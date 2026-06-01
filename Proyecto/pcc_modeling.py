import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from IPython.display import Image, display
from Metrics import Metrics
import time


class PCC_model:
    def __init__(self,n=3,L=0.063,m=0.034,K_val=0.56,D_val=0.1066):
        self.n = n
        self.L = L
        self.m_nominal = m
        self.m_real = 0.042
        self.g = 9.81

        self.K = K_val * np.eye(self.n)
        self.D = D_val * np.eye(self.n)


    @staticmethod
    def sinc(x):
        return cas.if_else(cas.fabs(x) < 1e-6, 1-((x**2)/6), cas.sin(x)/x)
    
    @staticmethod
    def R(theta):
        R = cas.vertcat(cas.horzcat(cas.cos(theta), -cas.sin(theta)),
                        cas.horzcat(cas.sin(theta), cas.cos(theta)))
        return R
    
    def build_augmented_model(self,mass_value):     
        xi_sym = cas.MX.sym('xi', 4*self.n)
        pcs = []
        current = cas.MX.zeros(2)
        phi = 0
        for i in range(self.n):
            theta1 = xi_sym[4*i + 0]
            d1     = xi_sym[4*i + 1]
            d2     = xi_sym[4*i + 2]
            theta2 = xi_sym[4*i + 3]

            phi = phi + theta1
            pc = current + self.R(phi) @ cas.vertcat(d1,0)
            pcs.append(pc)
            current = (current+self.R(phi) @ cas.vertcat(d1+d2,0))
            phi = phi + theta2

        # --------------------------------------------------------
        # augmented Jacobians
        # --------------------------------------------------------

        Jvs = []
        for i in range(self.n):
            Jv = cas.jacobian(pcs[i],xi_sym)
            Jvs.append(Jv)

        # --------------------------------------------------------
        # inertia
        # --------------------------------------------------------

        Bxi = cas.MX.zeros(4*self.n,4*self.n)
        for Jv in Jvs:
            Bxi += mass_value*(Jv.T @ Jv)

        # --------------------------------------------------------
        # gravity
        # --------------------------------------------------------
        V = 0
        for pc in pcs:
            V += mass_value*self.g*pc[1]
        Gxi = cas.gradient(V, xi_sym)

        # --------------------------------------------------------
        # compiled functions
        # --------------------------------------------------------

        Bxi_fun = cas.Function(f'Bxi_{str(mass_value).replace(".","_")}',[xi_sym],[Bxi])
        Gxi_fun = cas.Function(f'Gxi_{str(mass_value).replace(".","_")}',[xi_sym],[Gxi])

        return Bxi_fun, Gxi_fun
    
    def create_model(self):
        Bxi_plant_fun, Gxi_plant_fun = self.build_augmented_model(self.m_real)
        Bxi_ctrl_fun, Gxi_ctrl_fun = self.build_augmented_model(self.m_nominal)
        self.q = cas.MX.sym('q', self.n)
        self.qd = cas.MX.sym('qd', self.n)
        self.eta = cas.MX.sym('eta', self.n)
        self.tau = cas.MX.sym('tau', self.n)
        self.t = cas.MX.sym('t')
        xi_list = []
        for i in range(self.n):
            qi = self.q[i]
            di = self.L*self.sinc(qi/2)
            xi_list.extend([
                qi/2,
                di,
                di,
                qi/2
            ])
        xi = cas.vertcat(*xi_list)

        # ============================================================
        # PROJECTION JACOBIAN
        # ============================================================
        Jm = cas.jacobian(xi,self.q)

        # ============================================================
        # Jm_dot
        # ============================================================

        Jm_flat = cas.reshape(Jm,Jm.numel(),1)
        Jm_dot_flat = cas.jtimes(Jm_flat,self.q,self.qd)
        Jm_dot = cas.reshape(Jm_dot_flat,4*self.n,self.n)

        Bxi_plant = Bxi_plant_fun(xi)
        Gxi_plant = Gxi_plant_fun(xi)
        self.B_plant = (Jm.T @ Bxi_plant @ Jm)
        self.B_plant += 1e-5*cas.MX.eye(self.n)
        self.h_plant = (Jm.T @ Bxi_plant @ Jm_dot @ self.qd)
        self.G_plant = Jm.T @ Gxi_plant

        Bxi_ctrl = Bxi_ctrl_fun(xi)
        Gxi_ctrl = Gxi_ctrl_fun(xi)
        self.B_ctrl = (Jm.T @ Bxi_ctrl @ Jm)
        self.B_ctrl += 1e-5*cas.MX.eye(self.n)
        self.h_ctrl = (Jm.T @ Bxi_ctrl @ Jm_dot @ self.qd)
        self.G_ctrl = Jm.T @ Gxi_ctrl

        self.x = cas.vertcat(self.q,self.qd)
        self.u = self.tau

        B_reg = self.B_plant + 1e-2*cas.MX.eye(self.n)

        self.qdd = cas.inv(B_reg) @ (self.u - self.h_plant - self.D @ self.qd - self.G_plant - self.K @ self.q)

        #self.qdd = cas.solve(self.B_plant,self.u - self.h_plant - self.D @ self.qd - self.G_plant - self.K @ self.q)
        self.xdot = cas.vertcat(self.qd, self.qdd)

        # ====================================================
        # CONTINUOUS DYNAMICS FUNCTION
        # ====================================================

        self.f = cas.Function('f',[self.x, self.u],[self.xdot])
        self.B_func = cas.Function(
            'B_func',
            [self.q],
            [self.B_plant]
        )

        self.h_func = cas.Function(
            'h_func',
            [self.q, self.qd],
            [self.h_plant]
        )

        self.G_func = cas.Function(
            'G_func',
            [self.q],
            [self.G_plant]
        )
    
    def create_discretization(self, dt):
        # x = cas.MX.sym('X', 2*self.n)
        # u = cas.MX.sym('U', self.n)
        x = self.x
        u = self.u
        f = self.f

        # RK4 integration
        k1 = f(x,u)
        k2 = f(x + dt/2 * k1,u)
        k3 = f(x + dt/2 * k2,u)
        k4 = f(x + dt * k3,u)
        # x_next = x + dt/6 * (k1+ 2*k2 + 2*k3 + k4)
        x_next = x + dt*self.f(x,u)

        self.f_discrete = cas.Function('f_discrete',[x,u],[x_next])

    @property
    def nx(self):
        return self.x.size1()
    
    @property
    def nu(self):
        return self.u.size1()


class PCC_Controller:
    def __init__(self,pcc_model,qref,Iq=0.0,dt=0.03):
        self.model = pcc_model
        self.qref_fun, self.qdref_fun, self.qddref_fun = qref[0], qref[1], qref[2]
        self.Iq = Iq*np.eye(self.model.n)
        self.dt = dt

    def create_controller(self):
        q_ref = self.qref_fun(self.model.t)
        qd_ref = self.qdref_fun(self.model.t)
        qdd_ref = self.qddref_fun(self.model.t)

        e = q_ref - self.model.q
        eta_dot = e
        

        # ====================================================
        # CONTROLLER
        # ====================================================

        tau_cmd = (self.model.K @ q_ref + self.model.D @ qd_ref + self.model.h_ctrl + 
                   self.model.B_ctrl @ qdd_ref + self.model.G_ctrl + self.Iq @ self.model.eta)
        
        tau_state_dot = 20*(tau_cmd - self.model.tau)

        qdd = cas.solve(self.model.B_plant,
                        self.model.tau - self.model.h_plant - self.model.D @ self.model.qd - self.model.G_plant - self.model.K @ self.model.q)
        
        x_full = cas.vertcat(self.model.q,self.model.qd,self.model.eta,self.model.tau)
        xdot = cas.vertcat(self.model.qd,qdd,eta_dot,tau_state_dot)

        self.x_full = x_full
        self.xdot = xdot
        self.tau_cmd = tau_cmd
        self.q_ref = q_ref
        # ====================================================
        # COMPILED FUNCTIONS
        # ====================================================

        self.tau_fun = cas.Function('tau_fun',[x_full,self.model.t],[self.model.tau])
        self.qref_eval = cas.Function('qref_eval',[self.model.t],[q_ref])
        self.e_eval = cas.Function('e_eval',[x_full, self.model.t],[q_ref - self.model.q])
        # ====================================================
        # INTEGRATOR
        # ====================================================
        dae = {'x': x_full,'p': self.model.t,'ode': xdot}
        self.integrator = cas.integrator('integrator','cvodes', dae,{'tf': self.dt,'abstol': 1e-5,'reltol': 1e-3})


class PCCSimulation:
    def __init__(self,controller,T=12, x0=None):

        self.controller = controller
        self.model = controller.model
        self.T = T
        # ====================================================
        # INITIAL CONDITION
        # ====================================================

        if x0 is None:
            self.x0 = np.concatenate([
                1e-3*np.ones(self.model.n),
                np.zeros(self.model.n),
                np.zeros(self.model.n),
                np.zeros(self.model.n)
            ])
        else:
            self.x0 = x0

    # ========================================================
    # RUN SIMULATION
    # ========================================================

    def simulate(self):
        N = int(self.T/self.controller.dt)
        xk = self.x0.copy()
        X = [xk]
        times = [0]
        taus = []
        qrefs = []
        e = []
        print("Running simulation...")
        start = time.time()      
        for k in range(N):
            tk = k*self.controller.dt        
            tau_k = np.array(self.controller.tau_fun(xk,tk)).flatten()
            taus.append(tau_k) 

            e_k = np.array(self.controller.e_eval(xk, tk)).flatten()
            e.append(e_k)          

            qref_k = np.array(self.controller.qref_eval(tk)).flatten()
            qrefs.append(qref_k)
            
            res = self.controller.integrator(x0=xk,p=tk)
            xk = np.array(res['xf']).flatten()
            X.append(xk)
            times.append(tk + self.controller.dt)
        
        self.X = np.array(X)
        self.taus = np.array(taus)
        self.qrefs = np.array(qrefs)
        self.times = np.array(times)
        self.e = np.array(e)
        print(f"Simulation completed in "
              f"{time.time()-start:.2f} s")
        
        print("\n================ KPIs ================\n")
        IAE = []
        ISE = []
        ITAE = []
        RMSE = []
        for i in range(self.model.n):
            iae = Metrics.IAE(self.times[:-1],self.e[:,i])
            ise = Metrics.ISE(self.times[:-1],self.e[:,i])
            itae = Metrics.ITAE(self.times[:-1],self.e[:,i])
            rmse = Metrics.RMSE(self.e[:,i])
            IAE.append(iae) 
            ISE.append(ise)  
            ITAE.append(itae)
            RMSE.append(rmse)                      
            print(f"Section {i+1}")
            print(f"IAE  = {iae:.6f}")
            print(f"ISE  = {ise:.6f}")
            print(f"ITAE = {itae:.6f}")
            print(f"RMSE = {rmse:.6f}")
            print("-----------------------------------")
        
        IAE_mean = np.mean(IAE)
        ITAE_mean = np.mean(ITAE)
        ISE_mean = np.mean(ISE)
        RMSE_mean = np.mean(RMSE)
        print(f"Metrics - Average along sections: ")
        print(f"IAE  = {IAE_mean:.6f}")
        print(f"ISE  = {ISE_mean:.6f}")
        print(f"ITAE = {ITAE_mean:.6f}")
        print(f"RMSE = {RMSE_mean:.6f}")

    
    def plot_results(self,filename=None,save=False):

        fig1 = plt.figure(figsize=(10,5))
        for i in range(self.model.n):
            plt.plot(self.times,self.X[:,i],label=rf'$q_{i+1}$')
            plt.plot(self.times[:-1],self.qrefs[:,i],'--',linewidth=2,label=rf'$q_{{{i+1},ref}}$')

        plt.xlabel('Time (s)')
        plt.ylabel(r'Curvature Evolution, $q$ [rad]')       
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save:
            fig1.savefig(f"{filename[0]}.pdf", dpi=500, bbox_inches='tight')
        plt.show()

        # ====================================================
        # TORQUE PLOT
        # ====================================================

        fig2 = plt.figure(figsize=(10,5))
        for i in range(self.model.n):
            plt.plot(self.times[:-1],self.taus[:,i],label=rf'$\tau_{i+1}$')

        plt.xlabel('Time [sec]')
        plt.ylabel(r'Actuation Torques, $\tau$ [Nm]')        
        plt.grid(True)
        plt.legend()
        if save:
            fig2.savefig(f"{filename[1]}.pdf", dpi=500, bbox_inches='tight')
        plt.show()

    


        



        

    