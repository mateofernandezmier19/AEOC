import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from IPython.display import Image, display
from Metrics import Metrics
import time

class NE_Seeking_Controller:

    def __init__(self,pcc_model,n_players, q_star, kp=None, kd=None, alpha=None,beta=None,k_es=None, omega=None,phi=None,omega_l=None,omega_h=None,b=None,
                 z_max=1.0, u_max=10.0,z0 = None):
        
        self.model = pcc_model
        self.N = self.model.n        
        self.q_star = q_star

        # ====================================================
        # VIRTUAL STIFFNESS/DAMPING
        # ====================================================

        self.kp = (8*np.ones(self.N) if kp is None else kp)
        self.kd = (2*np.ones(self.N) if kd is None else kd)

        # ====================================================
        # PAYOFF WEIGHTS
        # ====================================================

        self.alpha = (20*np.ones(self.N) if alpha is None else alpha)
        self.beta = (5*np.ones(self.N) if beta is None else beta)

        # ====================================================
        # ES PARAMETERS
        # ====================================================

        self.k_es = (0.01*np.ones(self.N) if k_es is None else k_es)
        self.omega = (np.array([5,7,9][:self.N]) if omega is None else omega)
        self.phi = (np.zeros(self.N) if phi is None else phi)
        self.omega_l = (0.05*np.ones(self.N) if omega_l is None else omega_l)
        self.omega_h = (1*np.ones(self.N) if omega_h is None else omega_h)
        self.b = (0.05*np.ones(self.N) if b is None else b)

        # ====================================================
        # LIMITS
        # ====================================================

        self.z_max = z_max
        self.u_max = u_max

        # ====================================================
        # INTERNAL STATES
        # ====================================================

        if z0 is None:
            self.z_hat = 0.15*np.ones(self.N)
        else:
            self.z_hat = z0

        self.a = 0.001*np.ones(self.N)
        self.n = np.zeros(self.N)

    # ========================================================
    # PAYOFF FUNCTION
    # ========================================================

    def payoff(self, q, q_dot, z):

        J = np.zeros(self.N)
        for i in range(self.N):
            equilibrium_error = (self.alpha[i] * (z[i] - self.q_star[i])**2)
            regulation_error = (2.0 * (q[i] - z[i])**2)
            velocity = (self.beta[i]*q_dot[i]**2)
            J[i] = -(self.alpha[i] * (q[i] - self.q_star[i])**2)

        return J

    # ========================================================
    # UPDATE
    # ========================================================

    def update(self,q,q_dot, t,dt):
        # ----------------------------------------------------
        # perturbation
        # ----------------------------------------------------
        eta = np.sin(self.omega*t + self.phi)

        # ----------------------------------------------------
        # virtual equilibria
        # ----------------------------------------------------

        z = (self.z_hat + self.a*eta)
        z = np.clip(z,-self.z_max,self.z_max)
        # ----------------------------------------------------
        # equilibrium-shaping control law
        # ----------------------------------------------------
        u = (self.kp * (z - q) - self.kd * q_dot)
        u = np.clip(u,-self.u_max,self.u_max)
        # ----------------------------------------------------
        # payoff
        # ----------------------------------------------------
        J = self.payoff(q, q_dot, z)
        # ----------------------------------------------------
        # adaptation
        # ----------------------------------------------------
        dz_hat = (self.k_es *(J - self.n) * eta)
        da = (-self.omega_l*self.a + self.b * self.omega_l* (J - self.n))
        dn = (-self.omega_h*self.n + self.omega_h*J)
        # ----------------------------------------------------
        # integrate
        # ----------------------------------------------------
        self.z_hat += dz_hat*dt
        self.a += da*dt
        self.n += dn*dt

        return u, z, J, da
    
    def rk4_step(self, x, u, dt):

        k1 = np.array(self.model.f(x,u)).astype(np.float64).flatten()
        k2 = np.array(self.model.f( x + 0.5*dt*k1,u)).astype(np.float64).flatten()
        k3 = np.array(self.model.f(x + 0.5*dt*k2,u)).astype(np.float64).flatten()
        k4 = np.array(self.model.f(x + dt*k3,u)).astype(np.float64).flatten()
        return (x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4))
    
class NE_Seeking_Simulation:
    
    def __init__(self,controller,T,dt):
        self.controller = controller
        self.T = T
        self.dt = dt
        self.steps = int(self.T/self.dt)

        self.x = np.zeros(2*self.controller.N)
    
    def simulate(self):
        t_hist = []
        q_hist = []
        u_hist = []
        z_hist = []
        e_hist = []
        J_hist = []
        da_hist = []

        print("Running simulation...")
        start = time.time()

        for k in range(self.steps):

            t = k*self.dt
            # --------------------------------------------------------
            # states
            # --------------------------------------------------------

            q = self.x[:self.controller.N]
            q_dot = self.x[self.controller.N:]

            # --------------------------------------------------------
            # controller
            # --------------------------------------------------------

            u, z, J, da = self.controller.update(q=q,q_dot=q_dot,t=t,dt=self.dt)

            # --------------------------------------------------------
            # nonlinear PCC dynamics
            # --------------------------------------------------------

            self.x = self.controller.rk4_step(self.x,u,self.dt)

            # --------------------------------------------------------
            # errors
            # --------------------------------------------------------

            e = self.controller.q_star - q
            # --------------------------------------------------------
            # store
            # --------------------------------------------------------

            t_hist.append(t)
            q_hist.append(q.copy())
            u_hist.append(u.copy())
            z_hist.append(z.copy())
            e_hist.append(e.copy())
            J_hist.append(J.copy())
            da_hist.append(da.copy())

        print(f"Simulation completed in "
            f"{time.time()-start:.2f} s")
        
        self.t_hist = np.array(t_hist)
        self.q_hist = np.array(q_hist)
        self.u_hist = np.array(u_hist)
        self.z_hist = np.array(z_hist)
        self.e_hist = np.array(e_hist)
        self.J_hist = np.array(J_hist)
        self.da_hist = np.array(da_hist)

        print("\n================ KPIs ================\n")
        IAE = []
        ISE = []
        ITAE = []
        RMSE = []
        for i in range(self.controller.model.n):
            iae = Metrics.IAE(self.t_hist,self.e_hist[:,i])
            ise = Metrics.ISE(self.t_hist,self.e_hist[:,i])
            itae = Metrics.ITAE(self.t_hist,self.e_hist[:,i])
            rmse = Metrics.RMSE(self.e_hist[:,i])
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
        fig1 = plt.figure(figsize=(12,8))
        for i in range(self.controller.N):
            plt.plot(self.t_hist,self.q_hist[:,i],linewidth=2,label=f"$q_{i+1}$")
            plt.axhline(self.controller.q_star[i],linestyle='--',linewidth=2,color='r',label=rf'$q_{{{i+1},ref}}$')

        plt.xlabel('Time (s)')
        plt.ylabel(r'Curvature Evolution, $q$ [rad]')       
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save:
            fig1.savefig(f"{filename[0]}.pdf", dpi=500, bbox_inches='tight')
        plt.show()

        
        fig2, ax = plt.subplots(figsize=(12,8))

        for i in range(self.controller.N):
            ax.plot(self.t_hist,self.u_hist[:, i],linewidth=2,label=rf"$u_{i+1}$")

        ax.set_xlabel('Time [sec]')
        ax.set_ylabel(r'Actuation Torques, $\tau$ [Nm]')
        ax.grid(True)
        ax.legend()

        # ----------------------------------------------------------
        # ZOOM-IN AXES
        # ----------------------------------------------------------
        
        axins = inset_axes(
            ax,
            width="25%", height="35%",
            loc='upper right',
            bbox_to_anchor=(-0.1, -0.55, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        for i in range(self.controller.N):
            axins.plot(self.t_hist,self.u_hist[:, i],linewidth=1.5)

        # Zoom near steady state
        idx = int(0.8 * len(self.t_hist))
        xmin = self.t_hist[idx]
        xmax = self.t_hist[-1]
        ymin = np.min(self.u_hist[idx:]) - 0.05
        ymax = np.max(self.u_hist[idx:]) + 0.05
        axins.set_xlim(xmin, xmax)
        axins.set_ylim(ymin, ymax)

        mark_inset(ax,axins,loc1=2,loc2=4,fc="none",ec="black")
        axins.grid()

        
        if save:
            fig2.savefig(f"{filename[1]}.pdf", dpi=500, bbox_inches='tight')
        plt.show()

    

        
                
    


