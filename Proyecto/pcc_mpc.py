import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from IPython.display import Image, display
from Metrics import Metrics
import time


class PCC_MPC_Controller:

    def __init__(self,model):
        self.model = model

        self.N = None  # Prediction Horizon
        self.dt = None

        # Matrices for error definition
        self.Q = None
        self.R = None
        self.S = None

        # Constrains
        self.x_lb = None
        self.x_ub = None
        self.u_lb = None
        self.u_ub = None

        self.constraints = [] # list of the constraints functions

        self.opti = None        
        self.reference_function = None

    def set_params(self,N,dt):
        self.N = N
        self.dt = dt

    def set_obj_function_params(self,Q,R,S,P=None):
        self.Q = Q
        self.R = R
        self.S = S
        self.P = P
    
    def set_state_bounds(self,lb,ub):
        """
        lb, ub: arrays of size nx
        """

        self.x_lb = lb
        self.x_ub = ub

    def set_input_bounds(self,lb,ub):
        """
        lb, ub: arrays of size nx
        """

        self.u_lb = lb
        self.u_ub = ub

    def add_constraint(self,constraint_obj):
        """
        func must return an expression of the form g(x,u) <= 0
        """
        
        self.constraints.append(constraint_obj)

    def setup_controller(self):

        nx = self.model.nx
        nu = self.model.nu

        self.opti = cas.Opti()

        # Decision variables
        X = self.opti.variable(nx, self.N + 1)
        U = self.opti.variable(nu, self.N)

        # Parameters
        x_0 = self.opti.parameter(nx)

        x_ref = self.opti.parameter(nx, self.N + 1)

        # Initial condition
        self.opti.subject_to(X[:,0] == x_0)

        cost = 0

        for k in range(self.N):

            x_next = self.model.f_discrete(X[:,k],U[:,k])

            # System Dynamic Constraints
            self.opti.subject_to(X[:,k+1] == x_next)

            error = X[:,k] - x_ref[:,k]            

            cost += (error.T @ self.Q @ error) + (U[:,k].T @ self.R @ U[:,k])

            if k > 0:
                du = U[:,k] - U[:,k-1]
                cost += du.T @ self.S @ du

            if self.x_lb is not None:
                self.opti.subject_to(self.opti.bounded(self.x_lb,X[:,k],self.x_ub))

            if self.u_lb is not None:
                self.opti.subject_to(self.opti.bounded(self.u_lb,U[:,k],self.u_ub))

            if len(self.constraints) >= 1:
                for constraint in self.constraints:
                    g_expr = constraint.casadi_expression(X[:,k],U[:,k])
                    self.opti.subject_to(g_expr <= 0)

        # Option for terminal cost
        if self.P is not None:            
            error_terminal = X[:,self.N] - x_ref[:,self.N]           
            cost += error_terminal.T @ self.P @ error_terminal

        self.opti.minimize(cost)

        p_opts = {}
        s_opts = {
            'max_iter': 2000,
            'print_level': 0,
            'tol': 1e-4,
            'acceptable_tol': 1e-3
        }
        self.opti.solver('ipopt',p_opts,s_opts)
        # self.opti.solver("ipopt")

        # Store
        self.X = X
        self.U = U
        self.x_0 = x_0
        self.x_ref = x_ref

    # ------------------------
    # Solve
    # ------------------------

    def make_step(self,x_0_val, x_ref_val):
        self.opti.set_value(self.x_0,x_0_val)
        self.opti.set_value(self.x_ref, x_ref_val)

        sol = self.opti.solve()

        return sol.value(self.U[:,0])

class PCC_MPC_Simulator:
    """
    This class handles:
        - Single step simulation
        - Closed-loop simulation
        - Plotting
        - Animation
    """

    def __init__(self, model):
        self.model = model

        self.reference_function = None
        
        # History
        self.x_history = []
        self.u_history = []
        self.t_history = []
        self.error_history = []

    # Single step
    def make_step(self,x,u):        
        x_next = self.model.f_discrete(x,u)
        
        return np.array(x_next.full()).flatten()
    
    # Closed-loop Simulation

    def run_closed_loop(self,controller, x_0, n_steps, X_ref = None, reference_function = None):
        x = np.array(x_0)

        self.x_history = []
        self.u_history = []
        self.t_history = []
        self.error_history = []
        self.ref_history = []

        self.controller = controller   
        print("Running simulation...")
        start = time.time()     
        
        for k in range(n_steps):
            t_0 = k*controller.dt
            
            if reference_function is None:
                raise ValueError("Trajectory mode requires reference_function")
            else:
                self.reference_function = reference_function
            
            x_ref = self.reference_function(t_0, controller.N, controller.dt)
            u = self.controller.make_step(x,x_ref)
            current_ref = x_ref[:,0]                      

            self.x_history.append(x)
            self.u_history.append(u)
            self.t_history.append(t_0)
            self.error_history.append(x-current_ref)
            self.ref_history.append(current_ref)

            x = self.make_step(x,u)

        self.x_history = np.array(self.x_history)
        self.u_history = np.array(self.u_history)
        self.t_history = np.array(self.t_history)
        self.error_history = np.array(self.error_history)
        self.ref_history = np.array(self.ref_history)

        print(f"Simulation completed in "
              f"{time.time()-start:.2f} s")
        
        print("\n================ KPIs ================\n")
        IAE = []
        ISE = []
        ITAE = []
        RMSE = []
        for i in range(self.model.n):
            iae = Metrics.IAE(self.t_history,self.error_history[:,i])
            ise = Metrics.ISE(self.t_history,self.error_history[:,i])
            itae = Metrics.ITAE(self.t_history,self.error_history[:,i])
            rmse = Metrics.RMSE(self.error_history[:,i])
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
        

        return self.x_history, self.u_history, self.t_history
    
    def plot_results(self,filename=None,save=False):

        q = self.x_history[:, :self.model.n]
        q_ref = self.ref_history[:, :self.model.n]
        u = self.u_history
        # ====================================================
        # STATES
        # ====================================================

        fig1 = plt.figure(figsize=(12,8))
        for i in range(self.model.n):            
            plt.plot(self.t_history, q[:,i],linewidth=2,label=rf'$q_{i+1}$')
            plt.plot(self.t_history,q_ref[:,i],'--',linewidth=2,label=rf'$q_{{{i+1},ref}}$')
            plt.ylabel('rad')
            plt.grid(True)
            plt.legend()
        
        plt.xlabel('Time (s)')
        plt.ylabel(r'Curvature Evolution, $q$ [rad]')       
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save:
            fig1.savefig(f"{filename[0]}.pdf", dpi=500, bbox_inches='tight')
        plt.show()

        # ====================================================
        # INPUTS
        # ====================================================

        fig2 = plt.figure(figsize=(12,8))
        for i in range(self.model.n):       
            plt.step(self.t_history,u[:,i],linewidth=2,label=rf'$\tau_{i+1}$')
           

        plt.xlabel('Time [sec]')
        plt.ylabel(r'Actuation Torques, $\tau$ [Nm]')        
        plt.grid(True)
        plt.legend()
        if save:
            fig2.savefig(f"{filename[1]}.pdf", dpi=500, bbox_inches='tight')
        plt.show()
        


