import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from IPython.display import Image, display
# import matplotlib.animation as animation


class model:
    """"
    Class for the definition of the dynamic model using symbolic Casadi
    
    - Define states, inputs, parameters
    - Define ODE or discrete dynamics
    - Build CasADi functions
    - RK4 discretization
    """

    def __init__(self,model_type:str="continuous"):
        """"
        model_type:
            "continuous" -> xdot = f(x,u,p)
            "discrete" -> x(k+1) = f(x,u,p)
        """

        assert isinstance(model_type,str) , "model_type must be string, {} was passed".format(type(model_type))
        assert model_type in ["continuous","discrete"] , "model_type must be either discrete or continuous, but {} was passed".format(model_type)

        self.model_type = model_type

        # Dictionaries storing the symbolic variables before stacking
        self._x = {} # states
        self._u = {} # inputs
        self._p = {} # parameters

        # Dictionaries storing equations
        self._rhs = {} # ODE

        # Stacked variables
        self.x = None
        self.u = None
        self.p = None

        self.rhs = None

        # CasADi functions
        self.f = None # continuous or discrete model
        self.f_discrete = None # discrete version suitable for MPC

        self.setup_done = False

    # -----------------------
    # Variable Definition
    # ----------------------- 

    def set_state(self,name,size=1):
        """
        Create symbolic state variable
        """

        var = cas.SX.sym(name,size)
        self._x[name] = var
        return var
    
    def set_input(self,name,size=1):
        """
        Create symbolic control input variable
        """

        var = cas.SX.sym(name,size)
        self._u[name] = var
        return var
    
    # ----------------------------
    # Equations Definition
    # ----------------------------

    def set_rhs(self,state_name,expression):
        """
        Define RHS for a state equation
        """
        self._rhs[state_name] = expression

    # ---------------------------------
    # Setup
    # ---------------------------------

    def setup(self):
        """
        Stack symbolic variables and build CasADi functions
        Must be called before using the model
        """

        # Stack variables
        self.x = cas.vertcat(*self._x.values()) if self._x else cas.SX()
        self.u = cas.vertcat(*self._u.values()) if self._u else cas.SX()
        self.p = cas.vertcat(*self._p.values()) if self._p else cas.SX()

        # Stack differential equations
        rhs_list = [self._rhs[name] for name in self._x]
        self.rhs = cas.vertcat(*rhs_list) if rhs_list else cas.SX()

        # Create CasADi function
        if self.model_type == "continuous":
            self.f = cas.Function("f",[self.x,self.u],[self.rhs])
        else:
            self.f = cas.Function("f",[self.x,self.u],[self.rhs])

        self.setup_done = True

    # Discretization

    def create_discretization(self,dt):
        """
        Create a discrete model using RK4 integration
        Required for continuous models before doing MPC
        """

        x = self.x
        u = self.u
        f = self.f

        # RK4 integration
        k1 = f(x,u)
        k2 = f(x + (dt/2)*k1,u)
        k3 = f(x + (dt/2)*k2,u)
        k4 = f(x + dt*k3,u)
        
        x_next = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        self.f_discrete = cas.Function("f_discrete",[x,u],[x_next])

    @property
    def nx(self):
        return self.x.size1()
    
    @property
    def nu(self):
        return self.u.size1()
    

class MPC_Controller:
    """
    MPC controller
    """

    def __init__(self,model):
        self.model = model

        self.N = None  # Prediction Horizon
        self.dt = None

        # Matrices for error definition
        self.Q = None
        self.R = None

        # Constrains
        self.x_lb = None
        self.x_ub = None
        self.u_lb = None
        self.u_ub = None

        self.constraints = [] # list of the constraints functions

        self.opti = None
        self.reference_mode = None
        self.reference_function = None

    # ----------------------
    # Configuration
    # ----------------------

    def set_params(self,N,dt):
        self.N = N
        self.dt = dt

    def set_obj_function_params(self,Q,R,P=None):
        self.Q = Q
        self.R = R
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

    def set_reference_mode(self,reference_mode="constant"):
        assert reference_mode in ["constant", "trajectory tracking"], "reference_mode must be constant or trajectory tracking, but {} was passed".format(reference_mode)

        self.reference_mode = reference_mode



    def setup_controller(self):

        nx = self.model.nx
        nu = self.model.nu

        self.opti = cas.Opti()

        # Decision variables
        X = self.opti.variable(nx, self.N + 1)
        U = self.opti.variable(nu, self.N)

        # Parameters
        x_0 = self.opti.parameter(nx)

        if self.reference_mode == "trajectory tracking":
            x_ref = self.opti.parameter(nx, self.N + 1)
        else:
            x_ref = self.opti.parameter(nx)

        # Initial condition
        self.opti.subject_to(X[:,0] == x_0)

        cost = 0

        for k in range(self.N):

            x_next = self.model.f_discrete(X[:,k],U[:,k])

            # System Dynamic Constraints
            self.opti.subject_to(X[:,k+1] == x_next)

            # Stage Cost
            if self.reference_mode == "trajectory tracking":
                error = X[:,k] - x_ref[:,k]
            else:
                error = X[:,k] - x_ref

            cost += (error.T @ self.Q @ error) + (U[:,k].T @ self.R @ U[:,k])

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
            if self.reference_mode == "trajectory tracking":
                error_terminal = X[:,self.N] - x_ref[:,self.N]
            else:
                error_terminal = X[:,self.N] - x_ref

            cost += error_terminal.T @ self.P @ error_terminal

        self.opti.minimize(cost)
        self.opti.solver("ipopt")

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
    

class MPC_Simulator:
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
        if self.model.model_type == "continuous":
            x_next = self.model.f_discrete(x,u)
        else:
            x_next = self.model.f(x,u)

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

        self.mode = self.controller.reference_mode

        
        for k in range(n_steps):
            t_0 = k*controller.dt

            if self.controller.reference_mode == "trajectory tracking":
                if reference_function is None:
                    raise ValueError("Trajectory mode requires reference_function")
                else:
                    self.reference_function = reference_function
                
                x_ref = self.reference_function(t_0, controller.N, controller.dt)
                u = self.controller.make_step(x,x_ref)
                current_ref = x_ref[:,0]

            else:
                if X_ref is None:
                    raise ValueError("Constant mode requires X_ref")
                x_ref = X_ref

                u = self.controller.make_step(x,x_ref)

                current_ref = x_ref

            

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

        return self.x_history, self.u_history, self.t_history
    
    def plot_results(self, state_labels=None, input_labels=None, 
                 error_labels=None, save=False, save_prefix="simulation"):
        """
        Plot results in three separate figures:

        1) States vs time (n_states subplots)
        2) Inputs vs time (n_inputs subplots)
        3) Error vs time (single plot, all states)

        Parameters
        ----------
        state_labels : list of strings
        input_labels : list of strings
        error_labels : list of strings
        save : bool
            If True, saves each figure as a separate file
        save_prefix : str
            Prefix for saved filenames
        """

        n_states = self.x_history.shape[1]
        n_inputs = self.u_history.shape[1]


        # ============================
        # STATES FIGURE
        # ============================

        fig1 = plt.figure(figsize=(8, 2.5*n_states))

        for i in range(n_states):
            plt.subplot(n_states, 1, i+1)
            plt.plot(self.t_history, self.x_history[:, i], label="Current Value")

            if self.mode == "trajectory tracking":
                plt.plot(self.t_history, self.ref_history[:,i], "--", label="Reference")

            plt.ylabel(state_labels[i] if state_labels else fr"$x_{i}$")
            
            plt.grid()

            if self.mode == "trajectory tracking":
                plt.legend()

            if i == n_states - 1:
                plt.xlabel("Time (s)")

        plt.tight_layout()

        if save:
            fig1.savefig(f"{save_prefix}_states.pdf", dpi=500,bbox_inches='tight')

        # ============================
        #   INPUTS FIGURE
        # ============================

        fig2 = plt.figure(figsize=(8, 2.5*n_states))

        for i in range(n_inputs):
            plt.subplot(n_inputs, 1, i+1)
            plt.step(self.t_history, self.u_history[:, i], label="Current Value")
            
            if input_labels:
                plt.ylabel(input_labels[i])
            else:
                plt.ylabel(fr"$u_{i}$")
            plt.grid()

            if i == n_inputs - 1:
                plt.xlabel("Time (s)")

        plt.tight_layout()

        if save:
            fig2.savefig(f"{save_prefix}_inputs.pdf", dpi=500,bbox_inches='tight')

        # ============================
        # ERROR FIGURE
        # ============================

        fig3 = plt.figure(figsize=(8, 4))

        for i in range(n_states):
            plt.plot(self.t_history, self.error_history[:, i],
                    label=error_labels[i] if error_labels else f"e_{i}")

        plt.xlabel("Time (s)")
        plt.ylabel("Trackig Error" if self.mode == "trajectory tracking" else "Error")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        if save:
            fig3.savefig(f"{save_prefix}_error.pdf", dpi=500,bbox_inches='tight')
        

        fig4 = plt.figure(figsize=(8,4))
        plt.plot(self.x_history[:,0], self.x_history[:,1], label="Trajectory", linewidth=2)
        for constraint in self.controller.constraints:
            if hasattr(constraint, "plot"):
                constraint.plot(plt.gca())

        # Plot reference trajectory
        if self.mode == "trajectory tracking":
            plt.plot(self.ref_history[:,0], self.ref_history[:,1], "--", label="Reference")
        else:
            plt.plot(self.ref_history[0,0], self.ref_history[0,1], "ro", label="Reference")

        plt.xlabel(r"$x$ (m)")
        plt.ylabel(r"$y$ (m)")
        plt.title("XY Trajectory")
        # plt.axis("equal")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        if save:
            fig4.savefig(f"{save_prefix}_xy.pdf", dpi=500, bbox_inches='tight')

        plt.show()

    def animate_results(self, filename="simulation.gif", fps=20):
        """
        Animate states, inputs, errors and XY trajectory
        in a 2x2 layout and save as GIF.
        """

        fig, axs = plt.subplots(2, 2, figsize=(10,8))
        axs = axs.flatten()

        t = self.t_history
        X = self.x_history
        U = self.u_history
        E = self.error_history

        n_states = X.shape[1]
        n_inputs = U.shape[1]

        # --- Prepare lines ---
        state_lines = []
        for i in range(n_states):
            line, = axs[0].plot([], [], label=fr"$x_{i}$")
            state_lines.append(line)

        input_lines = []
        for i in range(n_inputs):
            line, = axs[1].step([], [], where='post', label=fr"$u_{i}$")
            input_lines.append(line)

        error_lines = []
        for i in range(n_states):
            line, = axs[2].plot([], [], label=fr"$e_{i}$")
            error_lines.append(line)

        traj_line, = axs[3].plot([], [], linewidth=2)
        current_point, = axs[3].plot([], [], 'ro')

        # Formatting
        axs[0].set_title("States")
        axs[1].set_title("Inputs")
        axs[2].set_title("Errors")
        # axs[3].set_title("XY Trajectory")

        for ax in axs[:3]:
            ax.grid()

        axs[3].set_aspect("equal")
        axs[3].grid()

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

        # Set limits
        axs[0].set_xlim(t[0], t[-1])
        axs[1].set_xlim(t[0], t[-1])
        axs[2].set_xlim(t[0], t[-1])

        # Set Y limits for states
        axs[0].set_ylim(np.min(X) - 0.1*np.abs(np.min(X)),
                        np.max(X) + 0.1*np.abs(np.max(X)))

        # Set Y limits for inputs
        axs[1].set_ylim(np.min(U) - 0.1*np.abs(np.min(U)),
                        np.max(U) + 0.1*np.abs(np.max(U)))

        # Set Y limits for errors
        axs[2].set_ylim(np.min(E) - 0.1*np.abs(np.min(E)),
                        np.max(E) + 0.1*np.abs(np.max(E)))

        axs[3].set_xlim(np.min(X[:,0])-1, np.max(X[:,0])+1)
        axs[3].set_ylim(np.min(X[:,1])-1, np.max(X[:,1])+1)
        for constraint in self.controller.constraints:
            if hasattr(constraint, "plot"):
                constraint.plot(axs[3])

        def update(frame):

            for i in range(n_states):
                state_lines[i].set_data(t[:frame], X[:frame,i])

            for i in range(n_inputs):
                input_lines[i].set_data(t[:frame], U[:frame,i])

            for i in range(n_states):
                error_lines[i].set_data(t[:frame], E[:frame,i])

            
            traj_line.set_data(X[:frame,0], X[:frame,1])
            current_point.set_data(X[frame-1,0], X[frame-1,1])

            axs[3].set_title(f"XY Trajectory  |  t = {t[frame]:.2f}s")

            return state_lines + input_lines + error_lines + [traj_line, current_point]

        ani = FuncAnimation(
            fig, update,
            frames=len(t),
            interval=1000/fps,
            blit=True
        )

        writer = ImageMagickWriter(fps=fps)
        ani.save(filename, writer=writer)

        
        plt.close(fig)
        print(f"Gif save to {filename}")
        display(Image(filename=filename))


class Constraint:
    """
    Base class for constraints.

    Each constraint must implement:

        - casadi_expression(x, u)
        - plot(ax)
    """

    def casadi_expression(self, x, u):
        raise NotImplementedError

    def plot(self, ax):
        """
        Plot constraint region on given axis.
        """
        pass
