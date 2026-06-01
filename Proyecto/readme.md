# PCC Soft Robot Control Framework Tutorial

This repository provides a modular framework for modeling, simulation, and control of Piecewise Constant Curvature (PCC) soft robots.

The framework includes:

- A nonlinear PCC dynamic model
- A Model Predictive Controller (MPC)
- A Nash Equilibrium Seeking (NES) controller
- Simulation environments
- Performance evaluation metrics

The purpose of this tutorial is to demonstrate how to build a PCC robot model, configure controllers, execute simulations, and analyze results.

---

# Framework Architecture

```text
                 PCC_model
                      |
     ----------------------------------
     |                                |
PCC_MPC_Controller         NE_Seeking_Controller
     |                                |
PCC_MPC_Simulator       NE_Seeking_Simulation
```

The framework is organized around the `PCC_model` class, which provides the robot dynamics.

Controllers generate actuation torques based on the robot state and the desired objective.

Simulation classes execute the interaction between the controller and the robot dynamics while storing performance data.

---

# 1. Import Required Libraries

```python
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

from pcc_modeling import *
from pcc_mpc import *
from pcc_NE_seeking import *
from Metrics import *
```

---

# 2. Building the PCC Robot Model

The PCC robot is represented using curvature coordinates

$q = [q_1\; q_2 \; q_3]$

where each curvature variable represents one PCC section.

The robot dynamics are automatically generated from the geometric model.

## Create the Robot

```python
model = PCC_model()
```

The default parameters are:

- Number of sections: 3
- Section length: 0.063 m
- Elastic stiffness matrix $K$
- Damping matrix $D$

---

## Generate the Dynamic Model

```python
model.create_model()
```

This method generates:

### Inertia Matrix

$B(q)$

### Coriolis and Centrifugal Terms

$h(q,\dot q)$

### Gravity Vector

$G(q)$

The resulting dynamic model is

$B(q)\ddot{q}+h(q,\dot{q})+D\dot{q}+G(q)+Kq=\tau$

which is converted into the state-space form

\[
x=
\begin{bmatrix}
q\\
\dot q
\end{bmatrix}
\]

\[
\dot x = f(x,u)
\]

---

## Create the Discrete-Time Model

The MPC controller requires a discrete-time model.

```python
dt = 1e-2

model.create_discretization(dt)
```

This generates

\[
x_{k+1}=f_d(x_k,u_k)
\]

which is used inside the optimization problem.

---

# 3. Understanding the State Vector

The state vector is defined as

\[
x=
\begin{bmatrix}
q\\
\dot q
\end{bmatrix}
\]

For a three-section PCC robot:

```python
model.nx
```

returns

```text
6
```

corresponding to

```text
[q1 q2 q3 q1_dot q2_dot q3_dot]
```

The control input is

```text
[u1 u2 u3]
```

representing the actuation torques applied to each section.

---

# 4. Model Predictive Control

The MPC controller computes the optimal torque sequence by solving a finite-horizon optimization problem at every sampling instant.

---

## Create the Controller

```python
mpc = PCC_MPC_Controller(model)
```

---

## Define MPC Parameters

```python
N = 5

mpc.set_params(
    N=N,
    dt=dt
)
```

where:

- `N` is the prediction horizon
- `dt` is the controller sampling time

---

## Define Cost Function Weights

```python
Q = np.diag([100,100,100,1,1,1])
R = 0.01*np.eye(model.nu)
S = 0.1*np.eye(model.nu)
P = Q
```

The optimization problem minimizes

\[
J=
\sum_{k=0}^{N-1}
\left(
e_k^TQe_k
+
u_k^TRu_k
+
\Delta u_k^TS\Delta u_k
\right)
+
e_N^TPe_N
\]

where

\[
e_k=x_k-x_k^{ref}
\]

---

### Matrix Interpretation

#### Q

Penalizes state tracking error.

Large values improve tracking performance.

---

#### R

Penalizes control effort.

Large values reduce torque magnitude.

---

#### S

Penalizes input variations.

Large values produce smoother torque trajectories.

---

#### P

Terminal cost matrix.

Improves finite-horizon performance.

---

## Assign MPC Parameters

```python
mpc.set_obj_function_params(
    Q=Q,
    R=R,
    S=S,
    P=P
)
```

---

## Build the Optimization Problem

```python
mpc.setup_controller()
```

This method:

1. Creates decision variables.
2. Adds dynamic constraints.
3. Constructs the objective function.
4. Configures IPOPT.

The optimization variables are:

```python
X
```

Predicted state trajectory.

```python
U
```

Predicted torque sequence.

The dynamic constraints are

\[
x_{k+1}=f_d(x_k,u_k)
\]

for every prediction step.

---

# 5. Defining the Reference Trajectory

The MPC simulator requires a reference trajectory function.

The function must return

```python
X_ref.shape = (nx,N+1)
```

---

## Example: Constant Curvature Reference

```python
omega = (2/3)*np.pi

def reference_function(t0, N, dt):

    X_ref = np.zeros((model.nx, N+1))

    for k in range(N+1):

        q_ref = np.array([
            0.20,
            0.25,
            0.30
        ])

        qd_ref = np.zeros(model.n)

        X_ref[:,k] = np.concatenate([
            q_ref,
            qd_ref
        ])

    return X_ref
```

This reference corresponds to a desired static configuration.

---

# 6. Running an MPC Simulation

---

## Create Simulator

```python
sim = PCC_MPC_Simulator(model)
```

---

## Initial Condition

```python
x0 = np.zeros(model.nx)
```

corresponding to

\[
q(0)=0
\]

\[
\dot q(0)=0
\]

---

## Execute Closed-Loop Simulation

```python
T = 20

n_steps = int(T/dt)

sim.run_closed_loop(
    controller=mpc,
    x_0=x0,
    n_steps=n_steps,
    reference_function=reference_function
)
```

The simulation loop performs:

1. Reference generation.
2. MPC optimization.
3. Application of the first optimal input.
4. State propagation.
5. Data storage.

This is the classical receding horizon strategy.

---

## Plot Results

```python
sim.plot_results(
    filename=["pcc_mpc_q","pcc_mpc_tau"],
    save=True
)
```

The simulator automatically generates:

- Curvature trajectories
- Reference trajectories
- Torque profiles
- KPI statistics

---

# 7. Nash Equilibrium Seeking Control

The Nash Equilibrium Seeking controller does not solve an optimization problem online.

Instead, each PCC section is treated as a player in a dynamic game.

Each player seeks its desired equilibrium curvature

\[
q_i^\star
\]

through extremum-seeking adaptation.

---

## Create the Robot

```python
N = 3

robot = PCC_model(n=N)

robot.create_model()
```

---

## Define the Desired Equilibrium

```python
q_star = np.array([
    0.2,
    0.25,
    0.3
])
```

---

## Create the Controller

```python
NE_controller = NE_Seeking_Controller(
    pcc_model = robot,
    n_players=N,
    q_star=q_star,
    kp=1000*np.ones(N),
    kd=20*np.ones(N),
    alpha=150*np.ones(N),
    beta=5*np.ones(N),
    k_es=np.array([2.76, 2.66, 1.627]),
    omega=np.array([110,140,115]),
    omega_l=0.5*np.ones(N),
    omega_h=0.5*np.ones(N),
    b=0.0001*np.ones(N),
    z_max=1,
    u_max=10.0
)
```

---

# 8. How the NE Controller Works

The control law is

\[
u_i
=
k_{p,i}(z_i-q_i)
-
k_{d,i}\dot q_i
\]

where

\[
z_i
\]

is a virtual equilibrium point.

This law behaves as a virtual spring-damper system.

---

## Payoff Function

Each player maximizes

\[
J_i
=
-\alpha_i
(q_i-q_i^\star)^2
\]

which is equivalent to minimizing the tracking error.

The maximum payoff occurs when

\[
q_i=q_i^\star
\]

for every section.

---

## Extremum Seeking Adaptation

The equilibrium estimate evolves according to

\[
\dot{\hat z}_i
=
k_{es,i}
(J_i-n_i)
\sin(\omega_i t)
\]

where:

- \(k_{es}\) controls adaptation speed
- \(\omega_i\) is the perturbation frequency
- \(n_i\) is a filtered estimate of the payoff

The adaptation law drives

\[
z_i
\rightarrow
q_i^\star
\]

without requiring gradient information.

---

# 9. Running an NE-Seeking Simulation

---

## Create Simulation Environment

```python
T = 20

dt = 1e-3

NE_simulation = NE_Seeking_Simulation(
    controller=NE_controller,
    T=T,
    dt=dt
)
```

---

## Execute Simulation

```python
NE_simulation.simulate()
```

The simulation performs:

1. State extraction.
2. Payoff computation.
3. Equilibrium adaptation.
4. Torque generation.
5. RK4 integration.
6. KPI computation.

---

## Plot Results

```python
NE_simulation.plot_results(
    filename=["pcc_NE_q","pcc_NE_u"],
    save=True
)
```

Generated plots include:

- Curvature evolution
- Control torques
- Steady-state zoom regions
- Performance metrics

---

# 10. Performance Metrics

The framework automatically computes:

## Integral Absolute Error (IAE)

\[
IAE=
\int_0^T |e(t)|dt
\]

---

## Integral Squared Error (ISE)

\[
ISE=
\int_0^T e^2(t)dt
\]

---

## Integral Time Absolute Error (ITAE)

\[
ITAE=
\int_0^T t|e(t)|dt
\]

---

## Root Mean Square Error (RMSE)

\[
RMSE=
\sqrt{
\frac1N
\sum_{k=1}^{N}
e_k^2
}
\]

These metrics provide quantitative measures of tracking performance and control quality.

---

# 11. Recommended Workflow

```text
1. Create PCC_model

2. Build dynamics
   model.create_model()

3. Create discrete model
   model.create_discretization(dt)

4. Choose controller
   ├── PCC_MPC_Controller
   └── NE_Seeking_Controller

5. Configure parameters

6. Create simulator

7. Execute simulation

8. Analyze KPIs

9. Tune controller parameters
```

---

# Conclusion

This framework provides a unified environment for investigating advanced control techniques for soft robots.

Two fundamentally different control philosophies are implemented:

### Model Predictive Control

- Optimization-based
- Explicitly handles constraints
- Computes optimal actions online

### Nash Equilibrium Seeking Control

- Game-theoretic
- Optimization-free online implementation
- Learns equilibrium configurations through extremum seeking

Because both controllers operate on the same PCC dynamic model, the framework allows direct and fair comparison between optimization-based and game-theoretic approaches for soft robotic systems.