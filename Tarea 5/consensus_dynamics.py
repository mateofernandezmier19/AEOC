import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import solve_ivp
plt.rcParams['text.usetex'] = True


class Consensus_Dynamics:

    def __init__(self,number_of_nodes=100,k_nearest_neighbors=6,p=0.0):
        self.n = number_of_nodes
        self.k = k_nearest_neighbors
        self.p = p
    
    def define_graph(self):
        self.G = nx.watts_strogatz_graph(self.n,self.k,self.p,seed=1)
        self.L = self.G.laplacian_matrix(self.G).toarray()
        eig = np.sort(np.linalg.eigvals(self.L))
        self.algebraic_connectivity = np.real(eig[1])
        
    def define_dynamics(self,t,x,L):
        rhs = -L @ x
        return rhs
    
    def Simulate_Consensus_Dynamics(self,t_final=10, n_points=1000):
        
        x0 = np.arange(1,self.n+1)
        t_eval = np.linspace(0,t_final,n_points)

        sol = solve_ivp(self.define_dynamics,[0,t_final],x0,args=(self.L,),t_eval=t_eval)

        self.t_sol, self.x_sol = sol.t, sol.y
    

