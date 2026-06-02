import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from IPython.display import Image, display


class GeneticAlgorithm:
    """
    Genetic Algorithm (AGA)

    Real-valued genetic algorithm based on the implementation
    described by Kevin Passino.
    """

    def __init__(self,fitness_function,num_traits,low_trait,high_trait,pop_size=20,mutation_prob=0.05,
                 crossover_prob=0.8,elitism=False, max_generations=2000,delta=100,epsilon=0.01,seed=None,):

        self.fitness_function = fitness_function

        self.num_traits = num_traits
        self.low_trait = np.asarray(low_trait)
        self.high_trait = np.asarray(high_trait)

        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.elitism = elitism

        self.max_generations = max_generations
        self.delta = delta
        self.epsilon = epsilon

        self.rng = np.random.default_rng(seed)

        # History
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.worst_fitness_history = []
        self.best_individual_history = []
        self.population_history = []

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def initialize_population(self):
        """
        Uniform random population.
        Shape: (pop_size, num_traits)
        """
        population = self.rng.uniform(self.low_trait,self.high_trait,size=(self.pop_size, self.num_traits))
        return population

    # ==========================================================
    # FITNESS EVALUATION
    # ==========================================================

    def evaluate_population(self, population):
        """
        Evaluate all individuals.
        """
        fitness = np.array([self.fitness_function(ind) for ind in population])
        return fitness

    # ==========================================================
    # TOURNAMENT SELECTION
    # ==========================================================

    def select_parents(self, population, fitness):

        parents = np.zeros_like(population)
        tournament_size = 3
        for i in range(self.pop_size):
            candidates = self.rng.choice(self.pop_size,tournament_size,replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            parents[i] = population[winner]
        return parents

    # ==========================================================
    # CROSSOVER
    # ==========================================================

    def crossover(self, parents, best_idx=None):
        children = np.copy(parents)
        for i in range(self.pop_size):
            if self.elitism and i == best_idx:
                continue
            mate = i
            while mate == i:
                mate = self.rng.integers(0, self.pop_size)
            if self.rng.random() < self.crossover_prob:
                alpha = self.rng.random()
                children[i] = (alpha * parents[i]+ (1-alpha) * parents[mate])
                children[i] = np.clip(children[i],self.low_trait,self.high_trait)

        return children

    # ==========================================================
    # MUTATION
    # ==========================================================
    
    def mutate(self, children, best_idx=None):

        sigma = 0.03 * (self.high_trait - self.low_trait)
        for i in range(self.pop_size):
            if self.elitism and i == best_idx:
                continue
            if self.rng.random() < self.mutation_prob:
                children[i] += self.rng.normal(0,sigma,self.num_traits)
                children[i] = np.clip(children[i],self.low_trait,self.high_trait)

        return children

    # ==========================================================
    # MAIN OPTIMIZATION LOOP
    # ==========================================================

    def optimize(self):

        population = self.initialize_population()
        self.population_history = []       

        for generation in range(self.max_generations):
            self.population_history.append(population.copy())
            fitness = self.evaluate_population(population)

            best_idx = np.argmax(fitness)
            worst_idx = np.argmin(fitness)

            best_fitness = fitness[best_idx]
            avg_fitness = np.mean(fitness)
            worst_fitness = fitness[worst_idx]

            best_individual = population[best_idx].copy()

            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.worst_fitness_history.append(worst_fitness)
            self.best_individual_history.append(best_individual)            

            # Termination criterion
            if len(self.best_fitness_history) > self.delta:
                recent = np.array(self.best_fitness_history[-self.delta:])
                if np.max(np.abs(np.diff(recent))) <= self.epsilon:
                    print(f"Converged at generation {generation}")
                    break

            # Selection
            parents = self.select_parents(population,fitness)

            # Crossover
            children = self.crossover(parents,best_idx)

            # Mutation
            children = self.mutate(children,best_idx)

            population = children
        
        self.population = population.copy()


        return {"best_individual": best_individual,"best_fitness": best_fitness,"generation": generation,}
    
    def plot_results(self,resolution=300,levels=25,filename=None,save=False):
        """
        Visualize the optimization results.

        Figure 1:
            - 1D problems:
                Function + all visited individuals.
            - 2D problems:
                Contour map + all visited individuals.

        Figure 2:
            - Best / Average / Worst fitness
            - Best individual evolution
        """

        # ==========================================================
        # FIGURE 1
        # ==========================================================

        if self.num_traits == 1:
            fig1, ax = plt.subplots(figsize=(10, 6))
            x = np.linspace(self.low_trait[0],self.high_trait[0],resolution)
            y = np.array([self.fitness_function(np.array([xi])) for xi in x])

            ax.plot(x,y,'--',linewidth=2,label="Fitness Function")

            # ------------------------------------------------------
            # Population history
            # ------------------------------------------------------

            all_points = np.vstack(self.population_history)

            all_fitness = np.array([self.fitness_function(np.array([xi])) for xi in all_points[:, 0]])

            best = self.best_individual_history[-1]

            ax.scatter(best[0],self.fitness_function(best),marker="*",s=250,label="Best Solution",color='green')
            ax.scatter(all_points[:, 0],all_fitness,s=8,alpha=0.3,label="Population",color='red')          
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$f(x)$")
            ax.set_title("Fitness Function")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            if save:
                fig1.savefig(f"{filename[0]}.pdf", dpi=500, bbox_inches='tight')
            plt.show()

        # ==========================================================
        # 2D PROBLEM
        # ==========================================================

        elif self.num_traits == 2:
            # ======================================================
            # CREATE FITNESS LANDSCAPE
            # ======================================================

            x = np.linspace(self.low_trait[0],self.high_trait[0],resolution)
            y = np.linspace(self.low_trait[1],self.high_trait[1],resolution)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            for i in range(resolution):
                for j in range(resolution):
                    Z[i, j] = self.fitness_function(np.array([X[i, j],Y[i, j]]))

            best = self.best_individual_history[-1]

            # ======================================================
            # FIGURE 1
            # ======================================================

            fig1 = plt.figure(figsize=(14, 6))

            # ------------------------------------------------------
            # CONTOUR MAP
            # ------------------------------------------------------

            ax1 = fig1.add_subplot(121)
            contour = ax1.contour(X,Y,Z,levels=levels,cmap = 'viridis')           
            all_points = np.vstack(self.population_history)
            ax1.scatter(all_points[:, 0],all_points[:, 1],s=5,alpha=0.3,label="Population")
            ax1.scatter(best[0],best[1],marker='*',s=250,label="Best Solution")
            ax1.set_xlabel(r"$x$")
            ax1.set_ylabel(r"$y$")
            ax1.set_title("Contour Map")
            ax1.legend()
            ax1.grid(True)          

            # ------------------------------------------------------
            # 3D SURFACE
            # ------------------------------------------------------

            ax2 = fig1.add_subplot(122,projection='3d')
            ax2.plot_surface(X,Y,Z,cmap='viridis',edgecolor='none',alpha=0.9)
            z_best = self.fitness_function(best)
            ax2.scatter(best[0],best[1],z_best,marker='*',s=250,color='red')
            ax2.set_xlabel(r"$x$")
            ax2.set_ylabel(r"$y$")
            ax2.set_zlabel(r"$f(x,y)$")
            ax2.set_title("Fitness Surface")
            plt.tight_layout()
            if save:
                fig1.savefig(f"{filename[0]}.pdf", dpi=500, bbox_inches='tight')
            plt.show()

        else:
            raise ValueError(
                "Visualization only implemented for "
                "1D and 2D problems."
            )

        # ==========================================================
        # FIGURE 2
        # ==========================================================

        fig2, axs = plt.subplots(2,1,figsize=(10, 8),sharex=True)
        generations = np.arange(len(self.best_fitness_history))
        # ----------------------------------------------------------
        # Fitness evolution
        # ----------------------------------------------------------
        axs[0].plot(generations,self.best_fitness_history,label="Best")
        axs[0].plot(generations,self.avg_fitness_history,label="Average")
        axs[0].plot(generations,self.worst_fitness_history,label="Worst")
        axs[0].set_ylabel("Fitness")
        axs[0].set_title("Fitness Evolution")
        axs[0].grid(True)
        axs[0].legend()

        # ----------------------------------------------------------
        # Best individual evolution
        # ----------------------------------------------------------

        best_hist = np.array(self.best_individual_history)
        if self.num_traits == 1:
            axs[1].plot(generations,best_hist[:, 0],linewidth=2)
            axs[1].set_ylabel(r"Best $x$")
        else:
            for k in range(self.num_traits):
                axs[1].plot(generations,best_hist[:, k],label=rf"$x_{k+1}$")
            axs[1].legend()
        axs[1].set_xlabel("Generation")
        axs[1].set_ylabel("Best Individual")
        axs[1].set_title("Best Individual Evolution")
        axs[1].grid(True)
        plt.tight_layout()
        plt.tight_layout()
        if save:
            fig2.savefig(f"{filename[1]}.pdf", dpi=500, bbox_inches='tight')
        plt.show()
    