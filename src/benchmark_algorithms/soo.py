import numpy as np
from PyXAB.algos.SOO import SOO
from PyXAB.partition.BinaryPartition import BinaryPartition


class Soo:
    def __init__(self, f_obj, dimension, lower_bound, upper_bound, budget, h_max=100):
        """
        SOO algorithm (Munos, 2011)
        h_max : int, optional
            Maximum depth of the tree (default=100).
        """
        self.f_obj = f_obj
        self.dimension = dimension
        self.lower_bound = np.array(lower_bound).flatten()
        self.upper_bound = np.array(upper_bound).flatten()
        self.budget = int(budget)
        self.h_max = h_max

        # Domaine pour PyXAB
        self.domain = [[self.lower_bound[i], self.upper_bound[i]] for i in range(self.dimension)]

        self.best_solution = None
        self.best_value = float('inf')

    def run_soo(self):
        # Initialiser SOO
        algo = SOO(n=self.budget, h_max=self.h_max, domain=self.domain, partition=BinaryPartition)

        # Evaluation
        for t in range(self.budget):
            # Prochain point à évaluer
            point = algo.pull(t)
            point = np.array(point, dtype=float)
            point = np.clip(point, self.lower_bound, self.upper_bound)
            value = self.f_obj(point)

            # MAJ meilleure solution
            if value < self.best_value:
                self.best_value = value
                self.best_solution = np.array(point)

            # On utiliser -reward car nous on veut on minimise
            reward = -value
            algo.receive_reward(t, reward)

        self.f_obj(self.best_solution)

        return self.best_solution, self.best_value


def soo_optimizer(problem, dim, budget):
    """
    Fonction wrapper qui implémente l'interface requise par l'environnement BBOB (COCO).

    Arguments:
        problem: La fonction objectif COCO (un objet 'Problem' avec la méthode 'evaluate').
        dim: La dimension du problème.
        budget: Le nombre maximal d'évaluations de fonction (FEvals).
    """
    lower_bound = -5.0
    upper_bound = 5.0
    lower_bound = [lower_bound] * dim
    upper_bound = [upper_bound] * dim

    solver = Soo(
        f_obj=problem,
        dimension=dim,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        budget=budget,
        h_max=100
    )

    # Exécution de l'algorithme
    solver.run_soo()


"""
# Test
def sphere(x):
    return np.sum(x ** 2)

optimizer = Soo(sphere, dimension=2, lower_bound=[-5, -5], upper_bound=[5, 5], budget=5000)
best_sol, best_val = optimizer.run_soo()
print(f"Best solution: {best_sol}")
print(f"Best value: {best_val}")"""