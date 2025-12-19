import numpy as np
from scipy.optimize import direct, Bounds


class Direct:

    def __init__(self, f_obj, dimension, lower_bound, upper_bound, budget):
        """
                direct algorithm (Jones et al., 1993) (last updat 2007)

        """

        self.f_obj = f_obj
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.budget = budget

    def run_direct(self):

        # Définir les limites BBOB comme un objet Bounds (pour SciPy)
        bounds = Bounds(self.lower_bound, self.upper_bound)

        # SciPy.direct ne prend pas de budget direct, on utilise maxfev (max fun evaluations)
        max_evaluations = int(self.budget)

        # Exécuter l'algorithme DIRECT de SciPy
        result = direct(self.f_obj, bounds, maxfun=max_evaluations
                        )
        # MAJ de la meilleurs solution avec la meilleure solution retournée par direct
        self.f_obj(result.x)

        # Retourner la solution et la valeur
        return result.x, result.fun


def direct_optimizer(problem, dim, budget):
    """
    Fonction wrapper qui implémente l'interface requise par l'environnement BBOB (COCO).

    Arguments:
        problem: La fonction objectif COCO (un objet 'Problem' avec la méthode 'evaluate').
        dim: La dimension du problème.
        budget: Le nombre maximal d'évaluations de fonction (FEvals).
    """
    lower_bound = [-5.0] * dim
    upper_bound = [5.0] * dim

    solver = Direct(
        f_obj=problem,
        dimension=dim,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        budget=budget
    )

    # Exécution de l'algorithme
    best_sol, best_val = solver.run_direct()

    return best_sol, best_val


"""
# -- TEST---
def sphere(x):
    return np.sum(pow(x, 2))

best_sol, best_val = direct_optimizer(sphere, dim=2, budget=1000*2)
print(f"Meilleure solution : {best_sol}")
print(f"Meilleure valeur   : {best_val}")"""