import numpy as np
import math


class Fractop:
    def __init__(self, f_obj, dimension, lower_bound, upper_bound, budget, initial_sample_size=100, gamma=0.15, rho=0.8,
                 min_region_size=1e-6, max_levels=10):
        """
        Implémentation fidèle de FRACTOP selon l'article Demirhan et.al,1999.

        """
        self.f_obj = f_obj
        self.dim = dimension
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.budget = budget
        self.initial_sample_size = initial_sample_size
        self.gamma = gamma
        self.rho = rho
        self.min_region_size = min_region_size
        self.max_levels = max_levels

        # Stockage
        self.regions = []  # Régions actives
        self.best_solution = None
        self.best_value = float('inf')
        self.eval_count = 0

        # Initialisation : une seule région (tout l'espace)
        self.regions.append({
            'lb': self.lower_bound.copy(),
            'ub': self.upper_bound.copy(),
            'level': 0,
            'parent_bel': 0.0,  # Pas de parent
            'bel': 1.0,  # Bel(S) = 1
            'samples': None,
            'values': None,
            'id': 0
        })

        self.region_counter = 1

    def divide_region(self, region):
        """Divise une région en 2^dim sous-régions identiques (page 3)"""
        if self.is_region_too_small(region):
            return []  # Ne pas diviser si trop petite

        lb, ub = region['lb'], region['ub']
        mid = (lb + ub) / 2.0
        children = []

        # Générer les 2^dim combinaisons
        for mask in range(2 ** self.dim):
            new_lb = lb.copy()
            new_ub = ub.copy()

            for d in range(self.dim):
                if (mask >> d) & 1:
                    new_lb[d] = mid[d]
                else:
                    new_ub[d] = mid[d]

            children.append({
                'lb': new_lb,
                'ub': new_ub,
                'level': region['level'] + 1,
                'parent_bel': region['bel'],  # Hérite du bel du parent
                'bel': None,  # À calculer après échantillonnage
                'samples': None,
                'values': None,
                'id': self.region_counter
            })
            self.region_counter += 1

        return children

    def is_region_too_small(self, region):
        """Vérifie si la région est plus petite que min_region_size"""
        region_sizes = region['ub'] - region['lb']
        return np.any(region_sizes < self.min_region_size)

    def get_sample_size(self, region):
        if region['level'] == 0:
            return self.initial_sample_size
        # Taille proportionnelle
        region_size = np.prod(region['ub'] - region['lb'])
        initial_size = np.prod(self.upper_bound - self.lower_bound)
        ratio = region_size / initial_size

        s = max(30, int(self.initial_sample_size * ratio))

        # Ne pas dépasser le budget restant
        return min(s, self.budget - self.eval_count)

    def sample_region(self, region):
        """Random Search (RS) - Random Search"""
        s = self.get_sample_size(region)
        if s <= 0 or self.eval_count >= self.budget:
            return np.array([]), np.array([])

        lb, ub = region['lb'], region['ub']
        samples = []
        values = []

        # Générer s points aléatoires uniformes
        for _ in range(s):
            if self.eval_count >= self.budget:
                break

            x = np.random.uniform(lb, ub)
            fx = self.f_obj(x)

            samples.append(x)
            values.append(fx)
            self.eval_count += 1

            # Mettre à jour la meilleure solution globale
            if fx < self.best_value:
                self.best_value = fx
                self.best_solution = x.copy()

        return np.array(samples), np.array(values)

    def compute_m(self, values):
        """
        Fuzzy evidence measure m(A_it) : quantifie l'infomation apportée par la région
        m(A_it) = 1/s * Σ [(F_j/F*) * exp(1 - F_j/F*)]
        """
        if len(values) == 0:
            return 0.0

        # F* = meilleure valeur globale (article page 3)
        if self.best_value == float('inf'):
            F_star = np.min(values)  # Au début, prendre le min local
        else:
            F_star = self.best_value

        if F_star == 0:
            F_star = 1e-12 # Eviter la division par 0

        sum_evidence = 0.0
        for F_j in values:
            ratio = F_j / F_star

            # Éviter overflow pour grands ratios
            if ratio > 100:
                term = 0.0
            else:
                term = ratio * np.exp(1.0 - ratio)

            sum_evidence += term

        return sum_evidence / len(values)

    def compute_bel(self, region, m_value):
        """
        Bel(alpha_it) = γ * parent_bel + (1-γ) * m(A_it)
        """
        return self.gamma * region['parent_bel'] + (1 - self.gamma) * m_value

    def should_eliminate_region(self, region, best_bel):
        """
        Critère d'élimination (si région petite ou Bel(region) < ρ * Bel(best_region))
        """
        if self.is_region_too_small(region):
            return True
        if region['bel'] is None:
            return True
        if region['bel'] < self.rho * best_bel:
            return True

        return False

    def run_fractop(self):
        level = 0

        while self.eval_count < self.budget and level < self.max_levels and len(self.regions) > 0:

            # Echantillonnage et calcul des croyances
            for region in self.regions:
                if self.eval_count >= self.budget:
                    break

                # RS sampling
                samples, values = self.sample_region(region)
                region['samples'] = samples
                region['values'] = values

                # Calcul de Bel
                if len(values) > 0:
                    m_val = self.compute_m(values)
                    region['bel'] = self.compute_bel(region, m_val)

            # Trouver la meilleure région
            valid_regions = [r for r in self.regions if r['bel'] is not None]
            if not valid_regions:
                break

            best_region = max(valid_regions, key=lambda r: r['bel'])
            best_bel = best_region['bel']

            # Elimination
            survivors = []
            for region in self.regions:
                if not self.should_eliminate_region(region, best_bel):
                    survivors.append(region)

            # Division de la meilleure région
            new_regions = []
            # Vérifiez si la meilleure région est toujours dans les survivants ET n'est pas trop petite
            if best_region['id'] in [r['id'] for r in survivors] and not self.is_region_too_small(best_region):
                survivors = [r for r in survivors if r['id'] != best_region['id']]

                # Diviser en 2^dim enfants
                children = self.divide_region(best_region)
                new_regions.extend(children)

            # Mise à jour
            self.regions = survivors + new_regions
            level += 1

        return self.best_solution, self.best_value


def fractop_optimizer(problem, dim, budget):
    """
    Fonction wrapper qui implémente l'interface requise par l'environnement BBOB (COCO).

    Arguments:
        problem: La fonction objectif COCO (un objet 'Problem' avec la méthode 'evaluate').
        dim: La dimension du problème.
        budget: Le nombre maximal d'évaluations de fonction (FEvals).
    """

    lower_bound = [-5.0] * dim
    upper_bound = [5.0] * dim

    solver = Fractop(
        f_obj=problem,
        dimension=dim,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        budget=budget,
    )

    # Exécution de l'algorithme
    a, b =solver.run_fractop()

    return a, b

"""
# -- TEST---
def sphere(x):
    return np.sum(x**2)

best_sol, best_val = fractop_optimizer(sphere, dim=2, budget=1000)
print(f"Meilleure solution : {best_sol}")
print(f"Meilleure valeur   : {best_val}")"""