import copy
#import time
import numpy as np

from scipy.stats import qmc
from src.fda_v0.hypersphere import Hypersphere
from src.fda_v0.constance import *


class FdaBaselinePhsWsc:

    def __init__(self, f_obj, dimension, k_levels_max, lower_bound, upper_bound, ls_method, promising_sphere_method):
        self.f_obj = f_obj
        self.m_dimension = dimension
        self.m_k_levels_max = k_levels_max
        self.m_upper_bound = upper_bound
        self.m_lower_bound = lower_bound
        self.m_stopCriterion = 5000 * dimension
        self.NumberOfEvaluations = 0

        self.ls_method = ls_method
        self.promising_sphere_method = promising_sphere_method
        self.axis_scores = np.zeros(self.m_dimension)

        # Pour sauvegrader les résultats
        self.history_fitness = []
        self.history_nb_eval = []
        self.history_nb_explored_sphere = []
        self.history_nb_decomposed_sphere = []
        self.history_nb_exploited_sphere = []
        self.history_nb_eval_exploitation = []
        self.history_nb_eval_exploration = []
        self.history_nb_visited_sphere = []
        self.history_nb_sphere_visited = []

        self.nb_eval_at_best = 0

        self.nb_explored_sphere = 0
        self.nb_decomposed_sphere = 0
        self.nb_exploited_sphere = 0
        self.nb_eval_exploitation = 0
        self.nb_eval_exploration = 0
        self.nb_visited_sphere = 0
        self.nb_sphere_visited = 0

        # Pour nalyser l'équilibre ===
        self.history_balance_ratio = []
        # sauvegarde toutes les x éval
        if dimension <= 5:
            self.save_interval = 1000
        elif 20 >= dimension >= 10:
            self.save_interval = 5000
        else:
            self.save_interval = 10000

        self.m_CurrentLevel = 1
        self.m_stackMemory = {}
        self.m_indexes = {}

        temp_radius = (upper_bound - lower_bound) / 2
        temp_center = [lower_bound + (upper_bound - lower_bound) / 2] * dimension

        self.m_bestSolutionCoordinates = copy.copy(temp_center)
        self.m_bestSolutionCoordinatesNavigation = copy.copy(temp_center)
        self.m_bestSolutionFitness = float('inf')
        self.m_bestSolutionFitness = self.evaluation_fitness(self.m_bestSolutionCoordinates)
        self.nb_eval_exploration += 1
        self.m_bestSolutionFitnessNavigation = self.m_bestSolutionFitness

        self.m_CurrentHyperSphere = Hypersphere(dimension, temp_center, temp_radius, None)
        self.m_CurrentHyperSphere.m_fitness = self.m_bestSolutionFitness

    # --- Exploration stratey ---
    def decomposition_hypersphere(self, hypersphere):
        result_list = []
        r0 = hypersphere.m_radius / delta
        offset = hypersphere.m_radius - r0
        self.nb_decomposed_sphere += 1

        for i in range(hypersphere.m_dimension):
            for sign in (-1, +1):
                new_center = hypersphere.m_center.copy()
                new_center[i] += sign * offset
                subHyperSphere = Hypersphere(hypersphere.m_dimension, new_center, r0, hypersphere.belief_score)
                result_list.append(subHyperSphere)

        hypersphere.children_spheres = result_list
        return result_list

    # --- Exploitation stratey ---
    def intensive_local_search(self):

        s = copy.copy(self.m_CurrentHyperSphere.m_center)
        gamma = self.m_CurrentHyperSphere.m_radius
        solution_s = -math.inf
        best_temp_solution_s = -math.inf

        if self.NumberOfEvaluations < self.m_stopCriterion:
            solution_s = self.evaluation_fitness(s)
            self.nb_eval_exploitation += 1
            best_temp_solution_s = solution_s
            self.nb_exploited_sphere += 1

        while gamma > GAMMA_MIN and self.NumberOfEvaluations < self.m_stopCriterion:

            for d in range(self.m_dimension):
                if self.NumberOfEvaluations + 2 > self.m_stopCriterion: break

                s1 = copy.copy(s)
                s1[d] -= gamma
                s2 = copy.copy(s)
                s2[d] += gamma

                solution_s1 = self.evaluation_fitness(s1)
                solution_s2 = self.evaluation_fitness(s2)
                self.nb_eval_exploitation += 2

                # Trouver le meilleur candidat
                candidates = [(solution_s, s), (solution_s1, s1), (solution_s2, s2)]
                best_solution, best_coordinates = min(candidates, key=lambda item: item[0])

                if best_solution < solution_s:
                    solution_s = best_solution
                    s = best_coordinates

            if solution_s >= best_temp_solution_s:
                gamma *= step_size
            else:
                best_temp_solution_s = solution_s

        return {"coordinates": s, "solution": best_temp_solution_s}

    def goAndMoveUp(self):
        result = True
        nb_spheres = self.m_indexes[self.m_CurrentLevel - 1]
        while nb_spheres == (2 * self.m_dimension):
            self.goUp()
            nb_spheres = self.m_indexes.get(self.m_CurrentLevel - 1, 0)
        if self.m_CurrentLevel == 1:
            result = False
        elif self.NumberOfEvaluations < self.m_stopCriterion:
            self.moveUp(nb_spheres)
        return result

    def goUp(self):
        self.m_CurrentLevel -= 1

    def goDown(self):
        self.m_CurrentLevel += 1

    def moveUp(self, nbSphere):
        tempList_HyperLastLevel = self.m_stackMemory[self.m_CurrentLevel - 1]
        self.m_CurrentHyperSphere = tempList_HyperLastLevel[nbSphere]
        self.m_indexes[self.m_CurrentLevel - 1] += 1

    def save_current_state(self):
        self.history_fitness.append(self.m_bestSolutionFitness)
        self.history_nb_eval.append(self.NumberOfEvaluations)
        self.history_nb_explored_sphere.append(self.nb_explored_sphere)
        self.history_nb_decomposed_sphere.append(self.nb_decomposed_sphere)
        self.history_nb_exploited_sphere.append(self.nb_exploited_sphere)
        self.history_nb_eval_exploitation.append(self.nb_eval_exploitation)
        self.history_nb_eval_exploration.append(self.nb_eval_exploration)
        self.history_nb_visited_sphere.append(self.nb_visited_sphere)

        # Calcul du ratio d'équilibre
        total_eval = self.nb_eval_exploration + self.nb_eval_exploitation
        if total_eval > 0 and self.nb_eval_exploitation > 0:
            balance_ratio = self.nb_eval_exploration / self.nb_eval_exploitation
        else:
            balance_ratio = 1.0
        self.history_balance_ratio.append(balance_ratio)

    def sobol_sphere_sampling(self, sphere, ns, seed=None):

        # Initialisation du générateur pour le déterminisme
        rng = np.random.default_rng(seed)

        # Utiliser Sobol pour générer Ns points dans [0, 1]^1 (les alphas, pour le facteur radial)
        sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
        alphas = sampler.random(n=ns).flatten()

        # Générer Ns vecteurs normaux Z (pour la direction isotrope)
        z = rng.normal(0, 1, size=(ns, self.m_dimension))

        # Normaliser pour obtenir les directions (U sur la surface)
        normes = np.linalg.norm(z, axis=1)
        normes[normes == 0] = 1.0
        u = z / normes[:, np.newaxis]

        # Calculer le facteur radial r avec Sobol
        # r = Rayon * (alpha)^(1/D) --> pour assure l'uniformité dans le volume.
        r_factor = sphere.m_radius * (alphas ** (1 / self.m_dimension))

        # Obtenir les points p' centrés à l'origine
        p_prime = u * r_factor[:, np.newaxis]

        # Translater au centre de la sphère
        sample = p_prime + sphere.m_center

        return sample

    #Retourne la plus petite puissance de 2 supérieure ou égale à n
    @staticmethod
    def next_power_of_2(n):
        if n <= 0:
            return 1
        return 1 << (n - 1).bit_length()

    # ==================================
    # ==== Evaluation d'une fonction ===
    # ==================================
    def evaluation_fitness(self, solution):
        self.NumberOfEvaluations += 1
        f = -math.inf

        if self.NumberOfEvaluations % self.save_interval == 0:
            self.save_current_state()

        f = self.f_obj(solution)

        if f < self.m_bestSolutionFitness:
            self.nb_eval_at_best = self.NumberOfEvaluations

        return f

    def run_fda(self):
        #start_time = time.time()
        fini = True
        w_1 = 0  # poids de l'exploration
        w_2 = 0  # poids de l'exploitation
        nb_sphere_visited = 0
        nb_enter_level_max = 0

        method_actions = {
            "ils_baseline": self.intensive_local_search,
        }
        action_to_run = method_actions.get(self.ls_method)

        # Sauvegarde initiale
        self.save_current_state()

        while self.NumberOfEvaluations < self.m_stopCriterion and fini:
            list_hyper = self.decomposition_hypersphere(self.m_CurrentHyperSphere)
            for h in list_hyper:
                # h.compute_fitness(self)

                if self.promising_sphere_method in ['weighted_selection_criterion']:

                    n_opt = 5 * self.m_dimension
                    n = self.next_power_of_2(n_opt)
                    sample_points = self.sobol_sphere_sampling(sphere=h, ns=n, seed=42)
                    # Ajouter le centre de la sphere à l'ensemble des points échantillonnés
                    center_point = np.array(h.m_center).reshape(1, -1)
                    sample_points = np.vstack([sample_points, center_point])


                    # Calcul des pondérations : w_1 (poids de l'exploration) et w_2 (poids de l'exploitation)
                    w_2 = self.m_CurrentLevel / self.m_k_levels_max
                    w_1 = 1 - w_2
                    h.weighted_selection_criterion(self, sample_points, w_1, w_2)

                else:
                    # Récupérer la méthode par son nom depuis la classe Hypersphere
                    getattr(h, self.promising_sphere_method)(self)

            list_hyper.sort(key=lambda a: a.m_fitness)
            new_fitness = list_hyper[0].m_fitness
            current_fitness = self.m_CurrentHyperSphere.m_fitness

            if new_fitness > current_fitness:
                self.m_bestSolutionCoordinatesNavigation = copy.copy(list_hyper[0].m_fitnessCoordinates)
                self.m_bestSolutionFitnessNavigation = new_fitness

            list_hyper.sort(key=lambda a: a.selection_score, reverse=True)

            self.m_stackMemory[self.m_CurrentLevel] = copy.deepcopy(list_hyper)
            self.m_indexes[self.m_CurrentLevel] = 0

            self.m_CurrentHyperSphere = copy.copy(list_hyper[0])
            nb_sphere_visited += 1

            self.m_indexes[self.m_CurrentLevel] += 1
            if self.m_CurrentLevel == self.m_k_levels_max:
                temp_list_hyper_last_level = self.m_stackMemory[self.m_CurrentLevel]
                nb_sphere_visited -= 1
                for i in range(2 * self.m_dimension):
                    if self.NumberOfEvaluations >= self.m_stopCriterion: break
                    nb_enter_level_max += 1
                    self.m_CurrentHyperSphere = temp_list_hyper_last_level[i]
                    nb_sphere_visited += 1

                    result_solution = self.intensive_local_search()

                    if result_solution["solution"] < self.m_bestSolutionFitness:
                        self.m_bestSolutionCoordinates = copy.copy(result_solution["coordinates"])
                        self.m_bestSolutionFitness = result_solution["solution"]

                self.m_indexes[self.m_CurrentLevel] = 0
                fini = self.goAndMoveUp()
                if not fini:
                    print(f"ARBRE FINI : Nb sphères visitées : {nb_sphere_visited}")
            else:
                self.goDown()
        self.history_nb_visited_sphere.append(nb_sphere_visited)
        self.save_current_state()


# =================================================================
# =================== INTERFACE POUR BBOB/COCO ====================
# =================================================================
def fda_wsc_optimizer(problem, dim, budget):
    """
    Fonction wrapper qui implémente l'interface requise par l'environnement BBOB (COCO).

    Arguments:
      problem: La fonction objectif COCO (un objet 'Problem' avec la méthode 'evaluate').
      dim: La dimension du problème.
      budget: Le nombre maximal d'évaluations de fonction (FEvals).
    """
    k_levels_max = 5
    lower_bound = -5.0
    upper_bound = 5.0
    ls_method = "ils_baseline"
    promising_sphere_method = "weighted_selection_criterion"

    # Création de l'instance FdaBS (avec f comme fonction objectif)
    solver = FdaBaselinePhsWsc(
        f_obj=problem,
        dimension=dim,
        k_levels_max=k_levels_max,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        ls_method=ls_method,
        promising_sphere_method=promising_sphere_method
    )
    solver.m_stopCriterion = budget

    # Exécution de l'algorithme
    solver.run_fda()

    #return solver.m_bestSolutionCoordinates, solver.m_bestSolutionFitness
