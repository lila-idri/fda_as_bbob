from scipy.stats import norm
import numpy as np
import copy
from .constance import *


class Hypersphere:
    EPSILON_DIVISION_0 = 1e-20

    def __init__(self, dimension, center, radius, parent_belief_score):
        self.m_dimension = dimension
        self.m_center = copy.copy(center)
        self.m_radius = radius
        self.m_fitness = None
        self.m_fitnessCoordinates = []
        self.m_bestRatio = -1
        self.selection_score = math.inf
        #self.belief_score = None
        self.belief_score = 1.0 if parent_belief_score is None else None
        self.parent_belief_score = parent_belief_score
        self.children_spheres = []

        # Constantes pour les Opérateurs GA
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.tournament_rate = 3

    @staticmethod
    def calculate_distance(p1, p2):
        # tempSum = 10 + np.sum((np.array(p1) - np.array(p2)) ** 2)
        temp_sum = np.sum((np.array(p1) - np.array(p2)) ** 2)
        return math.sqrt(temp_sum)

    # Calculer la fitness d'une sphere
    def compute_fitness_only(self, context):
        self.m_radius *= inflation_coef
        context.nb_explored_sphere += 1
        self.m_fitness = context.evaluation_fitness(self.m_center)
        context.nb_eval_exploration += 1

        if self.m_fitness < context.m_bestSolutionFitness:
            context.m_bestSolutionCoordinates = copy.copy(self.m_fitnessCoordinates)
            context.m_bestSolutionFitness = self.m_fitness

    # Calculer la qualité d'une sphère
    def compute_fitness(self, context):
        self.m_radius *= inflation_coef
        context.nb_explored_sphere += 1

        solution_s = context.evaluation_fitness(self.m_center)
        distance_s = self.calculate_distance(self.m_center, context.m_bestSolutionCoordinatesNavigation)
        ratio_s = abs(solution_s / (distance_s + self.EPSILON_DIVISION_0))

        s1 = copy.copy(self.m_center)
        s2 = copy.copy(self.m_center)

        for i in range(self.m_dimension):
            s1[i] = self.m_center[i] + (self.m_radius / math.sqrt(self.m_dimension))
            s2[i] = self.m_center[i] - (self.m_radius / math.sqrt(self.m_dimension))

        solution_s1 = context.evaluation_fitness(s1)
        distance_s1 = self.calculate_distance(s1, context.m_bestSolutionCoordinatesNavigation)
        ratio_s1 = abs(solution_s1 / (distance_s1 + self.EPSILON_DIVISION_0))

        solution_s2 = context.evaluation_fitness(s2)
        distance_s2 = self.calculate_distance(s2, context.m_bestSolutionCoordinatesNavigation)
        ratio_s2 = abs(solution_s2 / (distance_s2 + self.EPSILON_DIVISION_0))

        self.m_fitness = min(solution_s, solution_s1, solution_s2)
        self.m_bestRatio = max(ratio_s, ratio_s1, ratio_s2)
        context.nb_eval_exploration += 3

        if self.m_fitness == solution_s:
            self.m_fitnessCoordinates = copy.copy(self.m_center)
        elif self.m_fitness == solution_s1:
            self.m_fitnessCoordinates = copy.copy(s1)
        else:
            self.m_fitnessCoordinates = copy.copy(s2)

        if self.m_fitness < context.m_bestSolutionFitness:
            context.m_bestSolutionCoordinates = copy.copy(self.m_fitnessCoordinates)
            context.m_bestSolutionFitness = self.m_fitness

    def compute_fitness_bis(self, context):
        self.m_radius *= inflation_coef
        context.nb_explored_sphere += 1

        solution_s = context.evaluation_fitness(self.m_center)
        distance_s = self.calculate_distance(self.m_center, context.m_bestSolutionCoordinatesNavigation)
        ratio_s = abs((context.m_bestSolutionFitnessNavigation - solution_s) / (distance_s + self.EPSILON_DIVISION_0))

        s1 = copy.copy(self.m_center)
        s2 = copy.copy(self.m_center)

        for i in range(self.m_dimension):
            s1[i] = self.m_center[i] + (self.m_radius / math.sqrt(self.m_dimension))
            s2[i] = self.m_center[i] - (self.m_radius / math.sqrt(self.m_dimension))

        solution_s1 = context.evaluation_fitness(s1)
        distance_s1 = self.calculate_distance(s1, context.m_bestSolutionCoordinatesNavigation)
        ratio_s1 = abs((context.m_bestSolutionFitnessNavigation - solution_s1) / (distance_s1 + self.EPSILON_DIVISION_0))

        solution_s2 = context.evaluation_fitness(s2)
        distance_s2 = self.calculate_distance(s2, context.m_bestSolutionCoordinatesNavigation)
        ratio_s2 = abs((context.m_bestSolutionFitnessNavigation - solution_s2) / (distance_s2 + self.EPSILON_DIVISION_0))

        self.m_fitness = min(solution_s, solution_s1, solution_s2)
        self.m_bestRatio = max(ratio_s, ratio_s1, ratio_s2)
        context.nb_eval_exploration += 3

        if self.m_fitness == solution_s:
            self.m_fitnessCoordinates = copy.copy(self.m_center)
        elif self.m_fitness == solution_s1:
            self.m_fitnessCoordinates = copy.copy(s1)
        else:
            self.m_fitnessCoordinates = copy.copy(s2)

        if self.m_fitness < context.m_bestSolutionFitness:
            context.m_bestSolutionCoordinates = copy.copy(self.m_fitnessCoordinates)
            context.m_bestSolutionFitness = self.m_fitness

    def weighted_selection_criterion(self, context, sample_points, w_exploration, w_exploitation):
        self.m_radius *= inflation_coef
        context.nb_explored_sphere += 1

        s1 = copy.copy(self.m_center)
        s2 = copy.copy(self.m_center)

        fitness_values = []

        # Evaluation de la fitness pour tous les points P
        for p in sample_points:
            solution_p = context.evaluation_fitness(p)
            fitness_values.append(solution_p)

            # Mise à jour du meilleur point GLOBAL
            if solution_p < context.m_bestSolutionFitness:
                context.m_bestSolutionCoordinates = copy.copy(p)
                context.m_bestSolutionFitness = solution_p

        # Mise à jour du compteur d'évaluation
        #context.nb_eval_exploration += len(sample_points)

        # Calcul des métriques statistiques
        fitness_array = np.array(fitness_values)
        mean_f = np.mean(fitness_array)  # Moyenne (mu)
        std_f = np.std(fitness_array)  # Ecart-type (sigma)

        # Calcul du score de sélection (m_fitness)
        # Formule pour la MAXIMISATION du Score (w_exploitation*sigma - w_exploitation*mu)
        self.selection_score = (w_exploration * std_f) - (w_exploitation * mean_f)

        # Le score de sélection remplace l'ancien critère de navigation (m_bestRatio)
        #self.m_bestRatio = self.selection_score

        # Mise à jour du meilleur point LOCAL trouvé dans la sphère
        min_index = np.argmin(fitness_array)
        self.m_fitness = np.min(fitness_array)
        self.m_fitnessCoordinates = copy.copy(sample_points[min_index])

    def weighted_selection_criterion_v0(self, context, w_exploration, w_exploitation):
        self.m_radius *= inflation_coef
        context.nb_explored_sphere += 1
        sample_points = []

        s1 = copy.copy(self.m_center)
        s2 = copy.copy(self.m_center)

        for i in range(self.m_dimension):
            s1[i] = self.m_center[i] + (self.m_radius / math.sqrt(self.m_dimension))
            s2[i] = self.m_center[i] - (self.m_radius / math.sqrt(self.m_dimension))

        sample_points.append(s1)
        sample_points.append(s2)

        fitness_values = []

        # Evaluation de la fitness pour tous les points P
        for p in sample_points:
            solution_p = context.evaluation_fitness(p)
            fitness_values.append(solution_p)

            # Mise à jour du meilleur point GLOBAL
            if solution_p < context.m_bestSolutionFitness:
                context.m_bestSolutionCoordinates = copy.copy(p)
                context.m_bestSolutionFitness = solution_p

        # Mise à jour du compteur d'évaluation
        # context.nb_eval_exploration += len(sample_points)

        # Calcul des métriques statistiques
        fitness_array = np.array(fitness_values)
        mean_f = np.mean(fitness_array)  # Moyenne (mu)
        std_f = np.std(fitness_array)  # Ecart-type (sigma)

        # Calcul du score de sélection (m_fitness)
        # Formule pour la MAXIMISATION du Score (w_exploitation*sigma - w_exploitation*mu)
        self.selection_score = (w_exploration * std_f) - (w_exploitation * mean_f)

        # Le score de sélection remplace l'ancien critère de navigation (m_bestRatio)
        # self.m_bestRatio = self.selection_score

        # Mise à jour du meilleur point LOCAL trouvé dans la sphère
        min_index = np.argmin(fitness_array)
        self.m_fitness = np.min(fitness_array)
        self.m_fitnessCoordinates = copy.copy(sample_points[min_index])


    def improvement_probability_selection(self, context, sample_points):
        self.m_radius *= inflation_coef
        context.nb_explored_sphere += 1

        fitness_values = []
        for p in sample_points:
            solution_p = context.evaluation_fitness(p)
            fitness_values.append(solution_p)

            # Mise à jour du meilleur point GLOBAL
            if solution_p < context.m_bestSolutionFitness:
                context.m_bestSolutionCoordinates = copy.copy(p)
                context.m_bestSolutionFitness = solution_p

        # Mise à jour du compteur d'évaluation
        context.nb_eval_exploration += len(sample_points)

        # Calcul des métriques statistiques (mu et sigma)
        fitness_array = np.array(fitness_values)
        mean_f = np.mean(fitness_array)  # mu
        std_f = np.std(fitness_array)  # sigma

        # Calcul du score PI
        f_best = context.m_bestSolutionFitness  # meilleure fitness global actuelle

        if std_f == 0.0:
            # si tous les points sont identiques, la probabilité est 0 ou 1
            # Pour une minimisation, on attribue un score faible si f > f_best
            self.selection_score = 0.0
        else:
            # Formule PI: P(f < f_best) = Phi((f_best - mu) / sigma)
            z = (f_best - mean_f) / std_f
            self.selection_score = norm.cdf(z)

        # Mise à jour des coordonnées et fitness locaux
        min_index = np.argmin(fitness_array)
        self.m_fitness = np.min(fitness_array)
        self.m_fitnessCoordinates = copy.copy(sample_points[min_index])

        # Note: On peut stocker m_bestRatio = self.selection_score pour la compatibilité
        # self.m_bestRatio = self.selection_score

    def fuzzy_belief_selection(self, context, sample_points, gamma_coef):
        """
        Calcule le score de croyance floue (Bel) de la sphère enfant
        en intégrant la croyance du parent (Bel(α'_it)) et la nouvelle évidence locale (m(A_it))."""

        
        # Évaluation des points et mise à jour du BSF
        fitness_values = []
        for p in sample_points:
            solution_p = context.evaluation_fitness(p)
            fitness_values.append(solution_p)

            # Mise à jour du meilleur point GLOBAL (F*)
            if solution_p < context.m_bestSolutionFitness:
                context.m_bestSolutionCoordinates = copy.copy(p)
                context.m_bestSolutionFitness = solution_p

        # Mise à jour du compteur d'évaluation de fcontion
        context.nb_eval_exploration += len(sample_points)

        F_star = context.m_bestSolutionFitness + self.EPSILON_DIVISION_0
        # Récupération de la croyance du parent (Bel(α'_it)).
        #parent_belief = self.parent_sphere.belief_score if self.parent_sphere is not None else 1.0

        # Calcul de la Mesure d'Évidence Floue (m(A_it))
        sum_evidence = 0.0
        for F_j in fitness_values:
            # Terme: (F_j / F*) * exp(1 - F_j / F*)
            ratio = F_j / F_star
            term = ratio * math.exp(1.0 - ratio)
            sum_evidence += term

        # m(A_it) = (1/s) * Somme[Termes]
        evidence_measure = sum_evidence / len(sample_points)

        # Calcul de la Mesure de Croyance Floue (Bel(α_it))
        # Bel(α_it) = γ * Bel(α'_it) + (1 − γ) * m(A_it)
        self.belief_score  = (gamma_coef * self.parent_belief_score) + ((1.0 - gamma_coef) * evidence_measure)

        # Le score de sélection de la sphère est la mesure de croyance.
        self.selection_score = self.belief_score

        # Mise à jour des coordonnées et fitness locaux
        min_index = np.argmin(fitness_values)
        self.m_fitness = np.min(fitness_values)
        self.m_fitnessCoordinates = copy.copy(sample_points[min_index])



    # Opérateurs GA
    def tournament_selection(self, population, k):
        # Sélectionne k individus au hasard et le gagnant est celui avec la meilleure fitness
        contenders = np.random.choice(population, size=k, replace=False)
        winner = min(contenders, key=lambda ind: ind['fitness'])
        return winner

    def arithmetic_crossover(self, parent1_coords, parent2_coords, dim, alpha=0.5):
        #Child = alpha*P1 + (1-alpha)*P2
        return alpha * parent1_coords + (1.0 - alpha) * parent2_coords

    def uniform_mutation(self, coords, mutation_rate, dim):
        #Ajouter un bruit uniforme si le taux de mutation est atteint
        mutated_coords = copy.copy(coords)
        for i in range(dim):
            if np.random.rand() < mutation_rate:
                mutated_coords[i] += np.random.uniform(-self.m_radius / 2.0, self.m_radius / 2.0)
        return mutated_coords

    def run_local_ga(self, center, max_eval_budget, context):
        current_evals = 0
        pop_size = context.m_dimension * 2  # P_init = 2D
        best_fitness = float('inf')
        best_coords = center
        all_evaluated_fitnesses = []

        current_population = []

        # Initialisation : points aléatoires dans la Sphère
        for _ in range(pop_size):
            coords = center + (np.random.rand(context.m_dimension) - 0.5) * 2 * context.m_radius
            fitness = context.evaluation_fitness(coords)
            all_evaluated_fitnesses.append(fitness)
            current_population.append({'coords': coords, 'fitness': fitness})

            if fitness < best_fitness:
                best_fitness = fitness
                best_coords = coords
        current_evals = pop_size

        while current_evals < max_eval_budget:
            pop_size_next = pop_size * 2
            new_evals_needed = pop_size_next

            if current_evals + new_evals_needed > max_eval_budget:
                break

            new_generation = []

            # Générer la nouvelle population
            while len(new_generation) < pop_size_next:
                # Sélection
                parent1 = self.tournament_selection(current_population, self.tournament_rate)
                parent2 = self.tournament_selection(current_population, self.tournament_rate)

                # CROISEMENT et MUTATION
                if np.random.rand() < self.crossover_rate:
                    child_coords = self.arithmetic_crossover(parent1['coords'], parent2['coords'], dim)
                else:
                    # Si pas de croisement, l'enfant est une copie mutée du meilleur parent
                    best_parent = parent1 if parent1['fitness'] < parent2['fitness'] else parent2
                    child_coords = copy.copy(best_parent['coords'])

                child_coords = self.uniform_mutation(child_coords, self.m_radius, self.mutation_rate, dim)

                # ÉVALUATION
                fitness = context.evaluation_fitness(child_coords)
                all_evaluated_fitnesses.append(fitness)
                current_evals += 1

                new_generation.append({'coords': child_coords, 'fitness': fitness})

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_coords = child_coords

            # Garder l'élite (le meilleur individu global) et remplacer la population
            # Combine ancienne et nouvelle population
            combined_pop = current_population + new_generation

            # Trie l'ensemble combiné par fitness (ascendant, le meilleur en premier)
            combined_pop.sort(key=lambda ind: ind['fitness'])

            # La nouvelle population pour la prochaine itération est constituée des meilleurs 'pop_size_next'
            current_population = combined_pop[:pop_size_next]
            pop_size = pop_size_next  # Mise à jour de la taille de la population courante

        context.nb_eval_exploration += current_evals

        return best_fitness, best_coords, all_evaluated_fitnesses


    def fuzzy_belief_with_ga_selection(self, context, sample_points, gamma_coef):
        """
        Calcule le score de croyance floue (Bel) de la sphère enfant
        en intégrant la croyance du parent (Bel(α'_it)) et la nouvelle évidence locale (m(A_it))."""

        # Évaluation des points et mise à jour du BSF
        fitness_values = []
        for p in sample_points:
            solution_p = context.evaluation_fitness(p)
            fitness_values.append(solution_p)

            # Mise à jour du meilleur point GLOBAL (F*)
            if solution_p < context.m_bestSolutionFitness:
                context.m_bestSolutionCoordinates = copy.copy(p)
                context.m_bestSolutionFitness = solution_p

        # Mise à jour du compteur d'évaluation de fcontion
        context.nb_eval_exploration += len(sample_points)

        F_star = context.m_bestSolutionFitness + self.EPSILON_DIVISION_0
        # Récupération de la croyance du parent (Bel(α'_it)).
        # parent_belief = self.parent_sphere.belief_score if self.parent_sphere is not None else 1.0

        # Calcul de la Mesure d'Évidence Floue (m(A_it))
        sum_evidence = 0.0
        for F_j in fitness_values:
            # Terme: (F_j / F*) * exp(1 - F_j / F*)
            ratio = F_j / F_star
            term = ratio * math.exp(1.0 - ratio)
            sum_evidence += term

        # m(A_it) = (1/s) * Somme[Termes]
        evidence_measure = sum_evidence / len(sample_points)

        # Calcul de la Mesure de Croyance Floue (Bel(α_it))
        # Bel(α_it) = γ * Bel(α'_it) + (1 − γ) * m(A_it)
        self.belief_score = (gamma_coef * self.parent_belief_score) + ((1.0 - gamma_coef) * evidence_measure)

        # Le score de sélection de la sphère est la mesure de croyance.
        self.selection_score = self.belief_score

        # Mise à jour des coordonnées et fitness locaux
        min_index = np.argmin(fitness_values)
        self.m_fitness = np.min(fitness_values)
        self.m_fitnessCoordinates = copy.copy(sample_points[min_index])





