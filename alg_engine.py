import copy

import numpy
import numpy as np
import random
import re
import pickle
from sklearn.cluster import KMeans
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.indicators.igd import IGD
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import FloatRandomSampling, BinaryRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

# operators for MOTSP
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.termination.default import DefaultSingleObjectiveTermination
from MOTSP.tsp_operator import StartFromZeroRepair

from sklearn import *
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible
from pymoo.indicators.hv import HV

from ga_src.env import GENERATION, MODULE, MAX_ARCHIVE, N_TRIALS, N_THREAD, POP_SIZE, OTHER, TP
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA

from ga_src.env import POP_SIZE
from ga_src.env import TP


from pymoo.core.repair import Repair


class ConsiderMaximumWeightRepair(Repair):

    def _do(self, problem, Z, **kwargs):

        # maximum capacity for the problem
        Q = problem.C

        # the corresponding weight of each individual
        weights = (Z * problem.W).sum(axis=1)

        # now repair each indvidiual i
        for i in range(len(Z)):

            # the packing plan for i
            z = Z[i]

            # while the maximum capacity violation holds
            while weights[i] > Q:

                # randomly select an item currently picked
                item_to_remove = np.random.choice(np.where(z)[0])

                # and remove it
                z[item_to_remove] = False

                # adjust the weight
                weights[i] -= problem.W[item_to_remove]

        return Z

class AlgoProfile:
    """
    Monitor the algorithm over the running of each trial to calculate some characteristics of the candidate MOEA, e.g.,
    the convergence speed, diversity change between generations, it characterizes the dynamic features of MOEA, which
    can be served as input information for LLM to further improve algorithm design
    """
    def __init__(self):
        pass

    def compute_single_feature(self):
        pass

    def compute(self,):
        pass


class AlgoPerformance:
    """
    Aggregate all the (final) performance of a candidate MOEA on a set of multi-objective optimization problems over
    a number of independent runs
    """
    def __init__(self):
        pass

    def _compute_on_single_trial(self):
        # compute performance of an MOEA on single problem of a single trial, e.g., hypervolume (HV)
        pass

    def compute(self):
        # aggregate all trials of all problem instances of an algorithm
        pass

def motspsearch(pops: dict, D_lst: np.ndarray, POP_SIZE: int, N_P: int):
    """
    Evolves the population by integrating advanced selection, crossover, and mutation strategies to generate a new population
    that aims to minimize travel costs across multiple objectives.

    :param pops: Dictionary with current population data, containing:
                 - 'individuals': numpy.ndarray of shape (POP_SIZE, N_P) with integer sequences.
                 - 'rankings': numpy.ndarray of shape (POP_SIZE,) with ranking scores.
    :param D_lst: 3D numpy.ndarray of shape (N_O, N_P, N_P) representing multiple cost matrices.
    :param POP_SIZE: Integer, total number of individuals in the population.
    :param N_P: Integer, number of cities to be visited.

    :return: numpy.ndarray of shape (POP_SIZE, N_P) with new population of individual sequences.
    """
    def select_parents(ranking, num_parents):
        """Selects parent individuals based on their rankings using a tournament selection strategy."""
        chosen = []
        for _ in range(num_parents):
            participants = np.random.choice(np.arange(len(ranking)), size=min(4, len(ranking)), replace=False)
            selected = participants[np.argmin(ranking[participants])]
            chosen.append(selected)
        return chosen

    def crossover(parent1, parent2):
        """Performs edge recombination crossover (ERX) on two parents to produce two offspring."""
        def build_adjacency_matrix(parent):
            adj_matrix = {i: set() for i in range(N_P)}
            length = len(parent)
            for i in range(length):
                left = parent[i - 1]
                right = parent[(i + 1) % length]
                adj_matrix[parent[i]].update([left, right])
            return adj_matrix

        def recombine(parent1, parent2):
            adj_matrix1 = build_adjacency_matrix(parent1)
            adj_matrix2 = build_adjacency_matrix(parent2)
            for key in adj_matrix1:
                adj_matrix1[key].update(adj_matrix2[key])

            current = np.random.choice(parent1)
            offspring = [current]
            while len(offspring) < N_P:
                if adj_matrix1[current]:
                    next_city = min(adj_matrix1[current], key=lambda x: len(adj_matrix1[x]))
                    offspring.append(next_city)
                    for adj in adj_matrix1.values():
                        adj.discard(current)
                    current = next_city
                else:
                    remaining_cities = set(range(N_P)) - set(offspring)
                    current = np.random.choice(list(remaining_cities))
                    offspring.append(current)
            return offspring

        offspring1 = recombine(parent1, parent2)
        offspring2 = recombine(parent2, parent1)
        return offspring1, offspring2

    def mutate(individual):
        """Mutates an individual by swapping two cities with a low probability."""
        if np.random.rand() < 0.1:  # mutation probability
            swap_indices = np.random.choice(N_P, size=2, replace=False)
            individual[swap_indices[0]], individual[swap_indices[1]] = individual[swap_indices[1]], individual[swap_indices[0]]

    # Selection
    num_parents = POP_SIZE // 2
    parent_indices = select_parents(pops['rankings'], num_parents * 2)
    np.random.shuffle(parent_indices)

    # Crossover
    new_pop = []
    for i in range(0, len(parent_indices), 2):
        parent1, parent2 = pops['individuals'][parent_indices[i]], pops['individuals'][parent_indices[i+1]]
        offspring1, offspring2 = crossover(parent1, parent2)
        new_pop.append(offspring1)
        new_pop.append(offspring2)

    # Mutation
    for individual in new_pop:
        mutate(individual)

    # Ensure all are valid permutations
    new_pop = np.array(new_pop, dtype=np.int32)

    return new_pop
def mokpsearch(pops: dict, W: np.ndarray, C: int, V: np.ndarray, POP_SIZE: int, N_P: int) -> np.ndarray:
    """
    Evolves a population for the multi-objective knapsack problem using advanced selection, crossover, and mutation strategies.

    :param pops: Current population and their rankings.
    :param W: Weights of each item.
    :param C: Maximum capacity of the knapsack.
    :param V: Profits for each item across multiple objectives.
    :param POP_SIZE: Size of the population.
    :param N_P: Number of decision variables (items).
    :return: New population matrix with improved potential solutions.
    """

    def select_parents(ranked_individuals):
        """Selects parents using a hybrid of tournament and elitism selection."""
        parents = []
        elite_size = max(2, POP_SIZE // 10)  # Ensure at least top 10% of individuals are considered elite
        elite_indices = np.argsort(ranked_individuals[:, -1])[:elite_size]
        for _ in range(POP_SIZE // 2):
            if np.random.rand() < 0.20:  # 20% chance to select purely from elites
                i1, i2 = np.random.choice(elite_indices, 2, replace=False)
            else:
                i1, i2 = np.random.choice(POP_SIZE, 2, replace=False)
            if ranked_individuals[i1][-1] < ranked_individuals[i2][-1]:
                parents.append(ranked_individuals[i1][:-1])
            else:
                parents.append(ranked_individuals[i2][:-1])
        return np.array(parents)

    def crossover(parents):
        """Performs uniform crossover with a probability of swapping each gene."""
        children = []
        for i in range(0, len(parents), 2):
            p1, p2 = parents[i], parents[i+1]
            mask = np.random.rand(N_P) < 0.5
            child1 = np.where(mask, p1, p2)
            child2 = np.where(mask, p2, p1)
            children.extend([child1, child2])
        return np.array(children)

    def mutate(offspring):
        """Mutates offspring by flipping bits with a dynamic mutation rate."""
        mutation_rate = 0.05  # Base mutation rate
        for i in range(len(offspring)):
            if np.random.rand() < mutation_rate * (1 - (i / len(offspring))):  # Decrease mutation rate for better individuals
                idx = np.random.randint(0, N_P)
                offspring[i][idx] = 1 - offspring[i][idx]
        return offspring

    def repair(offspring):
        """Repairs offspring to meet knapsack constraints using a greedy approach."""
        for i in range(len(offspring)):
            while np.dot(W, offspring[i]) > C:
                overweight_indices = np.where(offspring[i] == 1)[0]
                # Prefer to drop items with lower profit/weight ratio
                ratios = np.sum(V[overweight_indices], axis=1) / W[overweight_indices]  # Sum profits across all objectives
                drop_idx = overweight_indices[np.argmin(ratios)]
                offspring[i][drop_idx] = 0
        return offspring

    # Subsection Selection
    ranked_individuals = np.hstack((pops['individuals'], pops['rankings'].reshape(-1, 1)))
    parents = select_parents(ranked_individuals)

    # Subsection Crossover
    offspring = crossover(parents)

    # Subsection Mutation
    mutated_offspring = mutate(offspring)

    # Subsection Checking
    new_pop = repair(mutated_offspring)
    return new_pop.astype(np.int32)

def mopsearch(pops: dict, search_trajectory: dict, xlb: np.ndarray, xub: np.ndarray, POP_SIZE: int, N_P: int,
                    current_gen: int, max_gen: int) -> np.ndarray:
    """
    Evolves a new generation of individuals for multi-objective optimization problems using a hybrid strategy.
    :param pops: Current population and rankings
    :param search_trajectory: Historical search trajectory
    :param xlb: Lower bounds of decision variables
    :param xub: Upper bounds of decision variables
    :param POP_SIZE: Population size
    :param N_P: Number of decision variables
    :param current_gen: Current generation number
    :param max_gen: Maximum number of generations
    :return: New population as a numpy.ndarray
    """

    def differential_evolution_crossover(target, donor, cr=0.9):
        """
        Performs Differential Evolution crossover.
        """
        trial = np.copy(target)
        indices = np.arange(N_P)
        np.random.shuffle(indices)
        for i in indices[:int(N_P * cr)]:
            trial[i] = donor[i]
        return np.clip(trial, xlb, xub)

    def polynomial_mutation(individual, eta=20):
        """
        Applies Polynomial mutation to an individual.
        """
        for i in range(N_P):
            if np.random.rand() < 1.0 / N_P:
                delta = np.random.rand()
                if delta < 0.5:
                    delta_q = (2.0 * delta) ** (1.0 / (eta + 1)) - 1.0
                else:
                    delta_q = 1.0 - (2.0 * (1.0 - delta)) ** (1.0 / (eta + 1))
                individual[i] += delta_q * (xub[i] - xlb[i])
                individual[i] = np.clip(individual[i], xlb[i], xub[i])
        return individual

    def tournament_selection(population, rankings, tournament_size=2):
        """
        Selects individuals based on tournament selection.
        """
        winners = []
        for _ in range(POP_SIZE):
            participants = np.random.choice(range(POP_SIZE), size=tournament_size, replace=False)
            participants_ranking = rankings[participants]
            winner_index = participants[np.argmin(participants_ranking)]
            winners.append(population[winner_index])
        return np.array(winners)

    # Part 1: Selection
    parents = tournament_selection(pops['individuals'], pops['rankings'])

    # Part 2: Crossover
    offspring = np.empty((POP_SIZE, N_P))
    for i in range(0, POP_SIZE, 2):
        target, donor = parents[i], parents[(i + 1) % POP_SIZE]
        offspring[i] = differential_evolution_crossover(target, donor)
        offspring[i + 1] = differential_evolution_crossover(donor, target)

    # Part 3: Mutation
    for i in range(POP_SIZE):
        offspring[i] = polynomial_mutation(offspring[i])

    # Validation
    new_pops = np.clip(offspring, xlb, xub)

    return new_pops
best_alg = {
    'MOP': mopsearch,
    'MOKP': mokpsearch,
    'MOTSP': motspsearch,
}

import numpy as np


def test_next_generation(pops, N_P, D_lst, POP_SIZE):

    """
    Novel evolutionary algorithm that uses grouped path crossover and adaptive mutation for multi-objective TSP.
    """
    import numpy as np
    def calculate_costs(path):
        """
        Calculate total cost as a combination of all objectives for a given path.
        """
        total_costs = np.zeros((len(D_lst),))
        for i in range(N_P - 1):
            for o in range(len(D_lst)):
                total_costs[o] += D_lst[o][path[i]][path[i + 1]]
        # Considering returning to the start point
        for o in range(len(D_lst)):
            total_costs[o] += D_lst[o][path[-1]][path[0]]
        return np.sum(total_costs)

    def crossover(parent1, parent2):
        """
        Performs ordered crossover between two parents.
        """
        start, end = sorted(np.random.choice(range(N_P), 2, replace=False))
        child = [None] * N_P
        # Insert the slice from parent1 into the child
        child[start:end + 1] = parent1[start:end + 1]
        fill_index = (end + 1) % N_P
        # Fill the rest using cities from parent2
        for city in parent2:
            if city not in child:
                child[fill_index] = city
                fill_index = (fill_index + 1) % N_P
        return child

    def mutate(path):
        """
        Performs a swap mutation on a path.
        """
        idx1, idx2 = np.random.choice(range(N_P), 2, replace=False)
        path[idx1], path[idx2] = path[idx2], path[idx1]
        return path

    def adaptive_mutation(path):
        """
        Mutates path if its performance is below a certain threshold.
        """
        if calculate_costs(path) > np.mean([calculate_costs(ind) for ind in individuals]):
            return mutate(path)
        return path

    individuals = pops['individuals']
    rankings = pops['rankings']
    # New population
    new_pops = []
    elite_size = POP_SIZE // 10
    elite_individuals = individuals[np.argsort(rankings)[:elite_size]]
    # Elitism: directly pass the top elite_size individuals
    new_pops.extend(elite_individuals)
    # Crossover and mutation to create new offspring
    while len(new_pops) < POP_SIZE:
        parents = individuals[np.random.choice(range(POP_SIZE), 2, replace=False)]
        child = crossover(parents[0], parents[1])
        child = adaptive_mutation(child)
        new_pops.append(child)
    return new_pops

# investigate for time:
# class LLM_MOEA(GeneticAlgorithm):
#     def __init__(self, offspring_generation_code, sampling, **kwargs):
#         # kwargs['save_history'] = False
#         super().__init__(
#             pop_size=POP_SIZE,
#             sampling=sampling,
#             survival=RankAndCrowding(),
#             output=MultiObjectiveOutput(),
#             advance_after_initial_infill=True,
#             **kwargs
#         )
#         self.nds = NonDominatedSorting()
#         self.tournament_type = 'comp_by_dom_and_crowding'
#         self.next_generation = best_alg[TP]
#
#     def obtain_ranking(self, objectives):
#         fronts = self.nds.do(objectives)
#         length = len(objectives)
#         rankings = np.empty(length, dtype=np.int32)
#         for rank, ids in enumerate(fronts):
#             rankings[ids] = rank + 1
#         return rankings
#
#
#     def _infill(self):
#         # override
#         # :param 'pops', current population consists of {{'individuals': numpy.ndarray with shape(POP_SIZE, N_P), 'fitness': numpy.ndarray with shape(POP_SIZE,)}}
#         # :param 'search_trajectory', results gained along the evolutionary search that consists of {{'individuals': numpy.ndarray with shape(*, N_P), 'fitness': numpy.ndarray with shape(*, N_P)}} (can be empty arrays)
#         # :return: 'new_pops', new population in the format of numpy.ndarray (with shape(POP_SIZE, N_P)) that may achieve superior results on the optimization problems
#         PSIZE = self.pop_size
#         NP = self.problem.n_var
#         n_gen = self.n_gen
#
#         offsprings = None
#         if TP == 'MOP':
#             pops = {
#                 'individuals': np.array(self.pop.get('X')),
#                 f'{MODULE}': self.obtain_ranking(self.pop.get('F'))
#             }
#             xlb = self.problem.xl
#             xub = self.problem.xu
#             offsprings_array = self.next_generation(pops, None, xlb, xub, PSIZE, NP, n_gen, GENERATION)
#             for i in range(self.pop_size):
#                 offsprings_array[i,:] = np.clip(offsprings_array[i,:], xlb, xub)
#             offsprings = Population.new("X", offsprings_array)
#         elif TP == 'MOKP':
#             pops = {
#                 'individuals': np.array(self.pop.get('X')).astype(int),
#                 f'{MODULE}': self.obtain_ranking(self.pop.get('F'))
#             }
#             W, C, V = self.problem.W, self.problem.C, self.problem.V
#             offsprings_array = self.next_generation(pops, W, C, V, PSIZE, NP)
#             offsprings = Population.new("X", np.array(offsprings_array).astype(bool))
#         elif TP == 'MOTSP':
#             pops = {
#                 'individuals': np.array(self.pop.get('X')).astype(int),
#                 f'{MODULE}': self.obtain_ranking(self.pop.get('F'))
#             }
#             offsprings_array = self.next_generation(pops, None, PSIZE, NP)
#             offsprings = Population.new("X", np.array(offsprings_array).astype(int))
#
#         return offsprings
class LLM_MOEA(GeneticAlgorithm):
    def __init__(self, offspring_generation_code, sampling, **kwargs):
        # kwargs['save_history'] = False
        super().__init__(
            pop_size=POP_SIZE,
            sampling=sampling,
            survival=RankAndCrowding(),
            output=MultiObjectiveOutput(),
            advance_after_initial_infill=True,
            **kwargs
        )
        self.nds = NonDominatedSorting()
        self.tournament_type = 'comp_by_dom_and_crowding'
        self.offspring_generation_code = offspring_generation_code
        if self.offspring_generation_code is not None:
            self.compiled_code = compile(self.offspring_generation_code, '<string>', 'exec')
            self.local_namespace = {}
            exec(self.compiled_code, globals().copy(), self.local_namespace)
            self.next_generation = self.local_namespace['next_generation']
        else:
            assert False, 'offspring_generation_code is None'
        self.search_trajectory = {'individuals': None, 'objectives': None, f'{MODULE}': None}

    def obtain_ranking(self, objectives):
        fronts = self.nds.do(objectives)
        length = len(objectives)
        rankings = np.empty(length, dtype=np.int32)
        for rank, ids in enumerate(fronts):
            rankings[ids] = rank + 1
        return rankings

    def _post_advance(self):

        # update the current optimum of the algorithm
        self._set_optimum()

        # update the current termination condition of the algorithm
        self.termination.update(self)

        # display the output if defined by the algorithm
        self.display(self)

        # if a callback function is provided it is called after each iteration
        self.callback(self)

        if self.save_history:
            _hist, _callback, _display = self.history, self.callback, self.display

            self.history, self.callback, self.display = None, None, None
            # obj = copy.deepcopy(self)
            obj = copy.deepcopy(self.pop)

            self.history, self.callback, self.display = _hist, _callback, _display
            self.history.append(obj)

        self.n_iter += 1
    def _infill(self):
        # override
        # :param 'pops', current population consists of {{'individuals': numpy.ndarray with shape(POP_SIZE, N_P), 'fitness': numpy.ndarray with shape(POP_SIZE,)}}
        # :param 'search_trajectory', results gained along the evolutionary search that consists of {{'individuals': numpy.ndarray with shape(*, N_P), 'fitness': numpy.ndarray with shape(*, N_P)}} (can be empty arrays)
        # :return: 'new_pops', new population in the format of numpy.ndarray (with shape(POP_SIZE, N_P)) that may achieve superior results on the optimization problems
        PSIZE = self.pop_size
        NP = self.problem.n_var
        n_gen = self.n_gen

        offsprings = None
        if TP == 'MOP':
            pops = {
                'individuals': np.array(self.pop.get('X')),
                f'{MODULE}': self.obtain_ranking(self.pop.get('F'))
            }
            # search_trajectory = {'individuals': np.array([]), 'fitness': np.array([])}  # Under construction
            xlb = self.problem.xl
            xub = self.problem.xu
            c_search_trajectory = {
                'individuals': self.search_trajectory['individuals'],
                f'{MODULE}': self.search_trajectory[f'{MODULE}']
            }
            offsprings_array = self.next_generation(pops, c_search_trajectory, xlb, xub, PSIZE, NP, n_gen, GENERATION)
            # offsprings_array = test_next_generation(pops, c_search_trajectory, xlb, xub, PSIZE, NP, n_gen, GENERATION) # for testing
            for i in range(self.pop_size):
                offsprings_array[i,:] = np.clip(offsprings_array[i,:], xlb, xub)
            offsprings = Population.new("X", offsprings_array)

            # update search_trajectory
            if self.search_trajectory['individuals'] is not None:
                self.search_trajectory['individuals'] = np.append(self.search_trajectory['individuals'], self.pop.get('X'), axis=0)[:MAX_ARCHIVE]
                self.search_trajectory['objectives'] = np.append(self.search_trajectory['objectives'], self.pop.get('F'), axis=0)[:MAX_ARCHIVE]
            else:
                self.search_trajectory['individuals'] = np.copy(self.pop.get('X'))[:MAX_ARCHIVE]
                self.search_trajectory['objectives'] = np.copy(self.pop.get('F'))[:MAX_ARCHIVE]
            self.search_trajectory[f'{MODULE}'] = self.obtain_ranking(self.search_trajectory['objectives'])
        elif TP == 'MOKP':
            pops = {
                'individuals': np.array(self.pop.get('X')).astype(int),
                f'{MODULE}': self.obtain_ranking(self.pop.get('F'))
            }
            W, C, V = self.problem.W, self.problem.C, self.problem.V
            if self.offspring_generation_code is not None:
                offsprings_array = self.next_generation(pops, W, C, V, PSIZE, NP)
            else:
                # offsprings_array = next_generation_test(pops, W, C, V, PSIZE, NP)
                offsprings_array = None
            offsprings = Population.new("X", np.array(offsprings_array).astype(bool))
        elif TP == 'MOTSP':
            pops = {
                'individuals': np.array(self.pop.get('X')).astype(int),
                f'{MODULE}': self.obtain_ranking(self.pop.get('F'))
            }
            D_lst = np.array([tsp.D for tsp in self.problem.tsps])
            offsprings_array = self.next_generation(pops, D_lst, PSIZE, NP)
            # offsprings_array = test_next_generation(pops, D_lst, PSIZE, NP)
            for check in offsprings_array:
                if set(check) != set(range(NP)):
                    raise ValueError("Invalid permutation in the offspring.")
            offsprings = Population.new("X", np.array(offsprings_array).astype(int))

        return offsprings

def get_llmcode(algo_code, verbose):
    if TP == 'MOP':
        return LLM_MOEA(algo_code, FloatRandomSampling(), verbose=verbose)
    elif TP == 'MOKP':
        return LLM_MOEA(algo_code, BinaryRandomSampling(), verbose=verbose, eliminate_duplicates=True)
    elif TP == 'MOTSP':
        return LLM_MOEA(algo_code, PermutationRandomSampling(), verbose=verbose, eliminate_duplicates=True)
    else:
        assert False, 'no such problem to handle'

def get_nsga2(pop_size):
    if TP == 'MOP':
        return NSGA2(pop_size=pop_size)
    elif TP == 'MOKP':
        return NSGA2(
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            repair=ConsiderMaximumWeightRepair(),
            eliminate_duplicates=True
        )
    elif TP == 'MOTSP':
        return NSGA2(
            pop_size=pop_size,
            sampling=PermutationRandomSampling(),
            crossover=OrderCrossover(),
            mutation=InversionMutation(),
            repair=StartFromZeroRepair(),
            eliminate_duplicates=True,
        )
    else:
        assert False, 'no such problem to handle'

def get_agemoea(pop_size):
    if TP == 'MOP':
        return AGEMOEA(pop_size=pop_size)
    elif TP == 'MOKP':
        return AGEMOEA(
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            repair=ConsiderMaximumWeightRepair(),
            eliminate_duplicates=True
        )
    elif TP == 'MOTSP':
        return AGEMOEA(
            pop_size=pop_size,
            sampling=PermutationRandomSampling(),
            crossover=OrderCrossover(),
            mutation=InversionMutation(),
            repair=StartFromZeroRepair(),
            eliminate_duplicates=True
        )
    else:
        assert False, 'no such problem to handle'

def get_smsemoa(pop_size):
    if TP == 'MOP':
        return SMSEMOA(pop_size=pop_size)
    elif TP == 'MOKP':
        return SMSEMOA(
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            repair=ConsiderMaximumWeightRepair(),
            eliminate_duplicates=True
        )
    elif TP == 'MOTSP':
        return SMSEMOA(
            pop_size=pop_size,
            sampling=PermutationRandomSampling(),
            crossover=OrderCrossover(),
            mutation=InversionMutation(),
            repair=StartFromZeroRepair(),
            eliminate_duplicates=True
        )
    else:
        assert False, 'no such problem to handle'

def get_moead(pop_size, NO, seed):
    ref_dirs = get_reference_directions("energy", NO, pop_size, seed=seed)
    if TP == 'MOP':
        return MOEAD(ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7)
    elif TP == 'MOKP':
        return MOEAD(
            ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            repair=ConsiderMaximumWeightRepair(),
            # eliminate_duplicates=True
        )
    elif TP == 'MOTSP':
        return MOEAD(
            ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
            sampling=PermutationRandomSampling(),
            crossover=OrderCrossover(),
            mutation=InversionMutation(),
            repair=StartFromZeroRepair(),
            # eliminate_duplicates=True
        )
    else:
        assert False, 'no such problem to handle'

def get_ctaea(pop_size, NO, seed):
    ref_dirs = get_reference_directions("energy", NO, pop_size, seed=seed)
    if TP == 'MOP':
        return CTAEA(ref_dirs)
    elif TP == 'MOKP':
        return CTAEA(
            ref_dirs,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            repair=ConsiderMaximumWeightRepair(),
            eliminate_duplicates=True
        )
    elif TP == 'MOTSP':
        return CTAEA(
            ref_dirs,
            sampling=PermutationRandomSampling(),
            crossover=OrderCrossover(),
            mutation=InversionMutation(),
            repair=StartFromZeroRepair(),
            eliminate_duplicates=True
        )
    else:
        assert False, 'no such problem to handle'

def get_rvea(pop_size, NO, seed):
    ref_dirs = get_reference_directions("energy", NO, pop_size, seed=seed)
    if TP == 'MOP':
        return RVEA(ref_dirs)
    elif TP == 'MOKP':
        return RVEA(
            ref_dirs,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            repair=ConsiderMaximumWeightRepair(),
            eliminate_duplicates=True
        )
    elif TP == 'MOTSP':
        return RVEA(
            ref_dirs,
            sampling=PermutationRandomSampling(),
            crossover=OrderCrossover(),
            mutation=InversionMutation(),
            repair=StartFromZeroRepair(),
            eliminate_duplicates=True
        )
    else:
        assert False, 'no such problem to handle'