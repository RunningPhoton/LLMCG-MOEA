import copy
import warnings
from multiprocessing import Pool
import os
import numpy as np
import re
import pickle
from pymoo.core.problem import Problem
from pymoo.indicators.igd import IGD
from pymoo.problems import get_problem
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from sklearn import *
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible
from pymoo.indicators.hv import HV
from MOKP.data_generate import MOKnapsack
from alg_engine import get_nsga2, get_agemoea, get_smsemoa, get_moead, get_ctaea, get_rvea, get_llmcode
from ga_src.env import GENERATION, MODULE, MAX_ARCHIVE, N_TRIALS, N_THREAD, POP_SIZE, OTHER, TP
from MOTSP.motsp import TSP, MOTSP
from ga_src.env import POP_SIZE
from utils import open_pkl

# def run_single_trial(problem_inst:Problem, algo_code, seed, GEN=GENERATION):
#     algo_runner = LLM_MOEA(seed=seed, offspring_generation_code=algo_code, verbose=False)
#     algo_runner.setup(problem_inst, termination=('n_gen', GEN))
#     # actually execute the algorithm
#     res = algo_runner.run()
#     return res

def calc(problem, pof):
    npof = pof
    # lower value of perf denotes better performance !!!!!!
    if TP == 'MOP':
        indicator = IGD(problem.pareto_front())
        perf = indicator(pof)
    elif TP == 'MOKP':
        indicator = HV(ref_point=np.ones_like(problem.ref_point))
        pof = -pof / problem.ref_point
        perf = indicator(pof)
    elif TP == 'MOTSP':
        indicator = HV(ref_point=np.ones_like(problem.ref_point_min))
        pof = (pof - problem.ref_point_min) / (problem.ref_point_max - problem.ref_point_min)
        perf = 1 - indicator(pof)
    else:
        assert False, 'calc errors'


    return perf


def run_single_trial(problem_inst:Problem, algo_code, seed, GEN=GENERATION, verbose=False, compare=OTHER):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(algo_code) > 20:
            algo_runner = get_llmcode(algo_code, verbose)
        else:
            if algo_code == 'NSGA2':
                algo_runner = get_nsga2(POP_SIZE)
            elif algo_code == 'AGEMOEA':
                algo_runner = get_agemoea(POP_SIZE)
            elif algo_code == 'SMSEMOA':
                algo_runner = get_smsemoa(POP_SIZE)
            elif algo_code == 'MOEAD':
                algo_runner = get_moead(POP_SIZE, problem_inst.n_obj, seed)
            elif algo_code == 'CTAEA':
                algo_runner = get_ctaea(POP_SIZE, problem_inst.n_obj, seed)
            elif algo_code == 'RVEA':
                algo_runner = get_rvea(POP_SIZE, problem_inst.n_obj, seed)
            else:
                assert False, 'no such alg setting'
        algo_runner.setup(problem_inst, termination=('n_gen',GEN), verbose=verbose, save_history=compare)
        # actually execute the algorithm
        res = algo_runner.run()
        rpof = res.F
        perf = calc(problem_inst, rpof)
        val_record = []
        if compare:
            if len(algo_code) > 20:
                hist = res.history
            else:
                hist = [alg.pop for alg in res.history]
            for data in hist:
                val_record.append(calc(problem_inst, data.get('F')))
        return {'value': perf, 'record': val_record}

def get_problems_MOPs(DIM=None):

    names = [
        ['zdt1', DIM],
        ['zdt2', DIM],
        ['zdt3', DIM],
        ['zdt4', DIM],
        ['zdt5'],
        ['zdt6', DIM],
        ['dtlz1', DIM, 3],
        ['dtlz2', DIM, 3],
        ['dtlz3', DIM, 3],
        ['dtlz4', DIM, 3],
        ['dtlz5', DIM, 3],
        ['dtlz6', DIM, 3],
        ['dtlz7', DIM, 3],
    ]
    prob_insts = []
    for vals in names:
        if DIM is None or len(vals) == 1:
            prob_insts.append(get_problem(f'{vals[0]}'))
        elif len(vals) == 2:
            prob_insts.append(get_problem(f'{vals[0]}', n_var=vals[1]))
        else:
            prob_insts.append(get_problem(f'{vals[0]}', n_var=vals[1], n_obj=vals[2]))
    return prob_insts


def get_prob_inst_set(pname):
    if pname == 'MOP':
        prob_insts = get_problems_MOPs()
        for prob_inst in prob_insts:
            prob_inst.history_individuals = []
            prob_inst.history_fitness = []
    elif pname == 'MOKP':
        dataset = open_pkl(f'MOKP/MOKP.pkl')
        prob_insts = []
        for data in dataset:
            MOKP = MOKnapsack(data['W'], data['C'], data['V'], data['ref'])
            prob_insts.append(MOKP)
    elif pname == 'MOTSP':
        """Training instance set"""
        # current_file_path = os.path.abspath(__file__)
        # current_dir_path = os.path.dirname(current_file_path)
        # parent_dir_path = os.path.dirname(current_dir_path)
        # with open(os.path.join(parent_dir_path, 'MOTSP/motsp_train.pkl'), 'rb') as f:
        #     prob_insts = pickle.load(f)
        prob_insts = open_pkl(f'MOTSP/motsp_train.pkl')
    else:
        assert False, 'no such problem to handle'
    return prob_insts


def get_prob_test(pname):
    prob_insts = None
    if pname == 'MOP':
        prob_insts = get_problems_MOPs(DIM=50)
        for prob_inst in prob_insts:
            prob_inst.history_individuals = []
            prob_inst.history_fitness = []
    elif pname == 'MOKP':
        dataset = open_pkl(f'MOKP/MOKPTEST.pkl')
        prob_insts = []
        for data in dataset:
            MOKP = MOKnapsack(data['W'], data['C'], data['V'], data['ref'])
            prob_insts.append(MOKP)
    elif pname == 'MOTSP':
        """Testing instance set"""
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.dirname(current_file_path)
        parent_dir_path = os.path.dirname(current_dir_path)
        with open(os.path.join(parent_dir_path,'MOTSP/MOTSPTEST.pkl'), 'rb') as f:
            prob_insts = pickle.load(f)
    else:
        assert False, 'no such problem to handle'
    return prob_insts

def wrapped_run_single_trial(args):
   prob_inst = args['prob_inst']
   code = args['code']
   seed = args['seed']
   return run_single_trial(prob_inst, code, seed)


def organize_run_results(packed_run_res, crossover_code_pops, prob_insts, n_trials):
    organized_results = {}
    for i in range(len(crossover_code_pops)):
        organized_results[f"algo{i}"] = {f"prob{j+1}": [] for j in range(len(prob_insts))}

    # 填充organized_results字典
    res_index = 0
    for i in range(len(crossover_code_pops)):
        for j in range(len(prob_insts)):
            for seed in range(n_trials):
                result_for_this_seed = packed_run_res[res_index]
                organized_results[f"algo{i}"][f"prob{j+1}"].append(result_for_this_seed)
                res_index += 1

    return organized_results

def aggregate_run_results(packed_run_res, code_pops, prob_insts, n_trials):
    res_index = 0
    algo_perfs = []
    for i in range(len(code_pops)):
        algo_perf = []
        algo_record = []
        for j in range(len(prob_insts)):
            temp = [] # performance of one alg on one problem
            temp_record = []
            for seed in range(n_trials):
                temp.append(packed_run_res[res_index]['value'])
                temp_record.append(packed_run_res[res_index]['record'])
                res_index += 1
            algo_perf.append(temp)
            algo_record.append(temp_record)
        algo_perfs.append({'idx': i,
                           'data': algo_perf,
                           'record': algo_record})
    return algo_perfs
def simulate(code_pops, prob_insts, n_trials, pool):
    packed_run_args = []
    for code in code_pops:
        for prob_inst in prob_insts:
            for seed in range(n_trials):
                    args = {'prob_inst': copy.deepcopy(prob_inst), # key!!!!
                            'code': copy.deepcopy(code),
                            'seed': seed}
                    packed_run_args.append(args)


    packed_run_res = pool.map(wrapped_run_single_trial, packed_run_args)
    # res = copy.deepcopy(packed_run_res)
    # pool.close()
    # algo_scores = aggregate_run_results(packed_run_res, code_pops, prob_insts, n_trials)
    return aggregate_run_results(copy.deepcopy(packed_run_res), code_pops, prob_insts, n_trials)

def fun_search(code_pop, n_thread=N_THREAD, n_trials=N_TRIALS, **kwargs):
    import time

    # code_example = """import numpy as np\ndef next_generation(pops: {}, search_trajectory: {}, xlb: np.ndarray, xub: np.ndarray, pop_size: int, N_P: int):
    #     return np.random.rand(pop_size, N_P) * (xub - xlb) + xlb"""

    # define of a batch of initial candidate MOEAs
    # initial_code_pop = [xml_text_to_code(indi) for indi in open_pkl('gpt-4-1106-preview')]
    # if len(code_pop) == 1:
    #     print(f'The testing code is:\n{code_pop[0]}')
    initial_code_pop = code_pop
    # print(initial_code_pop[0])
    pool = Pool(n_thread)
    # get the problem instance set for evaluating the performance of a candidate MOEAs
    testing = kwargs.get('testing')
    pname = kwargs.get('pname')
    if testing is None or testing is False:
        prob_insts = get_prob_inst_set(pname)
    else:
        prob_insts = get_prob_test(pname)

    # just for testing:
    # ttt = run_single_trial(prob_insts[0], code_pop[0], seed=0)
    # print(ttt)
    # evaluate the performances of a batch of MOEAs
    t0 = time.time()
    run_res_dict = simulate(initial_code_pop, prob_insts, n_trials, pool)
    pool.close()
    # print(f"Consumed time: {time.time() - t0}")
    # print(run_res_dict)
    return run_res_dict

def target(queue, code, problems):
    state, error = run_test(code, problems)
    queue.put((state, error))
def run_test_with_timeout(code, pname, TIMEOUT):
    from multiprocessing import Process, Queue
    problems = get_prob_inst_set(pname)
    queue = Queue()
    process = Process(target=target, args=(queue, code, problems))
    process.start()
    process.join(TIMEOUT)

    if process.is_alive():
        process.terminate()
        process.join()
        print(f'Timeout: run_test function took longer than {TIMEOUT} seconds.')
        return False, 'Timeout'
    else:
        return queue.get()

def run_test(code, problems):
    import sys
    import traceback
    # problem = get_prob_inst_set()[-1]
    # run_single_trial(problem, code, 0, GEN=5)
    try:
        # print(f'testing code is:\n{code}\n\n')
        for problem in problems:
            perf = run_single_trial(problem, code, 0, GEN=20)
            val = perf['value']
            if not isinstance(val, (int, float)):
                return False, "The numerical operation may be wrong in 'crossover', 'mutation' as the inner evaluation function returned a nonnumerical result"
        return True, ''
    except Exception as e:
        # 获取异常信息
        exc_type, exc_value, exc_traceback = sys.exc_info()

        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)

        # 打印异常类型、异常信息和发生异常的行号
        output = f"An exception of type {exc_type.__name__} occurred in demo `run_test`. Message: {e}\n"
        output += f"Last few lines of the traceback:\n"
        start = False
        for line in tb_lines:
            if 'in next_generation' in line:
                start = True
            if start:
                output += f'{line}\n'
        return False, output

# def xml_text_to_code(information):
#     pattern = r"<next_generation>(?:\s*<!\[CDATA\[)?(.*?)(?:\]\]>\s*)?</next_generation>"
#     code_information = re.findall(pattern, information, re.DOTALL)
#     return code_information
#
#
# def write_pkl(data, filename):
#     with open(filename, "wb") as f:
#         pickle.dump(data, f)
#
#
# def open_pkl(filename):
#     with open(filename, "rb") as f:
#         data = pickle.load(f)
#         return data


code = '''
def next_generation(pops: dict, search_trajectory: dict, xlb: np.ndarray, xub: np.ndarray, POP_SIZE: int, N_P: int,
                    current_gen: int, max_gen: int):
    """
    Evolves a new population for the next generation using advanced evolutionary strategies.

    :param pops: Current population and rankings
    :param search_trajectory: Trajectory of search with individuals and rankings
    :param xlb: Lower bounds of decision variables
    :param xub: Upper bounds of decision variables
    :param POP_SIZE: Population size
    :param N_P: Number of decision variables
    :param current_gen: Current generation number
    :param max_gen: Maximum number of generations
    :return: New population array with shape (POP_SIZE, N_P)
    """

    def rank_based_selection(pop, s=1.5):
        """Select individuals using rank-based selection."""
        ranks = pop['rankings']
        probabilities = (2 - s) / len(ranks) + 2 * (ranks - np.min(ranks)) * (s - 1) / (
                    len(ranks) * (np.max(ranks) - np.min(ranks)))
        selected_indices = np.random.choice(len(ranks), size=POP_SIZE, p=probabilities / np.sum(probabilities),
                                            replace=True)
        return pop['individuals'][selected_indices]

    def blend_crossover(parent1, parent2, alpha=0.5):
        """Perform blend crossover."""
        gamma = (1. + 2. * alpha) * np.random.rand(N_P) - alpha
        offspring1 = (1 - gamma) * parent1 + gamma * parent2
        offspring2 = gamma * parent1 + (1 - gamma) * parent2
        return offspring1, offspring2

    def gaussian_mutation(individual, mutation_rate=0.1):
        """Mutate an individual using Gaussian mutation."""
        for i in range(N_P):
            if np.random.rand() < mutation_rate:
                individual[i] += np.random.normal(0, 1) * (xub[i] - xlb[i]) / 10.0
                individual[i] = np.clip(individual[i], xlb[i], xub[i])
        return individual

    # **Part 1: Selection**
    selected_parents = rank_based_selection(pops)

    # **Part 2: Crossover**
    offspring = []
    for i in range(0, POP_SIZE, 2):
        parent1, parent2 = selected_parents[i], selected_parents[min(i + 1, POP_SIZE - 1)]
        child1, child2 = blend_crossover(parent1, parent2)
        offspring.extend([child1, child2])

    # Ensure offspring is a numpy array with the correct shape
    offspring = np.array(offspring).reshape((POP_SIZE, N_P))

    # **Part 3: Mutation**
    for i in range(POP_SIZE):
        offspring[i] = gaussian_mutation(offspring[i])

    return offspring
'''


def run_moco():
    """moco - multi-objective combinatorial optimization"""
    pool = Pool(5)
    # TP = 'MOTSP'
    prob_insts = get_prob_inst_set(TP)
    print(f'retrieving the {TP} instance set with size {len(prob_insts)}')
    compared_algos = ['AGEMOEA']
    # run_res_dict = simulate(initial_code_pop, prob_insts, 2, pool)
    for algo_code in compared_algos:
        print(f'Running optimization trial by {algo_code}')
        for pro in prob_insts:
            run_single_trial(pro, algo_code, 1, GEN = 200, verbose = True)

code_test = f'''
def next_generation(pops: dict, D_lst: numpy.ndarray, POP_SIZE: int, N_P: int):
    import numpy as np

    def select_parents(rankings):
        """ Select parents based on rankings using tournament selection """
        parents = []
        for _ in range(POP_SIZE // 2):
            idx1, idx2 = np.random.choice(range(len(rankings)), 2, replace=False)
            if rankings[idx1] < rankings[idx2]:
                parent1 = idx1
            else:
                parent1 = idx2

            idx3, idx4 = np.random.choice(range(len(rankings)), 2, replace=False)
            if rankings[idx3] < rankings[idx4]:
                parent2 = idx3
            else:
                parent2 = idx4

            parents.append((parent1, parent2))
        return parents

    def crossover(parent1, parent2):
        """ Perform Order Crossover (OX) """
        start, end = sorted(np.random.choice(range(N_P), 2, replace=False))
        child = [None]*N_P
        child[start:end+1] = parent1[start:end+1]
        filled_positions = set(range(start, end+1))
        fill_idx = (end + 1) % N_P

        for i in range(N_P):
            candidate_idx = (end + 1 + i) % N_P
            if parent2[candidate_idx] not in child:
                child[fill_idx] = parent2[candidate_idx]
                fill_idx = (fill_idx + 1) % N_P

        return np.array(child, dtype=np.int32)

    def mutate(individual):
        """ Swap mutation """
        idx1, idx2 = np.random.choice(range(N_P), 2, replace=False)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    # Selection
    parent_pairs = select_parents(pops['rankings'])

    # Crossover
    new_pop = []
    for parent1_idx, parent2_idx in parent_pairs:
        parent1 = pops['individuals'][parent1_idx]
        parent2 = pops['individuals'][parent2_idx]
        child1 = crossover(parent1, parent2)
        child2 = crossover(parent2, parent1)
        new_pop.append(child1)
        new_pop.append(child2)

    # Mutation
    for i in range(POP_SIZE):
        if np.random.rand() < 0.1:  # Mutation probability
            new_pop[i] = mutate(new_pop[i])

    # Ensure all are valid permutations
    new_pop = [np.array(list(range(N_P)), dtype=np.int32) if not np.all(np.unique(indiv) == np.arange(N_P)) else indiv for indiv in new_pop]

    return np.array(new_pop, dtype=np.int32)
'''
if __name__ == '__main__':
    # Example usage:
    # res = fun_search(code_pop=[code], n_thread=1, n_trials=10)
    # print(res)
    # pass
    # run_moco()
    res = fun_search([code_test], n_thread=1, n_trials=1, pname='MOTSP')