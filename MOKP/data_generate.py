import pickle
from copy import deepcopy

import numpy as np
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize


class MOKnapsack(ElementwiseProblem):
    def __init__(self, W, C, V, ref_point, **kwargs):
        # profit values, np.ndarray with shape(NP, NO)
        self.V = V
        self.NP, self.NO = self.V.shape
        # weight capacity
        self.C = C
        # weights of items
        self.W = W # np.ndarray with shape(NP,)
        self.ref_point = ref_point

        self.rlst = list(range(self.NP))
        n_constr = 1
        super().__init__(
            n_var=self.NP,
            n_obj=self.NO,
            n_constr=n_constr,
            xl=0,
            xu=1,
            **kwargs
        )

    def repair_x(self, x):
        total_weight = np.vdot(x, self.W)
        import random
        random.shuffle(self.rlst)
        res = deepcopy(x)
        cid = 0
        while total_weight > self.C:
            to_remove = self.rlst[cid]
            cid += 1
            if res[to_remove] == 0: continue
            res[to_remove] = 0
            total_weight -= self.W[to_remove]
        return res


    def _evaluate(self, x, out, *args, **kwargs):
        nx = self.repair_x(x)
        fs = -np.matmul(nx, self.V)

        out['F'] = np.column_stack(fs)
        out["G"] = (np.vdot(self.W, nx) - self.C)


def knapsack_optimized(W, V, C):
    # Number of items
    n = len(W)

    # One-dimensional array to store the maximum profit for each subproblem
    K = [0] * (C + 1)

    # Build array K[] in a bottom-up manner
    for i in range(n):
        # Traverse the array from C to the weight of the current item
        for w in range(C, W[i] - 1, -1):
            # Update the maximum profit for the current weight
            K[w] = max(K[w], K[w - W[i]] + V[i])

    # The last entry of the array is the result
    return K[C]

def create_random_knapsack_problem(n_items, n_obj, seed):
    np.random.seed(seed)
    W = np.random.randint(1, 100, size=n_items)
    C = int(np.sum(W) / 10)
    # C = 2000
    V = np.random.randint(1, 100, size=(n_items, n_obj))

    ref_point = []
    for i in range(n_obj):
        ref_point.append(knapsack_optimized(W, V[:, i], C))
    print(ref_point)
    print(f'C: {C}')
    # problem = MOKnapsack(W, C, V, ref_point)
    return W, C, V, ref_point

def MOKP_Generate(filename):
    data = []
    N_case = 10
    for i in range(N_case):
        n_obj = np.random.randint(low=2, high=3)
        N_items = np.random.randint(low=100, high=200)
        # if i < 10:
        #     n_obj = np.random.randint(low=2, high=3)
        #     N_items = np.random.randint(low=50, high=100)
        # else:
        #     n_obj = np.random.randint(low=2, high=3+1)
        #     N_items = np.random.randint(low=100, high=200)
        W, C, V, ref = create_random_knapsack_problem(n_items=N_items, n_obj=n_obj, seed=i)
        data.append({
            'W': W,
            'C': C,
            'V': V,
            'ref': ref
        })
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def open_pkl(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
        return data

if __name__ == '__main__':
    MOKP_Generate('MOKPT.pkl')
    # dataset = open_pkl('MOKP.pkl')
    # prob_insts = []
    # for data in dataset:
    #     MOKP = MOKnapsack(data['W'], data['C'], data['V'], data['ref'])
    #     prob_insts.append(MOKP)
    # # mokp = create_random_knapsack_problem(100, 3, 1)
    # for mokp in prob_insts:
    #     alg = AGEMOEA(
    #         pop_size = 100,
    #         sampling=BinaryRandomSampling(),
    #         crossover=TwoPointCrossover(),
    #         mutation=BitflipMutation(),
    #         eliminate_duplicates=True
    #     )
    #
    #     res = minimize(mokp,
    #                    alg,
    #                    ('n_gen', 200),
    #                    verbose=True)
    #     from pymoo.indicators.hv import HV
    #     ind = HV(ref_point=np.ones_like(mokp.ref_point))
    #     pof = -res.F / mokp.ref_point
    #
    #     print(f'score: {1-ind(pof)}')