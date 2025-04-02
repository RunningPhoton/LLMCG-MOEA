from pymoo.core.problem import Problem, ElementwiseProblem
import pickle
from copy import deepcopy
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import os

class TSP(ElementwiseProblem):
    # migrated from pymoo.problems.single.traveling_salesman

    def __init__(self, cities, **kwargs):
        """
        A two-dimensional traveling salesman problem (TSP)

        Parameters
        ----------
        cities : numpy.array
            The cities with 2-dimensional coordinates provided by a matrix where where city is represented by a row.

        """
        n_cities, _ = cities.shape

        self.cities = cities
        self.D = cdist(cities, cities)

        super(TSP, self).__init__(
            n_var=n_cities,
            n_obj=1,
            xl=0,
            xu=n_cities,
            vtype=int,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = self.get_route_length(x)

    def get_route_length(self, x):
        n_cities = len(x)
        dist = 0
        for k in range(n_cities - 1):
            i, j = x[k], x[k + 1]
            dist += self.D[i, j]

        last, first = x[-1], x[0]
        dist += self.D[last, first]  # back to the initial city
        return dist

    def visualize(self, x, show=True, label=True):
        fig, ax = plt.subplots()

        # plot cities using scatter plot
        ax.scatter(self.cities[:, 0], self.cities[:, 1], s=250)
        if label:
            # annotate cities
            for i, c in enumerate(self.cities):
                ax.annotate(str(i), xy=c, fontsize=10, ha="center", va="center", color="white")

        # plot the line on the path
        for i in range(len(x)):
            current = x[i]
            next_ = x[(i + 1) % len(x)]
            ax.plot(self.cities[[current, next_], 0], self.cities[[current, next_], 1], 'r--')

        fig.suptitle("Route length: %.4f" % self.get_route_length(x))

        if show:
            plt.show()


class MOTSP(ElementwiseProblem):
    def __init__(self, n_city, n_obj, seed, is_permutation=True, **kwargs):
        # profit values, np.ndarray with shape(NP, NO)
        self.n_var = n_city
        self.n_obj = n_obj
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.is_permutation = is_permutation
        self.tsps = [TSP(self.rng.uniform(0, 1, size=(n_city, 2))) for _ in range(self.n_obj)]
        print(f"Creating an MOTSP instance with seed {self.seed}")
        self.ref_point_max = calculate_reference_point(self, ideal=False)
        self.ref_point_min = calculate_reference_point(self, ideal=True)
        print('ref_point_min', self.ref_point_min)
        print('ref_point_max', self.ref_point_max)
        super().__init__(n_var=self.n_var, n_obj=self.n_obj, xl=0,xu=1, **kwargs)

    def sample_solution(self):
        if self.is_permutation:
            return np.random.permutation(self.n_var)
        else:
            return np.random.rand(self.n_var)

    def _decode(self, x):
        # Translate the real-valued x into a permutation as a solution of the TSP
        assert np.all((x >= 0.0) & (x <= 1.0)) and len(x.shape) == 1
        tour = np.argsort(x)
        return tour

    def _evaluate(self, x, out, *args, **kwargs):
        '''
        :param x: A permutation/real-valued array encoding a candidate solution to the tsp
        :param out: A n_obj-dimensional array of containing costs of each TSP as an objective
        :param args:
        :param kwargs:
        :return:
        '''
        if self.is_permutation:
            x_permute = x
        else:
            x_permute = self._decode(x)
        out['F'] = np.array([self.tsps[tsp_id].get_route_length(x_permute) for tsp_id in range(self.n_obj)])

    def visualize(self, x, show=True, label=True):
        if self.is_permutation:
            x_permute = x
        else:
            print('real-valued solution', x)
            x_permute = self._decode(x)
        print('permutation', x_permute)

        fig, ax = plt.subplots(nrows=1, ncols=self.n_obj)

        for tsp_id in range(self.n_obj):
            # plot cities using scatter plot
            ax[tsp_id].scatter(self.tsps[tsp_id].cities[:, 0], self.tsps[tsp_id].cities[:, 1], s=250)
            if label:
                # annotate cities
                for i, c in enumerate(self.tsps[tsp_id].cities):
                    ax[tsp_id].annotate(str(i), xy=c, fontsize=10, ha="center", va="center", color="white")

            # plot the line on the path
            for i in range(len(x_permute)):
                current = x_permute[i]
                next_ = x_permute[(i + 1) % len(x_permute)]
                ax[tsp_id].plot(self.tsps[tsp_id].cities[[current, next_], 0], self.tsps[tsp_id].cities[[current, next_], 1], 'r--')

            ax[tsp_id].set_title("Route length: %.4f" % self.tsps[tsp_id].get_route_length(x_permute))

        out = {}
        self._evaluate(x, out)
        print('obj_val', out['F'])

        if show:
            plt.show()


class TSPSolver:
    def __init__(self, prob: TSP, reverse_weight=False):
        self.prob = prob
        self.reverse_weight = reverse_weight
        self.weight_multiplier = 10000
        self.dist_max = np.max(self.prob.D)
        self.tsp_data = {'distance_matrix': (self.dist_max - self.prob.D
                                             if self.reverse_weight else self.prob.D) * self.weight_multiplier,
                         'num_vehicles': 1,
                         'depot': 0}
        # print(self.tsp_data['distance_matrix'])

    def solve(self, show=False):
        manager = pywrapcp.RoutingIndexManager(
            len(self.tsp_data["distance_matrix"]), self.tsp_data["num_vehicles"], self.tsp_data["depot"]
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.tsp_data["distance_matrix"][from_node, to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        routes = get_routes(solution, routing, manager)
        route = routes[0][:-1]

        # print_solution(manager, routing, solution)
        # print(route, len(route))

        if show:
            self.prob.visualize(route)
        return self.prob.get_route_length(route)


def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    index = routing.Start(0)
    plan_output = "Route for vehicle 0:\n"
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += f" {manager.IndexToNode(index)} ->"
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += f" {manager.IndexToNode(index)}\n"
    plan_output += f"Route distance: {route_distance} miles"
    print(plan_output)


def calculate_reference_point(prob: MOTSP, ideal=True):
    reference_point = []
    reverse_weight = not ideal
    for i in range(prob.n_obj):
        # print(f"Calculating the best value for {i+1}-th objective as reference {'ideal' if ideal else 'worst'} point")
        or_solver = TSPSolver(deepcopy(prob.tsps[i]), reverse_weight=reverse_weight)
        reference_point.append(or_solver.solve(show=False))
    reference_point = np.array(reference_point)
    # print(f"Reference {'ideal' if ideal else 'worst'} point: {reference_point}")
    return reference_point


def generate_motsp_inst_set(dataset_dir, train_size=20, test_size=20):
    train_set, test_set = [], []
    for seed in list(np.arange(train_size) + 1000):
        train_n_city = 30
        train_set.append(MOTSP(train_n_city, 2, seed, is_permutation=True))

    for seed in list(np.arange(test_size) + 4396):
        test_n_city = np.random.randint(50,100)
        test_set.append(MOTSP(test_n_city, 2, seed, is_permutation=True))

    with open(os.path.join(dataset_dir, 'motsp_train.pkl'), 'wb') as f:
        pickle.dump(train_set, f)

    with open(os.path.join(dataset_dir, 'motsp_test-2.pkl'), 'wb') as f:
        pickle.dump(test_set, f)


if __name__ == '__main__':
    motsp = MOTSP(20, 2, 10, is_permutation=True)
    or_solver = TSPSolver(motsp.tsps[0], reverse_weight=True)
    or_solver.solve(show=True)


    motsp = MOTSP(20, 2, 10, is_permutation=True)
    or_solver = TSPSolver(motsp.tsps[0], reverse_weight=False)
    or_solver.solve(show=True)
    # generate_motsp_inst_set("/home/shwu/project/AutoAlgDesign/LLM-OPT-CG/moco",train_size=20, test_size=20)

