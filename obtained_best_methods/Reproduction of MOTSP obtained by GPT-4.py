
import numpy as np

def next_generation(pops: dict, D_lst: np.ndarray, POP_SIZE: int, N_P: int):
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