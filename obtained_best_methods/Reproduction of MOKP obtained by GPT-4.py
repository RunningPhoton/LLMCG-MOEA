import numpy as np


def next_generation(pops: {}, W: np.ndarray, C: int, V: np.ndarray, POP_SIZE: int, N_P: int):
    """
    Generate a new population for the multi-objective knapsack problem using evolutionary strategies.
    """

    def elitist_selection(rankings, elite_size=POP_SIZE // 5):
        """
        Selects the best-performing individuals to carry over to the next generation.
        """
        elite_indices = np.argpartition(rankings, elite_size)[:elite_size]
        return elite_indices

    def simulated_binary_crossover(parent1, parent2, eta=2.0):
        """
        Perform simulated binary crossover (SBX) to produce two offspring.
        """
        offspring1, offspring2 = np.copy(parent1), np.copy(parent2)
        for i in range(N_P):
            if np.random.rand() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 0:
                    y1 = min(parent1[i], parent2[i])
                    y2 = max(parent1[i], parent2[i])
                    rand = np.random.rand()
                    beta = 1.0 + (2.0 * (y1 - 0.0) / (y2 - y1))
                    alpha = 2.0 - beta ** -(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))

                    beta = 1.0 + (2.0 * (1.0 - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** -(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

                    c1 = min(max(c1, 0.0), 1.0)
                    c2 = min(max(c2, 0.0), 1.0)

                    if np.random.rand() <= 0.5:
                        offspring1[i] = c2
                        offspring2[i] = c1
                    else:
                        offspring1[i] = c1
                        offspring2[i] = c2
        return offspring1.astype(np.int32), offspring2.astype(np.int32)

    def polynomial_mutation(individual, eta=20.0):
        """
        Mutate an individual using polynomial mutation.
        """
        for i in range(N_P):
            if np.random.rand() < 1.0 / N_P:
                y = individual[i]
                delta1 = (y - 0.0) / (1.0 - 0.0)
                delta2 = (1.0 - y) / (1.0 - 0.0)
                rand = np.random.rand()
                mut_pow = 1.0 / (eta + 1.0)
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                    delta_q = 1.0 - val ** mut_pow
                y = y + delta_q * (1.0 - 0.0)
                y = min(max(y, 0.0), 1.0)
                individual[i] = y
        return individual.astype(np.int32)

    def greedy_repair(individual):
        """
        Greedy repair mechanism that adds items with the highest profit-to-weight ratio.
        """
        while np.dot(individual, W) > C:
            selected_items = np.where(individual == 1)[0]
            p_w_ratio = V[selected_items, 0] / W[selected_items]
            individual[selected_items[np.argmin(p_w_ratio)]] = 0
        remaining_capacity = C - np.dot(individual, W)
        non_selected_items = np.where(individual == 0)[0]
        for item in non_selected_items[np.argsort(-V[non_selected_items, 0] / W[non_selected_items])]:
            if W[item] <= remaining_capacity:
                individual[item] = 1
                remaining_capacity -= W[item]
        return individual

    # Elitism
    elite_indices = elitist_selection(pops['rankings'])
    elite_individuals = pops['individuals'][elite_indices]

    # Crossover and Mutation
    new_pop = []
    for i in range(0, POP_SIZE - len(elite_individuals), 2):
        idx1, idx2 = np.random.choice(POP_SIZE, 2, replace=False)
        parent1, parent2 = pops['individuals'][idx1], pops['individuals'][idx2]
        offspring1, offspring2 = simulated_binary_crossover(parent1, parent2)
        offspring1 = polynomial_mutation(offspring1)
        offspring2 = polynomial_mutation(offspring2)
        new_pop.extend([offspring1, offspring2])

    # Repair
    for i in range(len(new_pop)):
        new_pop[i] = greedy_repair(new_pop[i])

    # Merge elite individuals with new population and ensure binary representation
    new_pop = np.vstack((elite_individuals, new_pop))
    new_pop = new_pop[:POP_SIZE]  # Ensure population size
    new_pop = np.clip(new_pop, 0, 1).astype(np.int32)  # Enforce binary representation

    return new_pop