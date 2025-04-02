import numpy as np
def next_generation(pops: dict,
                    search_trajectory: dict,
                    xlb: np.ndarray,
                    xub: np.ndarray,
                    POP_SIZE: int,
                    N_P: int,
                    current_gen: int,
                    max_gen: int):
    """
    Evolves a new generation of individuals for multi-objective optimization using
    evolutionary strategies.

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

    def tournament_selection(pops, k=2):
        selected_indices = []
        for _ in range(POP_SIZE):
            participants = np.random.choice(len(pops['individuals']), k, replace=False)
            participants_fitness = pops['rankings'][participants]
            winner_index = participants[np.argmin(participants_fitness)]
            selected_indices.append(winner_index)
        return pops['individuals'][selected_indices]

    def simulated_binary_crossover(parent1, parent2, eta=15):
        child1, child2 = np.copy(parent1), np.copy(parent2)
        for i in range(N_P):
            if np.random.rand() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    x1 = min(parent1[i], parent2[i])
                    x2 = max(parent1[i], parent2[i])
                    xl = xlb[i]
                    xu = xub[i]
                    rand = np.random.rand()
                    beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                    alpha = 2.0 - beta**-(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha)**(1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))
                    c1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
                    beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                    alpha = 2.0 - beta**-(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha)**(1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))
                    c2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))
                    c1 = min(max(c1, xl), xu)
                    c2 = min(max(c2, xl), xu)
                    if np.random.rand() <= 0.5:
                        child1[i] = c2
                        child2[i] = c1
                    else:
                        child1[i] = c1
                        child2[i] = c2
        return child1, child2

    def polynomial_mutation(individual, eta=20):
        mutant = np.copy(individual)
        for i in range(N_P):
            if np.random.rand() < 1.0 / N_P:
                xl = xlb[i]
                xu = xub[i]
                delta1 = (individual[i] - xl) / (xu - xl)
                delta2 = (xu - individual[i]) / (xu - xl)
                rand = np.random.rand()
                mut_pow = 1.0 / (eta + 1.0)
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy**(eta + 1))
                    delta_q = val**mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy**(eta + 1))
                    delta_q = 1.0 - val**mut_pow
                mutant[i] = individual[i] + delta_q * (xu - xl)
                mutant[i] = min(max(mutant[i], xl), xu)
        return mutant

    # Selection
    parents = tournament_selection(pops)

    # Crossover
    offspring = []
    for i in range(0, POP_SIZE, 2):
        parent1, parent2 = parents[i], parents[min(i+1, POP_SIZE-1)]
        child1, child2 = simulated_binary_crossover(parent1, parent2)
        offspring.append(child1)
        offspring.append(child2)
    offspring = np.array(offspring)

    # Mutation
    for i in range(POP_SIZE):
        offspring[i] = polynomial_mutation(offspring[i])

    return offspring