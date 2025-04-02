import numpy as np


def next_generation(pops: dict, search_trajectory: dict, xlb: np.ndarray, xub: np.ndarray, POP_SIZE: int, N_P: int,
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