import numpy as np


def next_generation(pops: {}, search_trajectory: {}, xlb: np.ndarray, xub: np.ndarray, POP_SIZE: int, N_P: int,
                    current_gen: int, max_gen: int):
    """
    Generate the next generation using an adaptive evolutionary strategy that
    intelligently balances exploration and exploitation based on population
    diversity, fitness distribution, and search progress. Key components:
    - Intelligent selection fusing fitness ranking and search trajectory
    - Adaptive multi-operator crossover guided by diversity and convergence
    - Self-adaptive mutation with dynamic rates based on fitness and diversity
    - Boundary constraint handling
    """

    def intelligent_selection():
        """Select parents using ranking-based selection guided by search trajectory."""
        parents = []

        # Assign selection probabilities based on fitness ranking with adaptive pressure
        ranks = pops['rankings']
        pressure = 2 + 2 * np.tanh(3 * current_gen / max_gen - 1.5)  # increase over time
        sel_prob = (max(ranks) - ranks + 1e-6) ** pressure
        sel_prob /= np.sum(sel_prob)

        # Inject guidance from search trajectory with rate based on progress
        guide_rate = 0.1 * np.tanh(2 * current_gen / max_gen)

        for _ in range(POP_SIZE // 2):
            if search_trajectory['individuals'] is not None and np.random.rand() < guide_rate:
                p1 = search_trajectory['individuals'][np.random.choice(len(search_trajectory['individuals']))]
            else:
                p1 = pops['individuals'][np.random.choice(len(pops['individuals']), p=sel_prob)]

            if search_trajectory['individuals'] is not None and np.random.rand() < guide_rate:
                p2 = search_trajectory['individuals'][np.random.choice(len(search_trajectory['individuals']))]
            else:
                p2 = pops['individuals'][np.random.choice(len(pops['individuals']), p=sel_prob)]

            parents.append((p1, p2))

        return parents

    def adaptive_crossover(parents):
        """Perform adaptive crossover using multiple operators guided by diversity and convergence."""
        offspring = []

        # Measure population diversity and convergence
        diversity = np.mean(np.std(pops['individuals'], axis=0)) / (xub - xlb).mean()
        convergence = 1 - np.std(pops['rankings']) / np.mean(pops['rankings'])

        # Adapt crossover operator probabilities based on diversity and convergence
        if diversity < 0.1 and convergence > 0.8:
            cross_probs = [0.1, 0.2, 0.5, 0.2]  # low diversity & high convergence
        elif diversity < 0.2 and convergence > 0.6:
            cross_probs = [0.2, 0.3, 0.4, 0.1]  # medium diversity & convergence
        elif diversity < 0.3:
            cross_probs = [0.4, 0.3, 0.2, 0.1]  # high diversity
        else:
            cross_probs = [0.6, 0.2, 0.1, 0.1]  # very high diversity

        for p1, p2 in parents:
            operator = np.random.choice(['multi-point', 'uniform', 'differential', 'directional'], p=cross_probs)

            if operator == 'multi-point':
                num_points = np.random.randint(1, max(2, int(0.2 * diversity * N_P)))
                points = np.random.choice(range(1, N_P), num_points, replace=False)
                points.sort()
                c1, c2 = p1.copy(), p2.copy()
                for i in range(len(points)):
                    if i % 2 == 0:
                        c1[points[i]:] = p2[points[i]:]
                        c2[points[i]:] = p1[points[i]:]
                    else:
                        c1[points[i]:] = p1[points[i]:]
                        c2[points[i]:] = p2[points[i]:]
            elif operator == 'uniform':
                mask = np.random.rand(N_P) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            elif operator == 'differential':
                J = np.random.choice(len(parents))
                xj = parents[J][0]
                F = np.random.normal(0.5, 0.3)
                F = np.clip(F, 0, 2)
                c1 = p1 + F * (xj - p2)
                c2 = p2 + F * (xj - p1)
            else:  # directional
                x_best = pops['individuals'][0]  # assuming sorted
                mask = np.random.rand(N_P) < 0.5
                c1 = np.where(mask, p1, x_best)
                c2 = np.where(mask, p2, x_best)

            offspring.append(c1)
            offspring.append(c2)

        return np.array(offspring)

    def self_adaptive_mutation(offspring):
        """Mutate offspring with self-adaptive rate and step size based on fitness and diversity."""
        min_rate = 1 / (5 * N_P)
        max_rate = 1.5 / N_P

        # Adapt global mutation rate based on fitness and diversity
        rel_fitness = (max(pops['rankings']) - pops['rankings']) / max(pops['rankings'])
        diversity = np.mean(np.std(offspring, axis=0)) / (xub - xlb).mean()
        global_rate = min_rate + (max_rate - min_rate) * (1 - rel_fitness) * (1 - diversity)

        # Adapt mutation step size based on convergence
        min_step = 0.01 * (xub - xlb)
        max_step = 0.2 * (xub - xlb)
        convergence = 1 - np.std(pops['rankings']) / np.mean(pops['rankings'])
        step_size = min_step + (max_step - min_step) * convergence

        # Perform mutation with adaptive rates and step sizes
        tau = np.random.normal(0, 0.2, len(offspring))
        for i in range(len(offspring)):
            rate = global_rate[i] * np.exp(tau[i])
            mask = np.random.rand(N_P) < rate
            step = step_size * np.exp(0.5 * tau[i])
            offspring[i][mask] += np.random.normal(0, step[mask])

        return offspring

    def check_bounds(offspring):
        """Ensure offspring lie within decision variable bounds."""
        offspring = np.clip(offspring, xlb, xub)
        return offspring

    # Intelligent selection balancing fitness, diversity and search history
    parents = intelligent_selection()

    # Multi-operator adaptive crossover guided by diversity and convergence
    offspring = adaptive_crossover(parents)

    # Self-adaptive mutation based on fitness and diversity
    offspring = self_adaptive_mutation(offspring)

    # Check boundary constraints
    new_pop = check_bounds(offspring)

    return new_pop