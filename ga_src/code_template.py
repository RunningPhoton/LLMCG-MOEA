# import numpy
#
# def llm_selection(pops: {}, search_trajectory: {}):
#
#     # This operator receives a population and existing search_trajectory that gained along the evolutionary search, pairs the parent individuals for crossover according to pops and search_trajectory, and returns an array of two-tuples like [(0, 1),...] which illustrates that crossover will be conducted on pops['individual'][0] and pops['individual'][1]
#
#     # :param pops: current population consists of {'individuals': numpy.ndarray, 'fitness': numpy.ndarray}
#     # :param search_trajectory: results gained along the evolutionary search that consists of {'individuals': numpy.ndarray, 'fitness': numpy.ndarray} (can be empty arrays)
#     # :return: crossover_tuples: indices for conducting crossover
#
#
# def llm_crossover(pops: {}, search_trajectory: {}, indices: list, xlb: numpy.ndarray, xub: numpy.ndarray):
#
#     # This operator receives a population and existing search_trajectory that gained along the evolutionary search, and generate new population using strategies according existing population given by 'pops' and search experiences given by 'search_trajectory'
#
#     # :param pops: current population consists of {'individuals': numpy.ndarray, 'fitness': numpy.ndarray}
#     # :param search_trajectory: results gained along the evolutionary search that consists of {'individuals': numpy.ndarray, 'fitness': numpy.ndarray}
#     # :param indices: a list of parent indexes for conducting crossover
#     # :param xlb: lower_bound of decision variables
#     # :param xub: upper_bound of decision variables
#     # :return: new_pops: new population in the format of numpy.ndarray (with the same shape of 'pops['individuals']') that may achieve superior results on the optimization problems
#
#
# def llm_mutation(pops: numpy.ndarray, search_trajectory: {}, xlb: numpy.ndarray, xub: numpy.ndarray):
#
#     # This operator receives a population (without fitness) from crossover and existing search_trajectory that gained along the evolutionary search, and fine-tune the 'pops' using strategies learned from existing population given by 'pops' and search experiences given by 'search_trajectory'
#
#     # :param pops: current population consists of numpy.ndarray
#     # :param search_trajectory: results gained along the evolutionary search that consists of {'individuals': numpy.ndarray, 'fitness': numpy.ndarray}
#     # :param xlb: lower_bound of decision variables
#     # :param xub: upper_bound of decision variables
#     # :return: pops: fine-tuned new population in the format of numpy.ndarray that may achieve superior results on the optimization problems
#
#
# def llm_replacement(merged_pops: {}, search_trajectory: {}, POP_SIZE: int, **kwargs):
#
#     # This operator receives a merged population after evaluation and existing search_trajectory that gained along the evolutionary search, and select 'POP_SIZE' individuals from the merged population using strategies learned from existing population given by 'pops' and search experiences given by 'search_trajectory'
#
#     # :param merged_pops: merged population consists of {'individuals': numpy.ndarray, 'fitness': numpy.ndarray}
#     # :param search_trajectory: results gained along the evolutionary search that consists of {'individuals': numpy.ndarray, 'fitness': numpy.ndarray}
#     # :return: new_pops: new population for the next generation
#
#
# def llm_evaluation(pops: numpy.ndarray, xlb: numpy.ndarray, xub: numpy.ndarray):
#
#     # This operator receives a population (without fitness) from mutation, and evaluate the 'pops' in parallel
#
#     # :param pops: current population consists of numpy.ndarray
#     # :param search_trajectory: results gained along the evolutionary search that consists of {'individuals': numpy.ndarray, 'fitness': numpy.ndarray}
#     # :param xlb: lower_bound of decision variables
#     # :param xub: upper_bound of decision variables
#     # :return: pops: evaluated population consists of {'individuals': numpy.ndarray, 'fitness': numpy.ndarray}
#
#
