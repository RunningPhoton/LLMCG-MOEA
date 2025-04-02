import numpy as np

from ga_src.env import PROBLEM, MODULE

# This should be changed according to the problem that need to solve
PROB_DESP = {
'multi-objective problems': f'',

'multi-objective knapsack problems': f'''The aim of the {PROBLEM} is to maximize the aggregate profit across diverse categories, while complying with the weight constraint of the knapsack.''',

'multi-objective traveling salesman problems': f'The aim of the {PROBLEM} is to find a traveling path that minimize several objectives simultaneously, such as total travel distance and total travel time',
}
# This should be changed according to the function that you need to design (depend on your entire solving system)
P_FORMATS = {
'multi-objective problems': f'''
Input parameters and their format:\n
:param `pops`, current population consists of {{'individuals': numpy.ndarray with shape(POP_SIZE, N_P), '{MODULE}': numpy.ndarray with shape(POP_SIZE,)}}. Each row in pop['individuals'] represents an individual. Individuals have been sorted by {MODULE} in ascending order, and smaller {MODULE} indicate superior performance. {MODULE} are gained via non-dominated sorting, where all the individuals may share the same {MODULE}.\n
:param `search_trajectory`, trajectory gained along the evolutionary search that consists of {{'individuals': numpy.ndarray with shape(:, N_P) or None; '{MODULE}': numpy.ndarray with shape(:,) or None}}, representing the latest several populations collected throughout the evolutionary search to aid in design intelligent evolutionary operators.\n
:param `xlb`: the lower_bound of decision variables for each individual, i.e., numpy.ndarray with shape(N_P,)\n
:param `xub`: the upper_bound of decision variables for each individual, i.e., numpy.ndarray with shape(N_P,)\n
:param `POP_SIZE`: the number of individuals in each population\n
:param `N_P`: the number of decision variables\n
:param `current_gen`: the number of current generation\n
:param `max_gen`: the maximum number of generations\n

Output format:\n
:return `new_pops`, new population in the format of numpy.ndarray (with shape(POP_SIZE, N_P)) that may achieve superior results on the {PROBLEM}.\n

Functionality of the function:\n
'next_generation' receives evaluated current population and other necessary parameters to help generate a new population in the evolutionary search.\n

Note to improve optimization efficacy:\n
You may design innovative search strategies via sub-functions within the `next_generation`, to obtain superior search performance by harnessing informative inputs such as existing population given by `pops` and search trajectory given by `search_trajectory`.\n

''',

'multi-objective knapsack problems': f'''
Input parameters and their format:\n
:param `pops`: The current population, structured as a dictionary (i.e., {{'individuals': numpy.ndarray with shape(POP_SIZE, N_P), '{MODULE}': numpy.ndarray with shape(POP_SIZE,)}}) containing:\n
    - `individuals`: A numpy.ndarray (dtype=numpy.int32) with shape (POP_SIZE, N_P). Each row corresponds to an individual, with decision variables set to 1 (item included) or 0 (item excluded).\n
    - `{MODULE}`: A numpy.ndarray with shape (POP_SIZE,). Individuals are sorted by {MODULE} in ascending order, where a smaller {MODULE} value signifies better performance. {MODULE} values are assigned through non-dominated sorting, and it's possible for multiple individuals to have identical {MODULE} values.\n
:param `W`: A numpy.ndarray (dtype=numpy.int32) with shape (N_P,) representing the weights of the N_P items.\n
:param `C`: An integer defining the maximum weight capacity of the knapsack.\n
:param `V`: A numpy.ndarray (dtype=numpy.int32) with shape (N_P, N_O) indicating the N_O profit values for the N_P items.\n
:param `POP_SIZE`: The number of individuals in each population.\n
:param `N_P`: The number of decision variables\n

Output format:\n
:return `new_pops`, new population in the format of numpy.ndarray (with shape=(POP_SIZE, N_P), dtype=numpy.int32) that may achieve superior results on the {PROBLEM}\n

Functionality of the function:\n
`next_generation` receives evaluated current population and other necessary parameters to help generate a new population in the evolutionary search, with the goal of obtaining the highest profits at each of the N_O dimensions.\n

Note to improve optimization efficacy:\n
You may design innovative strategies using a sub-function within the `next_generation`. By exploiting the problem properties of {PROBLEM} in terms of weights given by `W`, knapsack capacity given by `C` and profits given by `V`, these strategies are expected to refine the checked solutions effectively and efficiently.\n

''',

'multi-objective traveling salesman problems': f'''
Input parameters and their format:\n
:param `pops`: The current population, structured as a dictionary (i.e., {{'individuals': numpy.ndarray with shape(POP_SIZE, N_P), '{MODULE}': numpy.ndarray with shape(POP_SIZE,)}}) containing:\n
    - `individuals`: A numpy.ndarray (dtype=numpy.int32) with shape (POP_SIZE, N_P). Each row represents a traveling sequence (i.e., a permutation start from 0 to N_P-1). For example, the sequence denoted by [0, 3, 1, 2] represents the traveling route is: 0->3->1->2->0\n
    - `{MODULE}`: A numpy.ndarray with shape (POP_SIZE,). Individuals are sorted by {MODULE} in ascending order, where a smaller {MODULE} value signifies better performance. {MODULE} values are assigned through non-dominated sorting, and it's possible for multiple individuals to have identical {MODULE} values.\n
:param `D_lst`: A 3-dimensional numpy.ndarray with data type numpy.float. It has the shape (N_O, N_P, N_P), where `N_O` represents the number of different travel cost matrices, and `N_P` is the number of cities. Each (N_P, N_P) matrix within `D_lst` contains the travel costs, such as time or distance, between each pair of cities.\n
:param `POP_SIZE`: The number of individuals in each population.\n
:param `N_P`: The number of cities that should be visited\n

Output format:\n
:return `new_pops`, new population in the format of numpy.ndarray (with shape=(POP_SIZE, N_P), dtype=numpy.int32) that may achieve superior results on the {PROBLEM}\n

Functionality of the function:\n
'next_generation' receives evaluated current population and other necessary parameters to help generate a new population in the evolutionary search, with the goal of obtaining the lowest traveling cost at each of the N_O dimensions.\n

Note to improve optimization efficacy:\n
To improve solution quality, it is also recommended to develop innovative strategies that extend beyond the scope of traditional evolutionary search operators. These strategies can be implemented as sub-functions within the next_generation method. By exploiting the problem properties of {PROBLEM} in terms of D_lst, these strategies are expected to refine the checked solutions effectively and efficiently.''',
}
F_FORMATS = {
'multi-objective problems': f'''
Format of the function is given by:\n
def next_generation(pops: {{}}, search_trajectory: {{}}, xlb: numpy.ndarray, xub: numpy.ndarray, POP_SIZE: int, N_P: int, current_gen: int, max_gen: int):\n
    # **Subsection Selection**: First of all, pair the parent individuals---potentially selected from both the populations and the search trajectory---for intelligent crossover. This may produce POP_SIZE/2 pairs of parent individuals.\n
    
    # **Subsection Crossover**: Secondly, conduct intelligent crossover between each pair of parent individuals based on the innovative search strategies, generate two offspring individuals for each pair, and thus obtain a new population `new_pop` with shape (POP_SIZE, N_P) that may achieve superior performance in optimization.\n
    
    # **Subsection Mutation**: Thirdly, fine-tune the `new_pop` gained from Crossover by harnessing the innovative search strategies, and return the modified population `new_pop` using numpy.ndarray with shape(POP_SIZE, N_P).\n
    
    # **Subsection Checking**: Ensure that the decision variable values in `new_pop` are within the `xlb` lower bounds and `xub` upper bounds before returning it.\n
''',

'multi-objective knapsack problems': f'''
Format of the function is given by:\n
def next_generation(pops: {{}}, W: numpy.ndarray, C: int, V: numpy.ndarray, POP_SIZE: int, N_P: int):\n
    # **Subsection Selection**: First of all, pair the parent individuals for intelligent. This may produce POP_SIZE/2 pairs of parent individuals.\n
    
    # **Subsection Crossover**: Secondly, conduct intelligent crossover between each pair of parent individuals based on innovative crossover strategies, generate two offspring individuals for each pair, and thus obtain a new population `new_pop` with shape (POP_SIZE, N_P) that may achieve superior performance in optimization.\n
    
    # **Subsection Mutation**: Thirdly, fine-tune the `new_pop` gained from Crossover by harnessing innovative mutation strategies, and return the modified population `new_pop` using numpy.ndarray (dtype=numpy.int32) with shape(POP_SIZE, N_P).\n

    # **Subsection Checking**: Ensure that `new_pop` contains only 0 (item removed) or 1 (item selected) for each decision variable before returning. You may design innovative search mechanisms here to improve the solution quality via leveraging on W, C and V.\n

''',


'multi-objective traveling salesman problems': f'''
def next_generation(pops: {{}}, D_lst: numpy.ndarray, POP_SIZE: int, N_P: int):\n
    # **Subsection Selection**: First of all, pair the parent individuals for intelligent crossover. This may produce POP_SIZE/2 pairs of parent individuals\n
    
    # **Subsection Crossover**: Secondly, conduct intelligent crossover between each pair of parent individuals based on innovative crossover strategies, generate two offspring individuals for each pair, and thus obtain a new population `new_pop` with shape (POP_SIZE, N_P) that may achieve superior performance in optimization. You need to design a suitable crossover sub-function that can handle the permutation-based data.\n
    
    # **Subsection Mutation**: Thirdly, fine-tune the `new_pop` gained from Crossover by harnessing innovative mutation strategies, and return the modified population `new_pop` using numpy.ndarray (dtype=numpy.int32) with shape(POP_SIZE, N_P). You need to design a suitable mutation sub-function that can handle the permutation-based data.\n
    
    # **Subsection Checking**: Ensure that each row of `new_pop` contains a permutation of N_P integers from 0 to N_P-1 without repeat numbers. You can also design innovative search mechanisms to improve the solution quality via leveraging on D_lst.\n
    
'''

}
PROMPTS = {
'system': f'''You are an expert in designing intelligent evolutionary search strategies that can solve {PROBLEM} efficiently and effectively. {PROB_DESP[PROBLEM]}\n
''',

'task': f'''Your task is to evolve a superior evolutionary operator with Python for tackling {PROBLEM}, with the goal of achieving top search performance across {PROBLEM}. You have to provide me the Python code with a single function (you can add sub-functions inner the single function namely 'next_generation') following the format and the requirements given below, which are matched with their functionalities.\n
{F_FORMATS[PROBLEM]}
{P_FORMATS[PROBLEM]}
\nRequirements:\n
You have to return me a single function namely `next_generation` (may involve several inner functions), keep the format of input and the format of output unchanged, and provide concise descriptions in the annotation.\n
Please return me a XML text using the following format:\n
<next_generation>\n
...\n
</next_generation>\n
where `...` gives only the entire code without any additional information. To enable direct compilation for the code given in `...`, please don't provide any other text except the single Python function namely `next_generation` with its annotation.\n
\n**No Further Explanation Needed!!**\n
    '''
}


def selection_code(scores, N):
    import random
    # Normalize scores
    adjusted_scores = np.exp(scores)
    adjusted_scores /= sum(adjusted_scores) + 1e-10
    chosen_indices = np.random.choice(range(len(scores)), size=N, replace=False, p=adjusted_scores)
    return chosen_indices


def roulette_wheel_selection(scores, N=2):
    total_score = np.sum(scores)
    selection_probs = np.array(scores) / total_score
    chosen_indices = np.random.choice(range(len(scores)), size=N, replace=False, p=selection_probs)
    return chosen_indices

def code_evolve_to_description_multi(algs, scores, low=2, high=5):
    stream = f'''
    I will showcase several evaluated 'next_generation' functions in XML format first, with their scores obtained on the {PROBLEM}. Then, your task is to conceive an advanced function with the same input/output formats, termed 'next_generation', that should draw inspiration from the high-qualified cases while differentiating itself from them.\n
    '''
    N = np.random.randint(low, high)
    indices = selection_code(scores, N)
    # indices = roulette_wheel_selection(scores)
    stream += f'''\nBelow, you will find the {N} evaluated `next_generation` functions in XML texts, each accompanied by its corresponding score.\n'''
    cid = 1
    for idx in indices:
        alg = algs[idx]
        score = scores[idx]
        stream += f"\n<next_generation>\n{alg}\n</next_generation>\n"
        stream += f"The score of {cid}-th `next_generation` function is: {score}\n\n"
        cid += 1

    stream += f'''
    Kindly devise an innovative `next_generation` method with XML text (<next_generation>\n...\n</next_generation>, where the `...` represents the code snippet.) that retains the identical input/output structure.
    \nNo Explanation Needed!!\n'''
    return stream

def code_mutation_to_description(alg):
    stream = f'''I will introduce an evolutionary search function namely `next_generation` in XML format.\n
    Your task is to meticulously refine this function and propose a novel one that may obtain superior search performance on {PROBLEM}, ensuring the input/output formats, function name, and core functionality remain unaltered.\n
    '''
    stream += f'The original function is given by:\n<next_generation>\n{alg}\n</next_generation>\n'
    stream += f'''Please return me an innovative `next_generation` function with the same XML format, i.e., \n<next_generation>\n...\n</next_generation>, where the `...` represents the code snippet.
    \nNo Explanation Needed!!\n'''
    return stream

def failed_refine(error):
    refine = f"The code you provided for me cannot pass my demo test on {PROBLEM}\nThe error is:\n{error}"
    refine += f"Can you correct the code according to the errors?\n"
    refine += f'''Please return me a refined `next_generation` with the same XML format, i.e., \n<next_generation>\n...\n</next_generation>, where the `...` represents the code snippet.
    \nNo Explanation Needed!!\n'''
    return refine


