import logging
import re
import pickle
import time
from copy import deepcopy
import openai
import numpy as np
from descriptions import *
from ga_src.env import GC_POP_SIZE
from MOKP.data_generate import MOKnapsack
from MOTSP.motsp import TSP, MOTSP

def write_pkl(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def open_pkl(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
        return data

def update(pop, scores, size):
    _args = np.argsort(-np.array(scores)).tolist()
    pop = np.array(pop)[_args].tolist()
    scores = np.array(scores)[_args].tolist()
    return pop[:size], scores[:size]


def score_calc(result, bias):
    # m * n, m testing problems, n trials
    data = np.array(result['data'])
    mean_pro = np.mean(data, axis=1)
    mean, std = np.mean(mean_pro), np.std(mean_pro)
    score = 1 - mean - std * bias
    return score


def get_logger(filename, verbosity=1, name=None, mode='w'):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # Remove all handlers if there are any
    while logger.hasHandlers() and len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])

    # File handler for outputting to a file
    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# def extract_function(text):
#     # Pattern to match the function definition, excluding the markdown code block syntax if present
#     pattern = r"(?:```python\s)?(.*?)(?:```)?"
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         return match.group(1).strip()  # Return the matched function definition without the markdown syntax
#     else:
#         return None  # Return None if no match is found
def extract_function(code):
    lines = code.split('\n')
    start_index = None
    end_index = None

    # Find the start of the function or import
    for i, line in enumerate(lines):
        if line.strip().startswith('import') or line.strip().startswith('def'):
            start_index = i
            break

    # Find the last return statement
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith('return'):
            end_index = i
            break
    # print(start_index, end_index)
    # Extract the relevant lines
    if start_index is not None and end_index is not None:
        extracted_code = '\n'.join(lines[start_index:end_index + 1])
        return extracted_code
    else:
        return "No valid function or import found."
def xml_text_to_code(information):
    pattern = r"<next_generation>(?:\s*<!\[CDATA\[)?(?:\s*```python)?(.*?)(?:```\s*)?(?:\]\]>\s*)?</next_generation>"
    # pattern = r"<next_generation>(?:```python\s)?(.*?)(?:```)?</next_generation>"
    code1 = re.findall(pattern, information, re.DOTALL)[0]
    # print(code1)
    code_information = extract_function(code1)
    # code_information = code1
    # print(code_information)
    return code_information
# aa = xml_text_to_code(ttt)



if __name__ == '__main__':
    inf1 = '''
<next_generation>
<![CDATA[
```python

import numpy as np

def next_generation(pops: dict, search_trajectory: dict, xlb: np.ndarray, xub: np.ndarray, POP_SIZE: int, N_P: int, current_gen: int, max_gen: int):

    offspring = None
    return offspring
```
]]>
</next_generation>
    '''
    inf2 = '''
<next_generation>
```python

import numpy as np

def next_generation(pops: dict, search_trajectory: dict, xlb: np.ndarray, xub: np.ndarray, POP_SIZE: int, N_P: int, current_gen: int, max_gen: int):

    offspring = None
    return offspring
```
</next_generation>
    '''
    inf3 = '''
<next_generation>
<![CDATA[

import numpy as np

def next_generation(pops: dict, search_trajectory: dict, xlb: np.ndarray, xub: np.ndarray, POP_SIZE: int, N_P: int, current_gen: int, max_gen: int):

    offspring = None
    return offspring
]]>
</next_generation>
    '''
    kk1 = xml_text_to_code(inf1)
    kk2 = xml_text_to_code(inf2)
    kk3 = xml_text_to_code(inf3)
    print(kk1)
    print(kk2)
    print(kk3)