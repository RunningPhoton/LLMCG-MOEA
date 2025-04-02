import sys
import time
import traceback
from copy import deepcopy
import openai

from descriptions import *
from ga_src.env import GC_POP_SIZE, APIUS_SLEEP, MAX_RUNNING_TIME, GEMINI, GPT, QWEN, MAX_MUTATION, MAX_REPAIR
from ga_src.pipeline import fun_search, run_test, run_test_with_timeout
from llm_sessions import my_sessions
from utils import score_calc, xml_text_to_code, write_pkl
from MOTSP.motsp import TSP, MOTSP

def obt_scores(algs, pname):
    try:
        results = fun_search(algs, pname=pname)
        scores = []
        # print(results)
        for res in results:
            # score = max([0, 1-res['IGD_mean']-res['IGD_std']])
            score = score_calc(res, bias=1)
            if np.isnan(score):
                output = f"Invalid results"
                return None, output
            scores.append(score)
        return scores, ''
    except Exception as e:

        # 获取异常信息
        exc_type, exc_value, exc_traceback = sys.exc_info()

        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)

        # 打印异常类型、异常信息和发生异常的行号
        output = f"An exception of type {exc_type.__name__} occurred in `obt_scores`. Message: {e}\n"
        output += f"Last few lines of the traceback:\n"
        start = False
        for line in tb_lines:
            if 'in next_generation' in line:
                start = True
            if start:
                output += f'{line}\n'
        return None, output

def fault_process(information, messages, Ps, MAX_TEST=MAX_REPAIR):
    _count = 0
    info = information
    while True:
        # assert _count < MAX_TEST, 'too many errors'
        try:
            code = xml_text_to_code(info.content)
        except Exception as e:
            print(f'code extraction failed, error: {e}')
            print(f'content is:\n {info.content}')
            return None, _count # 解析失败，无法修复，需重新访问LLM
        new_messages = deepcopy(messages)
        new_messages.append(info)
        # state, error = run_test(code, Ps['problem'])
        state, error = run_test_with_timeout(code, Ps['problem'], TIMEOUT=MAX_RUNNING_TIME)
        if 'nonnumerical result' in error:
            print(f'{error}')
            return None, _count # 发生运行中错误（除零等），需重新访问LLM
        if state == True: return code, _count # 试运行成功，返回代码和修复次数
        print(error)
        # print(f"\n{code}\n")

        if _count >= MAX_TEST:
            print('too many errors in fault process')
            return None, _count # 修复失败，返回修复次数

        info, _ = my_sessions[Ps["model_name"]](Ps['client'], new_messages, model_name=Ps['model_name'], temperature=Ps['temperature'], max_token=Ps['max_token'], add=None, _msg=failed_refine(error))
        Ps['fault'] += 1
        _count += 1

# initialization
def initialize(logger, Ps):
    initial_pop = []
    messages = [
        {"role": "system", "content": PROMPTS['system']},
        {"role": "user", "content": PROMPTS['task']},
    ]
    scores = []
    code_count = 0
    for _ in range(GC_POP_SIZE):
        refine = 0
        score_lst = None
        individual_code = None
        while score_lst is None:
            if refine > 0: print(f'refine: {refine}')
            information, messages = my_sessions[Ps["model_name"]](Ps['client'], messages, model_name=Ps['model_name'], temperature=Ps['temperature'], max_token=Ps['max_token'], add=None)
            individual_code, error_count = fault_process(information, messages, Ps)
            if individual_code is None:
                Ps['fault'] += 1
                print(f'init failed: individual code is None')
                continue
            # print(new_alg)
            score_lst, output = obt_scores([individual_code], pname=Ps['problem'])
            if score_lst is None:
                Ps['fault'] += 1
                print(output)
            else:
                code_count += 1
                print(f'{code_count}-th code initialized, score: {score_lst}')
            refine += 1

        scores.append(score_lst[0])
        initial_pop.append(individual_code)
        logger.info(individual_code)

    init_data = {
        'pop': initial_pop,
        'scores': scores,
        'fault': Ps['fault']
    }
    write_pkl(init_data, Ps['init_to_save'])
    write_pkl(messages, Ps['message_to_save'])

# generate a novel alg according to existing pop
def code_evolve(logger, pop, scores, messages, best_num, worst_num, Ps):
    # best_algs, best_scores = pop[:best_num], scores[:best_num]
    # worst_algs, worst_scores = pop[-worst_num:], scores[-worst_num:]
    # content = code_evolve_to_description(best_algs, best_scores, worst_algs, worst_scores)
    content = code_evolve_to_description_multi(algs=pop, scores=scores)

    _msg, model_name = content, Ps['model_name']
    # if model_name == GPT:
    #     _msg = [{
    #         "type": "text",
    #         "text": content
    #     }]
    # elif model_name == GEMINI:
    #     _msg = content
    # elif model_name == QWEN:
    #     _msg = content

    individual_code, EVO_count = None, 0
    while individual_code is None:
        EVO_count += 1
        print(f'EVO_count: {EVO_count}')
        information, messages = my_sessions[Ps["model_name"]](Ps['client'], messages, model_name=Ps['model_name'], temperature=Ps['temperature'], max_token=Ps['max_token'], add=None, _msg=_msg)
        individual_code, error_count = fault_process(information, messages, Ps)
        if individual_code is None:
            Ps['fault'] += 1
    # logger.info(content)
    # logger.info(individual_code)
    return individual_code

def code_mutation(logger, alg, messages, P_MU=5/GC_POP_SIZE, Ps=None):
    if np.random.rand() < P_MU:
        content = code_mutation_to_description(alg)
        _msg, model_name = content, Ps['model_name']
        # if model_name == GPT:
        #     _msg = [{
        #         "type": "text",
        #         "text": content
        #     }]
        # elif model_name == GEMINI:
        #     _msg = content
        # elif model_name == QWEN:
        #     _msg = content
        individual_code, MU_count = None, 0
        while individual_code is None:
            information, messages = my_sessions[Ps["model_name"]](Ps['client'], messages, model_name=Ps['model_name'], temperature=Ps['temperature'], max_token=Ps['max_token'], add=None, _msg=_msg)
            individual_code, error_count = fault_process(information, messages, Ps)
            MU_count += 1
            if individual_code is None:
                Ps['fault'] += 1
            print(f'MU_count: {MU_count}')
            if MU_count >= MAX_MUTATION:
                print('too many failures, mutation failed')
                return alg
        # logger.info(content)
        # logger.info(alg)
        return individual_code
    else:
        return alg


def fault_test(Ps, INV_NUM=20):
    messages = [
        {"role": "system", "content": PROMPTS['system']},
        {"role": "user", "content": PROMPTS['task']},
    ]
    initial_pop = []
    error_counts = []
    for _idx in range(INV_NUM):
        information, messages = my_sessions[Ps["model_name"]](Ps['client'], messages, model_name=Ps['model_name'], temperature=Ps['temperature'], max_token=Ps['max_token'], add=None)
        individual_code, error_count = fault_process(information, messages, Ps)
        initial_pop.append(individual_code)
        error_counts.append(error_count)
        print(f'code is: \n{individual_code}\n{error_count}\n')

    init_data = {
        'pop': initial_pop,
        'refined_counts': error_counts
    }
    return init_data