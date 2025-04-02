import copy
import os

import numpy as np
import openai

from ga_src.env import GC_POP_SIZE, GC_GENERATION, apikeys, base_urls
from ga_src.pipeline import fun_search, run_test
from utils import get_logger, open_pkl, update, write_pkl
from LLM_CodeGen_OP import initialize, code_evolve, code_mutation, fault_process, obt_scores, fault_test


def advanced_evolve(text):
    files = os.listdir('evolve/')
    # print(files)
    res = []
    gen_lst = []
    for file in files:
        if text in file:
            res.append(f'evolve/{file}')
            gen_lst.append(int(file.split('-')[-1]))
    if len(res) > 0:
        arg = np.argmax(gen_lst)
        return res[arg]
    else:
        return None
def obt_record(prob_name, output_file):
    files = os.listdir('evolve/')
    alg_record = []
    score_record = []
    for file in files:
        if prob_name in file:
            filename = f'evolve/{file}'
            data = open_pkl(filename)
            pop, scores = data['pop'], data['scores']
            alg_record.append(pop)
            score_record.append(scores)
            print(f'{file}: score: {scores[0]}')

    write_pkl({
        'alg_record': alg_record,
        'score_record': score_record
    }, output_file)
def openai_code_evolve(client, model_name, temperature, max_token, problem, R=''):
    # add_text = '-ox2'
    add_text = ''
    logging_file = f'responses/{R}{model_name}-{problem}{add_text}'
    init_to_save = f'init/{R}{model_name}-{problem}{add_text}'
    evolve_to_save = f'evolve/{R}{model_name}-{problem}{add_text}'
    message_to_save = f'messages/{R}{model_name}-{problem}{add_text}'
    restart_text = f'{R}{model_name}-{problem}{add_text}'
    Ps = {
        'client': client,
        'model_name': model_name,
        'temperature': temperature,
        'max_token': max_token,
        'init_to_save': init_to_save,
        'message_to_save': message_to_save,
        'problem': problem,
        'fault': 0
    }

    # initialization
    logger = get_logger(f'{logging_file}.log', name=f'{logging_file}', mode='w')
    logger.info('start')
    initialize(logger, Ps)

    # replace to initialization
    # logger = get_logger(f'{logging_file}.log', name=f'{logging_file}', mode='a')


    # logger.info('evolving-----------------------------------')
    init_filename = advanced_evolve(restart_text)
    # init_filename = None
    if init_filename is None:
        init_filename = init_to_save
        cur_gen = 0
    else:
        cur_gen = int(init_filename.split('-')[-1])
    init_data = open_pkl(init_filename)
    pop = init_data['pop']
    scores = init_data['scores']
    Ps['fault'] = init_data['fault']
    messages = open_pkl(message_to_save)
    pop, scores = update(pop, scores, GC_POP_SIZE)
    print(scores)
    # main loop

    for _g in range(1, GC_GENERATION+1):
        print(f'alg evolving generation: {_g}')
        if _g <= cur_gen: continue
        for _i in range(GC_POP_SIZE):
            refine = 0
            score_lst = None
            new_alg = None
            while score_lst is None:
                if refine > 0:
                    print(f'refine: {refine}')
                alg = code_evolve(logger, pop, scores, messages, best_num=None, worst_num=None, Ps=Ps)
                print('evolve end')
                # check for the quality of llm crossover
                # evo_score_lst, evo_output = obt_scores([alg], pname=Ps['problem'])
                # print(evo_score_lst)
                new_alg = code_mutation(logger, alg, messages, P_MU=1/GC_POP_SIZE, Ps=Ps)
                print('mutation end')
                # print(new_alg)
                score_lst, output = obt_scores([new_alg], pname=Ps['problem'])
                if score_lst is None:
                    Ps['fault'] += 1
                    print(output)
                refine += 1

            insert = True
            for i in range(len(scores)):
                temp = score_lst[0]
                if abs(scores[i] - temp) < 1e-2:
                    insert = False

            if insert == True or score_lst[0] > scores[0]:
                pop.append(new_alg)
                scores.append(score_lst[0])
                print(f'score: {score_lst[0]}')
                print('------------------------------------------------')
                pop, scores = update(pop, scores, GC_POP_SIZE)
                # logger.info(new_alg)
        print(f"Alg Evolving: {_g}, best score: {scores[0]}, fault: {Ps['fault']}")
        print(scores)
        cur_data = {
            'pop': copy.deepcopy(pop),
            'scores': copy.deepcopy(scores),
            'fault': Ps['fault']
        }
        write_pkl(data=cur_data, filename=evolve_to_save+f'-{_g}')

    logger.info('finish')


def fault_investigation(model_names, temperature, max_token, problem):
    code_records = {}
    for model_name in model_names:
        print(model_name)
        client = openai.OpenAI(api_key=apikeys[model_name], base_url=base_urls[model_name])
        Ps = {
            'client': client,
            'model_name': model_name,
            'temperature': temperature,
            'max_token': max_token,
            'problem': problem
        }
        code_records[model_name] = fault_test(Ps)
        record_to_save = f'investigation/records-{model_name}-{problem}'
        write_pkl(code_records, filename=record_to_save)