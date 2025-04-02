import copy
import os
import time
import json
from LLM_CodeGen_OP import obt_scores
from ga_src.env import N_TRIALS, TP, GPT4o, GEMINI, GLM, CLAUDE, QWEN, GPT, GC_GENERATION
from ga_src.pipeline import fun_search, get_prob_inst_set, run_single_trial, get_prob_test, run_test_with_timeout, \
    run_test
from interaction import obt_record
from utils import open_pkl, write_pkl, score_calc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from MOTSP.motsp import MOTSP, TSP
from scipy.stats import ranksums


def test(A, B, prob_mode):
    # 进行Wilcoxon秩和检验
    stat, p_value = ranksums(A, B)
    # 根据p值判断结果
    if p_value < 0.05:
        if prob_mode == 'MOP' or prob_mode == 'MOKP':
            if stat > 0:
                return '$-$' # A > B, A is significant worse than B
            else:
                return '$+$' # A < B, A is significant better than B
        elif prob_mode == 'MOTSP':
            if stat > 0:
                return '$+$' # A > B, A is significant worse than B
            else:
                return '$-$' # A < B, A is significant better than B
    else:
        return '$\\approx$'

OURMETHOD = 'LLMMOP'
# baselines = ['MOEAD', 'CTAEA', 'RVEA', 'NSGA2', 'AGEMOEA', 'SMSEMOA']
baselines = ['CTAEA', 'RVEA', 'NSGA2', 'AGEMOEA', 'SMSEMOA']
# 0, 2, 3, 4 'CTAEA', 'NSGA2', 'AGEMOEA', 'SMSEMOA'
def draw(llm_algs, handcrafted_algs, pname, fontsize=18):
    plt.clf()
    x = list(range(10, 101, 10))
    for llmalg in llm_algs.keys():
        if llmalg == OURMETHOD:
            plt.plot(x, llm_algs[llmalg], label=f'{llmalg}', color='red', ls='-.')
        else:
            plt.plot(x, llm_algs[llmalg], label=f'{llmalg}')
    for alg in handcrafted_algs.keys():
        plt.plot(x, np.ones_like(x) * handcrafted_algs[alg], label=f'{alg}')

    # plt.title("The convergence Curve of LLM-Generated Code")

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Fitness Evaluation Times", fontsize=fontsize)
    plt.ylabel(f"Averaged Score on {pname}", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='lower right')

    output_name = f'outputs/pictures/demo/{pname}'
    # plt.show()
    plt.savefig(output_name+'.svg', bbox_inches='tight')
    plt.savefig(output_name+'.eps', bbox_inches='tight')

def draw_investigate(cmp, cmp_names, add='', pname='MOP', fontsize=18):
    x = range(10, 101, 10)
    plt.clf()
    for i, alg in enumerate(cmp.keys()):
        plt.plot(x, np.ones_like(GC_GENERATION) * cmp[alg], label=f'{cmp_names[i]}')

    plt.title("Convergence Curve Obtained by LLMs")

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Fitness Evaluation Times", fontsize=fontsize)
    plt.ylabel(f"Scores on {pname}", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='lower right')

    output_name = f'outputs/pictures/{pname}'
    # plt.show()
    plt.savefig(output_name+f'{add}'+'.jpg', bbox_inches='tight')
    plt.savefig(output_name+f'{add}'+'.eps', bbox_inches='tight')

def compute(results):
    scores = []
    for data in results[:]:
        score = score_calc(data, bias=1)
        scores.append(score)
    return scores
def demo_compare(prob_name='MOP', file='gpt-4', choices=None):

    # obtain handcrafted algs' scores
    handnames = np.array(baselines)[choices].tolist()
    filename = f'outputs/{file}-{prob_name}'
    # ttest = open_pkl(filename+f'-summarize')
    handcrafted = np.array(open_pkl(filename+f'-summarize'))[choices].tolist()
    handscores = compute(handcrafted)
    hand_cmps = {}
    for k, name in enumerate(handnames):
        hand_cmps[name]=handscores[k]

    # read scores of OURMETHOD
    cruns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'x']
    RS = ['', 'R2-', 'R3-']
    best_score_lst = []
    for _add in cruns:
        temp = []
        for R in RS:
            rfile = f'evolve/{R}{file}-1106-preview-{prob_name}-{_add}'
            temp.append(open_pkl(rfile)['scores'][0])
        best_score_lst.append(np.mean(temp))
    # print()

    # records = open_pkl(filename)
    # alg_records, score_record = records['alg_record'], records['score_record']
    # best_score_lst = [v[0] for v in score_record]

    # read scores of EOH
    RSEOH = ['', '_R2', '_R3']
    eoh_score_lst = []
    for _add in cruns:
        temp = []
        for R in RSEOH:
            eoh_file = f'eoh_data/EOH_RES_{prob_name}{R}/results/pops_best/population_generation_{_add}.json'
            with open(eoh_file, 'r', encoding='utf-8') as f:
                temp.append(json.load(f)['objective'])
                # print()
        eoh_score_lst.append(np.mean(temp))

    # eoh_dir = f'eoh_solve/EOH_RES_{prob_name}/results/pops_best/'
    # eoh_files = os.listdir(eoh_dir)
    # eoh_score_lst = []
    # for file in eoh_files:
    #     with open(eoh_dir+file, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #         eoh_score_lst.append(data['objective'])
    llmalgs = {
        OURMETHOD: best_score_lst,
        'EOH': eoh_score_lst
    }
    draw(llmalgs, hand_cmps, prob_name)


def draw_box(data_dict, prob_name, output_dir):
    plt.clf()
    labels = list(data_dict.keys())
    data = list(data_dict.values())

    # 绘制箱线图
    plt.boxplot(data, patch_artist=True, medianprops=dict(color="red"),
                boxprops=dict(facecolor="lightblue", edgecolor="black"))

    # 设置标题和标签
    plt.title(f'Score Distribution on {prob_name}', fontsize=18)
    plt.ylabel(f'Score on {prob_name}', fontsize=18)
    plt.xticks(range(1, len(labels) + 1), labels, fontsize=18)

    # 设置y轴刻度
    plt.yticks(fontsize=18)
    plt.savefig(output_dir + '.eps', bbox_inches='tight')
    # 保存图片
    # plt.savefig(output_dir + '.jpg', bbox_inches='tight', dpi=150)


# def draw_box(data_dict, prob_name, output_dir):
#     plt.clf()
#     labels = list(data_dict.keys())
#     data = list(data_dict.values())
#     plt.boxplot(data)
#     # plt.title('')
#     plt.ylabel(f'Score on {prob_name}')
#     plt.xticks(range(1, len(labels) + 1), labels)
#     # plt.grid(True)
#     # plt.savefig(output_name+'.svg', bbox_inches='tight')
#     plt.savefig(output_dir + '.jpg', bbox_inches='tight')

def investigation_prompts(llm='gpt-4', prob_name='MOP'):
    result = {
        f'informative': open_pkl(f'outputs/{llm}-{prob_name}')['score_record'][0]
    }
    eoh_file = f'eoh_data/EOH_RES_{prob_name}/results/pops/population_generation_1.json'
    eoh_data = []
    with open(eoh_file) as f:
        temp = json.load(f)
        result['concise'] = [data['objective'] for data in temp]
    output_dir = f'outputs/pictures/inv_prompt/{prob_name}'
    draw_box(result, prob_name, output_dir)

def validation_process(prob_name='MOP', file='gpt-4', choices=None):

    # run our method
    our_file = f'outputs/{file}-{prob_name}-{OURMETHOD}'
    ournames = [OURMETHOD]
    score1 = -10000000
    ouralg = None
    cnt1 = 1
    for R in ['R2-', 'R3-']:
        cnt1 += 1
        filename = f'evolve/{R}gpt-4-1106-preview-{prob_name}-x'
        data = open_pkl(filename)
        ouralg = data['pop'][0]
        run_validation([ouralg], ournames, our_file+f'-R{cnt1}', prob_name)
        # if score1 < data['scores'][0]:
        #     score1 = data['scores'][0]
        #     ouralg = data['pop'][0]

        # print()
    # run_validation([ouralg], ournames, our_file, prob_name)

    # do = open_pkl(f'MOTSP/MOTSPTEST.pkl')
    # d1 = open_pkl(f'MOTSP/motsp_test_clustered.pkl')
    # d2 = open_pkl(f'MOTSP/motsp_test_exponential.pkl')
    # d3 = open_pkl(f'MOTSP/motsp_test_normal.pkl')
    # d4 = open_pkl(f'MOTSP/motsp_test_poisson.pkl')
    # ff = do[:8]
    # tt = [d1, d2, d3, d4]
    # for data in tt:
    #     for val in data:
    #         ff.append(val)
    # write_pkl(ff, f'MOTSP/MOTSPTEST.pkl')

    # run handcrafted algs
    # handnames = np.array(baselines)[choices].tolist()
    # handcrafted_file = f'outputs/{file}-{prob_name}-handcrafted'
    # # has = open_pkl(handcrafted_file)
    # run_validation(handnames, handnames, handcrafted_file, prob_name, has=None)
    # print()
    # # print(compute(open_pkl(handcrafted_file)))

    # run EOH
    eoh_file = f'outputs/{file}-{prob_name}-EOH'
    eohnames = ['EOH']
    eohalg = None
    score2 = -100000
    cnt2 = 1
    for R in ['_R2', '_R3']:
        cnt2 += 1
        with open(f'eoh_data/EOH_RES_{prob_name}{R}/results/pops_best/population_generation_x.json') as f:
            temp = json.load(f)
            eohalg = temp['code']
            run_validation([eohalg], eohnames, eoh_file+f'-R{cnt2}', prob_name)
            # if score2 < temp['objective']:
            #     score2 = temp['objective']
            #     eohalg = temp['code']
    # run_validation([eohalg], eohnames, eoh_file, prob_name)

def prob_draw(drawdata, xlabel, ylabel, fontsize=18):
    colors = ['r', 'b', 'g', 'c', 'm', 'k', 'orange']
    shapes = ['-', '--', '-.', ':']
    for pname, algdata in drawdata.items():
        plt.clf()
        count_ = -1
        for algname, data in algdata.items():
            count_ += 1
            x = list(range(len(data)))
            plt.title(f'{pname}', fontsize=fontsize)
            plt.plot(x, data, label=f'{algname}', ls=shapes[count_%len(shapes)], color=colors[count_%len(colors)])
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel(xlabel, fontsize=fontsize)
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.legend(fontsize=fontsize)
        # output_name = f'outputs/pictures/{pname}.jpg'
        output_name = f'outputs/pictures/{pname}'
        plt.savefig(output_name+'.jpg', bbox_inches='tight')
        plt.savefig(output_name+'.eps', bbox_inches='tight')

def MOPs_draw(prob_mode):
    dataset = open_pkl(f'outputs/gpt-4-{prob_mode}-summarize-test')
    cmp_alg_names = None
    problem_names = None
    ylabel = None
    if prob_mode == 'MOKP':
        ylabel = 'HV'
        cmp_alg_names = ['LLMOPT', 'CTAEA', 'RVEA', 'NSGA2', 'AGEMOEA', 'SMSEMOA']
        problem_names = ['MOKP1', 'MOKP2', 'MOKP3', 'MOKP4', 'MOKP5', 'MOKP6', 'MOKP7', 'MOKP8', 'MOKP9', 'MOKP10']
    elif prob_mode == 'MOP':
        ylabel = 'IGD'
        # problem_names = ['WFG1', 'WFG2', 'WFG3', 'WFG4', 'WFG5', 'WFG6', 'WFG7', 'WFG8', 'WFG9', 'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7']
        problem_names = ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt5', 'zdt6', 'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7']
        cmp_alg_names = ['LLMOPT', 'MOEAD', 'CTAEA', 'RVEA', 'NSGA2', 'AGEMOEA', 'SMSEMOA']
    # elif prob_mode == 'MOTSP':
    #     ylabel = 'HV'
    #     cmp_alg_names = ['LLMOPT', 'MOEAD', 'CTAEA', 'RVEA', 'NSGA2', 'AGEMOEA', 'SMSEMOA']
    #     problem_names = ['MOTSP1', 'MOTSP2', 'MOTSP3', 'MOTSP4', 'MOTSP5', 'MOTSP6', 'MOTSP7', 'MOTSP8', 'MOTSP9', 'MOTSP10']
    else:
        assert False
    drawdata = {pname: {} for pname in problem_names}
    for j, probname in enumerate(problem_names):
        for i, algname in enumerate(cmp_alg_names):
            mean_val = np.mean(np.array(dataset[i]['record'][j]), axis=0).tolist()
            drawdata[probname][algname] = mean_val
    prob_draw(drawdata, xlabel='Generation', ylabel=ylabel)

def MOPs_tabulate(prob_mode, choices, inf=None, R1='', R2=''):
    dataset = np.array(open_pkl(f'outputs/gpt-4-{prob_mode}-handcrafted'))[choices].tolist()
    dataset.append(open_pkl(f'outputs/gpt-4-{prob_mode}-EOH{R2}'))
    dataset.append(open_pkl(f'outputs/gpt-4-{prob_mode}-{OURMETHOD}{R1}'))
    # our = open_pkl(f'outputs/gpt-4-{prob_mode}-{OURMETHOD}')
    # eoh = open_pkl(f'outputs/gpt-4-{prob_mode}-EOH')

    Bcmp = copy.deepcopy(dataset[-1])
    cmp_alg_names = np.array(baselines)[choices].tolist() + ['EOH'] + [OURMETHOD]
    problem_names = None
    ylabel = None
    inverse = False
    if prob_mode == 'MOKP':
        ylabel = 'HV'
        problem_names = [f'MOKP{val}' for val in range(1, 21)]
    elif prob_mode == 'MOP':
        ylabel = 'IGD'
        problem_names = [f'zdt{val}' for val in range(1, 7)] + [f'dtlz{val}' for val in range(1, 8)]
    elif prob_mode == "MOTSP":
        inverse = True
        ylabel = 'HV'
        problem_names = [f'MOTSP{k}' for k in range(1, 21)]
    else:
        assert False
    collected = {pname: {} for pname in problem_names}
    collected_sign = {pname: {} for pname in problem_names}
    for j, probname in enumerate(problem_names):
        for i, algname in enumerate(cmp_alg_names):
            if inverse: # 200 generations
                ddata = (1 - np.array(dataset[i]['record'][j]))[:, -1]
                B = (1 - np.array(Bcmp['record'][j]))[:, -1]
                mean_val = np.mean(1 - np.array(dataset[i]['record'][j]), axis=0).tolist()[-1]
            else:
                ddata = np.array(dataset[i]['record'][j])[:, -1]
                B = np.array(Bcmp['record'][j])[:, -1]
                mean_val = np.mean(np.array(dataset[i]['record'][j]), axis=0).tolist()[-1]
            collected[probname][algname] = mean_val
            collected_sign[probname][algname] = test(ddata, B, prob_mode)

    output_write = f''
    records = {pname: [] for pname in cmp_alg_names}
    counting = -1
    for probname, cmps in collected.items():
        labels = []
        counting += 1
        output_write += f'{probname}'
        if inf is not None:
            for val in inf[counting]:
                output_write += f' & {val}'
        vals = []
        for key, val in cmps.items():
            # print(key, end=' ')
            vals.append(val)
            labels.append(collected_sign[probname][key])
            records[key].append(val)
        # print()
        if inverse:
            argmin = np.argmin(-np.array(vals))
        else:
            argmin = np.argmin(vals)
        for i, val in enumerate(vals):
            if i < len(vals)-1:
                label = labels[i]
            else:
                label = ''
            if i == argmin:
                output_write += f' & \\textbf{{{val:.3e}}}{label}'
            else:
                output_write += f' & {val:.3e}{label}'
        output_write += f' \\\\ \n'

    # end of
    output_write += f'Averaged {ylabel}'
    for _ in inf[0]:
        output_write += f' & - '
    for pname, val_lst in records.items():
        output_write += f' & {np.mean(val_lst):.3e}'
    output_write += f' \\\\ \n'

    print(output_write)

def preprocess(mode, prob_name, add=''):
    lst = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'x']
    pops, scores = [], []
    for idx in lst:
        filename = f'evolve/{add}{mode}-{prob_name}-{idx}'
        tdata = open_pkl(filename)
        pops.append(tdata['pop'])
        scores.append(tdata['scores'])
    # pops = np.array(pops)
    # scores = np.array(scores)
    outname = 'investigation/'+mode+f'-{prob_name}'
    write_pkl({'alg_record': pops, 'score_record': scores}, filename=outname)

def check_has(algname, has):
    if has is None:
        return None
    for data in has:
        if algname == data['alg']:
            return data
    return None

def run_validation(alg_pop, alg_names, filename, prob_name, has=None, testing=True):
    datas = []
    for idx, alg in enumerate(alg_pop):
        alg_name = alg_names[idx]
        other = check_has(alg_name, has)
        if other is not None:
            datas.append(other)
            continue
        time_slot = time.time()
        print(f'{alg_name} on {prob_name} started')
        result = fun_search([alg], testing=testing, pname=prob_name)[0]
        result['time'] = time.time() - time_slot
        result['alg'] = alg_name
        datas.append(result)
        print(f'{alg_name} on {prob_name} finished <score: {score_calc(result, bias=1)}> <time: {result["time"]}>')
    if len(alg_pop) == 1:
        write_pkl(datas[0], filename)
    else:
        write_pkl(datas, filename)

def data_property(pname):
    datasets = get_prob_inst_set(pname)
    # for idx, data in enumerate(datasets):
    #     print(f'{pname}-{idx+1}: n_var ({datasets[idx].n_var}) n_obj ({datasets[idx].n_obj})')
    # print('validation data:\n')

    datasets = get_prob_test(pname)

    infs = []
    if pname == 'MOP':
        for idx, data in enumerate(datasets):
            infs.append([datasets[idx].n_var])
        # return None
    elif pname == 'MOKP':
        for idx, data in enumerate(datasets):
            infs.append([datasets[idx].n_var, datasets[idx].C])
    elif pname == 'MOTSP':
        distris = ['uniform'] * 8 + ['clustered'] * 3 + ['exponential'] * 3 + ['normal'] * 3 + ['poisson'] * 3
        for idx, data in enumerate(datasets):
            infs.append([datasets[idx].n_var, distris[idx]])
    return infs

# investigate the performance of different LLMs
def llm_inv1(compared_llms, compared_names, prob='MOP'):
    cmps = {}
    for llm in compared_llms:
        filename = 'investigation/'+llm+f'-{prob}'
        records = open_pkl(filename)
        alg_records, score_record = records['alg_record'], records['score_record']
        cmps[llm] = [v[0] for v in score_record]
    print()
    draw_investigate(cmps, compared_names)

def llm_inv2(compared_llms, compared_names, prob='MOP'):
    for k, llm in enumerate(compared_llms):
        filename = 'investigation/records-' + llm + f'-{prob}'
        data = open_pkl(filename)[llm]
        length = len(data['pop'])
        success, failed, r_success = 0, 0, 0
        for idx in range(length):
            code = data['pop'][idx]
            re_time = data['refined_counts'][idx]
            if code is None: # failed repair
                failed += 1
            else:
                if re_time > 0: # repaired
                    failed += 1
                    r_success += 1
                else: # no repair
                    success += 1
        print(f'{compared_names[k]} & 20 & {success} & {failed} & {r_success/failed*100:.1f}\%')
def ablation1(mode=GPT, prob='MOP'):
    compared_files = [f'investigation/{mode}-{prob}', f'investigation/{mode}-ox2-{prob}']
    compared_llms = ['GPT4', 'GPT4-no-dc']
    cmps = {}
    for lid, llm in enumerate(compared_llms):
        filename = compared_files[lid]
        records = open_pkl(filename)
        alg_records, score_record = records['alg_record'], records['score_record']
        cmps[llm] = [v[0] for v in score_record]
    draw_investigate(cmps, compared_llms, add='-nodc')

def time_tabulate(choices, cmps):
    output = f''
    timetable = {}
    for k, prob_mode in enumerate(['MOP', 'MOKP', 'MOTSP']):
        dataset = np.array(open_pkl(f'outputs/gpt-4-{prob_mode}-handcrafted'))[choices].tolist()
        dataset.append(open_pkl(f'outputs/gpt-4-{prob_mode}-EOH{cmps[k][1]}'))
        dataset.append(open_pkl(f'outputs/gpt-4-{prob_mode}-{OURMETHOD}{cmps[k][0]}'))
        for data in dataset:
            algname, tmcost = data['alg'], data['time']
            if not algname in timetable.keys():
                timetable[algname] = [tmcost]
            else:
                timetable[algname].append(tmcost)
    for key, vals in timetable.items():
        output += f'{key}'
        for val in vals:
            output += f' & {np.round(val, 0)}'
        output += '\\\\\n'
    print(output)
def test_code():
    def load(file):
        output = f''
        with open(file, 'r') as f:
            for line in f.readlines():
                output += line
        return output

    original = load('MOEA')
    d_selection = load('MOEA-selection')
    d_crossover = load('MOEA-crossover')
    d_mutation = load('MOEA-mutation')
    # prob_valid = get_prob_inst_set(TP)
    # prob_tests = get_prob_test(TP)
    algs = [d_selection, d_crossover, d_mutation, original]
    names = ['selection', 'crossover', 'mutation', 'baseline']
    for i in range(len(algs)):
        our_file = f'outputs/MOEA-{OURMETHOD}-{names[i]}'
        ournames = [names[i]]
        ouralgs = [algs[i]]
        run_validation(ouralgs, ournames, our_file, TP, testing=True)

    # for code in [d_selection, d_crossover, d_mutation, original]:
    #     print('start')
    #     print(code)
    #     run_test_with_timeout(code, TP, 60)
    #     problems = get_prob_inst_set(TP)
    #     run_test(code, problems)
    #     print('end')
    # print(original)
def tabulate_testcode():
    files = ['outputs/MOEA-LLMMOP-baseline', 'outputs/MOEA-LLMMOP-selection', 'outputs/MOEA-LLMMOP-crossover', 'outputs/MOEA-LLMMOP-mutation']
    cmp_alg_names = ['baseline', 'selection', 'crossover', 'mutation']
    dataset = [open_pkl(file) for file in files]
    problem_names = [f'zdt{val}' for val in range(1, 7)] + [f'dtlz{val}' for val in range(1, 8)]

    collected = {pname: {} for pname in problem_names}
    for j, probname in enumerate(problem_names):
        for i, algname in enumerate(cmp_alg_names):
            mean_val = np.mean(np.array(dataset[i]['record'][j]), axis=0).tolist()[-1]
            collected[probname][algname] = mean_val

    output_write = f''
    records = {pname: [] for pname in cmp_alg_names}
    counting = -1
    for probname, cmps in collected.items():
        counting += 1
        output_write += f'{probname}'

        vals = []
        for key, val in cmps.items():
            # print(key, end=' ')
            vals.append(val)
            records[key].append(val)
        # print()

        argmin = np.argmin(vals)
        for i, val in enumerate(vals):
            if i == argmin:
                output_write += f' & \\textbf{{{val:.3e}}}'
            else:
                output_write += f' & {val:.3e}'
        output_write += f' \\\\ \n'

    # end of
    output_write += f'Averaged IGD'
    for pname, val_lst in records.items():
        output_write += f' & {np.mean(val_lst):.3e}'
    output_write += f' \\\\ \n'

    print(output_write)
def inv_gemini():
    for case in ['T1-', 'T2-', 'T3-']:
    # files = ['T1-api-gemini-1.5-pro-MOP-10', 'T2-api-gemini-1.5-pro-MOP-10', 'T3-api-gemini-1.5-pro-MOP-10']
        files = [f'{case}api-gemini-1.5-pro-MOP-{x}' for x in range(1, 11)]
        for file in files:
            filename = f'evolve/{file}'
            data = open_pkl(filename)
            print(data['fault'], data['scores'][0])
            # print(file)
            # print(data)
        print()

def show_code(IDS, problems):
    for k in range(len(IDS)):
        _id = IDS[k]
        prob = problems[k]
        filename = f'evolve/{_id}gpt-4-1106-preview-{prob}-x'
        code = open_pkl(filename)['pop'][0]
        print(f'GPT-4 for problem: {prob}. Code:\n')
        print(code)
    print(f'Claude for problem: MOP. Code:\n')
    filename = f'evolve/claude-3-opus-MOP-x'
    code = open_pkl(filename)['pop'][0]
    print(code)
if __name__ == '__main__':

    # show code
    show_code(IDS=['', 'R2-', ''], problems=['MOP', 'MOKP', 'MOTSP'])

    # cmps = [('', '-R2'), ('-R2', '-R2'), ('', '-R3')]
    # investigate on the initial prompts
    # investigation_prompts(prob_name='MOP')
    # investigation_prompts(prob_name='MOKP')
    # investigation_prompts(prob_name='MOTSP')

    # run for the testing set
    # validation_process(prob_name=TP, choices=[0,2,3,4]) # TP must equals to ...


    # investigate on the training set
    # demo_compare(prob_name='MOP', choices=[0,2,3,4])
    # demo_compare(prob_name='MOKP', choices=[0,2,3,4])
    # demo_compare(prob_name='MOTSP', choices=[0,2,3,4])

    # inv_gemini()
    # investigate on the running time
    # time_tabulate(choices=[0,2,3,4], cmps=cmps)

    # draw pictures and tables on validation sets
    # for prob in ['MOP', 'MOKP', 'MOTSP']:
    #     MOPs_tabulate(prob_mode=prob, choices=[0,2,3,4], inf=data_property(prob), add_='')
        # MOPs_draw(prob_mode=prob)
    # for kk, prob in enumerate(['MOP', 'MOKP', 'MOTSP']):
    #     R1, R2 = cmps[kk]
    #     MOPs_tabulate(prob_mode=prob, choices=[0,2,3,4], inf=data_property(prob), R1=R1, R2=R2)

    # investigation on different LLMs
    # for mode in [GPT4o, GEMINI, QWEN, CLAUDE]:
    #     add = ''
    #     if mode == GEMINI:
    #         add = 'T1-'
    #     preprocess(mode=mode, prob_name='MOP', add=add)
    # compared_llms = [GPT, GPT4o, GEMINI, CLAUDE]
    # compared_names = ['GPT4', 'GPT4O', 'GEMINI', 'CLAUDE']
    # llm_inv1(compared_llms, compared_names)
    # llm_inv2(compared_llms, compared_names)


    # test_code()
    # tabulate_testcode()
    # preprocess(mode=GPT, prob_name='MOP', add='-ox2-')
    # ablation1()
