from utils import open_pkl, write_pkl

import numpy as np
import matplotlib.pyplot as plt

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
        output_name = f'outputs/pictures/{pname}'
        plt.savefig(output_name+'.jpg', bbox_inches='tight')
        plt.savefig(output_name+'.eps', bbox_inches='tight')

def MOPs_draw(prob_mode, offset=0):
    dataset = open_pkl(f'outputs/gpt-4-{prob_mode}-summarize-test')
    cmp_alg_names = None
    problem_names = None
    ylabel = None
    if prob_mode == 'MOTSP':
        ylabel = 'HV'
        cmp_alg_names = ['LLMOPT', 'MOEAD', 'CTAEA', 'RVEA', 'NSGA2', 'AGEMOEA', 'SMSEMOA']
        problem_names = ['MOTSP1', 'MOTSP2', 'MOTSP3', 'MOTSP4', 'MOTSP5', 'MOTSP6', 'MOTSP7', 'MOTSP8', 'MOTSP9', 'MOTSP10']
    else:
        assert False
    drawdata = {pname: {} for pname in problem_names}
    for j, probname in enumerate(problem_names):
        for i, algname in enumerate(cmp_alg_names):
            mean_val = np.mean(1-np.array(dataset[i]['record'][j+offset]), axis=0).tolist()
            drawdata[probname][algname] = mean_val
    prob_draw(drawdata, xlabel='Generation', ylabel=ylabel)

def MOPs_tabulate(prob_mode, offset=0):
    dataset = open_pkl(f'outputs/gpt-4-{prob_mode}-summarize-test')
    cmp_alg_names = None
    problem_names = None
    ylabel = None
    if prob_mode == "MOTSP":
        ylabel = 'HV'
        problem_names = [f'MOTSP{k}' for k in range(1, 11)]
        cmp_alg_names = ['LLMOPT', 'MOEAD', 'CTAEA', 'RVEA', 'NSGA2', 'AGEMOEA', 'SMSEMOA']
    else:
        assert False
    collected = {pname: {} for pname in problem_names}
    for j, probname in enumerate(problem_names):
        for i, algname in enumerate(cmp_alg_names):
            mean_val = np.mean(1 - np.array(dataset[i]['record'][j+offset]), axis=0).tolist()[-1]
            collected[probname][algname] = mean_val

    output_write = f''
    records = {pname: [] for pname in cmp_alg_names}
    for probname, cmps in collected.items():
        output_write += f'{probname}'
        vals = []
        for key, val in cmps.items():
            print(key, end=' ')
            vals.append(val)
            records[key].append(val)
        print()
        argmin = np.argmax(vals)
        for i, val in enumerate(vals):
            if i == argmin:
                output_write += f' & \\textbf{{{val:.3e}}}'
            else:
                output_write += f' & {val:.3e}'
        output_write += f' \\\\ \n'

    # end of
    output_write += f'Averaged {ylabel}'
    for pname, val_lst in records.items():
        output_write += f' & {np.mean(val_lst):.3e}'
    output_write += f' \\\\ \n'

    print(output_write)

def showcase(file):
    filename = f'outputs/{file}'
    records = open_pkl(filename)
    alg_records, score_record = records['alg_record'], records['score_record']
    best_alg = alg_records[-1][0]
    print(best_alg)

def time_tabulate(llm_mode='gpt-4'):
    names = ['LLMOPT', 'MOEAD', 'CTAEA', 'RVEA', 'NSGA2', 'AGEMOEA', 'SMSEMOA']
    probs = ['MOP', 'MOKP', 'MOTSP']

    for name in names:
        output = f'{name} '
        for prob in probs:
            filename = f'outputs/{llm_mode}-{prob}-time'
            tm_record = open_pkl(filename)
            if name in tm_record.keys():
                output += f'& {tm_record[name]:.0f}'
            else:
                output += f'& ---'
        output += '\n'
        print(output)
# for MOTSP and time assessment
def read_data():
    head = ['', 'R2-', 'R3-']
    tail = ['x', '10', '10']
    probs = ['MOP', 'MOKP', 'MOTSP']
    model = 'gpt-4-1106-preview'

    # file = f'outputs/gpt-4-MOP'
    # da1 = open_pkl(file)
    # print()

    # for pname in probs:
    #     for k in range(len(head)):
    #         filename = f'evolve/{head[k]}{model}-{pname}-{tail[k]}'
    #         try:
    #             data = open_pkl(filename)
    #             print('xxx')
    #         except:
    #             print('no this data')
if __name__ == '__main__':
    # showcase(file='gpt-4-MOTSP')
    read_data()
    # MOPs_tabulate(prob_mode='MOTSP')
    # MOPs_draw(prob_mode='MOTSP')
    # time_tabulate()