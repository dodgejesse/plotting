import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import sample_histograms
import scatter_with_vert_lines
import os
from collections import OrderedDict
import json


def get_search_algos():
    search_algos = OrderedDict()
    search_algos['spearmint_seq'] = {'name':'spearmint_seq', 'color':'C3'}
    search_algos['spearmint_2'] = {'name':'spearmint_2', 'color':'C5'}
    search_algos['spearmint_kover2'] = {'name':'spearmint_kover2', 'color':'C2'}
    search_algos['mixed_dpp_rbf'] = {'name':'mixed_dpp_rbf', 'color':'C0'}
    
    #search_algos['bayes_opt'] = 'bayes_opt'
    #search_algos['random'] = 'random'


    #search_algos['sobol_noise'] = 'sobol_noise'
    return search_algos

def get_iters():
    return 20

def get_space():
    #return "bigcube"
    return "arch"
    #return "dropl2learn_bad_lr"




def make_line_plot():
    all_info = {}
    search_algos = get_search_algos()
    iters = get_iters()
    space = get_space()
    
    for algo in search_algos:
        accs, durs = get_avg_and_std_dev('/home/jessedd/projects/plotting/icml_2018/data/durations_and_accs/stanford_sentiment_binary,nmodels=1,mdl_tpe=cnn,srch_tpe={},spce={},iters={}.txt'.format(algo, space, iters), algo)
        avg_dur_to_best, avg_best = get_avg_dur_to_best(accs, durs, iters)
        print algo, avg_dur_to_best, avg_best
        all_info[algo] = {'accs':accs, 'durs':durs, 'avg_dur_to_best':avg_dur_to_best, 'avg_best':avg_best}

    
    fig = plt.figure()
    cur_ax = fig.add_subplot(1,1,1)

    
    scatter_with_vert_lines.add_scatter(all_info, search_algos, space, cur_ax)
    plt.tight_layout()
    save_loc = get_save_loc(search_algos)
    print("printing in {}".format(save_loc))
    plt.savefig(save_loc,bbox_inches='tight')


def get_avg_dur_to_best(accs, durs, iters):
    durs_to_best = []
    bests = []
    eval_counter = 0
    best_dur, best_acc = 0,0

    for i in range(len(accs)):
        if accs[i] > best_acc:
            best_dur = durs[i]
            best_acc = accs[i]
            
        eval_counter += 1
        if eval_counter == iters:
            durs_to_best.append(best_dur)
            bests.append(best_acc)
            eval_counter = 0
            best_dur = 0
            best_acc = 0
    avg_dur_to_best = sum(durs_to_best) * 1.0/len(durs_to_best)
    avg_best = sum(bests) * 1.0/len(bests)
    return avg_dur_to_best, avg_best
    

def get_save_loc(search_algos):
    save_loc = "plot_drafts/wall_clock_time__"
    for s_t in search_algos:
        save_loc += str(s_t) + "_"
    save_loc = save_loc[:-1]
    save_loc += ",space=" + get_space()
    save_loc += ",iters=" + str(get_iters())
    save_loc += '.pdf'
    return save_loc


def get_avg_and_std_dev(file_loc, algo):
    #print file_loc
    with open(file_loc) as f:
        lines = f.readlines()
        examples = extract_examples(lines, algo)

        
        return examples


def get_algos_by_type():
    algos = {}
    algos['sequential'] = ['spearmint_seq', 'bayes_opt']
    algos['batch'] = {'spearmint_kover2':2, 'spearmint_2':get_iters()/2}
    algos['parallel'] = ['mixed_dpp_rbf', 'random', 'sobol_noise']
    return algos
    
def extract_examples(lines, algo):
    assert (algo in get_algos_by_type()['sequential']
            or algo in get_algos_by_type()['batch']
            or algo in get_algos_by_type()['parallel'])
    accuracies = []
    durations = []
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        elif line[0] == '[':
            if algo in get_algos_by_type()['parallel']:
                durations += [x[0] for x in json.loads(line)]
            elif algo in get_algos_by_type()['sequential']:
                cur_durs = [x[0] for x in json.loads(line)]
                cur_dur_sum = 0
                for cur_dur in cur_durs:
                    durations.append(cur_dur_sum + cur_dur)
                    cur_dur_sum += cur_dur
            elif algo in get_algos_by_type()['batch']:
                cur_durs = [x[0] for x in json.loads(line)]
                num_batches = get_algos_by_type()['batch'][algo]
                batch_size = len(cur_durs) / num_batches
                assert len(cur_durs) % num_batches == 0

                prev_sum_dur = 0
                cur_max_dur = 0
                for i in range(len(cur_durs)):
                    if i % batch_size == 0:
                        prev_sum_dur += cur_max_dur
                        cur_max_dur = cur_durs[i]
                    else:
                        if cur_durs[i] > cur_max_dur:
                            cur_max_dur = cur_durs[i]
                    durations.append(prev_sum_dur + cur_durs[i])

                    
        else:
            line = line.replace(",", "")
            accuracies.append(float(line))

    return accuracies, durations




make_line_plot()
