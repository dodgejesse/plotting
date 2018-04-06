import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import sample_histograms
import line_plot
import os
from collections import OrderedDict



def get_search_algos():
    search_algos = OrderedDict()
    search_algos['mixed_dpp_rbf'] = 'k-DPP-RBF'
    search_algos['bayes_opt'] = 'BO-TPE'
    search_algos['random'] = 'Uniform'
    #search_algos['spearmint_seq'] = 'spearmint_seq'
    #search_algos['spearmint_kover2'] = 'spearmint_kover2'
    search_algos['sobol_noise'] = 'Sobol'
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
        things = get_avg_and_std_dev('/home/jessedd/projects/plotting/icml_2018/data/stanford_sentiment_binary,nmodels=1,mdl_tpe=cnn,srch_tpe={},spce={},iters={}.txt'.format(algo, space, iters))
        all_info[algo] = make_into_dict(*things)

        # DEBUG PRINTING
        best_acc = round(100*things[0][19],3)
        ci_range = round(100*things[1][19],3)
        best_acc_ci = [best_acc - ci_range, best_acc + ci_range]
        print("{}, in {}. num_iters={}, {}".format(best_acc, str(best_acc_ci), all_info[algo]['num_iters'], algo))



    fig = plt.figure()
    cur_ax = fig.add_subplot(1,1,1)
    matplotlib.rcParams.update({'font.size': 11})
    
    line_plot.add_line_plot(all_info, search_algos, space, cur_ax)
    plt.tight_layout()
    save_loc = get_save_loc(search_algos)
    print("printing in {}".format(save_loc))
    plt.savefig(save_loc,bbox_inches='tight')

def get_save_loc(search_algos):
    save_loc = "plot_drafts/"
    for s_t in search_algos:
        save_loc += str(s_t) + "_"
    save_loc = save_loc[:-1]
    save_loc += ",space=" + get_space()
    save_loc += ",iters=" + str(get_iters())
    save_loc += '.pdf'
    return save_loc


def get_avg_and_std_dev(file_loc):
    print file_loc
    with open(file_loc) as f:
        lines = f.readlines()
        examples = extract_examples(lines)
        avg_best, avg_best_ci = compute_avg_best_so_far(examples)
        
        return avg_best, avg_best_ci, len(examples), len(examples[0])


def extract_examples(lines):
    examples = [[]]
    counter = 0
    for line in lines:
        line = line.strip()
        if line == "":
            examples.append([])
            counter = 0
        else:
            counter += 1
            #DEBUG THIS IS FOR GETTING MORE DATA
            if counter > 20:
                continue
            example = line.split(',')
            example[len(example)-2] += "," + example[len(example)-1]
            del example[-1]
            example[0] = float(example[0])
            example[1] = float(example[1])
            example[2] = float(example[2])
            extract_single_example(example)
            examples[len(examples)-1].append(example)
            

    
    examples = [x for x in examples if x != []]
    return examples


def extract_single_example(example):
    example[4] = example[4].replace('7e-05', '0.00007')
    example[4] = example[4].replace('5e-05', '0.00005')
    hparams_string = example[4].split(' ')
    hparams = {}
    cur_hparam  = ''
    for hp in hparams_string:
        if hp == '':
            continue
        elif ':' in hp:
            cur_hparam = hp.split(':')[0]
            hparams[cur_hparam] = ''
        else :
            hparams[cur_hparam] += hp
    for hparam in hparams:
        if 'learning_rate' in hparam:
            hparams[hparam] = float(hparams[hparam])
        elif 'reg_strength' in hparam:
            hparams[hparam] = round(np.exp(float(hparams[hparam])),5)
        elif 'dropout' in hparam:
            hparams[hparam] = float(hparams[hparam])
        elif 'filters' in hparam:
            hparams[hparam] = int(hparams[hparam])
    example[4] = hparams

def compute_avg_best_so_far(examples):
    best_so_far = []
    for sample_of_k in examples:
        for i in range(len(sample_of_k)):
            if len(best_so_far) <= i:
                best_so_far.append([])
            best_so_far[i].append(float(sample_of_k[i][2]))
    avg_best = []
    avg_best_ci = []
    for i in range(len(best_so_far)):
        #avg_best_ci.append(1.96*np.std(np.asarray(best_so_far[i]))/np.sqrt(len(examples)))
        avg_best_ci.append(1.645*np.std(np.asarray(best_so_far[i]))/np.sqrt(len(examples)))
        avg_best.append(np.average(np.asarray(best_so_far[i])))

    print len(examples)

    return avg_best, avg_best_ci

def make_into_dict(avg_best, avg_best_ci, num_samples, num_iters):
    info = {}
    info['avg_best'] = avg_best
    info['ci'] = avg_best_ci
    info['num_samples'] = num_samples
    info['num_iters'] = num_iters
    return info


make_line_plot()
