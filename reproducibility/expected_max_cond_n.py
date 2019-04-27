import sys
sys.path.append("/home/jessedd/projects/reproducibility/scripts")
import samplemax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import numpy as np

max_x = 1
max_x_percent = True
x_axis_time = True
plot_errorbar = False

linestyle = ['-', '--']
data_name = collections.OrderedDict({"sst2_lstm_elmo":"LSTM", "sst2_biattentive_lstm_elmo":"Biattentive"}) #"ner_incomplete":"CoNLL 2003"}


def main():
    data, avg_time = get_data()
    
    fig = plt.figure()

    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0
    for data_size in data_sizes:
        counter += 1

        cur_ax = fig.add_subplot(sqrt_num_plots,sqrt_num_plots,counter)
        cur_ax.set_xscale('log')
        #cur_ax.set_yscale('log')

        experiment_counter = 0
        for cur_data in data_name:
            one_plot(data[cur_data][data_size], avg_time[cur_data][data_size], data_size, cur_ax, data_name[cur_data], experiment_counter)
            experiment_counter += 1

        #one_plot(data[1][data_size], avg_time[1][data_size], data_size, cur_ax, "BiAttentive")
        #cur_ax.get_yaxis().set_visible(False)

    classifiers = get_classifiers(data[list(data.keys())[0]])
    save_plot(data[list(data.keys())[0]].keys(), classifiers, True)
        

def get_data():
    with_replacement = True

    all_data = collections.OrderedDict()
    all_avg_time = collections.OrderedDict()
    
    for cur_name in data_name:
        data, avg_time = samplemax.compute_sample_maxes(cur_name, with_replacement, return_avg_time=True)

        all_data[cur_name] = data
        all_avg_time[cur_name] = avg_time

    return all_data, all_avg_time

    
def one_plot(data, avg_time, data_size, cur_ax, encoder_name, experiment_counter):

    classifiers = list(data.keys())
    classifiers.sort()

    #cur_max_x = get_max_x(avg_time)

    max_first_point = 0
    classifier_counter = 0
    for classifier in classifiers:

        cur_means = data[classifier]['mean']
        cur_vars = data[classifier]['var']
        
        if x_axis_time:
            times = [avg_time[classifier] * (i+1) for i in range(len(cur_means))]
        else:
            times = [i+1 for i in range(len(cur_means))]
        
        cur_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][classifier_counter]
        classifier_counter += 1

        cur_classifier_name = encoder_name + " " + classifier

        #import pdb; pdb.set_trace()
        if plot_errorbar:
            line = cur_ax.errorbar(
                times, cur_means, yerr=cur_vars, label=cur_classifier_name,
                linestyle=linestyle[experiment_counter], color=cur_color)
        else:
            line = cur_ax.plot(times, cur_means,
                               label=cur_classifier_name, linestyle=linestyle[experiment_counter], color=cur_color)
        
        #if max_x == "all":
        #    line = cur_ax.plot(times, cur_data, label=classifier)
        #else:
        #    line = cur_ax.plot(times[0:max_x], cur_data[0:max_x], label=classifier)

        
        #if cur_data[0] > max_first_point:
        #    max_first_point = cur_data[0]


    left, right = cur_ax.get_xlim()
    #cur_ax.set_xlim(left, right*max_x)
    #bottom, top = cur_ax.get_ylim()
    #cur_ax.set_ylim(max_first_point, top)
    #cur_ax.set_ylim(bottom=0.725)

    #cur_ax.set_title("SST2")

    
    #cur_ax.legend(bbox_to_anchor=(1,0,.5,1))
    cur_ax.legend(loc='lower right')
    
    plt.tight_layout()

    
def save_plot(data_sizes, classifiers, with_replacement):
    sizes = cat_list(data_sizes)
    cs = cat_list(classifiers)
    data_names = cat_list(data_name.keys())
    if x_axis_time:
        x_axis_units = "time"
    else:
        x_axis_units = "trials"

    if plot_errorbar:
        errorbar = "/errorbar"
    else:
        errorbar = ""
    save_loc = "plot_drafts/expected_max_dev{}/{}_{}_{}_x={}_maxx={}_replacement={}.pdf".format(
        errorbar, data_names, sizes, cs, x_axis_units, max_x, with_replacement)
    print("saving to {}...".format(save_loc))
    plt.savefig(save_loc)


def cat_list(l):
    cat_l = ""
    l = list(l)
    l.sort()
    for cur_l in l:
        cat_l += str(cur_l).replace(" ","") + ","
    cat_l = cat_l[0:len(cat_l) - 1]
    return cat_l

def get_classifiers(data):
    classifiers = set()
    for data_size in data:
        for c in data[data_size].keys():
            classifiers.add(c)


    cs = list(classifiers)
    cs.sort()
    return classifiers


        
if __name__ == "__main__":
    main()
