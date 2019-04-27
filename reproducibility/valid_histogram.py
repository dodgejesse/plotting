import sys
sys.path.append("/home/jessedd/projects/reproducibility/scripts")
import load_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import seaborn as sns


data_name = collections.OrderedDict({"sst2_biattentive_lstm_elmo":"Biattentive"})


def make_hist():
    data = get_data()
    fig = plt.figure()


    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0
    for data_size in data_sizes:
        counter += 1

        cur_ax = fig.add_subplot(sqrt_num_plots,sqrt_num_plots,counter)

        experiment_counter = 0
        for cur_data in data_name:
            one_hist(data[cur_data][data_size], data_size, cur_ax,
                     data_name[cur_data], experiment_counter)
            experiment_counter += 1

    save_plot()

def one_hist(data, data_size, cur_ax, encoder_name, experiment_counter):
    classifiers = list(data.keys())
    classifiers.sort()

    all_data = []
    for classifier in classifiers:
        all_data.append(data[classifier])
        cur_classifier_name = encoder_name + " " + classifier
        sns.kdeplot(np.array(data[classifier]), bw=0.01, label=cur_classifier_name)

        #cur_ax.hist(data[classifier], 100, label=cur_classifier_name)

        
    #cur_ax.hist(all_data, 25, histtype='step')


    
    #cur_ax.set_xlim(0,1.25)
    #cur_ax.legend(loc="upper right")


def get_data():
    data = collections.OrderedDict()
    for cur_name in data_name:
        data[cur_name] = load_data.from_file(cur_name, return_avg_time=False)

    return data

def save_plot():
    save_loc = "plot_drafts/valid_histograms/first_hist.pdf"
    print("saving in {}".format(save_loc))
    plt.savefig(save_loc)

if __name__ == "__main__":
    make_hist()
