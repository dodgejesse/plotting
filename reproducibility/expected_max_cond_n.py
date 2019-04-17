import sys
sys.path.append("/home/jessedd/projects/reproducibility/scripts")
import samplemax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

max_x = 1
max_x_percent=True

def main():
    data, avg_time = get_data()
    
    fig = plt.figure()

    data_sizes = list(data[0].keys())
    data_sizes.sort()
    
    counter = 0
    for data_size in data_sizes:
        counter += 1
        cur_ax = fig.add_subplot(1,1,counter)
        cur_ax.set_xscale('log')
        #cur_ax.set_yscale('log')

        one_plot(data[0][data_size], avg_time[0][data_size], data_size, cur_ax, "LSTM")
        one_plot(data[1][data_size], avg_time[1][data_size], data_size, cur_ax, "BiAttentive")
        #cur_ax.get_yaxis().set_visible(False)
        
    classifiers = get_classifiers(data[0])
    save_plot(data[0].keys(), classifiers, "sst2_biattentive_and_lstm_elmo_transformer", True)
        

def get_data():
    with_replacement = True
    data_name = ["sst2_elmo_transformer", "sst2_biattentive_elmo_transformer"] #["ag_news", "imdb", "sst_fiveway_glove", "hatespeech_10k"]:

    all_data = []
    all_avg_time = []
    
    for cur_name in data_name:
        data, avg_time = samplemax.compute_sample_maxes(cur_name, with_replacement, return_avg_time=True)
        
        all_data.append(data)
        all_avg_time.append(avg_time)

    return all_data, all_avg_time

    
def one_plot(data, avg_time, data_size, cur_ax, encoder_name):

    classifiers = list(data.keys())
    classifiers.sort()

    #cur_max_x = get_max_x(avg_time)

    max_first_point = 0
    classifier_counter = 0
    for classifier in classifiers:
        
        times = [avg_time[classifier] * (i+1) for i in range(len(data[classifier]))]
        #import pdb; pdb.set_trace()
        
        cur_data = data[classifier]

        cur_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][classifier_counter]
        classifier_counter += 1

        cur_classifier_name = encoder_name + " " + classifier

        if "LSTM" == encoder_name:
            cur_linestyle = "-"
        else:
            cur_linestyle = "--"
        
        line = cur_ax.plot(times, cur_data, label=cur_classifier_name, linestyle=cur_linestyle, color=cur_color)
        #if max_x == "all":
        #    line = cur_ax.plot(times, cur_data, label=classifier)
        #else:
        #    line = cur_ax.plot(times[0:max_x], cur_data[0:max_x], label=classifier)

        
        if cur_data[0] > max_first_point:
            max_first_point = cur_data[0]


    left, right = cur_ax.get_xlim()
    #cur_ax.set_xlim(left, right*max_x)
    #bottom, top = cur_ax.get_ylim()
    #cur_ax.set_ylim(max_first_point, top)
    #cur_ax.set_ylim(0.88, top)

    cur_ax.set_title(data_size)

    
    #cur_ax.legend(bbox_to_anchor=(1,0,.5,1))
    cur_ax.legend()
    
    plt.tight_layout()

    
def save_plot(data_sizes, classifiers, data_name, with_replacement):
    sizes = cat_list(data_sizes)
    cs = cat_list(classifiers)
    
    save_loc = "plot_drafts/expected_max_dev/{}_{}_{}_x=time_maxx={}_replacement={}.pdf".format(data_name,sizes, cs, max_x,
                                                                        with_replacement)
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
