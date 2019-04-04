import sys
sys.path.append("/home/jessedd/projects/reproducibility/scripts")
import samplemax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

max_n = "all"

def main():
    with_replacement = True
    data_name = "sst5" #["ag_news", "imdb", "sst_fiveway_glove", "hatespeech_10k"]:
    
    data = samplemax.compute_sample_maxes(data_name, with_replacement)
    fig = plt.figure()

    data_sizes = list(data.keys())
    data_sizes.sort()
    
    counter = 0
    for data_size in data_sizes:
        counter += 1
        cur_ax = fig.add_subplot(2,2,counter)
        one_plot(data[data_size], data_size, cur_ax)

    classifiers = get_classifiers(data)
    save_plot(data.keys(), classifiers, data_name, with_replacement)
        

def one_plot(data, data_size, cur_ax):

    classifiers = list(data.keys())
    classifiers.sort()
    for classifier in classifiers:
        
        cur_data = data[classifier]
        if max_n == "all":
            line = cur_ax.plot(cur_data, label=classifier)
        else:
            line = cur_ax.plot(cur_data[0:max_n], label=classifier)
            
    cur_ax.set_title(data_size)

    
    cur_ax.legend(bbox_to_anchor=(1,0,.5,1))
    
    plt.tight_layout()


def save_plot(data_sizes, classifiers, data_name, with_replacement):
    sizes = cat_list(data_sizes)
    cs = cat_list(classifiers)
    
    save_loc = "plot_drafts/expected_max_dev/{}_{}_{}_maxn={}_replacement={}.pdf".format(data_name,sizes, cs, max_n,
                                                                        with_replacement)
    print("saving to {}...".format(save_loc))
    plt.savefig(save_loc)


def cat_list(l):
    cat_l = ""
    l = list(l)
    l.sort()
    for cur_l in l:
        cat_l += str(cur_l) + ","
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
