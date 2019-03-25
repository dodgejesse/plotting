import sys
sys.path.append("/home/jessedd/projects/reproducibility/scripts")
import samplemax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



#data_sizes = [200,500,2500,5000,10000]
data_sizes = [32]
#classifiers = ['cnn', 'linear', 'lstm', 'boe']
classifiers = ['cnn', 'linear', 'lstm']

def main():
    for with_replacement in [True, False]:
        for data_name in ["sst_fiveway"]: #["ag_news", "imdb"]:
            data = samplemax.compute_sample_maxes(data_name, with_replacement)
            fig = plt.figure()
            
            for i in range(len(data_sizes)):
                one_plot(data[data_sizes[i]], data_sizes[i], i+1, fig)

            save_plot(data_sizes, classifiers, data_name, with_replacement)
        

def one_plot(data, data_size, plot_counter, fig):


    cur_ax = fig.add_subplot(1,1,plot_counter)
    for classifier in classifiers:
        
        cur_data = data[classifier][0:200]
        line = cur_ax.plot(cur_data, label=classifier)
            
    cur_ax.set_title(data_size)

    #if plot_counter == 5:
    cur_ax.legend(bbox_to_anchor=(1,0,.5,1))
    
    plt.tight_layout()






def save_plot(data_sizes, classifiers, data_name, with_replacement):
    sizes = cat_list(data_sizes)
    cs = cat_list(classifiers)
    
    
    save_loc = "plot_drafts/{}_{}_{}_replacement={}.pdf".format(data_name,sizes, cs, with_replacement)
    print("saving to {}...".format(save_loc))
    plt.savefig(save_loc)


def cat_list(l):
    cat_l = ""
    for cur_l in l:
        cat_l += str(cur_l) + ","
    cat_l = cat_l[0:len(cat_l) - 1]
    return cat_l



if __name__ == "__main__":
    main()
