import sys
sys.path.append("/home/jessedd/projects/reproducibility")
import samplemax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



data_sizes = [200,500,2500,5000,10000]
classifiers = ['cnn', 'lr', 'lstm', 'boe']

def main():
    data = samplemax.compute_sample_maxes()
    fig = plt.figure()

    for i in range(len(data_sizes)):
        one_plot(data[data_sizes[i]], data_sizes[i], i+1, fig)

    save_plot(data_sizes, classifiers)

def one_plot(data, data_size, plot_counter, fig):


    cur_ax = fig.add_subplot(3,3,plot_counter)
    for classifier in classifiers:
        
        cur_data = data[classifier]
        line = cur_ax.plot(cur_data, label=classifier)
            
    cur_ax.set_title(data_size)

    if plot_counter == 1:
        cur_ax.legend()
    
    plt.tight_layout()






def save_plot(data_sizes, classifiers):
    sizes = cat_list(data_sizes)
    cs = cat_list(classifiers)
    
    
    save_loc = "plot_drafts/{}_{}.pdf".format(sizes, cs)
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