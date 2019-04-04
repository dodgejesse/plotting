import sys
import expected_max_cond_n
sys.path.append("/home/jessedd/projects/reproducibility/scripts")
import load_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



#data_sizes = [200,500,2500,5000,10000]
#data_sizes = [10000]
#classifiers = ['cnn', 'linear', 'lstm', 'boe']
#classifiers = ['seq2seq', 'seq2vec']
#classifiers = ['cnn', 'linear', 'lstm']

def main():
    data_name = "sst5" #["ag_news", "imdb", "hatespeech_10k"]:

    unformatted_data = load_data.from_file(data_name, True)[1]
    
    data = format_data(unformatted_data)
    
    import pdb; pdb.set_trace()
    
    classifiers = expected_max_cond_n.get_classifiers(data)
    for data_size in data:

        fig = plt.figure()


        counter = 0
        for classifier in classifiers:
            counter += 1
            cur_ax = fig.add_subplot(2,2,counter)            

            one_plot(data[data_size][classifier], classifier, cur_ax)
        
        save_plot(data_size, classifiers, data_name)

def format_data(u_data):
    data = {}
    for data_size in u_data:
        data[data_size] = {}
        for classifier in u_data[data_size]:
            acc = []
            lr = []
            u_data[data_size][classifier].sort()
            for example in u_data[data_size][classifier]:
                acc.append(example[1])
                lr.append(example[0])
            data[data_size][classifier] = {"acc":acc, "lr":lr}
    return data

def one_plot(data, classifier, cur_ax):

    cur_ax.set_xscale("log")
    line = cur_ax.scatter(data["lr"], data["acc"])
            
    cur_ax.set_title(classifier)

    
    #cur_ax.legend(bbox_to_anchor=(1,0,.5,1))
    
    plt.tight_layout()



def save_plot(data_size, classifiers, data_name):
    cs = expected_max_cond_n.cat_list(classifiers)
    
    save_loc = "plot_drafts/accuracy_lr/accuracy_lr_plot_{}_{}_{}.pdf".format(data_name,data_size, cs)
    print("saving to {}...".format(save_loc))
    plt.savefig(save_loc)


if __name__ == "__main__":
    main()
