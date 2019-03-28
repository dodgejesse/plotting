import sys
sys.path.append("/home/jessedd/projects/reproducibility/scripts")
import load_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



#data_sizes = [200,500,2500,5000,10000]
#data_sizes = [10000]
#classifiers = ['cnn', 'linear', 'lstm', 'boe']
#classifiers = ['cnn', 'linear', 'lstm', 'boe']
classifiers = ['seq2seq', 'seq2vec']
#classifiers = ['cnn', 'linear', 'lstm']

def main():
    for data_name in ["battle_year"]: #["ag_news", "imdb", "hatespeech_10k"]:
        unformatted_data = load_data.from_file(data_name, True)[1]
        
        data = format_data(unformatted_data)

        import pdb; pdb.set_trace()
        
        fig = plt.figure()
        
        for i in range(len(classifiers)):
            one_plot(data[classifiers[i]], classifiers[i], i+1, fig)

        save_plot([""], classifiers, data_name)

def format_data(u_data):
    data = {}
    for classifier in u_data:
        acc = []
        lr = []
        u_data[classifier].sort()
        for example in u_data[classifier]:
            acc.append(example[1])
            lr.append(example[0])
        data[classifier] = {"acc":acc, "lr":lr}
    return data

def one_plot(data, classifier, plot_counter, fig):


    cur_ax = fig.add_subplot(2,2,plot_counter)
    cur_ax.set_xscale("log")
    line = cur_ax.scatter(data["lr"], data["acc"])
            
    cur_ax.set_title(classifier)

    
    #cur_ax.legend(bbox_to_anchor=(1,0,.5,1))
    
    plt.tight_layout()



def save_plot(data_sizes, classifiers, data_name):
    sizes = cat_list(data_sizes)
    cs = cat_list(classifiers)
    
    
    save_loc = "plot_drafts/accuracy_lr_plot_{}_{}_{}.pdf".format(data_name,sizes, cs)
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
