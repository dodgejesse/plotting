import get_data_from_rational_recurrence.load_category_errors as load_errors
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections

marker_map = {
    "none": {
        "4-gram": "s",
        "3-gram": "^",
        "2-gram": "|",
        "1-gram": ".",
        "1-gram,2-gram,3-gram,4-gram": "*"
    },
    "rho_entropy": {
        "4-gram": "+"
    },
    "learned":{
        "4-gram": "x"
    }
}
    


def set_colors(num_colors):
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/num_colors) for i in range(num_colors)]
    np.random.shuffle(colors)
    return colors

def main():
    data, worst, best = load_errors.get_data()
    print(data)
    train_nums = load_errors.num_training_examples()
    num_colors = len(train_nums)
    colors = set_colors(num_colors)


    for d_outs in [["24", "6,6,6,6"],["256","64,64,64,64"]]:
        for lr in [0.001]:
            one_plot(worst, best, data, train_nums, d_outs=d_outs, lr=lr, colors=colors)



# data is a map from d_out -> pattern -> lr -> sparsity -> category, train_num is a map from category -> num training samples
# d_outs is a list of d_out, like ["24", "6,6,6,6"]
# one plot per sum(d_out), one per lr
def one_plot(worst, best, data, train_nums, d_outs, lr, colors):
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim((best-0.01, worst))
    best_cat_val = {}
    best_cat_sparsity = {}

    x_categorical = True
    for d_out in d_outs:
        for pattern in data[d_out]:
            for sparsity in data[d_out][pattern][lr]:
                if len(data[d_out][pattern][lr][sparsity]) == 0:
                    continue

                
                xs = []
                ys = []
                for category in data[d_out][pattern][lr][sparsity]:
                    ys.append(data[d_out][pattern][lr][sparsity][category])
                    xs.append(train_nums[category])

                    
                    if category not in best_cat_val:
                        best_cat_val[category] = 1
                    if ys[-1] < best_cat_val[category] and sparsity != "belowpointnine":
                        best_cat_val[category] = ys[-1]
                        best_cat_sparsity[category] = sparsity + "," + pattern


                    
                        
                if x_categorical:
                    sorted_data = []
                    for i in range(len(xs)):
                        if xs[i] is not None:
                            sorted_data.append([xs[i], ys[i]])
                    sorted_data.sort()
                    sorted_data = np.asarray(sorted_data)
 
                    xs = np.exp(range(len(sorted_data[:,0])))
                    ys = sorted_data[:,1]
            
                cur_colors = colors[:len(xs)]
                if sparsity == "belowpointnine":
                    #print_entropy_on_plot(xs, ys, colors)
                    plt.xticks(xs, ys, rotation=75)
                else:
                    plt.scatter(xs, ys, marker=marker_map[sparsity][pattern], c=cur_colors, alpha=0.5)


    num_best_sparsity = collections.Counter()
    for category in best_cat_val:
        num_best_sparsity[best_cat_sparsity[category]] = num_best_sparsity[best_cat_sparsity[category]] + 1

    for item in num_best_sparsity:
        print(item, num_best_sparsity[item])
    save_plot(d_outs,lr)

def save_plot(d_outs, lr):
    d_outs_string = ""
    for d_out in d_outs:
        d_outs_string += d_out + ","
    d_outs_string = d_outs_string[:len(d_outs_string)-1]
    save_loc = "plot_drafts/category_errors/errors_by_category_dout={}_lr={}.pdf".format(d_outs_string, lr)
    print("saving to {}...".format(save_loc))
    plt.savefig(save_loc)
                







if __name__ == "__main__":
    main()
