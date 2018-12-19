import get_data_from_rational_recurrence.load_category_errors_from_hparamopt as load_errors
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections

def marker_map(sparsity, ngram):
    if sparsity == "none":
        if ngram == "4-gram": return "s"
        elif ngram == "3-gram": return "^"
        elif ngram == "2-gram": return "|"
        elif ngram == "1-gram": return "."
        elif ngram == "1-gram,2-gram,3-gram,4-gram": return "*"
    elif sparsity == "rho_entropy": return "+"
    elif sparsity == "rho": return "+"
    elif sparsity == "learned": return "x"
    


def set_colors(num_colors):
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/num_colors) for i in range(num_colors)]
    np.random.shuffle(colors)
    return colors

def main():
    data, categories, worst, best, learned_structures = load_errors.get_data()
    print(data)
    
    train_nums = load_errors.num_training_examples()
    num_colors = len(train_nums)
    colors = set_colors(num_colors)

    for d_out in [24]:#[["24", "6,6,6,6"],["256","64,64,64,64"]]:
        one_plot(worst, best, data, categories, train_nums, d_out=d_out, colors=colors)


# data is a map from d_out -> pattern -> lr -> sparsity -> category, train_num is a map from category -> num training samples
# d_outs is a list of d_out, like ["24", "6,6,6,6"]
# one plot per sum(d_out), one per lr
def one_plot(worst, best, data, categories, train_nums, d_out, colors):
    plt.figure()

    plt.yscale("log")
    plt.ylim((best-0.01, worst))
    best_cat_val = {}
    best_cat_sparsity = {}

    x_categorical = True
    if not x_categorical:
        plt.xscale("log")

    pattern_counter = 0
    for pattern in data[d_out]:
        for sparsity in data[d_out][pattern]:
            if len(data[d_out][pattern][sparsity]) == 0:
                continue


            xs = []
            ys = []
            std_devs = []
            #for category in categories:
            for category in data[d_out][pattern][sparsity]:
                #if pattern == "1-gram" and category == "toys_&_games/":
                #    import pdb; pdb.set_trace()
                
                avg = np.mean(data[d_out][pattern][sparsity][category])
                ys.append(avg)
                std_dev = np.std(data[d_out][pattern][sparsity][category])
                std_devs.append(std_dev)
                xs.append(train_nums[category])

                
                if category not in best_cat_val:
                    best_cat_val[category] = 1
                if ys[-1] < best_cat_val[category] and sparsity != "belowpointnine":
                    best_cat_val[category] = ys[-1]
                    best_cat_sparsity[category] = sparsity + "," + pattern


            x_train_nums = xs
            if x_categorical:
                sorted_data = []
                for i in range(len(xs)):
                    if xs[i] is not None:
                        sorted_data.append([xs[i], ys[i], std_devs[i]])
                sorted_data.sort()
                sorted_data = np.asarray(sorted_data)
                adjustment = (pattern_counter / 7.0 - .5)/2.0
                xs = [point + adjustment for point in range(len(sorted_data[:,0]))]
                
                #xs = np.exp(xs_before_exp)
                ys = sorted_data[:,1]
                std_devs = sorted_data[:,2]
            
            cur_colors = colors[:len(xs)]
            x_train_nums.sort()
            plt.xticks(xs, x_train_nums, rotation=75)
            #plt.scatter(xs, ys, marker=marker_map[sparsity][pattern], c=cur_colors, alpha=0.5)
            plt.errorbar(xs, ys, yerr=std_devs, fmt="o", marker=marker_map(sparsity,pattern), alpha=0.5)

            pattern_counter += 1
    

    num_best_sparsity = collections.Counter()
    for category in best_cat_val:

        num_best_sparsity[best_cat_sparsity[category]] = num_best_sparsity[best_cat_sparsity[category]] + 1

    for item in num_best_sparsity:
        print(item, num_best_sparsity[item])
    save_plot(d_out)

def save_plot(d_out):
    save_loc = "plot_drafts/category_errors/error_by_category_all_cs_dout={}_hparamopt.pdf".format(d_out)
    print("saving to {}...".format(save_loc))
    plt.savefig(save_loc)
                







if __name__ == "__main__":
    main()
