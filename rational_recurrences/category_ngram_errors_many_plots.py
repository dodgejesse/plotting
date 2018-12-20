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
    elif sparsity == "rho_entropy" or sparsity == "rho" or sparsity == "states": return "+"
    elif sparsity == "learned" or sparsity == "l1-learned": return "x"
    
def main():
    data, categories, worst, best, learned_structures = load_errors.get_data(count_num_params=True)
    formatted_data = format_data(data, categories)
    import pdb; pdb.set_trace()
    for d_out in [24]:#[["24", "6,6,6,6"],["256","64,64,64,64"]]:
        one_plot(worst, best, formatted_data, categories, d_out=d_out)


def format_data(data, categories):
    # ideal format:
    # for category:
    #   four points y=[1-gram, 2-gram, 3-gram, 4-gram],x=[24,2*24,3*24,4*24], yerr=[std_dev_1, std_dev_2, ...], wihch make a line, with associated
    #   one point y=[even_split_avgs], x=[6+12+18+24], yerr=[std_dev]
    #   one point y=[l1-learned_avgs], x=[learned ngrams], yerr[std_dev]
    #   one point y=[states_avgs], x=[learned_ngsams_avgs], yerr[std_dev]
    new_data = {}
    pattern_and_sparsity = set()
    for category in categories:
        new_data[category] = {}
        new_data[category]["baselines"] = {"x" : [0,0,0,0], "y": [0,0,0,0], "xerr": [0,0,0,0], "yerr":[0,0,0,0]}
        new_data[category]["experiments"] = {}
        
        format_one_category(data, category, new_data[category])
        
    return new_data
    
def format_one_category(data, category, new_data):
    for d_out in data:
        for pattern in data[d_out]:
            for sparsity in data[d_out][pattern]:
                cur = np.asarray(data[d_out][pattern][sparsity][category])
                avg = np.mean(cur[:,0])
                std_dev = np.std(cur[:,0])
                avg_num_params = np.mean(cur[:,1])
                std_dev_num_params = np.std(cur[:,1])

                add_one_point(avg, std_dev, avg_num_params, std_dev_num_params, pattern, sparsity, new_data)

                
def add_one_point(avg, std_dev, avg_num_params, std_dev_num_params, pattern, sparsity, new_data):
    if sparsity == "none" and pattern in ["1-gram","2-gram","3-gram","4-gram"]:
        cur_ngram = int(pattern[0]) - 1
        new_data["baselines"]["x"][cur_ngram] = avg_num_params
        new_data["baselines"]["xerr"][cur_ngram] = std_dev_num_params
        new_data["baselines"]["y"][cur_ngram] = avg
        new_data["baselines"]["yerr"][cur_ngram] = std_dev

    new_data["experiments"][sparsity + "~" + pattern] = {"x": [avg_num_params],
                                                         "y":[avg], "xerr":[std_dev_num_params], "yerr":[std_dev]}

def one_plot(worst, best, data, categories, d_out):
    fig = plt.figure()


    #plt.ylim((best-0.01, worst))

    counter = 0
    for category in categories:
        counter += 1 
        cur_ax = fig.add_subplot(2,2,counter)
        cur_ax.set_yscale("log")
        for experiment in data[category]["experiments"]:
            
            cur_data = data[category]["experiments"][experiment]
            sparsity, pattern = experiment.split("~")
            cur_ax.errorbar(cur_data["x"], cur_data["y"],
                            yerr=cur_data["yerr"], xerr=cur_data["xerr"],
                            fmt="o", alpha=0.5, marker=marker_map(sparsity, pattern))
            #plt.errorbar(xs, ys, yerr=std_devs, fmt="o", marker=marker_map(sparsity,pattern), alpha=0.5)

        cur_ax.plot(data[category]["baselines"]["x"], data[category]["baselines"]["y"])

            
        cur_ax.set_title(category)
        cur_ax.set_ylim([.08,.15])
        if counter == 2 or counter == 4:
            cur_ax.set_yticks([])

        if counter == 1 or counter == 2:
            cur_ax.set_xticks([])

    save_plot(d_out, categories)

def save_plot(d_out, categories):
    cat_names = ""
    for cat in categories:
        cat_names += cat[0:3] + ","
    cat_names = cat_names[0:len(cat_names) - 1]
    
    save_loc = "plot_drafts/category_errors/error_by_category_all_cs_dout={}_categories={}_manyplots_rho-entropy.pdf".format(d_out, cat_names)
    print("saving to {}...".format(save_loc))
    plt.savefig(save_loc)
                







if __name__ == "__main__":
    main()
