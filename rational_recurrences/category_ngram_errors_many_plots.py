#import get_data_from_rational_recurrence.load_category_errors_from_hparamopt as load_errors
import get_data_from_rational_recurrence.load_errors_from_sheet as load_errors
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
    elif sparsity == "learned" or sparsity == "l1-learned": return "x"
    elif sparsity == "rho_entropy" or sparsity == "rho" or sparsity == "states": return "+"

def experiment_marker_map(experiment):
    if experiment == "baselines": return "o"
    elif experiment == "learned": return "s"
    elif experiment == "regularized": return "D"

def experiment_color_map(experiment):
    if experiment == "baselines": return "green"
    elif experiment == "learned": return "blue"
    elif experiment == "regularized": return "orange"

def get_name(name):
    if "orig" in name:
        return "original_mix"
    elif "dvd" in name:
        return "dvd"
    elif "kitche" in name:
        return "kitchen"
    elif "book" in name:
        return "books"
    elif "BER" in name:
        return "kitchen BERT"
    
    
def main():
    data, categories, worst, best, learned_structures = load_errors.get_data(path = "./get_data_from_rational_recurrence/data/data_test.csv")
    formatted_data = format_data(data, categories)
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
        for experiment in ["baselines", "learned", "baseline_mixed"]:#, "regularized"]:
            new_data[category][experiment] = {"x" : [0,0,0,0], "y": [0,0,0,0], "xerr": [0,0,0,0], "yerr":[0,0,0,0]}
        new_data[category]["experiments"] = {}
        
        format_one_category(data, category, new_data[category])
        
    return new_data
    
def format_one_category(data, category, new_data):
    for d_out in data:
        for pattern in data[d_out]:
            for sparsity in data[d_out][pattern]:
                if category not in data[d_out][pattern][sparsity]:
                    continue
                cur = np.asarray(data[d_out][pattern][sparsity][category])
                avg = np.mean(cur[:,0])
                std_dev = np.std(cur[:,0])
                avg_num_params = np.mean(cur[:,1])
                std_dev_num_params = np.std(cur[:,1])

                add_one_point(avg, std_dev, avg_num_params, std_dev_num_params, pattern, sparsity, new_data)

                
def add_one_point(avg, std_dev, avg_num_params, std_dev_num_params, pattern, sparsity, new_data):
    if sparsity == "none" and pattern in ["1-gram","2-gram","3-gram","4-gram"]:
        cur_ngram = int(pattern[0]) - 1
        add_one_point_helper(avg, std_dev, avg_num_params, std_dev_num_params, new_data["baselines"], cur_ngram)
    elif "learned_goalparams" in sparsity:
        position = int(int(sparsity[-2])/2) - 1
        add_one_point_helper(avg, std_dev, avg_num_params, std_dev_num_params, new_data["learned"], position)
    elif sparsity == "none" and pattern == "1-gram,2-gram,3-gram,4-gram":
        new_data['baseline_mixed'] = {"x": [avg_num_params], "xerr": [0], "y": [avg], "yerr": [std_dev]}

    # this elif shouldn't happen now, since we're not plotting this
    elif "states_goalparams" in sparsity:
        position = int(int(sparsity[-2])/2) - 1
        add_one_point_helper(avg, std_dev, avg_num_params, std_dev_num_params, new_data["regularized"], position)

    new_data["experiments"][sparsity + "~" + pattern] = {"x": [avg_num_params],
                                                         "y":[avg], "xerr":[std_dev_num_params], "yerr":[std_dev]}

def add_one_point_helper(avg, std_dev, avg_num_params, std_dev_num_params, new_data, position):
    new_data["x"][position] = avg_num_params
    new_data["xerr"][position] = std_dev_num_params
    new_data["y"][position] = avg
    new_data["yerr"][position] = std_dev
    
    
def one_plot(worst, best, data, categories, d_out):

    fig = plt.figure(figsize=(12,3))

    #plt.locator_params(axis = 'y', nbins = 4)
    #plt.ylim((best-0.01, worst))
    #subplot_dim = int(np.ceil(np.sqrt(len(categories))))
    subplot_dim = 4
    counter = 0
    for category in categories:

        cur_worst = 100
        cur_best = 0
        
        counter += 1 
        cur_ax = fig.add_subplot(1,subplot_dim,counter)
        #cur_ax.set_yscale("log")
        
        plot_points = False
        if plot_points:
            for experiment in data[category]["experiments"]:
            
                cur_data = data[category]["experiments"][experiment]
                sparsity, pattern = experiment.split("~")
                cur_ax.errorbar(cur_data["x"], cur_data["y"],
                                yerr=cur_data["yerr"], xerr=cur_data["xerr"],
                                fmt="o", alpha=0.5, marker=experiment_marker_map(sparsity, pattern))
                #plt.errorbar(xs, ys, yerr=std_devs, fmt="o", marker=marker_map(sparsity,pattern), alpha=0.5)
                
            #import pdb; pdb.set_trace()
        
        for experiment in ["baselines", "learned", "baseline_mixed"]: #, "regularized"]:
            #if category == "original_mix/":
            #    import pdb; pdb.set_trace()
            cur_data = data[category][experiment]
            color = experiment_color_map(experiment)
            marker = experiment_marker_map(experiment)
            #cur_ax.plot(cur_data["x"], cur_data["y"],
            #            ls='-', marker=marker, color=color)

            if min(cur_data["y"]) < cur_worst:
                cur_worst = min(cur_data["y"])
            if max(cur_data["y"]) > cur_best:
                cur_best = max(cur_data["y"])

            if experiment == "baseline_mixed":
                #color = line[0].get_color()
                color = None
            else:
                color = None
            line = cur_ax.errorbar(cur_data["x"], cur_data["y"],
                                   yerr=cur_data["yerr"], xerr=cur_data["xerr"], marker=marker, alpha=0.75, markersize=0.5,
                                   color=color)

                
                            #fmt="o", alpha=0.5, marker=marker, color=color)
            #import pdb; pdb.set_trace()            

            
        cur_ax.set_title(get_name(category))
        #cur_ax.locator_params(axis='y', tight=True, )
        cur_ax.set_ylim([cur_worst-1, cur_best+1])
        if "BERT" in category:
            #cur_ax.set_xlim([10, 100])
            cur_ax.set_xticks([12, 24, 36, 48])
            #    cur_ax.set_yticks([9,91,92,93])

        else:
            #cur_ax.set_yticks(np.round(np.linspace(cur_worst,cur_best,4)))
            #cur_ax.set_yticklabels(np.round(np.linspace(cur_worst,cur_best,4)))
            cur_ax.set_xlim([10, 100])
            cur_ax.set_xticks([24, 48, 72, 96])
        
        #cur_ax.ticklabel_format(style='plain')

        #cur_ax.set_yticks([.88,.9,.91])
        #cur_ax.set_yticklabels([.9])
        #cur_ax.locator_params(axis = 'y', nbins = 4)
        #cur_ax.set_ylim([.08,.15])
        #if counter == 2 or counter == 4:
        #    cur_ax.set_yticklabels([])

        #if counter == 1 or counter == 2:
        #    cur_ax.set_xticks([])

        if counter == 1:
            cur_ax.set_ylabel("Classification Accuracy")
            cur_ax.set_xlabel("Number of Transitions")

    #fig.text(0.5, 0.04, 'Total number of WFSA states', ha='center')
    #fig.text(0.01, 0.5, 'Error on four datasets (lower is better)', va='center', rotation='vertical')
    plt.tight_layout()
    save_plot(d_out, categories)
    

def save_plot(d_out, categories):
    cat_names = ""
    for cat in categories:
        cat_names += cat[0:3] + ","
    cat_names = cat_names[0:len(cat_names) - 1]
    
    save_loc = "plot_drafts/category_errors/all_baselines_accuracy_{}.pdf".format(cat_names)
    print("saving to {}...".format(save_loc))
    plt.savefig(save_loc)







if __name__ == "__main__":
    main()
