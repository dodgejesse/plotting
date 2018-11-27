import get_data_from_rational_recurrence.load_category_errors as load_errors
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections

def main():
    data, worst, best = load_errors.get_data()

    train_nums = load_errors.num_training_examples()
    num_colors = len(train_nums)


    for d_outs in [["24", "6,6,6,6"], ["256", "64,64,64,64"]]:
        data_per_approach = get_unique_approaches(data, train_nums, 0.001, d_outs)
        convert_error_to_rank(data_per_approach)
        one_plot(data_per_approach, d_outs[0])

# returns a map from approach name to [xs, ys] list.
def get_unique_approaches(data, train_nums, lr, d_outs):
    x_categorical = True
    data_per_approach = {}
    for d_out in d_outs:
        for pattern in data[d_out]:
            for sparsity in data[d_out][pattern][lr]:
                if len(data[d_out][pattern][lr][sparsity]) == 0 or sparsity == "belowpointnine":
                    continue
                    
                cur_approach_name = sparsity + "," + pattern

                
                if cur_approach_name in data_per_approach:
                    import pdb; pdb.set_trace()
                    print(cur_approach_name)
                

                xs = []
                ys = []
                for category in data[d_out][pattern][lr][sparsity]:
                    ys.append(data[d_out][pattern][lr][sparsity][category])
                    xs.append(train_nums[category])

                if x_categorical:
                    sorted_data = []
                    for i in range(len(xs)):
                        if xs[i] is not None:
                            sorted_data.append([xs[i], ys[i]])
                    sorted_data.sort()
                    sorted_data = np.asarray(sorted_data)
 
                    xs = range(len(sorted_data[:,0]))
                    ys = sorted_data[:,1]

                data_per_approach[cur_approach_name] = [xs, ys]
                    

    return data_per_approach

def convert_error_to_rank(data):
    num_categories = len(data["none,1-gram"][1])
    for i in range(num_categories):
        cur_data = []
        for approach in data:
            cur_data.append([data[approach][1][i], approach])
        cur_data.sort()
        for approach in data:
            for j in range(len(cur_data)):
                if cur_data[j][1] == approach:
                    data[approach][1][i] = j + 1

names = {
    "none,1-gram": "1-gram",
    "none,2-gram": "2-gram",
    "none,3-gram": "3-gram",
    "none,4-gram": "4-gram",
    "none,1-gram,2-gram,3-gram,4-gram": "even mix",
    "learned,4-gram": "learned",
    "rho_entropy,4-gram": "entropy"
}

                    
# d_outs is a list of d_out, like ["24", "6,6,6,6"]
# one plot per sum(d_out), one per lr
def one_plot(data, d_out):
    matplotlib.rcParams.update({'font.size': 6})
    fig = plt.figure()
    
    counter = 0
    for approach in data:
        counter = counter + 1
        cur_ax = fig.add_subplot(len(data), 1, counter)
        cur_ax.bar(data[approach][0], data[approach][1])
        
        cur_ax.set_ylabel(names[approach], rotation=75)
        cur_ax.set_xticklabels([])    


    save_plot(d_out)

def save_plot(d_out):
    save_loc = "plot_drafts/category_ranks/ranks_by_category_dout={}.pdf".format(d_out)
    print("saving to {}...".format(save_loc))
    plt.savefig(save_loc)
                







if __name__ == "__main__":
    main()
