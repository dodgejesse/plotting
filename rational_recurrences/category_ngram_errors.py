import get_data_from_rational_recurrence.load_category_errors as load_errors
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

marker_map = {
    "4-gram": "s",
    "3-gram": "^",
    "2-gram": "|",
    "1-gram": ".",
    "4-gram,3-gram,2-gram,1-gram": "*"
}


def set_colors(num_colors):
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/num_colors) for i in range(num_colors)]
    np.random.shuffle(colors)
    return colors

def main():
    data = load_errors.get_data()
    print(data)

    train_nums = load_errors.num_training_examples()
    plot_data(data, train_nums)


# data is a map from d_out -> pattern -> category, train_num is a map from category -> num training samples
def plot_by_d_out(data, train_nums, d_out):
    x_categorical = True
    num_colors = len(data["24"]["4-gram"])
    colors = set_colors(num_colors)
    for d_out in ["24", "6,6,6,6"]:



        for pattern in data[d_out]:
            if len(data[d_out][pattern]) == 0:
                continue
            xs = []
            ys = []
            for category in data[d_out][pattern]:
                ys.append(data[d_out][pattern][category])
                xs.append(train_nums[category])

            
            if x_categorical:
                sorted_data = []
                for i in range(len(xs)):
                    if xs[i] is not None:
                        sorted_data.append([xs[i], ys[i]])
                sorted_data.sort()
                sorted_data = np.asarray(sorted_data)
                #import pdb; pdb.set_trace()
                xs = np.exp(range(len(sorted_data[:,0])))
                ys = sorted_data[:,1]
            
            cur_colors = colors[:len(xs)]
            plt.scatter(xs, ys, marker=marker_map[pattern], c=cur_colors, alpha=0.5)


            
    
    plt.xscale("log")
    plt.yscale("log")

    plt.savefig("plot_drafts/category_errors/plot.pdf")
                







if __name__ == "__main__":
    main()
