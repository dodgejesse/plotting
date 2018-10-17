import pickle, pprint
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


def get_data():
    
    data = pickle.load(open("/home/jessedd/projects/rational-recurrences/classification/logging/lr_regstr_validerr_testerr.p", 'rb'))
    lr = []
    reg_strength = []
    valid_err = []
    test_err = []
    
    for point in data:
        lr.append(point['lr'])
        reg_strength.append(point['reg_strength'])
        valid_err.append(point['valid_err'])
        test_err.append(point['test_err'])
    
    return np.asarray([lr, reg_strength, valid_err, test_err])

def get_one_plot_data(data, n, n_bins):
    # need one vector for x, one vector for y
    first_dim = []
    second_dim = []
    for snum in data:

        first_dim += [data[snum][n][k][0] for k in range(len(data[snum][n]))]
        second_dim += [data[snum][n][k][1] for k in range(len(data[snum][n]))]

    heatmap, xedge, yedge = np.histogram2d(first_dim, second_dim, bins=(n_bins,n_bins))
    
                   
    return heatmap, [0,1,0,1]
    

def make_avgs(data, bins):

    x_bound = [0.5, 0.0001]
    y_bound = [0.5, 0.00001]
    
    x_bins = np.logspace(np.log10(x_bound[0]),np.log10(x_bound[1]), num=bins)
    y_bins = np.logspace(np.log10(y_bound[0]),np.log10(y_bound[1]), num=bins)


    avgs = np.zeros((bins-1,bins-1))
    for i in range(bins-1):
        for j in range(bins-1):
            xs = np.logical_and(data[0] < x_bins[i], data[0] > x_bins[i+1])
            ys = np.logical_and(data[1] < y_bins[j], data[1] > y_bins[j+1])
            points = np.logical_and(xs,ys)
            avgs[i][j] = np.average(data[2][points])

            
    return avgs, x_bins, y_bins
    #print(avgs)
            



def plot_heat(data, x_bins, y_bins):
    matplotlib.rcParams.update({'font.size':7})
    fig = plt.figure(figsize=(10,10))
    #fig.suptitle("Heatmap of 5000 samples from sobol sequence with uniform noise.\n{}x{} bins,"\
    #             "plotted with bicubic interpoliation.".format(bins,bins), fontsize=8)
    cur_ax = fig.add_subplot(1,1,1)

    
    cur_ax.set_xscale("log")
    cur_ax.set_yscale("log")

        
    import pdb; pdb.set_trace()

    # DOESN'T WORK!
    cur_ax.imshow(data, interpolation='bicubic', cmap='BuPu')#,vmin=min_value,vmax=max_value)
    #cur_ax.set_title('n={}'.format(n))




    plt.tight_layout()

    out_fname = "/home/jessedd/projects/plotting/rational_recurrences/plot_drafts/heatmaps/heatmap_avgs.pdf"
    plt.savefig(out_fname)
    print("saving to {}".format(out_fname))



# this function is so you can see where your points are, for a starting point for the heatmap
def scatter(data):
    plt.scatter(data[0], data[1], c=100*data[2], cmap=cm.coolwarm)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([min(data[0]),max(data[0])])
    plt.ylim([min(data[1]),max(data[1])])
    plt.colorbar()


    plt.savefig("/home/jessedd/projects/plotting/rational_recurrences/plot_drafts/heatmaps/scatter_of_heatmap_data.pdf")


def print_data(data):
    #import pdb; pdb.set_trace()
    #good_vals = np.logical_and(np.logical_and(data[0] > 0.1, data[1] < 0.01), data[1] > 0.001)
    okay_vals = np.logical_and(data[0] > 0.3, data[1] < 0.01)
    #better_vals = np.logical_and(np.logical_and(data[0] > 0.3, data[1] < 0.004), data[1] > 0.003)
    import pdb; pdb.set_trace()
    
    scatter(data[:,okay_vals])
    
def main():
        
    data = get_data()
    if len(data) == 0:
        print('did not read data, probably filename incorrect')
        exit
    bins = 10

    #scatter(data)
    print_data(data)
    avgs, x_bins, y_bins = make_avgs(data, bins)
    #plot_heat(avgs, x_bins, y_bins)

    

if __name__ == "__main__":
    main()
