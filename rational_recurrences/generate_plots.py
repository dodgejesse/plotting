import sys
sys.path.append('/home/jessedd/projects/rational-recurrences/classification')
from experiment_params import ExperimentParams

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import os        
import get_data_from_rational_recurrence.load_groups_norms as load_norms


num_groups = 256
num_plots_per_axis = int(np.sqrt(num_groups))

def main():
    args = ExperimentParams()
    norms = load_norms.load_from_file(args)
    assert(len(norms) % num_groups == 0)

    data, largest_point, smallest_point = arrange_data(norms)
    sorted_data = sort_data_by_decrease(data)



    many_plots(sorted_data, largest_point, smallest_point, args)
    

def arrange_data(norms):

    data = []
    for i in range(len(norms)):
        if len(data) < num_groups:
            data.append([norms[i]])
        else:
            data[i % num_groups].append(norms[i])
    largest_point = max(norms)
    smallest_point = min(norms)
    
    return data, largest_point, smallest_point

def sort_data_by_decrease(data):
    return sorted(data, key=lambda datum: datum[0] - datum[-1])

def many_plots(data, largest_point, smallest_point, args):
    matplotlib.rcParams.update({'font.size':1})
    fig = plt.figure()

    for i in range(len(data)):
        counter = i + 1
        cur_ax = fig.add_subplot(num_plots_per_axis,num_plots_per_axis,counter)
        cur_ax.set_ylim([smallest_point, largest_point])
        cur_ax.get_xaxis().set_visible(False)
        cur_ax.get_yaxis().set_visible(False)
        
        cur_ax.plot(data[i])

    
    plt.tight_layout()
    plot_name = "plot_drafts/norms/{}.pdf".format(args.file_name())
    print("saving to {}".format(plot_name))
    plt.savefig(plot_name,bbox_inches='tight')
    
    


    
if __name__ == "__main__":
    main()
