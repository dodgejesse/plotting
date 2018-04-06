import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def add_line_plot(data, order, space, cur_ax, ylim_lower=None, ylim_upper=None, last_few=None):
    
    x = range(1,len(data[order.items()[0][0]]['avg_best'])+1)
    x = [xi for xi in x if xi % 5 == 0]
    print x
    diff_x = {}
    counter = 0
    width = .3/4
    for algo in order:
        counter += 1
        diff_x[algo] = [i + (width * counter) - 0.15 for i in range(len(x))]

    for thing in diff_x:
        print diff_x[thing]

        
    if ylim_lower is not None:
        cur_ax.set_ylim([ylim_lower,ylim_upper])
    #else:
    #    cur_ax.set_ylim([.545,.83])

    for algo in order:
        if algo in data:
            #cur_ax.plot(diff_x[algo], data[algo]['avg_best'], dashes=data[algo]['dash'], linewidth=1)
            #cur_ax.plot(diff_x[algo], data[algo]['avg_best'], linewidth=1)

            cur_y = [data[algo]['avg_best'][i-1] for i in x]
            cur_ci = [data[algo]['ci'][i-1] for i in x]
            #cur_ax.errorbar(diff_x[algo], data[algo]['avg_best'], yerr=data[algo]['ci'], fmt='o')
            cur_ax.errorbar(diff_x[algo], cur_y, yerr=cur_ci, fmt='o')

    cur_ax.locator_params(nbins=6)
    #print cur_ax.get_yticks()
    #cur_ax.set_xticks(x)
    cur_ax.set_xticklabels(['1','k=5','k=10','k=15','k=20'])
    cur_ax.tick_params(labelsize=11)
    for thing in cur_ax.get_xmajorticklabels():
        print thing

    if space == 'reg_bad_lr':
        cur_ax.set_title("Hard learning rate".format(space))
    elif space == 'reg_half_bad_lr':
        cur_ax.set_title("Medium learning rate".format(space))
    elif space == 'reg':
        cur_ax.set_title("Easy learning rate".format(space))
    
    cur_ax.legend([order[name] for name in order if name in data], loc='lower right')

    cur_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


