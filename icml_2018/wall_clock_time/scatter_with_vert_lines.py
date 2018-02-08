import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def add_scatter(data, order, space, cur_ax, ylim_lower=None, ylim_upper=None):
        

    for algo in order:
        if algo in data:
            
            cur_ax.scatter(data[algo]['durs'],data[algo]['accs'], marker='.', s=5, color=order[algo]['color'])

    min_acc = 1
    max_acc = 0
    max_dur = 0
    for algo in order:
        if algo in data:
            if min_acc > min(data[algo]['accs']):
                min_acc = min(data[algo]['accs'])
            if max_acc < max(data[algo]['accs']):
                max_acc = max(data[algo]['accs'])
            if max_dur < max(data[algo]['durs']):
                max_dur = max(data[algo]['durs'])

    max_acc += 0.0025
    min_acc -= 0.0025
    for algo in order:
        if algo in data:
            cur_avg_dur = [data[algo]['avg_dur_to_best'],data[algo]['avg_dur_to_best']]
            min_and_max_acc = [min_acc, max_acc]
            cur_ax.plot(cur_avg_dur, min_and_max_acc, color=order[algo]['color'])

    
    #for algo in order:
    #    if algo in data:
    #        min_and_max_durs = [data[algo]['avg_dur_to_best'], max_dur]
    #        cur_avg_best = [data[algo]['avg_best'],data[algo]['avg_best']]
    #        cur_ax.plot(min_and_max_durs, cur_avg_best, color=order[algo]['color'], linestyle=":")
            
    cur_ax.legend([order[name]['name'] for name in order if name in data], loc='lower right')



    cur_ax.set_xlim(left=0, right = max_dur + 100)

