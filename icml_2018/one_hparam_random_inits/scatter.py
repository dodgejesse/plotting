import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_data(f_name):
    f = open('data/{}'.format(f_name), 'r')
    lines = f.readlines()
    to_return = []
    for num in lines:
        to_return.append(float(num.strip()))
    return to_return

def make_plot():
    matplotlib.rcParams.update({'font.size':15})
    fig = plt.figure(figsize=(10,10))
    cur_ax = fig.add_subplot(1,1,1)


    X = read_data('times.txt')
    Y = read_data('accuracies.txt')
    
    cur_ax.scatter(X,Y)
    # the reported accuracy in the paper:
    reported_acc = [.469,.469]
    min_and_max_time = [min(X), max(X)]
    cur_ax.plot(min_and_max_time,reported_acc, color='red', label='reported accuracy')

    
    cur_ax.set_title('100 random restarts of DAN, default hyperparameter settings\non 5-way Stanford sentiment text cat\nhidden dim=300, num layers=3, dropout=0.3, regularize=1e-4')
    cur_ax.set_ylabel('accuracy')
    cur_ax.set_xlabel('training and evaluation time (seconds)')
    cur_ax.legend()
    
    save_loc = 'plots/100_random_restarts.pdf'
    print('saving to {}'.format(save_loc))
    plt.savefig(save_loc)



if __name__ == "__main__":
    make_plot()
