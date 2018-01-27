import scipy.stats
import numpy as np

def read_data(f_name):
    f = open('data/{}'.format(f_name), 'r')
    lines = f.readlines()
    to_return = []
    for num in lines:
        to_return.append(float(num.strip()))
    return to_return


def compute_corr():


    test = read_data('dan_accuracies.txt')
    dev = read_data('cnn_dev_accuracies.txt')
    time = read_data('cnn_times.txt')

    #(dev_test_corr, dev_test_p) = scipy.stats.pearsonr(test,dev)
    #(time_test_corr, time_test_p) = scipy.stats.pearsonr(test,time)
    #print("dev test corr: {}, p-value: {}".format(dev_test_corr, dev_test_p))
    #print("time test corr: {}, p-value: {}".format(time_test_corr, time_test_p))

    print("min: {}, max: {}, mean: {}, std dev: {}".format(min(test), max(test), np.mean(test), np.std(test)))

    

if __name__ == "__main__":
    compute_corr()
