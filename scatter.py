import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def make_plot():
    fig = plt.figure(figsize=(10,10))
    cur_ax = fig.add_subplot(1,1,1)

    # training time
    X = [100, 120, 95, 122]
    Y = [.8, .7, .6, .75]
    
    cur_ax.scatter(X,Y)
    cur_ax.set_title('title')
    cur_ax.set_ylabel('accuracy')
    cur_ax.set_xlabel('time')
    plt.savefig('example.pdf')



if __name__ == "__main__":
    make_plot()
