__author__ = 'fnaiser'
import pickle
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math

if __name__ == '__main__':
    with open('/Users/fnaiser/Documents/graphs2/chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)

    print len(chunks)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for ch in chunks:
        ch = np.array(ch)

        std_lim = 3.
        if np.std(ch[:, 0]) < std_lim and np.std(ch[:, 1]) < std_lim and ch.shape[0] > 1:
            continue

        if ch.shape[0] == 1:
            continue
        else:
            avg_a = np.mean(ch[:, 3])
            w_ = max(1, math.log(avg_a)-4)
            ax.plot(ch[:, 1], ch[:, 0], ch[:, 2], linewidth=w_)

        plt.hold(True)

    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
    plt.hold(False)
    plt.show()
