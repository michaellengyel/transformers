from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from scipy import interpolate


def main():

    for i in range(300):

        timestamp = np.sort(np.random.rand(4))
        distance = np.sort(np.random.rand(4))

        data = np.array((timestamp, distance))
        tck, u = interpolate.splprep(data, s=0)
        unew = np.arange(0, 1.01, 0.01)
        out = interpolate.splev(unew, tck)
        plt.plot(out[0], out[1], color='orange')

        plt.plot(data[0, :], data[1, :], 'ob')

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.show()

if __name__ == '__main__':
    main()
