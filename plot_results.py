import numpy as np
import argparse
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', help='The path to the directory containing the results.', type=str)
    args = parser.parse_args()

    path = args.results

    mean_loss = np.loadtxt(path+'learning.csv', delimiter=',', skiprows=1, usecols=2)
    results = np.loadtxt(path+'results.csv', delimiter=',', skiprows=1, usecols=(2,5))

    plt.figure(1)
    plt.plot(mean_loss)
    plt.title('MSE')
    plt.savefig(path+'mean_loss.png')

    plt.figure(2)
    plt.subplot(211)
    plt.plot(results[:,0])
    plt.title('Average length of an episode')
    plt.subplot(212)
    plt.plot(results[:,1])
    plt.title('Average score per episode')
    plt.tight_layout()
    plt.savefig(path+'results.png')
