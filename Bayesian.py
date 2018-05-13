import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

NUM_TRIALS = 2000
PROBABILITIES = [.3, .4, .5,.6]


class Generator:
    def __init__(self, p):
        self.p = p
        self.a = 1
        self.b = 1

    def generate(self):
        return np.random.random() < self.p

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x):
        self.a += x
        self.b += 1 - x

def plot(bandits, trail):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label="real p: %4f" % b.p)
    plt.title("Bandit distribution after {} trails".format(trail))
    plt.legend()
    plt.show()


def experiment():
    bandits = [Generator(p) for p in PROBABILITIES]

    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    for i in xrange(NUM_TRIALS):
        bestb = None
        maxsample = -1
        allsamples = []
        for b in bandits:
            sample = b.sample()
            allsamples.append("%.4f" % sample)
            if sample > maxsample:
                maxsample = sample
                bestb = b
        if i in sample_points:
            print "Samples: %s" % allsamples
            plot(bandits, i)
        x = bestb.generate()
        bestb.update(x)

if __name__ == '__main__':
    experiment()
