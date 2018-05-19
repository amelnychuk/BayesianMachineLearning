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
    generators = [Generator(p) for p in PROBABILITIES]

    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    for i in xrange(NUM_TRIALS):
        best_gen = None
        maxsample = -1
        allsamples = []
        for generator in generators:
            sample = generator.sample()
            allsamples.append("%.4f" % sample)
            if sample > maxsample:
                maxsample = sample
                best_gen = generator
        if i in sample_points:
            print "Samples: %s" % allsamples
            plot(generators, i)
        x = best_gen.generate()
        best_gen.update(x)


def thompson_convergence():
    NUM_TRIALS = 20000

    seeds = np.random.random(3)
    generators = [Generator(r) for r in seeds]

    data = np.empty(NUM_TRIALS)

    for i in xrange(NUM_TRIALS):
        j = np.argmax([g.sample() for g in generators])
        x = generators[j].generate()
        generators[j].update(x)
        data[i] = x

    cumulative_avg_ctr = np.cumsum(data) / (np.arange(NUM_TRIALS) +1)
    plt.plot(cumulative_avg_ctr)

    for seed in seeds:
        plt.plot(np.ones(NUM_TRIALS)*seed)
    plt.ylim((0,1))
    plt.xscale('log')
    plt.show()



if __name__ == '__main__':
    experiment()
    #thompson_convergence()
