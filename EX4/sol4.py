import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

LETTER_TRANS = {
    "A": 0,
    "T": 1,
    "C": 2,
    "G": 3,
}

RANGE = np.arange(4)


def sample_transition(a, epsilon):
    prob = np.random.binomial(1, 0.25 + 0.25 * 3 * np.exp(-4 * epsilon))

    if prob == 1:
        return a

    return np.random.choice(RANGE[RANGE != a])


def generate_n(n, a, epsilon):
    samples = np.array([sample_transition(a, epsilon) for i in range(n)])
    hist = np.histogram(samples, bins=4)
    return hist[0]


def test_sampler():
    for t in [0.04, 0.1, 0.3]:
        l = []
        for n in [10, 100, 1000, 10000]:
            l.append(generate_n(n, 0, t))
            # plt.bar(hist)
            # plt.show()
        # plot actual results:
        fig = plot_bar(l, t)
        fig.tight_layout()
        plt.show()
        # plot ratio:

    # ll = l[:4]
    # fig = plot_bar(ll, 0.04)
    #
    # fig.tight_layout()
    #
    # plt.show()


def plot_bar(ll, t):
    labels = ['A', 'T', 'G', 'C']
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3 / 2 * width, ll[0], width, label='10')
    rects2 = ax.bar(x - width / 2, ll[1], width, label='100')
    rects3 = ax.bar(x + width / 2, ll[2], width, label='1000')
    rects4 = ax.bar(x + width * 3 / 2, ll[3], width, label='10000')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency per size of sample with t: ' + str(t))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    return fig


def sample_pair_transition(epsilon):
    a = np.random.choice(RANGE)
    return a, sample_transition(a, epsilon)


def generate_n_pairs(n, epsilon):
    return np.array([sample_pair_transition(epsilon) for i in range(n)])


def estimate_t(m=100):
    n = 500
    l = np.empty((m, 3))
    epsilons = [0.04, 0.1, 0.3]

    for i in range(m):
        for j, t in enumerate(epsilons):
            pairs = generate_n_pairs(n, t)
            non_equals = np.count_nonzero(pairs[:, 0] - pairs[:, 1])
            equals = n - non_equals

            mle = -np.log((3 * equals - non_equals) / 3 * (equals + non_equals))
            l[i, j] = np.exp(mle)

    plt.boxplot(l, labels=epsilons)
    plt.show()

if __name__ == '__main__':
    estimate_t()
