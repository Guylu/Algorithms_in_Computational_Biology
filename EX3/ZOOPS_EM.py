import argparse
import os

import numpy as np
import pandas as pd
import motif_find as finder
import matplotlib.pyplot as plt

from scipy.special import logsumexp

LETTER_TRANS = {
    "^": 0,
    "A": 1,
    "T": 2,
    "C": 3,
    "G": 4,
    "$": 5,
}
DEBUG = False


def generate_transitions(hidden_states, p, q):
    tau = np.zeros((hidden_states + 4, hidden_states + 4))

    tau[0, 1] = q
    tau[0, hidden_states + 2] = 1 - q
    tau[1, 1] = 1 - p
    tau[1, 2] = p

    for i in range(2, hidden_states + 2):
        tau[i, i + 1] = 1

    tau[hidden_states + 2, hidden_states + 2] = 1 - p
    tau[hidden_states + 2, hidden_states + 3] = p

    return tau


def generate_emissions(motif, alpha):
    emissions = pd.DataFrame(np.zeros((4 + len(motif), 6)), columns=["^", "A", "T", "C", "G", "$"])

    emissions["^"][0] = 1
    emissions["$"][emissions.shape[0] - 1] = 1

    emissions.T[1][1:-1] = np.ones(4) / 4
    emissions.T[emissions.shape[0] - 2][1:-1] = np.ones(4) / 4

    for row, value in enumerate(motif):
        for col in "ACTG":
            emissions[col][row + 2] = 1 - 3 * alpha if value == col else alpha

    return emissions


def read_fasta(fasta):
    res = []
    with open(fasta, "r") as input_file:
        for line in input_file.readlines():
            if line.startswith(">"):
                continue

            res.append("^" + line.strip() + "$")

    return res


def _em_step(emissions, tau, samples):
    ll_sum = 0

    with np.errstate(divide='ignore'):
        nkx = np.log(np.zeros((tau.shape[0], len(emissions.columns))))
        nkl = np.log(np.zeros_like(tau))

    for seq in samples:
        f = finder.forward(seq, emissions, tau)
        b = finder.backward(seq, emissions, tau)
        ll = b[0, 0]
        ll_sum += ll

        for i, letter in enumerate(seq):
            emitted = LETTER_TRANS[letter]

            move = f[:, i] + b[:, i] - ll
            nkx[:, emitted] = np.logaddexp(nkx[:, emitted], move)

            if i < 1:
                continue

            vsum = f[:, i - 1].reshape(-1, 1) + b[:, i].reshape(1, -1) + np.array(emissions[letter]).reshape(1, -1)
            nkl = np.logaddexp(nkl, tau + vsum - ll)

    new_emissions = np.exp(nkx - np.array(logsumexp(nkx, axis=1)).reshape(-1, 1))

    p = np.exp(np.logaddexp(nkl[1, 2], nkl[-2, -1]) - np.logaddexp(logsumexp(nkl[1]), logsumexp(nkl[-2])))
    q = np.exp(nkl[0, 1] - logsumexp(nkl[0]))
    new_tau = generate_transitions(tau.shape[0] - 4, p, q)

    return new_emissions, new_tau, ll_sum


def baum_welch(tau, emissions, samples, epsilon):
    ll_history = []

    curr_emissions = emissions.copy()

    new_emissions, new_tau, ll = _em_step(np.log(emissions), np.log(tau), samples)
    ll_history.append(ll)

    while len(ll_history) < 2 or abs(ll_history[-1] - ll_history[-2]) > epsilon:
        curr_tau = new_tau.copy()
        curr_emissions[2:-2] = new_emissions[2:-2]

        new_emissions, new_tau, ll = _em_step(np.log(curr_emissions), np.log(curr_tau), samples)
        ll_history.append(ll)

    curr_tau = new_tau.copy()
    curr_emissions[2:-2] = new_emissions[2:-2]

    return curr_emissions, curr_tau, ll_history


def pretty_print(emissions, tau, ll_history, samples):
    with open("./ll_history.txt", "w") as output:
        for value in ll_history:
            output.write(str(value) + os.linesep)

    with open("./motif_profile.txt", "w") as output:
        for letter in "ACGT":
            output.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(*np.array(emissions[letter][2:-2])) + os.linesep)
        output.write("{:.4f}".format(tau[1, 2]) + os.linesep)
        output.write("{:.4f}".format(tau[0, 1]) + os.linesep)

    with open("./motif_positions.txt", "w") as output:
        for seq in samples:
            orig_seq = seq[1:-1]
            location = finder.viterbi(seq, emissions, tau)[0].find("M")
            if DEBUG:
                output.write("{}, {}, ({})".format(
                    str(location), orig_seq[location:location + tau.shape[0] - 4], orig_seq) + os.linesep)
            else:
                output.write(str(location) + os.linesep)


def parse_args():
    """
    Parse the command line arguments.
    :return: The parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta', help='File path with list of sequences (e.g. seqs_ATTA.fasta)')
    parser.add_argument('seed', help='Guess for the motif (e.g. ATTA)')
    parser.add_argument('p', type=float, help='Initial guess for the p transition probability (e.g. 0.01)')
    parser.add_argument('q', type=float, help='Initial guess for the q transition probability (e.g. 0.9)')
    parser.add_argument('alpha', type=float, help='Softening parameter for the initial profile (e.g. 0.1)')
    parser.add_argument('convergenceThr', type=float, help='ll improvement threshold for the stopping condition'
                                                           ' (e.g. 0.1)')
    parser.add_argument('--debug', action="store_const", const=True, default=False, help='run in debug mode with plots')
    return parser.parse_args()


def main():
    args = parse_args()
    global DEBUG
    DEBUG = args.debug

    # build transitions 
    tau = generate_transitions(len(args.seed), args.p, args.q)

    # build emissions
    emissions = generate_emissions(motif=args.seed, alpha=args.alpha)

    # load fasta
    samples = read_fasta(args.fasta)

    # run Baum-Welch
    final_emissions, final_tau, ll_history = baum_welch(tau, emissions, samples, args.convergenceThr)

    # dump results
    pretty_print(final_emissions, final_tau, ll_history, samples)

    if DEBUG:
        plt.plot(ll_history)
        plt.show()


if __name__ == "__main__":
    main()
