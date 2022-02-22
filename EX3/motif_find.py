import argparse
import pandas as pd
import numpy as np
from scipy.special import logsumexp

DEBUG = False
CHUNK_SIZE = 50


def logdot(a, b):
    max_a, max_b = np.max(a), np.max(b)
    exp_a, exp_b = a - max_a, b - max_b
    np.exp(exp_a, out=exp_a)
    np.exp(exp_b, out=exp_b)
    c = np.dot(exp_a, exp_b)
    np.log(c, out=c)
    c += max_a + max_b
    return c


def _print_debug(msg):
    if DEBUG:
        print(msg)


def forward(seq, emissions, tau):
    f = np.zeros(shape=(emissions.shape[0], len(seq)))
    f[0, 0] = 1

    with np.errstate(divide='ignore'):
        f[:, 0] = np.log(f[:, 0])

    for col in range(1, f.shape[1]):
        with np.errstate(divide='ignore'):
            temp = logdot(f[:, col - 1], tau)
        f[:, col] = temp + np.array(emissions[seq[col]])

    return f


def backward(seq, emissions, tau):
    b = np.zeros(shape=(emissions.shape[0], len(seq)))
    b[-1, -1] = 1
    # b[:, -1] = 1

    with np.errstate(divide='ignore'):
        b[:, -1] = np.log(b[:, -1])

    for col in range(b.shape[1] - 2, -1, -1):
        temp = b[:, col + 1].reshape(-1, 1) + tau.T + np.array(emissions[seq[col + 1]]).reshape(-1, 1)
        b[:, col] = logsumexp(temp, axis=0)

    return b


def posterior(seq, emissions, tau):
    f = forward(seq, emissions, tau)
    b = backward(seq, emissions, tau)

    path = ""
    for i in range(len(seq)):
        idx = np.argmax(f[:, i] + b[:, i])
        path += "M" if 1 < idx < emissions.shape[0] - 2 else "B"

    return path[1:-1], seq[1:-1]


def viterbi(seq, emissions, tau, run_log=True):
    v = np.zeros((emissions.shape[0], len(seq)))
    p = np.zeros((emissions.shape[0], len(seq)))
    # v[:, 0] = 1
    v[0, 0] = 1

    if run_log:
        with np.errstate(divide='ignore'):
            v[:, 0] = np.log(v[:, 0])
            tau = np.log(tau)
            emissions = np.log(emissions)

    for i in range(1, len(seq)):
        if run_log:
            v[:, i] = np.array(emissions[seq[i]]) + np.max(v[:, i - 1].reshape(-1, 1) + tau, axis=0)
            p[:, i] = np.argmax(v[:, i - 1].reshape(-1, 1) + tau, axis=0)
        else:
            for k in range(v.shape[0]):
                v[k, i] = emissions[seq[i]][k] * np.max(v[:, i - 1] * tau[:, k])
                p[k, i] = np.argmax(np.exp(v[:, i - 1].T) * tau[:, k])

    res = np.zeros(len(seq), dtype=np.int64)
    res[len(seq) - 1] = np.argmax(v[:, -1])

    for i in range(len(seq) - 1, 1, -1):
        res[i - 1] = p[res[i], i]

    path = ""
    for i in res:
        path += "M" if 1 < i < emissions.shape[0] - 2 else "B"

    return path[1:-1], seq[1:-1]


def generate_transitions(emission_path, p, q):
    pad = 4
    init_emissions = pd.read_csv(emission_path, sep='\t', index_col=False)
    columns = init_emissions.columns.append(pd.DataFrame(columns=["^", "$"]).columns)

    k = init_emissions.shape[0]
    emission = np.zeros((k + pad, len(columns)))
    emission[1, : -2] = np.ones(len(columns) - 2) * 0.25
    emission[-2, : -2] = np.ones(len(columns) - 2) * 0.25
    emission[2:-2, : -2] = np.array(init_emissions)
    emission[-1, -1] = 1
    emission[0, -2] = 1

    emission = pd.DataFrame(emission, columns=columns)

    tau = np.zeros((k + pad, k + pad))

    tau[0, 1] = q
    tau[0, k + (pad // 2)] = 1 - q
    tau[1, 1] = 1 - p
    tau[1, 2] = p

    for i in range(2, k + (pad // 2)):
        tau[i, i + 1] = 1

    tau[k + (pad // 2), k + (pad // 2)] = 1 - p
    tau[k + (pad // 2), k + 1 + (pad // 2)] = p

    return emission, tau


def pretty_print(posterior, seq):
    chunk = 0
    while chunk < len(posterior):
        print(posterior[chunk:chunk + CHUNK_SIZE])
        print(seq[chunk:chunk + CHUNK_SIZE])
        print()
        chunk += CHUNK_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', help='run in debug mode', action='store_const', const=True, default=False)
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
    parser.add_argument('seq', help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
    parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emission.tsv)')
    parser.add_argument('p', help='transition probability p (e.g. 0.01)', type=float)
    parser.add_argument('q', help='transition probability q (e.g. 0.5)', type=float)
    parser.add_argument('--log', help='don\'t run calculations in log scale', action='store_const',
                        const=False,
                        default=True)
    args = parser.parse_args()

    emissions, tau = generate_transitions(args.initial_emission, args.p, args.q)
    seq = "^{}$".format(args.seq)

    if args.alg == 'viterbi':
        pretty_print(*viterbi(seq, emissions, tau, args.log))
    elif args.alg == 'forward':
        vec = forward(seq, emissions, tau)[:, -1]
        print(logsumexp(vec) if args.log else np.log(np.sum(vec)))
    elif args.alg == 'backward':
        vec = backward(seq, emissions, tau)[:, 0]
        print(logsumexp(vec) if args.log else np.log(np.sum(vec)))
    elif args.alg == 'posterior':
        pretty_print(*posterior(seq, emissions, tau))


if __name__ == '__main__':
    main()
