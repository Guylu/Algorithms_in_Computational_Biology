import argparse
from enum import Enum
from itertools import groupby

import numpy as np
import pandas as pd

COMPLEMENT_DNA = {
    "A": "T",
    "G": "C",
    "T": "A",
    "C": "G",
}


class Direction(Enum):
    LEFT = 0
    CROSS = 1
    TOP = 2
    RESET = 3


DEBUG = False
CHUNK_SIZE = 50


def _print_debug(msg):
    if DEBUG:
        print(msg)


def fastaread(fasta_name):
    """
    Read a fasta file. For each sequence in the file, yield the header and the actual sequence.
    In Ex1 you may assume the fasta files contain only one sequence.
    You may keep this function, edit it, or delete it and implement your own reader.
    """
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq


def calculate_gap_seq(seq, start, end, at_other_trail, allow_term_gaps, score_m):
    if allow_term_gaps:
        if end == len(seq):
            end -= 1

        if start == 0:
            start += 1

    return np.sum(np.asarray([score_m[seq[i]]["-"] for i in range(start, end)])) * (
        not (at_other_trail and allow_term_gaps))


def general_align(seq_a, a_start, a_end, seq_b, b_start, b_end, score_m, allow_term_gaps):
    if a_end - a_start == 0 and b_end - b_start == 0:
        return [], 0

    if a_end - a_start == 0:
        _print_debug(calculate_gap_seq(seq_b, b_start, b_end, a_start == len(seq_a) - 1, allow_term_gaps, score_m))
        return [[("-", seq_b[idx]) for idx in range(b_start, b_end)],
                calculate_gap_seq(seq_b, b_start, b_end, a_start == len(seq_a) - 1, allow_term_gaps, score_m), ]

    if b_end - b_start == 0:
        _print_debug(calculate_gap_seq(seq_a, a_start, a_end, b_end == 0, allow_term_gaps, score_m))
        return [[(seq_a[idx], "-") for idx in range(a_start, a_end)],
                calculate_gap_seq(seq_a, a_start, a_end, b_end == 0, allow_term_gaps, score_m), ]

    if a_end - a_start == 1 and b_end - b_start == 1:
        return [(seq_a[a_start], seq_b[b_start])], score_m[seq_a[a_start]][seq_b[b_start]]

    vector_len = a_end - a_start + 1
    v = [[score_m['-'][seq_a[0]] * i * (not (allow_term_gaps and b_start == 0)), Direction.TOP.value]
         for i in range(vector_len)]
    c = [[i, Direction.LEFT] for i in range(vector_len)]
    b_cutoff = b_start + (b_end - b_start) // 2

    for col in range(b_start, b_end):
        b_char = seq_b[col]
        # calculate b_char vs gap
        v_above = [score_m['-'][b_char] * (col - b_start + 1),
                   Direction.LEFT.value]
        c_above = [0, Direction.LEFT]

        for idx in range(a_end - a_start):
            a_term = a_start + idx == len(seq_a) - 1

            a_char = seq_a[a_start + idx]

            scores = np.array([
                v[idx + 1][0] + (score_m['-'][b_char] * (not (allow_term_gaps and a_term))),  # left
                v[idx][0] + score_m[a_char][b_char],  # top_left
                v_above[0] + (score_m[a_char]['-']),  # from_above
            ])

            # chose max from left(v[idx + 1])/left-top(v[idx])/top
            v_curr = [np.max(scores), np.argmax(scores)]

            if col >= b_cutoff:
                c_curr = c_above
                if v_curr[1] == Direction.CROSS.value:
                    c_curr = c[idx][:]
                    if col == b_cutoff:
                        c_curr[1] = Direction.CROSS
                elif v_curr[1] == Direction.LEFT.value:
                    c_curr = c[idx + 1][:]

                c[idx] = c_above[:]
                c_above = c_curr[:]

                if idx == a_end - a_start - 1:
                    c[idx + 1] = c_curr[:]

            # setup for next iteration
            v[idx] = v_above[:]
            v_above = v_curr[:]

            # reached last cell in vector
            if idx == a_end - a_start - 1:
                v[idx + 1] = v_curr[:]

    a_cutoff = a_start + c[-1][0]
    direction = c[-1][1]

    _print_debug("call with: a_start - {}, a_cutoff - {}, b_start - {}, b_cutoff - {}".format(
        a_start, a_cutoff, b_start, b_cutoff))
    seq_start, sum_score = general_align(seq_a, a_start, a_cutoff, seq_b, b_start, b_cutoff, score_m, allow_term_gaps)

    if direction == Direction.CROSS:
        seq_start.append((seq_a[a_cutoff], seq_b[b_cutoff]))
        sum_score += score_m[seq_a[a_cutoff]][seq_b[b_cutoff]]
    else:
        seq_start.append(("-", seq_b[b_cutoff]))
        sum_score += score_m["-"][seq_b[b_cutoff]] * (not (allow_term_gaps and a_cutoff == len(seq_a)))

    _print_debug("call with: a_cutoff - {}, a_end - {}, b_cutoff - {}, b_end - {}".format(
        a_cutoff + direction.value, a_end, b_cutoff + 1, b_end))
    seq_end, _ = general_align(seq_a, a_cutoff + direction.value, a_end, seq_b, b_cutoff + 1, b_end, score_m,
                               allow_term_gaps)
    sum_score += _

    _print_debug("calculated score: {}\nsummed score:{}\n".format(v[-1][0], sum_score))
    seq_start.extend(seq_end)

    return seq_start, v[-1][0]


def global_align(seq_a, a_start, a_end, seq_b, b_start, b_end, score_m):
    return general_align(seq_a, a_start, a_end, seq_b, b_start, b_end, score_m, allow_term_gaps=False)


def overlap_align(seq_a, a_start, a_end, seq_b, b_start, b_end, score_m):
    return general_align(seq_a, a_start, a_end, seq_b, b_start, b_end, score_m, allow_term_gaps=True)


def local_align(seq_a, seq_b, score_m):
    V = np.zeros(shape=(len(seq_b) + 1, len(seq_a) + 1, 2))
    path_interpretation = {0: np.array([1, 0]), 1: np.array([1, 1]),
                           2: np.array([0, 1]),
                           3: None}

    for i in range(1, V.shape[0]):
        for j in range(1, V.shape[1]):
            list = [V[i - 1, j, 0] + score_m[seq_b[i - 1]]['-'],
                    V[i - 1, j - 1, 0] + score_m[seq_b[i - 1]][seq_a[j - 1]],
                    V[i, j - 1, 0] + score_m['-'][seq_a[j - 1]],
                    0]
            V[i][j] = np.max(list), np.argmax(list)

    score = np.max(V)
    y, x = np.unravel_index(V.T[0, :].argmax(), V.T[0, :].shape)
    path = []
    p = V[x][y][1]
    while p != Direction.RESET.value and x in range(1, V.T[0, :].shape[1]) and \
            y in range(1, V.T[0, :].shape[0]):
        if p == Direction.CROSS.value:
            path.append([seq_a[y - 1], seq_b[x - 1]])
        elif p == Direction.LEFT.value:
            path.append(["-", seq_b[x - 1]])
        elif p == Direction.TOP.value:
            path.append([seq_a[y - 1], "-"])

        x, y = np.array([x, y]) - path_interpretation[p]
        p = V[x][y][1]

    path.reverse()
    return path, score


def pretty_print(seq, align_type, score):
    formatted_a = ""
    formatted_b = ""

    for from_a, from_b in seq:
        formatted_a += from_a
        formatted_b += from_b

    chunk = 0
    while chunk < len(formatted_a):
        print(formatted_a[chunk:chunk + CHUNK_SIZE])
        print(formatted_b[chunk:chunk + CHUNK_SIZE])
        print()
        chunk += CHUNK_SIZE

    print("{}: {}".format(align_type, score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', help='run in debug mode', action='store_const', const=True, default=False)
    parser.add_argument('seq_a', help='Path to first FASTA file (e.g. fastas/HomoSapiens-SHH.fasta)')
    parser.add_argument('seq_b', help='Path to second FASTA file')
    parser.add_argument('--align_type', help='Alignment type (e.g. local)', required=True)
    parser.add_argument('--score', help='Score matrix in.tsv format (default is score_matrix.tsv) ',
                        default='score_matrix.tsv')
    command_args = parser.parse_args()

    if not command_args.debug:
        seq_a = fastaread(command_args.seq_a).__next__()[1]
        seq_b = fastaread(command_args.seq_b).__next__()[1]
    else:
        global DEBUG
        DEBUG = True
        import time

        start_time = time.time()
        seq_a = command_args.seq_a
        seq_b = command_args.seq_b

    score_m = pd.read_csv(command_args.score, sep='\t')
    score_m.index = score_m['Unnamed: 0']
    score_m = score_m.drop('Unnamed: 0', 1)

    seq, score = [], 0
    if command_args.align_type == 'global':
        seq, score = global_align(seq_a, 0, len(seq_a), seq_b, 0, len(seq_b), score_m)
    elif command_args.align_type == 'local':
        seq, score = local_align(seq_a, seq_b, score_m)
    elif command_args.align_type == 'overlap':
        seq, score = overlap_align(seq_a, 0, len(seq_a), seq_b, 0, len(seq_b), score_m)

    pretty_print(seq, command_args.align_type, score)

    if command_args.debug:
        import sys
        print("--- elapsed {} seconds ---".format(time.time() - start_time), file=sys.stderr)


if __name__ == '__main__':
    main()

