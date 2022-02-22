import argparse
from itertools import groupby

import numpy as np
import pandas as pd


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


def global_align(seq_a, a_start, a_end, seq_b, b_start, b_end, score_m):
    """

    @param seq_a:
    @param a_start:
    @param a_end:
    @param seq_b:
    @param b_start:
    @param b_end:
    @param score_m:
    @return:
    """
    if a_end - a_start == 0 and b_end - b_start == 0:
        return [], 0
    if a_end - a_start == 0:
        return [(0, b_end - b_start)], score_m['-'][0] * (b_end - b_start)
    if b_end - b_start == 0:
        return [(a_end - a_start, 0)], score_m['-'][0] * (a_end - a_start)

    if a_end - a_start == 1 and b_end - b_start == 1:
        return [(1, 1)], score_m[seq_a[a_start]][seq_b[b_start]]

    vector_len = a_end - a_start + 1
    v = [score_m['-'][0] * i for i in range(vector_len)]
    c = [i for i in range(vector_len)]
    b_cutoff = b_start + ((b_end - b_start) // 2)

    for col in range(b_start, b_end):
        # calculate b_char vs gap
        above = score_m['-'][0] * (col - b_start + 1)
        c_above = 0
        b_char = seq_b[col]

        for idx in range(a_end - a_start):
            a_char = seq_a[a_start + idx]

            top_left = v[idx] + score_m[a_char][b_char]
            left = v[idx + 1] + score_m['-'][b_char]

            # chose max from top/left-top(v[idx])/left(v[idx + 1])
            curr = max(above + score_m[a_char]['-'], top_left, left)

            if col >= b_cutoff:
                curr_c = c_above
                if curr == top_left:
                    curr_c = c[idx]
                elif curr == left:
                    curr_c = c[idx + 1]

                c[idx] = c_above
                c_above = curr_c

                if idx == a_end - a_start - 1:
                    c[idx + 1] = curr_c

            # setup for next iteration
            v[idx] = above
            above = curr

            # reached last cell in vector
            if idx == a_end - a_start - 1:
                v[idx + 1] = curr

    a_cutoff = a_start + c[-1]

    seq_start, sum_score = global_align(seq_a, a_start, a_cutoff, seq_b,
                                        b_start, b_cutoff, score_m)
    seq_end, _ = global_align(seq_a, a_cutoff, a_end, seq_b, b_cutoff, b_end,
                              score_m)

    # sum_score += _
    # print("calculated score: {}\nsummed score:{}\n".format(v[-1], sum_score))
    seq_start.extend(seq_end)

    return seq_start, v[-1]


def pretty_print(seq_a, seq_b, seq, align_type, score):
    formatted_a = ""
    formatted_b = ""
    a_read = 0
    b_read = 0

    for from_a, from_b in seq:
        formatted_a += seq_a[a_read:a_read + from_a]
        a_read += from_a
        formatted_b += seq_b[b_read:b_read + from_b]
        b_read += from_b

        if from_a == 0:
            formatted_a += "-" * from_b

        if from_b == 0:
            formatted_b += "-" * from_a

    chunk = 0
    while chunk < len(formatted_a):
        print(formatted_a[chunk:chunk + 50])
        print(formatted_b[chunk:chunk + 50])
        print()
        chunk += 50

    print("{}: {}".format(align_type, score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_a',
                        help='Path to first FASTA file (e.g. '
                             'fastas/HomoSapiens-SHH.fasta)')
    parser.add_argument('seq_b', help='Path to second FASTA file')
    parser.add_argument('--align_type', help='Alignment type (e.g. local)',
                        required=True)
    parser.add_argument('--score',
                        help='Score matrix in.tsv format (default is '
                             'score_matrix.tsv) ',
                        default='score_matrix.tsv')
    command_args = parser.parse_args()

    seq_a = fastaread(command_args.seq_a).__next__()[1]
    seq_b = fastaread(command_args.seq_b).__next__()[1]
    score_m = pd.read_csv(command_args.score, sep='\t')
    score_m.index = score_m['Unnamed: 0']
    score_m = score_m.drop('Unnamed: 0', 1)

    seq, score = [], 0
    if command_args.align_type == 'global':
        seq, score = global_align(seq_a, 0, len(seq_a), seq_b, 0, len(seq_b),
                                  score_m)
    elif command_args.align_type == 'local':
        raise NotImplementedError
    elif command_args.align_type == 'overlap':
        raise NotImplementedError

    pretty_print(seq_a, seq_b, seq, command_args.align_type, score)


if __name__ == '__main__':
    main()
