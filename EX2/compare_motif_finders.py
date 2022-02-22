#!/usr/bin/python3 -u

import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
from random import Random

"""
This is a sanity-test script for your motif_find.py.
No need to read, edit or submit it this file!

Better make sure you pass all 4 tests before submitting.
Usage:
    python3 sanity_test.py PATH_TO_MOTIF_FIND
"""

seq = 'CCAAAATT'
states = 'BBMMMMBB'
ll = -10.47
alphabet = ['A', 'C', 'T', 'G']
p_choices = [0.1, 0.05, 0.2, 0.01]
q_choices = [0.75, 0.8, 0.9, 0.99]
trial_sequences = ['GGATG',
                   'GTGTCCTCAT',
                   'CTAATGATGTCGGTA',
                   'AAGAGTCTACCCCGAATGAT',
                   'TATCTGAGTCTCCCATGAACCAAGT',
                   'CCGTGGTATAGTCCATACTCTGAACCAAAA',
                   'CAGATAAACCAGCAAGATACATTGCAGAAGCTTGC',
                   'CACCTTAGCAGGTTGTCAGATATCCGTTTCTGGAACTCCC',
                   'GGGAGGACGATCGGAAGTTGAGCACAGGTACAAACACTTCAGGAA',
                   'TGATCTACTAAACTTTAGGGTCCGTACCTTTTATAATCCTTGCTAGCATC',
                   'ATGTTGAAGGTTAGAGGATTCCGAAACCAGAAGTGGCGATCTCGCTAAAGCAGGT',
                   'CACCACGGTCAGCGGGTGGCCATTTACTCGTGAAAACCATAGTCCGTGAAAGCTGGGCAA',
                   'CTTTAGTTGGGACCCTTAAGGCGACTGAGGGAAGCAACTATCGGAAGTATCGTACAGGTCGTAAA',
                   'GTACCAGTACGGAAGAAGCAGGGAGTTATAATATTCACTACCACAATTACCCGAGTTCACTTGTTTCAAT',
                   'CGCCCTCCCTTGACAGAACGTGCGTTACGTAGGAGTGCTTGACATACGGCGGCCGTCTGAGCTAGGACTATCGGA',
                   'GCGTAATAATGGGATTTCAAATTTACCAGTTCCAGGTTGTCCAAGGGCTTGGCGGTGAGTCGACATGGAAAGATAAATTC',
                   'CTCAGGTGCTGGCGCTCCCGTGGGGCCGCAGACACTACCTATTGGAGGGTGCTTAAACTATACAGCGCGCTAATTGTTAACTACT',
                   'CCTTTGTGTCATAAGGGAGGGGAAACACGCGAGGACCGCCTTTGATCTGGTTCAAACGCCTAGAAGTATCTCCATTCTGTCCATTACGCC',
                   'ACCGCCCCGTCGAATGGTACCGGTATCGCTTGACATCTGCTTCTATACTAGAACAACTAATGCCGGCTTCTGGAGTGAAGGCACCATCCCACCAG']

p_list = [0.01,
          0.01,
          0.1,
          0.2,
          0.01,
          0.01,
          0.2,
          0.01,
          0.2,
          0.05,
          0.05,
          0.2,
          0.05,
          0.1,
          0.2,
          0.05,
          0.2,
          0.1,
          0.1]

q_list = [0.9,
          0.99,
          0.75,
          0.9,
          0.99,
          0.9,
          0.8,
          0.99,
          0.99,
          0.9,
          0.75,
          0.75,
          0.75,
          0.99,
          0.75,
          0.99,
          0.9,
          0.8,
          0.9]


def is_output_correct(ret, alg):
    """ validate user output against school solution """

    r = [l.strip() for l in ret.split('\n') if l.strip()]
    if alg in ('forward', 'backward'):
        return len(r) == 1 and np.round(float(r[0]), 2) == ll
    else:
        return len(r) == 2 and r[0] == states and r[1] == seq


def test_single_alg(mf, epath, alg):
    """ run motif_find.py on a single algorithm
        (forward/ backward/ posterior/ viterbi) """
    print(f'testing {alg}...', end=' ')

    # run test as subprocess
    try:
        cmd = f'\"{sys.executable}\" {mf} --alg {alg} {seq} {epath} .1 .99'
        ret = subprocess.check_output(cmd, shell=True).decode()
    except Exception as e:
        print('Failed to run motif_find.py as a subprocess! ', e)
        return False

    # validate return value and print SUCCESS/FAIL
    if is_output_correct(ret, alg):
        print('\033[32m{}\033[00m'.format('SUCCESS'))
    else:
        print('\033[31m{}\033[00m'.format('FAIL'))


def compare_results(ret1, ret2, alg):
    """ validate user output against school solution """

    r1 = [l.strip() for l in ret1.split('\n') if l.strip()]
    r2 = [l.strip() for l in ret2.split('\n') if l.strip()]
    if alg in ('forward', 'backward'):
        assert (len(r1) == 1 and len(r2) == 1)
        # Another option:
        # return np.isclose(r1[0], r2[0])
        return np.round(float(r1[0]), 2) == np.round(float(r2[0]), 2)
    else:
        # Compares the bare strings. Could compare r1 and r2 (split) instead
        return ret1 == ret2


def compare_single_alg(mf1, mf2, epath, alg, in_seq, in_p, in_q):
    """ run motif_find.py on a single algorithm
        (forward/ backward/ posterior/ viterbi) """
    print(f'testing {alg}...', end=' ')

    # run test as subprocess
    try:
        cmd1 = f'\"{sys.executable}\" {mf1} --alg {alg} {in_seq} {epath} {in_p} {in_q}'
        ret1 = subprocess.check_output(cmd1, shell=True).decode()
        cmd2 = f'\"{sys.executable}\" {mf2} --alg {alg} {in_seq} {epath} {in_p} {in_q}'
        ret2 = subprocess.check_output(cmd2, shell=True).decode()
    except Exception as e:
        print('Failed to run motif_find.py as a subprocess! ', e)
        return False

    # validate return value and print SUCCESS/FAIL
    if compare_results(ret1, ret2, alg):
        print('\033[32m{}\033[00m'.format('SUCCESS'))
    else:
        print('\033[31m{}\033[00m'.format('FAIL'))


def main(args):
    # make sure input motif_find.py exists
    mf1 = args.my_motif_find_path
    mf2 = args.friend_motif_find_path
    if not os.path.isfile(mf1) or not os.path.isfile(mf2):
        print(f'Invalid input files')
        return 1

    # generate and dump a trivial emission table
    epath = './emissions_AAAA.tsv'
    pd.DataFrame(data=np.array([[.97, .01, .01, .01]] * 4),
                 columns=list('ACGT')).to_csv(epath, sep='\t',
                                              index=None)

    # Test your script with the official sanity tester
    print('\033[32mRunning default sanity test (comparing to official results)\033[00m')
    # test all 4 algorithms
    for alg in ('forward', 'backward', 'posterior', 'viterbi'):
        test_single_alg(mf1, epath, alg)

    # Compare your script to a friend's
    zipped = zip(trial_sequences, p_list, q_list)
    for in_seq, in_p, in_q in zipped:
        print('\033[32mComparing scripts for input seq: {} with p {} and q {}\033[00m'.format(in_seq, in_p, in_q))
        for alg in ('forward', 'backward', 'posterior', 'viterbi'):
            compare_single_alg(mf1, mf2, epath, alg, in_seq, in_p, in_q)

    # cleanup
    os.remove(epath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('my_motif_find_path', help='Path to your motif_find.py script (e.g. ./motif_find.py)')
    parser.add_argument('friend_motif_find_path', help='Path to a friend\'s motif_find.pyc file (or .py)')
    main(parser.parse_args())
