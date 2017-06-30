#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import argparse
import subprocess
import json


WORD2VEC_OPTIONS = '-cbow 0 -window 5 -negative -hs 1 -sample 1e-3 -binary 0'

UNK = '<unk>'
EOS = '</s>'


def main():
    args = parse_user_args()

    print("Loading vocabulary")
    ext = os.path.splitext(args.vocab)[1]
    if ext == ".json":
        with open(args.vocab) as vocab_file:
            vocab = json.load(vocab_file)
    elif ext == ".yaml" or ext == ".yml":
        with open(args.vocab) as vocab_file:
            # custom YAML loader as PyYAML skips some entries
            vocab = load_yaml(vocab_file)
    else:
        print("Error: extension of vocabulary not recognized")
        exit(1)

    lines = sum(1 for line in open(args.vocab))
    print("  lines: {}".format(lines))
    print("  entries: {}".format(len(vocab)))

    print("Adding <unk> na </s> tokens to the corpus")
    prep_corpus = args.corpus + '.prep'
    with open(args.corpus) as cin, open(prep_corpus, 'w+') as cout:
        for line in cin:
            cout.write(replace_unks(line, vocab) + " " + EOS + "\n")

    print("Training word2vec")
    cmd = "{w2v} {opts} -train {i} -output {o} -size {s} -threads {t}" \
        .format(w2v=args.word2vec, opts=WORD2VEC_OPTIONS,
                i=args.corpus, o=args.output, s=args.dim_emb, t=args.threads)
    print("  with command: {}".format(cmd))

    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()

    print("Checking vectors")
    with open(args.output) as cin:
        for i, line in enumerate(cin):
            if i == 0:
                cout.write(line)
                continue
            word, tail = line.split(' ', 1)
            if word not in vocab:
                print("  warning: no word '{}' found in vocabulary")

    print("Finished")


def replace_unks(l, voc):
    return " ".join([w if w in voc else UNK for w in l.strip().split()])


def load_yaml(lines):
    vocab = {}
    for line in lines:
        word, idx = line.strip().split(': ')
        vocab[word.strip('"')] = int(idx)
    return vocab


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Corpus file to train word2vec")
    parser.add_argument("-v", "--vocab", help="Path to vocabulary in JSON or YAML format", required=True)
    parser.add_argument("-w", "--word2vec", help="Path to word2vec", required=True)
    parser.add_argument("-o", "--output", help="File to write trained vectors", required=True)
    parser.add_argument("-d", "--dim-emb", help="Size of embedding vector", default=512)
    parser.add_argument("-t", "--threads", help="Number of threads", default=16)
    return parser.parse_args()


if __name__ == '__main__':
    main()
