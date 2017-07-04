#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import argparse
import subprocess
import json

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
        sys.stderr.write("Error: extension of vocabulary not recognized\n")
        exit(1)

    lines = sum(1 for line in open(args.vocab))
    sys.stderr.write("  entries: {}\n".format(len(vocab)))

    if args.dim_voc is not None:
        vocab = {w: v for w, v in vocab.items() if v < args.dim_voc}
        sys.stderr.write("  loaded: {}\n".format(len(vocab)))

    sys.stderr.write("Adding <unk> and </s> tokens to the corpus\n")
    for line in sys.stdin:
        sys.stdout.write(replace_unks(line, vocab) + " " + EOS + "\n")

def replace_unks(l, voc):
    return " ".join([w if w in voc else UNK for w in l.strip().split()])


def load_yaml(lines):
    vocab = {}
    for line in lines:
        # all values are integers, so splitting by ':' from right should be safe
        word, idx = line.strip().rsplit(':', 1)
        vocab[word.strip('"')] = int(idx.strip())
    return vocab


def parse_user_args():
    desc = """Prepare corpus w.r.t to vocabulary, i.e. add <unk> and </s>."""
    note = """Examples:
  {0} -v vocab.yml -i corpus.txt -o output.txt -w path/to/word2vec
  {0} -v vocab.yml -i vectors.txt -o output.txt"""
    note = note.format(os.path.basename(__file__))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=desc,
        epilog=note)
    parser.add_argument("-v", "--vocab", help="path to vocabulary in JSON or YAML format", required=True)
    parser.add_argument("--dim-voc", help= "maximum number of words from vocabulary to be used, default: no limit", type=int)
    parser.add_argument("--quiet", help="skip printing warnings", action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
