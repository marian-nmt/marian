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

    if args.dim_voc is not None:
        vocab = {w: v for w, v in vocab.items() if v < args.dim_voc}
        print("  loaded: {}".format(len(vocab)))

    if args.word2vec:
        print("Adding <unk> and </s> tokens to the corpus")
        prep_corpus = args.input + '.prep'
        with open(args.input) as cin, open(prep_corpus, 'w+') as cout:
            for line in cin:
                cout.write(replace_unks(line, vocab) + " " + EOS + "\n")

        print("Training word2vec")
        orig_vectors = args.output + '.orig'
        cmd = "{w2v} {opts} -train {i} -output {o} -size {s} -threads {t}" \
            .format(w2v=args.word2vec, opts=WORD2VEC_OPTIONS,
                    i=prep_corpus, o=orig_vectors, s=args.dim_emb, t=args.threads)
        print("  with command: {}".format(cmd))

        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()
    else:
        print("No training")
        orig_vectors = args.input

    print("Replacing words with IDs in vector file")
    n = 0
    embs = []
    dim_emb = None
    with open(orig_vectors) as cin:
        for i, line in enumerate(cin):
            if i == 0:
                dim_emb = line.strip().split(' ')[-1]
                continue
            word, tail = line.split(' ', 1)
            if word in vocab:
                embs.append("{} {}".format(vocab[word], tail))
                n += 1
            else:
                if not args.quiet:
                    print("  no word '{}' in vocabulary, line {}".format(
                        word, i + 1))

    print("  words: {}".format(n))

    with open(args.output, 'w') as cout:
        cout.write("{} {}\n".format(n, dim_emb))
        for emb in embs:
            cout.write(emb)

    print("Finished")


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
    desc = """Process embeddings (in text format) trained with word2vec or train new
embedding vectors with regard to the word vocabulary."""
    note = """Examples:
  {0} -v vocab.yml -i corpus.txt -o output.txt -w path/to/word2vec
  {0} -v vocab.yml -i vectors.txt -o output.txt"""
    note = note.format(os.path.basename(__file__))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=desc,
        epilog=note)
    parser.add_argument("-i", "--input", help="embedding vectors or corpus for word2vec", required=True)
    parser.add_argument("-o", "--output", help="output embedding vectors", required=True)
    parser.add_argument("-v", "--vocab", help="path to vocabulary in JSON or YAML format", required=True)
    parser.add_argument("-w", "--word2vec", help="path to word2vec, assumes text corpus on input")
    parser.add_argument("--dim-emb", help="size of embedding vector, only for training", default=512)
    parser.add_argument("--dim-voc", help= "maximum number of words from vocabulary to be used, default: no limit", type=int)
    parser.add_argument("-t", "--threads", help="number of threads", default=16)
    parser.add_argument("--quiet", help="skip printing warnings", action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
