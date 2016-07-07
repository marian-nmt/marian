#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import urllib
import sys
import os

BASE_URL = "http://statmt.org/rsennrich/wmt16_systems/{}-{}/{}"


def parse_args():
    """ parse command arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", dest="workdir", default='.')
    parser.add_argument('-m', dest="model", default='en-de')
    parser.add_argument('-f', dest="force", default=False)
    return parser.parse_args()


def download_model(src, trg, workdir, force=False):
    """ download Rico Sennrich's WMT16 model: <src> to <trg>. """
    download_file(src, trg, "model.npz", workdir, force)
    download_file(src, trg, "vocab.{}.json".format(src), workdir, force)
    download_file(src, trg, "vocab.{}.json".format(trg), workdir, force)


def download_file(src, trg, name, workdir, force=False):
    path = os.path.join(workdir, name)
    if not os.path.exists(path):
        full_url = BASE_URL.format(src, trg, name)
        print >> sys.stderr, "Downloading: {} to {}".format(full_url, path)
        urllib.urlretrieve(full_url, path)
    elif force:
        full_url = BASE_URL.format(src, trg, name)
        print >> sys.stderr, "Force downloading: {}".format(full_url)
        urllib.urlretrieve(full_url, path)
    else:
        print >> sys.stderr, "File {} exists. Skipped".format(path)


def main():
    """ main """
    args = parse_args()
    src = args.model.split('-')[0]
    trg = args.model.split('-')[1]
    workdir = os.path.realpath(args.workdir)
    force = args.force

    print >> sys.stderr,  "Downloading {} to {}".format(args.model,
                                                        args.workdir)
    download_model(src, trg, workdir, force)


if __name__ == "__main__":
    main()
