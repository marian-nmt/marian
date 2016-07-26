#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import requests
from clint.textui import progress

BASE_URL = "http://data.statmt.org/rsennrich/wmt16_systems/{}-{}/{}"

CONFIG_TEMPLATE = """
# Paths are relative to config file location
relative-paths: yes

# performance settings
beam-size: 12
devices: [0]
normalize: yes
threads-per-device: 1

# scorer configuration
scorers:
  F0:
    path: ./model.npz
    type: Nematus

# scorer weights
weights:
  F0: 1.0

bpe: ./{}{}.bpe

# vocabularies
source-vocab: ./vocab.{}.json
target-vocab: ./vocab.{}.json
"""

def download_with_progress(path, url):
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size=(1024 ** 2)),
                                  expected_size=(total_length/(1024 ** 2)) + 1):
            if chunk:
                f.write(chunk)
                f.flush()


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
    download_file(src, trg, "{}{}.bpe".format(src, trg), workdir, force)
    download_file(src, trg, "truecase-model.{}".format(src), workdir, force)


def download_file(src, trg, name, workdir, force=False):
    path = os.path.join(workdir, name)
    if not os.path.exists(path):
        full_url = BASE_URL.format(src, trg, name)
        print >> sys.stderr, "Downloading: {} to {}".format(full_url, path)
        download_with_progress(path, full_url)
    elif force:
        full_url = BASE_URL.format(src, trg, name)
        print >> sys.stderr, "Force downloading: {}".format(full_url)
        download_with_progress(path, full_url)
    else:
        print >> sys.stderr, "File {} exists. Skipped".format(path)


def create_base_config(model, model_dir):
    src = model.split('-')[0]
    trg = model.split('-')[1]
    config = CONFIG_TEMPLATE.format(src, trg, src, trg)

    with open("{}/config.yml".format(model_dir), 'w') as config_file:
        config_file.write(config)


def main():
    """ main """
    args = parse_args()
    src = args.model.split('-')[0]
    trg = args.model.split('-')[1]
    workdir = os.path.abspath(args.workdir)
    force = args.force

    try:
        os.makedirs(workdir)
    except OSError:
        pass

    print >> sys.stderr,  "Downloading {} to {}".format(args.model,
                                                        args.workdir)
    download_model(src, trg, workdir, force)
    create_base_config(args.model, workdir)


if __name__ == "__main__":
    main()
