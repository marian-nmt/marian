#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import subprocess as sp
import os

CONFIG_TEMPLATE = """
# Paths are relative to config file location
relative-paths: no

# performance settings
beam-size: 12
devices: [0]
normalize: yes
threads-per-device: 1

# scorer configuration
scorers:
  F0:
    path: {}/model.npz
    type: Nematus

# scorer weights
weights:
  F0: 1.0

# vocabularies
source-vocab: {}/vocab.{}.json
target-vocab: {}/vocab.{}.json
"""

AMUNMT_PATH = "/opt/amunmt"
AMUNMT_PATH = "/home/tomaszd/codes/amunmt"


def download_models(model, model_dir):
    print >> sys.stderr, "Model dir:", model_dir
    try:
        print >> sys.stderr, "Create directory:", model_dir
        os.makedirs(model_dir)
    except OSError as e:
        print >> sys.stderr, "ERROR:", e.strerror
    command = ' '.join(['{}/scripts/download_models.py'.format(AMUNMT_PATH),
                        '-w ', model_dir, '-m ', model])
    print >> sys.stderr, "Command: ", command
    sp.call(command, shell=True)


def create_config(model, model_dir):
    src = model.split('-')[0]
    trg = model.split('-')[1]
    config = CONFIG_TEMPLATE.format(model_dir, model_dir, src, model_dir, trg)

    with open("{}/config.yml".format(model_dir), 'w') as config_file:
        config_file.write(config)


def run_amunmt(model_dir):
    while True:
        command = ' '.join(['PYTHONPATH={}/release/src'.format(AMUNMT_PATH),
                 'python', '{}/run.py'.format(AMUNMT_PATH),
                 '{}/config.yml'.format(model_dir)])
        print >> sys.stderr, "Running amuNMT: ", command
        sp.call(command, shell=True)


def main():
    """ main """
    print sys.argv
    model = sys.argv[1]
    model_dir = '/model/amunmt/{}'.format(model)
    model_dir = '/home/tomaszd/{}'.format(model)

    download_models(model, model_dir)
    create_config(model, model_dir)
    run_amunmt(model_dir)


if __name__ == "__main__":
    main()
