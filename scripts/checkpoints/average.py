#!/usr/bin/env python3
"""
This script takes multiple Marian *.npz model files and outputs an elementwise
average of the model, meant to do check-point averaging from:

https://www.aclweb.org/anthology/W16-2316

usage examples:

./average.py -m model.1.npz model.2.npz --output model.avg.npz
./average.py --from-log train.log 2 chrf --output model.avg.npz
"""

from __future__ import print_function

import argparse
import numpy as np
import os
import re
import sys


def main():
    args = parse_cmd_args()

    if args.from_log:
        models = find_best_models(*args.from_log)
    else:
        models = args.model

    print("Averaging models: {}".format(" ".join(models)))
    average = average_models(models)

    # Save averaged model to file
    print("Saving to {}".format(args.output))
    np.savez(args.output, **average)


def average_models(models):
    average = dict()  # Holds the model matrix
    n = len(models)  # No. of models.

    for filename in models:
        print("Loading {}".format(filename))
        with open(filename, "rb") as mfile:
            # Loads matrix from model file
            m = np.load(mfile)
            for k in m:
                if k != "history_errs":
                    # Initialize the key
                    if k not in average:
                        average[k] = m[k]
                    # Add to the appropriate value
                    elif average[k].shape == m[k].shape and "special" not in k:
                        average[k] += m[k]

    # Actual averaging
    for k in average:
        if "special" not in k:
            average[k] /= n

    return average


def find_best_models(logs, best=5, metric='chrf', order=None):
    best = int(best)
    if order is None:  # Try to set ordering automatically
        order = 'asc' if metric == 'perplexity' else 'desc'
    print(
        "Taking {} best checkpoints according to {}/{} from {}".format(
            best, metric, order, logs
        )
    )

    match_model = re.compile(
        r'Saving model weights and runtime parameters to (?P<model>.*\.iter\d+\.npz)'
    )
    match_valid = re.compile(
        r'\[valid\] Ep\. [\d\.]+ : '
        r'Up\. (?P<update>[\d\.]+) : '
        r'(?P<metric>[^ ]+) : '
        r'(?P<value>[\d\.]+) :'
    )
    # Find model.iterXYZ.npz files and validation scores
    lines = []  # [(checkpoint, update, { metric: value })]
    with open(logs, "r") as logfile:
        for line in logfile:
            m = match_model.search(line)
            if m:
                model = m.group("model")
                lines.append([model, None, {}])
                continue
            m = match_valid.search(line)
            if m:
                update = m.group("update")
                name = m.group("metric")
                value = float(m.group("value"))
                if metric not in lines[-1][-1]:
                    lines[-1][1] = update
                    lines[-1][-1][name] = value

    # Check if the requested metric is found
    metrics = lines[0][-1].keys()
    if metric not in metrics:
        raise ValueError(
            "metric '{}' not found in {}, choose from: {}".format(
                metric, logs, " ".join(metrics)
            )
        )
        exit(1)

    # Select best N checkpoints
    models_all = [(line[0], line[2][metric]) for line in lines]
    reverse = True if order.lower() == 'desc' else False
    models_top = sorted(models_all, key=lambda p: p[1], reverse=reverse)[:best]

    print("Selected checkpoints:")
    for model, value in models_top:
        print("  {} {}={:.4f}".format(model, metric, value))

    return [p[0] for p in models_top]


def parse_cmd_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', nargs='+', help="models to average")
    parser.add_argument(
        '--from-log',
        nargs='+',
        metavar="ARG",
        help="average from train.log, args: path N metric",
    )
    parser.add_argument('-o', '--output', required=True, help="output path")
    args = parser.parse_args()

    if (not args.model and not args.from_log) or (args.model and args.from_log):
        parser.error('either -m/--model or --from-log must be set')
    return args


if __name__ == "__main__":
    main()
