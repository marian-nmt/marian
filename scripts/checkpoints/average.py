#!/usr/bin/env python3
"""
This script takes multiple Marian *.npz model files and outputs an elementwise average of the model,
meant to do check-point averaging from: 

https://www.aclweb.org/anthology/W16-2316

usage:

./average.py -m model.1.npz model.2.npz --output model.avg.npz
"""

from __future__ import print_function

import os
import sys
import argparse

import numpy as np

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', nargs='+', required=True,
                    help="models to average")
parser.add_argument('-o', '--output', required=True,
                    help="output path")
args = parser.parse_args()

# *average* holds the model matrix
average = dict()
# No. of models.
n = len(args.model)

for filename in args.model:
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

# Save averaged model to file
print("Saving to {}".format(args.output))
np.savez(args.output, **average)
