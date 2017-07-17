#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import argparse

import numpy as np

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dropout', type=float, required=True,
                    help="dropout rate")
parser.add_argument('-i', '--input', required=True,
                    help="Input model")
parser.add_argument('-o', '--output', required=True,
                    help="Output model")
args = parser.parse_args()

# Set dropout multiplier.
multiplier = 1.0 - args.dropout
# *output* holds the output matrix that has been "dropped out".
output = dict()


print("Loading {} to multiply with {}".format(args.input, multiplier)

with open(args.input, "rb") as mfile:
    # Loads the matrix from model file.
    m = np.load(mfile)
    for k in m:
        # Initialize the key.
        if "history_errs" in k or "_b" in k or "c_tt" in k:
            output[k] = m[k]
        # Multiply the dropout multipier.
        else:
            output[k] = multiplier * m[k]

# Save the "dropped out" model.
print("Saving to {}".format(args.output))
np.savez(args.output, **output)
