#!/usr/bin/env python

import os, sys
import argparse
import numpy as np;

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dropout', type=float, required=True,
                    help="dropout rate")
parser.add_argument('-i', '--input', required=True,
                    help="Input model")
parser.add_argument('-o', '--output', required=True,
                    help="Output model")
args = parser.parse_args()


multiplier = 1.0 - args.dropout

output = dict()
print "Loading", args.input, "to multiple with", multiplier
with open(args.input, "rb") as mfile:
  m = np.load(mfile)
  for k in m:
    if "history_errs" in k or "_b" in k or "c_tt" in k:
      output[k] = m[k]
    else:
      output[k] = multiplier * m[k]

print "Saving to", args.output
np.savez(args.output, **output)
