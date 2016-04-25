#!/usr/bin/env python

import os, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', nargs='+', required=True,
                    help="Models to average")
parser.add_argument('-o', '--output', required=True,
                    help="Output path")
args = parser.parse_args()

import numpy as np;

average = dict()
n = len(args.model)
for filename in args.model:
  print "Loading", filename 
  with open(filename, "rb") as mfile:
    m = np.load(mfile)
    for k in m:
      if k != "history_errs":
        if k not in average:
          average[k] = m[k]
        elif average[k].shape == m[k].shape:
          average[k] += m[k]

for k in average:
  average[k] /= n

print "Saving to", args.output
np.savez(args.output, **average)
