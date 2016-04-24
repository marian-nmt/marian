#!/usr/bin/env python

import sys
import numpy as np;

average = dict()

for filename in sys.argv[1:-1]:
    with open(filename, "rb") as mfile:
        m = np.load(mfile)
        for k in m:
            if k not in average:
                average[k] = m[k]
            elif average[k].shape == m[k].shape:
                average[k] += m[k]

for k in average:
    average[k] /= len(sys.argv[1:-1])

np.savez(sys.argv[-1], **average)
