#!/usr/bin/env python

import sys
import numpy as np;

average = dict()

n = len(sys.argv[1:-1])
for filename in sys.argv[1:-1]:
    print "Loading", filename 
    with open(filename, "rb") as mfile:
        m = np.load(mfile)
        for k in m:
            if k not in average:
                average[k] = 1 / m[k]
            elif average[k].shape == m[k].shape:
                average[k] += 1 / m[k]

for k in average:
    average[k] = n / average[k]

print "Saving to", sys.argv[-1]
np.savez(sys.argv[-1], **average)
