#!/usr/bin/env python

import sys
import numpy as np;

average = dict()

n = len(argv[1:-1])
for filename in sys.argv[1:-1]:
    print "Loading", filename 
    with open(filename, "rb") as mfile:
        m = np.load(mfile)
        for k in m:
            if k not in average:
                average[k] = m[k] ** 1.0/n
            elif average[k].shape == m[k].shape:
                average[k] *= m[k] ** 1.0/n

print "Saving to", sys.argv[-1]
np.savez(sys.argv[-1], **average)
