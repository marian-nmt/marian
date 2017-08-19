#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "../../build"))
import libmariannmt as nmt

print >>sys.stderr, "marian-nmt version: ", nmt.version()

if len(sys.argv) == 1:
    print >>sys.stderr, "Specify s2s arguments"
    exit(1)

nmt.init(' '.join(sys.argv))
for line in sys.stdin:
    print nmt.translate([line.rstrip()])
