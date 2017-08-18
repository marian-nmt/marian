#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "../../build"))

import libmariannmt as nmt

print >>sys.stderr, "marian-nmt version: ", nmt.version()

nmt.init(' '.join(sys.argv))

for line in sys.stdin:
    print nmt.translate([line.rstrip()])
