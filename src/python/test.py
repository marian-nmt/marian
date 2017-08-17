#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "../../build"))

import libmariannmt as nmt

print >>sys.stderr, nmt.version()

nmt.init(' '.join(sys.argv))

nmt.translate(["my name is george ."])
# nmt.translate(["that was the second cat ."])
nmt.translate(["that was the second cat .\nit has two sentences ."])
nmt.translate(["The last one ."])

for line in sys.stdin:
    sentences = [line.rstrip()]
    # output = nmt.translate(sentences)
