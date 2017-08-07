#!/usr/bin/env python


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "../../build/src/python"))

import libmariannmt as nmt


print >>sys.stderr, nmt.version()

nmt.init("" if len(sys.argv) < 2 else ' '.join(sys.argv[1:]))

sentences = []
for line in sys.stdin:
    sentences.append(line.rstrip())

    output = nmt.translate(sentences)

    for line in output:
        print line
