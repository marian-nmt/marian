#!/usr/bin/env python


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "../../build"))

import libmariannmt as nmt


print >>sys.stderr, nmt.version()

nmt.init(' '.join(sys.argv))

sentences = []
for line in sys.stdin:
    sentences.append(line.rstrip())

output = nmt.translate(sentences)
for line in output:
    print line
