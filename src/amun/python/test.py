#!/usr/bin/env python

import libamunmt as nmt
import sys

nmt.init(sys.argv[1])

sentences = []
for line in sys.stdin:
    sentences.append(line.rstrip())

output = nmt.translate(sentences)

for line in output:
    sys.stdout.write(line)
