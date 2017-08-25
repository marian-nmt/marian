#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Barry Haddow
# Distributed under MIT license

#
# Normalise Romanian s-comma and t-comma

import io
import sys
istream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
ostream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

for line in istream:
  line = line.replace("\u015e", "\u0218").replace("\u015f", "\u0219")
  line = line.replace("\u0162", "\u021a").replace("\u0163", "\u021b")
  ostream.write(line)
