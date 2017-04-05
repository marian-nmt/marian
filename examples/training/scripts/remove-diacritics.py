#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Barry Haddow
# Distributed under MIT license

#
# Remove Romanian diacritics. Assumes s-comma and t-comma are normalised

import io
import sys
istream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
ostream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

for line in istream:
  line = line.replace("\u0218", "S").replace("\u0219", "s") #s-comma
  line = line.replace("\u021a", "T").replace("\u021b", "t") #t-comma
  line = line.replace("\u0102", "A").replace("\u0103", "a")
  line = line.replace("\u00C2", "A").replace("\u00E2", "a")
  line = line.replace("\u00CE", "I").replace("\u00EE", "i")
  ostream.write(line)
