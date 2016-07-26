#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import re
import sys

for line in sys.stdin:
    m = re.search(ur'<seg id="\d+">(.*)</seg>', line)
    if m:
        print m.group(1)
