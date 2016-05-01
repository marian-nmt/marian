#!/usr/bin/env python
import sys
import cPickle
import yaml
import operator

d = cPickle.load(open(sys.argv[1], 'r'))
yaml.safe_dump(d, sys.stdout,
               default_flow_style=False, allow_unicode=True)