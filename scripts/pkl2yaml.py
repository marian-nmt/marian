#!/usr/bin/env python

import sys
import yaml

try:
    import cPickle as pickle
except ImportError:
    import pickle

with open(sys.argv[1], 'r') as fin:
    d = pickle.load(fin)
    yaml.safe_dump(d, sys.stdout,
                   default_flow_style=False,
                   allow_unicode=True)
