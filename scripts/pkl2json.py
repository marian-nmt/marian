#!/usr/bin/env python

import sys
import json

try:
    import cPickle as pickle
except ImportError:
    import pickle

with open(sys.argv[1], 'r') as fin:
    d = pickle.load(fin)
    json.dump(d, sys.stdout)
