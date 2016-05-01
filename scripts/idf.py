import sys
import math
import yaml
from collections import Counter

c = Counter()
N = 0
for line in sys.stdin:
    uniq = set(line.split())
    for word in uniq:
       c[word] += 1
    N += 1

out = dict()
for word in c:
    idf = math.log(float(N) / float(c[word])) / math.log(N)
    out[word] = idf

yaml.safe_dump(out, sys.stdout, default_flow_style=False, allow_unicode=True)