import sys
import math
from collections import Counter

c = Counter()
N = 0
for line in sys.stdin:
    uniq = set(line.split())
    for word in uniq:
       c[word] += 1
    N += 1

keys = sorted([k for k in c])
for word in keys:
    idf = math.log(float(N) / float(c[word])) / math.log(N)
    print word, ":", idf
