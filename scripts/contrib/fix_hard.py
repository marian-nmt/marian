import numpy as np
import sys

d = dict()
m = np.load(sys.argv[1])
for k in m:
  if "ff_" == k[0:3]:
    d["decoder_" + k] = m[k]
  elif k == "special:model.yml":
   info = m[k].tobytes()
   info = info.replace("layers-dec", "dec-depth")
   info = info.replace("layers-enc", "enc-depth")
   d[k] = info
   print info
  else:
    d[k] = m[k]
np.savez(sys.argv[1] + ".fixed", **d)