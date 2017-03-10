#!/usr/bin/env python
# script by Ulrich Germann
# This script is meant to test the python interface of amun by emulating the amun executable.
import sys, os

if 'AMUN_PYLIB_DIR' in os.environ:
    sys.path.append(os.environ['AMUN_PYLIB_DIR'])
    pass

import libamunmt
if __name__ == "__main__":
    libamunmt.init(" ".join(sys.argv[1:]))
    print libamunmt.translate(sys.stdin.readlines())
    libamunmt.shutdown()
    
