import libamunmt as nmt
import sys

nmt.init(sys.argv[1])

for line in sys.stdin:
    output = nmt.translate(line)
    sys.stdout.write(output)
