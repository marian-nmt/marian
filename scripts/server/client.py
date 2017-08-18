#!/usr/bin/env python

import sys
import time
from websocket import create_connection

batchSize = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def translate(batch):
    ws = create_connection("ws://localhost:1234/translate")
    #batch = batch[:-1]
    print(batch)
    ws.send(batch)
    result = ws.recv()
    result = result[:-1]
    print(result)
    ws.close()
    #time.sleep(5)


if __name__ == "__main__":
    batchCount = 0
    batch = ""
    for line in sys.stdin:
        #print line
        batchCount = batchCount + 1
        batch += line
        if batchCount == batchSize:
            translate(batch)
            batchCount = 0
            batch = ""

    if batchCount:
        translate(batch)
