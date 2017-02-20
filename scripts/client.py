#!/usr/bin/env python

from websocket import create_connection
import time
import sys

filePath = sys.argv[1]
batchSize = int(sys.argv[2])
#print filePath
#print batchSize

with open(filePath) as f:
  batchCount = 0
  batch = ""
  for line in f:
    #print line
    batchCount = batchCount + 1
    batch = batch + line 
    if batchCount == batchSize:
      ws = create_connection("ws://localhost:8080/translate")

      batch = batch[:-1]
      #print batch
      ws.send(batch)
      result=ws.recv()
      result = result[:-1]
      print(result)
      ws.close()
      #time.sleep(5)

      batchCount = 0
      batch = ""

