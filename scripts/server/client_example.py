#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division

import sys
import time
import argparse

# pip install websocket_client  
from websocket import create_connection


if __name__ == "__main__":
    # handle command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    # open connection
    ws = create_connection("ws://localhost:{}/translate".format(args.port))

    count = 0
    batch = ""
    for line in sys.stdin:
        count += 1
        batch += line.decode('utf-8') if sys.version_info < (3, 0) else line
        if count == args.batch_size:
            # translate the batch
            ws.send(batch)
            result = ws.recv()
            print(result.rstrip())

            count = 0
            batch = ""

    if count:
        # translate the remaining sentences
        ws.send(batch)
        result = ws.recv()
        print(result.rstrip())

    # close connection
    ws.close()
