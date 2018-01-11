#!/usr/bin/env python

from __future__ import print_function, unicode_literals, division

import sys
import time
import argparse

from websocket import create_connection


def translate(batch, port=8080):
    ws = create_connection("ws://localhost:{}/translate".format(port))
    #print(batch.rstrip())
    ws.send(batch)
    result = ws.recv()
    print(result.rstrip())
    ws.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-p", "--port", type=int, default=8080)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    count = 0
    batch = ""
    for line in sys.stdin:
        count += 1
        batch += line.decode('utf-8') if sys.version_info < (3, 0) else line
        if count == args.batch_size:
            translate(batch, port=args.port)
            count = 0
            batch = ""

    if count:
        translate(batch, port=args.port)
