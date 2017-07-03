#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import numpy as np


def main():
    desc = """Export word embedding from model"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=desc)
    parser.add_argument("-m", "--model", help="Model file", required=True)
    parser.add_argument(
        "-o", "--output-prefix", help="Output files prefix", required=True)
    args = parser.parse_args()

    print("Loading model")
    model = np.load(args.model)

    with open(args.output_prefix + ".src", "w") as out:
        out.write("{0} {1}\n".format(*model["Wemb"].shape))
        for i in range(model["Wemb"].shape[0]):
            vec = " ".join("{0:.8f}".format(v) for v in model["Wemb"][i])
            out.write("{0} {1}\n".format(i, vec))

    with open(args.output_prefix + ".trg", "w") as out:
        out.write("{0} {1}\n".format(*model["Wemb_dec"].shape))
        for i in range(model["Wemb_dec"].shape[0]):
            vec = " ".join("{0:.8f}".format(v) for v in model["Wemb"][i])
            out.write("{0} {1}\n".format(i, vec))


if __name__ == '__main__':
    main()
