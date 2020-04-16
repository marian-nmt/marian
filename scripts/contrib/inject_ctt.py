#!/usr/bin/env python3

from __future__ import print_function

import sys
import argparse
import numpy as np

DESC = "Add 'decoder_c_tt' required by Amun to a model trained with Marian v1.6.0+"


def main():
    args = parse_args()

    print("Loading model {}".format(args.input))
    model = np.load(args.input)

    if "decoder_c_tt" in model:
        print("The model already contains 'decoder_c_tt'")
        exit()

    print("Adding 'decoder_c_tt' to the model")
    amun = {"decoder_c_tt": np.zeros((1, 0))}
    for tensor_name in model:
        amun[tensor_name] = model[tensor_name]

    print("Saving model...")
    np.savez(args.output, **amun)


def parse_args():
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("-i", "--input", help="input model", required=True)
    parser.add_argument("-o", "--output", help="output model", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
