#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import yaml


DESC = "Prints version and model type from model.npz file."
S2S_SPECIAL_NODE = "special:model.yml"


def main():
    args = parse_args()

    model = np.load(args.model)
    if S2S_SPECIAL_NODE not in model:
        print("No special Marian YAML node found in the model")
        exit(1)

    yaml_text = bytes(model[S2S_SPECIAL_NODE]).decode('ascii')
    if not args.key:
        print(yaml_text)
        exit(0)

    # fix the invalid trailing unicode character '#x0000' added to the YAML
    # string by the C++ cnpy library
    try:
        yaml_node = yaml.load(yaml_text)
    except yaml.reader.ReaderError:
        yaml_node = yaml.load(yaml_text[:-1])

    print(yaml_node[args.key])


def parse_args():
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("-m", "--model", help="model file", required=True)
    parser.add_argument("-k", "--key", help="print value for specific key")
    return parser.parse_args()


if __name__ == "__main__":
    main()
