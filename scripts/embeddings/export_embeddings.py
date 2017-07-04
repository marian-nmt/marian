#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import yaml
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
    special = yaml.load(model["special:model.yml"][:-1].tobytes())

    if special["type"] == "amun":
        enc_emb = model["Wemb"]
        dec_emb = model["Wemb_dec"]
    else:
        enc_emb = model["encoder_Wemb"]
        dec_emb = model["decoder_Wemb"]

    with open(args.output_prefix + ".src", "w") as out:
        out.write("{0} {1}\n".format(*enc_emb.shape))
        for i in range(enc_emb.shape[0]):
            vec = " ".join("{0:.8f}".format(v) for v in enc_emb[i])
            out.write("{0} {1}\n".format(i, vec))

    with open(args.output_prefix + ".trg", "w") as out:
        out.write("{0} {1}\n".format(*dec_emb.shape))
        for i in range(dec_emb.shape[0]):
            vec = " ".join("{0:.8f}".format(v) for v in dec_emb[i])
            out.write("{0} {1}\n".format(i, vec))


if __name__ == '__main__':
    main()
