#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import yaml
import numpy as np


def main():
    desc = """Export word embeddings from model"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=desc)
    parser.add_argument("-m", "--model", help="path to model.npz file", required=True)
    parser.add_argument("-o", "--output-prefix", help="prefix for output files", required=True)
    args = parser.parse_args()

    print("Loading model")
    model = np.load(args.model)
    special = yaml.load(model["special:model.yml"][:-1].tobytes())

    if special["tied-embeddings-all"] or special["tied-embeddings-src"]:
        all_emb = model["Wemb"]
        export_emb(args.output_prefix + ".all", all_emb)
        exit()

    if special["type"] == "amun":
        enc_emb = model["Wemb"]
        dec_emb = model["Wemb_dec"]
    else:
        enc_emb = model["encoder_Wemb"]
        dec_emb = model["decoder_Wemb"]

    export_emb(args.output_prefix + ".src", enc_emb)
    export_emb(args.output_prefix + ".trg", dec_emb)


def export_emb(filename, emb):
    with open(filename, "w") as out:
        out.write("{0} {1}\n".format(*emb.shape))
        for i in range(emb.shape[0]):
            vec = " ".join("{0:.8f}".format(v) for v in emb[i])
            out.write("{0} {1}\n".format(i, vec))


if __name__ == '__main__':
    main()
