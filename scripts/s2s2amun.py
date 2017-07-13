#!/usr/bin/python

from __future__ import print_function

import argparse

import numpy as np

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--s2s', required=True,
                    help="S2S Model")
parser.add_argument('-o', '--amun', required=True,
                    help="Amun Model")
args = parser.parse_args()

# Mappings from Seq2Seq to Amun.
mapping = { "decoder_cell1_U":  "decoder_U",
            "decoder_cell1_Ux": "decoder_Ux",
            "decoder_cell1_W":  "decoder_W",
            "decoder_cell1_Wx": "decoder_Wx",
            "decoder_cell1_b":  "decoder_b",
            "decoder_cell1_bx": "decoder_bx",
            "decoder_cell1_gamma1": "decoder_cell1_gamma1",
            "decoder_cell1_gamma2": "decoder_cell1_gamma2",

            "decoder_U_att": "decoder_U_att",
            "decoder_W_comb_att": "decoder_W_comb_att",
            "decoder_Wc_att": "decoder_Wc_att",
            "decoder_att_gamma1": "decoder_att_gamma1",
            "decoder_att_gamma2": "decoder_att_gamma2",
            "decoder_b_att": "decoder_b_att",

            "decoder_cell2_U":  "decoder_U_nl",
            "decoder_cell2_Ux": "decoder_Ux_nl",
            "decoder_cell2_W":  "decoder_Wc",
            "decoder_cell2_Wx": "decoder_Wcx",
            "decoder_cell2_b":  "decoder_b_nl",
            "decoder_cell2_bx": "decoder_bx_nl",
            "decoder_cell2_gamma1": "decoder_cell2_gamma1",
            "decoder_cell2_gamma2": "decoder_cell2_gamma2",

            "decoder_ff_logit_l1_W0": "ff_logit_prev_W",
            "decoder_ff_logit_l1_W1": "ff_logit_lstm_W",
            "decoder_ff_logit_l1_W2": "ff_logit_ctx_W",
            "decoder_ff_logit_l1_b0": "ff_logit_prev_b",
            "decoder_ff_logit_l1_b1": "ff_logit_lstm_b",
            "decoder_ff_logit_l1_b2": "ff_logit_ctx_b",
            "decoder_ff_logit_l1_gamma0": "ff_logit_l1_gamma0",
            "decoder_ff_logit_l1_gamma1": "ff_logit_l1_gamma1",
            "decoder_ff_logit_l1_gamma2": "ff_logit_l1_gamma2",
            "decoder_ff_logit_l2_W": "ff_logit_W",
            "decoder_ff_logit_l2_b": "ff_logit_b",
            "decoder_ff_state_W": "ff_state_W",
            "decoder_ff_state_b": "ff_state_b",
            "decoder_ff_state_gamma": "ff_state_gamma",

            "decoder_Wemb": "Wemb_dec",
            "encoder_Wemb": "Wemb",

            "encoder_bi_U": "encoder_U",
            "encoder_bi_Ux": "encoder_Ux",
            "encoder_bi_W": "encoder_W",
            "encoder_bi_Wx": "encoder_Wx",
            "encoder_bi_b": "encoder_b",
            "encoder_bi_bx": "encoder_bx",
            "encoder_bi_gamma1": "encoder_gamma1",
            "encoder_bi_gamma2": "encoder_gamma2",
            "encoder_bi_r_U": "encoder_r_U",
            "encoder_bi_r_Ux": "encoder_r_Ux",
            "encoder_bi_r_W": "encoder_r_W",
            "encoder_bi_r_Wx": "encoder_r_Wx",
            "encoder_bi_r_b": "encoder_r_b",
            "encoder_bi_r_bx": "encoder_r_bx",
            "encoder_bi_r_gamma1": "encoder_r_gamma1",
            "encoder_bi_r_gamma2": "encoder_r_gamma2" }


# Loads the Seq2Seq model.
print("[s2s2amun] Loading s2s model {}".format(args.s2s))
s2s_model = np.load(args.s2s)
# *amun_model* holds the output model.
amun_model = dict()

for tensor_name in s2s_model:
    # Substitute the mapping.
    if tensor_name in mapping:
        amun_model[ mapping[ tensor_name ]] = s2s_model[ tensor_name ]
    # Otherwise, notify user of unknown tensor.
    else:
        print("[s2s2amun] unknown: {}".format(tensor_name))

decoder_c_tt = np.array([0])
amun_model[ "decoder_c_tt" ] = decoder_c_tt

# Saves the Amun model.
print("[s2s2amun] Saving amun model: {}".format(args.amun))
np.savez(args.amun, **amun_model)
