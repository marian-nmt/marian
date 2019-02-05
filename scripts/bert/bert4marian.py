#!/usr/bin/env python3
"""
This script takes a Tensorflow BERT checkpoint and a model description in a JSON file and converts
it to a Marian weight file with numpy weights and an internal YAML description.

This works with checkpoints from https://github.com/google-research/bert

Assmung a BERT checkpoint like this:
drwxr-xr-x 2 marcinjd marcinjd 4.0K Nov 23 16:39 .
-rw-r--r-- 1 marcinjd marcinjd  521 Nov 23 16:38 bert_config.json
-rw-r--r-- 1 marcinjd marcinjd 682M Nov 23 16:39 bert_model.ckpt.data-00000-of-00001
-rw-r--r-- 1 marcinjd marcinjd 8.5K Nov 23 16:39 bert_model.ckpt.index
-rw-r--r-- 1 marcinjd marcinjd 888K Nov 23 16:39 bert_model.ckpt.meta
-rw-r--r-- 1 marcinjd marcinjd 973K Nov 23 16:37 vocab.txt

usage:

./bert.py --bert_prefix bert_model.ckpt  --bert_config bert_config.json --marian bert.npz
"""

import tensorflow as tf
import numpy as np
import sys
import yaml
import argparse

parser = argparse.ArgumentParser(description='Convert Tensorflow BERT model to Marian weight file.')
parser.add_argument('--bert_prefix', help='Prefix for Tensorflow BERT checkpoint', required=True)
parser.add_argument('--bert_config', help='Path to Tensorflow BERT JSON config', required=True)
parser.add_argument('--marian', help='Output path for Marian weight file', required=True)
args = parser.parse_args()

print("Loading TensorFlow config from %s" % (args.bert_config,))
bertConfig = yaml.load(open(args.bert_config))
bertConfigYamlStr = yaml.dump(bertConfig, default_flow_style=False)
print(bertConfigYamlStr)

print("Loading TensorFlow model from %s" % (args.bert_prefix,))

# Collect tensors from TF model as numpy matrices
tfModel = dict()
with tf.Session() as sess:
    preloader = tf.train.import_meta_graph(args.bert_prefix + ".meta")
    preloader.restore(sess, args.bert_prefix)
    vars = tf.global_variables()
    for v in vars:
        if len(v.shape) > 0:
            if "adam" not in v.name: # ignore adam parameters
                print(v.name, v.shape)
                tfModel[v.name] = sess.run(v.name) # get numpy matrix

# Prepare Marian model config
config = dict()
config["type"] = "bert"
config["input-types"] = ["sequence", "class"]
config["tied-embeddings-all"] = True
config["dim-emb"] = tfModel["bert/embeddings/word_embeddings:0"].shape[-1]
config["dim-vocabs"] = [ tfModel["bert/embeddings/word_embeddings:0"].shape[0],
                         tfModel["cls/seq_relationship/output_weights:0"].shape[0] ]

config["transformer-dim-ffn"] = tfModel["bert/encoder/layer_0/intermediate/dense/kernel:0"].shape[-1]
config["transformer-ffn-activation"] = bertConfig["hidden_act"]
config["transformer-ffn-depth"] = 2
config["transformer-heads"] = bertConfig["num_attention_heads"]
config["transformer-train-position-embeddings"] = True
config["transformer-preprocess"] = ""
config["transformer-postprocess"] = "dan"
config["transformer-postprocess-emb"] = "nd"
config["bert-train-type-embeddings"] = True
config["bert-type-vocab-size"] = tfModel["bert/embeddings/token_type_embeddings:0"].shape[0]
config["version"] = "bert4marian.py conversion"

# check number of layers
found = True
config["enc-depth"] = 0;
while found:
    found = False
    for key in tfModel:
        if "bert/encoder/layer_" + str(config["enc-depth"]) in key:
            config["enc-depth"] += 1
            found = True
            break

if config["enc-depth"] != bertConfig["num_hidden_layers"]:
    sys.exit("Number of layers in JSON config (%s) and number of layers found in checkpoint (%s) do not match!" % (config["enc-depth"], bertConfig["num_hidden_layers"]))

configYamlStr = yaml.dump(config, default_flow_style=False)
desc = list(configYamlStr)
npDesc = np.chararray((len(desc),))
npDesc[:] = desc
npDesc.dtype = np.int8

marianModel = dict()
marianModel["special:model.yml"] = npDesc

# Map model weights here #
# Embedding layers
marianModel["Wemb"]  = tfModel["bert/embeddings/word_embeddings:0"]
marianModel["Wpos"]  = tfModel["bert/embeddings/position_embeddings:0"]
marianModel["Wtype"] = tfModel["bert/embeddings/token_type_embeddings:0"]
marianModel["encoder_emb_ln_scale_pre"] = tfModel["bert/embeddings/LayerNorm/gamma:0"]
marianModel["encoder_emb_ln_bias_pre"]  = tfModel["bert/embeddings/LayerNorm/beta:0"]

for layer in range(config["enc-depth"]):
    marianPrefix = "encoder_l%s" % (layer + 1,)
    tfPrefix  = "bert/encoder/layer_%s" % (layer,)

    # Attention
    marianModel[marianPrefix + "_self_Wq"] = tfModel[tfPrefix + "/attention/self/query/kernel:0"]
    marianModel[marianPrefix + "_self_bq"] = tfModel[tfPrefix + "/attention/self/query/bias:0"]

    marianModel[marianPrefix + "_self_Wk"] = tfModel[tfPrefix + "/attention/self/key/kernel:0"]
    marianModel[marianPrefix + "_self_bk"] = tfModel[tfPrefix + "/attention/self/key/bias:0"]

    marianModel[marianPrefix + "_self_Wv"] = tfModel[tfPrefix + "/attention/self/value/kernel:0"]
    marianModel[marianPrefix + "_self_bv"] = tfModel[tfPrefix + "/attention/self/value/bias:0"]

    marianModel[marianPrefix + "_self_Wo"] = tfModel[tfPrefix + "/attention/output/dense/kernel:0"]
    marianModel[marianPrefix + "_self_bo"] = tfModel[tfPrefix + "/attention/output/dense/bias:0"]

    marianModel[marianPrefix + "_self_Wo_ln_scale"] = tfModel[tfPrefix + "/attention/output/LayerNorm/gamma:0"]
    marianModel[marianPrefix + "_self_Wo_ln_bias"]  = tfModel[tfPrefix + "/attention/output/LayerNorm/beta:0"]

    # FFN
    marianModel[marianPrefix + "_ffn_W1"] = tfModel[tfPrefix + "/intermediate/dense/kernel:0"]
    marianModel[marianPrefix + "_ffn_b1"] = tfModel[tfPrefix + "/intermediate/dense/bias:0"]

    marianModel[marianPrefix + "_ffn_W2"] = tfModel[tfPrefix + "/output/dense/kernel:0"]
    marianModel[marianPrefix + "_ffn_b2"] = tfModel[tfPrefix + "/output/dense/bias:0"]

    marianModel[marianPrefix + "_ffn_ffn_ln_scale"] = tfModel[tfPrefix + "/output/LayerNorm/gamma:0"]
    marianModel[marianPrefix + "_ffn_ffn_ln_bias"]  = tfModel[tfPrefix + "/output/LayerNorm/beta:0"]

    # Training objectives
    # Masked-LM output layer
    marianModel["masked-lm_ff_logit_l1_W"] = tfModel["cls/predictions/transform/dense/kernel:0"]
    marianModel["masked-lm_ff_logit_l1_b"] = tfModel["cls/predictions/transform/dense/bias:0"]

    marianModel["masked-lm_ff_ln_scale"] = tfModel["cls/predictions/transform/LayerNorm/gamma:0"]
    marianModel["masked-lm_ff_ln_bias"] = tfModel["cls/predictions/transform/LayerNorm/beta:0"]

    marianModel["masked-lm_ff_logit_l2_b"] = tfModel["cls/predictions/output_bias:0"]

    # Next Sentence classifier
    marianModel["next-sentence_ff_logit_l1_W"] = tfModel["bert/pooler/dense/kernel:0"]
    marianModel["next-sentence_ff_logit_l1_b"] = tfModel["bert/pooler/dense/bias:0"]

    marianModel["next-sentence_ff_logit_l2_W"] = np.transpose(tfModel["cls/seq_relationship/output_weights:0"]) # transpose?!
    marianModel["next-sentence_ff_logit_l2_b"] = tfModel["cls/seq_relationship/output_bias:0"]

print("\nMarian config:")
print(configYamlStr)
print("Saving Marian model to %s" % (args.marian,))
np.savez(args.marian, **marianModel)
