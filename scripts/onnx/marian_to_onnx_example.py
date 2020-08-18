# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Example program demonstrating how to convert a Marian model using the marian_to_onnx library
to a self-contained ONNX model that implements greedy search.
"""

import os, sys
import marian_to_onnx as mo

# The following variables would normally be command-line arguments.
# We use constants here to keep it simple. They reflect an example use. You must adjust these.
my_dir = os.path.expanduser("~/young/wngt 2019/")
marian_npz = my_dir + "model.base.npz"            # path to the Marian model to convert
num_decoder_layers = 6                            # number of decoder layers
marian_vocs = [my_dir + "en-de.wl"] * 2           # path to the vocabularies for source and target
onnx_model_path = my_dir + "model.base.opt.onnx"  # resulting model gets written here
quantize_to_bits = 8                              # None for no quantization

# export Marian model as multiple ONNX models
partial_models = mo.export_marian_model_components(marian_npz, marian_vocs)

# quantize if desired
if quantize_to_bits:
    mo.quantize_models_in_place(partial_models, to_bits=quantize_to_bits)

# use the ONNX models in a greedy-search
# The result is a fully self-contained model that implements greedy search.
onnx_model = mo.compose_model_components_with_greedy_search(partial_models, num_decoder_layers)

# save as ONNX file
onnx_model.save(onnx_model_path)

# run a test sentence
w2is = [{ word.rstrip(): id for id, word in enumerate(open(voc_path, "r").readlines()) } for voc_path in marian_vocs]
i2ws = [{ id: tok for tok, id in w2i.items() } for w2i in w2is]
src_tokens = "▁Republican ▁leaders ▁justifie d ▁their ▁policy ▁by ▁the ▁need ▁to ▁combat ▁electoral ▁fraud ▁.".split()
src_ids = [w2is[0][tok] for tok in src_tokens]
print(src_tokens)
print(src_ids)
Y = mo.apply_model(greedy_search_fn=onnx_model,
                   source_ids=src_ids + [w2is[0]["</s>"]],
                   target_eos_id=w2is[1]["</s>"])
print(Y.shape, Y)
tgt_tokens = [i2ws[1][y] for y in Y]
print(" ".join(tgt_tokens))
