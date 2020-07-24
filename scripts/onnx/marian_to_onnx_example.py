# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Example program demonstrating how to convert a Marian model using the marian_to_onnx library
to a self-contained ONNX model that implements greedy search.
"""

import os, sys
import marian_to_onnx as mo

# The following variables would normally be command-line arguments.
# We use constants here to keep it simple. Please just adjust these as needed.
my_dir = os.path.expanduser("~/")
marian_npz = my_dir + "model.npz.best-ce-mean-words.npz"        # path to the Marian model to convert
num_decoder_layers = 6                                          # number of decoder layers
marian_vocs = [my_dir + "vocab_v1.wl"] * 2                      # path to the vocabularies for source and target
onnx_model_path = my_dir + "model.npz.best-ce-mean-words.onnx"  # resulting model gets written here

# export Marian model as multiple ONNX models
partial_models = mo.export_marian_model_components(marian_npz, marian_vocs)

# use the ONNX models in a greedy-search
# The result is a fully self-contained model that implements greedy search.
onnx_model = mo.compose_model_components_with_greedy_search(partial_models, num_decoder_layers)

# save as ONNX file
onnx_model.save(onnx_model_path)

# run a test sentence
Y = mo.apply_model(greedy_search_fn=onnx_model,
                   source_ids=[274, 35, 52, 791, 59, 4060, 6, 2688, 2, 7744, 9, 2128, 7, 2, 4695, 9, 950, 2561, 3, 0],
                   target_eos_id=0)
print(Y.shape, Y)
