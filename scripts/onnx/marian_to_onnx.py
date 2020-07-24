# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Library for converting certain types of Marian models to a standalone ONNX model.

Because Marian and ONNX use very different philosophies, a conversion is not possible
for all possible Marian models. Specifically, currently we don't support recurrent
networks in the encoder.

This works by running a Marian decode for 2 output steps, and capturing pieces of
this graph that correspond to the encoder, the first decoding steps, and the second
decoding step. The graph of the second decoding step can be applied repeatedly in
order to decoder a variable-length sequence.

Prerequisites:
```
pip install onnxruntime
git clone https://github.com/microsoft/onnxconverter-common.git
```
"""

import os, sys, inspect, subprocess
from typing import List, Dict, Optional, Callable

# get the Marian root path
_marian_root_path = os.path.dirname(inspect.getfile(inspect.currentframe())) + "/../.."

# we assume onnxconverter-common to be available next to the marian-dev repo; you must adjust this if needed
sys.path.append(_marian_root_path + "/../onnxconverter-common")
from onnxconverter_common.onnx_fx import Graph
from onnxconverter_common.onnx_fx import GraphFunctionType as _Ty
import onnxruntime as _ort
def _ort_apply_model(model, inputs):  # ORT execution is a callback so that Graph itself does not need to depend on ORT
    sess = _ort.InferenceSession(model.SerializeToString())
    return sess.run(None, inputs)
Graph.inference_runtime = _ort_apply_model
Graph.opset = 11


def export_marian_model_components(marian_model_path: str, marian_vocab_paths: List[str],
                                   marian_executable_path: Optional[str]=None) -> Dict[str,Graph]:
    """
    Export the Marian graph to a set of models.

    Args:
        marian_model_path: path to Marian model to convert
        marian_vocab_paths: paths of vocab files (normally, this requires 2 entries, which may be identical)
        marian_executable_path: path to Marian executable; will default to THIS_SCRIPT_PATH/../../build/marian
    Returns:
        Dict of ONNX Graph instances corresponding to pieces of Marian models.
    """
    assert isinstance(marian_vocab_paths, list), "marian_vocab_paths must be a list of paths"
    # default marian executable is found relative to location of this script (Linux/CMake only)
    if marian_executable_path is None:
        marian_executable_path = _marian_root_path + "/build/marian"
    # partial models are written to /tmp
    output_path_stem = "/tmp/" + os.path.basename(marian_model_path)
    # exporting is done via invoking Marian via its command-line interface; models are written to tmp files
    command = marian_executable_path
    args = [
        "convert",
        "--from", marian_model_path,
        "--vocabs", *marian_vocab_paths,
        "--to", output_path_stem,
        "--export-as", "onnx-encode"
    ]
    subprocess.run([command] + args, check=True)
    # load the tmp files into Python bytes objects
    graph_names = ["encode_source", "decode_first", "decode_next"]                                # Marian generates graphs with these names
    output_paths = [output_path_stem + "." + graph_name + ".onnx" for graph_name in graph_names]  # form pathnames under which Marian wrote the files
    res = { graph_name: Graph.load(output_path) for graph_name, output_path in zip(graph_names, output_paths) }
    # clean up after ourselves
    for output_path in output_paths:
        os.unlink(output_path)
    return res


def combine_model_components_with_greedy_search(partial_models: Dict[str,bytes], num_decoder_layers: int):
    """
    Create an ONNX model that implements greedy search over the exported Marian pieces.

    Args:
        partial_models: models returned from export_marian_model_components()
        num_decoder_layers: must be specified, since it cannot be inferred from the model files presently (e.g. 6)
    Returns:
        ONNX model that can be called as
        result_ids = greedy_search_fn(np.array(source_ids, dtype=np.int64), np.array([target_eos_id], dtype=np.int64))[0]
    """
    # load our partial functions
    # ONNX graph inputs and outputs are named but not ordered. Therefore, we must define the parameter order here.
    def define_parameter_order(graph, inputs, outputs):
        tmppath = "/tmp/tmpmodel.onnx"
        graph.save(tmppath)  # unfortunately, Graph.load() cannot load from bytes, so use a tmp file
        graph = Graph.load(tmppath, inputs=inputs, outputs=outputs)
        os.unlink(tmppath)
        return graph
    encode_source = define_parameter_order(partial_models["encode_source"],
                                           inputs=['data_0', 'data_0_mask', 'data_0_posrange'],  # define the order of arguments
                                           outputs=['encoder_context_0'])
    decode_first = define_parameter_order(partial_models["decode_first"],
                                          inputs=['data_1_posrange', 'encoder_context_0', 'data_0_mask'],
                                          outputs=['first_logits'] +
                                                  [f"first_decoder_state_{i}" for i in range(num_decoder_layers)])
    decode_next = define_parameter_order(partial_models["decode_next"],
                                         inputs=['prev_word', 'data_1_posrange', 'encoder_context_0', 'data_0_mask'] +
                                                [f"decoder_state_{i}" for i in range(num_decoder_layers)],
                                         outputs=['next_logits'] +
                                                 [f"next_decoder_state_{i}" for i in range(num_decoder_layers)])

    # create an ONNX graph that implements full greedy search
    # The greedy search is implemented via the @Graph.trace decorator, which allows us to
    # author the greedy search in Python, similar to @CNTK.Function and PyTorch trace-based jit.
    # The decorator executes greedy_search() below on a dummy input in order to generate an ONNX graph
    # via invoking operators from the onnx.fx library.
    # The partial functions exported from Marian are invoked (=inlined) by this.
    # The result is a full ONNX graph that implements greedy search using the Marian model.
    @Graph.trace(
        input_types=[_Ty.I(shape=['N']), _Ty.I([1])],
        output_types=[_Ty.I(shape=['T'])],
        outputs="Y")
    def greedy_search(X, eos_id):
        """
        Args:
            X: sequence of input tokens, including EOS symbol, as integer indices into the input vocabulary
            eos_id: id of the EOS symbol in the output vocabulary
        """
        ox = X.ox
        data_0 = X
        data_0_shape = data_0.shape()
        data_0_mask = ox.constant_of_shape(data_0_shape, value=1.0)
        seq_len = data_0_shape[-1]
        data_0_index_range = ox.range([ox.constant(value=0), seq_len, ox.constant(value=1)]).cast(to=ox.float)
        data_0_index_range = ox.unsqueeze(data_0_index_range, axes=[1, 2])
        max_len = seq_len * 3

        encoder_context_0 = encode_source(data_0=data_0, data_0_mask=data_0_mask,
                                          data_0_posrange=data_0_index_range)

        y_len_0 = ox.constant(value=0.0)
        logp, *out_decoder_states = decode_first(data_1_posrange=y_len_0,
                                                 encoder_context_0=encoder_context_0, data_0_mask=data_0_mask)

        y_t = logp[0, 0, 0].argmax(axis=-1, keepdims=True)  # note: rank-1 tensor, not a scalar
        eos_token = eos_id + 0
        test_y_t = (y_t != eos_token)

        @Graph.trace(outputs=['ty_t', 'y_t_o', *(f'ods_{i}' for i in range(num_decoder_layers)), 'y_t_o2'],
                    output_types=[_Ty.b, _Ty.i] + [_Ty.f] * 6 + [_Ty.i],
                    input_types=[_Ty.I([1]), _Ty.b, _Ty.i] + [_Ty.f] * num_decoder_layers)
        def loop_body(iteration_count, condition,  # these are not actually used inside
                    y_t,
                    out_decoder_states_0, out_decoder_states_1, out_decoder_states_2, out_decoder_states_3, out_decoder_states_4, out_decoder_states_5):
            # Currently, we do not support variable number of arguments to the callable.
            # @TODO: We have the information from the type signature in Graph.trace(), so this should be possible.
            assert num_decoder_layers == 6, "Currently, decoder layers other than 6 require a manual code change"
            out_decoder_states = [out_decoder_states_0, out_decoder_states_1, out_decoder_states_2, out_decoder_states_3, out_decoder_states_4, out_decoder_states_5]
            """
            Loop body follows the requirements of ONNX Loop:

            "The graph run each iteration.
            It has 2+N inputs: (iteration_num, condition, loop carried dependencies...).
            It has 1+N+K outputs: (condition, loop carried dependencies..., scan_outputs...).
            Each scan_output is created by concatenating the value of the specified output value at the end of each iteration of the loop.
            It is an error if the dimensions or data type of these scan_outputs change across loop iterations."

            Inputs:
                iteration_num (not used by our function)
                test_y_t: condition (not used as an input)
                y_t, *out_decoder_states: N=(num_decoder_layers+1) loop-carried dependencies

            Outputs:
                test_y_t: condition, return True if there is more to decode
                y_t, *out_decoder_states: N=(num_decoder_layers+1) loop-carried dependencies (same as in the Inputs section)
                y_t: K=1 outputs
            """
            pos = iteration_count + 1
            data_1_posrange = pos.cast(to=1).unsqueeze(axes=[0, 1, 2])
            logp, *out_decoder_states = decode_next(
                prev_word=y_t, data_1_posrange=data_1_posrange,
                encoder_context_0=encoder_context_0, data_0_mask=data_0_mask,
                **{f"decoder_state_{i}": out_decoder_states[i] for i in range(len(out_decoder_states))})
            y_t = logp[0, 0, 0].argmax(axis=-1, keepdims=True)
            test_y_t = (y_t != eos_token)
            return [test_y_t, y_t] + out_decoder_states + [y_t]

        # "Final N loop carried dependency values then K scan_outputs"
        ret_vals = ox.loop(max_len, test_y_t, loop_body,
                           inputs=[y_t] + out_decoder_states,
                           outputs=['gy_t_o', *[f"gods_{i}" for i in range(len(out_decoder_states))], 'greedy_out'])
        y = ret_vals[-1]  # scan_output

        # we must prepend the very first token
        Y = ox.concat([ox.unsqueeze(y_t), y], axis=0)  # note: y_t are rank-1 tensors, not scalars (ORT concat fails with scalars)
        return ox.squeeze(Y, axes=[1])
    greedy_search.to_model()  # this triggers the model tracing (which is lazy)
    return greedy_search


def apply_model(greedy_search_fn: Callable, source_ids: List[int], target_eos_id: int) -> List[int]:
    """
    Apply model to an input sequence, e.g. run translation.

    Args:
        greedy_search_fn: ONNX model created with combine_model_components_with_greedy_search()\
        source_ids: list of source tokens, as indices into soure vocabulary, ending in EOS symbol
        target_eos_id: id of EOS symbol in target vocabulary
    Returns:
        Result as list of ids into target vocabulary
    """
    import numpy as np
    Y = greedy_search_fn(
        np.array(source_ids, dtype=np.int64),
        np.array([target_eos_id], dtype=np.int64))[0]
    return Y
