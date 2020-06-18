import onnxruntime as ort
import numpy as np
import onnx
import os, sys, time

os.environ['OMP_NUM_THREADS'] = '1'
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

def get_function(path, output_vars):
    print("Reading ONNX function from", path)
    #model = onnx.load(path)
    #print("Done", flush=True)
    #print(model)
    ort_sess = ort.InferenceSession(path, sess_options)
    output_defs = ort_sess.get_outputs()
    for input in ort_sess.get_inputs():
        print("  input: ", input.name, input.shape, input.type)
    for output in output_defs:
        print("  output: ", output.name, output.shape, output.type)
    def invoke_model(**kwargs):
        def to_numpy(val):
            arr = np.array(val)
            if arr.dtype == np.double:
                arr = arr.astype(np.float32)
            elif arr.dtype == np.int64:
                arr = arr.astype(np.int32)
            return arr
        kwargs = { name: to_numpy(val) for name, val in kwargs.items() }
        output_vals = ort_sess.run(None, kwargs)
        output_dict = { output_def.name : output_val for output_val, output_def in zip(output_vals, output_defs) }
        return [output_dict[output_var] for output_var in output_vars]
    return invoke_model

id2word = { id : word.rstrip() for id, word in enumerate(open('c:/work/marian-dev/local/model/vocab_v1.wl', encoding='utf-8').readlines()) }
word2id = { word : id for id, word in id2word.items() }
unk_id = word2id["<unk>"]

model_path_prefix = "c:/work/marian-dev/local/model/model.npz.best-ce-mean-words-debug-sin-uniq-notrans-nounk"
encode_source = get_function(model_path_prefix + '.encode_source.onnx',
                             ['encoder_context_0'])
decode_first  = get_function(model_path_prefix + '.decode_first.onnx',
                             ['first_logits', 'first_decoder_state_0', 'first_decoder_state_1', 'first_decoder_state_2', 'first_decoder_state_3', 'first_decoder_state_4', 'first_decoder_state_5'])
decode_next   = get_function(model_path_prefix + '.decode_next.onnx',
                             ['next_logits', 'next_decoder_state_0', 'next_decoder_state_1', 'next_decoder_state_2', 'next_decoder_state_3', 'next_decoder_state_4', 'next_decoder_state_5'])

def greedy_decode(data_0):
    if len(data_0) == 1:  # special handling for the empty sentence, like Marian
        return data_0
    data_0_mask = [[[1.]]] * len(data_0)
    data_0_index_range = [[[float(t)]] for t in range(len(data_0))]
    #print(data_0, data_0_mask, data_0_index_range)

    max_len = len(data_0) * 3
    Y = []
    encoder_context_0, *_ = encode_source(data_0=data_0, data_0_mask=data_0_mask, data_0_posrange=data_0_index_range)
    logp, *out_decoder_states = decode_first(data_1_posrange=[[[float(0)]]],
                                             encoder_context_0=encoder_context_0, data_0_mask=data_0_mask)
    logp[:,:,:,unk_id] = -1e8  # suppress <unk>, like Marian
    Y.append(np.argmax(logp[0][0]))
    while Y[-1] != 0 and len(Y) < max_len:
        logp, *out_decoder_states = decode_next(prev_word=[Y[-1]], data_1_posrange=[[[float(len(Y))]]],
                                                encoder_context_0=encoder_context_0, data_0_mask=data_0_mask,
                                                decoder_state_0=out_decoder_states[0], decoder_state_1=out_decoder_states[1],
                                                decoder_state_2=out_decoder_states[2], decoder_state_3=out_decoder_states[3],
                                                decoder_state_4=out_decoder_states[4], decoder_state_5=out_decoder_states[5])
        logp[:,:,:,unk_id] = -1e8
        Y.append(np.argmax(logp[0][0]))
    return Y

start_time = time.time()
with open("C:/work/marian-dev/local/model/predictions.out-onnx-debug-sin-notrans-first100-d.tok", 'wt', encoding='utf-8') as out_f:
    for line in open("C:/work/marian-dev/local/model/predictions.in-first100.tok", encoding='utf-8').readlines():
        data = [word2id.get(w, unk_id) for w in (line.rstrip() + " </s>").split(' ') if w]
        Y = greedy_decode(data)
        print("input: ", ' '.join(id2word[x] for x in data))
        print("output:", ' '.join(id2word[y] for y in Y))
        print(' '.join(id2word[y] for y in Y[:-1]), file=out_f, flush=True)  # strip </s> for output to file
print("--- %s seconds ---" % (time.time() - start_time))
