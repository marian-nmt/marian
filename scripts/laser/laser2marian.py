import numpy as np
import sys
import yaml
import argparse

import torch

parser = argparse.ArgumentParser(description='Convert LASER model to Marian weight file.')
parser.add_argument('--laser', help='Path to LASER PyTorch model', required=True)
parser.add_argument('--marian', help='Output path for Marian weight file', required=True)
args = parser.parse_args()

laser = torch.load(args.laser)
    
config = dict()
config["type"] = "laser"
config["input-types"] = ["sequence"]
config["dim-vocabs"] = [laser["params"]["num_embeddings"]]

config["version"] = "laser2marian.py conversion"

config["enc-depth"] = laser["params"]["num_layers"]
config["enc-cell"] = "lstm"
config["dim-emb"] = laser["params"]["embed_dim"]
config["dim-rnn"] = laser["params"]["hidden_size"]

yaml.dump(laser["dictionary"], open(args.marian + ".vocab.yml", "w"))

marianModel = dict()

def transposeOrder(mat):
    matT = np.transpose(mat) # just a view with changed row order
    return matT.flatten(order="C").reshape(matT.shape) # force row order change and reshape
    
def convert(pd, srcs, trg, transpose=True, bias=False, lstm=False):
    num = pd[srcs[0]].detach().numpy()
    for i in range(1, len(srcs)):
        num += pd[srcs[i]].detach().numpy()

    out = num
    if bias:
        num = np.atleast_2d(num)
    else:
        if transpose:
            num = transposeOrder(num) # transpose with row order change
        
    if lstm: # different order in pytorch than marian
        stateDim = int(num.shape[-1] / 4)
        i = np.copy(num[:, 0*stateDim:1*stateDim])
        f = np.copy(num[:, 1*stateDim:2*stateDim])
        num[:, 0*stateDim:1*stateDim] = f
        num[:, 1*stateDim:2*stateDim] = i

    marianModel[trg] = num

for k in laser:
    print(k)

for k in laser["model"]:
    print(k, laser["model"][k].shape)

convert(laser["model"], ["embed_tokens.weight"], "encoder_Wemb", transpose=False)
for i in range(laser["params"]["num_layers"]):
    convert(laser["model"], [f"lstm.weight_ih_l{i}"], f"encoder_lstm_l{i}_W", lstm=True)
    convert(laser["model"], [f"lstm.weight_hh_l{i}"], f"encoder_lstm_l{i}_U", lstm=True)
    convert(laser["model"], [f"lstm.bias_ih_l{i}", f"lstm.bias_hh_l{i}"], f"encoder_lstm_l{i}_b", bias=True, lstm=True) # needs to be summed!
    
    convert(laser["model"], [f"lstm.weight_ih_l{i}_reverse"], f"encoder_lstm_l{i}_reverse_W", lstm=True)
    convert(laser["model"], [f"lstm.weight_hh_l{i}_reverse"], f"encoder_lstm_l{i}_reverse_U", lstm=True)
    convert(laser["model"], [f"lstm.bias_ih_l{i}_reverse", f"lstm.bias_hh_l{i}_reverse"], f"encoder_lstm_l{i}_reverse_b", bias=True, lstm=True) # needs to be summed!

for m in marianModel:
    print(m, marianModel[m].shape)

configYamlStr = yaml.dump(config, default_flow_style=False)
desc = list(configYamlStr)
npDesc = np.chararray((len(desc),))
npDesc[:] = desc
npDesc.dtype = np.int8
marianModel["special:model.yml"] = npDesc

print("\nMarian config:")
print(configYamlStr)
print("Saving Marian model to %s" % (args.marian,))
np.savez(args.marian, **marianModel)