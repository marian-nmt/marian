#!/bin/bash

# set chosen gpus
GPUS=0
if [ $# -ne 0 ]
then
    GPUS=$@
fi
echo Using gpus $GPUS

if [ ! -e ../../build/amun ]
then
    echo "amun is not installed in ../../build, you need to compile the toolkit first."
    exit 1
fi

# download dependencies and data
if [ ! -e "moses-scripts" ]
then
    git clone https://github.com/amunmt/moses-scripts
fi

if [ ! -e "en-de/model.npz" ]
then
    wget -r -l 1 --cut-dirs=2 -e robots=off -nH -np -R index.html* http://data.statmt.org/rsennrich/wmt16_systems/en-de/
fi


# translate test set with single model
cat data/newstest2015.ende.en | \
    # preprocess
    moses-scripts/scripts/tokenizer/normalize-punctuation.perl -l en | \
    moses-scripts/scripts/tokenizer/tokenizer.perl -l en -penn | \
    moses-scripts/scripts/recaser/truecase.perl -model en-de/truecase-model.en | \
    # translate
    ../../build/amun -m en-de/model.npz -s en-de/vocab.en.json -t en-de/vocab.de.json \
    --mini-batch 50 --maxi-batch 1000 -d $GPUS --gpu-threads 1 -b 12 -n --bpe en-de/ende.bpe | \
    # postprocess
    moses-scripts/scripts/recaser/detruecase.perl | \
    moses-scripts/scripts/tokenizer/detokenizer.perl -l de > data/newstest2015.single.out

# create configuration file for model ensemble
../../build/amun -m en-de/model-ens?.npz -s en-de/vocab.en.json -t en-de/vocab.de.json \
    --mini-batch 1 --maxi-batch 1 -d $GPUS --gpu-threads 1 -b 12 -n --bpe en-de/ende.bpe \
    --relative-paths --dump-config > ensemble.yml

# translate test set with ensemble
cat data/newstest2015.ende.en | \
    # preprocess
    moses-scripts/scripts/tokenizer/normalize-punctuation.perl -l en | \
    moses-scripts/scripts/tokenizer/tokenizer.perl -l en -penn | \
    moses-scripts/scripts/recaser/truecase.perl -model en-de/truecase-model.en | \
    # translate
    ../../build/amun -c ensemble.yml --gpu-threads 1 | \
    # postprocess
    moses-scripts/scripts/recaser/detruecase.perl | \
    moses-scripts/scripts/tokenizer/detokenizer.perl -l de > data/newstest2015.ensemble.out

moses-scripts/scripts/generic/multi-bleu.perl data/newstest2015.ende.de < data/newstest2015.single.out
moses-scripts/scripts/generic/multi-bleu.perl data/newstest2015.ende.de < data/newstest2015.ensemble.out
