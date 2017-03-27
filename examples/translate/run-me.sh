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

# download depdencies and data
if [ ! -e "mosesdecoder" ]
then
    git clone https://github.com/moses-smt/mosesdecoder
fi

if [ ! -e "en-de/model.npz" ]
then
  wget -r -l 1 --cut-dirs=2 -e robots=off -nH -np -R index.html* http://data.statmt.org/rsennrich/wmt16_systems/en-de/
fi


# Translate test set with single model
cat data/newstest2015.ende.en | \
#preprocess
mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en | \
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -penn | \
mosesdecoder/scripts/recaser/truecase.perl -model en-de/truecase-model.en | \
# translate
../../build/amun -m en-de/model.npz -s en-de/vocab.en.json -t en-de/vocab.de.json \
 --mini-batch 50 --maxi-batch 1000 -d $GPUS -b 12 -n --bpe en-de/ende.bpe  | \
# postprocess
mosesdecoder/scripts/recaser/detruecase.perl | \
mosesdecoder/scripts/tokenizer/detokenizer.perl -l de > data/newstest2015.single.out

# Create configuration file for model ensemble
../../build/amun -m en-de/model-ens?.npz -s en-de/vocab.en.json -t en-de/vocab.de.json \
 --mini-batch 1 --maxi-batch 1 -d $GPUS -b 12 -n --bpe en-de/ende.bpe \
 --relative-paths --dump-config > ensemble.yml

# Translate test set with ensemble
cat data/newstest2015.ende.en | \
#preprocess
mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en | \
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -penn | \
mosesdecoder/scripts/recaser/truecase.perl -model en-de/truecase-model.en | \
# translate
../../build/amun -c ensemble.yml | \
# postprocess
mosesdecoder/scripts/recaser/detruecase.perl | \
mosesdecoder/scripts/tokenizer/detokenizer.perl -l de > data/newstest2015.ensemble.out

mosesdecoder/scripts/generic/multi-bleu.perl data/newstest2015.ende.de < data/newstest2015.single.out
mosesdecoder/scripts/generic/multi-bleu.perl data/newstest2015.ende.de < data/newstest2015.ensemble.out

