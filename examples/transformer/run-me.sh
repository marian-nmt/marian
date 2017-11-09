#!/bin/bash -v

# set chosen gpus
GPUS=0
if [ $# -ne 0 ]
then
    GPUS=$@
fi
echo Using GPUs: $GPUS

if [ ! -e ../../build/marian ]
then
    echo "marian is not installed in ../../build, you need to compile the toolkit first"
    exit 1
fi

# download dependencies and data
if [ ! -e moses-scripts ]; then git clone https://github.com/amunmt/moses-scripts; fi
if [ ! -e subword-nmt ];   then git clone https://github.com/rsennrich/subword-nmt; fi
if [ ! -e sacreBLEU ];     then git clone https://github.com/mjpost/sacreBLEU; fi

if [ ! -e "data/corpus.en" ]
then
    ./scripts/download-files.sh
fi

mkdir -p model

# preprocess data
if [ ! -e "data/corpus.bpe.en" ]
then
    ./sacreBLEU/sacrebleu.py -t wmt13 -l en-de --echo src > data/valid.en
    ./sacreBLEU/sacrebleu.py -t wmt13 -l en-de --echo ref > data/valid.de

    ./sacreBLEU/sacrebleu.py -t wmt14 -l en-de --echo src > data/test2014.en
    ./sacreBLEU/sacrebleu.py -t wmt15 -l en-de --echo src > data/test2015.en
    ./sacreBLEU/sacrebleu.py -t wmt16 -l en-de --echo src > data/test2016.en

    ./scripts/preprocess-data.sh
fi

# train model
if [ ! -e "model/model.npz" ]
then
    ../../build/marian \
        --devices $GPUS \
        --seed 1234 \
        --type transformer \
        --model model.model.npz \
        --train-sets data/corpus.en data/corpus.de \
        --vocabs model/vocab.en.yml model/vocab.de.yml \
        --max-length 100 \
        --mini-batch-fit -w 7000 --maxi-batch 1000 \
        --early-stopping 10 \
        --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
        --valid-metrics ce-mean perplexity ce-mean-words translation \
        --valid-sets data/valid.bpe.en data/valid.bpe.de \
        --valid-script-path ./script/evaluate.sh \
        --valid-max-length 100 \
        --log model/train.log --valid-log model/valid.log \
        --transformer-postprocess-emb d \
        --transformer-postprocess dhn \
        --transformer-heads 16 --transformer-dropout 0.3 \
        --transformer-dim-ffn 4096 \
        --enc-depth 6 --dec-depth 6 --dim-emb 1024 \
        --tied-embeddings-all \
        --label-smoothing 0.1 \
        --clip-norm 5 --optimizer-params 0.9 0.98 1e-09 \
        --learn-rate 0.0003 --lr-report --lr-warmup 16000 --lr-decay-inv-sqrt 16000 \
        --sync-sgd
fi

# find best model on dev set
ITER=`cat model/valid.log | grep translation | sort -rg -k8,8 -t' ' | cut -f4 -d' ' | head -n1`
cp model/model.iter$ITER.npz model/model.bestdev.npz

# translate test sets
for prefix in test2014 test2015 test2016
do
    cat data/$prefix.bpe.en \
        | ../../build/marian-decoder -c model/model.npz.decoder.yml -m model/model.bestdev.npz -d $GPUS -b 12 -n \
        | sed 's/\@\@ //g' \
        | moses-scripts/scripts/recaser/detruecase.perl \
        | moses-scripts/scripts/tokenizer/detokenizer.perl -l de \
        > data/$prefix.de.output
done

# calculate bleu scores on test sets
./sacreBLEU/sacrebleu.py -t wmt14 -l en-de < data/test2014.de.output
./sacreBLEU/sacrebleu.py -t wmt15 -l en-de < data/test2015.de.output
./sacreBLEU/sacrebleu.py -t wmt16 -l en-de < data/test2016.de.output
