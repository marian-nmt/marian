#!/bin/bash -v

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

if [ ! -e ../../build/marian ]
then
    echo "marian is not installed in ../../build, you need to compile the toolkit first."
    exit 1
fi

# download dependencies and data
if [ ! -e "moses-scripts" ]
then
    git clone https://github.com/amunmt/moses-scripts
fi

if [ ! -e "subword-nmt" ]
then
    git clone https://github.com/rsennrich/subword-nmt
fi

if [ ! -e "data/ro-en.tgz" ]
then
    ./scripts/download-files.sh
fi

mkdir -p model

# preprocess data
if [ ! -e "data/corpus.bpe.en" ]
then
    ./scripts/preprocess.sh
fi

# train model
if [ ! -e "model/model.npz" ]
then

    ../../build/marian \
        --model model/model.npz \
        --devices $GPUS --seed 0 \
        --train-sets data/corpus.bpe.ro data/corpus.bpe.en \
        --vocabs model/vocab.ro.yml model/vocab.en.yml \
        --dim-vocabs 66000 50000 \
        --dynamic-batching -w 3000 \
        --layer-normalization --dropout-rnn 0.2 --dropout-src 0.1 --dropout-trg 0.1 \
        --early-stopping 5 --moving-average \
        --valid-freq 10000 --save-freq 10000 --disp-freq 1000 \
        --valid-sets data/newsdev2016.bpe.ro data/newsdev2016.bpe.en \
        --valid-metrics cross-entropy valid-script \
        --valid-script-path ./scripts/validate.sh \
        --log model/train.log --valid-log model/valid.log

fi

# collect 4 best models on dev set
MODELS=`cat model/valid.log | grep valid-script | sort -rg -k8,8 -t ' ' | cut -f 4 -d ' ' | head -n 4 | xargs -I {} echo model/model.iter{}.npz | xargs`

# average 4 best models into single model
../../scripts/average.py -m $MODELS -o model/model.avg.npz

# translate dev set with averaged model
cat data/newsdev2016.bpe.ro \
    | ../../build/amun -c model/model.npz.amun.yml -m model/model.avg.npz -d $GPUS -b 12 -n --mini-batch 10 --maxi-batch 1000 \
    | sed 's/\@\@ //g' | moses-scripts/scripts/recaser/detruecase.perl > data/newsdev2016.bpe.ro.output.postprocessed

# translate test set with averaged model
cat data/newstest2016.bpe.ro \
    | ../../build/amun -c model/model.npz.amun.yml -m model/model.avg.npz -d $GPUS -b 12 -n --mini-batch 10 --maxi-batch 1000 \
    | sed 's/\@\@ //g' | moses-scripts/scripts/recaser/detruecase.perl > data/newstest2016.bpe.ro.output.postprocessed

# calculate bleu scores for dev and test set
./moses-scripts/scripts/generic/multi-bleu.perl data/newsdev2016.tok.en < data/newsdev2016.bpe.ro.output.postprocessed
./moses-scripts/scripts/generic/multi-bleu.perl data/newstest2016.tok.en < data/newstest2016.bpe.ro.output.postprocessed
