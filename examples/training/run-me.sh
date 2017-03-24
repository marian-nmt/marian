#!/bin/bash -v

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

if [ ! -e "mosesdecoder" ]
then
    git clone https://github.com/moses-smt/mosesdecoder
fi

if [ ! -e "subword-nmt" ]
then
    git clone https://github.com/rsennrich/subword-nmt
fi

if [ ! -e "data/ro-en.tgz" ]
then
    ./scripts/download-files.sh
fi

mkdir model

if [ ! -e "data/corpus.bpe.en" ]
then
    ./scripts/preprocess.sh
fi

if [ ! -e "model/model.npz" ]
then

../../build/marian \
 --model model/model.npz \
 --devices 0 \
 --train-sets data/corpus.bpe.ro data/corpus.bpe.en \
 --vocabs model/vocab.ro.yml model/vocab.en.yml \
 --dim-vocabs 50000 50000 \
 --mini-batch 80 \
 --layer-normalization \
 --after-batches 90000 \
 --valid-freq 10000 --save-freq 30000 --disp-freq 1000 \
 --valid-sets data/newsdev2016.bpe.ro data/newsdev2016.bpe.en \
 --valid-metrics cross-entropy valid-script \
 --valid-script-path ./scripts/validate.sh \
 --log model/train.log --valid-log model/valid.log

fi

