#!/bin/bash

# model prefix
prefix=model/model.npz

dev=data/newsdev2016.bpe.ro
ref=data/newsdev2016.tok.en

# decode

cat $dev | ../../build/amun -c $prefix.dev.npz.amun.yml -b 12 -n --mini-batch 10 --maxi-batch 100 2>/dev/null \
    | sed 's/\@\@ //g' | ./moses-scripts/scripts/recaser/detruecase.perl > $dev.output.postprocessed

# get BLEU
./moses-scripts/scripts/generic/multi-bleu.perl $ref < $dev.output.postprocessed | cut -f 3 -d ' ' | cut -f 1 -d ','
