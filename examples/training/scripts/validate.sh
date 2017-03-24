#!/bin/bash

#model prefix
prefix=model/model.npz

dev=data/newsdev2016.bpe.ro
ref=data/newsdev2016.tok.en

# decode

cat $dev | ../../build/amun -c $prefix.dev.npz.amun.yml -b 12 -n --mini-batch 10 --maxi-batch 100 2>/dev/null \
 | sed 's/\@\@ //g' | mosesdecoder/scripts/recaser/detruecase.perl > $dev.output.postprocessed.dev

## get BLEU
BLEU=`./mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output.postprocessed.dev | cut -f 3 -d ' ' | cut -f 1 -d ','`

echo $BLEU
