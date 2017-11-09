#!/bin/bash

cat $1 \
    | sed 's/\@\@ //g' \
    | moses-scripts/scripts/recaser/detruecase.perl \
    | moses-scripts/scripts/tokenizer/detokenizer.perl -l en \
    | moses-scripts/scripts/generic/multi-bleu-detok.perl data/valid.de \
    | sed -r 's/BLEU = ([0-9.]+),.*/\1/'
