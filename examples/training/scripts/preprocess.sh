#!/bin/bash -v

# this sample script preprocesses a sample corpus, including tokenization,
# truecasing, and subword segmentation.
# for application to a different language pair,
# change source and target prefix, optionally the number of BPE operations,
# and the file names (currently, data/corpus and data/newsdev2016 are being processed)

# in the tokenization step, you will want to remove Romanian-specific normalization / diacritic removal,
# and you may want to add your own.
# also, you may want to learn BPE segmentations separately for each language,
# especially if they differ in their alphabet

# suffix of source language files
SRC=ro

# suffix of target language files
TRG=en

# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=85000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=moses-scripts

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=subword-nmt

# tokenize
for prefix in corpus newsdev2016 newstest2016
do
    cat data/$prefix.$SRC \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC \
        | ./scripts/normalise-romanian.py \
        | ./scripts/remove-diacritics.py \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $SRC > data/$prefix.tok.$SRC

    cat data/$prefix.$TRG \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG > data/$prefix.tok.$TRG

done

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
$mosesdecoder/scripts/training/clean-corpus-n.perl data/corpus.tok $SRC $TRG data/corpus.tok.clean 1 80

# train truecaser
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus data/corpus.tok.clean.$SRC -model model/truecase-model.$SRC
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus data/corpus.tok.clean.$TRG -model model/truecase-model.$TRG

# apply truecaser (cleaned training corpus)
for prefix in corpus
do
    $mosesdecoder/scripts/recaser/truecase.perl -model model/truecase-model.$SRC < data/$prefix.tok.clean.$SRC > data/$prefix.tc.$SRC
    $mosesdecoder/scripts/recaser/truecase.perl -model model/truecase-model.$TRG < data/$prefix.tok.clean.$TRG > data/$prefix.tc.$TRG
done

# apply truecaser (dev/test files)
for prefix in newsdev2016 newstest2016
do
    $mosesdecoder/scripts/recaser/truecase.perl -model model/truecase-model.$SRC < data/$prefix.tok.$SRC > data/$prefix.tc.$SRC
    $mosesdecoder/scripts/recaser/truecase.perl -model model/truecase-model.$TRG < data/$prefix.tok.$TRG > data/$prefix.tc.$TRG
done

# train BPE
cat data/corpus.tc.$SRC data/corpus.tc.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRC$TRG.bpe

# apply BPE

for prefix in corpus newsdev2016 newstest2016
do
    $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < data/$prefix.tc.$SRC > data/$prefix.bpe.$SRC
    $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < data/$prefix.tc.$TRG > data/$prefix.bpe.$TRG
done
