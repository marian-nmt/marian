# Example for training with Marian

Files and scripts in this folder have been adapted from the Romanian-English sample from https://github.com/rsennrich/wmt16-scripts. 

```
./run-me.sh
```

which downloads the Romanian-English training files and preprocesses them (tokenization, truecasing, segmentation into subwords units). 

Next it executes a training run with `marian`:

```
../../build/marian \
 --model model/model.npz \
 --devices 0 \
 --train-sets data/corpus.bpe.ro data/corpus.bpe.en \
 --vocabs model/vocab.ro.yml model/vocab.en.yml \
 --dim-vocabs 32000 32000 \
 --mini-batch 80 \
 --layer-normalization \
 --after-batches 10000 \
 --valid-freq 10000 --save-freq 30000 --disp-freq 1000 \
 --valid-sets data/newsdev2016.bpe.ro data/newsdev2016.bpe.en \
 --valid-metrics cross-entropy valid-script \
 --valid-script-path ./scripts/validate.sh \
 --log model/train.log --valid-log model/valid.log
```
After training for 90000 updates (mini-batches) the final model is used to translate the WMT2016 test set with `amun`:

```
cat data/newstest2016.bpe.ro \
 | ../../build/amun -c model/model.npz.amun.yml -b 12 -n --mini-batch 100 --maxi-batch 1000 \
 | sed 's/\@\@ //g' | mosesdecoder/scripts/recaser/detruecase.perl \
 > data/newstest2016.bpe.ro.output
```
after which BLEU scores for the test set are reported. 

## Custom validation script

The validation script `scripts/validate.sh` is a quick example how to write a custom validation script. The training pauses until the validation script finishes executing. A validation script should not output anything to `stdout` apart from the final single score (last line): 

```
#!/bin/bash

#model prefix
prefix=model/model.npz

dev=data/newsdev2016.bpe.ro
ref=data/newsdev2016.tok.en

# decode

cat $dev | ../../build/amun -c $prefix.dev.npz.amun.yml --mini-batch 10 --maxi-batch 100 2>/dev/null \
 | sed 's/\@\@ //g' | mosesdecoder/scripts/recaser/detruecase.perl > $dev.output.postprocessed

## get BLEU
./mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output.postprocessed \
| cut -f 3 -d ' ' | cut -f 1 -d ','
```
