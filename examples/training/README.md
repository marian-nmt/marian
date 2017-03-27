# Example for training with Marian

Files and scripts in this folder have been adapted from the Romanian-English sample 
from https://github.com/rsennrich/wmt16-scripts. We also add the back-translated data from
http://data.statmt.org/rsennrich/wmt16_backtranslations/ as desribed in 
http://www.aclweb.org/anthology/W16-2323. The resulting system should be competitive 
or even slightly better than reported in the Edinburgh WMT2016 paper. 

To execute the complete example type:

```
./run-me.sh
```

which downloads the Romanian-English training files and preprocesses them (tokenization, 
truecasing, segmentation into subwords units). 

To use with a different GPU than device 0 or more GPUs (here 0 1 2 3) type the command below. 
Training time on 1 NVIDIA GTX 1080 GPU should be roughly 24 hours.

```
./run-me.sh 0 1 2 3
```

Next it executes a training run with `marian`:

```
../../build/marian \
 --model model/model.npz \
 --devices $GPUS \
 --train-sets data/corpus.bpe.ro data/corpus.bpe.en \
 --vocabs model/vocab.ro.yml model/vocab.en.yml \
 --dim-vocabs 66000 50000 \
 --mini-batch 80 \
 --layer-normalization --dropout-rnn 0.2 --dropout-src 0.1 --dropout-trg 0.1 \
 --early-stopping 5 --moving-average \
 --valid-freq 10000 --save-freq 10000 --disp-freq 1000 \
 --valid-sets data/newsdev2016.bpe.ro data/newsdev2016.bpe.en \
 --valid-metrics cross-entropy valid-script \
 --valid-script-path ./scripts/validate.sh \
 --log model/train.log --valid-log model/valid.log
```
After training (the training should stop if cross-entropy on the validation set stops improving) a final model 
`model/model.avg.npz` is created from the 4 best models on the validation sets (by element-wise averaging). This model is used to 
translate the WMT2016 dev set and test set with `amun`:

```
cat data/newstest2016.bpe.ro \
 | ../../build/amun -c model/model.npz.amun.yml -m model/model.avg.npz -b 12 -n --mini-batch 100 --maxi-batch 1000 \
 | sed 's/\@\@ //g' | mosesdecoder/scripts/recaser/detruecase.perl \
 > data/newstest2016.bpe.ro.output
```
after which BLEU scores for the dev and test set are reported. Results should be somewhere in the area of:

```
newsdev2016:
BLEU = 35.88, 67.4/42.3/28.8/20.2 (BP=1.000, ratio=1.012, hyp_len=51085, ref_len=50483)

newstest2016:
BLEU = 34.53, 66.0/40.7/27.5/19.2 (BP=1.000, ratio=1.015, hyp_len=49258, ref_len=48531)
```

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
 | sed 's/\@\@ //g' | ./mosesdecoder/scripts/recaser/detruecase.perl > $dev.output.postprocessed

## get BLEU
./mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output.postprocessed \
| cut -f 3 -d ' ' | cut -f 1 -d ','
```
