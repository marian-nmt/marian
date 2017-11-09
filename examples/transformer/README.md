# Example for training transformer model

Files and scripts in this folder shows how to run transformer model ([Vaswani
et al, 2017](https://arxiv.org/abs/1706.03762)) on WMT-17 English-German data.
The problem-set is adapted from
[tensor2tensor](https://github.com/tensorflow/tensor2tensor) repository from
Google, i.e. 32,000 common BPE subwords for both languages.
No back-translations are used.


Assuming, you have four GPUs available (here 0 1 2 3), type the command below
to execute the complete example:

```
./run-me.sh 0 1 2 3
```

It executes a training run with `marian` using the following command:

```
../../build/marian \
    --devices $GPUS \
    --seed 1234 \
    --type transformer \
    --model model/model.npz \
    --train-sets data/corpus.en data/corpus.de \
    --vocabs model/vocab.en.yml model/vocab.de.yml \
    --max-length 100 \
    --mini-batch-fit -w 7000 --maxi-batch 1000 \
    --early-stopping 10 \
    --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
    --valid-metrics ce-mean perplexity ce-mean-words translation \
    --valid-sets data/valid.bpe.en data/valid.bpe.de \
    --valid-script-path ./scripts/evaluate.sh \
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
```

This reproduces a system roughly equivalent to the basic 6-layer transformer
described in the original paper.

The training setting includes:
* Fitting mini-batch sizes to 7GB of GPU memory, which results in large mini-batches
* Validation on external data set using cross-entropy, perplexity and BLEU
* 6-layer encoder and 6-layer decoder
* Tied embeddings
* Label smoothing
* Learning rate warmup
* Multi-GPU training with synchronous SGD


The evaluation is performed on WMT test sets from 2014, 2015 and 2016 using
[sacreBLEU](https://github.com/mjpost/sacreBLEU), which provides hassle-free
computation of shareable, comparable, and reproducible BLEU scores.  The
WMT-213 test set is used as the validation set.

See the basic training example (`marian/examples/training/`) for more details.
