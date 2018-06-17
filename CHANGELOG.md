# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.0] - 2018-06-17

### Added

- Average Attention Networks for Transformer model
- 16-bit matrix multiplication on CPU
- Memoization for constant nodes for decoding
- Autotuning for decoding

### Fixed

- GPU decoding optimizations, about 2x faster decoding of transformer models

## [1.4.0] - 2018-03-13

### Added
- Data weighting with `--data-weighting` at sentence or word level
- Persistent SQLite3 corpus storage with `--sqlite file.db`
- Experimental multi-node asynchronous training
- Restoring optimizer and training parameters such as learning rate, validation
  results, etc.
- Experimental multi-CPU training/translation/scoring with `--cpu-threads=N`
- Restoring corpus iteration after training is restarted
- N-best-list scoring in marian-scorer

### Fixed
- Deterministic data shuffling with specific seed for SQLite3 corpus storage
- Mini-batch fitting with binary search for faster fitting
- Better batch packing due to sorting


## [1.3.1] - 2018-02-04

### Fixed
- Missing final validation when done with training
- Differing summaries for marian-scorer when used with multiple GPUs

## [1.3.0] - 2018-01-24

### Added
- SQLite3 based corpus storage for on-disk shuffling etc. with `--sqlite`
- Asynchronous maxi-batch preloading
- Using transpose in SGEMM to tie embeddings in output layer

## [1.2.1] - 2018-01-19

### Fixed
- Use valid-mini-batch size during validation with "translation" instead of mini-batch
- Normalize gradients with multi-gpu synchronous SGD
- Fix divergence between saved models and validated models in asynchronous SGD

## [1.2.0] - 2018-01-13

### Added
- Option `--pretrained-model` to be used for network weights initialization
  with a pretrained model
- Version number saved in the model file
- CMake option `-DCOMPILE_SERVER=ON`
- Right-to-left training, scoring, decoding with `--right-left`

### Fixed
- Fixed marian-server compilation with Boost 1.66
- Fixed compilation on g++-4.8.4
- Fixed compilation without marian-server if openssl is not available

## [1.1.3] - 2017-12-06

### Added
- Added back gradient-dropping

### Fixed
- Fixed parameters initialization for `--tied-embeddings` during translation

## [1.1.2] - 2017-12-05

### Fixed
- Fixed ensembling with language model and batched decoding
- Fixed attention reduction kernel with large matrices (added missing
  `syncthreads()`), which should fix stability with large batches and beam-size
  during batched decoding.

## [1.1.1] - 2017-11-30

### Added
- Option `--max-length-crop` to be used together with `--max-length N` to crop
  sentences to length N rather than omitting them.
- Experimental model with convolution over input characters

### Fixed
- Fixed a number of bugs for vocabulary and directory handling

## [1.1.0] - 2017-11-21

### Added
- Batched translation for all model types, significant translation speed-up
- Batched translation during validation with translation
- `--maxi-batch-sort` option for `marian-decoder`
- Support for CUBLAS_TENSOR_OP_MATH mode for cublas in cuda 9.0
- The "marian-vocab" tool to create vocabularies

## [1.0.0] - 2017-11-13

### Added
- Multi-gpu validation, scorer and in-training translation
- summary-mode for scorer
- New "transformer" model based on [Attention is all you
  need](https://arxiv.org/abs/1706.03762)
- Options specific for the transformer model
- Linear learning rate warmup with and without initial value
- Cyclic learning rate warmup
- More options for learning rate decay, including: optimizer history reset,
  repeated warmup
- Continuous inverted square root decay of learning (`--lr-decay-inv-sqrt`)
  rate based on number of updates
- Exposed optimizer parameters (e.g. momentum etc. for Adam)
- Version of deep RNN-based models compatible with Nematus (`--type nematus`)
- Synchronous SGD training for multi-gpu (enable with `--sync-sgd`)
- Dynamic construction of complex models with different encoders and decoders,
  currently only available through the C++ API
- Option `--quiet` to suppress output to stderr
- Option to choose different variants of optimization criterion: mean
  cross-entropy, perplexity, cross-entropy sum
- In-process translation for validation, uses the same memory as training
- Label Smoothing
- CHANGELOG.md
- CONTRIBUTING.md
- Swish activation function default for Transformer
  (https://arxiv.org/pdf/1710.05941.pdf)

### Changed
- Changed shape organization to follow numpy.
- Changed option `--moving-average` to `--exponential-smoothing` and inverted formula to `s_t = (1 - \alpha) * s_{t-1} + \alpha * x_t`, `\alpha` is now
  `1-e4` by default
- Got rid of thrust for compile-time mathematical expressions
- Changed boolean option `--normalize` to `--normalize [arg=1] (=0)`. New
  behaviour is backwards-compatible and can also be specified as
  `--normalize=0.6`
- Renamed "s2s" binary to "marian-decoder"
- Renamed "rescorer" binary to "marian-scorer"
- Renamed "server" binary to "marian-server"
- Renamed option name `--dynamic-batching` to `--mini-batch-fit`
- Unified cross-entropy-based validation, supports now perplexity and other CE
- Changed `--normalize (bool)` to `--normalize (float)arg`, allow to change
  length normalization weight as `score / pow(length, arg)`.

### Removed
- Temporarily removed gradient dropping (`--drop-rate X`) until refactoring.
