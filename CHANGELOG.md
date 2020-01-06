# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added
- Add CMAKE options to disable compilation for specific GPU SM types
- An option to print word-level translation scores
- An option to turn off automatic detokenization from SentencePiece
- Separate quantization types for 8-bit FBGEMM for AVX2 and AVX512
- Sequence-level unliklihood training
- Allow file name templated valid-translation-output files
- Support for lexical shortlists in marian-server
- Support for 8-bit matrix multiplication with FBGEMM
- CMakeLists.txt now looks for SSE 4.2
- Purging of finished hypotheses during beam-search. A lot faster for large batches.
- Faster option look-up, up to 20-30% faster translation
- Added --cite and --authors flag
- Added optional support for ccache
- Switch to change abort to exception, only to be used in library mode
- Support for 16-bit packed models with FBGEMM
- Multiple separated parameter types in ExpressionGraph, currently inference-only
- Safe handling of sigterm signal
- Automatic vectorization of elementwise operations on CPU for tensors dims that 
  are divisible by 4 (AVX) and 8 (AVX2)
- Replacing std::shared_ptr<T> with custom IntrusivePtr<T> for small objects like 
  Tensors, Hypotheses and Expressions.
- Fp16 inference working for translation
- Gradient-checkpointing

### Fixed
- Gather-operation for all index sizes
- Fix word weighting with max length cropping
- Fixed compilation on CPUs without support for AVX
- FastOpt now reads "n" and "y" values as strings, not as boolean values
- Fixed multiple reduction kernels on GPU
- Fixed guided-alignment training with cross-entropy
- Replace IntrusivePtr with std::uniq_ptr in FastOpt, fixes random segfaults 
  due to thread-non-safty of reference counting.
- Make sure that items are 256-byte aligned during saving
- Make explicit matmul functions respect setting of cublasMathMode
- Fix memory mapping for mixed paramter models
- Removed naked pointer and potential memory-leak from file_stream.{cpp,h}
- Compilation for GCC >= 7 due to exception thrown in destructor
- Sort parameters by lexicographical order during allocation to ensure consistent 
  memory-layout during allocation, loading, saving.
- Output empty line when input is empty line. Previous behavior might result in 
  hallucinated outputs.
- Compilation with CUDA 10.1

### Changed
- Downgrade NCCL to 2.3.7 as 2.4.2 is buggy (hangs with larger models)
- Return error signal on SIGTERM
- Dropped support for CUDA 8.0, CUDA 9.0 is now minimal requirement
- Removed autotuner for now, will be switched back on later
- Boost depdendency is now optional and only required for marian_server 
- Dropped support for g++-4.9
- Simplified file stream and temporary file handling
- Unified node intializers, same function API.

## [1.8.0] - 2019-09-04

### Added
- Alias options and new --task option
- Automatic detection of CPU intrisics when building with -arch=native
- First version of BERT-training and BERT-classifier, currently not compatible with TF models
- New reduction operators
- Use Cmake's ExternalProject to build NCCL and potentially other external libs
- Code for Factored Vocabulary, currently not usable yet without outside tools

### Fixed
- Issue with relative paths in automatically generated decoder config files
- Bug with overlapping CXX flags and building spm_train executable
- Compilation with gcc 8
- Overwriting and unsetting vector options
- Windows build with recent changes
- Bug with read-ahead buffer
- Handling of "dump-config: false" in YAML config
- Errors due to warnings
- Issue concerning failed saving with single GPU training and --sync-sgd option.
- NaN problem when training with Tensor Cores on Volta GPUs
- Fix pipe-handling
- Fix compilation with GCC 9.1
- Fix CMake build types

### Changed
- Error message when using left-to-right and right-to-left models together in ensembles
- Regression tests included as a submodule
- Update NCCL to 2.4.2
- Add zlib source to Marian's source tree, builds now as object lib
- -DUSE_STATIC_LIBS=on now also looks for static versions of CUDA libraries
- Include NCCL build from github.com/marian-nmt/nccl and compile within source tree
- Set nearly all warnings as errors for Marian's own targets. Disable warnings for 3rd party
- Refactored beam search

## [1.7.0] - 2018-11-27

### Added
- Word alignment generation in scorer
- Attention output generation in decoder and scorer with `--alignment soft`
- Support for SentencePiece vocabularies and run-time segmentation/desegmentation
- Support for SentencePiece vocabulary training during model training
- Group training files by filename when creating vocabularies for joint vocabularies
- Updated examples
- Synchronous multi-node training (early version)

### Fixed
- Delayed output in line-by-line translation

### Changed
- Generated word alignments include alignments for target EOS tokens
- Boost::program_options has been replaced by another CLI library
- Replace boost::file_system with Pathie
- Expansion of unambiguous command-line arguments is no longer supported

## [1.6.0] - 2018-08-08

### Added
- Faster training (20-30%) by optimizing gradient popagation of biases
- Returning Moses-style hard alignments during decoding single models,
  ensembles and n-best lists
- Hard alignment extraction strategy taking source words that have the
  attention value greater than the threshold
- Refactored sync sgd for easier communication and integration with NCCL
- Smaller memory-overhead for sync-sgd
- NCCL integration (version 2.2.13)
- New binary format for saving/load of models, can be used with _*.bin_
  extension (can be memory mapped)
- Memory-mapping of graphs for inferece with `ExpressionGraph::mmap(const void*
  ptr)` function. (assumes _*.bin_ model is mapped or in buffer)
- Added SRU (--dec-cell sru) and ReLU (--dec-cell relu) cells to inventory of
  RNN cells
- RNN auto-regression layers in transformer (`--transformer-decoder-autreg
  rnn`), work with gru, lstm, tanh, relu, sru cells
- Recurrently stacked layers in transformer (`--transformer-tied-layers 1 1 1 2
  2 2` means 6 layers with 1-3 and 4-6 tied parameters, two groups of
  parameters)
- Seamless training continuation with exponential smoothing

### Fixed
- A couple of bugs in "selection" (transpose, shift, cols, rows) operators
  during back-prob for a very specific case: one of the operators is the first
  operator after a branch, in that case gradient propgation might be
  interrupted. This did not affect any of the existing models as such a case
  was not present, but might have caused future models to not train properly
- Bug in mini-batch-fit, tied embeddings would result in identical embeddings
  in fake source and target batch. Caused under-estimation of memory usage and
  re-allocation

## [1.5.0] - 2018-06-17

### Added
- Average Attention Networks for Transformer model
- 16-bit matrix multiplication on CPU
- Memoization for constant nodes for decoding
- Autotuning for decoding

### Fixed
- GPU decoding optimizations, about 2x faster decoding of transformer models
- Multi-node MPI-based training on GPUs

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
- Use valid-mini-batch size during validation with "translation" instead of
  mini-batch
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
  during batched decoding

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
- Changed option `--moving-average` to `--exponential-smoothing` and inverted
  formula to `s_t = (1 - \alpha) * s_{t-1} + \alpha * x_t`, `\alpha` is now
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
  length normalization weight as `score / pow(length, arg)`

### Removed
- Temporarily removed gradient dropping (`--drop-rate X`) until refactoring.
