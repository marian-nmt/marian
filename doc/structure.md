# Code Organisation

This purpose of this document is to outline the organisational structure of the Marian codebase. Each section of this document approaches an architectural component and highlights a subset of directories that are relevant to it.


## Operating Modes
```
marian/src
├── command
├── rescorer
├── training
└── translator
```
The Marian toolkit provides several commands, covering different modes of operation. These are:
  - `marian`
  - `marian-decoder`
  - `marian-server`
  - `marian-scorer`
  - `marian-vocab`
  - `marian-conv`

Each of which has a corresponding file in the `command` directory.

The main `marian` command is capable of running all other modes (except server), see `marian-main.cpp` for the implementation. By default, it operates in `train` mode and corresponds to `marian-train.cpp`. Other modes may be accessed by calling `marian <X>` instead of `marian-<X>`.

Training is covered by the main `marian` command, with relevant implementation details kept inside the `training` subdirectory. Translation is facilitated by code in the `translator` subdirectory and is handled by the `marian-decoder` command, as well as `marian-server` which provides a web-socket service. `marian-scorer` is the tool used to re-score parallel inputs or n-best lists, and uses code in the `rescorer` subdirectory.

The remaining commands `marian-vocab` and `marian-conv` provide useful auxiliary functions.  `marian-vocab` is a tool to create a vocabulary file from a given text corpus. This uses components described in the Data section of this document.
`marian-conv` exists to convert Marian model files from `.npz`, `.bin` as well as lexical shortlists to binary shortlists. It is also possible to use this command to emit an ONNX-compliant model representation. In addition to components defined in the Data section, this also makes use of Model specific components.

Finally, the implementation of the command-line-interface for these commands is described in the Utility section.


## Data
```
marian/src
└── data
```
Data refers to the handling and representation of the text input to Marian.
This consists of source code for the representation of the corpus, vocabulary and batches.

Internally, tokens are represented as indices, or `Words`; some indices are reserved for special tokens, such as `EOS`, `UNK`. Vocabulary implementations are responsible for encoding and decoding sentences to and from the internal representation, whether that be a SentencePiece, Factors or Plain Text/YAML defined vocabulary file.

This directory is also responsible for generating batches from a corpus and performing any shuffling of the corpus or batches, as requested. Furthermore, when using a shortlist, their behaviour is also defined here.

Once the batches are generated they are passed as input to the expression graph.


## Expression Graph
```
marian/src
├── functional
├── graph
├── optimizers
└── tensors
```

Marian implements a reverse-mode auto-differentiation computation graph. The relevant components reside in these subdirectories. The `graph` subdirectory concerns the structure of the graph, its nodes: operators, parameters and constants, as well as how to traverse it, both forwards and backwards. Moreover, it defines the APIs for operations that the graph is able to perform.

The `tensors` and `functional` subdirectories contain the implementation of operations for the graph.

One component of the `functional` subdirectory describes how functions operate on the underlying data types. This is a combination of standard operations on fundamental types, and SIMD intrinsics on extended types where available. The `functional` namespace also provides useful abstractions that enable generic formulas to be written. It defines variable-like objects `_1,_2`, such that `_1 * cos(_2)` represents the product of the argument at index 1 with the cosine of the argument at index 2.

The `tensors` subdirectory contains the definition of a tensor object. In Marian, a tensor is a piece of memory which is ascribed a shape and type which is associated with a backend (the compute device).
This directory also contains the implementations of tensor operations on CPU and GPU, as well as universal functions that dispatches the call to the relevant device.

More specific documentation is available that describes the [graph][graph], and how its [operators][graph_ops] are implemented.


## Model
```
marian/src
├── models
├── layers
└── rnn
```
The subdirectories above constitute the components of a Model. There are two main types of model:
  - `IModel`, which maps inputs to predictions
  - `ICriterionFunction`, which maps (inputs, references) to losses

The usage of these interfaces sometimes combined. As an example, `Trainer`, an implementation of the `ICriterionFunction` interface used in training contains an `IModel` member from which it then computes the loss.

An important specialisation of `IModel` is `IEncoderDecoder`, this specifies the interface for the `EncoderDecoder` class. `EncoderDecoder` consists of a set of Encoders and Decoders objects, which implement the interface of `EncoderBase` and `DecoderBase`, respectively. This composite object defines the behaviour of general Encoder-Decoder models. For instance, the `s2s` models implement a `EncoderS2S` and `DecoderS2S`, while `transformer` models implement a `EncoderTransformer` `DecoderTransformer`. These two use cases are both encapsulated in the `EncoderDecoder` framework. The addition of new encoder-decoder models only need implement their encoder and decoder classes. The `EncoderDecoder` models are constructed using a factory pattern in `src/models/model_factory.cpp`.

The export of an ONNX-compliant model is handled by code here.
```
marian/src
└── onnx
```


## Utility
```
marian/src
└── common
```
The `common` subdirectory contains many useful helper functions and classes.
The majority of which fall under one of these categories:
  - Command-line interface definition an Options object
  - Definitions, macros and typedefs
  - Filesystem and IO helpers
  - Logging
  - Memory management
  - Signal handling
  - Text manipulation
  - Type-based dispatching and properties

Beyond these areas, this folder also contains metadata, such as the program version, list of contributors, and the build flags used to compile it.


## External Libraries
```
marian/src
└── 3rd_party
  ```
Many of the external libraries that Marian depends on are contained in `3rd_party`.

These libraries are either copied into place here and version-controlled via the marian repository, or are included here as a submodule. Of these submodules, many have been forked and are maintained under the marian-nmt organisation.


## Tests and Examples
```
marian/src
├── examples
└── tests
```
There are basic tests and examples contained in `marian/src`.

The unit tests cover basic graph functionality, checks on the output of operators, and the implementation of RNN attention, as well IO of binary files and manipulation of the options structure.

The examples in this subdirectory demonstrate Marian's functionality using common datasets: Iris and MNIST. The Iris example, builds a simple dense feedforward network to perform a classification task. Over 200 epochs, it trains the network on target using mean cross-entropy. It then reports the accuracy of the model on the test-set. The MNIST example showcases more advanced features of Marian. It offers a choice of models (FFNN, LeNet), can leverage multi-device environments and uses a validator during training. This example more closely replicates the workflow of a typical Marian model, with batching of data and a model implemented in terms of Marian's model interfaces.

```
marian
├── examples
└── regression-tests
```
Further tests and examples are contained in the root of the marian source code. The examples here are end-to-end tutorials on how to use Marian. These range from covering the basics of training a Marian model, to replicating the types of models presented at the Conference on Machine Translation (WMT).
Similarly, the tests in `regression-tests` are more numerous and detailed. They cover some 250+ areas of the code. While the unit tests described above check basic consistency of certain functions, the regression tests offer end-to-end verification of the functionality of Marian.

<!-- Links -->
[graph]: https://marian-nmt.github.io/docs/api/graph.html
[graph_ops]: https://marian-nmt.github.io/docs/api/operators.html
