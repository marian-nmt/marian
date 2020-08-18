#include "marian.h"

// This contains the main function for the aggregate command line that allows to specify
// one of the Marian executables as the first argument. This is done by including all
// individual .cpp files into a single .cpp, using a #define to rename the respective
// main functions.
// For example, the following two are equivalent:
//  marian-scorer ARGS
//  marian score  ARGS
// The supported sub-commands are:
//  train
//  decode
//  score
//  embed
//  vocab
//  convert
// Currently, marian_server is not supported, since it is a special use case with lots of extra dependencies.

#define main mainTrainer
#include "marian_train.cpp"
#undef main
#define main mainDecoder
#include "marian_decoder.cpp"
#undef main
#define main mainScorer
#include "marian_scorer.cpp"
#undef main
#define main mainEmbedder
#include "marian_embedder.cpp"
#undef main
#define main mainVocab
#include "marian_vocab.cpp"
#undef main
#define main mainConv
#include "marian_conv.cpp"
#undef main

#include "3rd_party/ExceptionWithCallStack.h"

int main(int argc, char** argv) {
  using namespace marian;

  if(argc > 1 && argv[1][0] != '-') {
    std::string cmd = argv[1];
    argc--;
    argv[1] = argv[0];
    argv++;
    if(cmd == "train")           return mainTrainer(argc, argv);
    else if(cmd == "decode")     return mainDecoder(argc, argv);
    else if (cmd == "score")     return mainScorer(argc, argv);
    else if (cmd == "embed")     return mainEmbedder(argc, argv);
    else if (cmd == "vocab")     return mainVocab(argc, argv);
    else if (cmd == "convert")   return mainConv(argc, argv);
    std::cerr << "Command must be train, decode, score, embed, vocab, or convert." << std::endl;
    exit(1);
  } else
    return mainTrainer(argc, argv);
}
