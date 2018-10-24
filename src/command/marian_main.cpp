#include "marian.h"

// @TODO: rename these functions actually
#define main mainTrainer
#include "marian.cpp"
#undef main
#define main mainDecoder
#include "marian_decoder.cpp"
#undef main
//#define main mainScorer // commented out for now since it would require more intrusive code changes
//#include "marian_scorer.cpp"
//#undef main
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
    // else if (cmd == "score")     return mainScorer(argc, argv);
    else if (cmd == "vocab")     return mainVocab(argc, argv);
    else if (cmd == "convert")   return mainConv(argc, argv);
    std::cerr << "Command must be train, decode, score, vocab, or convert." << std::endl;
    exit(1);
  } else
    return mainTrainer(argc, argv);
}
