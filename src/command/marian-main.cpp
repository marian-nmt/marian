#include "marian.h"

// TODO: rename these functions actually
#define main mainTrainer
#include "marian.cpp"
#undef main
#define main mainDecoder
#include "marian_decoder.cpp"
#undef main
//#define main mainScorer
//#include "marian_scorer.cpp"
//#undef main
//#define main mainVocab
//#include "marian_vocab.cpp"
//#undef main

int main(int argc, char** argv) {
  using namespace marian;

  if(argc > 1 && argv[1][0] != '-') {
    std::string cmd = argv[1];
    argc--;
    argv[1] = argv[0];
    argv++;
    if(cmd == "train")
      return mainTrainer(argc, argv);
    else if(cmd == "decode")
      return mainDecoder(argc, argv);
    // else if (cmd == "score")  return mainScorer(argc, argv);
    // else if (cmd == "vocab")  return mainVocab(argc, argv);
    std::cerr << "Command must be train, decode, score, or vocab.";
    exit(1);
  } else
    return mainTrainer(argc, argv);
}
