#pragma once
#include <cstdint>

namespace marian {
namespace spm {

// Describes an individual token in a sentencepiece encoding
struct Native_SentencePiecePiece {
  int id;
  int begin;
  int end;
  char* surface;
  char* piece;
};

// Mirrors the SentencePieceText protobuf struct returned by SPM
// and provides individual piece and corresponding surface details
struct Native_SentencePieceText {
  char* text;
  int num_pieces;
  Native_SentencePiecePiece** pieces;
};

int SentencePieceInteropFreeNativeSentencePieceText(Native_SentencePieceText* spt);
intptr_t SentencePieceInteropLoadModel(const uint16_t* modelPath,
                                       const uint16_t** vocab,
                                       size_t vocabSize);
int SentencePieceInteropDecodeAligned(intptr_t object,
                                      int num_tokens,
                                      char** tokens,
                                      Native_SentencePieceText** nSpt);
int SentencePieceInteropEncodeAligned(intptr_t object, char* word, Native_SentencePieceText** nSpt);
int SentencePieceInteropGetPieceID(intptr_t object, char* word);
int SentencePieceInteropUnloadModel(intptr_t object);
int SentencepieceInteropTrainModel(char* args);

}  // namespace spm
}  // namespace marian