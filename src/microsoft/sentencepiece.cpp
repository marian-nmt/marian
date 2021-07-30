#include <sstream>
#include <memory>
#include <string>
#include <vector>

#ifdef USE_SENTENCEPIECE
#include "sentencepiece.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif

#include "sentencepiece/src/builtin_pb/sentencepiece.pb.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "sentencepiece/src/sentencepiece_processor.h"
#include "sentencepiece/src/sentencepiece_trainer.h"
#include "unicode_conversions.h"

namespace marian {
namespace spm {
class SentencePieceInternal {
  std::unique_ptr<sentencepiece::SentencePieceProcessor> m_processor;

  void checkStatus(sentencepiece::util::Status status, const char* what) {
    if(status.ok())
      return;
    std::string err = status.ToString();
    std::cerr << err << std::endl;
    throw std::runtime_error(std::string("SentencePiece error ") + what + ": " + err);
  }

  int createNativeSentencePieceText(sentencepiece::SentencePieceText& spt, Native_SentencePieceText** outSpt) {
    Native_SentencePieceText* spt_ret = new Native_SentencePieceText();

    spt_ret->text = new char[spt.text().size() + 1];
    ::strcpy(spt_ret->text, spt.text().c_str());

    spt_ret->num_pieces = spt.pieces().size();
    spt_ret->pieces     = new Native_SentencePiecePiece*[spt_ret->num_pieces];

    int counter = 0;
    for(auto& piece : spt.pieces()) {
      spt_ret->pieces[counter]          = new Native_SentencePiecePiece();
      spt_ret->pieces[counter]->id      = piece.id();
      spt_ret->pieces[counter]->begin   = piece.begin();
      spt_ret->pieces[counter]->end     = piece.end();
      spt_ret->pieces[counter]->surface = new char[piece.surface().size() + 1];
      ::strcpy((spt_ret->pieces)[counter]->surface, (char*)piece.surface().c_str());
      spt_ret->pieces[counter]->piece = new char[piece.piece().size() + 1];
      ::strcpy((spt_ret->pieces)[counter]->piece, (char*)piece.piece().c_str());
      counter++;
    }
    *outSpt = spt_ret;
    return 0;
  }

public:

  SentencePieceInternal(const uint16_t* modelPath, const uint16_t** vocab, size_t vocabSize) {
    m_processor.reset(new sentencepiece::SentencePieceProcessor());
    // load the model file
    const auto status = m_processor->Load(utf16_to_utf8(utf16string(modelPath)));
    // implant the restricted vocabulary, if given
    if(vocab && vocabSize > 0) {
      std::vector<std::string> vocab_str;
      for(size_t i = 0; i < vocabSize; i++)
        vocab_str.push_back(utf16_to_utf8(utf16string(vocab[i])));

      m_processor->SetVocabulary(vocab_str);
    }
    checkStatus(status, "loading");
  }

  int getPieceID(char* sentence) {
    std::string sentInUtf8(sentence);
    return m_processor->PieceToId(absl::string_view(sentInUtf8));
  }

  int encodeAligned(char* sentence, Native_SentencePieceText** nSpt) {
    sentencepiece::SentencePieceText spt;
    std::string sentInUtf8(sentence);
    m_processor->Encode(absl::string_view(sentInUtf8), &spt);

    return createNativeSentencePieceText(spt, nSpt);
  }

  int decodeAligned(int num_tokens, char** inp_tokens, Native_SentencePieceText** nSpt) {
    sentencepiece::SentencePieceText spt;
    std::vector<std::string> tokens;
    for(int i = 0; i < num_tokens; i++) {
      std::string tok((char*)inp_tokens[i]);
      tokens.push_back(tok);
    }
    m_processor->Decode(tokens, &spt);
    return createNativeSentencePieceText(spt, nSpt);
  }
};

int SentencePieceInteropFreeNativeSentencePieceText(Native_SentencePieceText* spt) {
  auto num_pieces = (*spt).num_pieces;
  for(int i = 0; i < num_pieces; i++) {
    Native_SentencePiecePiece* piece = (*spt).pieces[i];
    delete(piece->surface);
    delete(piece->piece);
    delete(piece);
  }
  delete[]((*spt).pieces);
  delete[]((*spt).text);
  delete(spt);
  spt = NULL;
  return 0;
}

intptr_t SentencePieceInteropLoadModel(const uint16_t* modelPath,
                                       const uint16_t** vocab,
                                       size_t vocabSize) {
  try {
    return (intptr_t) new SentencePieceInternal(modelPath, vocab, vocabSize);
  }
  catch(...) { return (intptr_t) nullptr; }
}

int SentencePieceInteropDecodeAligned(intptr_t object,
                                      int num_tokens,
                                      char** tokens,
                                      Native_SentencePieceText** nSpt) {
  try {
    return ((SentencePieceInternal*)object)->decodeAligned(num_tokens, tokens, nSpt);
  }
  catch(...) { return -1; }
}

int SentencePieceInteropEncodeAligned(intptr_t object,
                                      char* word,
                                      Native_SentencePieceText** nSpt) {
  try {
    return ((SentencePieceInternal*)object)->encodeAligned(word, nSpt);
  }
  catch(...) { return -1; }
}

int SentencePieceInteropGetPieceID(intptr_t object, char* word) {
  try {
    return ((SentencePieceInternal*)object)->getPieceID(word);
  }
  catch(...) { return -1; }
}

int SentencePieceInteropUnloadModel(intptr_t object) {
  delete(SentencePieceInternal*)object;
  return 0;
}

int SentencepieceInteropTrainModel(char* args) {
  std::stringstream command;
  command << std::string(args);
  auto status = sentencepiece::SentencePieceTrainer::Train(command.str());
  return (int)status.code();
}

}  // namespace spm
}  // namespace marian

#endif