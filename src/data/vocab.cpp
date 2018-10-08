#include "data/vocab.h"
#include "data/default_vocab.h"

#ifdef USE_SENTENCEPIECE
#include "data/sentencepiece_vocab.h"
#endif

namespace marian {

Ptr<BaseVocab> vocabFactory(const std::string& vocabPath) {
  bool isSentencePiece = regex::regex_search(vocabPath, regex::regex("\\.(spm)$"));
  if(isSentencePiece) {
#ifdef USE_SENTENCEPIECE
    return New<SentencePieceVocab>();
#else
    ABORT("*.spm suffix in path {} reserved for SentencePiece models, "
          "but support for SentencePiece is not compile into Marian.", 
          vocabPath);
#endif
  }
  return New<DefaultVocab>();
}

int Vocab::loadOrCreate(const std::string& vocabPath,
                        const std::string& textPath,
                        int max) {
  if(!vImpl_)
    vImpl_ = vocabFactory(vocabPath);
  vImpl_->loadOrCreate(vocabPath, textPath, max);
}

int Vocab::load(const std::string& vocabPath, int max) {
  if(!vImpl_)
    vImpl_ = vocabFactory(vocabPath);
  return vImpl_->load(vocabPath, max);
}

void Vocab::create(const std::string& vocabPath, const std::string& trainPath) {
  if(!vImpl_)
    vImpl_ = vocabFactory(vocabPath);
  vImpl_->create(vocabPath, trainPath);
}

void Vocab::create(io::InputFileStream& trainStrm,
                   io::OutputFileStream& vocabStrm,
                   size_t maxSize) {
  if(!vImpl_)
    vImpl_ = New<DefaultVocab>(); // Only DefaultVocab can be built from streams
  vImpl_->create(trainStrm, vocabStrm, maxSize);
}

void Vocab::createFake() {
  if(!vImpl_)
    vImpl_ = New<DefaultVocab>(); // DefaultVocab is OK here
  vImpl_->createFake();
}

}  // namespace marian
