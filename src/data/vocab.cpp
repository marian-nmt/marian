#include "data/vocab.h"
#include "data/default_vocab.h"

namespace marian {

int Vocab::loadOrCreate(const std::string& vocabPath,
                        const std::string& textPath,
                        int max) {
  if(!vImpl_)
    vImpl_ = New<DefaultVocab>();
  vImpl_->loadOrCreate(vocabPath, textPath, max);
}

int Vocab::load(const std::string& vocabPath, int max) {
  if(!vImpl_)
    vImpl_ = New<DefaultVocab>();
  return vImpl_->load(vocabPath, max);
}

void Vocab::create(const std::string& vocabPath, const std::string& trainPath) {
  if(!vImpl_)
    vImpl_ = New<DefaultVocab>();
  vImpl_->create(vocabPath, trainPath);
}

void Vocab::create(io::InputFileStream& trainStrm,
                   io::OutputFileStream& vocabStrm,
                   size_t maxSize) {
  if(!vImpl_)
    vImpl_ = New<DefaultVocab>();
  vImpl_->create(trainStrm, vocabStrm, maxSize);
}

void Vocab::createFake() {
  if(!vImpl_)
    vImpl_ = New<DefaultVocab>();
  vImpl_->createFake();
}

}  // namespace marian
