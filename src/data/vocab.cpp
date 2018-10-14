#include "data/vocab.h"
#include "data/vocab_impl.h"
#include "data/default_vocab.h"

#ifdef USE_SENTENCEPIECE
#include "data/sentencepiece_vocab.h"
#endif

namespace marian {

// @TODO: make each vocab peek on type
Ptr<VocabImpl> vocabFactory(const std::string& vocabPath, Ptr<Options> options, size_t batchIndex) {
  bool isSentencePiece = regex::regex_search(vocabPath, regex::regex("\\.(spm)$"));
  if(isSentencePiece) {
#ifdef USE_SENTENCEPIECE
    return New<SentencePieceVocab>(options, batchIndex);
#else
    ABORT("*.spm suffix in path {} reserved for SentencePiece models, "
          "but support for SentencePiece is not compiled into Marian. "
          "Try to recompile after `cmake .. -DUSE_SENTENCEPIECE=on [...]`",
          vocabPath);
#endif
  }
  return New<DefaultVocab>();
}

int Vocab::loadOrCreate(const std::string& vocabPath,
                        const std::string& trainPath,
                        int max) {
  size_t size = 0;
  if(vocabPath.empty()) {
    // No vocabulary path was given, attempt to first find a vocabulary
    // for trainPath + possible suffixes. If not found attempt to create
    // as trainPath + canonical suffix.

    LOG(info,
        "No vocabulary path given; "
        "trying to find default vocabulary based on data path {}",
        trainPath);

    vImpl_ = New<DefaultVocab>();
    size = vImpl_->findAndLoad(trainPath, max);

    if(size == 0) {
      auto path = trainPath + vImpl_->canonicalSuffix();
      LOG(info,
          "No vocabulary path given; "
          "trying to find vocabulary based on data path {}",
          trainPath);
      vImpl_->create(path, trainPath);
      size = vImpl_->load(path, max);
    }
  } else {
    if(!filesystem::exists(vocabPath)) {
      // Vocabulary path was given, but no vocabulary present,
      // attempt to create in specified location.
      create(vocabPath, trainPath);
    }
    // Vocabulary path exists, attempting to load
    size = load(vocabPath, max);
  }
  LOG(info, "[data] Setting vocabulary size for input {} to {}", batchIndex_, size);
  return size;
}

int Vocab::load(const std::string& vocabPath, int max) {
  if(!vImpl_)
    vImpl_ = vocabFactory(vocabPath, options_, batchIndex_);
  return vImpl_->load(vocabPath, max);
}

void Vocab::create(const std::string& vocabPath, const std::string& trainPath) {
  if(!vImpl_)
    vImpl_ = vocabFactory(vocabPath, options_, batchIndex_);
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
