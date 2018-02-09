#include <random>

#include "data/corpus.h"

namespace marian {
namespace data {

typedef std::vector<size_t> WordBatch;
typedef std::vector<float> MaskBatch;
typedef std::pair<WordBatch, MaskBatch> WordMask;
typedef std::vector<WordMask> SentBatch;

CorpusIterator::CorpusIterator() : pos_(-1), tup_(0) {}

CorpusIterator::CorpusIterator(CorpusBase* corpus)
    : corpus_(corpus), pos_(0), tup_(corpus_->next()) {}

void CorpusIterator::increment() {
  tup_ = corpus_->next();
  pos_++;
}

bool CorpusIterator::equal(CorpusIterator const& other) const {
  return this->pos_ == other.pos_ || (this->tup_.empty() && other.tup_.empty());
}

const SentenceTuple& CorpusIterator::dereference() const {
  return tup_;
}

// @TODO: remove options_ from CorpusBase
CorpusBase::CorpusBase(std::vector<std::string> paths,
                       std::vector<Ptr<Vocab>> vocabs,
                       Ptr<Config> options,
                       size_t maxLength)
    : DatasetBase(paths),
      options_(options),
      vocabs_(vocabs),
      maxLength_(maxLength ? maxLength : options_->get<size_t>("max-length")),
      maxLengthCrop_(options_->get<bool>("max-length-crop")),
      rightLeft_(options_->get<bool>("right-left")) {
  ABORT_IF(paths_.size() != vocabs_.size(),
           "Number of corpus files and vocab files does not agree");

  for(auto path : paths_) {
    files_.emplace_back(new InputFileStream(path));
  }
}

CorpusBase::CorpusBase(Ptr<Config> options, bool translate /*= false*/)
    : options_(options),
      maxLength_(options_->get<size_t>("max-length")),
      maxLengthCrop_(options_->get<bool>("max-length-crop")),
      rightLeft_(options_->get<bool>("right-left")) {
  bool training = !translate;

  if(training)
    paths_ = options_->get<std::vector<std::string>>("train-sets");
  else
    paths_ = options_->get<std::vector<std::string>>("input");

  std::vector<std::string> vocabPaths;
  if(options_->has("vocabs"))
    vocabPaths = options_->get<std::vector<std::string>>("vocabs");

  if(training) {
    ABORT_IF(!vocabPaths.empty() && paths_.size() != vocabPaths.size(),
             "Number of corpus files and vocab files does not agree");
  }

  std::vector<int> maxVocabs = options_->get<std::vector<int>>("dim-vocabs");

  if(training) {  // training or scoring
    std::vector<Vocab> vocabs;

    if(vocabPaths.empty()) {
      if(maxVocabs.size() < paths_.size())
        maxVocabs.resize(paths_.size(), 0);

      // Create vocabs if not provided
      for(size_t i = 0; i < paths_.size(); ++i) {
        Ptr<Vocab> vocab = New<Vocab>();
        int vocSize = vocab->loadOrCreate("", paths_[i], maxVocabs[i]);
        LOG(info,
            "[data] Setting vocabulary size for input {} to {}",
            i,
            vocSize);
        options_->get()["dim-vocabs"][i] = vocSize;

        options_->get()["vocabs"].push_back(paths_[i] + ".yml");
        vocabs_.emplace_back(vocab);
      }
    } else {
      // Load all vocabs
      if(maxVocabs.size() < vocabPaths.size())
        maxVocabs.resize(paths_.size(), 0);

      for(size_t i = 0; i < vocabPaths.size(); ++i) {
        Ptr<Vocab> vocab = New<Vocab>();
        int vocSize
            = vocab->loadOrCreate(vocabPaths[i], paths_[i], maxVocabs[i]);
        LOG(info,
            "[data] Setting vocabulary size for input {} to {}",
            i,
            vocSize);
        options_->get()["dim-vocabs"][i] = vocSize;

        vocabs_.emplace_back(vocab);
      }
    }
  } else {  // i.e., if translating
    ABORT_IF(vocabPaths.empty(),
             "Translating, but vocabularies are not given!");

    if(maxVocabs.size() < vocabPaths.size())
      maxVocabs.resize(paths_.size(), 0);

    for(size_t i = 0; i + 1 < vocabPaths.size(); ++i) {
      Ptr<Vocab> vocab = New<Vocab>();
      int vocSize = vocab->load(vocabPaths[i], maxVocabs[i]);
      LOG(info,
          "[data] Setting vocabulary size for input {} to {}",
          i,
          vocSize);
      options_->get()["dim-vocabs"][i] = vocSize;

      vocabs_.emplace_back(vocab);
    }
  }

  for(auto path : paths_) {
    if(path == "stdin")
      files_.emplace_back(new InputFileStream(std::cin));
    else {
      files_.emplace_back(new InputFileStream(path));
      ABORT_IF(files_.back()->empty(), "File '{}' is empty", path);
    }
  }

  if(training) {
    ABORT_IF(vocabs_.size() != files_.size(),
             "Number of corpus files ({}) and vocab files ({}) does not agree",
             files_.size(),
             vocabs_.size());
  } else {
    ABORT_IF(
        vocabs_.size() != files_.size(),
        "Number of input files ({}) and input vocab files ({}) does not agree",
        files_.size(),
        vocabs_.size());
  }

  // @TODO: check if files exist!
  if(options_->has("guided-alignment")) {
    auto path = options_->get<std::string>("guidedAlignment");
    LOG(info, "[data] Using word alignments from file {}", path);

    alignFileIdx_ = paths_.size();
    paths_.emplace_back(path);
    files_.emplace_back(new InputFileStream(path));
  }
  if(options_->has("data-weighting")) {
    auto path = options_->get<std::string>("data-weighting");
    LOG(info, "[data] Using weights from file {}", path);

    weightFileIdx_ = paths_.size();
    paths_.emplace_back(path);
    files_.emplace_back(new InputFileStream(path));
  }
}
}
}
