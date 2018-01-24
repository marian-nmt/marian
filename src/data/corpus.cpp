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

Corpus::Corpus(Ptr<Config> options, bool translate)
    : options_(options),
      maxLength_(options_->get<size_t>("max-length")),
      maxLengthCrop_(options_->get<bool>("max-length-crop")),
      rightLeft_(options_->get<bool>("right-left")),
      g_(Config::seed) {
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

  if(training) { // training or scoring
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
    ABORT_IF(vocabPaths.empty(), "Translating, but vocabularies are not given!");

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
             files_.size(), vocabs_.size());
  }
  else {
    ABORT_IF(vocabs_.size() != files_.size(),
             "Number of input files ({}) and input vocab files ({}) does not agree",
             files_.size(), vocabs_.size());
  }
}

Corpus::Corpus(std::vector<std::string> paths,
               std::vector<Ptr<Vocab>> vocabs,
               Ptr<Config> options,
               size_t maxLength)
    : CorpusBase(paths),
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

SentenceTuple Corpus::next() {
  bool cont = true;
  while(cont) {
    // get index of the current sentence
    size_t curId = pos_;
    // if corpus has been shuffled, ids_ contains sentence indexes
    if(pos_ < ids_.size())
      curId = ids_[pos_];
    pos_++;

    // fill up the sentence tuple with sentences from all input files
    SentenceTuple tup(curId);
    for(size_t i = 0; i < files_.size(); ++i) {
      std::string line;
      if(std::getline((std::istream&)*files_[i], line)) {
        Words words = (*vocabs_[i])(line);

        if(words.empty())
          words.push_back(0);

        if(maxLengthCrop_ && words.size() > maxLength_) {
          words.resize(maxLength_);
          words.back() = 0;
        }
        
        if(rightLeft_)
          std::reverse(words.begin(), words.end() - 1);

        tup.push_back(words);
      }
    }

    // continue only if each input file has provided an example
    cont = tup.size() == files_.size();

    // continue if all sentences are no longer than maximum allowed length
    if(cont && std::all_of(tup.begin(), tup.end(), [=](const Words& words) {
         return words.size() > 0 && words.size() <= maxLength_;
       }))
      return tup;
  }
  return SentenceTuple(0);
}

void Corpus::shuffle() {
  shuffleFiles(paths_);
}

void Corpus::reset() {
  files_.clear();
  ids_.clear();
  pos_ = 0;
  for(auto& path : paths_) {
    if(path == "stdin")
      files_.emplace_back(new InputFileStream(std::cin));
    else
      files_.emplace_back(new InputFileStream(path));
  }
}

void Corpus::shuffleFiles(const std::vector<std::string>& paths) {
  LOG(info, "[data] Shuffling files");

  std::vector<std::vector<std::string>> corpus;

  files_.clear();
  for(auto path : paths) {
    files_.emplace_back(new InputFileStream(path));
  }

  bool cont = true;
  while(cont) {
    std::vector<std::string> lines(files_.size());
    for(size_t i = 0; i < files_.size(); ++i) {
      cont = cont && std::getline((std::istream&)*files_[i], lines[i]);
    }
    if(cont)
      corpus.push_back(lines);
  }

  pos_ = 0;
  ids_.resize(corpus.size());
  std::iota(ids_.begin(), ids_.end(), 0);
  std::shuffle(ids_.begin(), ids_.end(), g_);

  tempFiles_.clear();

  std::vector<UPtr<OutputFileStream>> outs;
  for(size_t i = 0; i < files_.size(); ++i) {
    tempFiles_.emplace_back(
        new TemporaryFile(options_->get<std::string>("tempdir")));
    outs.emplace_back(new OutputFileStream(*tempFiles_[i]));
  }

  for(auto id : ids_) {
    auto& lines = corpus[id];
    size_t i = 0;
    for(auto& line : lines) {
      (std::ostream&)*outs[i++] << line << std::endl;
    }
  }

  files_.clear();
  for(size_t i = 0; i < outs.size(); ++i) {
    files_.emplace_back(new InputFileStream(*tempFiles_[i]));
  }

  LOG(info, "[data] Done");
}
}
}
