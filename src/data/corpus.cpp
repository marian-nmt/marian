#include <random>

#include "data/corpus.h"

namespace marian {
namespace data {

typedef std::vector<size_t> WordBatch;
typedef std::vector<float> MaskBatch;
typedef std::pair<WordBatch, MaskBatch> WordMask;
typedef std::vector<WordMask> SentBatch;

typedef std::vector<Words> SentenceTuple;

CorpusIterator::CorpusIterator() : pos_(-1) {}

CorpusIterator::CorpusIterator(Corpus& corpus)
 : corpus_(&corpus), pos_(0) {
  tup_ = corpus_->next();
}

void CorpusIterator::increment() {
  tup_ = corpus_->next();
  pos_++;
}

bool CorpusIterator::equal(CorpusIterator const& other) const
{
  return this->pos_ == other.pos_ ||
    (this->tup_.empty() && other.tup_.empty());
}

const SentenceTuple& CorpusIterator::dereference() const {
  return tup_;
}

Corpus::Corpus(Ptr<Config> options, bool translate)
  : options_(options),
    maxLength_(options_->get<size_t>("max-length")),
    g_(rd_()) {

  if(!translate)
    textPaths_ = options_->get<std::vector<std::string>>("train-sets");
  else
    textPaths_ = options_->get<std::vector<std::string>>("input");

  g_.seed(Config::seed);

  std::vector<std::string> vocabPaths;
  if(options_->has("vocabs"))
    vocabPaths = options_->get<std::vector<std::string>>("vocabs");

  if(!translate) {
    UTIL_THROW_IF2(!vocabPaths.empty() && textPaths_.size() != vocabPaths.size(),
                   "Number of corpus files and vocab files does not agree");
  }

  std::vector<int> maxVocabs =
    options_->get<std::vector<int>>("dim-vocabs");

  if(!translate) {
    std::vector<Vocab> vocabs;
    if(vocabPaths.empty()) {
      for(int i = 0; i < textPaths_.size(); ++i) {
        Ptr<Vocab> vocab = New<Vocab>();
        vocab->loadOrCreate("", textPaths_[i], maxVocabs[i]);
        options_->get()["vocabs"].push_back(textPaths_[i] + ".yml");
        vocabs_.emplace_back(vocab);
      }
    }
    else {
      for(int i = 0; i < vocabPaths.size(); ++i) {
        Ptr<Vocab> vocab = New<Vocab>();
        vocab->loadOrCreate(vocabPaths[i], textPaths_[i], maxVocabs[i]);
        vocabs_.emplace_back(vocab);
      }
    }
  }
  else {
    for(int i = 0; i < vocabPaths.size() - 1; ++i) {
      Ptr<Vocab> vocab = New<Vocab>();
      vocab->loadOrCreate(vocabPaths[i], textPaths_[i], maxVocabs[i]);
      vocabs_.emplace_back(vocab);
    }
  }

  for(auto path : textPaths_) {
    if(path == "stdin")
      files_.emplace_back(new InputFileStream(std::cin));
    else
      files_.emplace_back(new InputFileStream(path));
  }
}

Corpus::Corpus(std::vector<std::string> paths,
               std::vector<Ptr<Vocab>> vocabs,
               Ptr<Config> options,
               size_t maxLength)
  : options_(options),
    textPaths_(paths),
    vocabs_(vocabs),
    maxLength_(maxLength ? maxLength : options_->get<size_t>("max-length")) {

  UTIL_THROW_IF2(textPaths_.size() != vocabs_.size(),
                 "Number of corpus files and vocab files does not agree");

  for(auto path : textPaths_) {
    files_.emplace_back(new InputFileStream(path));
  }

}

SentenceTuple Corpus::next() {
  bool cont = true;
  while(cont) {
    SentenceTuple tup;
    for(int i = 0; i < files_.size(); ++i) {
      std::string line;
      if(std::getline((std::istream&)*files_[i], line)) {
        Words words = (*vocabs_[i])(line);
        if(words.empty())
          words.push_back(0);
        tup.push_back(words);
      }
    }
    cont = tup.size() == files_.size();
    if(cont && std::all_of(tup.begin(), tup.end(),
                           [=](const Words& words) {
                             return words.size() > 0 &&
                             words.size() <= maxLength_;
                            }))
      return tup;
  }
  return SentenceTuple();
}

void Corpus::shuffle() {
  shuffleFiles(textPaths_);
}

void Corpus::reset() {
  files_.clear();
  for(auto& path : textPaths_) {
    if(path == "stdin")
      files_.emplace_back(new InputFileStream(std::cin));
    else
      files_.emplace_back(new InputFileStream(path));
  }
}

void Corpus::shuffleFiles(const std::vector<std::string>& paths) {
  LOG(data, "Shuffling files");
  std::vector<std::vector<std::string>> corpus;

  files_.clear();
  for(auto path : paths) {
    files_.emplace_back(new InputFileStream(path));
  }

  bool cont = true;
  while(cont) {
    std::vector<std::string> lines(files_.size());
    for(int i = 0; i < files_.size(); ++i) {
      cont = cont && std::getline((std::istream&)*files_[i],
                                  lines[i]);
    }
    if(cont)
      corpus.push_back(lines);
  }

  std::shuffle(corpus.begin(), corpus.end(), g_);
  
  tempFiles_.clear();

  std::vector<UPtr<OutputFileStream>> outs;
  for(int i = 0; i < files_.size(); ++i) {
    tempFiles_.emplace_back(new TemporaryFile(options_->get<std::string>("tempdir")));
    outs.emplace_back(new OutputFileStream(*tempFiles_[i]));
  }
  
  for(auto& lines : corpus) {
    size_t i = 0;
    for(auto& line : lines) {
      (std::ostream&)*outs[i++] << line << std::endl;
    }
  }

  files_.clear();
  for(int i = 0; i < outs.size(); ++i) {
    files_.emplace_back(new InputFileStream(*tempFiles_[i]));
  }
  
  LOG(data, "Done");
}

}
}
