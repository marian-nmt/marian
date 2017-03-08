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

Corpus::Corpus(Ptr<Config> options)
  : options_(options),
    textPaths_(options_->get<std::vector<std::string>>("train-sets")),
    maxLength_(options_->get<size_t>("max-length")) {

  std::vector<std::string> vocabPaths;
  if(options_->has("vocabs"))
    vocabPaths = options_->get<std::vector<std::string>>("vocabs");

  UTIL_THROW_IF2(!vocabPaths.empty() && textPaths_.size() != vocabPaths.size(),
                 "Number of corpus files and vocab files does not agree");

  std::vector<int> maxVocabs =
    options_->get<std::vector<int>>("dim-vocabs");

  std::vector<Vocab> vocabs;
  if(vocabPaths.empty()) {
    for(int i = 0; i < textPaths_.size(); ++i) {
      Ptr<Vocab> vocab = New<Vocab>();
      vocab->loadOrCreate(textPaths_[i], maxVocabs[i]);
      vocabs_.emplace_back(vocab);
    }
  }
  else {
    for(int i = 0; i < vocabPaths.size(); ++i) {
      Ptr<Vocab> vocab = New<Vocab>();
      vocab->load(vocabPaths[i], maxVocabs[i]);
      vocabs_.emplace_back(vocab);
    }
  }


  for(auto path : textPaths_) {
    files_.emplace_back(new InputFileStream(path));
  }
}

Corpus::Corpus(std::vector<std::string> paths,
               std::vector<Ptr<Vocab>> vocabs,
               Ptr<Config> options)
  : options_(options),
    textPaths_(paths),
    vocabs_(vocabs),
    maxLength_(options_->get<size_t>("max-length")) {

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

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(corpus.begin(), corpus.end(), g);

  std::vector<UPtr<OutputFileStream>> outs;
  for(int i = 0; i < files_.size(); ++i) {
    auto path = files_[i]->path();
    outs.emplace_back(new OutputFileStream(path + ".shuf"));
  }
  files_.clear();

  for(auto& lines : corpus) {
    size_t i = 0;
    for(auto& line : lines) {
      (std::ostream&)*outs[i++] << line << std::endl;
    }

    std::vector<std::string> empty;
    lines.swap(empty);
  }

  for(int i = 0; i < outs.size(); ++i) {
    auto path = outs[i]->path();
    outs[i].reset();
    files_.emplace_back(new InputFileStream(path));
  }

  LOG(data, "Done");
}

}
}
