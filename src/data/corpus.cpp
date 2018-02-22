#include <random>

#include "data/corpus.h"

namespace marian {
namespace data {

Corpus::Corpus(Ptr<Config> options, bool translate /*= false*/)
    : CorpusBase(options, translate) {}

Corpus::Corpus(std::vector<std::string> paths,
               std::vector<Ptr<Vocab>> vocabs,
               Ptr<Config> options)
    : CorpusBase(paths, vocabs, options) {}

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
        if(i > 0 && i == alignFileIdx_) {
          addAlignmentToSentenceTuple(line, tup);
        } else if(i > 0 && i == weightFileIdx_) {
          addWeightsToSentenceTuple(line, tup);
        } else {
          addWordsToSentenceTuple(line, i, tup);
        }
      }
    }

    // continue only if each input file provides an example
    size_t expectedSize = files_.size();
    if(weightFileIdx_ > 0)
      expectedSize -= 1;
    if(alignFileIdx_ > 0)
      expectedSize -= 1;
    cont = tup.size() == expectedSize;

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

void Corpus::restore(Ptr<TrainingState> ts) {
  setRNGState(ts->seedCorpus);
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
  std::shuffle(ids_.begin(), ids_.end(), eng_);

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
