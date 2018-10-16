#include "data/corpus.h"

#include <numeric>
#include <random>

#include "common/utils.h"
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
  for (;;) { // (this is a retry loop for skipping invalid sentences)
    // get index of the current sentence
    size_t curId = pos_;
    // if corpus has been shuffled, ids_ contains sentence indexes
    if(pos_ < ids_.size())
      curId = ids_[pos_];
    pos_++;

    // fill up the sentence tuple with sentences from all input files
    SentenceTuple tup(curId);
    size_t eofsHit = 0;
    for(size_t i = 0; i < files_.size(); ++i) {
      std::string line;

      bool gotLine = io::getline(*files_[i], line);
      if(gotLine) {
        if(i > 0 && i == alignFileIdx_) { // @TODO: alignFileIdx == 0 possible?
          addAlignmentToSentenceTuple(line, tup);
        } else if(i > 0 && i == weightFileIdx_) {
          addWeightsToSentenceTuple(line, tup);
        } else {
          addWordsToSentenceTuple(line, i, tup);
        }
      }
      else
        eofsHit++;
    }

    if (eofsHit == files_.size())
      return SentenceTuple(0);
    ABORT_IF(eofsHit != 0, "not all input files have the same number of lines");

    // check if all streams are valid, that is, non-empty and no longer than maximum allowed length
    if(std::all_of(tup.begin(), tup.end(), [=](const Words& words) {
         return words.size() > 0 && words.size() <= maxLength_;
       }))
      return tup;

    // otherwise skip this sentence and try the next one
    // @TODO: tail recursion?
  }
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
      files_.emplace_back(new io::InputFileStream(std::cin));
    else
      files_.emplace_back(new io::InputFileStream(path));
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
    files_.emplace_back(new io::InputFileStream(path));
  }

  bool cont = true;
  while(cont) {
    std::vector<std::string> lines(files_.size());
    for(size_t i = 0; i < files_.size(); ++i) {
      cont = cont && io::getline(*files_[i], lines[i]);
    }
    if(cont)
      corpus.push_back(lines);
  }

  pos_ = 0;
  ids_.resize(corpus.size());
  std::iota(ids_.begin(), ids_.end(), 0);
  std::shuffle(ids_.begin(), ids_.end(), eng_);

  tempFiles_.clear();

  std::vector<UPtr<io::OutputFileStream>> outs;
  for(size_t i = 0; i < files_.size(); ++i) {
    tempFiles_.emplace_back(
        new io::TemporaryFile(options_->get<std::string>("tempdir")));
    outs.emplace_back(new io::OutputFileStream(*tempFiles_[i]));
  }

  for(auto id : ids_) {
    auto& lines = corpus[id];
    size_t i = 0;
    for(auto& line : lines) {
      *outs[i++] << line << std::endl;
    }
  }

  files_.clear();
  for(size_t i = 0; i < outs.size(); ++i) {
    files_.emplace_back(new io::InputFileStream(*tempFiles_[i]));
  }

  LOG(info, "[data] Done");
}
}  // namespace data
}  // namespace marian
