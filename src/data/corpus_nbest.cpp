#include <random>

#include "common/utils.h"
#include "data/corpus_nbest.h"

namespace marian {
namespace data {

CorpusNBest::CorpusNBest(Ptr<Options> options, bool translate /*= false*/)
    : CorpusBase(options, translate) {}

CorpusNBest::CorpusNBest(std::vector<std::string> paths,
                         std::vector<Ptr<Vocab>> vocabs,
                         Ptr<Options> options)
    : CorpusBase(paths, vocabs, options) {}

int numFromNbest(const std::string& line) {
  auto fields = utils::split(line, " ||| ", true);
  ABORT_IF(fields.size() < 4,
           "Too few fields ({}) in line \"{}\", is this a correct n-best list?",
           fields.size(),
           line);
  return std::stoi(fields[0]);
}

std::string lineFromNbest(const std::string& line) {
  auto fields = utils::split(line, " ||| ", true);
  ABORT_IF(fields.size() < 4,
           "Too few fields ({}) in line \"{}\", is this a correct n-best list?",
           fields.size(),
           line);
  return fields[1];
}

SentenceTuple CorpusNBest::next() {
  bool cont = true;
  while(cont) {
    // get index of the current sentence
    size_t curId = pos_;
    // if corpus has been shuffled, ids_ contains sentence indexes
    if(pos_ < ids_.size())
      curId = ids_[pos_];
    pos_++;

    // fill up the sentence tuple with sentences from all input files
    SentenceTupleImpl tup(curId);

    std::string line;
    lastLines_.resize(files_.size() - 1);
    size_t last = files_.size() - 1;

    if(io::getline(*files_[last], line)) {
      int curr_num = numFromNbest(line);
      std::string curr_text = lineFromNbest(line);

      for(size_t i = 0; i < last; ++i) {
        if(curr_num > lastNum_) {
          ABORT_IF(!std::getline(*files_[i], lastLines_[i]),
                   "Too few lines in input {}",
                   i);
        }
        addWordsToSentenceTuple(lastLines_[i], i, tup);
      }
      addWordsToSentenceTuple(curr_text, last, tup);
      lastNum_ = curr_num;
    }

    // continue only if each input file provides an example
    size_t expectedSize = files_.size();

    cont = tup.size() == expectedSize;

    // continue if all sentences are no longer than maximum allowed length
    if(cont && std::all_of(tup.begin(), tup.end(), [=](const Words& words) {
         return words.size() > 0 && words.size() <= maxLength_;
       }))
      return SentenceTuple(tup);
  }

  return SentenceTuple();
}

void CorpusNBest::reset() {
  files_.clear();
  ids_.clear();
  pos_ = 0;
  lastNum_ = -1;
  for(auto& path : paths_) {
    if(path == "stdin")
      files_.emplace_back(new std::istream(std::cin.rdbuf()));
    else
      files_.emplace_back(new io::InputFileStream(path));
  }
}
}  // namespace data
}  // namespace marian
