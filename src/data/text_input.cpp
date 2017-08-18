#include "data/text_input.h"

namespace marian {
namespace data {

TextIterator::TextIterator() : pos_(-1), tup_(0) {}

TextIterator::TextIterator(TextInput& corpus)
    : corpus_(&corpus), pos_(0), tup_(corpus_->next()) {}

void TextIterator::increment() {
  tup_ = corpus_->next();
  pos_++;
}

bool TextIterator::equal(TextIterator const& other) const {
  return this->pos_ == other.pos_ || (this->tup_.empty() && other.tup_.empty());
}

const SentenceTuple& TextIterator::dereference() const {
  return tup_;
}

TextInput::TextInput(std::vector<std::string> paths,
                     std::vector<Ptr<Vocab>> vocabs,
                     Ptr<Config> options,
                     size_t maxLength)
    : DatasetBase(paths),
      vocabs_(vocabs),
      options_(options),
      maxLength_(maxLength ? maxLength : options_->get<size_t>("max-length")) {

  for(auto path : paths_) {
    files_.emplace_back(new std::istringstream(path));
  }
}

SentenceTuple TextInput::next() {
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
      if(std::getline(*files_[i], line)) {
        Words words = (*vocabs_[i])(line);
        if(words.empty())
          words.push_back(0);
        tup.push_back(words);
      }
    }

    // continue only if each input file has provided an example
    cont = tup.size() == files_.size();

    // TODO: we don't need this for TextInput
    // continue if all sentences are no longer than maximum allowed length
    if(cont && std::all_of(tup.begin(), tup.end(), [=](const Words& words) {
         return words.size() > 0 && words.size() <= maxLength_;
       }))
      return tup;
  }
  return SentenceTuple(0);
}

}
}
