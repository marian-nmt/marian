#include "data/text_input.h"
#include "common/utils.h"

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

TextInput::TextInput(std::vector<std::string> inputs,
                     std::vector<Ptr<Vocab>> vocabs,
                     Ptr<Config> options)
    // TODO: fix this: input text is stored in an inherited variable named
    // paths_ that is very confusing
    : DatasetBase(inputs),
      vocabs_(vocabs),
      options_(options) {
  for(const auto& text : paths_)
    files_.emplace_back(new std::istringstream(text));
}

SentenceTuple TextInput::next() {
  bool cont = true;
  while(cont) {
    // get index of the current sentence
    size_t curId = pos_++;

    // fill up the sentence tuple with sentences from all input files
    SentenceTuple tup(curId);
    for(size_t i = 0; i < files_.size(); ++i) {
      std::string line;
      io::InputFileStream dummyStream(*files_[i]);
      if(io::getline(dummyStream, line)) {
        Words words = (*vocabs_[i])(line);
        if(words.empty())
          words.push_back(0);
        tup.push_back(words);
      }
    }

    // continue only if each input file has provided an example
    cont = tup.size() == files_.size();
    if(cont)
      return tup;
  }
  return SentenceTuple(0);
}
}  // namespace data
}  // namespace marian
