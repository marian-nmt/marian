#include "data/text_input.h"
#include "common/utils.h"

namespace marian {
namespace data {

TextIterator::TextIterator() : pos_(-1), tup_(0) {}
TextIterator::TextIterator(TextInput& corpus) : corpus_(&corpus), pos_(0), tup_(corpus_->next()) {}

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
                     Ptr<Options> options)
    : DatasetBase(inputs, options), vocabs_(vocabs) {
  // note: inputs are automatically stored in the inherited variable named paths_, but these are
  // texts not paths!
  for(const auto& text : paths_)
    files_.emplace_back(new std::istringstream(text));
}

// TextInput is mainly used for inference in the server mode, not for training, so skipping too long
// or ill-formed inputs is not necessary here
SentenceTuple TextInput::next() {
  // get index of the current sentence
  size_t curId = pos_++;

  // fill up the sentence tuple with source and/or target sentences
  SentenceTuple tup(curId);
  for(size_t i = 0; i < files_.size(); ++i) {
    std::string line;
    if(io::getline(*files_[i], line)) {
      Words words = vocabs_[i]->encode(line, /*addEOS =*/ true, /*inference =*/ inference_);
      if(words.empty())
        words.push_back(Word::ZERO); // @TODO: What is this for? @BUGBUG: addEOS=true, so this can never happen, right?
      tup.push_back(words);
    }
  }

  // check if each input file provided an example
  if(tup.size() == files_.size())
    return tup;
  return SentenceTuple(0);
}

}  // namespace data
}  // namespace marian
