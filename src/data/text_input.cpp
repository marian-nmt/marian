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
  return this->pos_ == other.pos_ || (!this->tup_.valid() && !other.tup_.valid());
}

const SentenceTuple& TextIterator::dereference() const {
  return tup_;
}

TextInput::TextInput(std::vector<std::string> inputs,
                     std::vector<Ptr<Vocab>> vocabs,
                     Ptr<Options> options)
    : DatasetBase(inputs, options),
      vocabs_(vocabs),
      maxLength_(options_->get<size_t>("max-length")),
      maxLengthCrop_(options_->get<bool>("max-length-crop")) {
  // Note: inputs are automatically stored in the inherited variable named paths_, but these are
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
  SentenceTupleImpl tup(curId);
  for(size_t i = 0; i < files_.size(); ++i) {
    std::string line;
    if(io::getline(*files_[i], line)) {
      Words words = vocabs_[i]->encode(line, /*addEOS=*/true, /*inference=*/inference_);
      if(this->maxLengthCrop_ && words.size() > this->maxLength_) {
        words.resize(maxLength_);
        words.back() = vocabs_.back()->getEosId();  // note: this will not work with class-labels
      }

      ABORT_IF(words.empty(),   "No words (not even EOS) found in string??");
      ABORT_IF(tup.size() != i, "Previous tuple elements are missing.");
      tup.push_back(words);
    }
  }

  if(tup.size() == files_.size()) // check if each input file provided an example
    return SentenceTuple(tup);
  else if(tup.size() == 0) // if no file provided examples we are done
    return SentenceTupleImpl(); // return an empty tuple if above test does not pass();
  else // neither all nor none => we have at least on missing entry
    ABORT("There are missing entries in the text tuples.");
}

}  // namespace data
}  // namespace marian
