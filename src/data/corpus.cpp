#include "data/corpus.h"

#include <numeric>
#include <random>

#include "common/utils.h"
#include "common/filesystem.h"

#include "data/corpus.h"

namespace marian {
namespace data {

Corpus::Corpus(Ptr<Options> options, bool translate /*= false*/)
    : CorpusBase(options, translate),
        shuffleInRAM_(options_->get<bool>("shuffle-in-ram", false)),
        allCapsEvery_(options_->get<size_t>("all-caps-every", 0)),
        titleCaseEvery_(options_->get<size_t>("english-title-case-every", 0)) {}

Corpus::Corpus(std::vector<std::string> paths,
               std::vector<Ptr<Vocab>> vocabs,
               Ptr<Options> options)
    : CorpusBase(paths, vocabs, options),
        shuffleInRAM_(options_->get<bool>("shuffle-in-ram", false)),
        allCapsEvery_(options_->get<size_t>("all-caps-every", 0)),
        titleCaseEvery_(options_->get<size_t>("english-title-case-every", 0)) {}

void Corpus::preprocessLine(std::string& line, size_t streamId) {
  if (allCapsEvery_ != 0 && pos_ % allCapsEvery_ == 0 && !inference_) {
    line = vocabs_[streamId]->toUpper(line);
    if (streamId == 0)
      LOG_ONCE(info, "[data] Source all-caps'ed line to: {}", line);
    else
      LOG_ONCE(info, "[data] Target all-caps'ed line to: {}", line);
  }
  else if (titleCaseEvery_ != 0 && pos_ % titleCaseEvery_ == 1 && !inference_ && streamId == 0) {
    // Only applied to stream 0 (source) since this feature is aimed at robustness against
    // title case in the source (and not at translating into title case).
    // Note: It is user's responsibility to not enable this if the source language is not English.
    line = vocabs_[streamId]->toEnglishTitleCase(line);
    if (streamId == 0)
      LOG_ONCE(info, "[data] Source English-title-case'd line to: {}", line);
    else
      LOG_ONCE(info, "[data] Target English-title-case'd line to: {}", line);
  }
}

SentenceTuple Corpus::next() {
  for(;;) { // (this is a retry loop for skipping invalid sentences)
    // get index of the current sentence
    size_t curId = pos_; // note: at end, pos_  == total size
    // if corpus has been shuffled, ids_ contains sentence indexes
    if(pos_ < ids_.size())
      curId = ids_[pos_];
    pos_++;

    // fill up the sentence tuple with sentences from all input files
    SentenceTuple tup(curId);
    size_t eofsHit = 0;
    size_t numStreams = corpusInRAM_.empty() ? files_.size() : corpusInRAM_.size();
    for(size_t i = 0; i < numStreams; ++i) {
      std::string line;

      // fetch line, from cached copy in RAM or actual file
      if (!corpusInRAM_.empty()) {
        if (curId < corpusInRAM_[i].size())
          line = corpusInRAM_[i][curId];
        else {
          eofsHit++;
          continue;
        }
      }
      else {
        bool gotLine = io::getline(*files_[i], line).good();
        if(!gotLine) {
          eofsHit++;
          continue;
        }
      }

      if(i > 0 && i == alignFileIdx_) { // @TODO: alignFileIdx == 0 possible?
        addAlignmentToSentenceTuple(line, tup);
      } else if(i > 0 && i == weightFileIdx_) {
        addWeightsToSentenceTuple(line, tup);
      } else {
        preprocessLine(line, i);
        addWordsToSentenceTuple(line, i, tup);
      }
    }

    if (eofsHit == numStreams)
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

// reset and initialize shuffled reading
// Call either reset() or shuffle().
// @TODO: merge with reset() below to clarify mutual exclusiveness with reset()
void Corpus::shuffle() {
  shuffleData(paths_);
}

// reset to regular, non-shuffled reading
// Call either reset() or shuffle().
// @TODO: make shuffle() private, instad pass a shuffle() flag to reset(), to clarify mutual exclusiveness with shuffle()
void Corpus::reset() {
  corpusInRAM_.clear();
  ids_.clear();
  if (pos_ == 0) // no data read yet
    return;
  pos_ = 0;
  for (size_t i = 0; i < paths_.size(); ++i) {
      if(paths_[i] == "stdin") {
        files_[i].reset(new std::istream(std::cin.rdbuf()));
        // Probably not necessary, unless there are some buffers
        // that we want flushed.
      }
      else {
        ABORT_IF(files_[i] && filesystem::is_fifo(paths_[i]),
                 "File '", paths_[i], "' is a pipe and cannot be re-opened.");
        // Do NOT reset named pipes; that closes them and triggers a SIGPIPE
        // (lost pipe) at the writing end, which may do whatever it wants
        // in this situation.
        files_[i].reset(new io::InputFileStream(paths_[i]));
      }
    }
}

void Corpus::restore(Ptr<TrainingState> ts) {
  setRNGState(ts->seedCorpus);
}

void Corpus::shuffleData(const std::vector<std::string>& paths) {
  LOG(info, "[data] Shuffling data");

  size_t numStreams = paths.size();

  size_t numSentences;
  std::vector<std::vector<std::string>> corpus(numStreams); // [stream][id]
  if (!corpusInRAM_.empty()) { // when caching, we use what we have instead
    corpus = std::move(corpusInRAM_); // temporarily move ownership here, will be moved back
    numSentences = corpus[0].size();
  }
  else {
    files_.resize(numStreams);
    for(size_t i = 0; i < numStreams; ++i) {
      UPtr<io::InputFileStream> strm(new io::InputFileStream(paths[i]));
      strm->setbufsize(10000000);  // huge read-ahead buffer to avoid network round-trips
      files_[i] = std::move(strm);
    }

    // read entire corpus into RAM
    std::string lineBuf;
    for (;;) {
      size_t eofsHit = 0;
      for(size_t i = 0; i < numStreams; ++i) {
        bool gotLine = io::getline(*files_[i], lineBuf).good();
        if (gotLine)
          corpus[i].push_back(lineBuf);
        else
          eofsHit++;
      }
      if (eofsHit == numStreams)
        break;
      ABORT_IF(eofsHit != 0, "Not all input files have the same number of lines");
    }
    files_.clear();
    numSentences = corpus[0].size();
    LOG(info, "[data] Done reading {} sentences", numSentences);
  }

  // randomize sequence ids, and remember them
  ids_.resize(numSentences);
  std::iota(ids_.begin(), ids_.end(), 0);
  std::shuffle(ids_.begin(), ids_.end(), eng_);

  if (shuffleInRAM_) {
    // when shuffling in RAM, we keep no files_, instead but the data itself
    corpusInRAM_ = std::move(corpus);
    LOG(info, "[data] Done shuffling {} sentences (cached in RAM)", numSentences);
  }
  else {
    // create temp files that contain the data in randomized order
    tempFiles_.resize(numStreams);
    for(size_t i = 0; i < numStreams; ++i) {
      tempFiles_[i].reset(new io::TemporaryFile(options_->get<std::string>("tempdir")));
      io::TemporaryFile &out = *tempFiles_[i];
      const auto& corpusStream = corpus[i];
      for(auto id : ids_) {
        out << corpusStream[id] << std::endl;
      }
    }

    // replace files_[] by the tempfiles we just created
    files_.resize(numStreams);
    for(size_t i = 0; i < numStreams; ++i) {
      auto inputStream = tempFiles_[i]->getInputStream();
      inputStream->setbufsize(10000000);
      files_[i] = std::move(inputStream);
    }
    LOG(info, "[data] Done shuffling {} sentences to temp files", numSentences);
  }
  pos_ = 0;
}

CorpusBase::batch_ptr Corpus::toBatch(const std::vector<Sample>& batchVector) {
  size_t batchSize = batchVector.size();

  std::vector<size_t> sentenceIds;

  std::vector<int> maxDims;      // @TODO: What's this? widths? maxLengths?
  for(auto& ex : batchVector) {  // @TODO: rename 'ex' to 'sample' or 'sentenceTuple'
    if(maxDims.size() < ex.size())
      maxDims.resize(ex.size(), 0);
    for(size_t i = 0; i < ex.size(); ++i) {
      if(ex[i].size() > (size_t)maxDims[i])
        maxDims[i] = (int)ex[i].size();
    }
    sentenceIds.push_back(ex.getId());
  }

  std::vector<Ptr<SubBatch>> subBatches;
  for(size_t j = 0; j < maxDims.size(); ++j) {
    subBatches.emplace_back(New<SubBatch>(batchSize, maxDims[j], vocabs_[j]));
  }

  std::vector<size_t> words(maxDims.size(), 0);
  for(size_t b = 0; b < batchSize; ++b) {                    // loop over batch entries
    for(size_t j = 0; j < maxDims.size(); ++j) {             // loop over streams
      auto subBatch = subBatches[j];
      for(size_t s = 0; s < batchVector[b][j].size(); ++s) { // loop over word positions
        subBatch->data()[subBatch->locate(/*batchIdx=*/b, /*wordPos=*/s)/*s * batchSize + b*/] = batchVector[b][j][s];
        subBatch->mask()[subBatch->locate(/*batchIdx=*/b, /*wordPos=*/s)/*s * batchSize + b*/] = 1.f;
        words[j]++;
      }
    }
  }

  for(size_t j = 0; j < maxDims.size(); ++j)
    subBatches[j]->setWords(words[j]);

  auto batch = batch_ptr(new batch_type(subBatches));
  batch->setSentenceIds(sentenceIds);

  if(options_->get("guided-alignment", std::string("none")) != "none" && alignFileIdx_)
    addAlignmentsToBatch(batch, batchVector);
  if(options_->hasAndNotEmpty("data-weighting") && weightFileIdx_)
    addWeightsToBatch(batch, batchVector);

  return batch;
}

}  // namespace data
}  // namespace marian
