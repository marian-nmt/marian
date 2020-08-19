#include "rescorer/score_collector.h"

#include "common/logging.h"
#include "common/utils.h"

#include <iostream>

namespace marian {

ScoreCollector::ScoreCollector(const Ptr<Options>& options)
    : nextId_(0),
      alignment_(options->get<std::string>("alignment", "")),
      alignmentThreshold_(getAlignmentThreshold(alignment_)) {

    if(options->get<std::string>("output") == "stdout")
      outStrm_.reset(new std::ostream(std::cout.rdbuf()));
    else
      outStrm_.reset(new io::OutputFileStream(options->get<std::string>("output")));
  }

void ScoreCollector::Write(long id, const std::string& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(id == nextId_) {
    *outStrm_ << message << std::endl;

    ++nextId_;

    typename Outputs::const_iterator iter, iterNext;
    iter = outputs_.begin();
    while(iter != outputs_.end()) {
      long currId = iter->first;

      if(currId == nextId_) {
        // 1st element in the map is the next
        *outStrm_ << iter->second << std::endl;

        ++nextId_;

        // delete current record, move iter on 1
        iterNext = iter;
        ++iterNext;
        outputs_.erase(iter);
        iter = iterNext;
      } else {
        // not the next. stop iterating
        assert(nextId_ < currId);
        break;
      }
    }

  } else {
    // save for later
    outputs_[id] = message;
  }
}

void ScoreCollector::Write(long id,
                           float score,
                           const data::SoftAlignment& align /*= {}*/,
                           const std::vector<float>& wordScores /*= {}*/) {
  auto msg = std::to_string(score);
  if(!alignment_.empty() && !align.empty())
    msg += " ||| " + getAlignment(align);
  if(!wordScores.empty())
    msg += " ||| WordScores= " + utils::join(wordScores, " ");
  Write(id, msg);
}

std::string ScoreCollector::getAlignment(const data::SoftAlignment& align) {
  if(alignment_ == "soft") {
    return data::SoftAlignToString(align);
  } else if(alignment_ == "hard") {
    return data::ConvertSoftAlignToHardAlign(align, 1.f).toString();
  } else if(alignmentThreshold_ > 0.f) {
    return data::ConvertSoftAlignToHardAlign(align, alignmentThreshold_)
        .toString();
  } else {
    ABORT("Unrecognized word alignment type");
  }
}

ScoreCollectorNBest::ScoreCollectorNBest(const Ptr<Options>& options)
    : ScoreCollector(options),
      nBestList_(options->get<std::vector<std::string>>("train-sets").back()),
      fname_(options->get<std::string>("n-best-feature")) {
  file_.reset(new io::InputFileStream(nBestList_));
}

void ScoreCollectorNBest::Write(long id,
                                float score,
                                const data::SoftAlignment& align /*= {}*/,
                                const std::vector<float>& wordScores /*= {}*/) {
  std::string line;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = buffer_.find(id);
    if(iter == buffer_.end()) {
      ABORT_IF(lastRead_ >= id,
               "Entry {} < {} already read but not in buffer",
               id,
               lastRead_);
      while(lastRead_ < id && io::getline(*file_, line)) {
        lastRead_++;
        iter = buffer_.emplace(lastRead_, line).first;
      }
    }

    line = iter->second;
    buffer_.erase(iter);
  }

  ScoreCollector::Write(id, addToNBest(line, fname_, score, align, wordScores));
}

std::string ScoreCollectorNBest::addToNBest(const std::string nbest,
                                            const std::string feature,
                                            float score,
                                            const data::SoftAlignment& align /*= {}*/,
                                            const std::vector<float>& wordScores /*= {}*/) {
  auto fields = utils::split(nbest, "|||");
  std::stringstream ss;
  if(!alignment_.empty() && !align.empty())
    ss << " " << getAlignment(align) << " |||";
  if(!wordScores.empty())
    ss << " WordScores= " + utils::join(wordScores, " ") << " |||";
  ss << fields[2] << feature << "= " << score << " ";
  fields[2] = ss.str();
  return utils::join(fields, "|||");
}

}  // namespace marian
