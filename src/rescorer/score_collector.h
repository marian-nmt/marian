#pragma once

#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>
#include <iostream>
#include <map>

#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/logging.h"
#include "common/utils.h"
#include "data/alignment.h"

namespace marian {

class ScoreCollector {
public:
  ScoreCollector(const Ptr<Config>& options)
      : nextId_(0),
        outStrm_(new OutputFileStream(std::cout)),
        alignment_(options->get<std::string>("alignment", "")),
        alignmentThreshold_(getAlignmentThreshold(alignment_)){};

  virtual void Write(long id, const std::string& message) {
    boost::mutex::scoped_lock lock(mutex_);
    if(id == nextId_) {
      ((std::ostream&)*outStrm_) << message << std::endl;

      ++nextId_;

      typename Outputs::const_iterator iter, iterNext;
      iter = outputs_.begin();
      while(iter != outputs_.end()) {
        long currId = iter->first;

        if(currId == nextId_) {
          // 1st element in the map is the next
          ((std::ostream&)*outStrm_) << iter->second << std::endl;

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

  virtual void Write(long id,
                     float score,
                     const data::SoftAlignment& align = {}) {
    auto msg = std::to_string(score);
    if(!alignment_.empty() && !align.empty())
      msg += " ||| " + getAlignment(align);
    Write(id, msg);
  }

protected:
  long nextId_{0};
  UPtr<OutputFileStream> outStrm_;
  boost::mutex mutex_;

  typedef std::map<long, std::string> Outputs;
  Outputs outputs_;

  std::string alignment_;
  float alignmentThreshold_{0.f};

  float getAlignmentThreshold(const std::string& str) {
    try {
      return std::max(std::stof(str), 0.f);
    } catch(...) {
      return 0.f;
    }
  }

  std::string getAlignment(const data::SoftAlignment& align) {
    if(alignment_ == "soft") {
      return data::SoftAlignToString(align, true);
    } else if(alignment_ == "hard") {
      return data::ConvertSoftAlignToHardAlign(align, 1.f, true).toString();
    } else if(alignmentThreshold_ > 0.f) {
      return data::ConvertSoftAlignToHardAlign(align, alignmentThreshold_, true)
          .toString();
    } else {
      ABORT("Unrecognized word alignment type");
    }
    return "";
  }
};

class ScoreCollectorNBest : public ScoreCollector {
private:
  Ptr<Config> options_;

  std::string nBestList_;
  std::string fname_;
  long lastRead_{-1};
  UPtr<InputFileStream> file_;
  std::map<long, std::string> buffer_;

public:
  ScoreCollectorNBest() = delete;

  // TODO: get rid of the options_ attribute
  ScoreCollectorNBest(const Ptr<Config>& options)
      : ScoreCollector(options), options_(options) {
    auto paths = options_->get<std::vector<std::string>>("train-sets");
    nBestList_ = paths.back();
    fname_ = options_->get<std::string>("n-best-feature");
    file_.reset(new InputFileStream(nBestList_));
  }

  ScoreCollectorNBest(const ScoreCollectorNBest&) = delete;

  std::string addToNBest(const std::string nbest,
                         const std::string feature,
                         float score,
                         const data::SoftAlignment& align = {}) {
    std::vector<std::string> fields;
    utils::Split(nbest, fields, "|||");
    std::stringstream ss;
    if(!alignment_.empty() && !align.empty())
      ss << " " << getAlignment(align) << " |||";
    ss << fields[2] << feature << "= " << score << " ";
    fields[2] = ss.str();
    return utils::Join(fields, "|||");
  }

  virtual void Write(long id, float score, const data::SoftAlignment& align) {
    std::string line;
    {
      boost::mutex::scoped_lock lock(mutex_);
      auto iter = buffer_.find(id);
      if(iter == buffer_.end()) {
        ABORT_IF(lastRead_ >= id,
                 "Entry {} < {} already read but not in buffer",
                 id,
                 lastRead_);
        std::string line;
        while(lastRead_ < id && utils::GetLine((std::istream&)*file_, line)) {
          lastRead_++;
          iter = buffer_.emplace(lastRead_, line).first;
        }
      }

      line = iter->second;
      buffer_.erase(iter);
    }

    ScoreCollector::Write(id, addToNBest(line, fname_, score, align));
  }
};
}  // namespace marian
