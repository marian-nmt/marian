#pragma once

#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>
#include <iostream>
#include <map>

#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/logging.h"

namespace marian {

template <class Message>
class ScoreCollector {
public:
  ScoreCollector()
  : nextId_(0), outStrm_(new OutputFileStream(std::cout)) {};

  virtual void Write(long id, Message message) {
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


protected:
  long nextId_{0};
  UPtr<OutputFileStream> outStrm_;
  boost::mutex mutex_;

  typedef std::map<long, Message> Outputs;
  Outputs outputs_;
};

class ScoreCollectorNBest : private ScoreCollector<std::string> {
private:
  std::string nBestList_;
  std::string fname_;
  long lastRead_{-1};
  UPtr<InputFileStream> file_;
  std::map<long, std::string> buffer_;

public:
  ScoreCollectorNBest() = delete;
  ScoreCollectorNBest(const std::string& nBestList, const std::string& fname)
  : nBestList_(nBestList), fname_(fname), file_{new InputFileStream(fname_)} {}

  ScoreCollectorNBest(const ScoreCollectorNBest&) = delete;

  virtual void Write(long id, float score) {

    auto iter = buffer_.find(id);
    if(iter == buffer_.end()) {
      ABORT_IF(lastRead_ >= id, "Entry already read but not in buffer");
      std::string line;
      while(std::getline((std::istream&)*file_, line) && lastRead_ < id) {
        lastRead_++;
        iter = buffer_.emplace(id, line).first;
      }
    }

    std::string line = iter->second;
    buffer_.erase(iter);

    ScoreCollector::Write(id, line + " ||| " + fname_ + "= " + std::to_string(score));
  }

};
}
