#include "output_collector.h"
#include "common/file_stream.h"
#include "common/logging.h"

#include <cassert>

namespace marian {

OutputCollector::OutputCollector()
  : nextId_(0),
    printing_(new DefaultPrinting()) {}

OutputCollector::OutputCollector(std::string outFile)
  : nextId_(0),
    outStrm_(new std::ostream(std::cout.rdbuf())),
    printing_(new DefaultPrinting()) {
  if (outFile != "stdout")
    outStrm_.reset(new io::OutputFileStream(outFile));
}

void OutputCollector::Write(long sourceId,
                            const std::string& best1,
                            const std::string& bestn,
                            bool nbest) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(sourceId == nextId_) {
    if(printing_->shouldBePrinted(sourceId))
      LOG(info, "Best translation {} : {}", sourceId, best1);

    if(outStrm_) {
      if(nbest)
        *outStrm_ << bestn << std::endl;
      else
        *outStrm_ << best1 << std::endl;
    }

    ++nextId_;

    Outputs::const_iterator iter, iterNext;
    iter = outputs_.begin();
    while(iter != outputs_.end()) {
      long currId = iter->first;

      if(currId == nextId_) {
        // 1st element in the map is the next
        const auto& currOutput = iter->second;
        if(printing_->shouldBePrinted(currId))
          LOG(info, "Best translation {} : {}", currId, currOutput.first);

        if(outStrm_) {
          if(nbest)
            *outStrm_ << currOutput.second << std::endl;
          else
            *outStrm_ << currOutput.first << std::endl;
        }

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

    // for 1-best, flush stdout so that we can consume this immediately from an
    // external process
    if(outStrm_ && !nbest)
      *outStrm_ << std::flush;

  } else {
    // save for later
    outputs_[sourceId] = std::make_pair(best1, bestn);
  }
}

StringCollector::StringCollector(bool quiet /*=false*/) : maxId_(-1), quiet_(quiet) {}

void StringCollector::add(long sourceId,
                          const std::string& best1,
                          const std::string& bestn) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(!quiet_)
    LOG(info, "Best translation {} : {}", sourceId, best1);
  outputs_[sourceId] = std::make_pair(best1, bestn);
  if(maxId_ <= sourceId)
    maxId_ = sourceId;
}

std::vector<std::string> StringCollector::collect(bool nbest) {
  std::vector<std::string> outputs;
  for(int id = 0; id <= maxId_; ++id)
    outputs.emplace_back(nbest ? outputs_[id].second : outputs_[id].first);
  return outputs;
}
}  // namespace marian
