#include "output_collector.h"
#include <cassert>
#include "common/file_stream.h"
#include "common/logging.h"

namespace marian {

OutputCollector::OutputCollector()
    : nextId_(0), outStrm_(new OutputFileStream(std::cout)) {}

void OutputCollector::Write(long sourceId,
                            const std::string& best1,
                            const std::string& bestn,
                            bool nbest) {
  boost::mutex::scoped_lock lock(mutex_);
  if(sourceId == nextId_) {
    LOG(translate)->info("Best translation {} : {}", sourceId, best1);

    if(nbest)
      ((std::ostream&)*outStrm_) << bestn << std::endl;
    else
      ((std::ostream&)*outStrm_) << best1 << std::endl;

    ++nextId_;

    Outputs::const_iterator iter, iterNext;
    iter = outputs_.begin();
    while(iter != outputs_.end()) {
      long currId = iter->first;

      if(currId == nextId_) {
        // 1st element in the map is the next
        const auto& currOutput = iter->second;
        LOG(translate)->info("Best translation {} : {}", currId, currOutput.first);
        if(nbest)
          ((std::ostream&)*outStrm_) << currOutput.second << std::endl;
        else
          ((std::ostream&)*outStrm_) << currOutput.first << std::endl;

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
    outputs_[sourceId] = std::make_pair(best1, bestn);
  }
}
}
