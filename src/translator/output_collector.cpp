#include <cassert>
#include "output_collector.h"
#include "common/logging.h"
#include "common/file_stream.h"

namespace marian {

OutputCollector::OutputCollector()
 : nextId_(0), outStrm_(new OutputFileStream(std::cout)) {}

void OutputCollector::Write(long sourceId, const std::string& output)
{
  boost::mutex::scoped_lock lock(mutex_);
  if (sourceId == nextId_) {
    LOG(translate, "Best translation {} : {}", sourceId, output);
    ((std::ostream&)*outStrm_) << output << std::endl;
    
    ++nextId_;

    Outputs::const_iterator iter, iterNext;
    iter = outputs_.begin();
    while (iter != outputs_.end()) {
      long currId = iter->first;

      if (currId == nextId_) {
        // 1st element in the map is the next
        const std::string &currOutput = iter->second;
        LOG(translate, "Best translation {} : {}", currId, currOutput);
        ((std::ostream&)*outStrm_) << currOutput << std::endl;

        ++nextId_;

        // delete current record, move iter on 1
        iterNext = iter;
        ++iterNext;
        outputs_.erase(iter);
        iter = iterNext;
      }
      else {
        // not the next. stop iterating
        assert(nextId_ < currId);
        break;
      }
    }

  }
  else {
    // save for later
    outputs_[sourceId] = output;
  }
}

}

