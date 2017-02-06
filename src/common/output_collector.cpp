#include <cassert>
#include "output_collector.h"

using namespace std;

namespace amunmt {

OutputCollector::OutputCollector()
 : nextId_(0),
  outStrm_(&std::cout)
{
}

void OutputCollector::Write(long sourceId, const std::string& output)
{
  boost::mutex::scoped_lock lock(mutex_);
  if (sourceId == nextId_) {
    *outStrm_ << output << std::flush;
    ++nextId_;

    Outputs::const_iterator iter, iterNext;
    iter = outputs_.begin();
    while (iter != outputs_.end()) {
      long currId = iter->first;

      if (currId == nextId_) {
        // 1st element in the map is the next
        const string &currOutput = iter->second;
        *outStrm_ << currOutput << std::flush;
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

