#include "embedder/vector_collector.h"

#include "common/logging.h"
#include "common/utils.h"

#include <iostream>
#include <iomanip>

namespace marian {

// This class manages multi-threaded writing of embedded vectors to stdout or an output file.
// It will either output string versions of float vectors or binary equal length versions depending
// on its binary_ flag.

VectorCollector::VectorCollector(const Ptr<Options>& options)
    : nextId_(0), binary_{options->get<bool>("binary", false)} {
    if(options->get<std::string>("output") == "stdout")
      outStrm_.reset(new std::ostream(std::cout.rdbuf()));
    else
      outStrm_.reset(new io::OutputFileStream(options->get<std::string>("output")));
  }

void VectorCollector::Write(long id, const std::vector<float>& vec) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(id == nextId_) {
    WriteVector(vec);

    ++nextId_;

    typename Outputs::const_iterator iter, iterNext;
    iter = outputs_.begin();
    while(iter != outputs_.end()) {
      long currId = iter->first;

      if(currId == nextId_) {
        // 1st element in the map is the next
        WriteVector(iter->second);
        
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
    outputs_[id] = vec;
  }
}

void VectorCollector::WriteVector(const std::vector<float>& vec) {
  if(binary_) {
    outStrm_->write((char*)vec.data(), vec.size() * sizeof(float));
  } else {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(8);
    for(auto v : vec)
      *outStrm_ << v << " ";
    *outStrm_ << std::endl;
  }
}

}  // namespace marian
