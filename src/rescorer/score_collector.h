#pragma once

#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>
#include <iostream>
#include <map>

#include "common/definitions.h"
#include "common/file_stream.h"

namespace marian {

class ScoreCollector {
public:
  ScoreCollector();
  ScoreCollector(const ScoreCollector&) = delete;

  void Write(long id, float score);

protected:
  long nextId_{0};
  UPtr<OutputFileStream> outStrm_;
  boost::mutex mutex_;

  typedef std::map<long, float> Outputs;
  Outputs outputs_;
};
}
