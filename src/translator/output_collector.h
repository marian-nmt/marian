#pragma once

#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>
#include <iostream>
#include <map>

#include "common/definitions.h"
#include "common/file_stream.h"

namespace marian {

class OutputCollector {
public:
  OutputCollector();

  template <class T>
  OutputCollector(T&& arg) : nextId_(0), outStrm_(new OutputFileStream(arg)) {}

  OutputCollector(const OutputCollector&) = delete;

  void Write(long sourceId,
             const std::string& best1,
             const std::string& bestn,
             bool nbest);

protected:
  UPtr<OutputFileStream> outStrm_;
  boost::mutex mutex_;
  long nextId_;

  typedef std::map<long, std::pair<std::string, std::string>> Outputs;
  Outputs outputs_;
};
}
