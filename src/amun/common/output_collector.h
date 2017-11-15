#pragma once

#include <iostream>
#include <map>
#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>

namespace amunmt {

class OutputCollector {
 public:
  OutputCollector();
  OutputCollector(const OutputCollector&) = delete;

  void Write(long sourceId, const std::string& output);

 protected:
  std::ostream* outStrm_;
  boost::mutex mutex_;
  long nextId_;

  typedef std::map<long, std::string> Outputs;
  Outputs outputs_;

};

}

