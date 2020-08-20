#pragma once

#include "common/options.h"
#include "common/definitions.h"
#include "common/file_stream.h"

#include <map>
#include <mutex>

namespace marian {

// This class manages multi-threaded writing of embedded vectors to stdout or an output file.
// It will either output string versions of float vectors or binary equal length versions depending
// on its binary_ flag.
class VectorCollector {
public:
  VectorCollector(const Ptr<Options>& options);
  virtual ~VectorCollector() {}
  
  virtual void Write(long id, const std::vector<float>& vec);

protected:
  long nextId_{0};
  UPtr<std::ostream> outStrm_;
  bool binary_; // output binary floating point vectors if set

  std::mutex mutex_;

  typedef std::map<long, std::vector<float>> Outputs;
  Outputs outputs_;

  virtual void WriteVector(const std::vector<float>& vec);
};
}  // namespace marian
