#pragma once

#include <boost/thread/mutex.hpp>
#include <map>

#include "common/config.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "data/alignment.h"

namespace marian {

class ScoreCollector {
public:
  ScoreCollector(const Ptr<Config>& options);

  virtual void Write(long id, const std::string& message);
  virtual void Write(long id,
                     float score,
                     const data::SoftAlignment& align = {});

protected:
  long nextId_{0};
  UPtr<OutputFileStream> outStrm_;
  boost::mutex mutex_;

  typedef std::map<long, std::string> Outputs;
  Outputs outputs_;

  std::string alignment_;
  float alignmentThreshold_{0.f};

  std::string getAlignment(const data::SoftAlignment& align);

  float getAlignmentThreshold(const std::string& str) {
    try {
      return std::max(std::stof(str), 0.f);
    } catch(...) {
      return 0.f;
    }
  }
};

class ScoreCollectorNBest : public ScoreCollector {
public:
  ScoreCollectorNBest() = delete;

  ScoreCollectorNBest(const Ptr<Config>& options);
  ScoreCollectorNBest(const ScoreCollectorNBest&) = delete;

  virtual void Write(long id,
                     float score,
                     const data::SoftAlignment& align = {}) override;

private:
  std::string nBestList_;
  std::string fname_;
  long lastRead_{-1};
  UPtr<InputFileStream> file_;
  std::map<long, std::string> buffer_;

  std::string addToNBest(const std::string nbest,
                         const std::string feature,
                         float score,
                         const data::SoftAlignment& align = {});
};
}  // namespace marian
