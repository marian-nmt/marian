#pragma once

#include "common/options.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "data/alignment.h"

#include <map>
#include <mutex>

namespace marian {

class ScoreCollector {
public:
  ScoreCollector(const Ptr<Options>& options);
  virtual ~ScoreCollector() {}

  virtual void Write(long id, const std::string& message);
  virtual void Write(long id,
                     float score,
                     const data::SoftAlignment& align = {},
                     const std::vector<float>& wordScores = {});

protected:
  long nextId_{0};
  UPtr<std::ostream> outStrm_;
  std::mutex mutex_;

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

  ScoreCollectorNBest(const Ptr<Options>& options);
  ScoreCollectorNBest(const ScoreCollectorNBest&) = delete;

  virtual void Write(long id,
                     float score,
                     const data::SoftAlignment& align = {},
                     const std::vector<float>& wordScores = {}) override;

private:
  std::string nBestList_;
  std::string fname_;
  long lastRead_{-1};
  UPtr<io::InputFileStream> file_;
  std::map<long, std::string> buffer_;

  std::string addToNBest(const std::string nbest,
                         const std::string feature,
                         float score,
                         const data::SoftAlignment& align = {},
                         const std::vector<float>& wordScores = {});
};
}  // namespace marian
