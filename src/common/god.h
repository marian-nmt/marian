#pragma once
#include <memory>
#include <iostream>
#include <boost/thread/tss.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/locks.hpp>

#include "common/processor/processor.h"
#include "common/config.h"
#include "common/loader.h"
#include "common/logging.h"
#include "common/scorer.h"
#include "common/types.h"
#include "common/base_best_hyps.h"
#include "common/output_collector.h"
#include "common/vocab.h"
#include "common/threadpool.h"
#include "common/file_stream.h"
#include "common/filter.h"
#include "common/processor/bpe.h"
#include "common/utils.h"

namespace amunmt {

class Search;
class Weights;
class Vocab;
class Filter;
class InputFileStream;

class God {
  public:
	God();
    virtual ~God();

    God& Init(const std::string&);
    God& Init(int argc, char** argv);


    bool Has(const std::string& key) const {
      return config_.Has(key);
    }

    template <typename T>
    T Get(const std::string& key) const {
      return config_.Get<T>(key);
    }

    YAML::Node Get(const std::string& key) const {
      return config_.Get(key);
    }

    Vocab& GetSourceVocab(size_t i = 0) const;
    Vocab& GetTargetVocab() const;

    std::istream& GetInputStream() const;
    OutputCollector& GetOutputCollector() const;

    const Filter& GetFilter() const;

    BestHypsBasePtr GetBestHyps(const DeviceInfo &deviceInfo) const;

    std::vector<ScorerPtr> GetScorers(const DeviceInfo &deviceInfo) const;
    std::vector<std::string> GetScorerNames() const;
    const std::map<std::string, float>& GetScorerWeights() const;

    std::vector<std::string> Preprocess(size_t i, const std::vector<std::string>& input) const;
    std::vector<std::string> Postprocess(const std::vector<std::string>& input) const;

    void CleanUp();

    void LoadWeights(const std::string& path);

    DeviceInfo GetNextDevice() const;
    Search &GetSearch() const;

  private:
    void LoadScorers();
    void LoadFiltering();
    void LoadPrePostProcessing();


    Config config_;

    mutable std::vector<std::unique_ptr<Vocab>> sourceVocabs_;
    mutable std::unique_ptr<Vocab> targetVocab_;

    std::unique_ptr<const Filter> filter_;

    std::vector<std::vector<PreprocessorPtr>> preprocessors_;
    std::vector<PostprocessorPtr> postprocessors_;

    typedef std::map<std::string, LoaderPtr> Loaders;
    Loaders cpuLoaders_;
    Loaders gpuLoaders_;
    std::map<std::string, float> weights_;

    std::shared_ptr<spdlog::logger> info_;
    std::shared_ptr<spdlog::logger> progress_;

    mutable std::unique_ptr<InputFileStream> inputStream_;
    mutable OutputCollector outputCollector_;

    mutable size_t threadIncr_;

    mutable boost::thread_specific_ptr<Search> search_;
    mutable boost::shared_mutex accessLock_;
};

}

