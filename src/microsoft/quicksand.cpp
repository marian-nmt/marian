#include "quicksand.h"
#include "marian.h"

#ifdef MKL_FOUND
#include "mkl.h"
#endif

#include "data/shortlist.h"
#include "translator/beam_search.h"
#include "translator/scorers.h"

namespace marian {

namespace quicksand {

template <class T>
void set(Ptr<Options> options, const std::string& key, const T& value) {
  options->set(key, value);
}

template void set(Ptr<Options> options, const std::string& key, const size_t&);
template void set(Ptr<Options> options, const std::string& key, const int&);
template void set(Ptr<Options> options, const std::string& key, const std::string&);
template void set(Ptr<Options> options, const std::string& key, const bool&);
template void set(Ptr<Options> options, const std::string& key, const std::vector<std::string>&);
template void set(Ptr<Options> options, const std::string& key, const float&);
template void set(Ptr<Options> options, const std::string& key, const double&);

Ptr<Options> newOptions() {
  return New<Options>();
}

class BeamSearchDecoder : public IBeamSearchDecoder {
private:
  Ptr<ExpressionGraph> graph_;
  Ptr<cpu::WrappedDevice> device_;

  std::vector<Ptr<Scorer>> scorers_;

public:
  BeamSearchDecoder(Ptr<Options> options,
                    const std::vector<const void*>& ptrs,
                    Word eos)
      : IBeamSearchDecoder(options, ptrs, eos) {
    
    // setting 16-bit optimization to false for now. Re-enable with better caching or pre-computation
    graph_ = New<ExpressionGraph>(/*inference=*/true, /*optimize=*/false);

    DeviceId deviceId{0, DeviceType::cpu};
    device_ = New<cpu::WrappedDevice>(deviceId);
    graph_->setDevice(deviceId, device_);

#ifdef MKL_FOUND
    mkl_set_num_threads(options->get<size_t>("mkl-threads", 1));
#endif

    std::vector<std::string> models
        = options_->get<std::vector<std::string>>("model");

    for(int i = 0; i < models.size(); ++i) {
      Ptr<Options> modelOpts = New<Options>();

      YAML::Node config;
      if(io::isBin(models[i]) && ptrs_[i] != nullptr)
        io::getYamlFromModel(config, "special:model.yml", ptrs_[i]);
      else
        io::getYamlFromModel(config, "special:model.yml", models[i]);

      modelOpts->merge(options_);
      modelOpts->merge(config);

      auto encdec = models::from_options(modelOpts, models::usage::translation);

      if(io::isBin(models[i]) && ptrs_[i] != nullptr) {
        // if file ends in *.bin and has been mapped by QuickSAND
        scorers_.push_back(New<ScorerWrapper>(
            encdec, "F" + std::to_string(scorers_.size()), 1, ptrs[i]));
      } else {
        // it's a *.npz file or has not been mapped by QuickSAND
        scorers_.push_back(New<ScorerWrapper>(
            encdec, "F" + std::to_string(scorers_.size()), 1, models[i]));
      }
    }

    for(auto scorer : scorers_) {
      scorer->init(graph_);
    }
  }

  void setWorkspace(uint8_t* data, size_t size) { device_->set(data, size); }

  QSNBestBatch decode(const QSBatch& qsBatch,
                      size_t maxLength,
                      const std::unordered_set<Word>& shortlist) {
    if(shortlist.size() > 0) {
      auto shortListGen = New<data::FakeShortlistGenerator>(shortlist);
      for(auto scorer : scorers_)
        scorer->setShortlistGenerator(shortListGen);
    }

    // form source batch, by interleaving the words over sentences in the batch, and setting the mask
    size_t batchSize = qsBatch.size();
    auto subBatch = New<data::SubBatch>(batchSize, maxLength, nullptr);
    for(size_t i = 0; i < maxLength; ++i) {
      for(size_t j = 0; j < batchSize; ++j) {
        const auto& sent = qsBatch[j];
        if(i < sent.size()) {
          size_t idx = i * batchSize + j;
          subBatch->data()[idx] = sent[i];
          subBatch->mask()[idx] = 1;
        }
      }
    }
    std::vector<Ptr<data::SubBatch>> subBatches;
    subBatches.push_back(subBatch);
    std::vector<size_t> sentIds(batchSize, 0);

    auto batch = New<data::CorpusBatch>(subBatches);
    batch->setSentenceIds(sentIds);

    // decode
    auto search = New<BeamSearch>(options_, scorers_, eos_);
    auto histories = search->search(graph_, batch);

    // convert to QuickSAND format
    QSNBestBatch qsNbestBatch;
    for(const auto& history : histories) { // loop over batch entries
      QSNBest qsNbest;
      auto nbestHyps = history->NBest(SIZE_MAX); // request as many N as we have
      for (const auto& hyp : nbestHyps) { // loop over N-best entries
        qsNbest.push_back(std::make_tuple(std::get<0>(hyp),
                                          std::get<2>(hyp)));
      }
      qsNbestBatch.push_back(qsNbest);
    }

    return qsNbestBatch;
  }
};

Ptr<IBeamSearchDecoder> newDecoder(Ptr<Options> options,
                                   const std::vector<const void*>& ptrs,
                                   Word eos) {
  return New<BeamSearchDecoder>(options, ptrs, eos);
}

}  // namespace quicksand
}  // namespace marian
