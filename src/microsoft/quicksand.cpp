#include "quicksand.h"
#include "marian.h"

#ifdef MKL_FOUND
#include "mkl.h"
#endif

#include "data/shortlist.h"
#include "translator/beam_search.h"
#include "translator/scorers.h"
#include "data/alignment.h"

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
    mkl_set_num_threads(options->get<int>("mkl-threads", 1));
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

      std::cerr << modelOpts->str() << std::flush;

      auto encdec = models::from_options(modelOpts, models::usage::translation);

      if(io::isBin(models[i]) && ptrs_[i] != nullptr) {
        // if file ends in *.bin and has been mapped by QuickSAND
        scorers_.push_back(New<ScorerWrapper>(
          encdec, "F" + std::to_string(scorers_.size()), /*weight=*/1.0f, ptrs[i]));
      } else {
        // it's a *.npz file or has not been mapped by QuickSAND
        scorers_.push_back(New<ScorerWrapper>(
          encdec, "F" + std::to_string(scorers_.size()), /*weight=*/1.0f, models[i]));
      }
    }

    for(auto scorer : scorers_) {
      scorer->init(graph_);
    }
  }

  void setWorkspace(uint8_t* data, size_t size) override { device_->set(data, size); }

  QSNBestBatch decode(const QSBatch& qsBatch,
                      size_t maxLength,
                      const std::unordered_set<Word>& shortlist) override {
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
          subBatch->data()[idx] = (unsigned int)sent[i];
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
    Histories histories = search->search(graph_, batch);

    // convert to QuickSAND format
    QSNBestBatch qsNbestBatch;
    for(const auto& history : histories) { // loop over batch entries
      QSNBest qsNbest;
      NBestList nbestHyps = history->NBest(SIZE_MAX); // request as many N as we have
      for (const Result& result : nbestHyps) { // loop over N-best entries
        // get hypothesis word sequence and normalized sentence score
        auto words = std::get<0>(result);
        auto score = std::get<2>(result);
        // determine alignment if present
        AlignmentSets alignmentSets;
        if (options_->hasAndNotEmpty("alignment"))
        {
          float alignmentThreshold;
          auto alignment = options_->get<std::string>("alignment"); // @TODO: this logic now exists three times in Marian
          if (alignment == "soft")
            alignmentThreshold = 0.0f;
          else if (alignment == "hard")
            alignmentThreshold = 1.0f;
          else
            alignmentThreshold = std::max(std::stof(alignment), 0.f);
          auto hyp = std::get<1>(result);
          data::WordAlignment align = data::ConvertSoftAlignToHardAlign(hyp->TracebackAlignment(), alignmentThreshold);
          // convert to QuickSAND format
          alignmentSets.resize(words.size());
          for (const auto& p : align) // @TODO: Does the feature_model param max_alignment_links apply here?
              alignmentSets[p.tgtPos].insert({p.srcPos, p.prob});
        }
        // form hypothesis to return
        qsNbest.emplace_back(words, std::move(alignmentSets), score);
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
