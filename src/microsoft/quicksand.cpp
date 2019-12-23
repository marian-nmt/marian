#include "quicksand.h"
#include "marian.h"

#if MKL_FOUND
#include "mkl.h"
#endif

#include "data/shortlist.h"
#include "translator/beam_search.h"
#include "translator/scorers.h"
#include "data/alignment.h"
#include "data/vocab_base.h"
#include "tensors/cpu/fbgemm/expression_graph_packable.h"

#if USE_FBGEMM
#include "fbgemm/Utils.h"
#endif

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

class VocabWrapper : public IVocabWrapper {
  Ptr<Vocab> pImpl_;
public:
  VocabWrapper(Ptr<Vocab> vocab) : pImpl_(vocab) {}
  WordIndex encode(const std::string& word) const override { return (*pImpl_)[word].toWordIndex(); }
  std::string decode(WordIndex id) const override { return (*pImpl_)[Word::fromWordIndex(id)]; }
  size_t size() const override { return pImpl_->size(); }
  void transcodeToShortlistInPlace(WordIndex* ptr, size_t num) const override { pImpl_->transcodeToShortlistInPlace(ptr, num); }
  Ptr<Vocab> getVocab() const { return pImpl_; }
};

class BeamSearchDecoder : public IBeamSearchDecoder {
private:
  Ptr<ExpressionGraph> graph_;
  Ptr<cpu::WrappedDevice> device_;

  std::vector<Ptr<Scorer>> scorers_;

  std::vector<Ptr<Vocab>> vocabs_;

public:
  BeamSearchDecoder(Ptr<Options> options,
                    const std::vector<const void*>& ptrs,
                    const std::vector<Ptr<IVocabWrapper>>& vocabs)
      : IBeamSearchDecoder(options, ptrs) {

    // copy the vocabs
    for (auto vi : vocabs)
      vocabs_.push_back(std::dynamic_pointer_cast<VocabWrapper>(vi)->getVocab());

    // setting 16-bit optimization to false for now. Re-enable with better caching or pre-computation
    graph_ = New<ExpressionGraph>(/*inference=*/true);

    DeviceId deviceId{0, DeviceType::cpu};
    device_ = New<cpu::WrappedDevice>(deviceId);
    graph_->setDevice(deviceId, device_);

#if MKL_FOUND
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

      std::cerr << modelOpts->asYamlString() << std::flush; // @TODO: take a look at why this is even here.

      auto encdec = models::createModelFromOptions(modelOpts, models::usage::translation);

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
                      const std::unordered_set<WordIndex>& shortlist) override {
    if(shortlist.size() > 0) {
      auto shortListGen = New<data::FakeShortlistGenerator>(shortlist);
      for(auto scorer : scorers_)
        scorer->setShortlistGenerator(shortListGen);
    }

    // form source batch, by interleaving the words over sentences in the batch, and setting the mask
    size_t batchSize = qsBatch.size();
    auto subBatch = New<data::SubBatch>(batchSize, maxLength, vocabs_[0]);
    for(size_t i = 0; i < maxLength; ++i) {
      for(size_t j = 0; j < batchSize; ++j) {
        const auto& sent = qsBatch[j];
        if(i < sent.size()) {
          size_t idx = i * batchSize + j;
          subBatch->data()[idx] = marian::Word::fromWordIndex(sent[i]);
          subBatch->mask()[idx] = 1;
        }
      }
    }
    auto tgtSubBatch = New<data::SubBatch>(batchSize, 0, vocabs_[1]); // only holds a vocab, but data is dummy
    std::vector<Ptr<data::SubBatch>> subBatches{ subBatch, tgtSubBatch };
    std::vector<size_t> sentIds(batchSize, 0);

    auto batch = New<data::CorpusBatch>(subBatches);
    batch->setSentenceIds(sentIds);

    // decode
    auto search = New<BeamSearch>(options_, scorers_, vocabs_[1]);
    Histories histories = search->search(graph_, batch);

    // convert to QuickSAND format
    QSNBestBatch qsNbestBatch;
    for(const auto& history : histories) { // loop over batch entries
      QSNBest qsNbest;
      NBestList nbestHyps = history->nBest(SIZE_MAX); // request as many N as we have
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
          data::WordAlignment align = data::ConvertSoftAlignToHardAlign(hyp->tracebackAlignment(), alignmentThreshold);
          // convert to QuickSAND format
          alignmentSets.resize(words.size());
          for (const auto& p : align)
            alignmentSets[p.tgtPos].insert({p.srcPos, p.prob}); // [trgPos] -> {(srcPos, P(srcPos|trgPos))}
        }
        // form hypothesis to return
        qsNbest.emplace_back(toWordIndexVector(words), std::move(alignmentSets), score);
      }
      qsNbestBatch.push_back(qsNbest);
    }

    return qsNbestBatch;
  }
};

Ptr<IBeamSearchDecoder> newDecoder(Ptr<Options> options,
                                   const std::vector<const void*>& ptrs,
                                   const std::vector<Ptr<IVocabWrapper>>& vocabs,
                                   WordIndex eosDummy) { // @TODO: remove this parameter
  marian::setThrowExceptionOnAbort(true); // globally defined to throw now
  ABORT_IF(marian::Word::fromWordIndex(eosDummy) != std::dynamic_pointer_cast<VocabWrapper>(vocabs[1])->getVocab()->getEosId(), "Inconsistent eos vs. vocabs_[1]");

  return New<BeamSearchDecoder>(options, ptrs, vocabs/*, eos*/);
}

std::vector<Ptr<IVocabWrapper>> loadVocabs(const std::vector<std::string>& vocabPaths) {
  std::vector<Ptr<IVocabWrapper>> res(vocabPaths.size());
  for (size_t i = 0; i < vocabPaths.size(); i++) {
    if (i > 0 && vocabPaths[i] == vocabPaths[i-1]) {
      res[i] = res[i-1];
      LOG(info, "[data] Input {} sharing vocabulary with input {}", i, i-1);
    }
    else {
      auto vocab = New<Vocab>(New<Options>(), i); // (empty options, since they are only used for creating vocabs)
      auto size = vocab->load(vocabPaths[i]);
      LOG(info, "[data] Loaded vocabulary size for input {} of size {}", i, size);
      res[i] = New<VocabWrapper>(vocab);
    }
  }
  return res;
}

// query CPU AVX version
DecoderCpuAvxVersion getCpuAvxVersion() {
#if USE_FBGEMM
  // Default value is AVX
  DecoderCpuAvxVersion cpuAvxVer = DecoderCpuAvxVersion::AVX;
  if (fbgemm::fbgemmHasAvx512Support())
    cpuAvxVer = DecoderCpuAvxVersion::AVX512;
  else if (fbgemm::fbgemmHasAvx2Support())
    cpuAvxVer = DecoderCpuAvxVersion::AVX2;

  return cpuAvxVer;
#else
  // Default value is AVX
  return DecoderCpuAvxVersion::AVX;
#endif
}

DecoderCpuAvxVersion parseCpuAvxVersion(std::string name) {
  if (name == "avx") {
    return DecoderCpuAvxVersion::AVX;
  } else if (name == "avx2") {
    return DecoderCpuAvxVersion::AVX2;
  } else if (name == "avx512") {
    return DecoderCpuAvxVersion::AVX512;
  } else {
    ABORT("Unknown CPU Instruction Set: {}", name);
    return DecoderCpuAvxVersion::AVX;
  }
}

// @TODO: clean-up this code and unify with marian-conv. The targetPrec parameter is not clear enought etc. 
bool convertModel(std::string inputFile, std::string outputFile, int32_t targetPrec) {
  std::cout << "Converting from: " << inputFile << ", to: " << outputFile << std::endl;

  YAML::Node config;
  std::stringstream configStr;
  marian::io::getYamlFromModel(config, "special:model.yml", inputFile);
  configStr << config;

  auto graph = New<ExpressionGraphPackable>();
  graph->setDevice(CPU0);
  graph->getBackend()->setOptimized(false);

  graph->load(inputFile);
  graph->forward();
  auto saveGemmType = Type::float32;
  if (targetPrec == 16)
    saveGemmType = Type::packed16;
  else if (targetPrec == 8)
    saveGemmType = Type::packed8avx2; // We currently use avx2 by default.

  // added a flag if the weights needs to be packed or not
  graph->packAndSave(outputFile, configStr.str(), saveGemmType);

  std::cout << "Conversion Finished." << std::endl;

  return true;
}

}  // namespace quicksand
}  // namespace marian
