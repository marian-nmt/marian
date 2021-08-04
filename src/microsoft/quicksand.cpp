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
#include "tensors/cpu/expression_graph_packable.h"
#include "layers/lsh.h"

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
template void set(Ptr<Options> options, const std::string& key, const std::vector<int>&);
template void set(Ptr<Options> options, const std::string& key, const float&);
template void set(Ptr<Options> options, const std::string& key, const double&);

Ptr<Options> newOptions() {
  return New<Options>();
}

class VocabWrapper : public IVocabWrapper {
  Ptr<Vocab> pImpl_;
public:
  VocabWrapper(Ptr<Vocab> vocab) : pImpl_(vocab) {}
  virtual ~VocabWrapper() {}
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
    mkl_set_num_threads(options_->get<int>("mkl-threads", 1));
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

    // run parameter init once, this is required for graph_->get("parameter name") to work correctly
    graph_->forward();
  }

  void setWorkspace(uint8_t* data, size_t size) override { device_->set(data, size); }

  QSNBestBatch decode(const QSBatch& qsBatch,
                      size_t maxLength,
                      const std::unordered_set<WordIndex>& shortlist) override {
    
    std::vector<int> lshOpts = options_->get<std::vector<int>>("output-approx-knn", {});
    ABORT_IF(lshOpts.size() != 0 && lshOpts.size() != 2, "--output-approx-knn takes 2 parameters");
    ABORT_IF(lshOpts.size() == 2 && shortlist.size() > 0, "LSH and shortlist cannot be used at the same time");

    if(lshOpts.size() == 2 || shortlist.size() > 0) {
      Ptr<data::ShortlistGenerator> shortListGen;
      // both ShortListGenerators are thin wrappers, hence no problem with calling this per query
      if(lshOpts.size() == 2) {
        // Setting abortIfDynamic to true disallows memory allocation for LSH parameters, this is specifically for use in Quicksand.
        // If we want to use the LSH in Quicksand we need to create a binary model that contains the LSH parameters via conversion.
        shortListGen = New<data::LSHShortlistGenerator>(lshOpts[0], lshOpts[1], vocabs_[1]->lemmaSize(), /*abortIfDynamic=*/true);
      } else {
        shortListGen = New<data::FakeShortlistGenerator>(shortlist);
      } 
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

// This function converts an fp32 model into an FBGEMM based packed model.
// marian defined types are used for external project as well.
// The targetPrec is passed as int32_t for the exported function definition.
bool convertModel(std::string inputFile, std::string outputFile, int32_t targetPrec, int32_t lshNBits) {
  std::cerr << "Converting from: " << inputFile << ", to: " << outputFile << ", precision: " << targetPrec << std::endl;

  YAML::Node config;
  std::stringstream configStr;
  marian::io::getYamlFromModel(config, "special:model.yml", inputFile);
  configStr << config;

  auto graph = New<ExpressionGraphPackable>();
  graph->setDevice(CPU0);

  graph->load(inputFile);

  // MJD: Note, this is a default settings which we might want to change or expose. Use this only with Polonium students.
  // The LSH will not be used by default even if it exists in the model. That has to be enabled in the decoder config.
  std::string lshOutputWeights = "Wemb";
  bool addLsh = lshNBits > 0;
  if(addLsh) {
    std::cerr << "Adding LSH to model with hash size " << lshNBits << std::endl;
    // Add dummy parameters for the LSH before the model gets actually initialized.
    // This create the parameters with useless values in the tensors, but it gives us the memory we need.
    graph->setReloaded(false);
    lsh::addDummyParameters(graph, /*weights=*/lshOutputWeights, /*nBits=*/lshNBits);
    graph->setReloaded(true);
  }

  graph->forward();  // run the initializers

  if(addLsh) {
    // After initialization, hijack the paramters for the LSH and force-overwrite with correct values.
    // Once this is done we can just pack and save as normal.
    lsh::overwriteDummyParameters(graph, /*weights=*/lshOutputWeights);
  }

  Type targetPrecType = (Type) targetPrec;
  if (targetPrecType == Type::packed16 
      || targetPrecType == Type::packed8avx2 
      || targetPrecType == Type::packed8avx512
      || (targetPrecType == Type::float32 && addLsh)) { // only allow non-conversion to float32 if we also use the LSH
    graph->packAndSave(outputFile, configStr.str(), targetPrecType);
    std::cerr << "Conversion Finished." << std::endl;
  } else {
    ABORT("Target type is not supported in this funcion: {}", targetPrec);
    return false;
  }

  return true;
}

}  // namespace quicksand
}  // namespace marian
