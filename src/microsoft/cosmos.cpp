#include "cosmos.h"

#include "models/model_base.h"
#include "models/model_factory.h"
#include "data/text_input.h"

#if MKL_FOUND
#include "mkl.h"
#endif

namespace marian {

// Thin wrapper around IModel that makes sure model can be cast to an EncoderPooler
// These poolers know how to collect embeddings from a seq2seq encoder.
class EmbedderModel {
private:
  Ptr<models::IModel> model_;

public:
  EmbedderModel(Ptr<Options> options)
    : model_(createModelFromOptions(options, models::usage::embedding)) {}

  void load(Ptr<ExpressionGraph> graph, const std::string& modelFile) {
    model_->load(graph, modelFile);
  }

  Expr build(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {
    auto embedder = std::dynamic_pointer_cast<EncoderPooler>(model_);
    ABORT_IF(!embedder, "Could not cast to EncoderPooler");
    return embedder->apply(graph, batch, /*clearGraph=*/true)[0];
  }
};

namespace cosmos {

const size_t MAX_BATCH_SIZE =  32;
const size_t MAX_LENGTH     = 256;

/** 
 * Single CPU-core implementation of an Embedder/Similiarity scorer. Turns sets of '\n' strings
 * into parallel batches and either outputs embedding vectors or similarity scores.
 */
class Embedder {
private: 
  Ptr<Options> options_;
  Ptr<ExpressionGraph> graph_;
  Ptr<Vocab> vocab_;

  Ptr<EmbedderModel> model_;
  
public:
  Embedder(const std::string& modelPath, const std::string& vocabPath, bool computeSimilarity = false) {
    options_ = New<Options>("inference", true, 
                            "shuffle", "none",
                            "mini-batch", MAX_BATCH_SIZE,
                            "maxi-batch", 100,
                            "maxi-batch-sort", "src",
                            "max-length", MAX_LENGTH,
                            "max-length-crop", true,
                            "compute-similarity", computeSimilarity,
                            "vocabs", std::vector<std::string>(computeSimilarity ? 2 : 1, vocabPath));
  
    vocab_ = New<Vocab>(options_, 0);
    vocab_->load(vocabPath, 0);

    graph_ = New<ExpressionGraph>(/*inference=*/true);
    graph_->setDevice(CPU0);
    graph_->reserveWorkspaceMB(512);

    YAML::Node config;
    io::getYamlFromModel(config, "special:model.yml", modelPath);
    
    Ptr<Options> modelOpts = New<Options>();
    modelOpts->merge(options_);
    modelOpts->merge(config);

    model_ = New<EmbedderModel>(modelOpts);
    model_->load(graph_, modelPath);
  }

  // Compute embedding vectors for a batch of sentences
  std::vector<std::vector<float>> embed(const std::string& input) {
    auto text = New<data::TextInput>(std::vector<std::string>({input}), 
                                     std::vector<Ptr<Vocab>>({vocab_}),
                                     options_);
    // we set runAsync=false as we are throwing exceptions instead of aborts. Exceptions and threading do not mix well.
    data::BatchGenerator<data::TextInput> batchGenerator(text, options_, /*stats=*/nullptr, /*runAsync=*/false);
    batchGenerator.prepare();

    std::vector<std::vector<float>> output;

    for(auto batch : batchGenerator) {
      auto embeddings = model_->build(graph_, batch);
      graph_->forward();

      std::vector<float> sentVectors;
      embeddings->val()->get(sentVectors);

      // collect embedding vector per sentence.
      // if we compute similarities this is only one similarity per sentence pair.
      for(size_t i = 0; i < batch->size(); ++i) {
        auto batchIdx = batch->getSentenceIds()[i];
        if(output.size() <= batchIdx)
          output.resize(batchIdx + 1);
        
        int embSize = embeddings->shape()[-1];
        size_t beg = i * embSize;
        size_t end = (i + 1) * embSize;
        std::vector<float> sentVector(sentVectors.begin() + beg, sentVectors.begin() + end);
        output[batchIdx] = sentVector;
      }
    }

    return output;
  }

  // Compute cosine similarity scores for a two batches of corresponding sentences
  std::vector<float> similarity(const std::string& input1, const std::string& input2) {
    auto text = New<data::TextInput>(std::vector<std::string>({input1, input2}), 
                                     std::vector<Ptr<Vocab>>({vocab_, vocab_}),
                                     options_);
    // we set runAsync=false as we are throwing exceptions instead of aborts. Exceptions and threading do not mix well.
    data::BatchGenerator<data::TextInput> batchGenerator(text, options_, /*stats=*/nullptr, /*runAsync=*/false);
    batchGenerator.prepare();

    std::vector<float> output;

    for(auto batch : batchGenerator) {
      auto similarities = model_->build(graph_, batch);
      graph_->forward();

      std::vector<float> vSimilarities;
      similarities->val()->get(vSimilarities);

      // collect similarity score per sentence pair.
      for(size_t i = 0; i < batch->size(); ++i) {
        auto batchIdx = batch->getSentenceIds()[i];
        if(output.size() <= batchIdx)
          output.resize(batchIdx + 1);
        output[batchIdx] = vSimilarities[i];
      }
    }

    return output;
  };
};

/* Interface functions ***************************************************************************/

MarianEmbedder::MarianEmbedder() {
#if MKL_FOUND
  mkl_set_num_threads(1);
#endif
  marian::setThrowExceptionOnAbort(true); // globally defined to throw now
}

std::vector<std::vector<float>> MarianEmbedder::embed(const std::string& input) {
  ABORT_IF(!embedder_, "Embedder is not defined??");
  return embedder_->embed(input);
}

bool MarianEmbedder::load(const std::string& modelPath, const std::string& vocabPath) {
  embedder_ = New<Embedder>(modelPath, vocabPath, /*computeSimilarity*/false);
  ABORT_IF(!embedder_, "Embedder is not defined??");
  return true;
}

MarianCosineScorer::MarianCosineScorer() {
#if MKL_FOUND
  mkl_set_num_threads(1);
#endif
  marian::setThrowExceptionOnAbort(true); // globally defined to throw now
}

std::vector<float> MarianCosineScorer::score(const std::string& input1, const std::string& input2) {
  ABORT_IF(!embedder_, "Embedder is not defined??");
  return embedder_->similarity(input1, input2);
};

bool MarianCosineScorer::load(const std::string& modelPath, const std::string& vocabPath) {
  embedder_ = New<Embedder>(modelPath, vocabPath, /*computeSimilarity*/true);
  ABORT_IF(!embedder_, "Embedder is not defined??");
  return true;
}

} // namespace cosmos
} // namespace marian
