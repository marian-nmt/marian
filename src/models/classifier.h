#pragma once

#include "marian.h"
#include "models/states.h"
#include "layers/constructors.h"
#include "layers/factory.h"

namespace marian {

class ClassifierBase {
protected:
  Ptr<Options> options_;
  std::string prefix_{"classifier"};
  bool inference_{false};
  size_t batchIndex_{0};

public:
  ClassifierBase(Ptr<Options> options)
      : options_(options),
        prefix_(options->get<std::string>("prefix", "classifier")),
        inference_(options->get<bool>("inference", false)),
        batchIndex_(options->get<size_t>("index", 1)) {} // assume that training input has batch index 0 and labels has 1

  virtual ~ClassifierBase() {}

  virtual Ptr<ClassifierState> apply(Ptr<ExpressionGraph>, Ptr<data::CorpusBatch>, const std::vector<Ptr<EncoderState>>&) = 0;

  template <typename T>
  T opt(const std::string& key) const {
    return options_->get<T>(key);
  }

  virtual void clear() = 0;
};

// This is a model that pretrains BERT for classification
class BertMaskedLM : public ClassifierBase {
public:
  BertMaskedLM(Ptr<Options> options) : ClassifierBase(options) {}

  Ptr<ClassifierState> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> bertBatch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    ABORT_IF(encoderStates.size() != 1, "Currently we only support a single encoder BERT model");

    auto context = encoderStates[0]->getContext();
    
    // Since this is a classifier we are not masking anything on the target. We can (mis)use the mask to hold 
    // indices of words in the encoder that have been masked out for BERT's masked LM training. 
    // Masks are floats, hence the conversion to IndexType.
    const auto& maskedRowsFloats = (*bertBatch)[batchIndex_]->mask(); 
    std::vector<IndexType> maskedRows(maskedRowsFloats.size());
    std::copy(maskedRowsFloats.begin(), maskedRowsFloats.end(), maskedRows.begin());

    auto classEmbeddings = rows(context, graph->indices(maskedRows)); // subselect stuff that has actually been masked out;
    
    int dimModel = classEmbeddings->shape()[-1];

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_ - 1]; // unsafe

    auto layerTanh = mlp::dense(graph)   //
        ("dim", dimModel)                //
        ("activation", mlp::act::tanh); //

    auto layerOut = mlp::output(graph)  //
        ("dim", dimVoc);

    layerOut.tie_transposed("W", "Wemb"); // We are a BERT model, hence tie with input

    // [-4: beam depth=1, -3: max length, -2: batch size, -1: vocab dim]
    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto output = mlp::mlp(graph)                 //
        ("prefix", prefix_ + "_ff_logit_bert_out") //
        .push_back(layerTanh)                      // @TODO: do we actually need this?
        .push_back(layerOut)                       //
        .construct();
    
    auto logits = output->apply(classEmbeddings);
    
    auto state = New<ClassifierState>();
    state->setLogProbs(logits);

    // filled automatically during masking, these are the vocab indices of masked words
    const auto& classLabels = (*bertBatch)[batchIndex_]->data();
    state->setTargetIndices(graph->indices(classLabels));

    return state;
  }

  virtual void clear() override {}
};

// This is a model that uses a pre-trained BERT model to build a classifier on top of the encoder
// Can be used for next sentence prediction task
class BertClassifier : public ClassifierBase {
public:
  BertClassifier(Ptr<Options> options) : ClassifierBase(options) {}

  // The batch has been filled with external classifier labels, @TODO: figure out how to do that
  Ptr<ClassifierState> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> bertBatch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    ABORT_IF(encoderStates.size() != 1, "Currently we only support a single encoder BERT model");

    auto context = encoderStates[0]->getContext();
    auto classEmbeddings = step(context, /*i=*/0, /*axis=*/-2); // [CLS] symbol is first symbol in each sequence
    
    int dimModel = classEmbeddings->shape()[-1];
    int dimTrgCls = opt<int>("bert-classes");

    auto output = mlp::mlp(graph)                                 //
                    ("prefix", prefix_ + "_ff_logit")             //
                    .push_back(mlp::dense(graph)                  //
                                 ("dim", dimModel)                //
                                 ("activation", mlp::act::tanh))  // @TODO: do we actually need this?
                    .push_back(mlp::dense(graph)                  //
                                 ("dim", dimTrgCls))              //
                    .construct();
    
    auto logits = output->apply(classEmbeddings); // class logits for each batch entry
    
    auto state = New<ClassifierState>();
    state->setLogProbs(logits);

    // filled externally, for BERT these are NextSentence prediction labels
    const auto& classLabels = (*bertBatch)[batchIndex_]->data();
    state->setTargetIndices(graph->indices(classLabels));

    return state;
  }

  virtual void clear() override {}
};

}