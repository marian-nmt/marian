#pragma once

#include "data/corpus_base.h"
#include "models/encoder_classifier.h"
#include "models/transformer.h"

namespace marian {
namespace data {

class BertBatch : public CorpusBatch {
private:
  std::vector<IndexType> maskedPositions_;
  std::vector<IndexType> maskedIndices_;
  std::vector<IndexType> sentenceIndices_;

  void init() {
    ABORT("Not implemented");
  }

public:
  BertBatch(Ptr<CorpusBatch> batch) : CorpusBatch(*batch) {
    std::cerr << "Creating BERT batch" << std::endl;
    init();
  }

  const std::vector<IndexType>& bertMaskedPositions() { return maskedPositions_; }
  const std::vector<IndexType>& bertMaskedIndices()   { return maskedIndices_; }
  const std::vector<IndexType>& bertSentenceIndices() { return sentenceIndices_; }
};

}

class BertEncoderClassifier : public EncoderClassifier {
public:
  BertEncoderClassifier(Ptr<Options> options) : EncoderClassifier(options) {}

  std::vector<Ptr<ClassifierState>> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, bool clearGraph) override {
    Ptr<data::BertBatch> bertBatch = New<data::BertBatch>(batch); // intercept batch and anotate with BERT-specific concepts
    return EncoderClassifier::apply(graph, bertBatch, clearGraph);
  }
};

class BertEncoder : public EncoderTransformer {
public:
  BertEncoder(Ptr<Options> options) : EncoderTransformer(options) {}

  Expr addSentenceEmbeddings(Expr embeddings, int start, Ptr<data::CorpusBatch> batch) const {
    Ptr<data::BertBatch> bertBatch = std::dynamic_pointer_cast<data::BertBatch>(batch);

    ABORT_IF(!bertBatch, "Batch could not be converted for BERT training");

    int dimEmb = embeddings->shape()[-1];

    auto sentenceEmbeddings = embedding(graph_)
                                ("prefix", "Wsent")
                                ("dimVocab", 2) // sentence A or sentence B
                                ("dimEmb", dimEmb)
                                .construct();

    // @TODO: note this is going to be really slow due to atomicAdd in backward step
    // with only two classes;
    // instead two masked reduce operations, maybe in parallel streams?
    auto sentenceIndices = graph_->indices(bertBatch->bertSentenceIndices());
    auto signal = rows(sentenceEmbeddings, sentenceIndices); 
    return embeddings + signal;
  }

  virtual Expr addSpecialEmbeddings(Expr input, int start = 0, Ptr<data::CorpusBatch> batch = nullptr) const override {
    input = addPositionalEmbeddings(input, start, true); // true for BERT
    input = addSentenceEmbeddings(input, start, batch);
    return input;
  }
};

// Can be used for next sentence prediction task
class BertClassifier : public ClassifierBase {
public:
  BertClassifier(Ptr<Options> options) : ClassifierBase(options) {}

  // The batch has been filled with external classifier labels, @TODO: figure out how to do that
  Ptr<ClassifierState> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    ABORT_IF(encoderStates.size() != 1, "Currently we only support a single encoder BERT model");

    auto context = encoderStates[0]->getContext();
    auto classEmbeddings = step(context, /*i=*/0, /*axis=*/-2); // [CLS] symbol is first symbol in each sequence
    
    int dimModel = classEmbeddings->shape()[-1];
    int dimTrgCls = opt<std::vector<int>>("dim-vocabs")[batchIndex_]; // Target vocab is used as class labels

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
    const auto& classLabels = (*batch)[batchIndex_]->data();
    state->setTargetIndices(graph->indices(classLabels));

    return state;
  }

  virtual void clear() override {}
};

// This is a model that pretrains BERT for classification
class BertMaskedLM : public ClassifierBase {
public:
  BertMaskedLM(Ptr<Options> options) : ClassifierBase(options) {}

  Ptr<ClassifierState> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    Ptr<data::BertBatch> bertBatch = std::dynamic_pointer_cast<data::BertBatch>(batch);

    ABORT_IF(!bertBatch, "Batch could not be converted to batch for BERT training");
    ABORT_IF(encoderStates.size() != 1, "Currently we only support a single encoder BERT model");

    auto context = encoderStates[0]->getContext();
    
    auto bertMaskedPositions = graph->indices(bertBatch->bertMaskedPositions()); // positions in batch of masked entries
    auto bertMaskedIndices   = graph->indices(bertBatch->bertMaskedIndices());   // vocab ids of entries that have been masked
    
    auto classEmbeddings = rows(context, bertMaskedPositions); // subselect stuff that has actually been masked out;
    
    int dimModel = classEmbeddings->shape()[-1];

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_]; 

    auto layerTanh = mlp::dense(graph)    //
        ("dim", dimModel)                 //
        ("activation", mlp::act::tanh);   //
    auto layerOut = mlp::output(graph)    //
        ("dim", dimVoc);                  //
    layerOut.tie_transposed("W", "Wemb"); // We are a BERT model, hence tie with input

    // [-4: beam depth=1, -3: max length, -2: batch size, -1: vocab dim]
    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto output = mlp::mlp(graph)                      //
        ("prefix", prefix_ + "_ff_logit_maskedlm_out") //
        .push_back(layerTanh)                          // @TODO: do we actually need this?
        .push_back(layerOut)                           //
        .construct();
    
    auto logits = output->apply(classEmbeddings);
    
    auto state = New<ClassifierState>();
    state->setLogProbs(logits);
    state->setTargetIndices(bertMaskedIndices);

    return state;
  }

  virtual void clear() override {}
};

}