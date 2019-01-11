#pragma once

#include "data/corpus_base.h"
#include "models/encoder_classifier.h"
#include "models/transformer.h"
#include "data/rng_engine.h"

namespace marian {
namespace data {

class BertBatch : public CorpusBatch {
private:
  std::mt19937& eng_;

  std::vector<IndexType> maskedPositions_;
  std::vector<IndexType> maskedWords_;
  std::vector<IndexType> sentenceIndices_;

  std::string maskSymbol_;
  std::string sepSymbol_;
  std::string clsSymbol_;

  std::unique_ptr<std::uniform_int_distribution<Word>> randomWord_;
  std::unique_ptr<std::uniform_int_distribution<int>> randomPercent_;

  std::unordered_set<Word> dontMask_;

  Word maskOut(Word word, Word mask) {
    auto subBatch = subBatches_.front();

    int r = (*randomPercent_)(eng_);
    if (r < 10) { // for 10% of cases return same word
      return word;
    } else if (r < 20) { // for 10% return random word
      Word randWord = (*randomWord_)(eng_);
      if(dontMask_.count(randWord) > 0) // the random word is a forbidden word
        return mask;                    // hence return mask symbol
      else
        return randWord;                // else return the random word
    } else { // for 80% of words apply mask symbol
      return mask;
    }
  }

public:
  BertBatch(Ptr<CorpusBatch> batch,
            std::mt19937& engine,
            float maskFraction,
            const std::string& maskSymbol,
            const std::string& sepSymbol,
            const std::string& clsSymbol)
    : CorpusBatch(*batch), eng_(engine),
      maskSymbol_(maskSymbol), sepSymbol_(sepSymbol), clsSymbol_(clsSymbol) {

    auto subBatch = subBatches_.front();

    randomWord_.reset(new std::uniform_int_distribution<Word>(0, subBatch->vocab()->size()));
    randomPercent_.reset(new std::uniform_int_distribution<int>(0, 100));

    auto& words = subBatch->data();

    Word maskId  = (*subBatch->vocab())[maskSymbol_];
    Word clsId   = (*subBatch->vocab())[clsSymbol_];
    Word sepId   = (*subBatch->vocab())[sepSymbol_];

    ABORT_IF(maskId == subBatch->vocab()->getUnkId(),
             "BERT masking symbol {} not found in vocabulary", maskSymbol_);

    ABORT_IF(sepId == subBatch->vocab()->getUnkId(),
             "BERT separator symbol {} not found in vocabulary", sepSymbol_);

    ABORT_IF(clsId == subBatch->vocab()->getUnkId(),
             "BERT class symbol {} not found in vocabulary", clsSymbol_);

    dontMask_.insert(clsId); // don't mask class token
    dontMask_.insert(sepId); // don't mask separator token
    dontMask_.insert(subBatch->vocab()->getEosId()); // don't mask </s>
    // it's ok to mask <unk>

    std::vector<int> selected;
    selected.reserve(words.size());
    for(int i = 0; i < words.size(); ++i) // collect words among which we will mask
      if(dontMask_.count(words[i]) == 0)  // do not add indices of special words
        selected.push_back(i);
    std::shuffle(selected.begin(), selected.end(), eng_);
    selected.resize(std::ceil(selected.size() * maskFraction)); // select first x percent from shuffled indices

    for(int i : selected) {
      maskedPositions_.push_back(i);        // where is the original word?
      maskedWords_.push_back(words[i]);     // what is the original word?
      words[i] = maskOut(words[i], maskId); // mask that position
    }

    int dimBatch = subBatch->batchSize();
    int dimWords = subBatch->batchWidth();

    sentenceIndices_.resize(words.size());
    std::vector<IndexType> sentPos(dimBatch, 0);
    for(int i = 0; i < dimWords; ++i) {
      for(int j = 0; j < dimBatch; ++j) {
        int k = i * dimBatch + j;
        sentenceIndices_[k] = sentPos[j];
        if(words[k] == sepId)
          sentPos[j]++;
      }
    }
  }

  const std::vector<IndexType>& bertMaskedPositions() { return maskedPositions_; }
  const std::vector<IndexType>& bertMaskedWords()     { return maskedWords_; }
  const std::vector<IndexType>& bertSentenceIndices() { return sentenceIndices_; }
};

}

class BertEncoderClassifier : public EncoderClassifier, public data::RNGEngine {
public:
  BertEncoderClassifier(Ptr<Options> options)
  : EncoderClassifier(options) {}

  std::vector<Ptr<ClassifierState>> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, bool clearGraph) override {
    // intercept batch and anotate with BERT-specific concepts
    auto bertBatch = New<data::BertBatch>(batch,
                                          eng_,
                                          opt<float>("bert-masking-fraction"),
                                          opt<std::string>("bert-mask-symbol"),
                                          opt<std::string>("bert-sep-symbol"),
                                          opt<std::string>("bert-class-symbol"));
    return EncoderClassifier::apply(graph, bertBatch, clearGraph);
  }

  // for externally created BertBatch for instance in BertValidator
  std::vector<Ptr<ClassifierState>> apply(Ptr<ExpressionGraph> graph, Ptr<data::BertBatch> bertBatch, bool clearGraph) {
    return EncoderClassifier::apply(graph, bertBatch, clearGraph);
  }
};

// @TODO: this should be in transformer.h
class BertEncoder : public EncoderTransformer {
public:
  BertEncoder(Ptr<Options> options) : EncoderTransformer(options) {}

  Expr addSentenceEmbeddings(Expr embeddings,
                             Ptr<data::CorpusBatch> batch,
                             bool learnedPosEmbeddings) const {
    Ptr<data::BertBatch> bertBatch = std::dynamic_pointer_cast<data::BertBatch>(batch);

    ABORT_IF(!bertBatch, "Batch could not be converted for BERT training");

    int dimEmb = embeddings->shape()[-1];
    int dimBatch = embeddings->shape()[-2];
    int dimWords = embeddings->shape()[-3];

    Expr sentenceEmbeddings;
    if(learnedPosEmbeddings) {
      sentenceEmbeddings = embedding(graph_)
                               ("prefix", "Wsent")
                               ("dimVocab", 2) // sentence A or sentence B
                               ("dimEmb", dimEmb)
                               .construct();
    } else {
      // trigonometric positions, no backprob
      sentenceEmbeddings = graph_->constant({2, dimEmb}, inits::positions(0));
    }

    auto sentenceIndices = graph_->indices(bertBatch->bertSentenceIndices());

    auto signal = rows(sentenceEmbeddings, sentenceIndices);
    signal = reshape(signal, {dimWords, dimBatch, dimEmb});
    return embeddings + signal;
  }

  virtual Expr addSpecialEmbeddings(Expr input, int start = 0, Ptr<data::CorpusBatch> batch = nullptr) const override {
    bool learnedPosEmbeddings = opt<bool>("transformer-learned-positions", true);
    input = addPositionalEmbeddings(input, start, learnedPosEmbeddings);
    input = addSentenceEmbeddings(input, batch, learnedPosEmbeddings);
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
    auto classEmbeddings = step(context, /*i=*/0, /*axis=*/-3); // [CLS] symbol is first symbol in each sequence

    int dimModel = classEmbeddings->shape()[-1];
    int dimTrgCls = opt<std::vector<int>>("dim-vocabs")[batchIndex_]; // Target vocab is used as class labels

    auto output = mlp::mlp(graph)                                     //
                    .push_back(mlp::dense(graph)                      //
                                 ("prefix", prefix_ + "_ff_logit_l1") //
                                 ("dim", dimModel)                    //
                                 ("activation", mlp::act::tanh))      // @TODO: do we actually need this?
                    .push_back(mlp::output(graph)                     //
                                 ("dim", dimTrgCls))                  //
                                 ("prefix", prefix_ + "_ff_logit_l2") //
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
    auto bertMaskedWords     = graph->indices(bertBatch->bertMaskedWords());   // vocab ids of entries that have been masked

    int dimModel = context->shape()[-1];
    int dimBatch = context->shape()[-2];
    int dimTime  = context->shape()[-3];

    auto maskedEmbeddings = rows(reshape(context, {dimBatch * dimTime, dimModel}), bertMaskedPositions); // subselect stuff that has actually been masked out;

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    auto layerTanh = mlp::dense(graph)    //
        ("prefix", prefix_ + "_ff_logit_maskedlm_out_l1") //
        ("dim", dimModel)                 //
        ("activation", mlp::act::tanh);   //
    auto layerOut = mlp::output(graph)    //
        ("prefix", prefix_ + "_ff_logit_maskedlm_out_l2") //
        ("dim", dimVoc);                  //
    layerOut.tie_transposed("W", "Wemb"); // We are a BERT model, hence tie with input

    // [-4: beam depth=1, -3: max length, -2: batch size, -1: vocab dim]
    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto output = mlp::mlp(graph)                      //
        .push_back(layerTanh)                          // @TODO: do we actually need this?
        .push_back(layerOut)                           //
        .construct();

    auto logits = output->apply(maskedEmbeddings);

    auto state = New<ClassifierState>();
    state->setLogProbs(logits);
    state->setTargetIndices(bertMaskedWords);

    return state;
  }

  virtual void clear() override {}
};

}