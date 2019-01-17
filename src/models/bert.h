#pragma once

#include "data/corpus_base.h"
#include "models/encoder_classifier.h"
#include "models/transformer.h"
#include "data/rng_engine.h"

namespace marian {

/** 
 * This file contains nearly all BERT-related code and adds BERT-funtionality
 * on top of existing classes like TansformerEncoder and Classifier. 
 */

namespace data {

/**
 * BERT-specific mini-batch that computes masking for Masked LM training.
 * Expects symbols [MASK], [SEP], [CLS] to be present in vocabularies unless
 * other symbols are specified on the command line.
 * 
 * This takes a normal CorpusBatch and extends it with additional data. Luckily
 * all the BERT-functionality can be inferred from a CorpusBatch alone.
 */
class BertBatch : public CorpusBatch {
private:
  std::mt19937& eng_;

  std::vector<IndexType> maskedPositions_;
  std::vector<IndexType> maskedWords_;
  std::vector<IndexType> sentenceIndices_;

  std::string maskSymbol_;
  std::string sepSymbol_;
  std::string clsSymbol_;

  // Selects a random word from the vocabulary
  std::unique_ptr<std::uniform_int_distribution<Word>> randomWord_;

  // Selects a random integer between 0 and 99
  std::unique_ptr<std::uniform_int_distribution<int>> randomPercent_;

  // Word ids of words that should not be masked, e.g. separators, padding
  std::unordered_set<Word> dontMask_;

  // Masking function
  Word maskOut(Word word, Word mask) {
    auto subBatch = subBatches_.front();

    // @TODO: turn those threshold into parameters, adjustable from command line
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

  // Takes a corpus batch, random engine (for deterministic behavior) and the masking percentage.
  // Also sets special vocabulary items given on command line.
  BertBatch(Ptr<CorpusBatch> batch,
            std::mt19937& engine,
            float maskFraction,
            const std::string& maskSymbol,
            const std::string& sepSymbol,
            const std::string& clsSymbol)
    : CorpusBatch(*batch), eng_(engine),
      maskSymbol_(maskSymbol), sepSymbol_(sepSymbol), clsSymbol_(clsSymbol) {

    auto subBatch = subBatches_.front();

    // Initialize to sample random vocab id
    randomWord_.reset(new std::uniform_int_distribution<Word>(0, subBatch->vocab()->size()));

    // Intialize to sample random percentage
    randomPercent_.reset(new std::uniform_int_distribution<int>(0, 100));

    auto& words = subBatch->data();

    // Get word id of special symbols
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
    std::shuffle(selected.begin(), selected.end(), eng_); // randomize positions
    selected.resize(std::ceil(selected.size() * maskFraction)); // select first x percent from shuffled indices

    for(int i : selected) {
      maskedPositions_.push_back(i);        // where is the original word?
      maskedWords_.push_back(words[i]);     // what is the original word?
      words[i] = maskOut(words[i], maskId); // mask that position
    }

    int dimBatch = subBatch->batchSize();
    int dimWords = subBatch->batchWidth();

    // create indices for BERT sentence embeddings A and B
    sentenceIndices_.resize(words.size()); // each word is either in sentence A or B
    std::vector<IndexType> sentPos(dimBatch, 0); // initialize each batch entry with being A [0]
    for(int i = 0; i < dimWords; ++i) {   // advance word-wise
      for(int j = 0; j < dimBatch; ++j) { // scan batch-wise
        int k = i * dimBatch + j;
        sentenceIndices_[k] = sentPos[j]; // set to current sentence position for batch entry
        if(words[k] == sepId) {           // if current word is a separator 
          sentPos[j]++;                   // then increase sentence position for batch entry (probably to B [1])
          ABORT_IF(sentPos[i] > 1, "Currently we only support sequences of up to two sentences in BERT, not {}", sentPos[i]);
        }
      }
    }
  }

  const std::vector<IndexType>& bertMaskedPositions() { return maskedPositions_; }
  const std::vector<IndexType>& bertMaskedWords()     { return maskedWords_; }
  const std::vector<IndexType>& bertSentenceIndices() { return sentenceIndices_; }
};

}

/**
 * BERT-specific version of EncoderClassifier, mostly here to automatically convert a
 * CorpusBatch to BertBatch. 
 */
class BertEncoderClassifier : public EncoderClassifier, public data::RNGEngine {
public:
  BertEncoderClassifier(Ptr<Options> options)
  : EncoderClassifier(options) {}

  std::vector<Ptr<ClassifierState>> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, bool clearGraph) override {
    // intercept batch and anotate with BERT-specific concepts
    auto bertBatch = New<data::BertBatch>(batch,
                                          eng_,
                                          opt<float>("bert-masking-fraction", 0.15f), // 15% by default according to paper
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

/**
 * BERT-specific modifications to EncoderTransformer
 * Actually all that is needed is to intercept the creation of special embeddings,
 * here sentence embeddings for sentence A and B. 
 */
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

    Expr signal;
    if(learnedPosEmbeddings) {
      auto sentenceEmbeddings = embedding()
                               ("prefix", "Wsent")
                               ("dimVocab", 2) // sentence A or sentence B, @TODO: should rather be a parameter
                               ("dimEmb", dimEmb)
                               .construct(graph_);
      signal = sentenceEmbeddings->apply(bertBatch->bertSentenceIndices(), {dimWords, dimBatch, dimEmb});
    } else {
      // @TODO: factory for postional embeddings?
      // trigonometric positions, no backprob
      auto sentenceEmbeddingsExpr = graph_->constant({2, dimEmb}, inits::positions(0));
      signal = rows(sentenceEmbeddingsExpr, bertBatch->bertSentenceIndices());
      signal = reshape(signal, {dimWords, dimBatch, dimEmb});
    }

    return embeddings + signal;
  }

  virtual Expr addSpecialEmbeddings(Expr input, int start = 0, Ptr<data::CorpusBatch> batch = nullptr) const override {
    bool learnedPosEmbeddings = opt<bool>("transformer-learned-positions", true);
    input = addPositionalEmbeddings(input, start, learnedPosEmbeddings);
    input = addSentenceEmbeddings(input, batch, learnedPosEmbeddings); // @TODO: separately set learnable pos and sent embeddings
    return input;
  }
};

/**
 * BERT-specific classifier 
 * Can be used for next sentence prediction task or other fine-tuned down-stream tasks
 * Does not actually need a BertBatch, works with CorpusBatch.
 * 
 * @TODO: This is in fact so generic that we might move it out of here as the typical classifier implementation
 */
class BertClassifier : public ClassifierBase {
public:
  BertClassifier(Ptr<Options> options) : ClassifierBase(options) {}

  Ptr<ClassifierState> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    ABORT_IF(encoderStates.size() != 1, "Currently we only support a single encoder BERT model");

    auto context = encoderStates[0]->getContext();
    auto classEmbeddings = step(context, /*i=*/0, /*axis=*/-3); // [CLS] symbol is first symbol in each sequence

    int dimModel = classEmbeddings->shape()[-1];
    int dimTrgCls = opt<std::vector<int>>("dim-vocabs")[batchIndex_]; // Target vocab is used as class labels

    auto output = mlp::mlp()                                          //
                    .push_back(mlp::dense()                           //
                                 ("prefix", prefix_ + "_ff_logit_l1") //
                                 ("dim", dimModel)                    //
                                 ("activation", mlp::act::tanh))      // @TODO: do we actually need this?
                    .push_back(mlp::output()                          //
                                 ("dim", dimTrgCls))                  //
                                 ("prefix", prefix_ + "_ff_logit_l2") //
                    .construct(graph);

    auto logits = output->apply(classEmbeddings); // class logits for each batch entry

    auto state = New<ClassifierState>();
    state->setLogProbs(logits);

    // Filled externally, for BERT these are NextSentence prediction labels
    const auto& classLabels = (*batch)[batchIndex_]->data();
    state->setTargetIndices(graph->indices(classLabels));

    return state;
  }

  virtual void clear() override {}
};

/**
 * This is a model that pretrains BERT for classification.
 * This is also a Classifier, but compared to the one above needs the BERT-specific information from BertBatch
 * as this is self-generating its labels from the source. Labels are dynamically created as complements of the
 * masking process. 
 */
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

    auto layerTanh = mlp::dense()                         //
        ("prefix", prefix_ + "_ff_logit_maskedlm_out_l1") //
        ("dim", dimModel)                                 //
        ("activation", mlp::act::tanh);                   // @TODO: again, check if this layer is present in original code
    auto layerOut = mlp::output()                         //
        ("prefix", prefix_ + "_ff_logit_maskedlm_out_l2") //
        ("dim", dimVoc);                  //
    layerOut.tieTransposed("Wemb"); // We are a BERT model, hence tie with input, @TODO: check if this is actually what Google does

    // [-4: beam depth=1, -3: max length, -2: batch size, -1: vocab dim]
    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto output = mlp::mlp()  //
        .push_back(layerTanh) // 
        .push_back(layerOut)  //
        .construct(graph);

    auto logits = output->apply(maskedEmbeddings);

    auto state = New<ClassifierState>();
    state->setLogProbs(logits);
    state->setTargetIndices(bertMaskedWords);

    return state;
  }

  virtual void clear() override {}
};

}