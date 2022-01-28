#pragma once

#include "data/corpus_base.h"
#include "models/encoder_classifier.h"
#include "models/transformer.h"   // @BUGBUG: transformer.h is large and was meant to be compiled separately
#include "data/rng_engine.h"

namespace marian {

/**
 * This file contains nearly all BERT-related code and adds BERT-functionality
 * on top of existing classes like TansformerEncoder and Classifier.
 */

namespace data {

/**
 * BERT-specific mini-batch that computes masking for Masked LM training.
 * Expects symbols [MASK], [SEP], [CLS] to be present in vocabularies unless
 * other symbols are specified in the config.
 *
 * This takes a normal CorpusBatch and extends it with additional data. Luckily
 * all the BERT-functionality can be inferred from a CorpusBatch alone.
 */
class BertBatch : public CorpusBatch {
private:
  std::vector<IndexType> maskedPositions_;
  Words maskedWords_;
  std::vector<IndexType> sentenceIndices_;

  std::string maskSymbol_;
  std::string sepSymbol_;
  std::string clsSymbol_;

  // Selects a random word from the vocabulary
  std::unique_ptr<std::uniform_int_distribution<WordIndex>> randomWord_;

  // Selects a random integer between 0 and 99
  std::unique_ptr<std::uniform_real_distribution<float>> randomPercent_;

  // Word ids of words that should not be masked, e.g. separators, padding
  std::unordered_set<Word> dontMask_;

  // Masking function, i.e. replaces a chosen word with either
  // a [MASK] symbol, itself or a random word
  Word maskOut(Word word, Word mask, std::mt19937& engine) {
    auto subBatch = subBatches_.front();

    // @TODO: turn those threshold into parameters, adjustable from command line
    float r = (*randomPercent_)(engine);
    if (r < 0.1f) { // for 10% of cases return same word
      return word;
    } else if (r < 0.2f) { // for 10% return random word
      Word randWord = Word::fromWordIndex((*randomWord_)(engine));
      if(dontMask_.count(randWord) > 0) // some words, e.g. [CLS] or </s>, may not be used as random words
        return mask;                    // for those, return the mask symbol instead
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
            const std::string& clsSymbol,
            int dimTypeVocab)
    : CorpusBatch(*batch),
      maskSymbol_(maskSymbol), sepSymbol_(sepSymbol), clsSymbol_(clsSymbol) {

    // BERT expects a textual first stream and a second stream with class labels
    auto subBatch = subBatches_.front();
    const auto& vocab = *subBatch->vocab();

    // Initialize to sample random vocab id
    randomWord_.reset(new std::uniform_int_distribution<WordIndex>(0, (WordIndex)vocab.size()));

    // Initialize to sample random percentage
    randomPercent_.reset(new std::uniform_real_distribution<float>(0.f, 1.f));

    auto& words = subBatch->data();

    // Get word id of special symbols
    Word maskId  = vocab[maskSymbol_];
    Word clsId   = vocab[clsSymbol_];
    Word sepId   = vocab[sepSymbol_];

    ABORT_IF(maskId == vocab.getUnkId(),
             "BERT masking symbol {} not found in vocabulary", maskSymbol_);

    ABORT_IF(sepId == vocab.getUnkId(),
             "BERT separator symbol {} not found in vocabulary", sepSymbol_);

    ABORT_IF(clsId == vocab.getUnkId(),
             "BERT class symbol {} not found in vocabulary", clsSymbol_);

    dontMask_.insert(clsId); // don't mask class token
    dontMask_.insert(sepId); // don't mask separator token
    dontMask_.insert(vocab.getEosId()); // don't mask </s>
    // it's ok to mask <unk>

    std::vector<int> selected;
    selected.reserve(words.size());
    for(int i = 0; i < words.size(); ++i) // collect words among which we will mask
      if(dontMask_.count(words[i]) == 0)  // do not add indices of special words
        selected.push_back(i);
    std::shuffle(selected.begin(), selected.end(), engine); // randomize positions
    selected.resize((size_t)std::ceil(selected.size() * maskFraction)); // select first x percent from shuffled indices

    for(int i : selected) {
      maskedPositions_.push_back(i);                // where is the original word?
      maskedWords_.push_back(words[i]);             // what is the original word?
      words[i] = maskOut(words[i], maskId, engine); // mask that position
    }

    annotateSentenceIndices(dimTypeVocab);
  }

  BertBatch(Ptr<CorpusBatch> batch,
            const std::string& sepSymbol,
            const std::string& clsSymbol,
            int dimTypeVocab)
    : CorpusBatch(*batch),
      maskSymbol_("dummy"), sepSymbol_(sepSymbol), clsSymbol_(clsSymbol) {
    annotateSentenceIndices(dimTypeVocab);
  }

  void annotateSentenceIndices(int dimTypeVocab) {
    // BERT expects a textual first stream and a second stream with class labels
    auto subBatch = subBatches_.front();
    const auto& vocab = *subBatch->vocab();
    auto& words = subBatch->data();

    // Get word id of special symbols
    Word sepId   = vocab[sepSymbol_];
    ABORT_IF(sepId == vocab.getUnkId(),
             "BERT separator symbol {} not found in vocabulary", sepSymbol_);

    int dimBatch = (int)subBatch->batchSize();
    int dimWords = (int)subBatch->batchWidth();

    const size_t maxSentPos = dimTypeVocab;

    // create indices for BERT sentence embeddings A and B
    sentenceIndices_.resize(words.size()); // each word is either in sentence A or B
    std::vector<IndexType> sentPos(dimBatch, 0); // initialize each batch entry with being A [0]
    for(int i = 0; i < dimWords; ++i) {   // advance word-wise
      for(int j = 0; j < dimBatch; ++j) { // scan batch-wise
        int k = i * dimBatch + j;
        sentenceIndices_[k] = sentPos[j]; // set to current sentence position for batch entry, max position 1.
        if(words[k] == sepId && sentPos[j] < maxSentPos) { // if current word is a separator and not beyond range
          sentPos[j]++;                   // then increase sentence position for batch entry (to B [1])
        }
      }
    }
  }

  const std::vector<IndexType>& bertMaskedPositions() { return maskedPositions_; }
  const Words& bertMaskedWords() { return maskedWords_; }
  const std::vector<IndexType>& bertSentenceIndices() { return sentenceIndices_; }
};

}

/**
 * BERT-specific version of EncoderClassifier, mostly here to automatically convert a
 * CorpusBatch to BertBatch.
 */
class BertEncoderClassifier : public EncoderClassifier, public data::RNGEngine { // @TODO: this random engine is not being serialized right now
public:
  BertEncoderClassifier(Ptr<Options> options)
  : EncoderClassifier(options) {}

  std::vector<Ptr<ClassifierState>> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, bool clearGraph) override {
    std::string modelType = opt<std::string>("type");
    int dimTypeVocab = opt<int>("bert-type-vocab-size");

    // intercept batch and annotate with BERT-specific concepts
    Ptr<data::BertBatch> bertBatch;
    if(modelType == "bert") { // full BERT pre-training
      bertBatch = New<data::BertBatch>(batch,
                                       eng_,
                                       opt<float>("bert-masking-fraction", 0.15f), // 15% by default according to paper
                                       opt<std::string>("bert-mask-symbol"),
                                       opt<std::string>("bert-sep-symbol"),
                                       opt<std::string>("bert-class-symbol"),
                                       dimTypeVocab);
    } else if(modelType == "bert-classifier") { // we are probably fine-tuning a BERT model for a classification task
      bertBatch = New<data::BertBatch>(batch,
                                       opt<std::string>("bert-sep-symbol"),
                                       opt<std::string>("bert-class-symbol"),
                                       dimTypeVocab); // only annotate sentence separators
    } else {
      ABORT("Unknown BERT-style model: {}", modelType);
    }

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
 * @BUGBUG: transformer.h was meant to be compiled separately. I.e., one cannot derive from it.
 *          Is there a way to maybe instead include a reference in here, instead of deriving from it?
 */
class BertEncoder : public EncoderTransformer {
  using EncoderTransformer::EncoderTransformer;
public:
  Expr addSentenceEmbeddings(Expr embeddings,
                             Ptr<data::CorpusBatch> batch,
                             bool learnedPosEmbeddings) const {
    Ptr<data::BertBatch> bertBatch = std::dynamic_pointer_cast<data::BertBatch>(batch);
    ABORT_IF(!bertBatch, "Batch must be BertBatch for BERT training or fine-tuning");

    int dimEmb = embeddings->shape()[-1];
    int dimBatch = embeddings->shape()[-2];
    int dimWords = embeddings->shape()[-3];

    int dimTypeVocab = opt<int>("bert-type-vocab-size", 2);

    Expr signal;
    if(learnedPosEmbeddings) {
      auto sentenceEmbeddings = embedding()
                               ("prefix", "Wtype")
                               ("dimVocab", dimTypeVocab) // sentence A or sentence B
                               ("dimEmb", dimEmb)
                               .construct(graph_);
      signal = sentenceEmbeddings->applyIndices(bertBatch->bertSentenceIndices(), {dimWords, dimBatch, dimEmb});
    } else {
      // @TODO: factory for positional embeddings?
      // constant sinusoidal position embeddings, no backprob
      auto sentenceEmbeddingsExpr = graph_->constant({2, dimEmb}, inits::sinusoidalPositionEmbeddings(0));
      signal = rows(sentenceEmbeddingsExpr, bertBatch->bertSentenceIndices());
      signal = reshape(signal, {dimWords, dimBatch, dimEmb});
    }

    return embeddings + signal;
  }

  virtual Expr addSpecialEmbeddings(Expr input, int start = 0, Ptr<data::CorpusBatch> batch = nullptr) const override {
    bool trainPosEmbeddings = opt<bool>("transformer-train-position-embeddings", true);
    bool trainTypeEmbeddings = opt<bool>("bert-train-type-embeddings", true);
    input = addPositionalEmbeddings(input, start, trainPosEmbeddings);
    input = addSentenceEmbeddings(input, batch, trainTypeEmbeddings);
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
  using ClassifierBase::ClassifierBase;
public:
  Ptr<ClassifierState> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    ABORT_IF(encoderStates.size() != 1, "Currently we only support a single encoder BERT model");

    auto context = encoderStates[0]->getContext();
    auto classEmbeddings = slice(context, /*axis=*/-3, /*i=*/0); // [CLS] symbol is first symbol in each sequence

    int dimModel = classEmbeddings->shape()[-1];
    int dimTrgCls = opt<std::vector<int>>("dim-vocabs")[batchIndex_]; // Target vocab is used as class labels

    auto output = mlp::mlp()                                          //
                    .push_back(mlp::dense()                           //
                                 ("prefix", prefix_ + "_ff_logit_l1") //
                                 ("dim", dimModel)                    //
                                 ("activation", (int)mlp::act::tanh))      // @TODO: do we actually need this?
                    .push_back(mlp::output()                          //
                                 ("dim", dimTrgCls))                  //
                                 ("prefix", prefix_ + "_ff_logit_l2") //
                    .construct(graph);

    auto logits = output->apply(classEmbeddings); // class logits for each batch entry

    auto state = New<ClassifierState>();
    state->setLogProbs(logits);

    // Filled externally, for BERT these are NextSentence prediction labels
    const auto& classLabels = (*batch)[batchIndex_]->data();
    state->setTargetWords(classLabels);

    return state;
  }

  virtual void clear() override {}
};

/**
 * This is a model that pretrains BERT for classification.
 * This is also a Classifier, but compared to the BertClassifier above needs the BERT-specific information from BertBatch
 * as this is self-generating its labels from the source. Labels are dynamically created as complements of the masking process.
 */
class BertMaskedLM : public ClassifierBase {
  using ClassifierBase::ClassifierBase;
public:
  Ptr<ClassifierState> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    Ptr<data::BertBatch> bertBatch = std::dynamic_pointer_cast<data::BertBatch>(batch);

    ABORT_IF(!bertBatch, "Batch must be BertBatch for BERT training");
    ABORT_IF(encoderStates.size() != 1, "Currently we only support a single encoder BERT model");

    auto context = encoderStates[0]->getContext();

    auto bertMaskedPositions    = graph->indices(bertBatch->bertMaskedPositions()); // positions in batch of masked entries
    const auto& bertMaskedWords = bertBatch->bertMaskedWords();   // vocab ids of entries that have been masked

    int dimModel = context->shape()[-1];
    int dimBatch = context->shape()[-2];
    int dimTime  = context->shape()[-3];

    auto maskedContext = rows(reshape(context, {dimBatch * dimTime, dimModel}), bertMaskedPositions); // subselect stuff that has actually been masked out

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    auto layer1 = mlp::mlp()
      .push_back(mlp::dense()
                 ("prefix", prefix_ + "_ff_logit_l1")
                 ("dim", dimModel))
                 .construct(graph);

    auto intermediate = layer1->apply(maskedContext);

    std::string activationType = opt<std::string>("transformer-ffn-activation");
    if(activationType == "relu")
      intermediate = relu(intermediate);
    else if(activationType == "swish")
      intermediate = swish(intermediate);
    else if(activationType == "gelu")
      intermediate = gelu(intermediate);
    else
      ABORT("Activation function {} not supported in BERT masked LM", activationType);

    auto gamma = graph->param(prefix_ + "_ff_ln_scale", {1, dimModel}, inits::ones());
    auto beta  = graph->param(prefix_ + "_ff_ln_bias",  {1, dimModel}, inits::zeros());
    intermediate = layerNorm(intermediate, gamma, beta);

    auto layer2 = mlp::mlp()
      .push_back(mlp::output(
                  "prefix", prefix_ + "_ff_logit_l2",
                  "dim", dimVoc)
                 .tieTransposed("Wemb"))
      .construct(graph);

    auto logits = layer2->apply(intermediate); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vocab dim]

    auto state = New<ClassifierState>();
    state->setLogProbs(logits);
    state->setTargetWords(bertMaskedWords);

    return state;
  }

  virtual void clear() override {}
};

}
