#pragma once

#include "marian.h"

namespace marian {

class EncoderS2S : public EncoderBase {
public:

  Expr applyEncoderRNN(Ptr<ExpressionGraph> graph,
                       Expr embeddings, Expr mask,
                       std::string type) {

    int first, second;
    if(type == "bidirectional" || type == "alternating") {
      // build two separate stacks, concatenate top output
      first = opt<int>("enc-depth");
      second = 0;
    }
    else {
      // build 1-layer bidirectional stack, concatenate,
      // build n-1 layer unidirectional stack
      first = 1;
      second = opt<int>("enc-depth") - first;
    }

    auto forward = type == "alternating" ?
      rnn::dir::alternating_forward : rnn::dir::forward;

    auto backward = type == "alternating" ?
      rnn::dir::alternating_backward : rnn::dir::backward;

    using namespace keywords;
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");

    auto rnnFw = rnn::rnn(graph)
                 ("type", opt<std::string>("enc-cell"))
                 ("direction", forward)
                 ("dimInput", opt<int>("dim-emb"))
                 ("dimState", opt<int>("dim-rnn"))
                 ("dropout", dropoutRnn)
                 ("layer-normalization", opt<bool>("layer-normalization"))
                 ("skip", opt<bool>("skip"));

    for(int i = 1; i <= first; ++i) {
      auto stacked = rnn::stacked_cell(graph);
      for(int j = 1; j <= opt<int>("enc-cell-depth"); ++j) {
        std::string paramPrefix = prefix_ + "_bi";
        if(i > 1)
          paramPrefix += "_l" + std::to_string(i);
        if(i > 1 || j > 1)
          paramPrefix += "_cell" + std::to_string(j);
        stacked.push_back(rnn::cell(graph)
                          ("prefix", paramPrefix));
      }
      rnnFw.push_back(stacked);
    }

    auto rnnBw = rnn::rnn(graph)
                 ("type", opt<std::string>("enc-cell"))
                 ("direction", backward)
                 ("dimInput", opt<int>("dim-emb"))
                 ("dimState", opt<int>("dim-rnn"))
                 ("dropout", dropoutRnn)
                 ("layer-normalization", opt<bool>("layer-normalization"))
                 ("skip", opt<bool>("skip"));

    for(int i = 1; i <= first; ++i) {
      auto stacked = rnn::stacked_cell(graph);
      for(int j = 1; j <= opt<int>("enc-cell-depth"); ++j) {
        std::string paramPrefix = prefix_ + "_bi_r";
        if(i > 1)
          paramPrefix += "_l" + std::to_string(i);
        if(i > 1 || j > 1)
          paramPrefix += "_cell" + std::to_string(j);
        stacked.push_back(rnn::cell(graph)
                          ("prefix", paramPrefix));
      }
      rnnBw.push_back(stacked);
    }

    auto context = concatenate({rnnFw->transduce(embeddings, mask),
                                rnnBw->transduce(embeddings, mask)},
                                axis=1);

    if(second > 0) {
      // add more layers (unidirectional) by transducing the output of the
      // previous bidirectional RNN through multiple layers

      // construct RNN first
      auto rnnUni = rnn::rnn(graph)
                    ("type", opt<std::string>("enc-cell"))
                    ("dimInput", 2 * opt<int>("dim-rnn"))
                    ("dimState", opt<int>("dim-rnn"))
                    ("dropout", dropoutRnn)
                    ("layer-normalization", opt<bool>("layer-normalization"))
                    ("skip", opt<bool>("skip"));

      for(int i = first + 1; i <= second + first; ++i) {
        auto stacked = rnn::stacked_cell(graph);
        for(int j = 1; j <= opt<int>("enc-cell-depth"); ++j) {
          std::string paramPrefix = prefix_ + "_l" + std::to_string(i) + "_cell" + std::to_string(j);
          stacked.push_back(rnn::cell(graph)
                            ("prefix", paramPrefix));
        }
        rnnUni.push_back(stacked);
      }

      // transduce context to new context
      context = rnnUni->transduce(context);
    }
    return context;
  }

  template <class... Args>
  EncoderS2S(Ptr<Config> options, Args... args)
      : EncoderBase(options, args...) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch,
                          size_t encoderIndex) {
    using namespace keywords;

    // create source embeddings
    int dimVoc = opt<std::vector<int>>("dim-vocabs")[encoderIndex];
    int dimEmb = opt<int>("dim-emb");

    auto embFactory = embedding(graph)
                      ("prefix", prefix_ + "_Wemb")
                      ("dimVocab", dimVoc)
                      ("dimEmb", dimEmb);

    if(options_->has("embedding-fix-src"))
      embFactory
        ("fixed", opt<bool>("embedding-fix-src"));

    if(options_->has("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      embFactory
        ("embFile", embFiles[encoderIndex])
        ("normalization", opt<bool>("embedding-normalization"));
    }

    auto embeddings = embFactory.construct();

    // select embeddings that occur in the batch
    Expr batchEmbeddings, batchMask;
    std::tie(batchEmbeddings, batchMask)
      = EncoderBase::lookup(embeddings, batch, encoderIndex);

    // apply dropout over source words
    float dropProb = inference_ ? 0 : opt<float>("dropout-src");
    if(dropProb) {
      int srcWords = batchEmbeddings->shape()[2];
      auto dropMask = graph->dropout(dropProb, {1, 1, srcWords});
      batchEmbeddings = dropout(batchEmbeddings, mask = dropMask);
    }

    float noiseStddev = inference_ ? 0 : opt<float>("noise-src");
    if(noiseStddev) {
      int dimRnn = batchEmbeddings->shape()[1];
      int srcWords = batchEmbeddings->shape()[2];
      auto noiseMask = graph->gaussian(0.f, noiseStddev, {1, dimRnn, srcWords});
      batchEmbeddings = batchEmbeddings + noiseMask;
    }

    Expr context = applyEncoderRNN(graph, batchEmbeddings, batchMask,
                                   opt<std::string>("enc-type"));

    return New<EncoderState>(context, batchMask, batch);
  }
};

class DecoderS2S : public DecoderBase {
private:
  Ptr<rnn::RNN> rnn_;
  Expr tiedOutputWeights_;

Ptr<rnn::RNN> constructDecoderRNN(Ptr<ExpressionGraph> graph,
                                  Ptr<DecoderState> state) {
  float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
  auto rnn = rnn::rnn(graph)
             ("type", opt<std::string>("dec-cell"))
             ("dimInput", opt<int>("dim-emb"))
             ("dimState", opt<int>("dim-rnn"))
             ("dropout", dropoutRnn)
             ("layer-normalization", opt<bool>("layer-normalization"))
             ("skip", opt<bool>("skip"));

  size_t decoderLayers = opt<size_t>("dec-depth");
  size_t decoderBaseDepth = opt<size_t>("dec-cell-base-depth");
  size_t decoderHighDepth = opt<size_t>("dec-cell-high-depth");

  // setting up conditional (transitional) cell
  auto baseCell = rnn::stacked_cell(graph);
  for(int i = 1; i <= decoderBaseDepth; ++i) {
    auto paramPrefix = prefix_ + "_cell" + std::to_string(i);
    baseCell.push_back(rnn::cell(graph)
                       ("prefix", paramPrefix)
                       ("final", i > 1));
    if(i == 1)
      baseCell.push_back(rnn::attention(graph)
                         ("prefix", prefix_)
                         .set_state(state->getEncoderState()));
  }
  // Add cell to RNN (first layer)
  rnn.push_back(baseCell);

  // Add more cells to RNN (stacked RNN)
  for(int i = 2; i <= decoderLayers; ++i) {
    // deep transition
    auto highCell = rnn::stacked_cell(graph);

    for(int j = 1; j <= decoderHighDepth; j++) {
      auto paramPrefix = prefix_ + "_l" + std::to_string(i) + "_cell" + std::to_string(j);
      highCell.push_back(rnn::cell(graph)
                         ("prefix", paramPrefix));
    }

    // @TODO:
    // dec-high-context : none repeat conditional conditional-repeat
    // conditional and conditional-repeat require dec-cell-high-depth > 1

    // Repeat attention output as input for each layer
    //if(opt<std::string>("dec-high-context") == "repeat") {
    //  highCell.add_input(
    //    [](Ptr<rnn::RNN> rnn) {
    //      return rnn->at(0)->as<rnn::StackedCell>()
    //                ->at(1)->as<rnn::Attention>()
    //                ->getContext();
    //    }
    //  );
    //  highCell("dimInputExtra", 2 * opt<int>("dim-rnn"));
    //}

    // Add cell to RNN (more layers)
    rnn.push_back(highCell);
  }

  return rnn.construct();
}

public:
  template <class... Args>
  DecoderS2S(Ptr<Config> options, Args... args)
      : DecoderBase(options, args...) {}

  virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) {
    using namespace keywords;

    // average the source context weighted by the batch mask
    // this will remove padded zeros from the average
    auto meanContext = weighted_average(encState->getContext(),
                                        encState->getMask(),
                                        axis = 2);

    auto graph = meanContext->graph();

    // apply single layer network to mean to map into decoder space
    auto mlp = mlp::mlp(graph)
               .push_back(mlp::dense(graph)
                          ("prefix", prefix_ + "_ff_state")
                          ("dim", opt<int>("dim-rnn"))
                          ("activation", mlp::act::tanh)
                          ("layer-normalization", opt<bool>("layer-normalization")));
    auto start = mlp->apply(meanContext);

    rnn::States startStates(opt<size_t>("dec-depth"), {start, start});
    return New<DecoderState>(startStates, nullptr, encState);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    auto embeddings = state->getTargetEmbeddings();

    // dropout target words
    float dropoutTrg = inference_ ? 0 : opt<float>("dropout-trg");
    if(dropoutTrg) {
      int trgWords = embeddings->shape()[2];
      auto trgWordDrop = graph->dropout(dropoutTrg, {1, 1, trgWords});
      embeddings = dropout(embeddings, mask = trgWordDrop);
    }

    if(!rnn_)
      rnn_ = constructDecoderRNN(graph, state);

    // apply RNN to embeddings, initialized with encoder context mapped into
    // decoder space
    auto decoderContext = rnn_->transduce(embeddings, state->getStates());

    // retrieve the last state per layer. They are required during translation
    // in order to continue decoding for the next word
    rnn::States decoderStates = rnn_->lastCellStates();

    // retrieve all the aligned contexts computed by the attention mechanism
    auto att = rnn_->at(0)->as<rnn::StackedCell>()->at(1)->as<rnn::Attention>();
    auto alignedContext = att->getContext();

    // construct deep output multi-layer network layer-wise
    auto layer1 = mlp::dense(graph)
                  ("prefix", prefix_ + "_ff_logit_l1")
                  ("dim", opt<int>("dim-emb"))
                  ("activation", mlp::act::tanh)
                  ("layer-normalization", opt<bool>("layer-normalization"));

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs").back();

    auto layer2 = mlp::dense(graph)
                  ("prefix", prefix_ + "_ff_logit_l2")
                  ("dim", dimTrgVoc);
    if(opt<bool>("tied-embeddings"))
      layer2.tie_transposed("W", prefix_ + "_Wemb");

    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto logits = mlp::mlp(graph)
                  .push_back(layer1)
                  .push_back(layer2)
                  ->apply(embeddings, decoderContext, alignedContext);

    // return unormalized(!) probabilities
    return New<DecoderState>(decoderStates, logits, state->getEncoderState());
  }

  // helper function for guided alignment
  virtual const std::vector<Expr> getAlignments() {
    auto att = rnn_->at(0)->as<rnn::StackedCell>()->at(1)->as<rnn::Attention>();
    return att->getAlignments();
  }
};

typedef EncoderDecoder<EncoderS2S, DecoderS2S> S2S;

}
