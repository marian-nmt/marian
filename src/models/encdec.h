#pragma once

#include "data/corpus.h"
#include "data/batch_generator.h"

#include "training/config.h"
#include "graph/expression_graph.h"
#include "layers/rnn.h"
#include "layers/param_initializers.h"
#include "layers/generic.h"
#include "common/logging.h"
#include "models/lex_probs.h"

namespace marian {

struct EncoderState {
  virtual Expr getContext() = 0;
  virtual Expr getMask() = 0;
};

struct DecoderState {
  virtual Ptr<EncoderState> getEncoderState() = 0;
  virtual Expr getProbs() = 0;
  virtual void setProbs(Expr) = 0;
  virtual Ptr<DecoderState> select(const std::vector<size_t>&) = 0;
};

class EncoderBase {
  protected:
    Ptr<Config> options_;
    std::string prefix_{"encoder"};
    bool inference_{false};

    virtual std::tuple<Expr, Expr>
    prepareSource(Expr emb, Ptr<data::CorpusBatch> batch, size_t index) {
      using namespace keywords;
      
      auto subBatch = (*batch)[index];
      
      int dimBatch = subBatch->batchSize();
      int dimEmb = emb->shape()[1];
      int dimWords = subBatch->batchWidth();

      auto graph = emb->graph();
      auto x = reshape(rows(emb, subBatch->indeces()), {dimBatch, dimEmb, dimWords});
      auto xMask = graph->constant(shape={dimBatch, 1, dimWords},
                                   init=inits::from_vector(subBatch->mask()));
      return std::make_tuple(x, xMask);
    }

  public:
    template <class ...Args>
    EncoderBase(Ptr<Config> options, Args ...args)
     : options_(options),
       prefix_(Get(keywords::prefix, "encoder", args...)),
       inference_(Get(keywords::inference, false, args...))
      {}

    virtual Ptr<EncoderState>
    build(Ptr<ExpressionGraph>, Ptr<data::CorpusBatch>, size_t = 0) = 0;
};

class DecoderBase {
  protected:
    Ptr<Config> options_;
    Ptr<LexProbs> lexProbs_;
    Ptr<sparse::CSR> lf_;
    bool inference_{false};
    
  public:
    template <class ...Args>
    DecoderBase(Ptr<Config> options, Args ...args)
     : options_(options),
       lexProbs_(Get(keywords::lex_probs, nullptr, args...)),
       inference_(Get(keywords::inference, false, args...)) {}
    
    virtual std::tuple<Expr, Expr, Expr>
    groundTruth(Ptr<ExpressionGraph> graph,
                Ptr<data::CorpusBatch> batch) {
      using namespace keywords;

      int dimVoc = options_->get<std::vector<int>>("dim-vocabs").back();
      int dimEmb = options_->get<int>("dim-emb");

      auto yEmb = Embedding("Wemb_dec", dimVoc, dimEmb)(graph);
      
      auto subBatch = batch->back();
      int dimBatch = subBatch->batchSize();
      int dimWords = subBatch->batchWidth();

      auto y = reshape(rows(yEmb, subBatch->indeces()),
                       {dimBatch, dimEmb, dimWords});

      auto yMask = graph->constant(shape={dimBatch, 1, dimWords},
                                   init=inits::from_vector(subBatch->mask()));
          
      auto yIdx = graph->constant(shape={(int)subBatch->indeces().size(), 1},
                                  init=inits::from_vector(subBatch->indeces()));
    
      auto yShifted = shift(y, {0, 0, 1, 0});
      
      return std::make_tuple(yShifted, yMask, yIdx);
    }

    virtual void setLexicalProbabilites(Ptr<data::CorpusBatch> batch) {
      if(lexProbs_)
        lf_ = lexProbs_->Lf(batch);
    }
    
    virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) = 0;
    
    virtual Expr selectEmbeddings(Ptr<ExpressionGraph> graph,
                                 const std::vector<size_t>& embIdx) {
      using namespace keywords;
      
      int dimTrgEmb = options_->get<int>("dim-emb");
      int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();

      Expr selectedEmbs;
      if(embIdx.empty()) {
        selectedEmbs = graph->constant(shape={1, dimTrgEmb},
                                       init=inits::zeros);
      }
      else {
        auto yEmb = Embedding("Wemb_dec", dimTrgVoc, dimTrgEmb)(graph);
        selectedEmbs = reshape(rows(yEmb, embIdx),
                               {1, yEmb->shape()[1], 1, (int)embIdx.size()});
      }
      
      return selectedEmbs;
    }
    
    virtual Ptr<LexProbs> getLexProbs() {
      return lexProbs_;
    }
    
    virtual Ptr<DecoderState> step(Expr embeddings,
                                   Ptr<DecoderState>,
                                   bool single=false) = 0;
};

class EncoderDecoderBase {
  public:
    
    virtual void load(Ptr<ExpressionGraph>,
                      const std::string&) = 0;

    virtual void save(Ptr<ExpressionGraph>,
                      const std::string&) = 0;
    
    virtual void save(Ptr<ExpressionGraph>,
                      const std::string&, bool) = 0;

    virtual Expr selectEmbeddings(Ptr<ExpressionGraph> graph,
                                  const std::vector<size_t>&) = 0;
    
    virtual Ptr<DecoderState>
    step(Expr, Ptr<DecoderState>, bool=false) = 0;

    virtual Expr build(Ptr<ExpressionGraph> graph,
                       Ptr<data::CorpusBatch> batch) = 0;
    
    virtual Ptr<EncoderBase> getEncoder() = 0;
    virtual Ptr<DecoderBase> getDecoder() = 0;
};

template <class Encoder, class Decoder>
class EncoderDecoder : public EncoderDecoderBase {
  protected:
    Ptr<Config> options_;
    Ptr<EncoderBase> encoder_;
    Ptr<DecoderBase> decoder_;
    Ptr<LexProbs> lexProbs_;
    bool inference_{false};

  public:

    template <class ...Args>
    EncoderDecoder(Ptr<Config> options, Args ...args)
     : options_(options),
       encoder_(New<Encoder>(options, args...)),
       decoder_(New<Decoder>(options, args...)),
       lexProbs_(Get(keywords::lex_probs, nullptr, args...)),
       inference_(Get(keywords::inference, false, args...))
    { }
    
    Ptr<EncoderBase> getEncoder() {
      return encoder_;
    }

    Ptr<DecoderBase> getDecoder() {
      return decoder_;
    }
    
    virtual void load(Ptr<ExpressionGraph> graph,
                       const std::string& name) {
      graph->load(name);
    }
    
    virtual void save(Ptr<ExpressionGraph> graph,
                      const std::string& name,
                      bool saveTranslatorConfig) {
      // ignore config for now
      graph->save(name);
      options_->saveModelParameters(name);
    }
    
    virtual void save(Ptr<ExpressionGraph> graph,
                      const std::string& name) {
      graph->save(name);
      options_->saveModelParameters(name);
    }
    
    virtual void clear(Ptr<ExpressionGraph> graph) {
      graph->clear();
      encoder_ = New<Encoder>(options_,
                              keywords::lex_probs=lexProbs_,
                              keywords::inference=inference_);
      decoder_ = New<Decoder>(options_,
                              keywords::lex_probs=lexProbs_,
                              keywords::inference=inference_);
    }

    virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                         Ptr<data::CorpusBatch> batch) {
      decoder_->setLexicalProbabilites(batch);
      return decoder_->startState(encoder_->build(graph, batch));
    }
    
    virtual Ptr<DecoderState>
    step(Expr embeddings,
         Ptr<DecoderState> state,
         bool single=false) {
      return decoder_->step(embeddings, state, single);
    }
    
    virtual Expr selectEmbeddings(Ptr<ExpressionGraph> graph,
                                  const std::vector<size_t>& embIdx) {
      return decoder_->selectEmbeddings(graph, embIdx);
    }

    virtual Expr build(Ptr<ExpressionGraph> graph,
                       Ptr<data::CorpusBatch> batch) {
      using namespace keywords;

      clear(graph);
      auto state = startState(graph, batch);
      
      Expr trgEmbeddings, trgMask, trgIdx;
      std::tie(trgEmbeddings, trgMask, trgIdx) = decoder_->groundTruth(graph, batch);
      
      auto nextState = step(trgEmbeddings, state);
      
      auto cost = CrossEntropyCost("cost")(nextState->getProbs(),
                                           trgIdx, mask=trgMask);

      return cost;
    }

    Ptr<data::BatchStats> collectStats(Ptr<ExpressionGraph> graph) {
      auto stats = New<data::BatchStats>();
      
      size_t step = 10;
      size_t maxLength = options_->get<size_t>("max-length");
      size_t numFiles = options_->get<std::vector<std::string>>("train-sets").size();
      for(size_t i = step; i <= maxLength; i += step) {
        size_t batchSize = step;
        std::vector<size_t> lengths(numFiles, i);
        bool fits = true;
        do {
          auto batch = data::CorpusBatch::fakeBatch(lengths, batchSize);
          build(graph, batch);
          fits = graph->fits();
          if(fits)
            stats->add(batch);
          batchSize += step;
        }
        while(fits);
      }
      return stats;
    }
};

}
