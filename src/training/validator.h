#pragma once

#include "training/config.h"
#include "graph/expression_graph.h"
#include "data/corpus.h"
#include "data/batch_generator.h"

namespace marian {
  
  class Validator {
    protected:
      Ptr<Config> options_;
      Ptr<data::BatchGenerator<data::Corpus>> batchGenerator_;
    
    public:
      Validator(std::vector<Ptr<Vocab>> vocabs,
                Ptr<Config> options)
       : options_(options) {
        
        using namespace data;
        
        auto validPaths = options_->get<std::vector<std::string>>("valid-sets");
        auto corpus = New<Corpus>(validPaths, vocabs, options);
        batchGenerator_ = New<BatchGenerator<Corpus>>(corpus, options);
      }
      
      virtual std::string type() = 0;
      virtual float validate(Ptr<ExpressionGraph>) = 0;
    
  };
  
  template <class Builder>
  class CrossEntropyValidator : public Validator {
    private:
      Ptr<Builder> builder_;
    
    public:
      CrossEntropyValidator(std::vector<Ptr<Vocab>> vocabs,
                            Ptr<Config> options)
       : Validator(vocabs, options),
         builder_(New<Builder>(options)) {}
       
      float validate(Ptr<ExpressionGraph> graph) {
        float cost = 0;
        size_t samples = 0;
        
        batchGenerator_->prepare(false);
        while(*batchGenerator_) {
          auto batch = batchGenerator_->next();
          builder_->build(graph, batch);
          graph->forward();
          
          cost += graph->topNode()->scalar() * batch->size();
          samples += batch->size();
        }
        
        return cost / samples;
      }
      
      std::string type() { return "cross-entropy"; }
  };
  
  template <class Builder>
  class PerplexityValidator : public Validator {
    private:
      Ptr<Builder> builder_;
    
    public:
      PerplexityValidator(std::vector<Ptr<Vocab>> vocabs,
                          Ptr<Config> options)
       : Validator(vocabs, options),
         builder_(New<Builder>(options)) {}
       
      float validate(Ptr<ExpressionGraph> graph) {
        float cost = 0;
        size_t words = 0;
        
        batchGenerator_->prepare(false);
        while(*batchGenerator_) {
          auto batch = batchGenerator_->next();
          builder_->build(graph, batch);
          graph->forward();
          
          cost += graph->topNode()->scalar() * batch->size();
          words += batch->words();
        }
        
        return expf(cost / words);
      }
      
      std::string type() { return "perplexity"; }
      
  };
  
  template <class Builder>
  std::vector<Ptr<Validator>> Validators(std::vector<Ptr<Vocab>> vocabs,
                                         Ptr<Config> options) {
    std::vector<Ptr<Validator>> validators;
    
    auto validMetrics = options->get<std::vector<std::string>>("valid-metrics");
    for(auto metric : validMetrics) {
      if(metric == "cross-entropy") {
        auto validator = New<CrossEntropyValidator<Builder>>(vocabs, options);
        validators.push_back(validator);
      }
      if(metric == "perplexity") {
        auto validator = New<PerplexityValidator<Builder>>(vocabs, options);
        validators.push_back(validator);        
      }
    }
    return validators;
  }
  
}