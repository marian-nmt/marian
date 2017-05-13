#pragma once

#include "data/batch_generator.h"
#include "data/corpus.h"

#include "translator/history.h"
#include "translator/printer.h"
#include "translator/output_collector.h"
#include "3rd_party/threadpool.h"
#include "models/lex_probs.h"

#include "translator/scorers.h"

namespace marian {

Ptr<Scorer> wrapScorer(std::string fname, float weight,
                       std::string model, Ptr<Config> options) {
  
  std::string type = options->get<std::string>("type");
  
  std::cerr << type << std::endl;
  
  if(type == "s2s") {
    
    // hacky hack
    std::vector<size_t> idx = {0, 2};
      
    if(fname == "F5")
      idx = {1, 2};
    
    return New<ScorerWrapper<S2S>>(fname, weight,
                                   model, options,
                                   idx);
  }
  else if(type == "multi-att") {
    return New<ScorerWrapper<MultiS2S>>(fname, weight,
                                         model, options,
                                         std::vector<size_t>({0, 1, 2}));
  }
  else if(type == "hard-att") {
    return New<ScorerWrapper<HardAtt>>(fname, weight,
                                       model, options);
  }
  else if(type == "hard-soft-att") {
    return New<ScorerWrapper<HardSoftAtt>>(fname, weight,
                                           model, options);
  }
  else if(type == "multi-hard-att") {
    return New<ScorerWrapper<MultiHardSoftAtt>>(fname, weight,
                                                model, options,
                                                std::vector<size_t>({0, 1, 2}));
  }
  else {
    return New<ScorerWrapper<S2S>>(fname, weight,
                                   model, options);
  }
}

std::vector<Ptr<Scorer>>
createScorers(Ptr<Config> options) {
  std::vector<Ptr<Scorer>> scorers;
  
  auto models = options->get<std::vector<std::string>>("models");
  int dimVocab = options->get<std::vector<int>>("dim-vocabs").back();
  
  std::vector<float> weights = { 1, 1, 1, 1, 0, 0, 0, 0 };
  
  int i = 0;
  for(auto model : models) {
    std::string fname = "F" + std::to_string(i);
    
    auto mOptions = New<Config>(*options);
    mOptions->loadModelParameters(model);
    scorers.push_back(wrapScorer(fname, weights[i], model, mOptions));
    i++;
  }
  
  scorers.push_back(New<WordPenalty>("F6", weights[i++], dimVocab));
  scorers.push_back(New<UnseenWordPenalty>("F7", weights[i++], dimVocab, 0));
  
  return scorers;
}

template <class Search>
class TranslateMultiGPU : public ModelTask {
  private:
    Ptr<Config> options_;
    std::vector<Ptr<ExpressionGraph>> graphs_;
    std::vector<std::vector<Ptr<Scorer>>> scorers_;
    
    Ptr<data::Corpus> corpus_;
    Ptr<Vocab> trgVocab_;
    //Ptr<LexProbs> lexProbs_;
    
  public:  
    TranslateMultiGPU(Ptr<Config> options)
    : options_(options),
      corpus_(New<data::Corpus>(options_, true)),
      trgVocab_(New<Vocab>()) {
      
      auto vocabs = options_->get<std::vector<std::string>>("vocabs");
      trgVocab_->load(vocabs.back());

      //if(options_->has("lexical-table"))
      //  lexProbs_ = New<LexProbs>(options_,
      //                       corpus_->getVocabs().front(),
      //                       trgVocab_);
        
      auto devices = options_->get<std::vector<int>>("devices");
      for(auto& device : devices) {
        auto graph = New<ExpressionGraph>(true);
        graph->setDevice(device);
        graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
        graphs_.push_back(graph);
        
        auto scorers = createScorers(options);    
        for(auto scorer : scorers)
          scorer->init(graph);
        scorers_.push_back(scorers);
      }
      
    }
    
    void run() {
      
      data::BatchGenerator<data::Corpus> bg(corpus_, options_);
      
      auto devices = options_->get<std::vector<int>>("devices");
      ThreadPool threadPool(devices.size(), devices.size());
      
      auto collector = New<OutputCollector>();
      size_t sentenceId = 0;
      
      bg.prepare(false);
      while(bg) {
        auto batch = bg.next();
        
        auto task = [=](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local std::vector<Ptr<Scorer>> scorers;
          
          if(!graph) {
            graph = graphs_[id % devices.size()];
            cudaSetDevice(graph->getDevice());
            scorers = scorers_[id % devices.size()];
          }
          
          auto search = New<Search>(options_, scorers);
          auto history = search->search(graph, batch, id);
      
          std::stringstream ss;
          Printer(options_, trgVocab_, history, ss);
          collector->Write(history->GetLineNum(), ss.str());
        };
        
        threadPool.enqueue(task, sentenceId);
        
        sentenceId++;
      }
    }
};

template <class Search>
class TranslateSingleGPU : public ModelTask {
  private:
    Ptr<Config> options_;
    Ptr<ExpressionGraph> graph_;
    Ptr<data::Corpus> corpus_;
    Ptr<Vocab> trgVocab_;
    Ptr<LexProbs> lexProbs_;
    
  public:  
    TranslateSingleGPU(Ptr<Config> options)
    : options_(options),
      corpus_(New<data::Corpus>(options_, true)),
      trgVocab_(New<Vocab>()) {
        
      auto vocabs = options_->get<std::vector<std::string>>("vocabs");
      trgVocab_->load(vocabs.back());

      auto devices = options_->get<std::vector<int>>("devices");
      size_t device = devices[0];
      
      if(options_->has("lexical-table"))
        lexProbs_ = New<LexProbs>(options_,
                             corpus_->getVocabs().front(),
                             trgVocab_);
      
      graph_ = New<ExpressionGraph>(true);
      graph_->setDevice(device);
      graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      
      typedef typename Search::model_type Model;
      auto model = New<Model>(options_,
                              keywords::inference=true,
                              keywords::lex_probs=lexProbs_);
      model->load(graph_, options_->get<std::string>("model"));
    }
    
    void run() {
      data::BatchGenerator<data::Corpus> bg(corpus_, options_);
      
      auto collector = New<OutputCollector>();
      size_t sentenceId = 0;
      
      bg.prepare(false);
      while(bg) {
        auto batch = bg.next();
                  
        auto search = New<Search>(options_,
                                  keywords::lex_probs=lexProbs_);
        auto history = search->search(graph_, batch, sentenceId);
    
        std::stringstream ss;
        Printer(options_, trgVocab_, history, ss);
        collector->Write(history->GetLineNum(), ss.str());
        
        sentenceId++;
      }
    }
};


}