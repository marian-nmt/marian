#pragma once

#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/text_input.h"

#include "3rd_party/threadpool.h"
#include "translator/history.h"
#include "translator/output_collector.h"
#include "translator/printer.h"

#include "models/model_task.h"
#include "translator/scorers.h"

namespace marian {

template <class Search>
class TranslateMultiGPU : public ModelTask {
private:
  Ptr<Config> options_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<std::vector<Ptr<Scorer>>> scorers_;

  Ptr<data::Corpus> corpus_;
  Ptr<Vocab> trgVocab_;
  // Ptr<LexProbs> lexProbs_;

public:
  TranslateMultiGPU(Ptr<Config> options)
      : options_(options),
        corpus_(New<data::Corpus>(options_, true)),
        trgVocab_(New<Vocab>()) {
    auto vocabs = options_->get<std::vector<std::string>>("vocabs");
    trgVocab_->load(vocabs.back());

    // if(options_->has("lexical-table"))
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
          graph->getBackend()->setDevice(graph->getDevice());
          scorers = scorers_[id % devices.size()];
        }

        auto search = New<Search>(options_, scorers);
        auto history = search->search(graph, batch, id);

        std::stringstream best1;
        std::stringstream bestn;
        Printer(options_, trgVocab_, history, best1, bestn);
        collector->Write(history->GetLineNum(),
                         best1.str(),
                         bestn.str(),
                         options_->get<bool>("n-best"));
      };

      threadPool.enqueue(task, sentenceId);

      sentenceId++;
    }
  }
};

template <class Search>
class TranslateServiceMultiGPU : public ModelServiceTask {
private:
  Ptr<Config> options_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<std::vector<Ptr<Scorer>>> scorers_;

  std::vector<size_t> devices_;
  std::vector<Ptr<Vocab>> srcVocabs_;
  Ptr<Vocab> trgVocab_;

public:
  virtual ~TranslateServiceMultiGPU() {}

  TranslateServiceMultiGPU(Ptr<Config> options)
      : options_(options),
        devices_(options_->get<std::vector<size_t>>("devices")),
        trgVocab_(New<Vocab>()) {
    init();
  }

  void init() {
    // initialize vocabs
    auto vocabPaths = options_->get<std::vector<std::string>>("vocabs");
    std::vector<int> maxVocabs = options_->get<std::vector<int>>("dim-vocabs");
    for(size_t i = 0; i < vocabPaths.size() - 1; ++i) {
      Ptr<Vocab> vocab = New<Vocab>();
      vocab->load(vocabPaths[i], maxVocabs[i]);
      srcVocabs_.emplace_back(vocab);
    }
    trgVocab_->load(vocabPaths.back());

    // initialize scorers
    for(auto& device : devices_) {
      auto graph = New<ExpressionGraph>(true);
      graph->setDevice(device);
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graphs_.push_back(graph);

      auto scorers = createScorers(options_);
      for(auto scorer : scorers)
        scorer->init(graph);
      scorers_.push_back(scorers);
    }
  }

  std::vector<std::string> run(const std::vector<std::string>& inputs) {
    auto corpus_ = New<data::TextInput>(inputs, srcVocabs_, options_);
    data::BatchGenerator<data::TextInput> bg(corpus_, options_);

    auto collector = New<StringCollector>();
    size_t sentenceId = 0;

    bg.prepare(false);

    {
      ThreadPool threadPool_(devices_.size(), devices_.size());

      while(bg) {
        auto batch = bg.next();

        auto task = [=](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local std::vector<Ptr<Scorer>> scorers;

          if(!graph) {
            graph = graphs_[id % devices_.size()];
            graph->getBackend()->setDevice(graph->getDevice());
            scorers = scorers_[id % devices_.size()];
          }

          auto search = New<Search>(options_, scorers);
          auto history = search->search(graph, batch, id);

          std::stringstream best1;
          std::stringstream bestn;
          Printer(options_, trgVocab_, history, best1, bestn);
          collector->add(history->GetLineNum(), best1.str(), bestn.str());
        };

        threadPool_.enqueue(task, sentenceId);
        sentenceId++;
      }
    }

    return collector->collect(options_->get<bool>("n-best"));
  }
};
}
