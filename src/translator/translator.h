#pragma once

#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/shortlist.h"
#include "data/text_input.h"

#include "3rd_party/threadpool.h"
#include "translator/history.h"
#include "translator/output_collector.h"
#include "translator/output_printer.h"

#include "models/model_task.h"
#include "translator/scorers.h"

namespace marian {

template <class Search>
class Translate : public ModelTask {
private:
  Ptr<Config> options_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<std::vector<Ptr<Scorer>>> scorers_;

  Ptr<data::Corpus> corpus_;
  Ptr<Vocab> trgVocab_;
  Ptr<data::ShortlistGenerator> shortlistGenerator_;

public:
  Translate(Ptr<Config> options)
      : options_(options),
        corpus_(New<data::Corpus>(options_, true)),
        trgVocab_(New<Vocab>()) {
    auto vocabs = options_->get<std::vector<std::string>>("vocabs");
    trgVocab_->load(vocabs.back());

    auto srcVocab = corpus_->getVocabs()[0];

    if(options_->has("shortlist"))
      shortlistGenerator_ = New<data::LexicalShortlistGenerator>(
          options_, srcVocab, trgVocab_, 0, 1, vocabs.front() == vocabs.back());

    auto devices = options_->getDevices();

    ThreadPool threadPool(devices.size(), devices.size());
    scorers_.resize(devices.size());
    graphs_.resize(devices.size());

    size_t id = 0;
    for(auto device : devices) {
      auto task = [&](DeviceId device, size_t id) {
        auto graph
            = New<ExpressionGraph>(true, options_->get<bool>("optimize"));
        graph->setDevice(device);
        graph->getBackend()->setClip(options_->get<float>("clip-gemm"));
        graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
        graphs_[id] = graph;

        auto scorers = createScorers(options_);
        for(auto scorer : scorers) {
          scorer->init(graph);
          if(shortlistGenerator_)
            scorer->setShortlistGenerator(shortlistGenerator_);
        }

        scorers_[id] = scorers;
        graph->forward();
      };

      threadPool.enqueue(task, device, id++);
    }
  }

  void run() override {
    data::BatchGenerator<data::Corpus> bg(corpus_, options_);

    auto devices = options_->getDevices();

    ThreadPool threadPool(devices.size(), devices.size());

    size_t batchId = 0;
    auto collector = New<OutputCollector>();
    auto printer = New<OutputPrinter>(options_, trgVocab_);
    if(options_->get<bool>("quiet-translation"))
      collector->setPrintingStrategy(New<QuietPrinting>());

    bg.prepare(false);

    // @TODO: unify this and get rid of Config object.
    auto tOptions = New<Options>();
    tOptions->merge(options_);

    while(bg) {
      auto batch = bg.next();

      auto task = [=](size_t id) {
        thread_local Ptr<ExpressionGraph> graph;
        thread_local std::vector<Ptr<Scorer>> scorers;

        if(!graph) {
          graph = graphs_[id % devices.size()];
          scorers = scorers_[id % devices.size()];
        }

        auto search = New<Search>(
            tOptions, scorers, trgVocab_->GetEosId(), trgVocab_->GetUnkId());

        auto histories = search->search(graph, batch);

        for(auto history : histories) {
          std::stringstream best1;
          std::stringstream bestn;
          printer->print(history, best1, bestn);
          collector->Write((long)history->GetLineNum(),
                           best1.str(),
                           bestn.str(),
                           options_->get<bool>("n-best"));
        }
      };

      threadPool.enqueue(task, batchId++);
    }
  }
};

template <class Search>
class TranslateService : public ModelServiceTask {
private:
  Ptr<Config> options_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<std::vector<Ptr<Scorer>>> scorers_;

  std::vector<DeviceId> devices_;
  std::vector<Ptr<Vocab>> srcVocabs_;
  Ptr<Vocab> trgVocab_;

public:
  virtual ~TranslateService() {}

  TranslateService(Ptr<Config> options)
      : options_(options),
        devices_(options_->getDevices()),
        trgVocab_(New<Vocab>()) {
    init();
  }

  void init() override {
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
    for(auto device : devices_) {
      auto graph = New<ExpressionGraph>(true, options_->get<bool>("optimize"));
      graph->setDevice(device);
      graph->getBackend()->setClip(options_->get<float>("clip-gemm"));
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graphs_.push_back(graph);

      auto scorers = createScorers(options_);
      for(auto scorer : scorers)
        scorer->init(graph);
      scorers_.push_back(scorers);
    }
  }

  std::vector<std::string> run(const std::vector<std::string>& inputs) override {
    auto corpus_ = New<data::TextInput>(inputs, srcVocabs_, options_);
    data::BatchGenerator<data::TextInput> bg(corpus_, options_);

    auto collector = New<StringCollector>();
    auto printer = New<OutputPrinter>(options_, trgVocab_);
    size_t batchId = 0;

    // @TODO: unify this and get rid of Config object.
    auto tOptions = New<Options>();
    tOptions->merge(options_);

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
            scorers = scorers_[id % devices_.size()];
          }

          auto search = New<Search>(
              tOptions, scorers, trgVocab_->GetEosId(), trgVocab_->GetUnkId());
          auto histories = search->search(graph, batch);

          for(auto history : histories) {
            std::stringstream best1;
            std::stringstream bestn;
            printer->print(history, best1, bestn);
            collector->add(history->GetLineNum(), best1.str(), bestn.str());
          }
        };

        threadPool_.enqueue(task, batchId);
        batchId++;
      }
    }

    return collector->collect(options_->get<bool>("n-best"));
  }
};
}  // namespace marian
