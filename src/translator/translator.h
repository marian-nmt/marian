#pragma once

#include <string>

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

// currently for diagnostics only, will try to mmap files ending in *.bin suffix when enabled.
// @TODO: add this as an actual feature.
#define MMAP 0

#if MMAP
#include "3rd_party/mio/mio.hpp"
#endif

namespace marian {

template <class Search>
class Translate : public ModelTask {
private:
  Ptr<Options> options_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<std::vector<Ptr<Scorer>>> scorers_;

  Ptr<data::Corpus> corpus_;
  Ptr<Vocab> trgVocab_;
  Ptr<const data::ShortlistGenerator> shortlistGenerator_;

  size_t numDevices_;

#if MMAP
  std::vector<mio::mmap_source> mmaps_;
#endif

public:
  Translate(Ptr<Options> options)
    : options_(New<Options>(options->clone())) { // @TODO: clone should return Ptr<Options> same as "with"?
    // This is currently safe as the translator is either created stand-alone or
    // or config is created anew from Options in the validator

    options_->set("inference", true,
                  "shuffle", "none");

    corpus_ = New<data::Corpus>(options_, true);

    auto vocabs = options_->get<std::vector<std::string>>("vocabs");
    trgVocab_ = New<Vocab>(options_, vocabs.size() - 1);
    trgVocab_->load(vocabs.back());
    auto srcVocab = corpus_->getVocabs()[0];

    if(options_->hasAndNotEmpty("shortlist"))
      shortlistGenerator_ = New<data::LexicalShortlistGenerator>(
          options_, srcVocab, trgVocab_, 0, 1, vocabs.front() == vocabs.back());

    auto devices = Config::getDevices(options_);
    numDevices_ = devices.size();

    ThreadPool threadPool(numDevices_, numDevices_);
    scorers_.resize(numDevices_);
    graphs_.resize(numDevices_);

#if MMAP
    auto models = options->get<std::vector<std::string>>("models");
    for(auto model : models) {
      marian::filesystem::Path modelPath(model);
      ABORT_IF(modelPath.extension() != marian::filesystem::Path(".bin"),
              "Non-binarized models cannot be mmapped");
      mmaps_.push_back(std::move(mio::mmap_source(model)));
    }
#endif

    size_t id = 0;
    for(auto device : devices) {
      auto task = [&](DeviceId device, size_t id) {
        auto graph = New<ExpressionGraph>(true);
        auto prec = options_->get<std::vector<std::string>>("precision", {"float32"});
        graph->setDefaultElementType(typeFromString(prec[0]));
        graph->setDevice(device);
        graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
        graphs_[id] = graph;

#if MMAP
        auto scorers = createScorers(options_, mmaps_);
#else
        auto scorers = createScorers(options_);
#endif
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

    if(options_->get<bool>("output-sampling", false)) {
      if(options_->get<size_t>("beam-size") > 1)
        LOG(warn,
            "[warning] Output sampling and beam search (beam-size > 1) are contradictory methods "
            "and using them together is not recommended. Set beam-size to 1");
      if(options_->get<std::vector<std::string>>("models").size() > 1)
        LOG(warn,
            "[warning] Output sampling and model ensembling are contradictory methods and using "
            "them together is not recommended. Use a single model");
    }
  }

  void run() override {
    data::BatchGenerator<data::Corpus> bg(corpus_, options_);

    ThreadPool threadPool(numDevices_, numDevices_);

    size_t batchId = 0;
    auto collector = New<OutputCollector>(options_->get<std::string>("output"));
    auto printer = New<OutputPrinter>(options_, trgVocab_);
    if(options_->get<bool>("quiet-translation"))
      collector->setPrintingStrategy(New<QuietPrinting>());

    bg.prepare();

    bool doNbest = options_->get<bool>("n-best");
    for(auto batch : bg) {
      auto task = [=](size_t id) {
        thread_local Ptr<ExpressionGraph> graph;
        thread_local std::vector<Ptr<Scorer>> scorers;

        if(!graph) {
          graph = graphs_[id % numDevices_];
          scorers = scorers_[id % numDevices_];
        }

        auto search = New<Search>(options_, scorers, trgVocab_);
        auto histories = search->search(graph, batch);

        for(auto history : histories) {
          std::stringstream best1;
          std::stringstream bestn;
          printer->print(history, best1, bestn);
          collector->Write((long)history->getLineNum(),
                           best1.str(),
                           bestn.str(),
                           doNbest);
        }


        // progress heartbeat for MS-internal Philly compute cluster
        // otherwise this job may be killed prematurely if no log for 4 hrs
        if (getenv("PHILLY_JOB_ID")   // this environment variable exists when running on the cluster
            && id % 1000 == 0)  // hard beat once every 1000 batches
        {
          auto progress = 0.f; //fake progress for now
          fprintf(stderr, "PROGRESS: %.2f%%\n", progress);
          fflush(stderr);
        }
      };

      threadPool.enqueue(task, batchId++);

    }
  }
};

template <class Search>
class TranslateService : public ModelServiceTask {
private:
  Ptr<Options> options_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<std::vector<Ptr<Scorer>>> scorers_;

  std::vector<Ptr<Vocab>> srcVocabs_;
  Ptr<Vocab> trgVocab_;
  Ptr<const data::ShortlistGenerator> shortlistGenerator_;

  size_t numDevices_;

public:
  virtual ~TranslateService() {}

  TranslateService(Ptr<Options> options)
    : options_(New<Options>(options->clone())) {
    // initialize vocabs
    options_->set("inference", true);
    options_->set("shuffle", "none");

    auto vocabPaths = options_->get<std::vector<std::string>>("vocabs");
    std::vector<int> maxVocabs = options_->get<std::vector<int>>("dim-vocabs");

    for(size_t i = 0; i < vocabPaths.size() - 1; ++i) {
      Ptr<Vocab> vocab = New<Vocab>(options_, i);
      vocab->load(vocabPaths[i], maxVocabs[i]);
      srcVocabs_.emplace_back(vocab);
    }

    trgVocab_ = New<Vocab>(options_, vocabPaths.size() - 1);
    trgVocab_->load(vocabPaths.back());

    // load lexical shortlist
    if(options_->hasAndNotEmpty("shortlist"))
      shortlistGenerator_ = New<data::LexicalShortlistGenerator>(
          options_, srcVocabs_.front(), trgVocab_, 0, 1, vocabPaths.front() == vocabPaths.back());

    // get device IDs
    auto devices = Config::getDevices(options_);
    numDevices_ = devices.size();

    // initialize scorers
    for(auto device : devices) {
      auto graph = New<ExpressionGraph>(true);

      auto precison = options_->get<std::vector<std::string>>("precision", {"float32"});
      graph->setDefaultElementType(typeFromString(precison[0])); // only use first type, used for parameter type in graph
      graph->setDevice(device);
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graphs_.push_back(graph);

      auto scorers = createScorers(options_);
      for(auto scorer : scorers) {
        scorer->init(graph);
        if(shortlistGenerator_)
          scorer->setShortlistGenerator(shortlistGenerator_);
      }
      scorers_.push_back(scorers);
    }
  }

  std::string run(const std::string& input) override {
    // split tab-separated input into fields if necessary
    auto inputs = options_->get<bool>("tsv", false)
                      ? convertTsvToLists(input, options_->get<size_t>("tsv-fields", 1))
                      : std::vector<std::string>({input});
    auto corpus_ = New<data::TextInput>(inputs, srcVocabs_, options_);
    data::BatchGenerator<data::TextInput> batchGenerator(corpus_, options_);

    auto collector = New<StringCollector>(options_->get<bool>("quiet-translation", false));
    auto printer = New<OutputPrinter>(options_, trgVocab_);
    size_t batchId = 0;

    batchGenerator.prepare();

    {
      ThreadPool threadPool_(numDevices_, numDevices_);

      for(auto batch : batchGenerator) {
        auto task = [=](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local std::vector<Ptr<Scorer>> scorers;

          if(!graph) {
            graph = graphs_[id % numDevices_];
            scorers = scorers_[id % numDevices_];
          }

          auto search = New<Search>(options_, scorers, trgVocab_);
          auto histories = search->search(graph, batch);

          for(auto history : histories) {
            std::stringstream best1;
            std::stringstream bestn;
            printer->print(history, best1, bestn);
            collector->add((long)history->getLineNum(), best1.str(), bestn.str());
          }
        };

        threadPool_.enqueue(task, batchId);
        batchId++;
      }
    }

    auto translations = collector->collect(options_->get<bool>("n-best"));
    return utils::join(translations, "\n");
  }

private:
  // Converts a multi-line input with tab-separated source(s) and target sentences into separate lists
  // of sentences from source(s) and target sides, e.g.
  // "src1 \t trg1 \n src2 \t trg2" -> ["src1 \n src2", "trg1 \n trg2"]
  std::vector<std::string> convertTsvToLists(const std::string& inputText, size_t numFields) {
    std::vector<std::string> outputFields(numFields);

    std::string line;
    std::vector<std::string> lineFields(numFields);
    std::istringstream inputStream(inputText);
    bool first = true;
    while(std::getline(inputStream, line)) {
      utils::splitTsv(line, lineFields, numFields);
      for(size_t i = 0; i < numFields; ++i) {
        if(!first)
          outputFields[i] += "\n";  // join sentences with a new line sign
        outputFields[i] += lineFields[i];
      }
      if(first)
        first = false;
    }

    return outputFields;
  }
};
}  // namespace marian
