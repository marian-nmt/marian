#pragma once

#include <limits>
#include <cstdio>
#include <cstdlib>

#include "training/config.h"
#include "graph/expression_graph.h"
#include "data/corpus.h"
#include "data/batch_generator.h"

namespace marian {

  class Validator {
    protected:
      Ptr<Config> options_;
      std::vector<Ptr<Vocab>> vocabs_;
      float lastBest_;
      size_t stalled_{0};

    public:
      Validator(std::vector<Ptr<Vocab>> vocabs,
                Ptr<Config> options)
       : options_(options),
         vocabs_(vocabs),
         lastBest_{lowerIsBetter() ?
          std::numeric_limits<float>::max() :
          std::numeric_limits<float>::lowest() } {
      }

      virtual std::string type() = 0;

      virtual bool lowerIsBetter() {
        return true;
      }

      size_t stalled() {
        return stalled_;
      }

      virtual float validate(Ptr<ExpressionGraph> graph) {
        using namespace data;
        auto validPaths = options_->get<std::vector<std::string>>("valid-sets");
        auto corpus = New<Corpus>(validPaths, vocabs_, options_);
        Ptr<BatchGenerator<Corpus>> batchGenerator
          = New<BatchGenerator<Corpus>>(corpus, options_);
        batchGenerator->prepare(false);

        float val = validateBG(graph, batchGenerator);
        if(lowerIsBetter() && lastBest_ > val ||
           !lowerIsBetter() && lastBest_ < val) {
            stalled_ = 0;
            lastBest_ = val;
        }
        else {
          stalled_++;
        }
        return val;
      };

      virtual float validateBG(Ptr<ExpressionGraph>,
                               Ptr<data::BatchGenerator<data::Corpus>>) = 0;

  };

  template <class Builder>
  class CrossEntropyValidator : public Validator {
    private:
      Ptr<Builder> builder_;

    public:
      CrossEntropyValidator(std::vector<Ptr<Vocab>> vocabs,
                            Ptr<Config> options)
       : Validator(vocabs, options),
         builder_(New<Builder>(options, keywords::inference=true)) {}

      virtual float validateBG(Ptr<ExpressionGraph> graph,
                               Ptr<data::BatchGenerator<data::Corpus>> batchGenerator) {
        float cost = 0;
        size_t samples = 0;

        while(*batchGenerator) {
          auto batch = batchGenerator->next();
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
         builder_(New<Builder>(options, keywords::inference=true)) {}

      virtual float validateBG(Ptr<ExpressionGraph> graph,
                               Ptr<data::BatchGenerator<data::Corpus>> batchGenerator) {
        float cost = 0;
        size_t words = 0;

        while(*batchGenerator) {
          auto batch = batchGenerator->next();
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
  class ScriptValidator : public Validator {
    private:
      Ptr<Builder> builder_;

    public:
      ScriptValidator(std::vector<Ptr<Vocab>> vocabs,
                      Ptr<Config> options)
       : Validator(vocabs, options),
         builder_(New<Builder>(options, keywords::inference=true)) {}

      std::string exec(const std::string& cmd) {
        std::array<char, 128> buffer;
        std::string result;
        std::shared_ptr<std::FILE> pipe(popen(cmd.c_str(), "r"), pclose);
        if(!pipe)
          UTIL_THROW2("popen() failed!");

        while(!std::feof(pipe.get())) {
          if(std::fgets(buffer.data(), 128, pipe.get()) != NULL)
            result += buffer.data();
        }
        return result;
      }

      virtual float validate(Ptr<ExpressionGraph> graph) {
        using namespace data;
        auto model = options_->get<std::string>("model");

        builder_->save(graph, model + ".dev.npz");

        UTIL_THROW_IF2(!options_->has("valid-script-path"),
                       "valid-script metric but no script given");
        auto command = options_->get<std::string>("valid-script-path");

        auto valStr = exec(command);
        float val = std::atof(valStr.c_str());

        if(lowerIsBetter() && lastBest_ > val ||
           !lowerIsBetter() && lastBest_ < val) {
            stalled_ = 0;
            lastBest_ = val;
        }
        else {
          stalled_++;
        }

        return val;
      };


      virtual float validateBG(Ptr<ExpressionGraph> graph,
                               Ptr<data::BatchGenerator<data::Corpus>> batchGenerator) {
        return 0;
      }

      std::string type() { return "valid-script"; }
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
      if(metric == "valid-script") {
        auto validator = New<ScriptValidator<Builder>>(vocabs, options);
        validators.push_back(validator);
      }
    }

    return validators;
  }

}
