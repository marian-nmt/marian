#pragma once

#include "training/config.h"
#include "training/validator.h"

namespace marian {

class ReporterBase {
  public:
    virtual bool keepGoing() = 0;
    virtual void increaseEpoch() = 0;
    virtual size_t batches() = 0;

    virtual size_t stalled() = 0;
    virtual void finished() = 0;
    virtual void update(float cost, Ptr<data::Batch> batch) = 0;

    virtual bool validating() const = 0;
    virtual void validate(Ptr<ExpressionGraph> graph) = 0;

    virtual bool saving() const = 0;
    virtual void save(const std::string& name) = 0;

    virtual void load(const std::string& name) = 0;
};


//template <class Validator>
class Reporter {
  public:
    YAML::Node progress;

    Ptr<Config> options_;
    std::vector<Ptr<Validator>> validators_;

    float costSum{0};

    size_t epochs{1};
    size_t samples{0};
    size_t samplesDisp{0};
    size_t wordsDisp{0};
    size_t batches{0};

    boost::timer::cpu_timer timer;

  public:
    Reporter(Ptr<Config> options) : options_(options) {}

    bool keepGoing() {
      // stop if it reached the maximum number of epochs
      if(options_->get<size_t>("after-epochs") > 0
         && epochs > options_->get<size_t>("after-epochs"))
        return false;

      // stop if it reached the maximum number of batch updates
      if(options_->get<size_t>("after-batches") > 0
         && batches >= options_->get<size_t>("after-batches"))
        return false;

      // stop if the first validator did not improve for a given number of checks
      if(options_->get<size_t>("early-stopping") > 0
         && !validators_.empty()
         && validators_[0]->stalled() >= options_->get<size_t>("early-stopping"))
        return false;

      return true;
    }

    void increaseEpoch() {
      LOG(info, "Seen {} samples", samples);

      epochs++;
      samples = 0;

      LOG(info, "Starting epoch {}", epochs);
    }

    void finished() {
      LOG(info, "Training finshed");
    }

    void addValidator(Ptr<Validator> validator) {
      validators_.push_back(validator);
    }

    bool validating() {
      return (!validators_.empty() && batches % options_->get<size_t>("valid-freq") == 0);
    }

    bool saving() {
      return (batches % options_->get<size_t>("save-freq") == 0);
    }

    void validate(Ptr<ExpressionGraph> graph) {
      if(batches % options_->get<size_t>("valid-freq") == 0) {
        for(auto validator : validators_) {
          if(validator) {
            size_t stalledPrev = validator->stalled();
            float value = validator->validate(graph);
            if(validator->stalled() > 0)
              LOG(valid, "{} : {} : {} : stalled {} times", batches,
                validator->type(), value, validator->stalled());
            else
              LOG(valid, "{} : {} : {} : new best", batches,
                validator->type(), value);
          }
        }
      }
    }

    size_t stalled() {
      for(auto validator : validators_)
        if(validator)
          return validator->stalled();
      return 0;
    }

    void update(float cost, Ptr<data::Batch> batch) {
      costSum += cost * batch->size();
      samples += batch->size();
      samplesDisp += batch->size();
      wordsDisp += batch->words();
      batches++;

      if(batches % options_->get<size_t>("disp-freq") == 0) {
        LOG(info, "Ep. {} : Up. {} : Sen. {} : Cost {:.2f} : Time {} : {:.2f} words/s",
            epochs, batches, samples, costSum / samplesDisp,
            timer.format(2, "%ws"), wordsDisp / std::stof(timer.format(5, "%w")));
        timer.start();
        costSum = 0;
        wordsDisp = 0;
        samplesDisp = 0;
      }
    }

    void load(const std::string& name) {
      std::string nameYaml = name + ".yml";
      if(boost::filesystem::exists(nameYaml)) {
        YAML::Node config = YAML::LoadFile(nameYaml);
        epochs  = config["progress"]["epochs"].as<size_t>();
        batches = config["progress"]["batches"].as<size_t>();
      }
    }

    void save(const std::string& name) {
      YAML::Node config = options_->get();
      config["progress"]["epochs"] = epochs;
      config["progress"]["batches"] = batches;

      std::string nameYaml = name + ".yml";
      std::ofstream fout(nameYaml);
      fout << config;
    }
};

}
