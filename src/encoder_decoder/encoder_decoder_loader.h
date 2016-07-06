#pragma once

#include "decoder/loader.h"
#include "encoder_decoder/encoder_decoder.h"

class EncoderDecoderLoader : public Loader {
  public:
    EncoderDecoderLoader(const std::string name,
                         const YAML::Node& config)
     : Loader(name, config) {}

    virtual void Load() {
      std::string path = Get<std::string>("path");
      auto devices = God::Get<std::vector<size_t>>("devices");
      ThreadPool devicePool(devices.size());
      weights_.resize(devices.size());

      size_t i = 0;
      for(auto d : devices) {
        devicePool.enqueue([i, d, &path, this] {
          LOG(info) << "Loading model " << path;
          weights_[i].reset(new Weights(path, d));
        });
        ++i;
      }
    }

    virtual ScorerPtr NewScorer(size_t taskId) {
      size_t i = taskId % weights_.size();
      size_t d = weights_[i]->GetDevice();
      size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
      return ScorerPtr(new EncoderDecoder(name_, config_,
                                          tab, *weights_[i]));
    }

  private:
    std::vector<std::unique_ptr<Weights>> weights_;
};
