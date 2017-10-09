#pragma once

#include "layers/factory.h"
#include "models/amun.h"
#include "models/model_base.h"
#include "models/nematus.h"
#include "models/s2s.h"
#include "models/transformer.h"
#include "models/transformer_gru.h"

#include "examples/mnist/model.h"
#ifdef CUDNN
#include "examples/mnist/model_lenet.h"
#endif

#define REGISTER_ENCODER_DECODER(name, className)\
do {\
if(options_->get<std::string>("type") == name)\
  encdec = New<className>(options_);\
} while(0)


namespace marian {

namespace models {

class EncoderFactory : public Factory {
public:
  EncoderFactory(Ptr<ExpressionGraph> graph = nullptr) : Factory(graph) {}

  virtual Ptr<EncoderBase> construct() {
    if(options_->get<std::string>("type") == "s2s")
      return New<EncoderS2S>(options_);
    if(options_->get<std::string>("type") == "transformer")
      return New<EncoderTransformer>(options_);
    if(options_->get<std::string>("type") == "transformer_gru")
      return New<EncoderTransformerGRU>(options_);

    UTIL_THROW2("Unknown encoder type");
  }
};

typedef Accumulator<EncoderFactory> encoder;

class DecoderFactory : public Factory {
public:
  DecoderFactory(Ptr<ExpressionGraph> graph = nullptr) : Factory(graph) {}

  virtual Ptr<DecoderBase> construct() {

    if(options_->get<std::string>("type") == "s2s")
      return New<DecoderS2S>(options_);
    if(options_->get<std::string>("type") == "transformer")
      return New<DecoderTransformer>(options_);
    if(options_->get<std::string>("type") == "transformer_gru")
      return New<DecoderTransformerGRU>(options_);

    UTIL_THROW2("Unknown decoder type");
  }
};

typedef Accumulator<DecoderFactory> decoder;

class EncoderDecoderFactory : public Factory {
private:
  std::vector<encoder> encoders_;
  std::vector<decoder> decoders_;

public:
  EncoderDecoderFactory(Ptr<ExpressionGraph> graph = nullptr) : Factory(graph) {}

  Accumulator<EncoderDecoderFactory> push_back(encoder enc) {
    encoders_.push_back(enc);
    return Accumulator<EncoderDecoderFactory>(*this);
  }

  Accumulator<EncoderDecoderFactory> push_back(decoder dec) {
    decoders_.push_back(dec);
    return Accumulator<EncoderDecoderFactory>(*this);
  }

  virtual Ptr<EncoderDecoder> construct() {
    Ptr<EncoderDecoder> encdec;

    if(options_->get<std::string>("type") == "amun")
      encdec = New<Amun>(options_);
    if(options_->get<std::string>("type") == "nematus")
      encdec = New<Nematus>(options_);

    if(!encdec)
      encdec = New<EncoderDecoder>(options_);

    for(auto& ef: encoders_)
      encdec->push_back(ef(options_).construct());

    for(auto& df: decoders_)
      encdec->push_back(df(options_).construct());

    return encdec;
  }
};

typedef Accumulator<EncoderDecoderFactory> encoder_decoder;

Ptr<ModelBase> by_type(std::string type,
                       Ptr<Options> options) {

  if(type == "s2s" || type == "amun" || type == "nematus") {
    return models::encoder_decoder()(options)
        .push_back(models::encoder()("type", "s2s")("original-type", type))
        .push_back(models::decoder()("type", "s2s")("original-type", type))
        .construct();
  }

  if(type == "transformer") {
    return models::encoder_decoder()(options)
        .push_back(models::encoder()("type", "transformer"))
        .push_back(models::decoder()("type", "transformer"))
        .construct();
  }

  if(type == "transformer_gru") {
    return models::encoder_decoder()(options)
        .push_back(models::encoder()("type", "transformer_gru"))
        .push_back(models::decoder()("type", "transformer_gru"))
        .construct();
  }

  if(type == "transformer_s2s") {
    return models::encoder_decoder()(options)
        .push_back(models::encoder()("type", "transformer"))
        .push_back(models::decoder()("type", "s2s"))
        .construct();
  }

  if(type == "lm") {
    auto idx = options->has("index") ? options->get<size_t>("index") : 0;
    return models::encoder_decoder()(options)("type", "s2s")
        .push_back(models::decoder()("index", idx))
        .construct();
  }

  if(type == "multi-s2s") {
    size_t numEncoders = 2;
    auto ms2sFactory = models::encoder_decoder()(options)("type", "s2s");

    for(size_t i = 0; i < numEncoders; ++i) {
      auto prefix = "encoder" + std::to_string(i + 1);
      ms2sFactory.push_back(models::encoder()("prefix", prefix)("index", i));
    }

    ms2sFactory.push_back(models::decoder()("index", numEncoders));

    return ms2sFactory.construct();
  }

  // @TODO: examples should be compiled optionally
  if(type == "mnist-ffnn") {
    return New<MnistFeedForwardNet>(options);
  }
#ifdef CUDNN
  if(type == "mnist-lenet") {
    return New<MnistLeNet>(options);
  }
#endif

  UTIL_THROW2("Unknown model type: " + type);
}

Ptr<ModelBase> from_options(Ptr<Options> options) {
  std::string type = options->get<std::string>("type");
  return by_type(type, options);
}

Ptr<ModelBase> from_config(Ptr<Config> config) {
  Ptr<Options> options = New<Options>();
  options->merge(config);
  return from_options(options);
}

}
}
