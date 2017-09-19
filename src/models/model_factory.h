#pragma once

#include "layers/factory.h"
#include "models/s2s.h"
#include "models/amun.h"
#include "models/nematus.h"
#include "models/transformer.h"

#define REGISTER_ENCODER(name, className)\
do {\
if(options_->get<std::string>("type") == name)\
  return New<className>(options_);\
} while(0)

#define REGISTER_DECODER(name, className)\
do {\
if(options_->get<std::string>("type") == name)\
  return New<className>(options_);\
} while(0)

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

    REGISTER_ENCODER("s2s", EncoderS2S);
    REGISTER_ENCODER("transformer", EncoderTransformer);

    UTIL_THROW2("Unknown encoder type");
  }
};

typedef Accumulator<EncoderFactory> encoder;

class DecoderFactory : public Factory {
public:
  DecoderFactory(Ptr<ExpressionGraph> graph = nullptr) : Factory(graph) {}

  virtual Ptr<DecoderBase> construct() {

    REGISTER_DECODER("s2s", DecoderS2S);
    REGISTER_DECODER("transformer", DecoderTransformer);

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

    REGISTER_ENCODER_DECODER("amun", Amun);
    REGISTER_ENCODER_DECODER("nematus", Nematus);

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

Ptr<EncoderDecoder> by_type(std::string type,
                            Ptr<Options> options) {

  if(type == "s2s" || type == "amun" || type == "nematus") {
    return models::encoder_decoder()
           (options)
           .push_back(models::encoder()
                      ("type", "s2s")
                      ("original-type", type))
           .push_back(models::decoder()
                      ("type", "s2s")
                      ("original-type", type))
           .construct();
  }

  if(type == "transformer") {
    return models::encoder_decoder()
           (options)
           .push_back(models::encoder()
                      ("type", "transformer"))
           .push_back(models::decoder()
                      ("type", "s2s"))
           .construct();
  }

  if(type == "transformer2") {
    return models::encoder_decoder()
           (options)
           .push_back(models::encoder()
                      ("type", "transformer"))
           .push_back(models::decoder()
                      ("type", "transformer"))
           .construct();
  }

  if(type == "lm") {
    return models::encoder_decoder()
           (options)
           ("type", "s2s")
           .push_back(models::decoder()
                      ("index", options->has("index") ?
                                options->get<size_t>("index") : 0))
           .construct();
  }

  if(type == "multi-s2s") {
    size_t numEncoders = 2;
    auto ms2sFactory = models::encoder_decoder()
                       (options)
                       ("type", "s2s");

    for(size_t i = 0; i < numEncoders; ++i)
      ms2sFactory.push_back(models::encoder()
                            ("prefix", "encoder" + std::to_string(i + 1))
                            ("index", i));

    ms2sFactory.push_back(models::decoder()
                          ("index", numEncoders));

    return ms2sFactory.construct();
  }

  UTIL_THROW2("Unknown model type: " + type);
}

Ptr<EncoderDecoder> from_options(Ptr<Options> options) {
  std::string type = options->get<std::string>("type");
  return by_type(type, options);
}

Ptr<EncoderDecoder> from_config(Ptr<Config> config) {
  Ptr<Options> options = New<Options>();
  options->merge(config);
  return from_options(options);
}

}

}
