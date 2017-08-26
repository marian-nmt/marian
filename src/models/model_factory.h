#pragma once

#include "layers/factory.h"
#include "models/s2s.h"

namespace marian {

namespace models {

class EncoderFactory : public Factory {
public:
  EncoderFactory(Ptr<ExpressionGraph> graph) : Factory(graph) {}

  virtual Ptr<EncoderBase> construct() {
    std::string type = options_->get<std::string>("type");
    if(type == "s2s") {
      return New<EncoderS2S>(options_);
    } else {
      UTIL_THROW2("Unknown encoder type");
    }
  }
};

typedef Accumulator<EncoderFactory> encoder;

class DecoderFactory : public Factory {
public:
  DecoderFactory(Ptr<ExpressionGraph> graph) : Factory(graph) {}

  virtual Ptr<DecoderBase> construct() {
    std::string type = options_->get<std::string>("type");
    if(type == "s2s") {
      return New<DecoderS2S>(options_);
    } else {
      UTIL_THROW2("Unknown decoder type");
    }
  }
};

typedef Accumulator<DecoderFactory> decoder;

class EncoderDecoderFactory : public Factory {
private:
  std::vector<encoder> encoders_;
  std::vector<decoder> decoders_;

public:
  EncoderDecoderFactory(Ptr<ExpressionGraph> graph) : Factory(graph) {}

  Accumulator<EncoderDecoderFactory> push_back(encoder enc) {
    encoders_.push_back(enc);
    return Accumulator<EncoderDecoderFactory>(*this);
  }

  Accumulator<EncoderDecoderFactory> push_back(decoder dec) {
    decoders_.push_back(dec);
    return Accumulator<EncoderDecoderFactory>(*this);
  }

  virtual Ptr<EncoderDecoder> construct() {
    auto encdec = New<EncoderDecoder>(options_);

    for(auto& ef: encoders_)
      encdec->push_back(ef(options_).construct());

    for(auto& df: decoders_)
      encdec->push_back(df(options_).construct());

    return encdec;
  }
};

typedef Accumulator<EncoderDecoderFactory> encoder_decoder;



}

}