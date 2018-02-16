#include "marian.h"
#include "models/model_factory.h"

#include "models/s2s.h"
#include "models/char_s2s.h"
#include "models/transformer.h"
#include "models/hardatt.h"
#include "models/amun.h"
#include "models/nematus.h"
#include "models/encdec.h"

#include "examples/mnist/model.h"
#include "examples/mnist/model_lenet.h"

namespace marian {
namespace models {

Ptr<EncoderBase> EncoderFactory::construct() {
  if(options_->get<std::string>("type") == "s2s")
    return New<EncoderS2S>(options_);
  if(options_->get<std::string>("type") == "char-s2s")
    return New<CharS2SEncoder>(options_);
  if(options_->get<std::string>("type") == "transformer")
    return New<EncoderTransformer>(options_);

  ABORT("Unknown encoder type");
}

Ptr<DecoderBase> DecoderFactory::construct() {
  if(options_->get<std::string>("type") == "s2s")
    return New<DecoderS2S>(options_);
  if(options_->get<std::string>("type") == "transformer")
    return New<DecoderTransformer>(options_);
  if(options_->get<std::string>("type") == "hard-att")
    return New<DecoderHardAtt>(options_);
  if(options_->get<std::string>("type") == "hard-soft-att")
    return New<DecoderHardAtt>(options_);

  ABORT("Unknown decoder type");
}

Ptr<EncoderDecoder> EncoderDecoderFactory::construct() {
  Ptr<EncoderDecoder> encdec;

  if(options_->get<std::string>("type") == "amun")
    encdec = New<Amun>(options_);
  if(options_->get<std::string>("type") == "nematus")
    encdec = New<Nematus>(options_);

  if(!encdec)
    encdec = New<EncoderDecoder>(options_);

  for(auto& ef : encoders_)
    encdec->push_back(ef(options_).construct());

  for(auto& df : decoders_)
    encdec->push_back(df(options_).construct());

  return encdec;
}

Ptr<ModelBase> by_type(std::string type, Ptr<Options> options) {
  // clang-format off
  if(type == "s2s" || type == "amun" || type == "nematus") {
    return models::encoder_decoder()(options)
        ("original-type", type)
            .push_back(models::encoder()("type", "s2s"))
            .push_back(models::decoder()("type", "s2s"))
            .construct();
  }

  if(type == "transformer") {
    return models::encoder_decoder()(options)
        .push_back(models::encoder()("type", "transformer"))
        .push_back(models::decoder()("type", "transformer"))
        .construct();
  }

  if(type == "transformer_s2s") {
    return models::encoder_decoder()(options)
        ("original-type", type)
            .push_back(models::encoder()("type", "transformer"))
            .push_back(models::decoder()("type", "s2s"))
            .construct();
  }

  if(type == "lm") {
    auto idx = options->has("index") ? options->get<size_t>("index") : 0;
    std::vector<int> dimVocabs = options->get<std::vector<int>>("dim-vocabs");
    if(idx > 0)
      dimVocabs.resize(idx + 1, dimVocabs.back());
    return models::encoder_decoder()(options)
        ("type", "s2s")
        ("original-type", type)
            .push_back(models::decoder()
                       ("index", idx)
                       ("dim-vocabs", dimVocabs))
            .construct();
  }

  if(type == "hard-att") {
    return models::encoder_decoder()(options)
        ("original-type", type)
            .push_back(models::encoder()("type", "s2s"))
            .push_back(models::decoder()("type", "hard-att"))
            .construct();
  }

  if(type == "hard-soft-att") {
    return models::encoder_decoder()(options)
        ("original-type", type)
            .push_back(models::encoder()("type", "s2s"))
            .push_back(models::decoder()("type", "hard-soft-att"))
            .construct();
  }

  if(type == "multi-s2s") {
    size_t numEncoders = 2;
    auto ms2sFactory = models::encoder_decoder()(options)
        ("type", "s2s")
        ("original-type", type);

    for(size_t i = 0; i < numEncoders; ++i) {
      auto prefix = "encoder" + std::to_string(i + 1);
      ms2sFactory.push_back(models::encoder()("prefix", prefix)("index", i));
    }

    ms2sFactory.push_back(models::decoder()("index", numEncoders));

    return ms2sFactory.construct();
  }

  if(type == "multi-hard-att") {
    size_t numEncoders = 2;
    auto ms2sFactory = models::encoder_decoder()(options)
        ("type", "s2s")
        ("original-type", type);

    for(size_t i = 0; i < numEncoders; ++i) {
      auto prefix = "encoder" + std::to_string(i + 1);
      ms2sFactory.push_back(models::encoder()("prefix", prefix)("index", i));
    }

    ms2sFactory.push_back(models::decoder()
                          ("index", numEncoders)
                          ("type", "hard-soft-att"));

    return ms2sFactory.construct();
  }

  if(type == "multi-transformer") {
    size_t numEncoders = 2;
    auto mtransFactory = models::encoder_decoder()(options)
        ("type", "transformer")
        ("original-type", type);

    for(size_t i = 0; i < numEncoders; ++i) {
      auto prefix = "encoder" + std::to_string(i + 1);
      mtransFactory.push_back(models::encoder()("prefix", prefix)("index", i));
    }
    mtransFactory.push_back(models::decoder()("index", numEncoders));

    return mtransFactory.construct();
  }

  if(type == "lm-transformer") {
    auto idx = options->has("index") ? options->get<size_t>("index") : 0;
    return models::encoder_decoder()(options)
        ("type", "transformer")
        ("original-type", type)
            .push_back(models::decoder()("index", idx))
            .construct();
  }

  // @TODO: examples should be compiled optionally
  if(type == "mnist-ffnn") {
    return New<MnistFeedForwardNet>(options);
  }

  if(type == "mnist-lenet") {
    return New<MnistLeNet>(options);
  }

  if(type == "char-s2s") {
    return models::encoder_decoder()(options)
        ("original-type", type)
            .push_back(models::encoder()("type", "char-s2s"))
            .push_back(models::decoder()("type", "s2s"))
            .construct();
  }

  // clang-format on
  ABORT("Unknown model type: {}", type);
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
