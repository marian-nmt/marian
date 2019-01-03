#include "marian.h"

#include "models/model_factory.h"
#include "models/encoder_decoder.h"
#include "models/encoder_classifier.h"

#include "models/costs.h"

#include "models/amun.h"
#include "models/nematus.h"
#include "models/s2s.h"
#include "models/transformer_factory.h"

#ifdef CUDNN
#include "models/char_s2s.h"
#endif

#ifdef COMPILE_EXAMPLES
#include "examples/mnist/model.h"
#ifdef CUDNN
#include "examples/mnist/model_lenet.h"
#endif
#endif

namespace marian {
namespace models {

Ptr<EncoderBase> EncoderFactory::construct() {
  if(options_->get<std::string>("type") == "s2s")
    return New<EncoderS2S>(options_);

#ifdef CUDNN
  if(options_->get<std::string>("type") == "char-s2s")
    return New<CharS2SEncoder>(options_);
#endif

  if(options_->get<std::string>("type") == "transformer")
    // return New<EncoderTransformer>(options_);
    return NewEncoderTransformer(options_);

  ABORT("Unknown encoder type");
}

Ptr<DecoderBase> DecoderFactory::construct() {
  if(options_->get<std::string>("type") == "s2s")
    return New<DecoderS2S>(options_);
  if(options_->get<std::string>("type") == "transformer")
    // return New<DecoderTransformer>(options_);
    return NewDecoderTransformer(options_);
  ABORT("Unknown decoder type");
}

Ptr<ClassifierBase> ClassifierFactory::construct() {
  if(options_->get<std::string>("type") == "bert-masked-lm")
    return New<BertMaskedLM>(options_);
  if(options_->get<std::string>("type") == "bert-classifier")
    return New<BertClassifier>(options_);
  ABORT("Unknown classifier type");
}


Ptr<ModelBase> EncoderDecoderFactory::construct() {
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

  return add_cost(encdec, options_);
}

Ptr<ModelBase> EncoderClassifierFactory::construct() {
  auto enccls = New<EncoderClassifier>(options_);

  for(auto& ef : encoders_)
    enccls->push_back(ef(options_).construct());

  for(auto& cf : classifiers_)
    enccls->push_back(cf(options_).construct());

  return add_cost(enccls, options_);
}

Ptr<ModelBase> by_type(std::string type, usage use, Ptr<Options> options) {
  // clang-format off
  if(type == "s2s" || type == "amun" || type == "nematus") {
    return models::encoder_decoder()(options)
        ("usage", use)
        ("original-type", type)
            .push_back(models::encoder()("type", "s2s"))
            .push_back(models::decoder()("type", "s2s"))
            .construct();
  }

  if(type == "transformer") {
    return models::encoder_decoder()(options)
        ("usage", use)
        .push_back(models::encoder()("type", "transformer"))
        .push_back(models::decoder()("type", "transformer"))
        .construct();
  }

  if(type == "transformer_s2s") {
    return models::encoder_decoder()(options)
        ("usage", use)
        ("original-type", type)
            .push_back(models::encoder()("type", "transformer"))
            .push_back(models::decoder()("type", "s2s"))
            .construct();
  }

  if(type == "lm") {
    auto idx = options->has("index") ? options->get<size_t>("index") : 0;
    std::vector<int> dimVocabs = options->get<std::vector<int>>("dim-vocabs");
    int vocab = dimVocabs[0];
    dimVocabs.resize(idx + 1);
    std::fill(dimVocabs.begin(), dimVocabs.end(), vocab);

    return models::encoder_decoder()(options)
        ("usage", use)
        ("type", "s2s")
        ("original-type", type)
            .push_back(models::decoder()
                       ("index", idx)
                       ("dim-vocabs", dimVocabs))
            .construct();
  }

  if(type == "multi-s2s") {
    size_t numEncoders = 2;
    auto ms2sFactory = models::encoder_decoder()(options)
        ("usage", use)
        ("type", "s2s")
        ("original-type", type);

    for(size_t i = 0; i < numEncoders; ++i) {
      auto prefix = "encoder" + std::to_string(i + 1);
      ms2sFactory.push_back(models::encoder()("prefix", prefix)("index", i));
    }

    ms2sFactory.push_back(models::decoder()("index", numEncoders));

    return ms2sFactory.construct();
  }

  if(type == "shared-multi-s2s") {
    size_t numEncoders = 2;
    auto ms2sFactory = models::encoder_decoder()(options)
        ("usage", use)
        ("type", "s2s")
        ("original-type", type);

    for(size_t i = 0; i < numEncoders; ++i) {
      auto prefix = "encoder";
      ms2sFactory.push_back(models::encoder()("prefix", prefix)("index", i));
    }

    ms2sFactory.push_back(models::decoder()("index", numEncoders));

    return ms2sFactory.construct();
  }

  if(type == "multi-transformer") {
    size_t numEncoders = 2;
    auto mtransFactory = models::encoder_decoder()(options)
        ("usage", use)
        ("type", "transformer")
        ("original-type", type);

    for(size_t i = 0; i < numEncoders; ++i) {
      auto prefix = "encoder" + std::to_string(i + 1);
      mtransFactory.push_back(models::encoder()("prefix", prefix)("index", i));
    }
    mtransFactory.push_back(models::decoder()("index", numEncoders));

    return mtransFactory.construct();
  }

  if(type == "shared-multi-transformer") {
    size_t numEncoders = 2;
    auto mtransFactory = models::encoder_decoder()(options)
        ("usage", use)
        ("type", "transformer")
        ("original-type", type);

    for(size_t i = 0; i < numEncoders; ++i) {
      auto prefix = "encoder";
      mtransFactory.push_back(models::encoder()("prefix", prefix)("index", i));
    }
    mtransFactory.push_back(models::decoder()("index", numEncoders));

    return mtransFactory.construct();
  }

  if(type == "lm-transformer") {
    auto idx = options->has("index") ? options->get<size_t>("index") : 0;
    std::vector<int> dimVocabs = options->get<std::vector<int>>("dim-vocabs");
    int vocab = dimVocabs[0];
    dimVocabs.resize(idx + 1);
    std::fill(dimVocabs.begin(), dimVocabs.end(), vocab);

    return models::encoder_decoder()(options)
        ("usage", use)
        ("type", "transformer")
        ("original-type", type)
            .push_back(models::decoder()
                       ("index", idx)
                       ("dim-vocabs", dimVocabs))
            .construct();
  }

  if(type == "bert") {
    return models::encoder_classifier()(options) //
        ("usage", use)                           //
        .push_back(models::encoder()             //
                    ("type", "transformer")      //
                    ("original-type", type)      //
                    ("index", 0))                // close to original transformer encoder
        .push_back(models::classifier()          //
                    ("type", "bert-masked-lm")   //
                    ("index", 1))                // multi-task learning with MaskedLM
        .push_back(models::classifier()          //
                    ("type", "bert-classifier")  //
                    ("bert-classes", 2)          //
                    ("index", 2))                // and next sentence prediction 
        .construct();
  }

#ifdef COMPILE_EXAMPLES
  // @TODO: examples should be compiled optionally
  if(type == "mnist-ffnn") {
    auto mnist = New<MnistFeedForwardNet>(options);
    if(use == usage::scoring)
      return New<Scorer>(mnist, New<MNISTLogsoftmax>());
    else if(use == usage::training)
      return New<Trainer>(mnist, New<MNISTCrossEntropyCost>());
    else
      return mnist;
  }
#endif

#ifdef CUDNN
#ifdef COMPILE_EXAMPLES
  if(type == "mnist-lenet") {
    auto mnist = New<MnistLeNet>(options);
    if(use == usage::scoring)
      return New<Scorer>(mnist, New<MNISTLogsoftmax>());
    else if(use == usage::training)
      return New<Trainer>(mnist, New<MNISTCrossEntropyCost>());
    else
      return mnist;
  }
#endif
  if(type == "char-s2s") {
    return models::encoder_decoder()(options)
        ("usage", use)
        ("original-type", type)
            .push_back(models::encoder()("type", "char-s2s"))
            .push_back(models::decoder()("type", "s2s"))
            .construct();
  }
#endif

  // clang-format on
  ABORT("Unknown model type: {}", type);
}

Ptr<ModelBase> from_options(Ptr<Options> options, usage use) {
  std::string type = options->get<std::string>("type");
  return by_type(type, use, options);
}

}  // namespace models
}  // namespace marian
