#pragma once

#include "marian.h"

#include "layers/factory.h"
#include "models/encoder_decoder.h"
#include "models/encoder_classifier.h"

namespace marian {
namespace models {

class EncoderFactory : public Factory {
public:
  EncoderFactory(Ptr<ExpressionGraph> graph = nullptr) : Factory() {}

  virtual Ptr<EncoderBase> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<EncoderFactory> encoder;

class DecoderFactory : public Factory {
public:
  DecoderFactory(Ptr<ExpressionGraph> graph = nullptr) : Factory() {}

  virtual Ptr<DecoderBase> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<DecoderFactory> decoder;

class ClassifierFactory : public Factory {
public:
  ClassifierFactory(Ptr<ExpressionGraph> graph = nullptr) 
     : Factory() {}

  virtual Ptr<ClassifierBase> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<ClassifierFactory> classifier;

class EncoderDecoderFactory : public Factory {
private:
  std::vector<encoder> encoders_;
  std::vector<decoder> decoders_;

public:
  EncoderDecoderFactory(Ptr<ExpressionGraph> graph = nullptr)
      : Factory() {}

  Accumulator<EncoderDecoderFactory> push_back(encoder enc) {
    encoders_.push_back(enc);
    return Accumulator<EncoderDecoderFactory>(*this);
  }

  Accumulator<EncoderDecoderFactory> push_back(decoder dec) {
    decoders_.push_back(dec);
    return Accumulator<EncoderDecoderFactory>(*this);
  }

  virtual Ptr<ModelBase> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<EncoderDecoderFactory> encoder_decoder;

class EncoderClassifierFactory : public Factory {
private:
  std::vector<encoder> encoders_;
  std::vector<classifier> classifiers_;

public:
  EncoderClassifierFactory(Ptr<ExpressionGraph> graph = nullptr)
      : Factory() {}

  Accumulator<EncoderClassifierFactory> push_back(encoder enc) {
    encoders_.push_back(enc);
    return Accumulator<EncoderClassifierFactory>(*this);
  }

  Accumulator<EncoderClassifierFactory> push_back(classifier cls) {
    classifiers_.push_back(cls);
    return Accumulator<EncoderClassifierFactory>(*this);
  }

  virtual Ptr<ModelBase> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<EncoderClassifierFactory> encoder_classifier;

Ptr<ModelBase> by_type(std::string type, usage, Ptr<Options> options);

Ptr<ModelBase> from_options(Ptr<Options> options, usage);
}  // namespace models
}  // namespace marian
