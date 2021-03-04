#pragma once

#include "marian.h"

#include "layers/factory.h"
#include "models/encoder_decoder.h"
#include "models/encoder_classifier.h"
#include "models/encoder_pooler.h"

namespace marian {
namespace models {

class EncoderFactory : public Factory {
  using Factory::Factory;
public:
  virtual Ptr<EncoderBase> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<EncoderFactory> encoder;

class DecoderFactory : public Factory {
  using Factory::Factory;
public:
  virtual Ptr<DecoderBase> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<DecoderFactory> decoder;

class ClassifierFactory : public Factory {
  using Factory::Factory;
public:
  virtual Ptr<ClassifierBase> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<ClassifierFactory> classifier;

class PoolerFactory : public Factory {
  using Factory::Factory;
public:
  virtual Ptr<PoolerBase> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<PoolerFactory> pooler;

class EncoderDecoderFactory : public Factory {
  using Factory::Factory;
private:
  std::vector<encoder> encoders_;
  std::vector<decoder> decoders_;

public:
  Accumulator<EncoderDecoderFactory> push_back(encoder enc) {
    encoders_.push_back(enc);
    return Accumulator<EncoderDecoderFactory>(*this);
  }

  Accumulator<EncoderDecoderFactory> push_back(decoder dec) {
    decoders_.push_back(dec);
    return Accumulator<EncoderDecoderFactory>(*this);
  }

  virtual Ptr<IModel> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<EncoderDecoderFactory> encoder_decoder;

class EncoderClassifierFactory : public Factory {
  using Factory::Factory;
private:
  std::vector<encoder> encoders_;
  std::vector<classifier> classifiers_;

public:
  Accumulator<EncoderClassifierFactory> push_back(encoder enc) {
    encoders_.push_back(enc);
    return Accumulator<EncoderClassifierFactory>(*this);
  }

  Accumulator<EncoderClassifierFactory> push_back(classifier cls) {
    classifiers_.push_back(cls);
    return Accumulator<EncoderClassifierFactory>(*this);
  }

  virtual Ptr<IModel> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<EncoderClassifierFactory> encoder_classifier;

class EncoderPoolerFactory : public Factory {
  using Factory::Factory;
private:
  std::vector<encoder> encoders_;
  std::vector<pooler> poolers_;

public:
  Accumulator<EncoderPoolerFactory> push_back(encoder enc) {
    encoders_.push_back(enc);
    return Accumulator<EncoderPoolerFactory>(*this);
  }

  Accumulator<EncoderPoolerFactory> push_back(pooler cls) {
    poolers_.push_back(cls);
    return Accumulator<EncoderPoolerFactory>(*this);
  }

  virtual Ptr<IModel> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<EncoderPoolerFactory> encoder_pooler;

Ptr<IModel> createBaseModelByType(std::string type, usage, Ptr<Options> options);

Ptr<IModel> createModelFromOptions(Ptr<Options> options, usage);

Ptr<ICriterionFunction> createCriterionFunctionFromOptions(Ptr<Options> options, usage);
}  // namespace models
}  // namespace marian
