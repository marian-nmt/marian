#pragma once
#include <memory>
#include <vector>
#include <string>
#include <tuple>

namespace marian {

template <typename T> using Ptr = std::shared_ptr<T>;

class Options;

namespace quicksand {

typedef std::vector<std::string> Sentence;
typedef std::tuple<Sentence, float> SentenceWithProb;
typedef std::vector<SentenceWithProb> NBest;

Ptr<Options> newOptions();

template <class T>
void set(Ptr<Options> options, const std::string& key, T value);

class BeamSearchDecoderBase {
  protected:
    Ptr<Options> options_;

  public:
    BeamSearchDecoderBase(Ptr<Options> options)
    : options_(options) {}

    virtual NBest decode(const Sentence& sentence) = 0;
};

Ptr<BeamSearchDecoderBase> newDecoder(Ptr<Options> options);

}
}


