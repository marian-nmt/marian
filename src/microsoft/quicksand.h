#pragma once
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>
#include <set>

namespace marian {

template <typename T>
using Ptr = std::shared_ptr<T>;

class Options;

namespace quicksand {

typedef uint32_t IndexType;
typedef IndexType WordIndex;
typedef std::vector<WordIndex> WordIndices;
typedef std::vector<WordIndices> QSBatch;
typedef std::vector<std::set<std::pair<size_t, float>>> AlignmentSets; // [trgPos] -> set of (srcPos, P(srcPos|trgPos))

typedef std::tuple<WordIndices, AlignmentSets, float> QSSentenceWithProb;
typedef std::vector<QSSentenceWithProb> QSNBest;
typedef std::vector<QSNBest> QSNBestBatch;

enum class DecoderCpuAvxVersion {
  AVX,
  AVX2,
  AVX512
};

Ptr<Options> newOptions();

template <class T>
void set(Ptr<Options> options, const std::string& key, const T& value);

class IVocabWrapper {
public:
  virtual WordIndex encode(const std::string& word) const = 0;
  virtual std::string decode(WordIndex id) const = 0;
  virtual size_t size() const = 0;
  virtual void transcodeToShortlistInPlace(WordIndex* ptr, size_t num) const = 0;
};

class IBeamSearchDecoder {
protected:
  Ptr<Options> options_;
  std::vector<const void*> ptrs_;

public:
  IBeamSearchDecoder(Ptr<Options> options,
                     const std::vector<const void*>& ptrs)
      : options_(options), ptrs_(ptrs) {}

  virtual ~IBeamSearchDecoder() {}

  virtual QSNBestBatch decode(const QSBatch& qsBatch,
                              size_t maxLength,
                              const std::unordered_set<WordIndex>& shortlist)
      = 0;

  virtual void setWorkspace(uint8_t* data, size_t size) = 0;
};

Ptr<IBeamSearchDecoder> newDecoder(Ptr<Options> options,
                                   const std::vector<const void*>& ptrs,
                                   const std::vector<Ptr<IVocabWrapper>>& vocabs,
                                   WordIndex eos/*dummy --@TODO: remove*/);

// load src and tgt vocabs
std::vector<Ptr<IVocabWrapper>> loadVocabs(const std::vector<std::string>& vocabPaths);

// query CPU AVX version
DecoderCpuAvxVersion getCpuAvxVersion();
DecoderCpuAvxVersion parseCpuAvxVersion(std::string name);

// MJD: added "addLsh" which will now break whatever compilation after update. That's on purpose.
// The calling code should be adapted, not this interface. If you need to fix things in QS because of this
// talk to me first!
bool convertModel(std::string inputFile, std::string outputFile, int32_t targetPrec, int32_t lshNBits);

}  // namespace quicksand
}  // namespace marian
