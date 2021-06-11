#include "data/shortlist.h"
#include "microsoft/shortlist/utils/ParameterTree.h"
#include "marian.h"

#if BLAS_FOUND
#include "3rd_party/faiss/IndexLSH.h"
#endif

namespace marian {
namespace data {

// cast current void pointer to T pointer and move forward by num elements 
template <typename T>
const T* get(const void*& current, size_t num = 1) {
  const T* ptr = (const T*)current;
  current = (const T*)current + num;
  return ptr;
}

//////////////////////////////////////////////////////////////////////////////////////
Shortlist::Shortlist(const std::vector<WordIndex>& indices)
  : indices_(indices)
  , done_(false) {}

Shortlist::~Shortlist() {}

WordIndex Shortlist::reverseMap(int , int , int idx) const { return indices_[idx]; }

WordIndex Shortlist::tryForwardMap(int , int , WordIndex wIdx) const {
  auto first = std::lower_bound(indices_.begin(), indices_.end(), wIdx);
  if(first != indices_.end() && *first == wIdx)         // check if element not less than wIdx has been found and if equal to wIdx
    return (int)std::distance(indices_.begin(), first); // return coordinate if found
  else
    return npos;                                        // return npos if not found, @TODO: replace with std::optional once we switch to C++17?
}

void Shortlist::filter(Expr input, Expr weights, bool isLegacyUntransposedW, Expr b, Expr lemmaEt) {
  if (done_) {
    return;
  }

  auto forward = [this](Expr out, const std::vector<Expr>& ) {
    out->val()->set(indices_);
  };

  int k = (int) indices_.size();
  Shape kShape({k});
  indicesExpr_ = lambda({input, weights}, kShape, Type::uint32, forward);

  //std::cerr << "indicesExpr_=" << indicesExpr_->shape() << std::endl;
  broadcast(weights, isLegacyUntransposedW, b, lemmaEt, k);
  done_ = true;
}

Expr Shortlist::getIndicesExpr(int batchSize, int beamSize) const {
  int k = indicesExpr_->shape()[0];
  Expr ones = indicesExpr_->graph()->constant({batchSize, beamSize, 1}, inits::ones(), Type::float32);

  Expr tmp = reshape(indicesExpr_, {1, k});
  tmp = cast(tmp, Type::float32);

  Expr out = ones * tmp;
  //debug(out, "out.1");

  auto forward = [](Expr out, const std::vector<Expr>& inputs) {
    Expr in = inputs[0];
    const Shape &shape = in->shape();
    const float *inPtr = in->val()->data();
    uint32_t *outPtr = out->val()->data<uint32_t>();

    for (int i = 0; i < shape.elements(); ++i) {
        const float &val = inPtr[i];
        uint32_t valConv = (uint32_t)val;
        uint32_t &valOut = outPtr[i];
        valOut = valConv;
        //std::cerr << val << " " << valConv << " " << valOut << std::endl;
    }
  };
  out = lambda({out}, out->shape(), Type::uint32, forward);
  //debug(out, "out.2");
  //out = cast(out, Type::uint32);
  //std::cerr << "getIndicesExpr.2=" << out->shape() << std::endl;
  //out = reshape(out, {k});

  return out;
}

void Shortlist::broadcast(Expr weights,
                          bool isLegacyUntransposedW,
                          Expr b,
                          Expr lemmaEt,
                          int k) {
  //std::cerr << "isLegacyUntransposedW=" << isLegacyUntransposedW << std::endl;
  ABORT_IF(isLegacyUntransposedW, "Legacy untranspose W not yet tested");

  //std::cerr << "currBeamSize=" << currBeamSize << " batchSize=" << batchSize << std::endl;
  //std::cerr << "weights=" << weights->shape() << std::endl;
  cachedShortWt_ = index_select(weights, isLegacyUntransposedW ? -1 : 0, indicesExpr_);
  //std::cerr << "cachedShortWt_.1=" << cachedShortWt_->shape() << std::endl;
  cachedShortWt_ = reshape(cachedShortWt_, {1, 1, cachedShortWt_->shape()[0], cachedShortWt_->shape()[1]});

  if (b) {
    ABORT("Bias not yet tested");
    cachedShortb_ = index_select(b, -1, indicesExpr_);
    cachedShortb_ = reshape(cachedShortb_, {1, k, 1, cachedShortb_->shape()[1]}); // not tested
  }

  //std::cerr << "lemmaEt.1_=" << lemmaEt->shape() << std::endl;
  cachedShortLemmaEt_ = index_select(lemmaEt, -1, indicesExpr_);
  //std::cerr << "cachedShortLemmaEt.1_=" << cachedShortLemmaEt_->shape() << std::endl;
  cachedShortLemmaEt_ = reshape(cachedShortLemmaEt_, {1, 1, cachedShortLemmaEt_->shape()[0], k});
  //std::cerr << "cachedShortLemmaEt.2_=" << cachedShortLemmaEt_->shape() << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////
Ptr<faiss::IndexLSH> LSHShortlist::index_;

LSHShortlist::LSHShortlist(int k, int nbits)
: Shortlist(std::vector<WordIndex>()) 
, k_(k), nbits_(nbits) {
  //std::cerr << "LSHShortlist" << std::endl;
  /*
  for (int i = 0; i < k_; ++i) {
    indices_.push_back(i);
  }
  */
}

//#define BLAS_FOUND 1

WordIndex LSHShortlist::reverseMap(int batchIdx, int beamIdx, int idx) const {
  //int currBeamSize = indicesExpr_->shape()[0];
  int currBatchSize = indicesExpr_->shape()[1];
  idx = (k_ * currBatchSize * beamIdx) + (k_ * batchIdx) + idx;
  assert(idx < indices_.size());
  return indices_[idx]; 
}

WordIndex LSHShortlist::tryForwardMap(int , int , WordIndex wIdx) const {
  //utils::Debug(indices_, "LSHShortlist::tryForwardMap indices_");
  auto first = std::lower_bound(indices_.begin(), indices_.end(), wIdx);
  bool found = first != indices_.end();
  if(found && *first == wIdx)         // check if element not less than wIdx has been found and if equal to wIdx
    return (int)std::distance(indices_.begin(), first); // return coordinate if found
  else
    return npos;                                        // return npos if not found, @TODO: replace with std::optional once we switch to C++17?
}

Expr LSHShortlist::getIndicesExpr(int batchSize, int currBeamSize) const {
  assert(indicesExpr_->shape()[0] == currBeamSize);
  assert(indicesExpr_->shape()[1] == batchSize);
  Expr ret = transpose(indicesExpr_, {1, 0, 2});
  return ret;
}

#define BLAS_FOUND 1

void LSHShortlist::filter(Expr input, Expr weights, bool isLegacyUntransposedW, Expr b, Expr lemmaEt) {
#if BLAS_FOUND
  ABORT_IF(input->graph()->getDeviceId().type == DeviceType::gpu,
           "LSH index (--output-approx-knn) currently not implemented for GPU");

  int currBeamSize = input->shape()[0];
  int batchSize = input->shape()[2];
  int numHypos = currBeamSize * batchSize;

  auto forward = [this, numHypos](Expr out, const std::vector<Expr>& inputs) {
    auto query  = inputs[0];
    auto values = inputs[1];
    int dim = values->shape()[-1];

    if(!index_) {
      //std::cerr << "build lsh index" << std::endl;
      LOG(info, "Building LSH index for vector dim {} and with hash size {} bits", dim, nbits_);
      index_.reset(new faiss::IndexLSH(dim, nbits_, 
                                       /*rotate=*/dim != nbits_, 
                                       /*train_thesholds*/false));
      int vRows = 32121; //47960; //values->shape().elements() / dim;
      index_->train(vRows, values->val()->data<float>());
      index_->add(  vRows, values->val()->data<float>());
    }

    int qRows = query->shape().elements() / dim;
    std::vector<float> distances(qRows * k_);
    std::vector<faiss::Index::idx_t> ids(qRows * k_);

    index_->search(qRows, query->val()->data<float>(), k_,
                   distances.data(), ids.data());
    
    indices_.clear();
    for(auto iter = ids.begin(); iter != ids.end(); ++iter) {
      faiss::Index::idx_t id = *iter;
      indices_.push_back((WordIndex)id);
    }

    for (size_t hypoIdx = 0; hypoIdx < numHypos; ++hypoIdx) {
      size_t startIdx = k_ * hypoIdx;
      size_t endIdx = startIdx + k_;
      std::sort(indices_.begin() + startIdx, indices_.begin() + endIdx);
    }
    out->val()->set(indices_);
  };

  Shape kShape({currBeamSize, batchSize, k_});

  indicesExpr_ = lambda({input, weights}, kShape, Type::uint32, forward);
  //std::cerr << "indicesExpr_=" << indicesExpr_->shape() << std::endl;

  broadcast(weights, isLegacyUntransposedW, b, lemmaEt, indicesExpr_, k_);

#else
  input; weights; isLegacyUntransposedW; b; lemmaEt;
  ABORT("LSH output layer requires a CPU BLAS library");
#endif
}

void LSHShortlist::broadcast(Expr weights,
                          bool isLegacyUntransposedW,
                          Expr b,
                          Expr lemmaEt,
                          Expr indicesExprBC,
                          int k) {
  int currBeamSize = indicesExprBC->shape()[0];
  int batchSize = indicesExprBC->shape()[1];
  //int numHypos = batchSize * currBeamSize;
  //std::cerr << "batchSize=" << batchSize << std::endl;
  //std::cerr << "currBeamSize=" << currBeamSize << std::endl;
  //std::cerr << "isLegacyUntransposedW=" << isLegacyUntransposedW << std::endl;
  ABORT_IF(isLegacyUntransposedW, "Legacy untranspose W not yet tested");

  indicesExprBC = reshape(indicesExprBC, {indicesExprBC->shape().elements()});
  //std::cerr << "indicesExprBC.2=" << indicesExprBC->shape() << std::endl;

  cachedShortWt_ = index_select(weights, isLegacyUntransposedW ? -1 : 0, indicesExprBC);
  cachedShortWt_ = reshape(cachedShortWt_, {currBeamSize, batchSize, k, cachedShortWt_->shape()[1]});

  if (b) {
    ABORT("Bias not yet tested");
    cachedShortb_ = index_select(b, -1, indicesExprBC);
    cachedShortb_ = reshape(cachedShortb_, {currBeamSize, k, batchSize, cachedShortb_->shape()[1]}); // not tested
  }

  cachedShortLemmaEt_ = index_select(lemmaEt, -1, indicesExprBC);
  cachedShortLemmaEt_ = reshape(cachedShortLemmaEt_, {cachedShortLemmaEt_->shape()[0], batchSize, currBeamSize, k});
  cachedShortLemmaEt_ = transpose(cachedShortLemmaEt_, {2, 1, 0, 3});
}

LSHShortlistGenerator::LSHShortlistGenerator(int k, int nbits) 
  : k_(k), nbits_(nbits) {
  //std::cerr << "LSHShortlistGenerator" << std::endl;
}

Ptr<Shortlist> LSHShortlistGenerator::generate(Ptr<data::CorpusBatch> batch) const {
  return New<LSHShortlist>(k_, nbits_);
}

//////////////////////////////////////////////////////////////////////////////////////
QuicksandShortlistGenerator::QuicksandShortlistGenerator(Ptr<Options> options,
                                                         Ptr<const Vocab> srcVocab,
                                                         Ptr<const Vocab> trgVocab,
                                                         size_t srcIdx,
                                                         size_t /*trgIdx*/,
                                                         bool /*shared*/)
    : options_(options),
      srcVocab_(srcVocab),
      trgVocab_(trgVocab),
      srcIdx_(srcIdx) {
  std::vector<std::string> vals = options_->get<std::vector<std::string>>("shortlist");

  ABORT_IF(vals.empty(), "No path to filter path given");
  std::string fname = vals[0];

  auto firstNum   = vals.size() > 1 ? std::stoi(vals[1]) : 0;
  auto bestNum    = vals.size() > 2 ? std::stoi(vals[2]) : 0;
  float threshold = vals.size() > 3 ? std::stof(vals[3]) : 0;

  if(firstNum != 0 || bestNum != 0 || threshold != 0) {
    LOG(warn, "You have provided additional parameters for the Quicksand shortlist, but they are ignored.");
  }

  mmap_ = mio::mmap_source(fname); // memory-map the binary file once
  const void* current = mmap_.data(); // pointer iterator over binary file
  
  // compare magic number in binary file to make sure we are reading the right thing
  const int32_t MAGIC_NUMBER = 1234567890;
  int32_t header_magic_number = *get<int32_t>(current);
  ABORT_IF(header_magic_number != MAGIC_NUMBER, "Trying to mmap Quicksand shortlist but encountered wrong magic number");

  auto config = ::quicksand::ParameterTree::FromBinaryReader(current);
  use16bit_ = config->GetBoolReq("use_16_bit");
  
  LOG(info, "[data] Mapping Quicksand shortlist from {}", fname);

  idSize_ = sizeof(int32_t);
  if (use16bit_) {
    idSize_ = sizeof(uint16_t);
  }

  // mmap the binary shortlist pieces
  numDefaultIds_        = *get<int32_t>(current);
  defaultIds_           =  get<int32_t>(current, numDefaultIds_);
  numSourceIds_         = *get<int32_t>(current);
  sourceLengths_        =  get<int32_t>(current, numSourceIds_);
  sourceOffsets_        =  get<int32_t>(current, numSourceIds_);
  numShortlistIds_      = *get<int32_t>(current);
  sourceToShortlistIds_ =  get<uint8_t>(current, idSize_ * numShortlistIds_);
  
  // display parameters
  LOG(info, 
      "[data] Quicksand shortlist has {} source ids, {} default ids and {} shortlist ids",
      numSourceIds_, 
      numDefaultIds_, 
      numShortlistIds_);
}

Ptr<Shortlist> QuicksandShortlistGenerator::generate(Ptr<data::CorpusBatch> batch) const {
  auto srcBatch = (*batch)[srcIdx_];
  auto maxShortlistSize = trgVocab_->size();

  std::unordered_set<int32_t> indexSet;
  for(int32_t i = 0; i < numDefaultIds_ && i < maxShortlistSize; ++i) {
    int32_t id = defaultIds_[i];
    indexSet.insert(id);
  }

  // State
  std::vector<std::pair<const uint8_t*, int32_t>> curShortlists(maxShortlistSize);
  auto curShortlistIt = curShortlists.begin();

  // Because we might fill up our shortlist before reaching max_shortlist_size, we fill the shortlist in order of rank.
  // E.g., first rank of word 0, first rank of word 1, ... second rank of word 0, ...
  int32_t maxLength = 0;
  for (Word word : srcBatch->data()) {
    int32_t sourceId = (int32_t)word.toWordIndex();
    srcVocab_->transcodeToShortlistInPlace((WordIndex*)&sourceId, 1);

    if (sourceId < numSourceIds_) { // if it's a valid source id
      const uint8_t* curShortlistIds = sourceToShortlistIds_ + idSize_ * sourceOffsets_[sourceId]; // start position for mapping
      int32_t length = sourceLengths_[sourceId]; // how many mappings are there
      curShortlistIt->first  = curShortlistIds;
      curShortlistIt->second = length;
      curShortlistIt++;
      
      if (length > maxLength)
        maxLength = length;
    }
  }
        
  // collect the actual shortlist mappings
  for (int32_t i = 0; i < maxLength && indexSet.size() < maxShortlistSize; i++) {
    for (int32_t j = 0; j < curShortlists.size() && indexSet.size() < maxShortlistSize; j++) {
      int32_t length = curShortlists[j].second;
      if (i < length) {
        const uint8_t* source_shortlist_ids_bytes = curShortlists[j].first;
        int32_t id = 0;
        if (use16bit_) {
          const uint16_t* source_shortlist_ids = reinterpret_cast<const uint16_t*>(source_shortlist_ids_bytes);
          id = (int32_t)source_shortlist_ids[i];
        }
        else {
          const int32_t* source_shortlist_ids = reinterpret_cast<const int32_t*>(source_shortlist_ids_bytes);
          id = source_shortlist_ids[i];
        }
        indexSet.insert(id);
      }
    }
  }

  // turn into vector and sort (selected indices)
  std::vector<WordIndex> indices;
  indices.reserve(indexSet.size());
  for(auto i : indexSet)
    indices.push_back((WordIndex)i);

  std::sort(indices.begin(), indices.end());
  return New<Shortlist>(indices);
}

Ptr<ShortlistGenerator> createShortlistGenerator(Ptr<Options> options,
                                                 Ptr<const Vocab> srcVocab,
                                                 Ptr<const Vocab> trgVocab,
                                                 const std::vector<int> &lshOpts,
                                                 size_t srcIdx,
                                                 size_t trgIdx,
                                                 bool shared) {
  if (lshOpts.size() == 2) {
    return New<LSHShortlistGenerator>(lshOpts[0], lshOpts[1]);
  }
  else {                                                   
    std::vector<std::string> vals = options->get<std::vector<std::string>>("shortlist");
    ABORT_IF(vals.empty(), "No path to shortlist given");
    std::string fname = vals[0];
    if(filesystem::Path(fname).extension().string() == ".bin") {
      return New<QuicksandShortlistGenerator>(options, srcVocab, trgVocab, srcIdx, trgIdx, shared);
    } else {
      return New<LexicalShortlistGenerator>(options, srcVocab, trgVocab, srcIdx, trgIdx, shared);
    }
  }
}

}  // namespace data
}  // namespace marian
