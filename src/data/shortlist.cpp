#include "data/shortlist.h"
#include "microsoft/shortlist/utils/ParameterTree.h"
#include "marian.h"
#include "layers/lsh.h"

#include <queue>

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
  : indices_(indices), 
    initialized_(false) {}

Shortlist::~Shortlist() {}

WordIndex Shortlist::reverseMap(int /*beamIdx*/, int /*batchIdx*/, int idx) const { return indices_[idx]; }

WordIndex Shortlist::tryForwardMap(WordIndex wIdx) const {
  auto first = std::lower_bound(indices_.begin(), indices_.end(), wIdx);
  if(first != indices_.end() && *first == wIdx)         // check if element not less than wIdx has been found and if equal to wIdx
    return (int)std::distance(indices_.begin(), first); // return coordinate if found
  else
    return npos;                                        // return npos if not found, @TODO: replace with std::optional once we switch to C++17?
}

void Shortlist::filter(Expr input, Expr weights, bool isLegacyUntransposedW, Expr b, Expr lemmaEt) {
  if (initialized_) {
    return;
  }

  auto forward = [this](Expr out, const std::vector<Expr>& ) {
    out->val()->set(indices_);
  };

  int k = (int) indices_.size();
  Shape kShape({k});
  indicesExpr_ = lambda({input, weights}, kShape, Type::uint32, forward);

  createCachedTensors(weights, isLegacyUntransposedW, b, lemmaEt, k);
  initialized_ = true;
}

Expr Shortlist::getIndicesExpr() const {
  int k = indicesExpr_->shape()[0];
  Expr out = reshape(indicesExpr_, {1, 1, k});
  return out;
}

void Shortlist::createCachedTensors(Expr weights,
                          bool isLegacyUntransposedW,
                          Expr b,
                          Expr lemmaEt,
                          int k) {
  ABORT_IF(isLegacyUntransposedW, "Legacy untranspose W not yet tested");
  cachedShortWt_ = index_select(weights, isLegacyUntransposedW ? -1 : 0, indicesExpr_);
  cachedShortWt_ = reshape(cachedShortWt_, {1, 1, cachedShortWt_->shape()[0], cachedShortWt_->shape()[1]});

  if (b) {
    cachedShortb_ = index_select(b, -1, indicesExpr_);
  }

  if (lemmaEt) {
    cachedShortLemmaEt_ = index_select(lemmaEt, -1, indicesExpr_);
    cachedShortLemmaEt_ = reshape(cachedShortLemmaEt_, {1, 1, cachedShortLemmaEt_->shape()[0], k});
  }
}

///////////////////////////////////////////////////////////////////////////////////

LSHShortlist::LSHShortlist(int k, int nbits, size_t lemmaSize, bool abortIfDynamic)
: Shortlist(std::vector<WordIndex>()), 
  k_(k), nbits_(nbits), lemmaSize_(lemmaSize), abortIfDynamic_(abortIfDynamic) {
}

WordIndex LSHShortlist::reverseMap(int beamIdx, int batchIdx, int idx) const {
  //int currBeamSize = indicesExpr_->shape()[0];
  int currBatchSize = indicesExpr_->shape()[1];
  idx = (k_ * currBatchSize * beamIdx) + (k_ * batchIdx) + idx;
  assert(idx < indices_.size());
  return indices_[idx]; 
}

Expr LSHShortlist::getIndicesExpr() const {
  return indicesExpr_;
}

void LSHShortlist::filter(Expr input, Expr weights, bool isLegacyUntransposedW, Expr b, Expr lemmaEt) {

  ABORT_IF(input->graph()->getDeviceId().type == DeviceType::gpu,
           "LSH index (--output-approx-knn) currently not implemented for GPU");

  indicesExpr_ = callback(lsh::search(input, weights, k_, nbits_, (int)lemmaSize_, abortIfDynamic_),
                          [this](Expr node) { 
                            node->val()->get(indices_); // set the value of the field indices_ whenever the graph traverses this node
                          });

  createCachedTensors(weights, isLegacyUntransposedW, b, lemmaEt, k_);
}

void LSHShortlist::createCachedTensors(Expr weights,
                                       bool isLegacyUntransposedW,
                                       Expr b,
                                       Expr lemmaEt,
                                       int k) {
  int currBeamSize = indicesExpr_->shape()[0];
  int batchSize = indicesExpr_->shape()[1];
  ABORT_IF(isLegacyUntransposedW, "Legacy untranspose W not yet tested");

  Expr indicesExprFlatten = reshape(indicesExpr_, {indicesExpr_->shape().elements()});

  cachedShortWt_ = index_select(weights, isLegacyUntransposedW ? -1 : 0, indicesExprFlatten);
  cachedShortWt_ = reshape(cachedShortWt_, {currBeamSize, batchSize, k, cachedShortWt_->shape()[1]});

  if (b) {
    ABORT("Bias not supported with LSH");
    cachedShortb_ = index_select(b, -1, indicesExprFlatten);
    cachedShortb_ = reshape(cachedShortb_, {currBeamSize, batchSize, k, cachedShortb_->shape()[0]}); // not tested
  }

  if (lemmaEt) {
    int dim = lemmaEt->shape()[0];
    cachedShortLemmaEt_ = index_select(lemmaEt, -1, indicesExprFlatten);
    cachedShortLemmaEt_ = reshape(cachedShortLemmaEt_, {dim, currBeamSize, batchSize, k});
    cachedShortLemmaEt_ = transpose(cachedShortLemmaEt_, {1, 2, 0, 3});
  }
}

LSHShortlistGenerator::LSHShortlistGenerator(int k, int nbits, size_t lemmaSize, bool abortIfDynamic) 
  : k_(k), nbits_(nbits), lemmaSize_(lemmaSize), abortIfDynamic_(abortIfDynamic) {
}

Ptr<Shortlist> LSHShortlistGenerator::generate(Ptr<data::CorpusBatch> batch) const {
  return New<LSHShortlist>(k_, nbits_, lemmaSize_, abortIfDynamic_);
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

  auto config = marian::quicksand::ParameterTree::FromBinaryReader(current);
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
  if (lshOpts.size()) {
    assert(lshOpts.size() == 2);
    size_t lemmaSize = trgVocab->lemmaSize();
    return New<LSHShortlistGenerator>(lshOpts[0], lshOpts[1], lemmaSize, /*abortIfDynamic=*/false);
  }
  else {                                                   
    std::vector<std::string> vals = options->get<std::vector<std::string>>("shortlist");
    ABORT_IF(vals.empty(), "No path to shortlist given");
    std::string fname = vals[0];
    if(isBinaryShortlist(fname)){
        return New<BinaryShortlistGenerator>(options, srcVocab, trgVocab, srcIdx, trgIdx, shared);
    } else if(filesystem::Path(fname).extension().string() == ".bin") {
      return New<QuicksandShortlistGenerator>(options, srcVocab, trgVocab, srcIdx, trgIdx, shared);
    } else {
      return New<LexicalShortlistGenerator>(options, srcVocab, trgVocab, srcIdx, trgIdx, shared);
    }
  }
}

bool isBinaryShortlist(const std::string& fileName){
  uint64_t magic;
  io::InputFileStream in(fileName);
  in.read((char*)(&magic), sizeof(magic));
  return in && (magic == BINARY_SHORTLIST_MAGIC);
}

void BinaryShortlistGenerator::contentCheck() {
  bool failFlag = 0;
  // The offset table has to be within the size of shortlists.
  for(int i = 0; i < wordToOffsetSize_-1; i++)
    failFlag |= wordToOffset_[i] >= shortListsSize_;

  // The last element of wordToOffset_ must equal shortListsSize_
  failFlag |= wordToOffset_[wordToOffsetSize_-1] != shortListsSize_;

  // The vocabulary indices have to be within the vocabulary size.
  size_t vSize = trgVocab_->size();
  for(int j = 0; j < shortListsSize_; j++)
    failFlag |= shortLists_[j] >= vSize;
  ABORT_IF(failFlag, "Error: shortlist indices are out of bounds");
}

// load shortlist from buffer
void BinaryShortlistGenerator::load(const void* ptr_void, size_t blobSize, bool check /*= true*/) {
  /* File layout:
   * header
   * wordToOffset array
   * shortLists array
   */
  ABORT_IF(blobSize < sizeof(Header), "Shortlist length {} too short to have a header", blobSize);

  const char *ptr = static_cast<const char*>(ptr_void);
  const Header &header = *reinterpret_cast<const Header*>(ptr);
  ptr += sizeof(Header);
  ABORT_IF(header.magic != BINARY_SHORTLIST_MAGIC, "Incorrect magic in binary shortlist");

  uint64_t expectedSize = sizeof(Header) + header.wordToOffsetSize * sizeof(uint64_t) + header.shortListsSize * sizeof(WordIndex);
  ABORT_IF(expectedSize != blobSize, "Shortlist header claims file size should be {} but file is {}", expectedSize, blobSize);

  if (check) {
    uint64_t checksumActual = util::hashMem<uint64_t, uint64_t>(&header.firstNum, (blobSize - sizeof(header.magic) - sizeof(header.checksum)) / sizeof(uint64_t));
    ABORT_IF(checksumActual != header.checksum, "checksum check failed: this binary shortlist is corrupted");
  }

  firstNum_ = header.firstNum;
  bestNum_ = header.bestNum;
  LOG(info, "[data] Lexical short list firstNum {} and bestNum {}", firstNum_, bestNum_);

  wordToOffsetSize_ = header.wordToOffsetSize;
  shortListsSize_ = header.shortListsSize;

  // Offsets right after header.
  wordToOffset_ = reinterpret_cast<const uint64_t*>(ptr);
  ptr += wordToOffsetSize_ * sizeof(uint64_t);

  shortLists_ = reinterpret_cast<const WordIndex*>(ptr);

  // Verify offsets and vocab ids are within bounds if requested by user.
  if(check)
    contentCheck();
}

// load shortlist from file
void BinaryShortlistGenerator::load(const std::string& filename, bool check /*=true*/) {
  std::error_code error;
  mmapMem_.map(filename, error);
  ABORT_IF(error, "Error mapping file: {}", error.message());
  load(mmapMem_.data(), mmapMem_.mapped_length(), check);
}

BinaryShortlistGenerator::BinaryShortlistGenerator(Ptr<Options> options,
                                                   Ptr<const Vocab> srcVocab,
                                                   Ptr<const Vocab> trgVocab,
                                                   size_t srcIdx /*= 0*/,
                                                   size_t /*trgIdx = 1*/,
                                                   bool shared /*= false*/)
    : options_(options),
      srcVocab_(srcVocab),
      trgVocab_(trgVocab),
      srcIdx_(srcIdx),
      shared_(shared) {

  std::vector<std::string> vals = options_->get<std::vector<std::string>>("shortlist");
  ABORT_IF(vals.empty(), "No path to shortlist file given");
  std::string fname = vals[0];

  if(isBinaryShortlist(fname)){
    bool check = vals.size() > 1 ? std::stoi(vals[1]) : 1;
    LOG(info, "[data] Loading binary shortlist as {} {}", fname, check);
    load(fname, check);
  }
  else{
    firstNum_ = vals.size() > 1 ? std::stoi(vals[1]) : 100;
    bestNum_ = vals.size() > 2 ? std::stoi(vals[2]) : 100;
    float threshold = vals.size() > 3 ? std::stof(vals[3]) : 0;
    LOG(info, "[data] Importing text lexical shortlist as {} {} {} {}",
        fname, firstNum_, bestNum_, threshold);
    import(fname, threshold);
  }
}

BinaryShortlistGenerator::BinaryShortlistGenerator(const void *ptr_void,
                                                   const size_t blobSize,
                                                   Ptr<const Vocab> srcVocab,
                                                   Ptr<const Vocab> trgVocab,
                                                   size_t srcIdx /*= 0*/,
                                                   size_t /*trgIdx = 1*/,
                                                   bool shared /*= false*/,
                                                   bool check /*= true*/)
    : srcVocab_(srcVocab),
      trgVocab_(trgVocab),
      srcIdx_(srcIdx),
      shared_(shared) {
  load(ptr_void, blobSize, check);
}

Ptr<Shortlist> BinaryShortlistGenerator::generate(Ptr<data::CorpusBatch> batch) const {
  auto srcBatch = (*batch)[srcIdx_];
  size_t srcVocabSize = srcVocab_->size();
  size_t trgVocabSize = trgVocab_->size();

  // Since V=trgVocab_->size() is not large, anchor the time and space complexity to O(V).
  // Attempt to squeeze the truth tables into CPU cache
  std::vector<bool> srcTruthTable(srcVocabSize, 0);  // holds selected source words
  std::vector<bool> trgTruthTable(trgVocabSize, 0);  // holds selected target words

  // add firstNum most frequent words
  for(WordIndex i = 0; i < firstNum_ && i < trgVocabSize; ++i)
    trgTruthTable[i] = 1;

  // collect unique words from source
  // add aligned target words: mark trgTruthTable[word] to 1
  for(auto word : srcBatch->data()) {
    WordIndex srcIndex = word.toWordIndex();
    if(shared_)
      trgTruthTable[srcIndex] = 1;
    // If srcIndex has not been encountered, add the corresponding target words
    if (!srcTruthTable[srcIndex]) {
      for (uint64_t j = wordToOffset_[srcIndex]; j < wordToOffset_[srcIndex+1]; j++)
        trgTruthTable[shortLists_[j]] = 1;
      srcTruthTable[srcIndex] = 1;
    }
  }

  // Due to the 'multiple-of-eight' issue, the following O(N) patch is inserted
  size_t trgTruthTableOnes = 0;   // counter for no. of selected target words
  for (size_t i = 0; i < trgVocabSize; i++) {
    if(trgTruthTable[i])
      trgTruthTableOnes++;
  }

  // Ensure that the generated vocabulary items from a shortlist are a multiple-of-eight
  // This is necessary until intgemm supports non-multiple-of-eight matrices.
  for (size_t i = firstNum_; i < trgVocabSize && trgTruthTableOnes%8!=0; i++){
    if (!trgTruthTable[i]){
      trgTruthTable[i] = 1;
      trgTruthTableOnes++;
    }
  }

  // turn selected indices into vector and sort (Bucket sort: O(V))
  std::vector<WordIndex> indices;
  for (WordIndex i = 0; i < trgVocabSize; i++) {
    if(trgTruthTable[i])
      indices.push_back(i);
  }

  return New<Shortlist>(indices);
}

void BinaryShortlistGenerator::dump(const std::string& fileName) const {
  ABORT_IF(mmapMem_.is_open(),"No need to dump again");
  LOG(info, "[data] Saving binary shortlist dump to {}", fileName);
  saveBlobToFile(fileName);
}

void BinaryShortlistGenerator::import(const std::string& filename, double threshold) {
  io::InputFileStream in(filename);
  std::string src, trg;

  // Read text file
  std::vector<std::unordered_map<WordIndex, float>> srcTgtProbTable(srcVocab_->size());
  float prob;

  while(in >> trg >> src >> prob) {
    if(src == "NULL" || trg == "NULL")
      continue;

    auto sId = (*srcVocab_)[src].toWordIndex();
    auto tId = (*trgVocab_)[trg].toWordIndex();

    if(srcTgtProbTable[sId][tId] < prob)
      srcTgtProbTable[sId][tId] = prob;
  }

  // Create priority queue and count
  std::vector<std::priority_queue<std::pair<float, WordIndex>>> vpq;
  uint64_t shortListsSize = 0;

  vpq.resize(srcTgtProbTable.size());
  for(WordIndex sId = 0; sId < srcTgtProbTable.size(); sId++) {
    uint64_t shortListsSizeCurrent = 0;
    for(auto entry : srcTgtProbTable[sId]) {
      if (entry.first>=threshold) {
        vpq[sId].push(std::make_pair(entry.second, entry.first));
        if(shortListsSizeCurrent < bestNum_)
          shortListsSizeCurrent++;
      }
    }
    shortListsSize += shortListsSizeCurrent;
  }

  wordToOffsetSize_ = vpq.size() + 1;
  shortListsSize_ = shortListsSize;

  // Generate a binary blob
  blob_.resize(sizeof(Header) + wordToOffsetSize_ * sizeof(uint64_t) + shortListsSize_ * sizeof(WordIndex));
  struct Header* pHeader = (struct Header *)blob_.data();
  pHeader->magic = BINARY_SHORTLIST_MAGIC;
  pHeader->firstNum = firstNum_;
  pHeader->bestNum = bestNum_;
  pHeader->wordToOffsetSize = wordToOffsetSize_;
  pHeader->shortListsSize = shortListsSize_;
  uint64_t* wordToOffset = (uint64_t*)((char *)pHeader + sizeof(Header));
  WordIndex* shortLists = (WordIndex*)((char*)wordToOffset + wordToOffsetSize_*sizeof(uint64_t));

  uint64_t shortlistIdx = 0;
  for (size_t i = 0; i < wordToOffsetSize_ - 1; i++) {
    wordToOffset[i] = shortlistIdx;
    for(int popcnt = 0; popcnt < bestNum_ && !vpq[i].empty(); popcnt++) {
      shortLists[shortlistIdx] = vpq[i].top().second;
      shortlistIdx++;
      vpq[i].pop();
    }
  }
  wordToOffset[wordToOffsetSize_-1] = shortlistIdx;

  // Sort word indices for each shortlist
  for(int i = 1; i < wordToOffsetSize_; i++) {
    std::sort(&shortLists[wordToOffset[i-1]], &shortLists[wordToOffset[i]]);
  }
  pHeader->checksum = (uint64_t)util::hashMem<uint64_t>((uint64_t *)blob_.data()+2,
                                                        blob_.size()/sizeof(uint64_t)-2);

  wordToOffset_ = wordToOffset;
  shortLists_ = shortLists;
}

void BinaryShortlistGenerator::saveBlobToFile(const std::string& fileName) const {
  io::OutputFileStream outTop(fileName);
  outTop.write(blob_.data(), blob_.size());
}

}  // namespace data
}  // namespace marian
