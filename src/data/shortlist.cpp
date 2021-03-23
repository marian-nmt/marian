#include "data/shortlist.h"
#include "microsoft/shortlist/utils/ParameterTree.h"

namespace marian {
namespace data {

// cast current void pointer to T pointer and move forward by num elements 
template <typename T>
const T* get(const void*& current, size_t num = 1) {
  const T* ptr = (const T*)current;
  current = (const T*)current + num;
  return ptr;
}

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
                                                 size_t srcIdx,
                                                 size_t trgIdx,
                                                 bool shared) {
  std::vector<std::string> vals = options->get<std::vector<std::string>>("shortlist");
  ABORT_IF(vals.empty(), "No path to shortlist given");
  std::string fname = vals[0];
  if(filesystem::Path(fname).extension().string() == ".bin") {
    return New<QuicksandShortlistGenerator>(options, srcVocab, trgVocab, srcIdx, trgIdx, shared);
  } else {
    return New<LexicalShortlistGenerator>(options, srcVocab, trgVocab, srcIdx, trgIdx, shared);
  }
}

}  // namespace data
}  // namespace marian
