#pragma once

#include <vector>

#ifdef _MSC_VER
#define __builtin_popcountl __popcnt64
#define __builtin_popcount __popcnt
#endif

namespace marian {
namespace lsh {

  struct Parameters {
    int k;
    uint8_t* queryRows;
    int numQueryRows;
    uint8_t* codeRows;
    int numCodeRows;
    int bytesPerVector;
  };

  typedef uint32_t DistType;
  typedef uint64_t ChunkType;

  inline DistType popcount(const ChunkType& chunk) {
    switch (sizeof(ChunkType)) {
      case 8 : return (DistType)__builtin_popcountl((uint64_t)chunk);
      case 4 : return (DistType)__builtin_popcount((uint32_t)chunk);
      default: ABORT("Size {} not supported", sizeof(ChunkType));
    }
  }

  // return the number of full bytes required to encoded that many bits
  inline int bytesPerVector(int nBits);

  // compute top-k hamming distances for given query and weight binary codes. Faster than FAISS version, especially for larger k nearly constant wrt. k. 
  template <int StaticValue = 0, bool Dynamic=true, typename T>
  inline constexpr T getStaticOrDynamic(T dynamicValue) {
    return Dynamic ? dynamicValue : StaticValue;
  }

  template <size_t StepsStatic, bool Dynamic=false> 
  inline DistType hamming(ChunkType* queryRow, ChunkType* codeRow, int stepsDynamic = 0) {
    static_assert(Dynamic == true || StepsStatic != 0, "Either define dynamic use of steps or provide non-zero template argument");
    DistType dist = 0;
    for(int i = 0; i < getStaticOrDynamic<StepsStatic, Dynamic>(stepsDynamic); ++i)
      dist += popcount(queryRow[i] ^ codeRow[i]);
    return dist;
  }

  template <int warpSize, int NumCodeRows, int BytesPerVector, bool Dynamic, class Functor>
  inline void hammingTopKUnrollWarp(int queryOffset, const Parameters& parameters, const Functor& gather) {
    const int numBits = getStaticOrDynamic<BytesPerVector, Dynamic>(parameters.bytesPerVector) * 8;
    ABORT_IF(numBits % 64 != 0, "LSH hash size has to be a multiple of 64");

    // counter to keep track of seen hamming distances
    std::vector<std::vector<DistType>> counter(warpSize, std::vector<DistType>(numBits, 0));
    // buffer the distances for query vector warpRowId to all weight weight vectors codeRowId
    std::vector<std::vector<DistType>> distBuffer(warpSize, std::vector<DistType>(getStaticOrDynamic<NumCodeRows, Dynamic>(parameters.numCodeRows), 0));
    // minimal distances per query
    std::vector<DistType> minDist(warpSize);

    constexpr int StepStatic = BytesPerVector / sizeof(ChunkType);
    int stepDynamic = parameters.bytesPerVector / sizeof(ChunkType);

    ChunkType* codeRow = (ChunkType*)parameters.codeRows;
    
    for(int warpRowId = 0; warpRowId < warpSize; warpRowId++) {
      std::memset(counter[warpRowId].data(), 0, numBits * sizeof(DistType)); // Reset the counter via memset to 0
      minDist[warpRowId] = (DistType)numBits;
    }

    for(IndexType codeRowId = 0; codeRowId < (IndexType)getStaticOrDynamic<NumCodeRows, Dynamic>(parameters.numCodeRows); ++codeRowId, codeRow += getStaticOrDynamic<StepStatic, Dynamic>(stepDynamic)) {
      ChunkType* queryRow = (ChunkType*)parameters.queryRows;
      for(IndexType warpRowId = 0; warpRowId < warpSize; warpRowId++, queryRow += getStaticOrDynamic<StepStatic, Dynamic>(stepDynamic)) {
        // Compute the bit-wise hamming distance
        DistType dist = hamming<StepStatic, Dynamic>(queryRow, codeRow, stepDynamic);
      
        // Record the minimal distance seen for this query vector wrt. all weight vectors
        if(dist < minDist[warpRowId]) {
          minDist[warpRowId] = dist;
        }

        // Record the number of weight vectors that have this distance from the query vector.
        // Note, because there is at most numBits different distances this can be trivially done.
        // Not the case for generic distances like float.
        counter[warpRowId][dist]++;

        // Record the distance for this weight vector
        distBuffer[warpRowId][codeRowId] = dist;
      }
    }
    // warp finished, harvest k top distances

    for(int warpRowId = 0; warpRowId < warpSize; warpRowId++) {
      // Here we search for the distance at which we have seen equal or more than k elements with 
      // smaller distances. We start with the minimal distance from above which is its own address 
      // to the counter.
      DistType maxDist = minDist[warpRowId];
      size_t cummulativeDistances = 0;

      // Accumulate number of elements until we reach k in growing distance order. Note that
      // counter is indexed by hamming distance - from lowest to highest. Some slots will be 0.
      // The cumulative sum from position a to b tells you how many elements have distances smaller
      // than the distance at b. 
      while(cummulativeDistances < parameters.k)
        cummulativeDistances += counter[warpRowId][maxDist++];
      if(cummulativeDistances) 
        maxDist--; // fix overcounting

      // Usually, we overshoot by a couple of elements and we need to take care of the distance at which the k-th 
      // element sits. This elements has more neighbors at the same distance, but we only care for them
      // as long we have not reached k elements in total. 
      // By contrast, we trivially collect all elements below that distance -- these are always safe.

      // This is the number of elements we need to collect at the last distance. 
      DistType maxDistLimit = /*number of elements at maxDist=*/counter[warpRowId][maxDist] - /*overflow=*/((DistType)cummulativeDistances - (DistType)parameters.k);
      IndexType kSeen = 0;
      IndexType kSeenAtKDist = 0;

      for(IndexType codeRowId = 0; kSeen < (IndexType)parameters.k && codeRowId < (IndexType)getStaticOrDynamic<NumCodeRows, Dynamic>(parameters.numCodeRows); ++codeRowId) {
        DistType dist = distBuffer[warpRowId][codeRowId];
        // - if the current distance is smaller than the maxDist, just consume.
        // - if the distance is equal to maxDist, make sure to only consume maxDistLimit elements at maxDist
        //   and ignore the rest (smaller indices make it in first).
        // - after we finish this loop we have exactly k top values for every query row in original index order.
        int queryRowId = queryOffset + warpRowId;
        if(dist < maxDist) {
          gather(queryRowId, (IndexType)kSeen, codeRowId, dist);
          kSeen++;
        } else if(dist == maxDist && kSeenAtKDist < (DistType)maxDistLimit) {
          gather(queryRowId, (IndexType)kSeen, codeRowId, dist);
          kSeen++;
          kSeenAtKDist++;
        }
      }
    }
  }

  // Faster top-k search for hamming distance. The idea here is that instead of sorting the elements we find a hamming distances at which it is safe
  // to copy the given index. Copying only the indices below that distance is guaranteed to results in no more than k elements. For elements at that
  // distance we need to correct for overshooting. 
  // Once we have that distance we only need to traverse the set of distances. In the end we get exactly k elements per queryRows vector.
  template <int NumCodeRows, int BytesPerVector, bool Dynamic, class Functor>
  inline void hammingTopKUnroll(const Parameters& parameters, const Functor& gather) {
    static_assert(Dynamic == true || (NumCodeRows != 0 && BytesPerVector != 0), "Either define dynamic use of variables or provide non-zero template arguments");

    int warpSize = 4; // starting warpSize of 4 seems optimal
    auto warpParameters = parameters;
    for(int queryOffset = 0; queryOffset < parameters.numQueryRows; queryOffset += warpSize) {
      while(parameters.numQueryRows - queryOffset < warpSize)
        warpSize /= 2;

      int step = getStaticOrDynamic<BytesPerVector, Dynamic>(parameters.bytesPerVector);
      warpParameters.queryRows    = parameters.queryRows + queryOffset * step;
      warpParameters.numQueryRows = warpSize;
      switch(warpSize) {
        case 8 : hammingTopKUnrollWarp<8, NumCodeRows, BytesPerVector, Dynamic>(queryOffset, warpParameters, gather); break;
        case 4 : hammingTopKUnrollWarp<4, NumCodeRows, BytesPerVector, Dynamic>(queryOffset, warpParameters, gather); break;
        case 2 : hammingTopKUnrollWarp<2, NumCodeRows, BytesPerVector, Dynamic>(queryOffset, warpParameters, gather); break;
        case 1 : hammingTopKUnrollWarp<1, NumCodeRows, BytesPerVector, Dynamic>(queryOffset, warpParameters, gather); break;
        default: ABORT("Unhandled warpSize = {}??", warpSize);
      }
    }
  }

  template <class Functor>
  inline void hammingTopK(const Parameters& parameters, const Functor& gather) {
    if(parameters.numCodeRows      ==  2048 && parameters.bytesPerVector ==  64)
      hammingTopKUnroll< 2048,  64, false>(parameters, gather);
    else if(parameters.numCodeRows ==  4096 && parameters.bytesPerVector ==  64)
      hammingTopKUnroll< 4096,  64, false>(parameters, gather);
    else if(parameters.numCodeRows ==  6144 && parameters.bytesPerVector ==  64)
      hammingTopKUnroll< 6144,  64, false>(parameters, gather);
    else if(parameters.numCodeRows ==  8192 && parameters.bytesPerVector ==  64)
      hammingTopKUnroll< 8192,  64, false>(parameters, gather);
    else if(parameters.numCodeRows == 32000 && parameters.bytesPerVector ==  64)
      hammingTopKUnroll<32000,  64, false>(parameters, gather);
    else if(parameters.numCodeRows == 32000 && parameters.bytesPerVector == 128)
      hammingTopKUnroll<32000, 128, false>(parameters, gather);
    else
      hammingTopKUnroll<    0,   0, true>(parameters, gather);
  }

} // namespace lsh
} // namespace marian