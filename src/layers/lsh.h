#pragma once

#include "graph/expression_operators.h"
#include "graph/node_initializers.h"

#include <vector>

/**
 * In this file we bascially take the faiss::IndexLSH and pick it apart so that the individual steps
 * can be implemented as Marian inference operators. We can encode the inputs and weights into their
 * bitwise equivalents, apply the hashing rotation (if required), and perform the actual search. 
 * 
 * This also allows to create parameters that get dumped into the model weight file. This is currently 
 * a bit hacky (see marian-conv), but once this is done the model can memory-map the LSH with existing
 * mechanisms and no additional memory is consumed to build the index or rotation matrix.
 */

namespace marian {
namespace lsh {
  // encodes an input as a bit vector, with optional rotation
  Expr encode(Expr input, Expr rotator = nullptr);

  // compute the rotation matrix (maps weights->shape()[-1] to nbits floats)
  Expr rotator(Expr weights, int inDim, int nbits);

  // perform the LSH search on fully encoded input and weights, return k results (indices) per input row
  // @TODO: add a top-k like operator that also returns the bitwise computed distances
  Expr searchEncoded(Expr encodedQuery, Expr encodedWeights, int k, int firstNRows = 0, bool noSort = false);

  // same as above, but performs encoding on the fly
  Expr search(Expr query, Expr weights, int k, int nbits, int firstNRows = 0, bool abortIfDynamic = false);
  
  // struct for parameter conversion used in marian-conv
  struct ParamConvInfo {
    std::string name;
    std::string codesName;
    std::string rotationName;
    int nBits;
    bool transpose;

    ParamConvInfo(const std::string& name, const std::string& codesName, const std::string& rotationName, int nBits, bool transpose = false) 
     : name(name), codesName(codesName), rotationName(rotationName), nBits(nBits), transpose(transpose) {}
  };

  // These are helper functions for encoding the LSH into the binary Marian model, used by marian-conv
  void addDummyParameters(Ptr<ExpressionGraph> graph, ParamConvInfo paramInfo);
  void overwriteDummyParameters(Ptr<ExpressionGraph> graph, ParamConvInfo paramInfo);

  /**
   * Computes a random rotation matrix for LSH hashing.
   * This is part of a hash function. The values are orthonormal and computed via
   * QR decomposition.
   */
  Ptr<inits::NodeInitializer> randomRotation();
}

}