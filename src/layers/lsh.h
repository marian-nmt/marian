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

  // return the number of full bytes required to encoded that many bits
  int bytesPerVector(int nBits);

  // encodes an input as a bit vector, with optional rotation
  Expr encode(Expr input, Expr rotator = nullptr);

  // compute the rotation matrix (maps weights->shape()[-1] to nbits floats)
  Expr rotator(Expr weights, int nbits);

  // perform the LSH search on fully encoded input and weights, return k results (indices) per input row
  // @TODO: add a top-k like operator that also returns the bitwise computed distances
  Expr searchEncoded(Expr encodedQuery, Expr encodedWeights, int k, int firstNRows = 0);

  // same as above, but performs encoding on the fly
  Expr search(Expr query, Expr weights, int k, int nbits, int firstNRows = 0, bool abortIfDynamic = false);
  
  // These are helper functions for encoding the LSH into the binary Marian model, used by marian-conv
  void addDummyParameters(Ptr<ExpressionGraph> graph, std::string weightsName, int nBits);
  void overwriteDummyParameters(Ptr<ExpressionGraph> graph, std::string weightsName);

  /**
   * Computes a random rotation matrix for LSH hashing.
   * This is part of a hash function. The values are orthonormal and computed via
   * QR decomposition.
   */
  Ptr<inits::NodeInitializer> randomRotation();
}

}