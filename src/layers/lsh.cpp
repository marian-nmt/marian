#include "layers/lsh.h"
#include "graph/expression_operators.h"
#include "tensors/cpu/prod_blas.h"

#if BLAS_FOUND
#include "3rd_party/faiss/IndexLSH.h"
#endif

namespace marian {

Expr LSH::apply(Expr input, Expr W, Expr b) {
  auto idx = search(input, W);
  return affine(idx, input, W, b);
}

Expr LSH::search(Expr query, Expr values) {
#if BLAS_FOUND
  ABORT_IF(query->graph()->getDeviceId().type == DeviceType::gpu,
           "LSH index (--output-approx-knn) currently not implemented for GPU");

  auto kShape = query->shape();
  kShape.set(-1, k_);

  auto forward = [this](Expr out, const std::vector<Expr>& inputs) {
    auto query  = inputs[0];
    auto values = inputs[1];

    int dim = values->shape()[-1];

    if(!index_ || indexHash_ != values->hash()) {
      LOG(info, "Building LSH index for vector dim {} and with hash size {} bits", dim, nbits_);
      index_.reset(new faiss::IndexLSH(dim, nbits_, 
                                       /*rotate=*/dim != nbits_, 
                                       /*train_thesholds*/false));
      int vRows = values->shape().elements() / dim;
      index_->train(vRows, values->val()->data<float>());
      index_->add(  vRows, values->val()->data<float>());
      indexHash_ = values->hash();
    }

    int qRows = query->shape().elements() / dim;
    std::vector<float> distances(qRows * k_);
    std::vector<faiss::Index::idx_t> ids(qRows * k_);

    index_->search(qRows, query->val()->data<float>(), k_,
                   distances.data(), ids.data());
    
    std::vector<IndexType> vOut;
    vOut.reserve(ids.size());
    for(auto id : ids)
      vOut.push_back((IndexType)id);

    out->val()->set(vOut);
  };

  return lambda({query, values}, kShape, Type::uint32, forward);
#else
  query; values;
  ABORT("LSH output layer requires a CPU BLAS library");
#endif
}

Expr LSH::affine(Expr idx, Expr input, Expr W, Expr b) {
  auto outShape = input->shape();
  int dimVoc    = W->shape()[-2];
  outShape.set(-1, dimVoc);

  auto forward = [this](Expr out, const std::vector<Expr>& inputs) {
    auto lowest = NumericLimits<float>(out->value_type()).lowest;
    out->val()->set(lowest);

    int dimIn   = inputs[1]->shape()[-1];
    int dimOut  = out->shape()[-1];
    int dimRows = out->shape().elements() / dimOut;
    
    auto outPtr   = out->val()->data<float>();
    auto idxPtr   = inputs[0]->val()->data<uint32_t>();
    auto queryPtr = inputs[1]->val()->data<float>();
    auto WPtr     = inputs[2]->val()->data<float>();
    auto bPtr     = inputs.size() > 3 ? inputs[3]->val()->data<float>() : nullptr; // nullptr if no bias given

    for(int row = 0; row < dimRows; ++row) {
      auto currIdxPtr    = idxPtr   + row * k_;     // move to next batch of k entries
      auto currQueryPtr  = queryPtr + row * dimIn;  // move to next input query vector
      auto currOutPtr    = outPtr   + row * dimOut; // move to next output position vector (of vocabulary size)
      for(int k = 0; k < k_; k++) {
        int relPos = currIdxPtr[k];                   // k-th best vocabulay item
        auto currWPtr      = WPtr + relPos * dimIn;   // offset for k-th best embedding
        currOutPtr[relPos] = bPtr ? bPtr[relPos] : 0; // write bias value to position, init to 0 if no bias given
        
        // proceed one vector product at a time writing to the correct position
        sgemm(false, true, 1, 1, dimIn, 1.0f, currQueryPtr, dimIn, currWPtr, dimIn, 1.0f, &currOutPtr[relPos], 1);
      }
    }
  };

  std::vector<Expr> nodes = {idx, input, W};
  if(b) // bias is optional
    nodes.push_back(b);

  return lambda(nodes, 
                outShape,
                input->value_type(),
                forward);
}

// @TODO: alternative version which does the same as above with Marian operators, currently missing "scatter".
// this uses more memory and likely to be slower. Would make sense to have a scatter node that actually creates
// the node instead of relying on an existing node, e.g. scatter(shape, defaultValue, axis, indices, values);
#if 0 
Expr LSH::affine(Expr idx, Expr input, Expr W, Expr b) {
  int dim  = input->shape()[-1];
  int bch  = idx->shape().elements() / k;

  auto W = reshape(rows(Wt_, flatten(idx)), {bch, k, dim}); // [rows, k, dim]
  auto b = reshape(cols(b_,  flatten(idx)), {bch, 1,   k}); // [rows, 1,   k]

  auto aff = reshape(bdot(reshape(input, {bch, 1, dim}), W, false, true) + b, idx->shape()); // [beam, time, batch, k]

  int dimVoc  = Wt_->shape()[-2];
  auto oShape = input->shape();
  oShape.set(-1, dimVoc);
  auto lowest = graph_->constant(oShape, 
                                 inits::fromValue(NumericLimits<float>(input->value_type()).lowest), 
                                 input->value_type());
  return scatter(lowest, -1, idx, aff);
}
#endif

}  // namespace marian