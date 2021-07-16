#include "layers/lsh.h"
#include "tensors/tensor_operators.h"
#include "common/utils.h"

#include "3rd_party/faiss/utils/hamming.h"
#include "3rd_party/faiss/Index.h"

#if BLAS_FOUND
#include "3rd_party/faiss/VectorTransform.h"
#endif


namespace marian {
namespace lsh {

int bytesPerVector(int nBits) {
  return (nBits + 7) / 8;
}

void fillRandomRotationMatrix(Tensor output, Ptr<Allocator> allocator) {
#if BLAS_FOUND
  int nRows = output->shape()[-2];
  int nBits = output->shape()[-1];

  // @TODO re-implement using Marian code so it uses the correct random generator etc.
  faiss::RandomRotationMatrix rrot(nRows, nBits);
  // Then we do not need to use this seed at all
  rrot.init(5); // currently set to 5 following the default from FAISS, this could be any number really.

  // The faiss random rotation matrix is column major, hence we create a temporary tensor, 
  // copy the rotation matrix into it and transpose to output.
  Shape tempShape = {nBits, nRows};
  auto memory = allocator->alloc(requiredBytes(tempShape, output->type()));
  auto temp = TensorBase::New(memory,
                              tempShape,
                              output->type(),
                              output->getBackend());
  temp->set(rrot.A);
  TransposeND(output, temp, {0, 1, 3, 2});
  allocator->free(memory);
#else
  output; allocator;
  ABORT("LSH with rotation matrix requires Marian to be compiled with a BLAS library");
#endif
}

void encode(Tensor output, Tensor input) {
  int nBits = input->shape()[-1]; // number of bits is equal last dimension of float matrix
  int nRows = input->shape().elements() / nBits;
  faiss::fvecs2bitvecs(input->data<float>(), output->data<uint8_t>(), (size_t)nBits, (size_t)nRows);
}

void encodeWithRotation(Tensor output, Tensor input, Tensor rotation, Ptr<Allocator> allocator) {
  int nBits = input->shape()[-1]; // number of bits is equal last dimension of float matrix unless we rotate
  int nRows = input->shape().elements() / nBits;

  Tensor tempInput = input;
  MemoryPiece::PtrType memory;
  if(rotation) {
    int nBitsRot = rotation->shape()[-1];
    Shape tempShape = {nRows, nBitsRot};
    memory = allocator->alloc(requiredBytes(tempShape, rotation->type()));
    tempInput = TensorBase::New(memory, tempShape, rotation->type(), rotation->getBackend());
    Prod(tempInput, input, rotation, false, false, 0.f, 1.f);
  }
  encode(output, tempInput);

  if(memory)
    allocator->free(memory);
};

Expr encode(Expr input, Expr rotation) {
  auto encodeFwd = [](Expr out, const std::vector<Expr>& inputs) {
    if(inputs.size() == 1) {
      encode(out->val(), inputs[0]->val());
    } else if(inputs.size() == 2) {
      encodeWithRotation(out->val(), inputs[0]->val(), inputs[1]->val(), out->graph()->allocator());
    } else {
      ABORT("Too many inputs to encode??");
    }
  };

  // Use the address of the first lambda function as an immutable hash. Making it static and const makes sure
  // that this hash value will not change. Next pass the hash into the lambda functor were it will be used
  // to identify this unique operation. Marian's ExpressionGraph can automatically memoize and identify nodes 
  // that operate only on immutable nodes (parameters) and have the same hash. This way we make sure that the 
  // codes node won't actually get recomputed throughout ExpressionGraph lifetime. `codes` will be reused 
  // and the body of the lambda will not be called again. This does however build one index per graph.
  static const size_t encodeHash = (size_t)&encodeFwd; 

  Shape encodedShape = input->shape();

  int nBits = rotation ? rotation->shape()[-1] : input->shape()[-1];
  encodedShape.set(-1, bytesPerVector(nBits));
  std::vector<Expr> inputs = {input};
  if(rotation)
    inputs.push_back(rotation);
  return lambda(inputs, encodedShape, Type::uint8, encodeFwd, encodeHash);
}

Expr rotator(Expr weights, int nBits) {
  auto rotator = [](Expr out, const std::vector<Expr>& inputs) {
    inputs;
    fillRandomRotationMatrix(out->val(), out->graph()->allocator());
  };

  static const size_t rotatorHash = (size_t)&rotator; 
  int dim = weights->shape()[-1];
  return lambda({weights}, {dim, nBits}, Type::float32, rotator, rotatorHash);
}

Expr searchEncoded(Expr encodedQuery, Expr encodedWeights, int k, int firstNRows) {
  ABORT_IF(encodedQuery->shape()[-1] != encodedWeights->shape()[-1],
           "Query and index bit vectors need to be of same size ({} != {})", encodedQuery->shape()[-1], encodedWeights->shape()[-1]);

  int currBeamSize = encodedQuery->shape()[0];
  int batchSize    = encodedQuery->shape()[2];
  int numHypos = currBeamSize * batchSize;

  auto search = [=](Expr out, const std::vector<Expr>& inputs) {
    Expr encodedQuery   = inputs[0];
    Expr encodedWeights = inputs[1];

    int bytesPerVector = encodedWeights->shape()[-1];
    int wRows = encodedWeights->shape().elements() / bytesPerVector;
    
    // we use this with Factored Segmenter to skip the factor embeddings at the end
    if(firstNRows != 0)
      wRows = firstNRows;

    int qRows = encodedQuery->shape().elements() / bytesPerVector;

    uint8_t* qCodes = encodedQuery->val()->data<uint8_t>();
    uint8_t* wCodes = encodedWeights->val()->data<uint8_t>();

    // use actual faiss code for performing the hamming search. 
    std::vector<int> distances(qRows * k);
    std::vector<faiss::Index::idx_t> ids(qRows * k);
    faiss::int_maxheap_array_t res = {(size_t)qRows, (size_t)k, ids.data(), distances.data()};
    faiss::hammings_knn_hc(&res, qCodes, wCodes, (size_t)wRows, (size_t)bytesPerVector, 0);

    // Copy int64_t indices to Marian index type and sort by increasing index value per hypothesis.
    // The sorting is required as we later do a binary search on those values for reverse look-up.
    uint32_t* outData = out->val()->data<uint32_t>();
    for (size_t hypoIdx = 0; hypoIdx < numHypos; ++hypoIdx) {
      size_t startIdx = k * hypoIdx;
      size_t endIdx = startIdx + k;
      for(size_t i = startIdx; i < endIdx; ++i)
        outData[i] = (uint32_t)ids[i];
      std::sort(outData + startIdx, outData + endIdx);
    }
  };

  Shape kShape({currBeamSize, batchSize, k});
  return lambda({encodedQuery, encodedWeights}, kShape, Type::uint32, search);
}

Expr search(Expr query, Expr weights, int k, int nBits, int firstNRows, bool abortIfDynamic) {
  int dim = weights->shape()[-1];
  
  Expr rotMat = nullptr;
  if(dim != nBits) {
    rotMat = weights->graph()->get("lsh_output_rotation");
    if(rotMat) {
      LOG_ONCE(info, "Reusing parameter LSH rotation matrix {} with shape {}", rotMat->name(), rotMat->shape());
    } else {
      ABORT_IF(abortIfDynamic, "Dynamic creation of LSH rotation matrix prohibited");
      LOG_ONCE(info, "Creating ad-hoc rotation matrix with shape {}", Shape({dim, nBits}));
      rotMat = rotator(weights, nBits);
    }
  }

  Expr encodedWeights = weights->graph()->get("lsh_output_codes");
  if(encodedWeights) {
    LOG_ONCE(info, "Reusing parameter LSH code matrix {} with shape {}", encodedWeights->name(), encodedWeights->shape());
  } else {
    ABORT_IF(abortIfDynamic, "Dynamic creation of LSH code matrix prohibited");
    LOG_ONCE(info, "Creating ad-hoc code matrix with shape {}", Shape({weights->shape()[-2], lsh::bytesPerVector(nBits)}));
    encodedWeights = encode(weights, rotMat);
  }
  
  return searchEncoded(encode(query, rotMat), encodedWeights, k, firstNRows);
}

class RandomRotation : public inits::NodeInitializer {
public:
  void apply(Tensor tensor) override {
    auto sharedAllocator = allocator_.lock();
    ABORT_IF(!sharedAllocator, "Allocator in RandomRotation has not been set or expired");
    fillRandomRotationMatrix(tensor, sharedAllocator);
  }
};

Ptr<inits::NodeInitializer> randomRotation() {
  return New<RandomRotation>();
}

void addDummyParameters(Ptr<ExpressionGraph> graph, std::string weightsName, int nBitsRot) {
  auto weights = graph->get(weightsName);

  ABORT_IF(!weights, "Trying to encode non-existing weights matrix {}??", weightsName);

  int nBits = weights->shape()[-1];
  int nRows = weights->shape().elements() / nBits;

  Expr rotation;
  if(nBits != nBitsRot) {
    LOG(info, "Adding LSH rotation parameter lsh_output_rotation with shape {}", Shape({nBits, nBitsRot}));
    rotation = graph->param("lsh_output_rotation", {nBits, nBitsRot}, inits::dummy(), Type::float32);
    nBits = nBitsRot;
  }
  
  int bytesPerVector = lsh::bytesPerVector(nBits);
  LOG(info, "Adding LSH encoded weights lsh_output_codes with shape {}", Shape({nRows, bytesPerVector}));
  auto codes = graph->param("lsh_output_codes", {nRows, bytesPerVector}, inits::dummy(), Type::uint8);
}

void overwriteDummyParameters(Ptr<ExpressionGraph> graph, std::string weightsName) {
  Expr weights  = graph->get(weightsName);
  Expr codes    = graph->get("lsh_output_codes");
  Expr rotation = graph->get("lsh_output_rotation");

  ABORT_IF(!weights, "Trying to encode non-existing weights matrix {}??", weightsName);
  ABORT_IF(!codes, "Trying to overwrite non-existing LSH parameters lsh_output_codes??");

  if(rotation) {
    fillRandomRotationMatrix(rotation->val(), weights->graph()->allocator());
    encodeWithRotation(codes->val(), weights->val(), rotation->val(), weights->graph()->allocator());
  } else {
    encode(codes->val(), weights->val());
  }
}

}
}