#pragma once

#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "integer_common.h"

namespace marian {

namespace cpu {
namespace integer {

#if COMPILE_CPU
/*
 * Prepare an activation matrix into intgemm8/16 format. For now the activation matrix is just quantized.
 * Expr input: The input tensor
 */
template<Type vtype>
static inline Expr prepareA(Expr a) {
  auto nodeOp = [](Expr out, const std::vector<Expr>& children) {
    Expr in = children[0];
    auto quantMult = computeQuantMult<vtype>(in->val());
    typedef typename intgemm_<vtype>::type Integer;
    intgemm_<vtype>::width::PrepareA(in->val()->data(), /*input*/
                                     out->val()->data<Integer>(), /*output*/
                                     quantMult, /*Quant Mult*/
                                     rows(in->val()),
                                     cols(in->val()));
    getQuantMult<vtype>(out->val()) = quantMult;
  };

  return lambda({a}, a->shape(), vtype, nodeOp);
}
#endif

/*	
 * This computes A*B (+ bias if available) in intgemm.	
 * Expr a: The activation matrix in intgemm format	
 * Expr b: The parameter matrix in intgemm fromat	
 * Expr bias: The bias	
 * bool transA - tranpose input A if true
 * bool transB - unused here (@TODO remove?)
 * float scale - scale the output by `scale`
 * the template argument controls whether we're doing 16bit integers or 8bit integers. 
 * It can be Type::intgemm8 or Type::intgemm16 and all hardware-specific variants	
 */
template<Type vtype>
static inline Expr affineOrDotTyped(Expr a, Expr bQuant, Expr bias, bool transA, bool /*transB*/, float scale) {
#if COMPILE_CPU
  ABORT_IF(!isFloat(a->value_type()), "Intgemm expects type of A to be float32 not {}", a->value_type());
  ABORT_IF(!isIntgemm(bQuant->value_type()), "Intgemm expects type of B to be a variant of intgemm not {}", bQuant->value_type());

  auto aQuant = prepareA<vtype>(transA ? transpose(a) : a); // A should not be quantized yet as seen above, hence quantize here
  
  // determine the output shape m x n for A: m x k and B: k x n
  // since we transpose A beforehand we don't need to take care of transposed shapes here 
  Shape outShape = aQuant->shape();
  outShape.set(-1, bQuant->shape()[-1]);

  // wrap the multiply finctions to be executed in the forward step of a Lambda node
  auto dotOrAffineNodeOp = [=](Expr out, const std::vector<Expr>& children) {
    Expr aQuant = children[0];
    Expr bQuant = children[1];
    Expr bias   = children.size() > 2 ? children[2] : nullptr;

    // when we arrive here, A and B are already quantized, so just get the multipliers
    float aQuantMult = getQuantMult<vtype>(aQuant->val());
    float bQuantMult = getQuantMult<vtype>(bQuant->val());
        
    float unquant_mult = 1.0f / (aQuantMult * bQuantMult);
    unquant_mult = unquant_mult * scale;

    typedef typename intgemm_<vtype>::type Integer;
    if(bias) { // dispatch a multiply with integrated bias addition i.e affine(...)
      intgemm_<vtype>::width::Multiply(/*A=*/aQuant->val()->data<Integer>(),
                                       /*B=*/bQuant->val()->data<Integer>(),
                                       rows(aQuant->val()),
                                       cols(aQuant->val()),
                                       cols(bQuant->val()),
                                       intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, /*bias=*/bias->val()->data(), /*output=*/out->val()->data()));
    } else { // dispatch a multiply without bias addition i.e dot(...)
      intgemm_<vtype>::width::Multiply(/*A=*/aQuant->val()->data<Integer>(),
                                       /*B=*/bQuant->val()->data<Integer>(),
                                       rows(aQuant->val()),
                                       cols(aQuant->val()),
                                       cols(bQuant->val()),
                                       intgemm::callbacks::UnquantizeAndWrite(unquant_mult, /*output=*/out->val()->data()));
    }
  };

  std::vector<Expr> children = {aQuant, bQuant};
  if(bias)
    children.push_back(bias);

  return lambda(children, outShape, Type::float32, dotOrAffineNodeOp); // inference-only Lambda node
#else
  a, bQuant, bias, transA, scale;
  ABORT("You need to enable CPU compilation to use this feature. Use cmake .. -DCOMPILE_CPU=ON");
#endif
}

// Dispatch correct hardware-agnostic or hardware-specific matrix multiplies
static inline Expr affineOrDot(Expr a, Expr bQuant, Expr bias, bool transA, bool transB, float scale) {
  Type bQuantElementType = bQuant->value_type();
  static const bool pass = cpu::integer::passOrAbort(bQuantElementType);
  pass; // We declare this variable as static so that passOrAbort is only ever run once during the initialization.
  switch(bQuantElementType) {
    //case Type::intgemm8 :  // The generic case selects CPU automatically, but we set all the types manually anyways.
    //  return cpu::integer::affineOrDotTyped<Type::intgemm8>(a, bQuant, bias, transA, transB, scale);    
    case Type::intgemm8ssse3 :
      return cpu::integer::affineOrDotTyped<Type::intgemm8ssse3>(a, bQuant, bias, transA, transB, scale);
    case Type::intgemm8avx2 :
      return cpu::integer::affineOrDotTyped<Type::intgemm8avx2>(a, bQuant, bias, transA, transB, scale);
    case Type::intgemm8avx512 :
      return cpu::integer::affineOrDotTyped<Type::intgemm8avx512>(a, bQuant, bias, transA, transB, scale);
    case Type::intgemm8avx512vnni :
      return cpu::integer::affineOrDotTyped<Type::intgemm8avx512vnni>(a, bQuant, bias, transA, transB, scale);
    //case Type::intgemm16 :  // The generic case selects CPU automatically, but we set all the types manually anyways.
    //  return cpu::integer::affineOrDotTyped<Type::intgemm16>(a, bQuant, bias, transA, transB, scale);
    case Type::intgemm16sse2 :
      return cpu::integer::affineOrDotTyped<Type::intgemm16sse2>(a, bQuant, bias, transA, transB, scale);
    case Type::intgemm16avx2 :
      return cpu::integer::affineOrDotTyped<Type::intgemm16avx2>(a, bQuant, bias, transA, transB, scale);
    case Type::intgemm16avx512 :
      return cpu::integer::affineOrDotTyped<Type::intgemm16avx512>(a, bQuant, bias, transA, transB, scale);
    default:
      ABORT("Unsupported type {} for Intgemm type??", bQuantElementType);
  }
}

}  // namespace integer
}  // namespace cpu
}  // namespace marian
