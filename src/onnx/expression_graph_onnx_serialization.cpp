#ifdef USE_ONNX

#include "onnx/expression_graph_onnx_exporter.h"
#include "graph/expression_operators.h"
#include "graph/node_operators_unary.h"
#include "graph/node_operators_binary.h"
#include "common/version.h"
#define AuxillaryParseTableField AuxiliaryParseTableField  // in protobuf 3.12, the generated source has a spelling error
#include "3rd_party/onnx/protobuf/onnx-ml.pb-wrapper.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

namespace marian {

  // collection of helper functions for accessing and converting Expr properties
  // This class is a friend of all node-op classes whose attributes we need to access.
  class SerializationHelpers {
  public:
    // helper for accessing class members in Marian's polymorphic node classes
    // If 'e' is of NNaryNodeOp then execute getFn() and return true.
    template<class NNaryNodeOp, typename F>
    static bool tryGetAttributes(Expr e, const F& getFn) {
      auto np = std::dynamic_pointer_cast<NNaryNodeOp>(e);
      if (!np)
        return false;
      getFn(np);
      return true;
    }

    template<class NNaryNodeOp>
    static bool tryGetScalarAttribute(Expr e, float& scalar) {
      return tryGetAttributes<NNaryNodeOp>(e, [&](IPtr<NNaryNodeOp> np) { scalar = np->scalar_; });
    }

    template<class NNaryNodeOp>
    static bool tryGetMatMulAttributes(Expr e, bool& transA, bool& transB, float& scalar) {
      return tryGetAttributes<NNaryNodeOp>(e, [&](IPtr<NNaryNodeOp> np) {
        transA = np->transA_;
        transB = np->transB_;
        scalar = np->scalar_;
      });
    }

    template<class NNaryNodeOp>
    static bool tryGetEpsilonAttribute(Expr e, float& eps) {
      return tryGetAttributes<NNaryNodeOp>(e, [&](IPtr<NNaryNodeOp> np) { eps = np->eps_; });
    }

    template<class NNaryNodeOp>
    static bool tryGetAxisAttribute(Expr e, size_t& axis) {
      return tryGetAttributes<NNaryNodeOp>(e, [&](IPtr<NNaryNodeOp> np) { axis = (size_t)e->shape().axis(np->axis_); });
    }

    template<class NNaryNodeOp>
    static bool tryGetAxesAttribute(Expr e, std::vector<size_t>& axes) {
      return tryGetAttributes<NNaryNodeOp>(e, [&](IPtr<NNaryNodeOp> np) {
        axes.clear();
        for (auto ax : np->axes_)
          axes.push_back((size_t)e->shape().axis(ax));
      });
    }

    template<class NNaryNodeOp>
    static bool tryGetShiftAttributes(Expr e, std::vector<int>& shift, float& padValue) {
      return tryGetAttributes<NNaryNodeOp>(e, [&](IPtr<NNaryNodeOp> np) {
        shift.assign(np->shift_.begin(), np->shift_.end());
        padValue = np->padValue_;
      });
    }

    template<class NNaryNodeOp>
    static bool tryGetSliceAttribute(Expr e, Slice& slice) {
      return tryGetAttributes<NNaryNodeOp>(e, [&](IPtr<NNaryNodeOp> np) { slice = np->slice_; });
    }

    template<class NNaryNodeOp>
    static bool tryGetReshapeeAttributePtr(Expr e, Expr*& ep) {
      return tryGetAttributes<NNaryNodeOp>(e, [&](IPtr<NNaryNodeOp> np) { ep = &np->reshapee_; });
    }
  
    template<class NNaryNodeOp>
    static bool tryGetStepNodeAttributePtr(Expr e, Expr*& ep) {
      return tryGetAttributes<NNaryNodeOp>(e, [&](IPtr<NNaryNodeOp> np) { ep = &np->stepNode_; });
    }

    template<class NNaryNodeOp>
    static bool tryGetMaskAttributePtr(Expr e, Expr*& ep) {
      return tryGetAttributes<NNaryNodeOp>(e, [&](IPtr<NNaryNodeOp> np) { ep = &np->mask_; });
    }

    // call this for mandatory parameters, e.g. tryGetMaskAttributePtr(...) || tryFailed("message", ...)
    template<typename... Args>
    static bool fail(Args&&... args) {
      ABORT(std::forward<Args>(args)...);
    }
    static bool fail() { return fail("an attempt to access a Marian node attribute unexpectedly failed due to a type mismatch"); }
  };
  using E = SerializationHelpers;

  struct InputsMap : public std::map<Expr, Expr> {
    Expr operator()(Expr e) const {
      auto iter = find(e); // redirect input if found
      if (iter != end())
        e = iter->second;
      return e;
    }
  };

  // helper for rebuildNodesForward()
  static void addNodeAndChildren(Expr node, std::list<Expr>& nodesForward, std::set<Expr>& visited, const InputsMap& inputsMap)
  {
    // check if this is an input
    // In that case, we generate a replacement node instead, which has no children and thus terminates the recursion.
    // All nodes that reference this input are, however, unmodified.
    // The tape is now inconsistent. The consumer of this tape must perform child mapping.
    auto replacementNode = inputsMap(node);
    if (replacementNode != node)
      node = replacementNode;
    // recursion terminates if we already visited a node
    // (Input mapping is taken into account already.)
    auto res = visited.insert(node);
    if (!res.second) // already in visited set: done
      return;
    for (auto& child : node->children()) // children come before node itself
      addNodeAndChildren(child, nodesForward, visited, inputsMap);
    nodesForward.push_back(node);
  }

  // rebuild nodesForward_ from a graph given by its set of roots
  // Also replaces the inputs by constants, but does not redirect references (leaving an invalid tape--must be corrected on the fly by the caller!).
  void ExpressionGraphONNXExporter::rebuildNodesForward(const InputsMap& inputsMap,
                                                        const std::vector<std::pair<std::string, Expr>>& outputDefs) {
    nodesForward_.clear();
    std::set<Expr> visited;
    for (auto& outputDef : outputDefs)
      addNodeAndChildren(outputDef.second, nodesForward_, visited, inputsMap);
  }

  class NodeReferenceRedirector {
    std::map<Expr, Expr> nodeMap; // [orig node] -> replacement nodes

  public:
    void addRedirect(const Expr& whichNode, const Expr& withWhichNode) {
      nodeMap[whichNode] = withWhichNode;
    }

    // in-place redirect an Expr reference, i.e. look up the redirect and replace the original with it
    void redirectReference(Expr& child) const {
      auto iter = nodeMap.find(child);
      if (iter != nodeMap.end()) {
        child = iter->second;       // redirect child to the replacement node
        ABORT_IF(nodeMap.find(child) != nodeMap.end(), "Nested macro expansion??");
      }
    };

    // redirect all references (=children and more in special cases)
    void redirectAllReferencesIn(Expr v) const {
      // redirect all children
      auto& children = v->children(); // this is a mutable reference
      for (auto& child : children) {  // child is a mutable reference
        redirectReference(child);
      }
      // redirect additional references tat some nodes hold
      Expr* ep{};
      if (E::tryGetReshapeeAttributePtr<ReshapeNodeOp>   (v, ep) ||
          //E::tryGetStepNodeAttributePtr<StepNodeOp>      (v, ep) ||    // @TODO: review all of these and update the names
          E::tryGetMaskAttributePtr<PoolingWithMaskingOp>(v, ep)) {
        redirectReference(*ep);
      }
    }
  };

  static Expr newConstant(Expr v, Shape shape, float val, std::string suffix) {
    auto expr = v->graph()->constant(shape, inits::fromVector(std::vector<float>(shape.elements(), val)));
    expr->set_name("const_" + v->type() + "_" + std::to_string(v->getId()) + "_" + suffix);
    // Note: By convention, all constants should be named const_ something (and all data inputs data_),
    // to distinguish them from trainable weight tensors.
    return expr;
  }

  // unroll higher-level operations for which no ONNX equivalent exists
  // This updates the functionDefs' root nodes in-place.
  // Note: This appends to nodesForward_ in-place. Some meta-information, like root node, is not updated correctly.
  void ExpressionGraphONNXExporter::expandMacroOpsForONNX(std::map<std::string, std::pair<std::vector<std::pair<std::string, Expr>>, std::vector<std::pair<std::string, Expr>> >>& functionDefs) {
    LOG(info, "[graph] Expanding macro ops into primitives. Current graph size is {}", nodesForward_.size());
    NodeReferenceRedirector nodeReferenceRedirector;
    // clear memoization cache, as it removes some children for ops that have not changed since last inference
    tensors_->clearLongtermMemory();
    // Note: expansions will add to the existing tape in-place. But we disallows nested expansions,
    // i.e. disallow looping over newly created nodes, because otherwise the nodeReferenceRedirector
    // becomes very complicated because those new nodes are no longer topo-sorted.
    // The for loop below loops also over newly-created nodes, but those may not
    // trigger another expansion, which will be caught in redirectReference() above.
    auto beg = nodesForward_.begin();
    auto end = nodesForward_.end();
    for (auto vi = beg; vi != end; ++vi) {
      auto& v = *vi;
      // redirect all children of this node, in case they got mapped in this process
      nodeReferenceRedirector.redirectAllReferencesIn(v);
      // expand macro ops
      Expr n;
#if 0 // For GC ONNX, some ops are still missing. Map these first.
      // @BUGBUG: These operators are not up-to-date
      if (v->type() == "highway") {
        // Replace Sigmoid by Softmax. The only sigmoid in the system comes from highway.
        auto y = v->child(0);       // something like [B, H, T, dim]
        auto x = v->child(1);
        auto t = v->child(2);
        auto shape = x->shape();
        ABORT_IF(y->shape() != shape || t->shape() != shape, "unexpected highway shapes??");
        // Softmax([x,0]) = (Sigmoid(x), 1-Sigmoid(x))
        // Softmax([x,y]) = e^x / (e^x + e^y)
        // Sigmoid(x) = e^x / (e^x + e^0)
        auto shape1 = Shape{shape.elements() / shape.back(), shape.back(), 1};
        t = reshape(t, shape1);
        auto tAug = concatenate({t, newConstant(v, t->shape(), 0.0f, "zero_row")}, -1); // [(B*H*T, dim, 2)]
        auto s = softmax(tAug, /*axis=*/-1); // = (Sigmoid(t), 1-Sigmoid(t)) : [(B*H*T, dim, 2)]
        s = swapAxes(s, 0, -1);              // step() only supports axis=0
        auto sy = step(s, 0, /*axis=*/0);
        auto sx = step(s, 1, /*axis=*/0);
        sy = swapAxes(sy, 0, -1);
        sx = swapAxes(sx, 0, -1);
        sy = reshape(sy, shape);
        sx = reshape(sx, shape);
        n = sy * y + sx * x;
        //LOG(info, "OVERWRITING highway, {} -> {} -> {} -> back", std::string(shape), std::string(shape1), std::string(tAug->shape()));
      }
      else if (v->type() == "sum") {
        // replace ReduceSum by a matrix product with a vector of ones
        auto x = v->child(0);
        auto shape = x->shape();
        size_t lastAxis = shape.size() - 1;
        size_t axis;
        E::tryGetAxisAttribute<SumNodeOp>(v, axis) || E::fail();
        if (axis != lastAxis)   // bring axis to be reduced into last dimension so that we can MatMul
          x = swapAxes(x, (int)axis, (int)lastAxis);
        auto ones = newConstant(v, {x->shape().back(), 1}, 1.0f, "ones");
        n = dot(x, ones);       // [..., D] * [D, 1] = [..., 1]
        if (axis != lastAxis)   // and swap it back
          n = swapAxes(n, (int)axis, (int)lastAxis);
        //LOG(info, "OVERWRITING sum {}/{}, {} -> {} -> . -> {}", axis, lastAxis, std::string(shape), std::string(x->shape()), std::string(n->shape()));
      }
      else if (v->type() == "layer_normalization") {
        // layerNorm along last axis
        auto x = v->child(0);
        auto s = v->child(1);
        auto b = v->child(2);
        auto vecDim = x->shape().back();
        // for summing up elements, we use MatMul
        auto onesOverDim = newConstant(v, {vecDim, 1}, 1.0f / vecDim, "ones_over_dim");
        // compute mean and variance
        auto mean = dot(x, onesOverDim);
        auto x0 = x - mean;
        auto var = dot(x0 * x0, onesOverDim);
        // variance-normalize
        float epsilon;
        E::tryGetEpsilonAttribute<LayerNormalizationOp>(v, epsilon) || E::fail();
        auto sigma = sqrt(newConstant(v, {}, epsilon, "epsilon") + var);
        auto xnorm = x0 / sigma;
        // and final scale/bias
        n = xnorm * s + b;
        //LOG(info, "OVERWRITING layerNorm {} -> {}", std::string(x->shape()), std::string(mean->shape()));
      }
      else
#endif
      if (v->type() == "scalar_add") {
        float scalar{};
        E::tryGetScalarAttribute<ScalarAddNodeOp>(v, scalar) || E::fail();
        n = v->child(0) + newConstant(v, {}, scalar, "scalar");
      }
      else if (v->type() == "scalar_mult") {
        float scalar{};
        E::tryGetScalarAttribute<ScalarMultNodeOp>(v, scalar) || E::fail();
        n = v->child(0) * newConstant(v, {}, scalar, "scalar");
      }
      else if (v->type() == "square") {
        auto x = v->child(0);
        n = x * x;
      }
#if 0  // @BUGBUG: not supported for now, since we don't aim at training. This requires a function called select() which no longer exists.
      else if (v->type() == "x-ent") {
        auto x = v->child(0); // logits : some_shape + (num_classes,)
        auto y = v->child(1); // indices: some_shape + (1,)
        // C = sum_{v in V}(-logsoftmax(A) * delta(v, i) = -logsoftmax(A)[i]
        auto xShape = x->shape();
        // note: indices are flattened into a vector
        auto yShape = xShape;   // true shape of y -> result shape
        yShape.back() = 1;
        auto nl = logsoftmax(x);
        //nl->debug("nl");
#if 1   // ONNX has no batched select/gather, so we must fake it.
        // We first flatten the batch to a vector.
        nl = flatten(nl); // now x: (totalWords, vocabSize), while  y: (totalWords,)
        // Then we create a constant with offsets into this vector
        auto vocabSize = xShape.back();
        auto totalWords = xShape.elements() / vocabSize; // total batch size across batch and length dimension
        std::vector<unsigned int> offs;
        for (size_t i = 0; i < totalWords; i++)
          offs.push_back((unsigned int)(i * vocabSize));
        auto offsExpr = v->graph()->indices(offs);
        offsExpr->set_name("const_" + v->type() + "_offsets_" + std::to_string(v->getId()));
        // Now form indices into the flattened vector using the offsets
        y = y + offsExpr; // -> [y0, y1 + V, y2 + 2V, ...]
        // Now we can select with this.
        n = -select(nl, y, /*axis=*/-1);
        n = reshape(n, yShape);
        //LOG(info, "x-ent: {}, {} -> {}", std::string(x->shape()), std::string(y->shape()), std::string(n->shape()));
#else   // better version, but unfortunately neither Marian nor ONNX support batched select/gather
        y = reshape(y, yShape);
        n = -select(nl, y, /*axis=*/-1); // @TODO: update if we ever add axis_ to x-ent
#endif
      }
#endif
      else if (v->type() == "highway") {
        auto y = v->child(0);
        auto x = v->child(1);
        auto t = v->child(2);
        auto s = sigmoid(t);
        auto oneExpr = newConstant(v, {}, 1.0f, "one");
        n = s * y + (oneExpr - s) * x;
      }
      else if ( v->type() == "bdot" ||
               (v->type() == "dot"  /*  && (v->child(0)->shape().size() != 2 || v->child(1)->shape().size() != 2)*/) ||
               (v->type() == "affine" && (v->child(0)->shape().size() != 2 || v->child(1)->shape().size() != 2 || v->child(2)->shape().size() > 2))) {
        // ONNX MatMul behaves like Numpy matmul, and therefore implements batched semantics.
        // ONNX MatMul has no transA/B/scale parameters, so we must handle those as explicit operations.
        // affine() could also be ONNX Gemm, but that does not support outer ranks, so we just expand it into dot().
        // @TODO: ^^ we can just reshape(). Code is already below, but ONNX Gemm always crashes, so this is disabled for now.
        auto a = v->child(0);
        auto b = v->child(1);
        bool transA{}, transB{}; float scalar{}; // (gcc complains without the initializers, which I think is a compiler bug)
        E::tryGetMatMulAttributes<DotNodeOp>       (v, transA, transB, scalar) ||
        E::tryGetMatMulAttributes<DotBatchedNodeOp>(v, transA, transB, scalar) ||
        E::tryGetMatMulAttributes<AffineNodeOp>    (v, transA, transB, scalar) || E::fail();
        //LOG(info, "{} {}={}x{} trans = {}, {} and scalar = {}",
        //          v->type(), std::string(v->shape()), std::string(a->shape()), std::string(b->shape()), transA, transB, scalar);
        if (transA || transB || scalar != 1.0f ||
            (v->type() == "affine" && (a->shape().size() != 2 || b->shape().size() != 2 || v->child(2)->shape().size() > 2))) {
          //LOG(info, "patching {} {}={}x{} due to trans = {}, {} and scalar = {}",
          //          v->type(), std::string(v->shape()), std::string(a->shape()), std::string(b->shape()), transA, transB, scalar);
          if (transA) {  // note: we don't optimize for this since it does not happen in present models
            a = swapAxes(a, -1, -2);
            transA = false;
          }
          // @BUGBUG: Gemm always crashes with ONNX runtime. So we can't do this optimization.
          //if (v->type() != "bdot" && b->shape().size() == 2) {        // [A,B,C,I,J] x [J,K] --> reshape into regular matrix product
          //  ABORT_IF(transA, "Transposition not mapped away??");
          //  a = reshape(a, Shape({ a->shape().elements() / a->shape()[-1], a->shape()[-1] }));  // now it's a regular matrix product, can use Gemm
          //}
          /*else*/ if (transB) {  // not a regular matrix product: cannot use Gemm, so must transpose manually
            b = swapAxes(b, -1, -2);
            transB = false;
          }
          float extraScalar = 1.0f;
          if (v->type() == "bdot") {  // this maps to ONNX MatMul
            extraScalar = scalar;     // must add extra scale operation at the end
            scalar = 1.0f;            // we cannot scale in ONNX MatMul
            ABORT_IF(transA || transB || scalar != 1.0f, "Transposition and/or scalar not mapped away??");
            n = bdot(a, b, transA, transB, scalar);
          }
          else { // dot, affine
            // @BUGBUG: Gemm always crashes with ONNX runtime. So we can't do this optimization.
            //if (a->shape().size() != 2 || b->shape().size() != 2) {  // not ONNX MatMul: must use explicit scale operation
              extraScalar = scalar;
              scalar = 1.0f;
            //}
            n = dot(a, b, transA, transB, scalar);
            //LOG(info, "{} {} x {} -> {}", v->type(), std::string(a->shape()), std::string(b->shape()), std::string(n->shape()));
            if (v->type() == "affine")
              n = n + v->child(2);
          }
          //if (v->type() == "affine")
          //  LOG(info, "{} + {} -> {}", v->type(), std::string(v->child(2)->shape()), std::string(n->shape()));
          if (extraScalar != 1.0f)
            n = n * newConstant(v, {}, extraScalar, "scalar");
          if (n->shape() != v->shape())
            n = reshape(n, v->shape());  // if we did some shaping to get a regular matrix product, reshape it back
        }
      }
      else if (v->type() == "affine" && v->children().size() > 3) {
        // affine() may have a redundant vector of ones, which we strip here
        // This then becomes Gemm.
        v->children().resize(3);
        ABORT("affine() can presently not stripped of its additional ones vector. Need to fix Marian first to run with this.");
        // Note: Cannot recreate affine() as a new node, because that will get that fourth axis again.
        // @BUGBUG: This will crash.
      }
#if 0 // @BUGBUG: select() no longer exists. Likely some other ops are missing now.
      else if (v->type() == "select") {
        // select maps to Gather, and is limited to non-batched and the last axis
        size_t axis;
        E::tryGetAxisAttribute<SelectNodeOp>(v, axis) || E::fail();
        auto data    = v->child(0);
        auto indices = v->child(1);
        auto dataShape = data->shape();
        auto dataRank = dataShape.size();
        auto indicesShape = indices->shape();
        auto indicesRank = indicesShape.size();
        auto indicesDim = indicesShape[(int)axis - (int)dataShape.size()];
        ABORT_IF(indicesShape.elements() != indicesDim, "ONNX does not support batched select()");
        if (indicesRank != 1 || axis != dataRank - 1) {
          if (indicesRank != 1)
            indices = flatten(indices); // (batched Gather is not supported)
          if (axis != dataRank - 1)
            data = swapAxes(data, (int)axis, (int)dataRank - 1); // swap select axis to back
          n = select(data, indices, -1);
          if (axis != dataRank - 1)
            n = swapAxes(n, (int)axis, (int)dataRank - 1);
        }
      }
#endif
      else if (v->type() == "layer_normalization" &&
               (v->child(0)->shape().size() != 3 || v->child(1)->shape().size() != 1 || (v->children().size() > 2 && v->child(2)->shape().size() != 1))) {
        // ONNX InferenceNormalization is layer norm for shapes (N, C, D, ...) where N and C are
        // batch dimensions, and D... all share normalization statistics ("mean and variance are
        // computed per instance per channel").
        // Marian layer_normalization normalizes along axis -1.
        // Hence, if the input rank is != 3, we must temporarily reshape.
        // Also, ONNX expects scale and bias to contain C values (one for each c), while Marian
        // shares scale and bias along C but uses vectors of dim D. Hence, we must apply them manually.
        // This op gets replaced by a sequence that includes the same op, but with
        // gamma and beta being scalars, which is invalid for Marian.
        // (This will fail if layerNorm is applied to a scalar, which makes no sense.)
        auto x = v->child(0);
        auto s = v->child(1);
        auto b = v->children().size() > 2 ? v->child(2) : nullptr;  // beta is optional
        auto outShape = x->shape();
        auto vecDim = outShape[-1];
        x = reshape(x, {outShape.elements() / vecDim, 1, vecDim}); // -> (N, C, D)
        ABORT_IF((s->shape().size() > 1 && s->shape()[-1] != s->shape().elements()) ||
                 (b && b->shape().size() > 1 && b->shape()[-1] != b->shape().elements()),
                 "scale and bias must be vectors or single rows");
        s = flatten(s);
        if (b)
          b = flatten(b);
        //LOG(info, "layer_normalization reshaped from {} to {}", std::string(outShape), std::string(x->shape()));
        float epsilon;
        E::tryGetEpsilonAttribute<LayerNormalizationOp>(v, epsilon) || E::fail();
        //LOG(info, "LNORM {}, {}, {} vs. {}, {}", std::string(x->shape()), std::string(oneExpr->shape()), std::string(zeroExpr->shape()), std::string(s->shape()), std::string(b->shape()));
        n = layerNorm(x, newConstant(v, {1}, 1.0f, "one"), newConstant(v, {1}, 0.0f, "zero"), epsilon);
        n = n * s;
        if (b)
          n = n + b;
        n = reshape(n, outShape);
      }
      else if (v->type() == "const" && v->name().find("dropout_mask_") == 0) {
        // This is a randomly generated mask. We must replace this by RandomUniform.
        // This is done in 3 steps:
        //  - We expand v as (uniform < keepProb) * scale; but because Marian has no "<", we use "-" instead for now. @HACKHACK 1
        //  - The uniform for now is a constant, which later gets converted as ONNX RandomUniform(0,1).  @HACKHACK 2
        //  - The "-" with left arg of v gets patched to become ONNX Less. @HACKHACK 1 fix-up
        auto pString = v->name();
        pString.erase(0, pString.find_last_of('_') + 1);
        float dropProb = std::stof(pString);
        //LOG(info, "Found dropProb constant {} -> {}", v->name(), dropProb);
        float keepProb = 1.f - dropProb;
        float scale = 1.f / keepProb;
        auto uniformExpr = v->graph()->constant(v->shape(), inits::zeros());
        uniformExpr->set_name("opRandomUniform_" + std::to_string(v->getId())); // not using newConstant because of special node name
        // (uniform(0,1) < keepProb) * scale
        n = (uniformExpr - newConstant(v, {}, keepProb, "keepProb")) * newConstant(v, {}, scale, "scale");
        // @HACKHACK 1: Marian has no "less than", so we use "-" instead. Must patch that back later.
        // @HACKHACK 2: We use a specially-named constant as the placeholder for uniform(0,1).
      }

      if (n) {
        // copy key properties
        if (v->name() != n->name()) // (this tests for the empty name)
          n->set_name(v->name() + "_expanded"); // (this branch is actually never taken presently)
        n->setTrainable(v->trainable());
        // register mapping
        nodeReferenceRedirector.addRedirect(v, n);
        LOG(info, "[graph] Macro op {} expanded with new root op {}", v->type(), n->type());
      }
    }
    for (auto& functionDef : functionDefs) {
      for (auto& output : functionDef.second.second)  // redirect outputs: a root may also have been a macro op
        nodeReferenceRedirector.redirectReference(output.second);
      for (auto& output : functionDef.second.first)   // redirect inputs: inputs may be the outputs of other functions
        nodeReferenceRedirector.redirectReference(output.second);
    }

    // Since we added the expanded ops to the end of nodesForward_, we must bring it
    // back into topologically sorted order.
    LOG(info, "[graph] After creating expanded nodes, we now have {} nodes", nodesForward_.size());
  }

  using namespace onnx; // all -Proto classes come from here

  const std::string LENGTH_AXIS_NAME = "SOURCE_LENGTH";  // the source length is a named (dynamic) axis with this name

  // C++ port of a subset of https://github.com/onnx/onnx/blob/master/onnx/helper.py
  static ValueInfoProto makeValueInfoProto(std::string name, TensorProto_DataType dataType, std::vector<size_t> shape, size_t sentinelDim) {
    ValueInfoProto valueInfo;
    valueInfo.set_name(name);
    auto* valueInfoType = valueInfo.mutable_type();
    auto* valueInfoTensorType = valueInfoType->mutable_tensor_type();
    valueInfoTensorType->set_elem_type(dataType);
    auto* valueInfoTensorTypeShape = valueInfoTensorType->mutable_shape();
    for (auto dim : shape)
      if (dim == sentinelDim)
        valueInfoTensorTypeShape->add_dim()->set_dim_param(LENGTH_AXIS_NAME);
      else
        valueInfoTensorTypeShape->add_dim()->set_dim_value(dim);
    return valueInfo;
  }

  template<typename T> // note: for now, must pass the matching dataType (not checked)
  static TensorProto makeTensorProto(std::string name, TensorProto_DataType dataType, std::vector<size_t> shape, std::vector<T> vals) {
    TensorProto tensor;
    tensor.set_name(name);
    tensor.set_data_type(dataType);
    for (auto dim : shape)
      tensor.add_dims(dim);
#if 0   // @HACKHACK for debugging: keep files small during debugging, so that we can load and view those files easily
    *tensor.mutable_raw_data() = std::string((char*)vals.data(), (char*)(vals.data() + std::min(size_t(10), vals.size())));
#else
    *tensor.mutable_raw_data() = std::string((char*)vals.data(), (char*)(vals.data() + vals.size()));
#endif
    return tensor;
  }

  static inline void addAttribute(NodeProto& node, std::string name, std::vector<size_t> val) {
    AttributeProto* attribute = node.add_attribute();
    attribute->set_name(name);
    attribute->set_type(AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
    for (auto i : val)
      attribute->add_ints(i);
  }
  static inline void addAttribute(NodeProto& node, std::string name, std::vector<int> val) {
    AttributeProto* attribute = node.add_attribute();
    attribute->set_name(name);
    attribute->set_type(AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
    for (auto i : val)
      attribute->add_ints(i);
  }
  static inline void addAttribute(NodeProto& node, std::string name, std::string val) {
    AttributeProto* attribute = node.add_attribute();
    attribute->set_name(name);
    attribute->set_type(AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
    attribute->set_s(val);
  }
  static inline void addAttribute(NodeProto& node, std::string name, float val) {
    AttributeProto* attribute = node.add_attribute();
    attribute->set_name(name);
    attribute->set_type(AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT);
    attribute->set_f(val);
  }
  static inline void addAttribute(NodeProto& node, std::string name, int val) {
      AttributeProto* attribute = node.add_attribute();
      attribute->set_name(name);
      attribute->set_type(AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
      attribute->set_i(val);
  }
  static inline void addAttribute(NodeProto& node, std::string name, size_t val) {
      AttributeProto* attribute = node.add_attribute();
      attribute->set_name(name);
      attribute->set_type(AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
      attribute->set_i(val);
  }
  static inline void addAttribute(NodeProto& node, std::string name, bool val) {
    AttributeProto* attribute = node.add_attribute();
    attribute->set_name(name);
    attribute->set_type(AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
    attribute->set_i(val ? 1 : 0); // bool is stored as int in ONNX
  }
  static void addAttributes(NodeProto&) { // end of recursion
  }
  template<typename T, typename... Attributes>
  static void addAttributes(NodeProto& node, std::string name, T val, Attributes&&... moreAttributes) {
    addAttribute(node, name, val);
    addAttributes(node, std::forward<Attributes>(moreAttributes)...);
  }

  template <typename... Attributes>
  static NodeProto makeNode(std::string opType, std::string nodeName,
      std::vector<std::string> inputs, std::vector<std::string> outputs,
      Attributes&&... attributes) {
    NodeProto node;
    node.mutable_op_type()->assign(opType);
    for (auto input : inputs)
      node.add_input(input);
    for (auto output : outputs)
      node.add_output(output);
    if (!nodeName.empty())
      node.set_name(nodeName);
    addAttributes(node, std::forward<Attributes>(attributes)...);
    return node;
  }

  static GraphProto makeGraph(const std::vector<NodeProto>& nodes, std::string name,
                              const std::vector<ValueInfoProto>& inputs,
                              const std::vector<ValueInfoProto>& outputs,
                              const std::vector<TensorProto>& initializers,
                              const std::vector<ValueInfoProto>& valueInfos) {
    GraphProto graph;
    for (auto& node : nodes)
      *graph.add_node() = node;
    graph.set_name(name);
    for (auto& input : inputs)
      *graph.add_input() = input;
    for (auto& output : outputs)
      *graph.add_output() = output;
    for (auto& initializer: initializers)
      *graph.add_initializer() = initializer;
    for (auto& valueInfo : valueInfos)
#if 0 // add some as explicit outputs for debugging
      if (valueInfo.name() == "opReshape_292" || valueInfo.name() == "opPad_294")
        *graph.add_output() = valueInfo;
      else
#endif
      *graph.add_value_info() = valueInfo;
    valueInfos;
    return graph;
  }

  static ModelProto makeModel(const GraphProto& graph, std::string producerName) {
    ModelProto model;
    model.set_ir_version(IR_VERSION);
    model.set_producer_name(producerName);
    model.mutable_graph()->CopyFrom(graph);
#define OPSET_IMPORT_VERSION 11
    model.add_opset_import()->set_version(OPSET_IMPORT_VERSION);
    return model;
  }

  static std::string mapExprOp(Expr e) {
    const static std::map<std::string, std::string> opMap = {
      {"+"                      , "Add"},
      {"-"                      , "Sub"},
      {"*"                      , "Mul"},
      {"/"                      , "Div"},
      {"negate"                 , "Neg"},
      {"ReLU"                   , "Relu"},
      {"reshape"                , "Reshape"},
      {"affine"                 , "Gemm"},    // @TODO: is this just a hack, or meant to be used for this? It is not really standard GEMM semantics.
      {"bdot"                   , "MatMul"},
      {"dot"                    , "MatMul"},
      {"sigmoid"                , "Sigmoid"},
      {"sqrt"                   , "Sqrt"},
      {"sin"                    , "Sin"},
      {"cos"                    , "Cos"},
      {"tan"                    , "Tan"},
      {"layer_normalization"    , "InstanceNormalization"},
      {"softmax"                , "Softmax"},
      {"logsoftmax"             , "LogSoftmax"},
      {"sum"                    , "ReduceSum"},
      {"transpose"              , "Transpose"},
      {"concat"                 , "Concat"},
      {"sliceView"              , "Slice"},
      {"shift"                  , "Pad"},
      {"rows"                   , "Gather"},
      {"select"                 , "Gather"},
      // The following are never emitted to ONNX. Keep our original type names to avoid special-casing lots of code.
      {"const"                  , "const"},
      {"param"                  , "param"}
    };
    auto iter = opMap.find(e->type());
    ABORT_IF(iter == opMap.end(), "ONNX export of operation {} is presently not supported", e->type());
    return iter->second;
  }

  // get a unique name for an Expr. Either an actual name, or OP_ID if not named.
  // 'nameOverrides' overrides that name. This is used for inputs and outputs.
  static std::string getExprName(Expr e, const std::map<Expr, std::string>& nameOverrides) {
    if (nameOverrides.find(e) != nameOverrides.end())
      return nameOverrides.at(e);
    std::string name = e->name();
    if (name == "none") // Marian assigns "none" to denote an unassigned name
      name = (e->type() == "const" ? "" : "op") + mapExprOp(e) + "_" + std::to_string(e->getId());
    // For 'const', do not prefix "op", so that all internal constants in the system
    // (i.e. not input data) have a prefix "const_" to distinguish them from weight tensors.
    return name;
  }

  // convert Marian shape into vector<size_t>
  static std::vector<size_t> getExprShape(Expr e) {
    const auto& shape = e->shape();
    return std::vector<size_t>(shape.begin(), shape.end());
  }

  // get TensorProto_DataType for an Expr
  // Note: We map Marian uint32_t to ONNX signed integers because those are only used
  // for indices for Gather operations, where Marian requires unsigned and ONNX signed.
  static TensorProto_DataType getExprDataType(Expr expr) {
    switch (expr->value_type()) {
      case marian::Type::float32: return TensorProto_DataType::TensorProto_DataType_FLOAT;
      //case marian::Type::uint32:  //return TensorProto_DataType::TensorProto_DataType_UINT32;
      case marian::Type::uint32:  // uint32 becomes ONNX INT32 as well (see above)
      case marian::Type::int32:   return TensorProto_DataType::TensorProto_DataType_INT32;
      default: ABORT("Tensor type not supported yet");
    }
  }

  // convert a Marian constant to an ONNX TensorProto
  static TensorProto makeExprTensorProto(Expr expr, const std::map<Expr, std::string>& nameOverrides) {
    auto dataType = getExprDataType(expr);
    auto name     = getExprName    (expr, nameOverrides);
    auto shape    = getExprShape   (expr);
    switch(expr->value_type()) {
    case marian::Type::float32: { // @TODO: template this?
      std::vector<float> valBuf;
      expr->val()->get(valBuf);
      return makeTensorProto(name, dataType, shape, valBuf);
    }
    case marian::Type::uint32: {
      std::vector<uint32_t> valBuf; // note: uint32_t still get passed to ONNX as signed INT32 (cf. getExprDataType())
      expr->val()->get(valBuf);
      return makeTensorProto(name, dataType, shape, valBuf);
    }
    case marian::Type::int32: {
      std::vector<int32_t> valBuf;
      expr->val()->get(valBuf);
      return makeTensorProto(name, dataType, shape, valBuf);
    }
    default:
      ABORT("Tensor type not supported yet");
    }
  }

  static void logNode(const NodeProto& node, const std::vector<size_t>& shape, size_t sentinelDim) {
    std::string s = node.name() + " = " + node.op_type() + "(";
    auto addComma = [&]() { if (s.back() != '(' && s.back() != '[') s += ", "; };
    for (int i = 0; i < node.input_size(); i++) {
      auto inputName = node.input(i);
      addComma();
      s += inputName;
    }
    for (int i = 0; i < node.attribute_size(); i++) {
      auto attribute = node.attribute(i);
      addComma();
      s += attribute.name() + "=?";
    }
    s += (") : [");
    for (auto dim : shape) {
      addComma();
      if (dim == sentinelDim)
          s += LENGTH_AXIS_NAME;
      else
          s += std::to_string(dim);
    }
    s.push_back(']');
    LOG(info, s);
  }

  // convert a Marian Expr to an ONNX node
  // This function needs inputs and initializers because the special case of Reshape needs
  // to create an extra input with initializer.
  static void addExprNode(Expr expr, std::vector<NodeProto>& nodes, std::vector<ValueInfoProto>& inputs,
                          std::vector<TensorProto>& initializers,
                          const std::map<Expr, std::string>& nameOverrides, const InputsMap& inputsMap,
                          size_t sentinelDim) {
    // get all children
    // These may reference inputs, and hence must be mapped right here.
    // The original child in this case is not on the tape.
    auto children = expr->children();
    for (auto& child : children)
      child = inputsMap(child);

    // inputs are referenced by their node names (also when they are leaves)
    std::vector<std::string> inputNames;
    for (const auto& child : children)
      inputNames.push_back(getExprName(child, nameOverrides));

    auto name = getExprName(expr, nameOverrides); // node name is used as both output name and node name
    auto op = mapExprOp(expr);

    //if (op == "MatMul" && expr->child(0)->shape().size() == 2 && expr->child(1)->shape().size() == 2) {
    //  op = "Gemm";
    //}

#if 1 // workaround for onnxruntime which does not handle Pad correctly
    if (op == "Pad") {
      // Implement Pad as Slice >> Concat
      std::vector<int> shifts;
      float padValue{}; // (compiler bug: without initialization, I get an uninit warning, yet it is correctly set)
      E::tryGetShiftAttributes<ShiftNodeOp>(expr, shifts, padValue) || E::fail();
      ABORT_IF(shifts[0] != 1, "can only shift by one");
      for (size_t i = 1; i < shifts.size(); i++)
        ABORT_IF(shifts[i] != 0, "can only shift along first axis");
      auto shape = getExprShape(children[0]);
      // Slice [0:-1,:,:]
      auto sliceName = name + "_Slice";
      auto sliceNode = makeNode("Slice", sliceName, inputNames, {sliceName});
      addAttribute(sliceNode, "axes",   std::vector<size_t>{0});
      addAttribute(sliceNode, "starts", std::vector<size_t>{0});
      addAttribute(sliceNode, "ends",   std::vector<size_t>{shape[0] - 1}); // drop last step
      nodes.push_back(sliceNode);
      LOG(info, "Pad slice op {}", sliceName);
      // create a padding constant
      auto paddingName = "const_" + name + "_Padding";
      shape[0] = 1;
      size_t n = 1;
      for (auto& dim : shape)
        n *= dim;
      std::vector<float> zeros(n);
      inputs.      push_back(makeValueInfoProto(paddingName, TensorProto_DataType::TensorProto_DataType_FLOAT, shape, sentinelDim));
      initializers.push_back(makeTensorProto   (paddingName, TensorProto_DataType::TensorProto_DataType_FLOAT, shape, zeros));
      LOG(info, "Pad constant {}", paddingName);
      // Concat([paddingNode, sliceNode], axis=0)
      auto node = makeNode("Concat", name, {paddingName, sliceName}, {name});
      addAttribute(node, "axis", 0);
      nodes.push_back(node);
      LOG(info, "Pad concat op {}", name);
      return;
    }
#endif

    auto node = makeNode(op, name, inputNames, {name});
    //LOG(info, "NODE {} {} -> {}", name, expr->type(), E::mapExprOp(expr));

    // add attributes needed by some operators

    // fix up inputs
    if (node.op_type() == "Reshape") { // Reshape requires the shape itself to be a tensor.
      auto shapeInputName = "const_" + getExprName(expr, {}) + "_shape_attr";
      *node.add_input() = shapeInputName;
      // create a new input and a new initializer
      auto shape = getExprShape(expr);
      auto shape64 = std::vector<int64_t>(shape.begin(), shape.end());
      for (auto& dim : shape64)
        if (dim == (int64_t)sentinelDim)
          dim = -1;  // means that this one is inferred at runtime
      std::vector<size_t> shapeShape{shape.size()}; // ONNX Reshape requires shape in INT64
      inputs.      push_back(makeValueInfoProto(shapeInputName, TensorProto_DataType::TensorProto_DataType_INT64, shapeShape, sentinelDim));
      initializers.push_back(makeTensorProto   (shapeInputName, TensorProto_DataType::TensorProto_DataType_INT64, shapeShape, shape64));
      std::string s = shapeInputName;
      for (auto& dim : shape64)
        s += " " + std::to_string(dim);
      LOG(info, s);
    }
    // axis attribute
    size_t axis{};
    std::vector<size_t> axes;
    if (E::tryGetAxisAttribute<ConcatenateNodeOp>(expr, axis)// ||
        //E::tryGetAxisAttribute<SelectNodeOp>(expr, axis)
        ) { // axis_ -> 'axis'
      addAttribute(node, "axis", axis);
    }
    else if (E::tryGetAxisAttribute<ReduceNodeOp>(expr, axis) ||
             E::tryGetAxisAttribute<SliceViewNodeOp>(expr, axis)) { // {axis_} -> 'axes'
      addAttribute(node, "axes", std::vector<size_t>{axis});
    }
    else if (E::tryGetAxesAttribute<TransposeNodeOp>(expr, axes)) { // here, the axes are called 'perm'
      addAttribute(node, "perm", axes);
    }
    else if (node.op_type() == "Softmax" || node.op_type() == "LogSoftmax") {
      // Note: ONNX (Log)Softmax is not along an axis; rather along all axes >= given axis (they get flattened).
      addAttribute(node, "axis", expr->shape().size()-1); // Marian softmax defaults to last axis. @TODO: update if we ever add an axis_ parameter.
    }
    else if (expr->type() == "rows") { // becomes Gather
      // Example, adopted from ONNX docs:
      //  axis = 0
      //  data = [ [1.0, 1.2], [2.3, 3.4], [4.5, 5.7], ]
      //  indices = [ 0, 1, 1, 2, ]
      //  output = [  [1.0, 1.2], [2.3, 3.4], [2.3, 3.4], [4.5, 5.7], ] 
      ABORT_IF(expr->shape().size() != 2, "Unexpected input shape for rows()");
      addAttribute(node, "axis", 0);
    }
    // slice attributes (starts, ends)
    Slice slice;
    if (E::tryGetSliceAttribute<SliceViewNodeOp>(expr, slice)) {
      addAttribute(node, "starts", std::vector<size_t>{(size_t)slice.begin});
      addAttribute(node, "ends"  , std::vector<size_t>{(size_t)slice.end});
      addAttribute(node, "steps" , std::vector<size_t>{(size_t)slice.stride});
    }
    // shift attributes (shift, padValue)
    std::vector<int> shifts;
    float padValue{}; // (compiler bug: without initialization, I get an uninit warning, yet it is correctly set)
    if (E::tryGetShiftAttributes<ShiftNodeOp>(expr, shifts, padValue)) {
      std::vector<int> pads;
      for (auto shift : shifts)
        pads.push_back(shift);   // shift = #padValues to insert at front (or, for, shift < 0, to remove at front)
      for (auto shift : shifts)
        pads.push_back(-shift);  // and #values to remove at end (or, for, shift < 0, to insert at end)
      ABORT_IF(pads.size() != 2 * expr->shape().size(), "Unexpected number of shift dimensions");
      addAttribute(node, "pads", pads);
      addAttribute(node, "value", padValue);
      addAttribute(node, "mode", std::string("constant"));
    }

    // matmul attributes
    bool transA, transB;
    float scalar;
    // @BUGBUG: I cannot get Gemm to work, ONNX runtime always crashes. So we will NEVER get here.
    if (node.op_type() == "Gemm") {  // we get here for affine() or dot()
      // Note: We only get here if Gemm can implement this configuration.
      ABORT_IF(children[0]->shape().size() != 2 || children[1]->shape().size() != 2 ||
               (children.size() > 2 && children[2]->shape().size() > 2),
               "Gemm unexpectedly used for non-matrix inputs");
      E::tryGetMatMulAttributes<AffineNodeOp>(expr, transA, transB, scalar) || 
      E::tryGetMatMulAttributes<DotNodeOp>   (expr, transA, transB, scalar) || E::fail();
      /*if (transA)        */ addAttribute(node, "transA", transA ? 1 : 0);
      /*if (transB)        */ addAttribute(node, "transB", transB ? 1 : 0);
      /*if (scalar != 1.0f)*/ addAttribute(node, "alpha", scalar);
      //addAttribute(node, "beta", 0.0f);
    }
    else if (E::tryGetMatMulAttributes<DotNodeOp>       (expr, transA, transB, scalar) ||
             E::tryGetMatMulAttributes<DotBatchedNodeOp>(expr, transA, transB, scalar)) {
      // transpose/scalar not supported by ONNX MatMul, must have been expanded before we get here
      ABORT_IF(transA || transB || scalar != 1.0f, "Unexpected transpose or scalar attributes for {}", expr->type());
    }
    // epsilon attribute
    float epsilon;
    if (E::tryGetEpsilonAttribute<LayerNormalizationOp>(expr, epsilon)) {
      addAttribute(node, "epsilon", epsilon);
    }
    // dropout patches
    if (node.op_type() == "Sub" && children[0]->type() == "const" && children[0]->name().find("opRandomUniform_") == 0) {
      // @HACKHACK 1: For dropout, we route a "<" operation through a Marian "-" because it has no "<".
      *node.mutable_op_type() = "Less";
      // Note: Since this is a hack, we don't bother to fix up the node name, which is still opSub_ID.
    }
    else if (expr->type() == "const" && expr->name().find("opRandomUniform_") == 0) {
      // @HACKHACK 2: The dropout weight, which is a 'const' in Marian, acts as a placeholder for
      // a RandomUniform operation. In place of a 'const', we generate a uniform(0,1) node
      // of the same shape.
      *node.mutable_op_type() = "RandomUniform";
      addAttribute(node, "shape", getExprShape(expr));
    }
    nodes.push_back(node);
  }

  // serialize the nodesForward_ of a graph right after build() into an ONNX-formatted file
  // We declare this to be ONNX operator set 9. @TODO: Which ONNX version does this correspond to?
  // The nodes must only contain operations supported by ONNX, so the caller must first call
  // expandMacroOpsForONNX().
  // One batch axis can be variable-length. It is recognized via a hack: by a special
  // dimension value that otherwise never naturally occurs, e.g. a larger prime number.
  // We will not recognize derivates of this value, such as value+1 or value x another dimension.
  // @TODO: This presently does not support variable batch dimensions. How does ONNX handle them?
  // @TODO: How to handle guided alignment? That's another input. Name? Shape?
  // This is based on the simple example in
  // https://github.com/onnx/onnx/blob/master/onnx/examples/make_model.ipynb
  void ExpressionGraphONNXExporter::serializeToONNX(const std::string& fileRoot, FunctionDefs&& functionDefs, size_t sentinelDim) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // @TODO: expansion must deal with multiple sub-tapes (encoder, init)
    // expand Marian macro operations such as "highway" or "scalar_add" that ONNX does not have
    // After this, nodesForward_ is not topologically sorted.
    expandMacroOpsForONNX(functionDefs);

    for (const auto& functionDef : functionDefs) {
      const auto& graphName = functionDef.first;
      const auto& inputDefs  = functionDef.second.first;
      const auto& outputDefs = functionDef.second.second;

      // some stats
      LOG(info, "[onnx] Exporting graph {}", graphName);

      std::map<Expr, std::string> nameOverrides; // we implant input and output names dynamically (instead of setting the name in Expr)

      // clear memoization caches
      tensors_->clearShorttermMemory();
      tensors_->clearLongtermMemory();

      // create new dummy const nodes for all function arguments
      // These nodes will be replaced in rebuildNodesForward() and act as recursion stops.
      // The actual child references are NOT replaced.
      // Also, we collect the nameOverrides for all input and output nodes.
      InputsMap inputsMap;
      for (auto& inputDef : inputDefs) {
        const auto& input = inputDef.second;
        ABORT_IF(inputsMap.find(input) != inputsMap.end(), "Duplicate inputDef expr??");
        auto arg = constant(input->shape(), inits::zeros(), input->value_type());
        inputsMap[input] = arg;
        nameOverrides[arg] = inputDef.first;
      }
      for (const auto& outputDef : outputDefs)
          nameOverrides[inputsMap(outputDef.second)] = outputDef.first;

      // regenerate nodesForward_ from the roots, only for the function under consideration
      // This redirects all items in inputsMap in the graph and in outputDefs as well.
      // I.e. actual inputs are already replaced by Constants on the tape, but other nodes'
      // references are not!
      // All references from this point on have to be run through inputsMap().
      rebuildNodesForward(inputsMap, outputDefs);
      LOG(info, "[graph] Topologically sorted, garbage-collected graph has size {}", nodesForward_.size());

      // sanity check: is the tape consistent, assuming the inputsMap?
      std::set<Expr> nodesOnTape;
      for (const auto& e : nodesForward_)
        nodesOnTape.insert(e);
      for (const auto& e : nodesForward_) for (const auto& c : e->children()) {
        if (nodesOnTape.find(c) == nodesOnTape.end())
          LOG(info, "Redirected child: {}, {}", c->getId(), c->name());
        ABORT_IF(nodesOnTape.find(inputsMap(c)) == nodesOnTape.end(),
                 "Node {} {} refers to child {} {} that is off tape??", e->getId(), e->name(), c->getId(), c->name());
      }

      // sanity check: did we consume all expected inputs?
      std::set<Expr> mappedInputSet;  // set of replacement Exprs (those constants) for inputs
      for (auto ee : inputsMap)
        mappedInputSet.insert(ee.second);
      std::set<Expr> seenMappedInputs;
      for (const auto& expr : nodesForward_) {
        ABORT_IF(inputsMap.find(expr) != inputsMap.end(), "An input node (id={}) was not mapped??", expr->getId());
        if (mappedInputSet.find(expr) != mappedInputSet.end())
          seenMappedInputs.insert(expr);
      }
      for (auto e : mappedInputSet)
        if (seenMappedInputs.find(e) == seenMappedInputs.end()) {
          LOG(info, "WARNING: Input {} not consumed in input graph", nameOverrides[e]);
          nodesForward_.push_back(e);
        }
        //ABORT_IF(seenMappedInputs.find(e) == seenMappedInputs.end(), "Input node {} not found in input graph??", nameOverrides[e]);

      // output set -- these nodes are exported differently
      std::set<Expr> outputsSet;
      for (const auto& outputDef : outputDefs)
        outputsSet.insert(inputsMap(outputDef.second));

      std::vector<ValueInfoProto> inputsParamsAndConstants; // parameters and constants all are considered inputs, just with initializers

      // Create a the nodes -> array of NodeProto
      std::vector<NodeProto> nodes;
      std::vector<TensorProto> initializers; // constants are inputs with initializers that hold their values. They go here.
      std::vector<ValueInfoProto> shapeInfos; // expected shapes of operations (for diagnostics only)
      std::vector<ValueInfoProto> outputs; // outputs' shapes
      for(const auto& expr : nodesForward_) {
        //LOG(info, "exporting node name {} op {} ({})", getExprName(expr), E::mapExprOp(expr), expr->children().size());
        if (expr->type() == "param" ||
            (expr->type() == "const" && expr->name().find("opRandomUniform_") != 0)) { // leaves are not nodes in ONNX (except for the uniform placeholder @HACKHACK 2)
          //LOG(info, "exporting leaf name {} op {} ({})", getExprName(expr), E::mapExprOp(expr), expr->children().size());
          auto shape = getExprShape(expr);
          inputsParamsAndConstants.push_back(makeValueInfoProto(getExprName(expr, nameOverrides), getExprDataType(expr), shape, sentinelDim));
          // don't create an initializers entry for inputs
          if (std::any_of(inputsMap.begin(), inputsMap.end(), [&](const std::pair<Expr, Expr>& inputMap) {
                return inputMap.second == expr;
              })) { // skip designated inputs
            ABORT_IF(expr->type() != "const", "Data inputs must be 'const' nodes");
            //LOG(info, "No initializer for data-input node {}", getExprName(expr));
            continue;
          }
          // run initializers, to realize value of consts (params already got theirs)
          expr->allocate();
          expr->init();
          expr->forward();
          ABORT_IF(!expr->val(), "Leaf '{}' of type {} unexpectedly lacks a value despite trying really hard", expr->name(), expr->type());
          initializers.push_back(makeExprTensorProto(expr, nameOverrides));
          continue;      // parameters must become initializers, name=input name
        }
        addExprNode(expr, nodes, inputsParamsAndConstants, initializers, nameOverrides, inputsMap, sentinelDim);
        logNode(nodes.back(), getExprShape(expr), sentinelDim);

        auto valueInfo = makeValueInfoProto(nodes.back().name(), getExprDataType(expr), getExprShape(expr), sentinelDim);
        if (outputsSet.find(expr) != outputsSet.end())
          outputs.push_back(valueInfo);
        //else // we add expected-shape information, to more easily be able to track down where it may fail
        //  shapeInfos.push_back(valueInfo);
      }

      //LOG(info, "total nodes: {}, incl. {} inputs, {} op shapes", nodesForward_.size(), inputs.size(), shapeInfos.size());

      // @TODO: write a log message with the inputs and output names (the function signature)

      // Create the graph -> GraphProto
      auto graphDef = makeGraph(nodes, graphName, inputsParamsAndConstants, outputs, initializers, shapeInfos);

      // Create the model -> ModelProto
      auto modelDef = makeModel(graphDef, /*producer_name=*/"Marian " + buildVersion());

      // save it
      auto filename = fileRoot + "." + graphName + ".onnx";
      auto s = modelDef.SerializeAsString();
      ABORT_IF(s.empty(), "Failed to serialize ONNX graph to string buffer", filename);
      std::ofstream o(filename, std::ios::binary);
      ABORT_IF(o.fail(), "Failed to create ONNX model file {}", filename);
      o.write(s.data(), s.size());
      o.close();
      ABORT_IF(o.fail(), "Failed to write ONNX model to {}", filename);
      LOG(info, "[onnx] ONNX graph '{}' written to {}", graphName, filename);
    }

    // tape has been destroyed many times, so clear it for good
    nodesForward_.clear();
  }

  Expr ExpressionGraphONNXExporter::tryFindForwardNodeByName(const std::string& nodeName) const {
    auto iter = std::find_if(nodesForward_.begin(), nodesForward_.end(), [&](Expr node) {return node->name() == nodeName; });
    if (iter == nodesForward_.end())
      return nullptr;
    else
      return *iter;
  }

}  // namespace marian

#endif // USE_ONNX

