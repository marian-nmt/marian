/* All or part of this file was contributed by NVIDIA under license:
 *   Copyright (C) 2020 NVIDIA Corporation
 *   SPDX-License-Identifier: MIT
 */
#include "catch.hpp"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"

#ifdef CUDA_FOUND
#include "tensors/gpu/backend.h"
#endif

#include <cmath>

using namespace marian;

template <typename T>
void tests(DeviceType device, Type floatType = Type::float32) {

// Checking for FP16 support and skipping if not supported.
#ifdef CUDA_FOUND
  if(device == DeviceType::gpu && floatType == Type::float16) {
    auto gpuBackend = New<gpu::Backend>(DeviceId({0, device}), /*seed=*/1234);
    auto cudaCompute = gpuBackend->getCudaComputeCapability();
    if(cudaCompute.major < 6) return;
  }
#endif

  auto floatApprox = [](T x, T y) -> bool { return x == Approx(y).margin(0.001f); };
  auto floatApprox2 = [](T x, T y) -> bool { return x == Approx(y).margin(0.01f); };
  auto floatEqual  = [](T x, T y) -> bool { return x == y; };

  Config::seed = 1234;
  auto graph = New<ExpressionGraph>();
  
  graph->setInference(true);
  graph->setDefaultElementType(floatType);
  graph->setDevice({0, device});
  graph->reserveWorkspaceMB(16);

  std::vector<T> values, values2;

  SECTION("elementwise unary and binary operators with scalars") {
    graph->clear();
    values.clear();

    std::vector<T> vA({1, -2, 3, -4});
    auto a = graph->constant({2, 2}, inits::fromVector(vA));

    auto compare = [&](Expr res, std::function<float(float)> f) -> bool {
      if (res->shape() != Shape({ 2, 2 }))
          return false;
      res->val()->get(values);
      std::vector<float> ref{f(vA[0]), f(vA[1]), f(vA[2]), f(vA[3])};
      return std::equal(values.begin(), values.end(), ref.begin(), floatEqual);
    };

    // @TODO: add all operators and scalar variants here for completeness
    auto rsmult = 2.f * a;
    auto rabs   = abs(a);
    auto rmax1  = maximum(a, 1);
    auto rmax2  = maximum(1, a);
    auto rmin1  = minimum(a, 1);
    auto rmin2  = minimum(1, a);

    graph->forward();

    CHECK(compare(rsmult, [](float a) {return 2.f * a;}));
    CHECK(compare(rabs,   [](float a) {return std::abs(a);}));
    CHECK(compare(rmax1,  [](float a) {return std::max(a, 1.f);}));
    CHECK(compare(rmax2,  [](float a) {return std::max(1.f, a);}));
    CHECK(compare(rmin1,  [](float a) {return std::min(a, 1.f);}));
    CHECK(compare(rmin2,  [](float a) {return std::min(1.f, a);}));
  }

  SECTION("Scalar reductions <= 32 Elements") {
    graph->clear();
    values.clear();

    std::vector<T> maxInp({-1, -2, -3, -4});
    std::vector<T> minInp({4, 100, 42, 420, 3, 14, 15, 926, 53});
    std::vector<T> prodInp({5, -1, 3, 2, 3, -1, 4, 2, 1});
    std::vector<T> sumInp({4, 4, 8, 16, 32, 4, 5, 10});

    std::vector<T> genericInp({8, 8, 16});

    auto maxInpExpr = graph->constant({1, 1, (int)maxInp.size()}, inits::fromVector(maxInp));
    auto minInpExpr = graph->constant({1, 1, (int)minInp.size()}, inits::fromVector(minInp));
    auto prodInpExpr = graph->constant({1, 1, (int)prodInp.size()}, inits::fromVector(prodInp));
    auto sumInpExpr = graph->constant({1, 1, (int)sumInp.size()}, inits::fromVector(sumInp));
    auto genericInpExpr = graph->constant({1, 1, (int)genericInp.size()}, inits::fromVector(genericInp));

    auto compare = [&](Expr res, std::vector<T> inp, std::function<float(float, float)> op) -> bool {
      if (res->shape().elements() != 1)
          return false;
      float val = res->val()->get(0);
      
      float reduced = inp[0];
      for(int i = 1; i < inp.size(); ++i) {
        reduced = op(reduced, inp[i]);
      }
      return floatApprox(reduced, val);
    };

    // @TODO: add all operators here for completeness
    auto maxReduce = max(maxInpExpr, /*axis*/ -1);
    auto minReduce = min(minInpExpr, /*axis*/ -1);
    auto prodReduce = prod(prodInpExpr, /*axis*/ -1);
    auto sumReduce = sum(sumInpExpr, /*axis*/ -1);
    auto meanReduce = mean(genericInpExpr, /*axis*/ -1);
    auto logSumExpReduce = logsumexp(genericInpExpr, /*axis*/ -1);

    graph->forward();
    
    // All values are computed using numpy 1.19.2 with Python 3.6.9
    constexpr float expectedMean = 10.66666; // np.mean(genericInp)
    constexpr float expectedLogSumExp = 16.000670700286076; // np.log(np.sum(np.exp(genericInp)))

    CHECK(compare(maxReduce, maxInp, [](float a, float b) {return std::max(a, b);}));
    CHECK(compare(minReduce, minInp, [](float a, float b) {return std::min(a, b);}));
    CHECK(compare(prodReduce, prodInp, [](float a, float b) {return a * b;}));
    CHECK(compare(sumReduce, sumInp, [](float a, float b) {return a + b;}));
    CHECK(floatApprox(expectedMean, meanReduce->val()->get(0)));
    CHECK(floatApprox(expectedLogSumExp, logSumExpReduce->val()->get(0)));
  }

  SECTION("elementwise binary operators with broadcasting") {
    graph->clear();
    values.clear();

    std::vector<T> vA({1, -2, 3, -4});
    std::vector<T> vB({0.5, 1.5});

    auto a = graph->constant({2, 2}, inits::fromVector(vA));
    auto b = graph->constant({2}, inits::fromVector(vB));

    // Two lambdas below differ in the use of floatEqual or floatApprox and
    // are not merged because MSVC compiler returns C2446: no conversion from
    // lambda_x to lambda_y
    auto compare = [&](Expr res, std::function<float(float,float)> f) -> bool {
      if (res->shape() != Shape({ 2, 2 }))
          return false;
      res->val()->get(values);
      std::vector<float> ref{f(vA[0], vB[0]), f(vA[1], vB[1]), f(vA[2], vB[0]), f(vA[3], vB[1])};
      return std::equal(values.begin(), values.end(), ref.begin(), floatEqual);
    };

    auto compareApprox = [&](Expr res, std::function<float(float, float)> f) -> bool {
      if(res->shape() != Shape({2, 2}))
        return false;
      res->val()->get(values);
      std::vector<float> ref{f(vA[0], vB[0]), f(vA[1], vB[1]), f(vA[2], vB[0]), f(vA[3], vB[1])};
      return std::equal(values.begin(), values.end(), ref.begin(), floatApprox);
    };

    auto rplus  = a + b;
    auto rminus = a - b;
    auto rmult  = a * b;
    auto rdiv   = a / b;
    auto rlae   = logaddexp(a, b);
    auto rmax   = maximum(a, b);
    auto rmin   = minimum(a, b);
    auto rlt    = lt(a, b);
    auto req    = eq(a, b);
    auto rgt    = gt(a, b);
    auto rge    = ge(a, b);
    auto rne    = ne(a, b);
    auto rle    = le(a, b);

    graph->forward();

    CHECK(compare(rplus,  [](float a, float b) {return a + b;}));
    CHECK(compare(rminus, [](float a, float b) {return a - b;}));
    CHECK(compare(rmult,  [](float a, float b) {return a * b;}));
    CHECK(compareApprox(rdiv,   [](float a, float b) {return a / b;}));
    CHECK(compareApprox(rlae,   [](float a, float b) {return logf(expf(a) + expf(b));}));
    CHECK(compare(rmax,   [](float a, float b) {return std::max(a, b);}));
    CHECK(compare(rmin,   [](float a, float b) {return std::min(a, b);}));
    CHECK(compare(rlt,    [](float a, float b) {return a <  b;}));
    CHECK(compare(req,    [](float a, float b) {return a == b;}));
    CHECK(compare(rgt,    [](float a, float b) {return a >  b;}));
    CHECK(compare(rge,    [](float a, float b) {return a >= b;}));
    CHECK(compare(rne,    [](float a, float b) {return a != b;}));
    CHECK(compare(rle,    [](float a, float b) {return a <= b;}));
  }

  SECTION("transposing and reshaping") {
    graph->clear();
    values.clear();

    std::vector<T> vA({1, 2, 3, 4, 5, 6, 7, 8});

    std::vector<T> vT1({1, 5, 2, 6, 3, 7, 4, 8});
    std::vector<T> vT3({1, 2, 5, 6, 3, 4, 7, 8});
    std::vector<T> vT4({1, 5, 3, 7, 2, 6, 4, 8});
    std::vector<T> vT5({1, 2, 5, 6, 3, 4, 7, 8});

    auto a = graph->constant({2, 4}, inits::fromVector(vA));

    auto t1 = transpose(a);
    auto t2 = transpose(t1);
    auto t3 = transpose(reshape(t1, {2, 2, 2}));

    auto t4 = transpose(reshape(a, {2, 1, 2, 2}), {1, 3, 2, 0});
    auto t5 = transpose(reshape(a, {2, 1, 2, 2}), {2, 0, 1, 3});

    auto t6 = stopGradient(a);

    graph->forward();

    CHECK(t1->shape() == Shape({4, 2}));
    CHECK(t2->shape() == Shape({2, 4}));
    CHECK(t3->shape() == Shape({2, 2, 2}));
    CHECK(t4->shape() == Shape({1, 2, 2, 2}));
    CHECK(t5->shape() == Shape({2, 2, 1, 2}));
    CHECK(t6->shape() == a->shape());

    t1->val()->get(values);
    CHECK( values == vT1 );

    t2->val()->get(values);
    CHECK( values == vA );

    t3->val()->get(values);
    CHECK( values == vT3 );

    t4->val()->get(values);
    CHECK( values == vT4 );

    t5->val()->get(values);
    CHECK( values == vT5 );

    t6->val()->get(values);
    CHECK(values == vA);
    CHECK(!t6->trainable());
  }

  SECTION("softmax and logsoftmax") {
    graph->clear();
    values.clear();
    std::vector<T> in({-.2, -.3, 4.5, 5.2, -10, 101.45, -100.05, 1.05e-5});

    std::vector<T> smOut({ 0.52498f, 0.47502f, 0.33181f, 0.66819f,
                               0.0f, 1.0f, 0.0f, 1.0f });

    std::vector<T> lsmOut({ -0.6444f, -0.7444f, -1.10319f, -0.40319f,
                                -111.45f, 0.0f, -100.05001f, 0.0f });

    auto input = graph->constant({2, 2, 2}, inits::fromVector(in));

    auto sm  = softmax(input);
    auto lsm = logsoftmax(input);

    graph->forward();

    CHECK(sm->shape() == Shape({2, 2, 2}));
    CHECK(lsm->shape() == Shape({2, 2, 2}));

    sm->val()->get(values);

    CHECK( std::equal(values.begin(), values.end(),
                      smOut.begin(), floatApprox) );

    lsm->val()->get(values);

    CHECK( std::equal(values.begin(), values.end(),
                      lsmOut.begin(), floatApprox) );
  }

  SECTION("layer normalization") {
    graph->clear();
    values.clear();

#ifdef CUDA_FOUND
    std::vector<T> vLn({
      -1.1962, 1.43061, 0.380288, -0.614697, 0.816638, 0.622649,
      -1.69679, 0.257504, -1.12563, -0.151387, 1.61181, -0.334796,
      1.07207, -0.622614, 0.862014, -1.31147
    });
#else
    std::vector<T> vLn({
      -1.49821, -0.152206, 0.394932, 1.25548, -1.51701, -0.28032,
      0.9483, 0.849025, 0.855183, 1.11657, -0.788354, -1.1834,
      -0.85939, -1.13109, 0.972076, 1.01841
    });
#endif

    auto a = graph->constant({2, 2, 4}, inits::glorotUniform());
    auto gamma = graph->param("gamma", {1, 4}, inits::ones());
    auto beta = graph->param("beta", {1, 4}, inits::zeros());
    auto ln = layerNorm(a, gamma, beta);

    graph->forward();

    CHECK(ln->shape() == Shape({2, 2, 4}));

    ln->val()->get(values);
    CHECK( std::equal(values.begin(), values.end(),
                      vLn.begin(), floatApprox) );

  }

  SECTION("RMS normalization") {
    graph->clear();
    values.clear();

    std::vector<T> init = {
      2.88794374, 4.67853451, 3.96257305, 3.28433037,
      0.37778997, 0.67662024, 4.24959183, 1.23910618,
      0.68929380, 2.00369596, 4.38251686, 1.75624943,
      4.96126175, 3.01947117, 4.72057724, 2.23017120
    };

    auto a1 = graph->param("test1", {2, 2, 4}, inits::fromVector(init));
    auto a2 = graph->param("test2", {2, 2, 4}, inits::fromVector(init));
    auto gamma = graph->param("gamma", {1, 4}, inits::ones());
    
    auto rms = rmsNorm(a1, gamma, nullptr, 1e-5f);
    auto rms2 = gamma * (a2 / sqrt(mean(a2 * a2, /*axis=*/-1) + 1e-5f));

    auto top = sum(flatten(rms + rms2));

    graph->forward();
    graph->backward();

    CHECK(rms->shape() == Shape({2, 2, 4}));

    std::vector<T> values2;

    // compare values of rms and rms2 to make sure forward computation is correct
    rms->val()->get(values);
    rms2->val()->get(values2);

    CHECK( std::equal(values.begin(), values.end(),
                      values2.begin(), floatApprox) );

    // compare adjoints of a1 and a2 (parameters) to makes sure gradient computation is correct
    a1->grad()->get(values);
    a2->grad()->get(values2);

    CHECK( std::equal(values.begin(), values.end(),
                      values2.begin(), floatApprox) );
  
  }

  SECTION("reductions") {
    graph->clear();
    values.clear();

    std::vector<T> vA({1, 6, 3, 8,
                       5, 2, 7, 4});
    // import numpy as np
    // a = np.array([[1, 6, 3, 8], [5, 2, 7, 4]])
    std::vector<T> vS1({6, 8, 10, 12});              // s1 = np.sum(a, axis=0)
    std::vector<T> vS2({18, 18});                    // np.sum(a, axis = 1)
    std::vector<T> vS4({2.6925824f, 1.80277564f});   // np.std(a, axis = 1)
    std::vector<T> vV5({7.25, 3.25});                // np.var(a, axis = 1)
    std::vector<T> vM6({8, 7});                      // np.max(a, axis = 1)
    std::vector<T> vM7({1, 2});                      // np.min(a, axis = 1)
    std::vector<T> vP8({144, 280});                  // np.prod(a, axis = 1)
    std::vector<T> vL9({8.13364336f, 7.17551536f});  // np.log(np.sum(np.exp(a), axis=1))
    std::vector<T> vW({5.0f, 4.55555556f});          // np.mean(a*s1,axis=-1) / np.mean(s1,axis=-1)

    auto a = graph->constant({2, 4}, inits::fromVector(vA));

    auto s1 = sum(a, /*axis=*/ 0);
    auto s2 = sum(a, /*axis=*/ 1);

    auto m3 = mean(s1, /*axis=*/ 1);

    auto s4 = marian::std(a, /*axis=*/ 1);
    auto v5 = var(a, /*axis=*/ 1);

    auto m6 = max(a, /*axis=*/ 1);
    auto m7 = min(a, /*axis=*/ 1);
    auto p8 = prod(a, /*axis=*/ 1);
    auto l9 = logsumexp(a, /*axis=*/ 1);

    auto sp = scalar_product(s2, s2, /*axis=*/ 0);

    auto wa = weighted_average(a, s1, /*axis=*/ -1);

    graph->forward();

    CHECK(s1->shape() == Shape({1, 4}));
    CHECK(s2->shape() == Shape({2, 1}));
    CHECK(m3->shape() == Shape({1, 1}));
    CHECK(s4->shape() == Shape({2, 1}));
    CHECK(v5->shape() == Shape({2, 1}));
    CHECK(m6->shape() == Shape({2, 1}));
    CHECK(m7->shape() == Shape({2, 1}));
    CHECK(p8->shape() == Shape({2, 1}));
    CHECK(l9->shape() == Shape({2, 1}));
    CHECK(sp->shape() == Shape({1, 1}));
    CHECK(wa->shape() == Shape({2, 1}));

    s1->val()->get(values); CHECK(values == vS1);
    s2->val()->get(values); CHECK(values == vS2);

    CHECK(m3->val()->scalar() == 9);

    // The two tests below were changed to use this approx function since they originally failed
    // on a Titan V. The margin was increased to allow the tests to pass.
    auto floatApproxLocal = [](T x, T y) -> bool { return x == Approx(y).margin(0.004); };

    s4->val()->get(values); CHECK(std::equal(values.begin(), values.end(), vS4.begin(), floatApproxLocal));
    v5->val()->get(values); CHECK(values == vV5);
    m6->val()->get(values); CHECK(values == vM6);
    m7->val()->get(values); CHECK(values == vM7);
    p8->val()->get(values); CHECK(values == vP8);
    l9->val()->get(values); CHECK(std::equal(values.begin(), values.end(), vL9.begin(), floatApproxLocal));

    CHECK(sp->val()->scalar() == 648);

    wa->val()->get(values); CHECK(std::equal(values.begin(), values.end(), vW.begin(), floatApprox));
  }

  SECTION("concatenation") {
    graph->clear();
    values.clear();

    std::vector<T> vO1({ 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                             3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
                             1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                             3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4});

    std::vector<T> vO2({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                            1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                            1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                            1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4});

    std::vector<T> vO3({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});

    std::vector<T> vO4({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});

    auto in1 = graph->constant({1, 2, 2, 3}, inits::fromValue(1));
    auto in2 = graph->constant({1, 2, 2, 3}, inits::fromValue(2));
    auto in3 = graph->constant({1, 2, 2, 3}, inits::fromValue(3));
    auto in4 = graph->constant({1, 2, 2, 3}, inits::fromValue(4));

    auto c1out1 = concatenate({in1, in2, in3, in4}, /*axis=*/ 2);
    auto c1out2 = concatenate({in1, in2, in3, in4}, /*axis=*/ -1);
    auto c1out3 = concatenate({in1, in2, in3, in4}, /*axis=*/ -3);
    auto c1out4 = concatenate({in1, in2, in3, in4}, /*axis=*/ 0);

    graph->forward();

    CHECK(c1out1->shape() == Shape({1, 2, 8, 3}));
    CHECK(c1out2->shape() == Shape({1, 2, 2, 12}));
    CHECK(c1out3->shape() == Shape({1, 8, 2, 3}));
    CHECK(c1out4->shape() == Shape({4, 2, 2, 3}));

    c1out1->val()->get(values);
    CHECK( values == vO1 );

    c1out2->val()->get(values);
    CHECK( values == vO2 );

    c1out3->val()->get(values);
    CHECK( values == vO3 );

    c1out4->val()->get(values);
    CHECK( values == vO4 );
  }

  SECTION("dot product") {
    graph->clear();
    values.clear();

    std::vector<T> vA({1, 2, 3,
                       4, 5, 6,
                       7, 8, 9,
                       10, 11, 12});
    std::vector<T> vB({1, 2,
                       3, 4,
                       5, 6});
    std::vector<T> vC({22, 28,
                       49, 64,
                       76, 100,
                       103, 136});

    auto A = graph->param("A", {2, 2, 3}, inits::fromVector(vA));
    auto B = graph->param("B", {3, 2}, inits::fromVector(vB));
    auto C = dot(A, B);

    CHECK(C->shape() == Shape({2, 2, 2}));

    graph->forward();

    C->val()->get(values);
    CHECK(values == vC);
  }

  // Currently no support for CPU
  // @TODO: support for fp16 is done internally via cast to fp16, not efficient.
  if(device == DeviceType::gpu) {
    SECTION("csr-dot product") {
      graph->clear();
      values.clear();
      // CSR dot product, tested against dense product on the same values
      std::vector<T> vS({1, 0, 0, 1,          // sparse
                         0, 0, 1, 1.5});
      std::vector<T> vD({1, 2, 3, 1.2, 5.6,   // dense
                         4, 5, 6, 2.3, 6.7,
                         7, 8, 9, 3.4, 7.8,
                         1, 1, 2, 4.5, 8.9});
      auto S  = graph->param("S",  { 2, 4 }, inits::fromVector(vS));
      auto D  = graph->param("D",  { 4, 5 }, inits::fromVector(vD));
      auto DT = graph->param("DT", { 5, 4 }, inits::fromVector(vD)); // example matrix with transposed dimensions
      std::vector<T> SV;    // create CSR version of S
      std::vector<IndexType> SI, SO;
      SO.push_back((IndexType)SI.size());
      for (IndexType i = 0; i < (IndexType)S->shape()[0]; i++) {
        for (IndexType j = 0; j < (IndexType)S->shape()[1]; j++) {
          auto k = 4 * i + j;
          if (vS[k] != (T)0.f) {
            SV.push_back(vS[k]);
            SI.push_back(j);
          }
        }
        SO.push_back((IndexType)SI.size());
      }

      auto SxDd    = dot(S, D);
      auto STxSxDd = dot(S, SxDd, /*transA=*/true);
      auto SxDs = csr_dot( // sparse x dense
            S->shape(),
            graph->constant({(int)SV.size()}, inits::fromVector(SV)),
            graph->constant({(int)SI.size()}, inits::fromVector(SI), Type::uint32),
            graph->constant({(int)SO.size()}, inits::fromVector(SO), Type::uint32),
            D);
      auto STxSxDs = csr_dot(   // transpose(sparse) x dense; we use result of previous since dimensions match
            S->shape(),
            graph->constant({(int)SV.size()}, inits::fromVector(SV)),
            graph->constant({(int)SI.size()}, inits::fromVector(SI), Type::uint32),
            graph->constant({(int)SO.size()}, inits::fromVector(SO), Type::uint32),
            SxDd, /*transS=*/true);

#if 0 // currently not used anywhere
      auto DTxSTd   = dot(DT,     S, /*transA=*/false, /*transB=*/true);
      auto DTxSTxSd = dot(DTxSTd, S);
      auto DTxSTs = dot_csr( // dense x sparse
            DT,
            S->shape(),
            graph->constant({(int)SV.size()}, inits::fromVector(SV)),
            graph->constant({(int)SI.size()}, inits::fromVector(SI), Type::uint32),
            graph->constant({(int)SO.size()}, inits::fromVector(SO), Type::uint32),
            /*transS=*/true);
      auto DTxSTxSs = dot_csr( // dense x transpose(sparse)
            DTxSTd,
            S->shape(),
            graph->constant({(int)SV.size()}, inits::fromVector(SV)),
            graph->constant({(int)SI.size()}, inits::fromVector(SI), Type::uint32),
            graph->constant({(int)SO.size()}, inits::fromVector(SO), Type::uint32));
#endif

      CHECK(SxDs->shape() == SxDd->shape());
      CHECK(STxSxDs->shape() == STxSxDd->shape());
#if 0
      CHECK(DTxSTs->shape() == DTxSTd->shape());
      CHECK(DTxSTxSs->shape() == DTxSTxSd->shape());
#endif

      graph->forward();

      // dense and sparse operation results must be the same
      SxDd    ->val()->get(values2); SxDs    ->val()->get(values); CHECK(values == values2);
      STxSxDd ->val()->get(values2); STxSxDs ->val()->get(values); CHECK(values == values2);
#if 0
      DTxSTd  ->val()->get(values2); DTxSTs  ->val()->get(values); CHECK(values == values2);
      DTxSTxSd->val()->get(values2); DTxSTxSs->val()->get(values); CHECK(values == values2);
#endif
    }
  }

  SECTION("affine transformation") {
    graph->clear();
    values.clear();

    std::vector<T> vA({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    std::vector<T> vB({1, -2, 3, 4, -5, 6});
    std::vector<T> vAff({-6, 26, -9, 50, -12, 74, -15, 98});
    std::vector<T> vAffRelu({0, 26, 0, 50, 0, 74, 0, 98});

    auto A = graph->param("A", {4, 3}, inits::fromVector(vA));
    auto B = graph->param("B", {3, 2}, inits::fromVector(vB));
    auto bias = graph->param("C", {1, 2}, inits::fromValue(2));

    auto aff1 = affine(A, B, bias);
    auto aff2 = dot(A, B) + bias;

    auto affRelu1 = affineWithRelu(A, B, bias);
    auto affRelu2 = relu(dot(A, B) + bias);

    graph->forward();

    CHECK(aff1->shape() == Shape({4, 2}));
    aff1->val()->get(values);
    CHECK(values == vAff);

    values2.clear();
    CHECK(aff2->shape() == aff1->shape());
    aff2->val()->get(values2);
    CHECK(values2 == values);

    affRelu1->val()->get(values);
    affRelu2->val()->get(values2);
    CHECK(values2 == vAffRelu);
    CHECK(values2 == values);
  }

  SECTION("bdot") {
    graph->clear();
    values.clear();

    std::vector<T> vA({ 1, 2, 
                        3, 4,
                        5, 6,
                        7, 8});

    std::vector<T> vB({ 1,  2, 
                        3,  4,
                        5,  6,
                        7,  8,
                        9, 10,
                       11, 12});

    std::vector<T> vC({  7,  10, 
                        15,  22, 
                        19,  22, 
                        43,  50, 
                        31,  34, 
                        71,  78, 
                        23,  34, 
                        31,  46, 
                        67,  78, 
                        91, 106, 
                       111, 122, 
                       151, 166});

    std::vector<T> vCt({   5,  11, 
                          11,  25, 
                          17,  23, 
                          39,  53, 
                          29,  35, 
                          67,  81, 
                          17,  39, 
                          23,  53, 
                          61,  83, 
                          83, 113, 
                         105, 127, 
                         143, 173});

    auto A = graph->param("A", {2, 1, 2, 2}, inits::fromVector(vA));
    auto B = graph->param("B", {1, 3, 2, 2}, inits::fromVector(vB));
    
    auto C  = bdot(A, B, /*transA=*/false, /*transB=*/false);
    auto Ct = bdot(A, B, /*transA=*/false, /*transB=*/true);

    graph->forward();

    CHECK(C->shape()  == Shape({2, 3, 2, 2}));
    CHECK(Ct->shape() == Shape({2, 3, 2, 2}));

    C->val()->get(values);
    CHECK(vC == values);

    Ct->val()->get(values);
    CHECK(vCt == values);
  }

  SECTION("repeat") {
    graph->clear();
    values.clear();

    std::vector<T> vA({1, 2, 3, 4, 5, 6});
    std::vector<T> vB({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
    std::vector<T> vC({1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6});

    auto A = graph->param("A", {2,3}, inits::fromVector(vA));
    auto I = repeat(A, 1, 0);
    auto B = repeat(A, 2, 0);
    auto C = repeat(A, 2, 1);
    graph->forward();

    CHECK(I->shape() == Shape({2, 3}));
    I->val()->get(values);
    CHECK(values == vA);

    CHECK(B->shape() == Shape({4, 3}));
    B->val()->get(values);
    CHECK(values == vB);

    CHECK(C->shape() == Shape({2, 6}));
    C->val()->get(values);
    CHECK(values == vC);
  }

  SECTION("flatten") {
    graph->clear();
    values.clear();

    std::vector<T> vIn({1, 2, 3, 4, 5, 6, 7, 8});

    auto A = graph->param("A", {2, 4}, inits::fromVector(vIn));
    auto Af = flatten(A);
    auto B = graph->param("B", {2, 2, 1, 2}, inits::fromVector(vIn));
    auto Bf = flatten(B);
    graph->forward();

    CHECK(Af->shape() == Shape({8}));
    Af->val()->get(values);
    CHECK(values == vIn);

    CHECK(Bf->shape() == Shape({8}));
    Bf->val()->get(values);
    CHECK(values == vIn);
  }

  SECTION("rows selection from 2d matrix") {
    graph->clear();
    values.clear();

    std::vector<T> vA({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    std::vector<IndexType> iB0({0});            // first row
    std::vector<IndexType> iB1({0, 1, 2});      // several consecutive rows
    std::vector<IndexType> iB2({0, 2});         // two nonconsecutive rows
    std::vector<IndexType> iB3({2, 1});         // reversed order
    std::vector<IndexType> iB4({1, 1});         // repeated rows
    std::vector<IndexType> iB5({0, 1, 2, 3});   // identity
    std::vector<IndexType> iB6({});             // empty
    std::vector<T> vB0({1, 2, 3});
    std::vector<T> vB1({1, 2, 3, 4, 5, 6, 7, 8, 9});
    std::vector<T> vB2({1, 2, 3, 7, 8, 9});
    std::vector<T> vB3({7, 8, 9, 4, 5, 6});
    std::vector<T> vB4({4, 5, 6, 4, 5, 6});
    std::vector<T> vB6;

    auto A = graph->param("A", {4, 3}, inits::fromVector(vA));
    auto B0 = rows(A, iB0);
    auto B1 = rows(A, iB1);
    auto B2 = rows(A, iB2);
    auto B3 = rows(A, iB3);
    auto B4 = rows(A, iB4);
    auto B5 = rows(A, iB5);
    auto B6 = rows(A, iB6);
    graph->forward();

    CHECK(B0->shape() == Shape({1, 3}));
    B0->val()->get(values);
    CHECK( values == vB0 );

    CHECK(B1->shape() == Shape({3, 3}));
    B1->val()->get(values);
    CHECK( values == vB1 );

    CHECK(B2->shape() == Shape({2, 3}));
    B2->val()->get(values);
    CHECK( values == vB2 );

    CHECK(B3->shape() == Shape({2, 3}));
    B3->val()->get(values);
    CHECK( values == vB3 );

    CHECK(B4->shape() == Shape({2, 3}));
    B4->val()->get(values);
    CHECK( values == vB4 );

    CHECK(B5->shape() == Shape({4, 3}));
    B5->val()->get(values);
    CHECK( values == vA );

    CHECK(B6->shape() == Shape({0, 3}));
    B6->val()->get(values);
    CHECK( values == vB6 );
  }

  SECTION("columns selection from 2d matrix") {
    graph->clear();
    values.clear();

    std::vector<T> vA({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    std::vector<IndexType> iB0({0});            // first column
    std::vector<IndexType> iB1({0, 1, 2});      // several consecutive columns
    std::vector<IndexType> iB2({0, 2});         // two nonconsecutive columns
    std::vector<IndexType> iB3({2, 1});         // reversed order
    std::vector<IndexType> iB4({1, 1});         // repeated columns
    std::vector<IndexType> iB5({0, 1, 2, 3});   // identity
    std::vector<IndexType> iB6({});             // empty

    std::vector<T> vB0({1, 5, 9});
    std::vector<T> vB1({1, 2, 3, 5, 6, 7, 9, 10, 11});
    std::vector<T> vB2({1, 3, 5, 7, 9, 11});
    std::vector<T> vB3({3, 2, 7, 6, 11, 10});
    std::vector<T> vB4({2, 2, 6, 6, 10, 10});
    std::vector<T> vB6;

    auto A = graph->param("A", {3, 4}, inits::fromVector(vA));
    auto B0 = cols(A, iB0);
    auto B1 = cols(A, iB1);
    auto B2 = cols(A, iB2);
    auto B3 = cols(A, iB3);
    auto B4 = cols(A, iB4);
    auto B5 = cols(A, iB5);
    auto B6 = cols(A, iB6);
    graph->forward();

    CHECK(B0->shape() == Shape({3, 1}));
    B0->val()->get(values);
    CHECK( values == vB0 );

    CHECK(B1->shape() == Shape({3, 3}));
    B1->val()->get(values);
    CHECK( values == vB1 );

    CHECK(B2->shape() == Shape({3, 2}));
    B2->val()->get(values);
    CHECK( values == vB2 );

    CHECK(B3->shape() == Shape({3, 2}));
    B3->val()->get(values);
    CHECK( values == vB3 );

    CHECK(B4->shape() == Shape({3, 2}));
    B4->val()->get(values);
    CHECK( values == vB4 );

    CHECK(B5->shape() == Shape({3, 4}));
    B5->val()->get(values);
    CHECK( values == vA );

    CHECK(B6->shape() == Shape({3, 0}));
    B6->val()->get(values);
    CHECK( values == vB6 );
  }

  SECTION("relation of rows and columns selection using transpose") {
    graph->clear();
    values.clear();
    values2.clear();

    std::vector<T> vA({0, .3333, -.2, -.3, 0, 4.5, 5.2, -10, 101.45, -100.05, 0, 1.05e-5});
    std::vector<IndexType> idx({0, 1});

    auto A1 = graph->param("4x3", {4,3}, inits::fromVector(vA));
    auto B1 = rows(transpose(A1), idx);
    auto C1 = transpose(cols(A1, idx));
    auto A2 = graph->param("6x2", {6,2}, inits::fromVector(vA));
    auto B2 = cols(transpose(A2), idx);
    auto C2 = transpose(rows(A2, idx));
    graph->forward();

    CHECK(B1->shape() == C1->shape());
    B1->val()->get(values);
    C1->val()->get(values2);
    CHECK( values == values2 );

    values.clear();
    values2.clear();

    CHECK(B2->shape() == C2->shape());
    B2->val()->get(values);
    C2->val()->get(values2);
    CHECK( values == values2 );
  }

  SECTION("select, step, slice operators") {
    using IndexVector = std::vector<IndexType>;

    graph->clear();
    values.clear();

    std::vector<T> vA({  1, -2,   3,
                        -4,  5,  -6,
                         7, -8,   9,
                       -10, 11, -12});
    std::vector<T> vC({ 1,  -2, // C = np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12]).reshape((2, 3, 2))
                        3,  -4,
                        5,  -6,

                        7,  -8,
                        9, -10,
                        11, -12 });
    std::vector<T> vB1({1, -2, 3});
    std::vector<T> vB2({1, -4, 7, -10});
    std::vector<T> vB3({-2, 5, -8, 11});
    std::vector<T> vB4({1, -2, 3, -4, 5, -6});
    std::vector<T> vD1(vB4);
    std::vector<T> vD2({5, -6, 11, -12});
    std::vector<T> vD3({1, -2, 5, -6, 7, -8, 11, -12}); // C[:,(0,2),:]
    std::vector<T> vD4({5, -6, 3, -4, 7, -8, 11, -12}); // [C[0,(2,1),:],C[1,(0,2),:]]
    std::vector<T> vS1({7, -8, 9});
    std::vector<T> vS2({-4, 5, -6, 7, -8, 9});
    std::vector<T> vS3({7, -8, 9, -10, 11, -12});

    auto A = graph->param("4x3", {4,3}, inits::fromVector(vA));
    auto B1a = index_select(A, 0, IndexVector({0})); // always uses gather()
    auto B1b = slice(A,  0, 0);                        // memory-consecutive view
    auto B2  = slice(A,  1, 0);                        // not memory-consecutive
    auto B3  = slice(A, -1, 1);
    auto B4a = index_select(A, 0, IndexVector({0, 1}));
    auto B4b = slice(A, 0, Slice(0, 2)); // this is memory-consecutive
    auto B5  = slice(A, 0, Slice(0, 4)); // this is a no-op
    CHECK(B1a->type() == "rows");      // actually optimized to rows()
    CHECK(B1b->type() == "sliceView"); // must use view
    CHECK(B2->type() == "gather");     // cannot use view
    CHECK(B4a->type() == "rows");
    CHECK(B4b->type() == "sliceView"); // must use view
    CHECK(B5.get() == A.get());        // must be no-op

    auto C = graph->param("2x3x2", {2, 3, 2}, inits::fromVector(vC));
    auto D1 = slice(C,  0, 0);
    auto D2 = slice(C, -2, 2);
    auto D3 = index_select(C, 1, IndexVector({0, 2})); // C[:,(0,2),:]
    CHECK(D1->type() == "sliceView");
    CHECK(D2->type() == "gather");
    // enable this once gather() supports batched indices:
    auto D4 = gather(C, 1, graph->constant({2, 2, 1}, // [C[0,(2,1),:],C[1,(0,2),:]]
                                          inits::fromVector(std::vector<IndexType>{
                                            2, 1,
                                            0, 2 }),
                                          Type::uint32));

    auto S1 = slice(A, 0, 2);
    auto S2 = narrow(A, 0, 1, 2);
    auto S3 = slice(A, 0, Slice(-2, Slice::END));

    graph->forward();

    CHECK(B1a->shape() == Shape({1, 3})); B1a->val()->get(values); CHECK( values == vB1 );
    CHECK(B1b->shape() == Shape({1, 3})); B1b->val()->get(values); CHECK( values == vB1 );
    CHECK(B2->shape() == Shape({4, 1})); B2->val()->get(values); CHECK( values == vB2 );
    CHECK(B3->shape() == Shape({4, 1})); B3->val()->get(values); CHECK( values == vB3 );
    CHECK(B4a->shape() == Shape({2, 3})); B4a->val()->get(values); CHECK( values == vB4 );
    CHECK(B4b->shape() == Shape({2, 3})); B4b->val()->get(values); CHECK( values == vB4 );

    CHECK(D1->shape() == Shape({1, 3, 2})); D1->val()->get(values); CHECK( values == vD1 );
    CHECK(D2->shape() == Shape({2, 1, 2})); D2->val()->get(values); CHECK( values == vD2 );
    CHECK(D3->shape() == Shape({2, 2, 2})); D3->val()->get(values); CHECK( values == vD3 );
    CHECK(D4->shape() == Shape({2, 2, 2})); D4->val()->get(values); CHECK( values == vD4 );

    CHECK(S1->shape() == Shape({1,3})); S1->val()->get(values); CHECK(values == vS1);
    CHECK(S2->shape() == Shape({2,3})); S2->val()->get(values); CHECK(values == vS2);
    CHECK(S3->shape() == Shape({2,3})); S3->val()->get(values); CHECK(values == vS3);
  }

  SECTION("rows/cols as gather operations") {
    graph->clear();
    values.clear();
    values2.clear();


    std::vector<T> vA({0, .3333, -.2, -.3, 0, 4.5, 5.2, -10, 101.45, -100.05, 0, 1.05e-5});
    std::vector<IndexType> indices({0, 2});

    auto A = graph->param("4x3", {4, 3}, inits::fromVector(vA));
    auto B1 = rows(A, indices);
    auto B2 = gather(A, 0, graph->indices(indices, A, 0));
    auto C1 = cols(A, indices);
    auto C2 = gather(A, 1, graph->indices(indices, A, 1));

    graph->forward();

    CHECK(B1->shape() == B2->shape());
    B1->val()->get(values);
    B2->val()->get(values2);
    CHECK( values == values2 );

    CHECK(C1->shape() == C2->shape());
    C1->val()->get(values);
    C2->val()->get(values2);
    CHECK( values == values2 );
  }

  SECTION("topk operations") {
    graph->clear();
    values.clear();

    std::vector<T> vA({   0,      .3333,   -.2,
                          -.3,   0,        4.5,
                          5.2, -10,      101.45,
                       -100.05,  0,        1.05e-5});

    auto a = graph->constant({2, 2, 3}, inits::fromVector(vA));

    // get top-k indices and values as a tuple
    auto rtopk1 = topk(a, /*k=*/2, /*axis=*/-1, /*descending=*/true);
    auto rval1  = get<0>(rtopk1);  // values from top-k
    auto ridx1  = get<1>(rtopk1);  // indices from top-k
    auto gval1  = gather(a, -1, ridx1); // get the same values via gather and indices

    auto ridx2  = get<1>(topk(a, /*k=*/2, /*axis=*/-1, /*descending=*/false));
    auto gval2  = gather(a, -1, ridx2); // get the same values via gather and indices

    auto ridx3  = get<1>(argmin(a, -1));
    auto ridx3_ = slice(ridx2, -1, 0); // slice and cast now support uint32_t/IndexType

    // @TODO: add integer types to more operators
    auto eq3 = eq(cast(ridx3, floatType), cast(ridx3_, floatType));

    auto rtopk4 = argmax(a, /*axis=*/-2); // axes other than -1 are currently implemented via inefficient transpose
    auto rval4  = get<0>(rtopk4);
    auto ridx4  = get<1>(rtopk4);
    auto gval4  = gather(a, -2, ridx4);

    graph->forward();

    CHECK(rval1 != gval1);
    CHECK(rval1->shape() == gval1->shape());
    CHECK(ridx1->shape() == gval1->shape());

    std::vector<T> vval1 = { 0.3333,  0,
                             4.5,     0,
                           101.45,    5.2,
                             1.05e-5, 0 };

    std::vector<T> rvalues;
    std::vector<T> gvalues;
    rval1->val()->get(rvalues);
    gval1->val()->get(gvalues);
    CHECK( rvalues == gvalues );
    CHECK( rvalues == vval1 );

    std::vector<T> vval2 = { -0.2,  0,
                             -0.3,  0,
                            -10.0,  5.2,
                           -100.05, 0 };
    gval2->val()->get(values);
    CHECK( values == vval2 );

    eq3->val()->get(values);
    CHECK( values == std::vector<T>({1, 1, 1, 1}) );

    std::vector<IndexType> vidx4;
    ridx4->val()->get(vidx4);
    CHECK( ridx4->shape() == Shape({2, 1, 3}) );
    CHECK( vidx4 == std::vector<IndexType>({0, 0, 1,
                                            0, 1, 0}) );

    std::vector<T> vval4 = { 0,   0.3333,   4.5,
                             5.2, 0,      101.45 };
    rval4->val()->get(values);
    CHECK( values == vval4 );

    gval4->val()->get(values);
    CHECK( values == vval4 );
  }

  SECTION("cross entropy with label smoothing vs logsoftmax with gather") {
    graph->clear();
    values.clear();
    values2.clear();
    
    std::vector<T> logitsVec = {
      -0.1, -1.2, -0.4,
       1.2,  2.3, -3.4,
      -2.2,  1.0, -1.2
    };
    std::vector<IndexType> yhatVec = { 0, 1, 2 };
  
    auto logits   = graph->param("logits",   {3, 3}, inits::fromVector(logitsVec));
    auto logitsGa = graph->param("logitsGa", {3, 3}, inits::fromVector(logitsVec));
    auto yhat     = graph->indices(yhatVec); // [3]
    auto yhatGa   = reshape(yhat, {3, 1});   // [3, 1]

    float lsAlpha = 0.1;
    auto ceOp = cast(cross_entropy(logits, yhat, /*labelSmoothing=*/lsAlpha), floatType);

    auto ceGa = -gather(logsoftmax(logitsGa), -1, yhatGa);
         ceGa = (1.f - lsAlpha) * ceGa - lsAlpha * mean(logsoftmax(logitsGa), /*axis=*/-1);

    auto top = sum(ceOp) + sum(ceGa); // cast to float16 if required as cross_entropy casts to float32 internally

    graph->forward();
    graph->backward();

    CHECK(ceOp->shape() == ceGa->shape());

    // compare forward values
    ceOp->val()->get(values);
    ceGa->val()->get(values2);

    CHECK( std::equal(values.begin(), values.end(),
                      values2.begin(), floatApprox2) );


    // compare parameter gradients
    logits->grad()->get(values);
    logitsGa->grad()->get(values2);
    CHECK( std::equal(values.begin(), values.end(),
                      values2.begin(), floatApprox2) );
  }
}

#ifdef CUDA_FOUND
TEST_CASE("Expression graph supports basic math operations (gpu)", "[operator]") {
  tests<float>(DeviceType::gpu);
}

#if COMPILE_FP16
TEST_CASE("Expression graph supports basic math operations (gpu fp16)", "[operator]") {
  tests<float16>(DeviceType::gpu, Type::float16);
}
#endif
#endif

#ifdef BLAS_FOUND
TEST_CASE("Expression graph supports basic math operations (cpu)", "[operator]") {
  tests<float>(DeviceType::cpu);
}
#endif

#ifdef BLAS_FOUND
#ifdef CUDA_FOUND

TEST_CASE("Compare aggregate operator", "[graph]") {
  auto floatApprox = [](float x, float y) -> bool { return x == Approx(y).margin(0.001f); };

  Config::seed = 1234;

  std::vector<float> initc;
  std::vector<float> inita;

  {
    auto graph = New<ExpressionGraph>();
    graph->setDevice({0, DeviceType::cpu});
    graph->reserveWorkspaceMB(40);

    auto chl = graph->param("1x10x512x2048", {1, 10, 512, 2048}, inits::normal());
    auto adj = graph->param("1x1x512x2048",  {1,  1, 512, 2048}, inits::normal());
    graph->forward();

    chl->val()->get(initc);
    adj->val()->get(inita);
  }

  SECTION("initializing with zero (cpu)") {
    std::vector<float> values1;
    std::vector<float> values2;

    auto graph1 = New<ExpressionGraph>();
    graph1->setDevice({0, DeviceType::cpu});
    graph1->reserveWorkspaceMB(40);

    auto graph2 = New<ExpressionGraph>();
    graph2->setDevice({0, DeviceType::gpu});
    graph2->reserveWorkspaceMB(40);

    auto chl1 = graph1->param("1x10x512x2048", {1, 10, 512, 2048}, inits::fromVector(initc));
    auto adj1 = graph1->param("1x1x512x2048",  {1,  1, 512, 2048}, inits::fromVector(inita));
    auto prod1 = scalar_product(chl1, adj1, -1);
    graph1->forward();

    auto chl2 = graph2->param("1x10x512x2048", {1, 10, 512, 2048}, inits::fromVector(initc));
    auto adj2 = graph2->param("1x1x512x2048",  {1,  1, 512, 2048}, inits::fromVector(inita));
    auto prod2 = scalar_product(chl2, adj2, -1);
    graph2->forward();

    prod1->val()->get(values1);
    prod2->val()->get(values2);

    CHECK( std::equal(values1.begin(), values1.end(), values2.begin(), floatApprox) );
  }
}

  #endif
  #endif
