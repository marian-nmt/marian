#include "catch.hpp"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"

using namespace marian;

void tests(DeviceType device) {
  auto floatApprox = [](float x, float y) { return x == Approx(y); };

  auto graph = New<ExpressionGraph>();
  graph->setDevice({0, device});
  graph->reserveWorkspaceMB(16);

  std::vector<float> values;

  SECTION("scalar multiplication") {
    graph->clear();
    values.clear();
    std::vector<float> vB({1, 2, 3, 4, 5, 6});

    auto B = graph->param("B", {3, 2}, inits::from_vector(vB));
    auto B2 = B * 2.0f;
    graph->forward();

    CHECK(B2->shape() == Shape({3, 2}));
    B2->val()->get(values);

    std::vector<float> vB2({2, 4, 6, 8, 10, 12});
    CHECK(values == vB2);
  }

  SECTION("elementwise binary operators with broadcasting") {
    graph->clear();
    values.clear();

    std::vector<float> vA({1, -2, 3, -4});
    std::vector<float> vB({0.5, 1.5});

    std::vector<float> vAdd({1.5, -0.5, 3.5, -2.5});
    std::vector<float> vMinus({-0.5, 3.5, -2.5, 5.5});
    std::vector<float> vMult({0.5, -3.0, 1.5, -6.0});
    std::vector<float> vDiv({2.0f, -1.33333f, 6.0f, -2.66667f});

    auto a = graph->constant({2, 2, 1}, inits::from_vector(vA));
    auto b = graph->constant({2, 1}, inits::from_vector(vB));

    auto add = a + b;
    auto minus = b - a;
    auto mult = a * b;
    auto div = a / b;

    graph->forward();

    CHECK(add->shape() == Shape({2, 2, 1}));
    CHECK(minus->shape() == Shape({2, 2, 1}));
    CHECK(mult->shape() == Shape({2, 2, 1}));
    CHECK(div->shape() == Shape({2, 2, 1}));

    add->val()->get(values);
    CHECK( values == vAdd );

    minus->val()->get(values);
    CHECK( values == vMinus );

    mult->val()->get(values);
    CHECK( values == vMult );

    div->val()->get(values);
    CHECK( std::equal(values.begin(), values.end(),
                      vDiv.begin(), floatApprox) );
  }

  SECTION("transposing and reshaping") {
    graph->clear();
    values.clear();

    std::vector<float> vA({1, 2, 3, 4, 5, 6, 7, 8});

    std::vector<float> vT1({1, 5, 2, 6, 3, 7, 4, 8});
    std::vector<float> vT3({1, 2, 5, 6, 3, 4, 7, 8});
    std::vector<float> vT4({1, 5, 3, 7, 2, 6, 4, 8});
    std::vector<float> vT5({1, 2, 5, 6, 3, 4, 7, 8});

    auto a = graph->constant({2, 4}, inits::from_vector(vA));

    auto t1 = transpose(a);
    auto t2 = transpose(t1);
    auto t3 = transpose(reshape(t1, {2, 2, 2}));

    auto t4 = transpose(reshape(a, {2, 1, 2, 2}), {1, 3, 2, 0});
    auto t5 = transpose(reshape(a, {2, 1, 2, 2}), {2, 0, 1, 3});

    graph->forward();

    CHECK(t1->shape() == Shape({4, 2}));
    CHECK(t2->shape() == Shape({2, 4}));
    CHECK(t3->shape() == Shape({2, 2, 2}));
    CHECK(t4->shape() == Shape({1, 2, 2, 2}));
    CHECK(t5->shape() == Shape({2, 2, 1, 2}));

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
  }

  SECTION("softmax and logsoftmax") {
    graph->clear();
    values.clear();
    std::vector<float> in({-.2, -.3, 4.5, 5.2, -10, 101.45, -100.05, 1.05e-5});

    std::vector<float> smOut({ 0.52498f, 0.47502f, 0.33181f, 0.66819f,
                               0.0f, 1.0f, 0.0f, 1.0f });

    std::vector<float> lsmOut({ -0.6444f, -0.7444f, -1.10319f, -0.40319f,
                                -111.45f, 0.0f, -100.05001f, 0.0f });

    auto input = graph->constant({2, 2, 2}, inits::from_vector(in));

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

    Config::seed = 1234;

    std::vector<float> vLn({
      -1.20521, -0.321409, -0.0363369, 1.56296,
      0.332987, -0.613398, -1.17766, 1.45807,
      -0.731601, -0.187812, -0.766431, 1.68584,
      -1.31923, -0.059028, 1.49732, -0.119065
    });

    auto a = graph->constant({2, 2, 4}, inits::glorot_uniform);

    auto gamma = graph->param("gamma", {1, 4}, inits::ones);
    auto beta = graph->param("beta", {1, 4}, inits::zeros);

    auto ln = layerNorm(a, gamma, beta);

    graph->forward();

    CHECK(ln->shape() == Shape({2, 2, 4}));


    ln->val()->get(values);
    CHECK( std::equal(values.begin(), values.end(),
                      vLn.begin(), floatApprox) );

  }

  SECTION("reductions") {
    graph->clear();
    values.clear();

    std::vector<float> vA({1, 2, 3, 4, 5, 6, 7, 8});
    std::vector<float> vS1({6, 8, 10, 12});
    std::vector<float> vS2({10, 26});

    std::vector<float> vW({2.77778f, 6.77778f});


    auto a = graph->constant({2, 4}, inits::from_vector(vA));

    auto s1 = sum(a, /*axis=*/ 0);
    auto s2 = sum(a, /*axis=*/ 1);

    auto m3 = mean(s1, /*axis=*/ 1);

    auto sp = scalar_product(s2, s2, /*axis=*/ 0);

    auto wa = weighted_average(a, s1, /*axis=*/ -1);

    graph->forward();

    CHECK(s1->shape() == Shape({1, 4}));
    CHECK(s2->shape() == Shape({2, 1}));
    CHECK(m3->shape() == Shape({1, 1}));
    CHECK(sp->shape() == Shape({1, 1}));
    CHECK(wa->shape() == Shape({2, 1}));

    s1->val()->get(values);
    CHECK( values == vS1 );

    s2->val()->get(values);
    CHECK( values == vS2 );

    CHECK( m3->val()->scalar() == 9 );
    CHECK( sp->val()->scalar() == 776 );

    wa->val()->get(values);
    CHECK( std::equal(values.begin(), values.end(),
                      vW.begin(), floatApprox) );
  }

  SECTION("concatenation") {
    graph->clear();
    values.clear();

    std::vector<float> vO1({ 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                             3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
                             1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                             3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4});

    std::vector<float> vO2({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                            1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                            1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                            1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4});

    std::vector<float> vO3({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});

    std::vector<float> vO4({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});

    auto in1 = graph->constant({1, 2, 2, 3}, inits::from_value(1));
    auto in2 = graph->constant({1, 2, 2, 3}, inits::from_value(2));
    auto in3 = graph->constant({1, 2, 2, 3}, inits::from_value(3));
    auto in4 = graph->constant({1, 2, 2, 3}, inits::from_value(4));

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

    std::vector<float> vA({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    std::vector<float> vB({1, 2, 3, 4, 5, 6});
    std::vector<float> vC({22, 28, 49, 64, 76, 100, 103, 136});

    auto A = graph->param("A", {2, 2, 3}, inits::from_vector(vA));
    auto B = graph->param("B", {3, 2}, inits::from_vector(vB));
    auto C = dot(A, B);
    graph->forward();

    CHECK(C->shape() == Shape({2, 2, 2}));
    C->val()->get(values);
    CHECK(values == vC);
  }

  SECTION("affine transformation") {
    graph->clear();
    values.clear();

    std::vector<float> vA({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    std::vector<float> vB({1, 2, 3, 4, 5, 6});
    std::vector<float> vAff({24, 30, 51, 66, 78, 102, 105, 138});

    auto A = graph->param("A", {4, 3}, inits::from_vector(vA));
    auto B = graph->param("B", {3, 2}, inits::from_vector(vB));
    auto C = graph->param("C", {4, 2}, inits::from_value(2));
    auto aff1 = affine(A, B, C);
    auto aff2 = dot(A, B) + C;
    graph->forward();

    CHECK(aff1->shape() == Shape({4, 2}));
    aff1->val()->get(values);
    CHECK(values == vAff);

    std::vector<float> values2;
    CHECK(aff2->shape() == aff1->shape());
    aff2->val()->get(values2);
    CHECK(values2 == values);
  }

  SECTION("repeat") {
    graph->clear();
    values.clear();

    std::vector<float> vA({1, 2, 3, 4, 5, 6});
    std::vector<float> vB({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
    std::vector<float> vC({1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6});

    auto A = graph->param("A", {2,3}, inits::from_vector(vA));
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

    std::vector<float> vIn({1, 2, 3, 4, 5, 6, 7, 8});

    auto A = graph->param("A", {2, 4}, inits::from_vector(vIn));
    auto Af = flatten(A);
    auto B = graph->param("B", {2, 2, 1, 2}, inits::from_vector(vIn));
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

    std::vector<float> vA({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    std::vector<IndexType> iB0({0});            // first row
    std::vector<IndexType> iB1({0, 1, 2});      // several consecutive rows
    std::vector<IndexType> iB2({0, 2});         // two nonconsecutive rows
    std::vector<IndexType> iB3({2, 1});         // reversed order
    std::vector<IndexType> iB4({1, 1});         // repeated rows
    std::vector<IndexType> iB5({0, 1, 2, 3});   // identity
    std::vector<IndexType> iB6({});             // empty
    std::vector<float> vB0({1, 2, 3});
    std::vector<float> vB1({1, 2, 3, 4, 5, 6, 7, 8, 9});
    std::vector<float> vB2({1, 2, 3, 7, 8, 9});
    std::vector<float> vB3({7, 8, 9, 4, 5, 6});
    std::vector<float> vB4({4, 5, 6, 4, 5, 6});
    std::vector<float> vB6;

    auto A = graph->param("A", {4, 3}, inits::from_vector(vA));
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

    std::vector<float> vA({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    std::vector<IndexType> iB0({0});            // first column
    std::vector<IndexType> iB1({0, 1, 2});      // several consecutive columns
    std::vector<IndexType> iB2({0, 2});         // two nonconsecutive columns
    std::vector<IndexType> iB3({2, 1});         // reversed order
    std::vector<IndexType> iB4({1, 1});         // repeated columns
    std::vector<IndexType> iB5({0, 1, 2, 3});   // identity
    std::vector<IndexType> iB6({});             // empty

    std::vector<float> vB0({1, 5, 9});
    std::vector<float> vB1({1, 2, 3, 5, 6, 7, 9, 10, 11});
    std::vector<float> vB2({1, 3, 5, 7, 9, 11});
    std::vector<float> vB3({3, 2, 7, 6, 11, 10});
    std::vector<float> vB4({2, 2, 6, 6, 10, 10});
    std::vector<float> vB6;

    auto A = graph->param("A", {3, 4}, inits::from_vector(vA));
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
    std::vector<float> values2;

    std::vector<float> vA({0, .3333, -.2, -.3, 0, 4.5, 5.2, -10, 101.45, -100.05, 0, 1.05e-5});
    std::vector<IndexType> idx({0, 1});

    auto A1 = graph->param("4x3", {4,3}, inits::from_vector(vA));
    auto B1 = rows(transpose(A1), idx);
    auto C1 = transpose(cols(A1, idx));
    auto A2 = graph->param("6x2", {6,2}, inits::from_vector(vA));
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

  SECTION("select operator") {
    using Indices = std::vector<IndexType>;

    graph->clear();
    values.clear();

    std::vector<float> in({1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12});
    std::vector<float> vB1({1, -2, 3});
    std::vector<float> vB2({1, -4, 7, -10});
    std::vector<float> vB3({-2, 5, -8, 11});
    std::vector<float> vB4({1, -2, 3, -4, 5, -6});
    std::vector<float> vD1(vB4);
    std::vector<float> vD2({5, -6, 11, -12});
    std::vector<float> vD3({1, -2, 5, -6, 7, -8, 11, -12});

    auto A = graph->param("4x3", {4,3}, inits::from_vector(in));
    auto B1 = select(A, Indices({0}), 0);
    auto B2 = select(A, Indices({0}), 1);
    auto B3 = select(A, Indices({1}), -1);
    auto B4 = select(A, Indices({0, 1}), 0);

    auto C = graph->param("2x3x2", {2, 3, 2}, inits::from_vector(in));
    auto D1 = select(C, Indices({0}), 0);
    auto D2 = select(C, Indices({2}), -2);
    auto D3 = select(C, Indices({0,2}), 1);
    graph->forward();

    CHECK(B1->shape() == Shape({1, 3}));
    B1->val()->get(values);
    CHECK( values == vB1 );

    CHECK(B2->shape() == Shape({4, 1}));
    B2->val()->get(values);
    CHECK( values == vB2 );

    CHECK(B3->shape() == Shape({4, 1}));
    B3->val()->get(values);
    CHECK( values == vB3 );

    CHECK(B4->shape() == Shape({2, 3}));
    B4->val()->get(values);
    CHECK( values == vB4 );

    values.clear();

    CHECK(D1->shape() == Shape({1, 3, 2}));
    D1->val()->get(values);
    CHECK( values == vD1 );

    CHECK(D2->shape() == Shape({2, 1, 2}));
    D2->val()->get(values);
    CHECK( values == vD2 );

    CHECK(D3->shape() == Shape({2, 2, 2}));
    D3->val()->get(values);
    CHECK( values == vD3 );
  }

  SECTION("rows/cols as select operations") {
    graph->clear();
    values.clear();
    std::vector<float> values2;

    std::vector<float> vA({0, .3333, -.2, -.3, 0, 4.5, 5.2, -10, 101.45, -100.05, 0, 1.05e-5});
    std::vector<IndexType> idx({0, 2});

    auto A = graph->param("4x3", {4, 3}, inits::from_vector(vA));
    auto B1 = rows(A, idx);
    auto B2 = select(A, idx, 0);
    auto C1 = cols(A, idx);
    auto C2 = select(A, idx, 1);
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
}

#ifdef CUDA_FOUND
TEST_CASE("Expression graph supports basic math operations (gpu)", "[operator]") {
  tests(DeviceType::gpu);
}
#endif

#ifdef BLAS_FOUND
TEST_CASE("Expression graph supports basic math operations (cpu)", "[operator]") {
  tests(DeviceType::cpu);
}
#endif
