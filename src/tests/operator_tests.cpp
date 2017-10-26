#include "catch.hpp"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"

using namespace marian;

TEST_CASE("Expression graph supports basic math operations", "[operator]") {

  auto floatApprox = [](float x, float y) -> bool { return x == Approx(y); };

  auto graph = New<ExpressionGraph>();
  graph->setDevice(0);
  graph->reserveWorkspaceMB(16);

  std::vector<float> vA({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  std::vector<float> vB({1, 2, 3, 4, 5, 6});
  std::vector<float> values;

  SECTION("dot product") {
    graph->clear();
    values.clear();
    std::vector<float> vC({22, 28, 49, 64, 76, 100, 103, 136});

    auto A = graph->param("A", {2, 3, 2}, keywords::init = inits::from_vector(vA));
    auto B = graph->param("B", {3, 2}, keywords::init = inits::from_vector(vB));
    auto C = dot(A, B);
    graph->forward();

    CHECK(C->shape() == Shape({2, 2, 2}));
    C->val()->get(values);
    CHECK(values == vC);
  }

  SECTION("scalar multiplication") {
    graph->clear();
    values.clear();
    std::vector<float> vB2({2, 4, 6, 8, 10, 12});

    auto B = graph->param("B", {3, 2}, keywords::init = inits::from_vector(vB));
    auto B2 = B * 2.0f;
    graph->forward();

    CHECK(B2->shape() == Shape({3, 2}));
    B2->val()->get(values);
    CHECK(values == vB2);
  }

  SECTION("softmax and logsoftmax") {
    graph->clear();
    values.clear();
    std::vector<float> in({-.2, -.3, 4.5, 5.2, -10, 101.45, -100.05, 1.05e-5});

    std::vector<float> smOut({ 0.52498f, 0.47502f, 0.33181f, 0.66819f,
                               0.0f, 1.0f, 0.0f, 1.0f });

    std::vector<float> lsmOut({ -0.6444f, -0.7444f, -1.10319f, -0.40319f,
                                -111.45f, 0.0f, -100.05001f, 0.0f });

    auto input = graph->constant({2, 2, 2}, keywords::init = inits::from_vector(in));

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

  SECTION("elementwise binary operators with broadcasting") {
    graph->clear();
    values.clear();

    std::vector<float> vA({1, -2, 3, -4});
    std::vector<float> vB({0.5, 1.5});

    std::vector<float> vAdd({1.5, -0.5, 3.5, -2.5});
    std::vector<float> vMinus({-0.5, 3.5, -2.5, 5.5});
    std::vector<float> vMult({0.5, -3.0, 1.5, -6.0});
    std::vector<float> vDiv({2.0f, -1.33333f, 6.0f, -2.66667f});

    auto a = graph->constant({2, 2, 1}, keywords::init = inits::from_vector(vA));
    auto b = graph->constant({1, 2, 1}, keywords::init = inits::from_vector(vB));

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
    std::vector<float> vT5({1, 5, 3, 7, 2, 6, 4, 8});

    auto a = graph->constant({2, 4}, keywords::init = inits::from_vector(vA));

    auto t1 = transpose(a);
    auto t2 = transpose(t1);
    auto t3 = transpose(reshape(t1, {2, 2, 2}));
    auto t4 = transpose(reshape(a, {2, 2, 1, 2}), {2, 3, 0, 1});
    auto t5 = transpose(reshape(a, {2, 2, 1, 2}), {0, 3, 1, 2});

    graph->forward();

    CHECK(t1->shape() == Shape({4, 2}));
    CHECK(t2->shape() == Shape({2, 4}));
    CHECK(t3->shape() == Shape({2, 2, 2}));
    CHECK(t4->shape() == Shape({1, 2, 2, 2}));
    CHECK(t5->shape() == Shape({2, 2, 2, 1}));

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
}
