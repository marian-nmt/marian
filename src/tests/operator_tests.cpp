#include "catch.hpp"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"

using namespace marian;

TEST_CASE("Expression graph supports basic math operations", "[operator]") {
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

    auto A = graph->param("A", {4, 3}, keywords::init = inits::from_vector(vA));
    auto B = graph->param("B", {3, 2}, keywords::init = inits::from_vector(vB));
    auto C = dot(A, B);
    graph->forward();

    REQUIRE(C->shape() == Shape({4, 2}));
    C->val()->get(values);
    REQUIRE(values == vC);
  }

  SECTION("scalar multiplication") {
    graph->clear();
    values.clear();
    std::vector<float> vB2({2, 4, 6, 8, 10, 12});

    auto B = graph->param("B", {3, 2}, keywords::init = inits::from_vector(vB));
    auto B2 = B * 2.0f;
    graph->forward();

    REQUIRE(B2->shape() == Shape({3, 2}));
    B2->val()->get(values);
    REQUIRE(values == vB2);
  }
}
