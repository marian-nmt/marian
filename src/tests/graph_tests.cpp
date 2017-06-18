#include "catch.hpp"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"

using namespace marian;

TEST_CASE("Graph device is set", "[graph]") {
  auto graph = New<ExpressionGraph>();

  graph->setDevice(0);
  REQUIRE(graph->getDevice() == 0);
  graph->setDevice(1);
  REQUIRE(graph->getDevice() == 1);
}

TEST_CASE("Expression graph can be initialized with constant values",
          "[graph]") {
  auto graph = New<ExpressionGraph>();
  graph->setDevice(0);
  graph->reserveWorkspaceMB(4);

  std::vector<float> values;

  SECTION("initializing with zeros") {
    graph->clear();
    values.clear();
    auto zeros = graph->param("0s", {2, 5}, keywords::init = inits::zeros);
    graph->forward();

    zeros->val()->get(values);
    REQUIRE(values == std::vector<float>(10, 0.0f));
  }

  SECTION("initializing with ones") {
    graph->clear();
    values.clear();
    auto ones = graph->param("1s", {2, 5}, keywords::init = inits::ones);
    graph->forward();

    ones->val()->get(values);
    REQUIRE(values == std::vector<float>(10, 1.0f));
  }

  SECTION("initializing from vector") {
    graph->clear();
    values.clear();
    std::vector<float> v({1, 2, 3, 4, 5, 6});
    auto vals = graph->param("vs", {2, 3}, keywords::init = inits::from_vector(v));
    graph->forward();

    REQUIRE(values.empty());
    vals->val()->get(values);
    REQUIRE(values == v);
  }
}
