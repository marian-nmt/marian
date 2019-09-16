#include "catch.hpp"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"

using namespace marian;

#ifdef CUDA_FOUND
TEST_CASE("Graph device is set", "[graph]") {
  auto graph = New<ExpressionGraph>();    
  graph->setDevice({0, DeviceType::gpu});
  
  DeviceId testId{0, DeviceType::gpu};
  REQUIRE(graph->getDeviceId() == testId);
}

TEST_CASE("Expression graph can be initialized with constant values",
          "[graph]") {

  for(auto type : std::vector<Type>({Type::float32, Type::float16})) {
    auto graph = New<ExpressionGraph>();
    graph->setDevice({0, DeviceType::gpu});
    graph->setParameterType(type);
    graph->reserveWorkspaceMB(4);

    std::vector<float> values;

    SECTION("initializing with zeros") {
      graph->clear();
      values.clear();
      auto zeros = graph->param("0s", {2, 5}, inits::zeros());
      graph->forward();

      zeros->val()->get(values);
      REQUIRE(values == std::vector<float>(10, 0.0f));
    }

    SECTION("initializing with ones") {
      graph->clear();
      values.clear();
      auto ones = graph->param("1s", {2, 5}, inits::ones());
      graph->forward();

      ones->val()->get(values);
      REQUIRE(values == std::vector<float>(10, 1.0f));
    }

    SECTION("initializing from vector") {
      graph->clear();
      values.clear();
      std::vector<float> v({1, 2, 3, 4, 5, 6});
      auto vals = graph->param("vs", {2, 3}, inits::fromVector(v));
      graph->forward();

      REQUIRE(values.empty());
      vals->val()->get(values);
      REQUIRE(values == v);
    }

    SECTION("initializing float16 node from vector") {
      // This does not test fp16 computation, only float16 to float32 conversion
      graph->clear();
      std::vector<float16> values16;
      std::vector<float16> v({1, 2, 3, 4, 5, 6});
      auto vals1 = graph->param("vs1", {2, 3}, inits::fromVector(v), Type::float16);
      auto vals2 = graph->param("vs2", {2, 3}, inits::fromValue(5), Type::float16);

      debug(vals1 + vals2);
      graph->forward();

      REQUIRE(values.empty());
      vals1->val()->get(values16);
      REQUIRE(values16 == v);
    }
  }
}
#endif

TEST_CASE("Graph device is set (cpu)", "[graph]") {
  auto graph = New<ExpressionGraph>();

  graph->setDevice({0, DeviceType::cpu});

  DeviceId testId{0, DeviceType::cpu};
  REQUIRE(graph->getDeviceId() == testId);
}

TEST_CASE("Expression graph can be initialized with constant values (cpu)",
          "[graph]") {
  auto graph = New<ExpressionGraph>();
  graph->setDevice({0, DeviceType::cpu});
  graph->reserveWorkspaceMB(4);

  std::vector<float> values;

  SECTION("initializing with zero (cpu)") {
    graph->clear();
    values.clear();
    auto zeros = graph->param("0s", {2, 5}, inits::zeros());
    graph->forward();

    zeros->val()->get(values);
    REQUIRE(values == std::vector<float>(10, 0.0f));
  }

  SECTION("initializing with ones (cpu)") {
    graph->clear();
    values.clear();
    auto ones = graph->param("1s", {2, 5}, inits::ones());
    graph->forward();

    ones->val()->get(values);
    REQUIRE(values == std::vector<float>(10, 1.0f));
  }

  SECTION("initializing from vector (cpu)") {
    graph->clear();
    values.clear();
    std::vector<float> v({1, 2, 3, 4, 5, 6});
    auto vals = graph->param("vs", {2, 3}, inits::fromVector(v));
    graph->forward();

    REQUIRE(values.empty());
    vals->val()->get(values);
    REQUIRE(values == v);
  }
}
