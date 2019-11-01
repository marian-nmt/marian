#include "catch.hpp"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"

#ifdef CUDA_FOUND
#include "tensors/gpu/backend.h"
#endif

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
    graph->setDefaultElementType(type);
    graph->setDevice({0, DeviceType::gpu});
    graph->reserveWorkspaceMB(4);

    if(type == Type::float16) {
      auto gpuBackend = std::dynamic_pointer_cast<gpu::Backend>(graph->getBackend());
      if(gpuBackend) {
        auto cudaCompute = gpuBackend->getCudaComputeCapability();
        if(cudaCompute.major < 6) continue;
      }
    }

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
