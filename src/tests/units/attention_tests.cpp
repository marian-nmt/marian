#include "catch.hpp"
#include "marian.h"

#ifdef CUDA_FOUND
#include "tensors/gpu/backend.h"
#endif

#include "rnn/rnn.h"
#include "rnn/constructors.h"
#include "rnn/attention.h"

using namespace marian;

template <typename T>
void tests(DeviceType type, Type floatType = Type::float32) {

// Checking for FP16 support and skipping if not supported.
#ifdef CUDA_FOUND
  if(type == DeviceType::gpu && floatType == Type::float16) {
    auto gpuBackend = New<gpu::Backend>(DeviceId({0, type}), /*seed=*/1234);
    auto cudaCompute = gpuBackend->getCudaComputeCapability();
    if(cudaCompute.major < 6) return;
  }
#endif

  auto floatApprox = [](T x, T y) { return x == Approx(y).margin(0.001f); };

  Config::seed = 1234;

  std::vector<IndexType> vWords = {
    43, 2, 83, 78,
    6, 38, 80, 40,
    40, 70, 26, 60,
    106, 13, 111, 32,
    126, 62, 115, 72,
    127, 82, 55, 0,
    86, 0, 124, 0,
    0, 0, 0, 0
  };

  std::vector<T> vMask = {
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 0,
    1, 0, 1, 0,
  };

  SECTION("Attention over encoder context") {
    auto graph = New<ExpressionGraph>();
    graph->setDefaultElementType(floatType);
    graph->setDevice({0, type});
    graph->reserveWorkspaceMB(16);

    std::vector<T> values;

    int dimEmb = 16;
    int dimBatch = 4;
    int dimTime = 8;

    auto emb = graph->param("Embeddings",
                            {128, dimEmb},
                            inits::glorotUniform());

    auto input = reshape(rows(emb, vWords), {dimTime, dimBatch, dimEmb});
    auto mask = graph->constant({dimTime, dimBatch, 1},
                                inits::fromVector(vMask));

    auto rnn = rnn::rnn()         //
          ("prefix", "rnntest")        //
          ("type", "gru")              //
          ("dimInput", 16)             //
          ("dimState", 8)              //
          .push_back(rnn::cell()) //
          .construct(graph);

    auto context = rnn->transduce(input, mask);

    auto encState = New<EncoderState>(context, mask, nullptr);

    auto options = New<Options>();
    options->set("dimState", 16);
    options->set("prefix", "rnntest_att");

    auto att = New<rnn::Attention>(graph, options, encState);

    std::vector<T> vState(64);
    std::generate(vState.begin(), vState.end(),
                  [](){ static int n = -32; return n++ / 64.f; });

    rnn::State state({graph->constant({1, 1, 4, 16},
                                     inits::fromVector(vState)),
                      nullptr});

    auto aligned = att->apply(state);

    graph->forward();

    CHECK(aligned->shape() == Shape({1, 1, 4, 8}));

#ifdef CUDA_FOUND
    std::vector<T> vAligned({
      0.0396688, -0.0124071, -0.0159668, -0.00080064,
      -0.0132853, 0.0240206, 0.0744701, -0.0248388,
      0.0258906, -0.00868394, -0.0374499, 0.0357639,
      -0.00104548, -0.0287227, 0.0969243, -0.0394901,
      0.012359, 0.0147176, -0.00715986, 0.0294099,
      -0.0116097, 0.0325059, 0.0392856, -0.00318991,
      0.0257503, 0.0406036, 0.0294813, 0.0753923,
      -0.0330807, 0.018745, 0.0341848, -0.0111661
    });
#else
    std::vector<T> vAligned({
      -0.061056, 0.0262615, -0.0393096, 0.115902,
      0.0941305, 0.00475613, -0.0159573, 0.00293181,
      -0.0919751, -0.018913, 0.00927365, -0.000343846,
      0.0272919, -0.068427, 0.0633098, -0.00352522,
      -0.0548061, -0.0233574, 0.0149257, -0.0348666,
      0.00910941, -0.0690751, -0.00950329, -0.028282,
      0.0146638, 0.000398567, 0.0439999, 0.00196685,
      -0.0649105, 0.0890289, 0.0288681, 0.0055663
    });
#endif

    CHECK( std::equal(values.begin(), values.end(),
                      vAligned.begin(), floatApprox) );
  }
}

#ifdef CUDA_FOUND
TEST_CASE("Model components, Attention (gpu)", "[attention]") {
  tests<float>(DeviceType::gpu);
}

#if COMPILE_FP16
TEST_CASE("Model components, Attention (gpu, fp16)", "[attention]") {
  tests<float16>(DeviceType::gpu, Type::float16);
}
#endif
#endif

#ifdef BLAS_FOUND
TEST_CASE("Model components, Attention (cpu)", "[attention]") {
  tests<float>(DeviceType::cpu);
}
#endif
