#include "catch.hpp"
#include "marian.h"

#include "rnn/rnn.h"
#include "rnn/constructors.h"
#include "rnn/attention.h"

using namespace marian;

void tests(DeviceType type) {
  auto floatApprox = [](float x, float y) { return x == Approx(y).epsilon(0.01); };

  Config::seed = 1234;

  Words vWords = {
    43, 2, 83, 78,
    6, 38, 80, 40,
    40, 70, 26, 60,
    106, 13, 111, 32,
    126, 62, 115, 72,
    127, 82, 55, 0,
    86, 0, 124, 0,
    0, 0, 0, 0
  };

  std::vector<float> vMask = {
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
    graph->setDevice({0, type});
    graph->reserveWorkspaceMB(16);

    std::vector<float> values;

    int dimEmb = 16;
    int dimBatch = 4;
    int dimTime = 8;

    auto emb = graph->param("Embeddings",
                            {128, dimEmb},
                            inits::glorot_uniform);

    auto input = reshape(rows(emb, vWords), {dimTime, dimBatch, dimEmb});
    auto mask = graph->constant({dimTime, dimBatch, 1},
                                inits::from_vector(vMask));

    auto rnn = rnn::rnn(graph)         //
          ("prefix", "rnntest")        //
          ("type", "gru")              //
          ("dimInput", 16)             //
          ("dimState", 8)              //
          .push_back(rnn::cell(graph)) //
          .construct();

    auto context = rnn->transduce(input, mask);

    auto encState = New<EncoderState>(context, mask, nullptr);

    auto options = New<Options>();
    options->set("dimState", 16);
    options->set("prefix", "rnntest_att");

    auto att = New<rnn::Attention>(graph, options, encState);

    std::vector<float> vState(64);
    std::generate(vState.begin(), vState.end(),
                  [](){ static int n = -32; return n++ / 64.f; });

    rnn::State state({graph->constant({1, 1, 4, 16},
                                     inits::from_vector(vState)),
                      nullptr});

    auto aligned = att->apply(state);

    graph->forward();

    CHECK(aligned->shape() == Shape({1, 1, 4, 8}));

#ifdef CUDA_FOUND
    std::vector<float> vAligned({
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
    std::vector<float> vAligned({
      0.0472522, 0.0284729, 0.0236836, 0.0825191,
      -0.00110186, -0.0789387, -0.0278057, 0.0330174,
      0.150085, 0.0341702, 0.08339, 0.175978,
      0.0278676, 0.151767, 0.0733506, -0.0520294,
      0.125744, 0.0394306, 0.00818361, 0.132748,
      -0.0733842, 0.045595, -0.0367922, -0.0555062,
      0.00244656, -0.0687288, 0.143586, 0.0745924,
      0.0779884, -0.0396876, 0.0592336, -0.0205889
    });
#endif

    aligned->val()->get(values);
    CHECK( std::equal(values.begin(), values.end(),
                      vAligned.begin(), floatApprox) );
  }
}

#ifdef CUDA_FOUND
TEST_CASE("Model components, Attention (gpu)", "[attention]") {
  tests(DeviceType::gpu);
}
#endif

#ifdef BLAS_FOUND
TEST_CASE("Model components, Attention (cpu)", "[attention]") {
  tests(DeviceType::cpu);
}
#endif
