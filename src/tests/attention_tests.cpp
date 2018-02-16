#include "catch.hpp"
#include "marian.h"

#include "rnn/rnn.h"
#include "rnn/constructors.h"
#include "rnn/attention.h"

using namespace marian;

TEST_CASE("Model components, Attention", "[attention]") {

  auto floatApprox = [](float x, float y) { return x == Approx(y).epsilon(0.01); };

  std::vector<size_t> vWords = {
    43, 2, 83, 78,
    6, 38, 80, 40,
    40, 70, 26, 60,
    106, 13, 111, 32,
    126, 62, 115, 72,
    127, 82, 55, 0,
    86, 0, 124, 0,
    0, 0, 0, 0
  };

  std::vector<size_t> vMask = {
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
    Config::seed = 1234;

    auto graph = New<ExpressionGraph>();
    graph->setDevice({0, DeviceType::gpu});
    graph->reserveWorkspaceMB(16);

    std::vector<float> values;

    int dimEmb = 16;
    int dimBatch = 4;
    int dimTime = 8;

    auto emb = graph->param("Embeddings",
                            {128, dimEmb},
                            keywords::init=inits::glorot_uniform);

    auto input = reshape(rows(emb, vWords), {dimTime, dimBatch, dimEmb});
    auto mask = graph->constant({dimTime, dimBatch, 1},
                                keywords::init=inits::from_vector(vMask));

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
                                     keywords::init=inits::from_vector(vState)),
                      nullptr});

    auto aligned = att->apply(state);

    graph->forward();

    CHECK(aligned->shape() == Shape({1, 1, 4, 8}));

    std::vector<float> vAligned({
      -0.0340144, -0.0146283, -0.00580449, -0.0407178,
      0.0188505, 0.0338587, 0.0208132, 0.0693136,
      -0.0112407, -0.0281177, 0.0645477, 0.0274245,
      0.0360024, -0.0322742, -0.00826242, 0.0249615,
      -0.0400672, 0.00215977, -0.0283565, -0.0179272,
      -0.0283309, 0.00586264, -0.00111255, 0.0732812,
      0.0277454, -0.0299964, 0.0428718, 0.0431121,
      0.0342281, -0.0223563, 0.00132206, -0.00461199
    });

    aligned->val()->get(values);

    //for(int i = 0; i < values.size(); ++i) {
    //  if(i && i % 4 == 0)
    //    std::cout << std::endl;
    //  std::cout << values[i] << ", ";
    //}

    CHECK( std::equal(values.begin(), values.end(),
                      vAligned.begin(), floatApprox) );
  }
}
