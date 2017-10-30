#include "catch.hpp"
#include "marian.h"

using namespace marian;

TEST_CASE("Model components, Attention", "[attention]") {

  auto floatApprox = [](float x, float y) { return x == Approx(y); };

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
    graph->setDevice(0);
    graph->reserveWorkspaceMB(16);

    std::vector<float> values;

    int dimEmb = 16;
    int dimBatch = 4;
    int dimTime = 8;

    auto emb = graph->param("Embeddings",
                            {128, dimEmb},
                            keywords::init=inits::glorot_uniform);

    auto input = reshape(rows(emb, vWords), {dimBatch, dimEmb, dimTime});
    auto mask = graph->constant({dimBatch, 1, dimTime},
                                keywords::init=inits::from_vector(vMask));

    auto rnn = rnn::rnn(graph)         //
          ("prefix", "rnntest")        //
          ("type", "gru")              //
          ("dimInput", 16)             //
          ("dimState", 8)             //
          .push_back(rnn::cell(graph)) //
          .construct();

    auto context = rnn->transduce(input, mask);

    auto encState = New<EncoderState>(context, mask, nullptr);

    auto options = New<Options>();
    options->set("dimState", 16);
    options->set("prefix", "rnntest_att");

    auto att = New<rnn::Attention>(graph, options, encState);

    rnn::State state({graph->constant({4, 16, 1},
                                     keywords::init=inits::glorot_uniform),
                      nullptr});

    auto aligned = att->apply(state);

    graph->forward();

    CHECK(aligned->shape() == Shape({4, 8, 1}));

    std::vector<float> vAligned({
      -0.0338987, -0.0146153, -0.00579572, -0.0406731,
      0.0190828, 0.0339625, 0.0206139, 0.0693264,
      -0.0108943, -0.0284214, 0.0651774, 0.027579,
      0.0362864, -0.0320309, -0.00822382, 0.0244689,
      -0.0399934, 0.00202525, -0.0283427, -0.0180207,
      -0.0284016, 0.00583279, -0.000959444, 0.073175,
      0.0276376, -0.0298874, 0.0427888, 0.0431906,
      0.034185, -0.0224177, 0.00124145, -0.0046362
    });

    aligned->val()->get(values);
    CHECK( std::equal(values.begin(), values.end(),
                      vAligned.begin(), floatApprox) );
  }
}
