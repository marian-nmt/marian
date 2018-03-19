#include "catch.hpp"
#include "marian.h"

#include "rnn/rnn.h"
#include "rnn/constructors.h"

using namespace marian;

void tests(DeviceType type) {
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

  SECTION("Simple RNN") {
    Config::seed = 1234;

    auto graph = New<ExpressionGraph>();
    graph->setDevice({0, type});
    graph->reserveWorkspaceMB(16);

    std::vector<float> values;

    auto input = graph->constant({4, 1, 4},
                                 inits::glorot_uniform);

    auto rnn = rnn::rnn(graph)         //
          ("prefix", "rnntest")        //
          ("type", "tanh")             //
          ("dimInput", 4)              //
          ("dimState", 4)              //
          .push_back(rnn::cell(graph)) //
          .construct();

    auto output = rnn->transduce(input);

    graph->forward();

    CHECK(output->shape() == Shape({4, 1, 4}));

    std::vector<float> vOutput({
      0.108774, 0.237905, -0.819769, -0.212601,
      -0.684652, 0.455977, 0.504662, -0.184837,
      0.769393, 0.28449, -0.200995, -0.260122,
      -0.324909, -0.337419, -0.959819, 0.559088
    });

    output->val()->get(values);
    CHECK( std::equal(values.begin(), values.end(),
                      vOutput.begin(), floatApprox) );
  }

  SECTION("S2S-style encoder") {
    Config::seed = 1234;

    auto graph = New<ExpressionGraph>();
    graph->setDevice({0, type});
    graph->reserveWorkspaceMB(16);

    std::vector<float> values;

    auto buildRnn = [&graph] (std::string prefix,
                              Expr input, Expr mask,
                              int dimRnn=32,
                              int depth=1,
                              int cellDepth=1,
                              std::string type="bidirectional",
                              std::string cellType="gru",
                              bool layerNorm=false,
                              bool skip=false) {

      int dimEmb = input->shape()[-1];

      int first, second;
      if(type == "bidirectional" || type == "alternating") {
        // build two separate stacks, concatenate top output
        first = depth;
        second = 0;
      } else {
        // build 1-layer bidirectional stack, concatenate,
        // build n-1 layer unidirectional stack
        first = 1;
        second = depth - first;
      }

      auto forward = type == "alternating" ? rnn::dir::alternating_forward
                                           : rnn::dir::forward;

      auto backward = type == "alternating" ? rnn::dir::alternating_backward
                                            : rnn::dir::backward;

      using namespace keywords;

      auto rnnFw = rnn::rnn(graph)           //
          ("type", cellType)                 //
          ("direction", forward)             //
          ("dimInput", dimEmb)               //
          ("dimState", dimRnn)               //
          ("layer-normalization", layerNorm) //
          ("skip", skip);

      for(int i = 1; i <= first; ++i) {
        auto stacked = rnn::stacked_cell(graph);
        for(int j = 1; j <= cellDepth; ++j) {
          std::string paramPrefix = prefix + "_bi";
          if(i > 1)
            paramPrefix += "_l" + std::to_string(i);
          if(i > 1 || j > 1)
            paramPrefix += "_cell" + std::to_string(j);

          stacked.push_back(rnn::cell(graph)("prefix", paramPrefix));
        }
        rnnFw.push_back(stacked);
      }

      auto rnnBw = rnn::rnn(graph)            //
          ("type", cellType)                  //
          ("direction", backward)             //
          ("dimInput", dimEmb)                //
          ("dimState", dimRnn)                //
          ("layer-normalization", layerNorm)  //
          ("skip", skip);

      for(int i = 1; i <= first; ++i) {
        auto stacked = rnn::stacked_cell(graph);
        for(int j = 1; j <= cellDepth; ++j) {
          std::string paramPrefix = prefix + "_bi_r";
          if(i > 1)
            paramPrefix += "_l" + std::to_string(i);
          if(i > 1 || j > 1)
            paramPrefix += "_cell" + std::to_string(j);

          stacked.push_back(rnn::cell(graph)("prefix", paramPrefix));
        }
        rnnBw.push_back(stacked);
      }

      auto context = concatenate({rnnFw->transduce(input, mask),
                                  rnnBw->transduce(input, mask)},
                                 axis = input->shape().size() - 1);

      if(second > 0) {
        // add more layers (unidirectional) by transducing the output of the
        // previous bidirectional RNN through multiple layers

        // construct RNN first
        auto rnnUni = rnn::rnn(graph)           //
            ("type", cellType)                  //
            ("dimInput", 2 * dimRnn)            //
            ("dimState", dimRnn)                //
            ("layer-normalization", layerNorm)  //
            ("skip", skip);

        for(int i = first + 1; i <= second + first; ++i) {
          auto stacked = rnn::stacked_cell(graph);
          for(int j = 1; j <= cellDepth; ++j) {
            std::string paramPrefix = prefix + "_l" + std::to_string(i) + "_cell"
                                      + std::to_string(j);
            stacked.push_back(rnn::cell(graph)("prefix", paramPrefix));
          }
          rnnUni.push_back(stacked);
        }

        // transduce context to new context
        context = rnnUni->transduce(context);
      }
      return context;
    };

    int dimEmb = 16;
    int dimBatch = 4;
    int dimTime = 8;

    auto emb = graph->param("Embeddings",
                            {128, dimEmb},
                            inits::glorot_uniform);

    auto input = reshape(rows(emb, vWords), {dimTime, dimBatch, dimEmb});
    auto mask = graph->constant({dimTime, dimBatch, 1},
                                inits::from_vector(vMask));

    int dimRnn = 32;
    auto context1 = buildRnn("enc1", input, mask, dimRnn);
    auto contextSum1 = sum(context1, keywords::axis=2);

    auto context2 = buildRnn("enc2", input, mask, dimRnn, 2, 2);
    auto contextSum2 = sum(context2, keywords::axis=2);

    // @TODO: why is this numerically instable on different machines?
    //auto context3 = buildRnn("enc3", input, mask,
    //                         dimRnn, 4, 4,
    //                         "alternating", "lstm",
    //                         true, true);
    //auto contextSum3 = sum(context3, keywords::axis=1);

    graph->forward();

    CHECK(context1->shape() == Shape({dimTime, dimBatch, 2 * dimRnn}));
    CHECK(contextSum1->shape() == Shape({dimTime, dimBatch, 1}));

    std::vector<float> vContextSum1({
      0.14076, -0.102, 0.22832, -0.42283,
      -0.15911, 0.33222, 0.74858, -0.59844,
      -0.70797, -0.12694, -0.14322, 0.25016,
      -0.91476, 0.39106, -0.75152, -0.02236,
      -0.59753, 0.17417, -0.24941, -0.36464,
      -0.62975, 0.35372, 0.12781, -0.79948,
      -0.33747, -0.54613, 0.32809, -0.63282,
      -0.78209, -0.37947, -0.50397, -0.63282
    });

    contextSum1->val()->get(values);

    CHECK( std::equal(values.begin(), values.end(),
                      vContextSum1.begin(), floatApprox) );

    CHECK(context2->shape() == Shape({dimTime, dimBatch, 2 * dimRnn}));
    CHECK(contextSum2->shape() == Shape({dimTime, dimBatch, 1}));

    std::vector<float> vContextSum2({
      -0.0168112, -0.0524664, -0.0196701, -0.0118004,
      0.00975164, -0.0470996, -0.014982, -0.0248614,
      -0.0110038, 0.00297422, -0.00327533, 0.0175996,
      0.0319444, 0.0196884, -0.0436654, -0.0257596,
      0.0131209, -0.0533302, -0.058655, 0.0666001,
      0.00823802, 0.0133473, -0.00715647, 0.119427,
      0.0282871, 0.104641, -0.0271743, 0.0658893,
      0.0687114, 0.0511032, 0.0673459, 0.0658893
    });

    contextSum2->val()->get(values);
    CHECK( std::equal(values.begin(), values.end(),
                      vContextSum2.begin(), floatApprox) );

    //CHECK(context3->shape() == Shape({dimBatch, 2 * dimRnn, dimTime}));
    //CHECK(contextSum3->shape() == Shape({dimBatch, 1, dimTime}));
    //
    //std::vector<float> vContextSum3({
    //  1.135, 2.40939, 2.37631, 2.03765,
    //  0.0583942, -4.89241, 5.31731, -1.52973,
    //  3.52754, 1.02098, -4.05162, -1.11594,
    //  6.28777, -5.55708, -4.09155, 2.00661,
    //  -0.571597, 0.153122, -3.46678, 0.0771322,
    //  -2.10868, -3.58708, -6.3728, 1.77672,
    //  -10.9653, -2.02775, -5.70838, 0.944819,
    //  -1.81441, -1.84383, 0.790335, 0.941206
    //});
    //
    //contextSum3->val()->get(values);
    //
    //for(int i = 0; i < values.size(); ++i) {
    //  if(i && i % 4 == 0)
    //    std::cout << std::endl;
    //
    //  std::cout << values[i] << ", ";
    //}
    //
    //CHECK( std::equal(values.begin(), values.end(),
    //                  vContextSum3.begin(), floatApprox) );
  }
}

#ifdef CUDA_FOUND
TEST_CASE("Model components, RNN etc. (gpu)", "[model]") {
  tests(DeviceType::gpu);
}
#endif

#ifdef BLAS_FOUND
TEST_CASE("Model components, RNN etc. (cpu)", "[model]") {
  tests(DeviceType::cpu);
}
#endif
