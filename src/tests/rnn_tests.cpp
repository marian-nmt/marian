#include "catch.hpp"
#include "marian.h"

#include "rnn/rnn.h"
#include "rnn/constructors.h"

using namespace marian;

void tests(DeviceType type) {
  auto floatApprox = [](float x, float y) { return x == Approx(y).epsilon(0.01); };

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

#ifdef CUDA_FOUND
    std::vector<float> vOutput({
      0.637288, 0.906478, 0.603604, 0.152291,
      -0.5333, -0.854558, 0.458454, -0.179582,
      0.736857, 0.964425, 0.43848, 0.0261131,
      -0.533659, -0.733491, -0.953666, -0.965717
    });
#else
    std::vector<float> vOutput({
      0.671833, -0.944205, -0.569858, 0.902453,
      -0.166113, -0.109117, -0.247899, 0.150481,
      0.0531123, -0.263492, 0.474677, 0.423597,
      -0.0685829, -0.904944, -0.851515, 0.911637
    });
#endif

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
                                  /*axis =*/ input->shape().size() - 1);

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
    auto contextSum1 = sum(context1, /*axis*/2);

    auto context2 = buildRnn("enc2", input, mask, dimRnn, 2, 2);
    auto contextSum2 = sum(context2, /*axis*/2);

    // @TODO: why is this numerically instable on different machines?
    //auto context3 = buildRnn("enc3", input, mask,
    //                         dimRnn, 4, 4,
    //                         "alternating", "lstm",
    //                         true, true);
    //auto contextSum3 = sum(context3, /*axis*/1);

    graph->forward();

    CHECK(context1->shape() == Shape({dimTime, dimBatch, 2 * dimRnn}));
    CHECK(contextSum1->shape() == Shape({dimTime, dimBatch, 1}));

#ifdef CUDA_FOUND
    std::vector<float> vContextSum1({
      -0.110829, -0.510232, 0.265193, 0.194025,
      -0.242112, 0.185029, 0.0530527, 0.359336,
      0.60218, 0.46511, -0.240092, 0.100453,
      0.609049, 0.491292, -0.32164, -0.482791,
      -0.0203674, 0.602481, 0.0259332, -0.477771,
      0.436479, 0.338244, 0.00689805, 0.155251,
      0.487821, 0.531054, 0.593997, 0.0469481,
      0.360119, 0.422752, 0.55825, 0.0469481
    });
#else
    std::vector<float> vContextSum1({
      -0.129259, -0.433844, 0.132807, -0.63203,
      -0.0185539, -0.478032, 0.065215, 0.252358,
      0.18597, 0.033865, -0.33843, 0.396619,
      -0.186539, -1.35947, -0.0856928, 0.70514,
      -0.234371, -1.13612, 0.140888, 1.1614,
      -0.822445, -0.219703, 0.645387, 0.465694,
      -0.489863, -0.186675, -1.3761, 0.402658,
      0.0161895, -0.249712, -0.18665, 0.402658
    });
#endif

    contextSum1->val()->get(values);
    CHECK( std::equal(values.begin(), values.end(),
                      vContextSum1.begin(), floatApprox) );

    CHECK(context2->shape() == Shape({dimTime, dimBatch, 2 * dimRnn}));
    CHECK(contextSum2->shape() == Shape({dimTime, dimBatch, 1}));

#ifdef CUDA_FOUND
    std::vector<float> vContextSum2({
      -0.0282316, 0.0219561, -0.012136, 0.0206684,
      -0.0755229, 0.00091961, 0.0206883, 0.0176061,
      -0.0272491, 0.0833994, 0.0279131, 0.0170246,
      0.0922298, 0.2057, 0.0155544, -0.0299952,
      0.0907423, 0.196588, 0.0820211, -0.0345194,
      0.0284086, 0.109867, 0.057752, 0.0592283,
      0.0918175, 0.0818634, 0.0174914, 0.0548368,
      0.123207, 0.0774718, 0.0741554, 0.0548368
    });
#else
    std::vector<float> vContextSum2({
      0.11039, -0.018452, 0.0153041, -0.00669695,
      0.0136421, 0.0140959, 0.0318346, -0.0315847,
      0.018701, -0.0116785, -0.0056591, -0.0473642,
      -0.0525274, -0.0731631, -0.0295185, -0.0131082,
      0.0446591, -0.0925775, 0.0465199, -0.00508343,
      0.142556, 0.00457316, 0.0780754, -0.0114262,
      0.102179, 0.0107549, 0.0902212, -0.016181,
      0.0501191, 0.006, 0.0344533, -0.016181
    });
#endif

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
