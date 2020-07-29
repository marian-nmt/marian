#include "catch.hpp"
#include "marian.h"

#ifdef CUDA_FOUND
#include "tensors/gpu/backend.h"
#endif

#include "rnn/rnn.h"
#include "rnn/constructors.h"

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

  auto floatApprox = [](T x, T y) { return x == Approx(y).epsilon(0.01f).scale(1.f); };

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

  SECTION("Simple RNN") {
    Config::seed = 1234;

    auto graph = New<ExpressionGraph>();
    graph->setDefaultElementType(floatType);
    graph->setDevice({0, type});
    graph->reserveWorkspaceMB(16);

    std::vector<T> values;

    auto input = graph->constant({4, 1, 4},
                                 inits::glorotUniform());

    auto rnn = rnn::rnn()         //
          ("prefix", "rnntest")   //
          ("type", "tanh")        //
          ("dimInput", 4)         //
          ("dimState", 4)         //
          .push_back(rnn::cell()) //
          .construct(graph);

    auto output = rnn->transduce(input);

    graph->forward();

    CHECK(output->shape() == Shape({4, 1, 4}));

#ifdef CUDA_FOUND
    std::vector<T> vOutput({
      0.637288, 0.906478, 0.603604, 0.152291,
      -0.5333, -0.854558, 0.458454, -0.179582,
      0.736857, 0.964425, 0.43848, 0.0261131,
      -0.533659, -0.733491, -0.953666, -0.965717
    });
#else
    std::vector<T> vOutput({
      -0.523228, 0.645143, 0.430939, 0.273439,
      -0.747293, 0.131912, 0.115222, 0.363874,
      0.367535, -0.819531, -0.313036, -0.387701,
      -0.459136, 0.962531, 0.0314726, 0.531492
    });
#endif

    output->val()->get(values);
    CHECK( std::equal(values.begin(), values.end(),
                      vOutput.begin(), floatApprox) );
  }

  SECTION("S2S-style encoder") {
    Config::seed = 1234;

    auto graph = New<ExpressionGraph>();
    graph->setDefaultElementType(floatType);
    graph->setDevice({0, type});
    graph->reserveWorkspaceMB(16);

    std::vector<T> values;

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

      auto rnnFw = rnn::rnn()                //
          ("type", cellType)                 //
          ("direction", forward)             //
          ("dimInput", dimEmb)               //
          ("dimState", dimRnn)               //
          ("layer-normalization", layerNorm) //
          ("skip", skip);

      for(int i = 1; i <= first; ++i) {
        auto stacked = rnn::stacked_cell();
        for(int j = 1; j <= cellDepth; ++j) {
          std::string paramPrefix = prefix + "_bi";
          if(i > 1)
            paramPrefix += "_l" + std::to_string(i);
          if(i > 1 || j > 1)
            paramPrefix += "_cell" + std::to_string(j);

          stacked.push_back(rnn::cell()("prefix", paramPrefix));
        }
        rnnFw.push_back(stacked);
      }


      auto rnnBw = rnn::rnn()                //
          ("type", cellType)                 //
          ("direction", backward)            //
          ("dimInput", dimEmb)               //
          ("dimState", dimRnn)               //
          ("layer-normalization", layerNorm) //
          ("skip", skip);

      for(int i = 1; i <= first; ++i) {
        auto stacked = rnn::stacked_cell();
        for(int j = 1; j <= cellDepth; ++j) {
          std::string paramPrefix = prefix + "_bi_r";
          if(i > 1)
            paramPrefix += "_l" + std::to_string(i);
          if(i > 1 || j > 1)
            paramPrefix += "_cell" + std::to_string(j);

          stacked.push_back(rnn::cell()("prefix", paramPrefix));
        }
        rnnBw.push_back(stacked);
      }

      auto context = concatenate({rnnFw.construct(graph)->transduce(input, mask),
                                  rnnBw.construct(graph)->transduce(input, mask)},
                                  /*axis =*/ (int)input->shape().size() - 1);

      if(second > 0) {
        // add more layers (unidirectional) by transducing the output of the
        // previous bidirectional RNN through multiple layers

        // construct RNN first
        auto rnnUni = rnn::rnn()               //
            ("type", cellType)                 //
            ("dimInput", 2 * dimRnn)           //
            ("dimState", dimRnn)               //
            ("layer-normalization", layerNorm) //
            ("skip", skip);

        for(int i = first + 1; i <= second + first; ++i) {
          auto stacked = rnn::stacked_cell();
          for(int j = 1; j <= cellDepth; ++j) {
            std::string paramPrefix = prefix + "_l" + std::to_string(i) + "_cell"
                                      + std::to_string(j);
            stacked.push_back(rnn::cell()("prefix", paramPrefix));
          }
          rnnUni.push_back(stacked);
        }

        // transduce context to new context
        context = rnnUni.construct(graph)->transduce(context);
      }
      return context;
    };

    int dimEmb = 16;
    int dimBatch = 4;
    int dimTime = 8;

    auto emb = graph->param("Embeddings",
                            {128, dimEmb},
                            inits::glorotUniform());

    auto input = reshape(rows(emb, vWords), {dimTime, dimBatch, dimEmb});
    auto mask = graph->constant({dimTime, dimBatch, 1},
                                inits::fromVector(vMask));

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
    std::vector<T> vContextSum1({
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
    std::vector<T> vContextSum1({
      -0.0674548, 0.383986, -0.613574, 0.226154,
      -0.819571, 0.47317, -1.39324, -0.401005,
      -0.24099, 0.64791, -0.120434, -0.818529,
      -0.312704, 0.441536, 0.199262, 0.436554,
      -0.157767, -0.277224, 0.786445, 0.777559,
      -0.213046, 0.294554, 0.507711, 0.61881,
      -0.626906, 0.440541, 0.178261, 0.765169,
      -0.290793, 0.5869, 0.313428, 0.765169
    });
#endif

    CHECK( std::equal(values.begin(), values.end(),
                      vContextSum1.begin(), floatApprox) );

    CHECK(context2->shape() == Shape({dimTime, dimBatch, 2 * dimRnn}));
    CHECK(contextSum2->shape() == Shape({dimTime, dimBatch, 1}));

#ifdef CUDA_FOUND
    std::vector<T> vContextSum2({
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
    std::vector<T> vContextSum2({
      0.0193405, -0.0580973, -0.0213983, 0.0381918,
      -0.0135365, -0.0934286, -0.0171637, 0.0198686,
      -0.0102693, -0.0865369, -0.0160779, 0.0393178,
      -0.0208074, 0.0371625, -0.031599, 0.0184805,
      0.0172931, 0.0145368, -0.0388733, 0.0226179,
      0.0270382, -0.0222009, -0.0240776, 0.018094,
      0.024001, -0.0116693, -0.0155723, 0.0574173,
      0.0544399, 0.0276539, 0.0487282, 0.0574173
    });
#endif

    CHECK( std::equal(values.begin(), values.end(),
                      vContextSum2.begin(), floatApprox) );

    //CHECK(context3->shape() == Shape({dimBatch, 2 * dimRnn, dimTime}));
    //CHECK(contextSum3->shape() == Shape({dimBatch, 1, dimTime}));
    //
    //std::vector<T> vContextSum3({
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
  tests<float>(DeviceType::gpu);
}

#if COMPILE_FP16
TEST_CASE("Model components, RNN etc. (gpu, fp16)", "[model]") {
  tests<float16>(DeviceType::gpu, Type::float16);
}
#endif
#endif

#ifdef BLAS_FOUND
TEST_CASE("Model components, RNN etc. (cpu)", "[model]") {
  tests<float>(DeviceType::cpu);
}
#endif
