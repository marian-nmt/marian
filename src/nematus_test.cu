#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "rnn.h"
#include "batch_generator.h"
#include "param_initializers.h"
#include "optimizers.h"

#include "cnpy/cnpy.h"

using namespace marian;
using namespace keywords;
using namespace data;

typedef DeviceVector<size_t> WordBatch;
typedef std::vector<WordBatch> SentBatch;

void load(ExpressionGraphPtr g, const std::string& name) {
  auto numpy = cnpy::npz_load(name);

  auto parameters = {
    // Source word embeddings
    "Wemb",

    // GRU in encoder
    "encoder_U", "encoder_W", "encoder_b",
    "encoder_Ux", "encoder_Wx", "encoder_bx",

    // GRU in encoder, reversed
    "encoder_r_U", "encoder_r_W", "encoder_r_b",
    "encoder_r_Ux", "encoder_r_Wx", "encoder_r_bx",

    // Transformation of decoder input state
    "ff_state_W", "ff_state_b",

    // Target word embeddings
    "Wemb_dec",

    // GRU layer 1 in decoder
    "decoder_U", "decoder_W", "decoder_b",
    "decoder_Ux", "decoder_Wx", "decoder_bx",

    // Attention
    "decoder_W_comb_att", "decoder_b_att",
    "decoder_Wc_att", "decoder_U_att",

    // GRU layer 2 in decoder
    "decoder_U_nl", "decoder_Wc", "decoder_b_nl",
    "decoder_Ux_nl", "decoder_Wcx", "decoder_bx_nl",

    // Read out
    "ff_logit_lstm_W", "ff_logit_lstm_b",
    "ff_logit_prev_W", "ff_logit_prev_b",
    "ff_logit_ctx_W", "ff_logit_ctx_b",
    "ff_logit_W", "ff_logit_b",
  };

  for(auto name : parameters) {
    Shape shape;
    if(numpy[name].shape.size() == 2) {
      shape.set(0, numpy[name].shape[0]);
      shape.set(1, numpy[name].shape[1]);
    }
    else if(numpy[name].shape.size() == 1) {
      shape.set(0, 1);
      shape.set(1, numpy[name].shape[0]);
    }

    g->param(name, shape, init=from_numpy(numpy[name]));
  }
}

void construct(ExpressionGraphPtr g,
               const SentBatch& srcSentenceBatch,
               const SentBatch& trgSentenceBatch) {
  g->clear();

  int dimSrcVoc = g->get("Wemb") ? g->get("Wemb")->shape()[0] : 85000;
  int dimSrcEmb = g->get("Wemb") ? g->get("Wemb")->shape()[1] : 500;
  int dimEncState = g->get("encoder_U") ? g->get("encoder_U")->shape()[0] : 1024;

  int dimTrgVoc = g->get("Wemb_dec") ? g->get("Wemb_dec")->shape()[0] : 85000;
  int dimTrgEmb = g->get("Wemb_dec") ? g->get("Wemb_dec")->shape()[1] : 500;
  int dimDecState = g->get("decoder_U") ? g->get("decoder_U")->shape()[0] : 1024;

  int dimBatch = 1;

  /****************************** encoder *********************************/

  auto Wemb = g->param("Wemb", {dimSrcVoc, dimSrcEmb}, init=glorot_uniform);

  std::vector<Expr> inputs;
  size_t i = 0;
  for(auto& srcWordBatch : srcSentenceBatch) {
    auto x = name(rows(Wemb, srcWordBatch), "x_" + std::to_string(i++));
    inputs.push_back(x);
    dimBatch = srcWordBatch.size();
  }

  auto encoderGRU = [=](const std::string& prefix) {
    auto U = g->param(prefix + "_U", {dimEncState, 2 * dimEncState},
                      init=glorot_uniform);

    auto W = g->param(prefix + "_W", {dimSrcEmb, 2 * dimEncState},
                      init=glorot_uniform);

    auto b = g->param(prefix + "_b", {1, 2 * dimEncState}, init=zeros);

    auto Ux = g->param(prefix + "_Ux", {dimEncState, dimEncState},
                      init=glorot_uniform);

    auto Wx = g->param(prefix + "_Wx", {dimSrcEmb, dimEncState},
                       init=glorot_uniform);

    auto bx = g->param(prefix + "_bx", {1, dimEncState}, init=zeros);

    ParametersGRUFast encParams;
    encParams.U = concatenate({U, Ux}, axis=1);
    encParams.W = concatenate({W, Wx}, axis=1);
    encParams.b = concatenate({b, bx}, axis=1);

    return RNN<GRUFast>(encParams);
  };

  auto encState0 = name(g->zeros(shape={dimBatch, dimEncState}), "start");

  auto statesFw = encoderGRU("encoder").apply(inputs.begin(),
                                              inputs.end(),
                                              encState0);

  auto statesBw = encoderGRU("encoder_r").apply(inputs.rbegin(),
                                                inputs.rend(),
                                                encState0);

  std::vector<Expr> biStates;
  auto itFw = statesFw.begin();
  auto itBw = statesBw.rbegin();
  while(itFw != statesFw.end())
    biStates.push_back(concatenate({*itFw++, *itBw++}, axis=1));

  // add proper axes and make this a 3D tensor
  auto encContext = name(concatenate(biStates, axis=2), "concat");
  auto meanContext = mean(encContext, axis=2);

  /*** decoder layer 1 ****************************************************/

  auto Wi = g->param("ff_state_W", {2 * dimEncState, dimDecState}, init=glorot_uniform);
  auto bi = g->param("ff_state_b", {1, dimDecState}, init=zeros);

  auto decState0 = tanh(dot(meanContext, Wi) + bi);

  auto Wemb_dec = g->param("Wemb_dec", {dimTrgVoc, dimTrgEmb}, init=glorot_uniform);

  std::vector<Expr> outputs;
  auto emptyEmbedding = name(g->zeros(shape={dimBatch, dimTrgEmb}), "emptyEmbedding");

  outputs.push_back(emptyEmbedding);

  i = 0;
  // @TODO: skip last

  std::vector<float> picks;
  for(auto& trgWordBatch : trgSentenceBatch) {
    for(auto w: trgWordBatch)
      picks.push_back((float)w);
    if(outputs.size() < trgSentenceBatch.size()) {
      auto y = name(rows(Wemb_dec, trgWordBatch), "y_" + std::to_string(i++));
      outputs.push_back(y);
    }
  }
  
  auto decoderGRUWithAttention = [=]() {
    ParametersGRUWithAttention decParams;

    auto U = g->param("decoder_U", {dimDecState, 2 * dimDecState},
                      init=glorot_uniform);

    auto W = g->param("decoder_W", {dimTrgEmb, 2 * dimDecState},
                      init=glorot_uniform);

    auto b = g->param("decoder_b", {1, 2 * dimDecState}, init=zeros);

    auto Ux = g->param("decoder_Ux", {dimDecState, dimDecState},
                      init=glorot_uniform);

    auto Wx = g->param("decoder_Wx", {dimTrgEmb, dimDecState},
                       init=glorot_uniform);

    auto bx = g->param("decoder_bx", {1, dimDecState}, init=zeros);

    decParams.U = concatenate({U, Ux}, axis=1);
    decParams.W = concatenate({W, Wx}, axis=1);
    decParams.b = concatenate({b, bx}, axis=1);

    decParams.Wa = g->param("decoder_W_comb_att", {dimDecState, 2 * dimDecState},
                            init=glorot_uniform);

    decParams.ba = g->param("decoder_b_att", {1, 2 * dimDecState},
                            init=zeros);

    decParams.Ua = g->param("decoder_Wc_att", {2 * dimEncState, 2 * dimDecState},
                            init=glorot_uniform);

    decParams.va = g->param("decoder_U_att", {2 * dimDecState, 1}, // ?
                            init=glorot_uniform);

    auto Uc = g->param("decoder_U_nl", {dimDecState, 2 * dimDecState},
                       init=glorot_uniform);

    auto Wc = g->param("decoder_Wc", {2 * dimEncState,  2 * dimDecState},
                       init=glorot_uniform);

    auto bc = g->param("decoder_b_nl", {1, 2 * dimDecState}, init=zeros);

    auto Uxc = g->param("decoder_Ux_nl", {dimDecState, dimDecState},
                        init=glorot_uniform);

    auto Wxc = g->param("decoder_Wcx", {2 * dimEncState, dimDecState},
                        init=glorot_uniform);

    auto bxc = g->param("decoder_bx_nl", {1, dimDecState}, init=zeros);

    decParams.Uc = concatenate({Uc, Uxc}, axis=1);
    decParams.Wc = concatenate({Wc, Wxc}, axis=1);
    decParams.bc = concatenate({bc, bxc}, axis=1);

    GRUWithAttention gruCell(decParams, encContext);
    return RNN<GRUWithAttention>(gruCell);
  };


  auto decoderGRU = decoderGRUWithAttention();

  auto decStates = decoderGRU.apply(outputs.begin(),
                                    outputs.end(),
                                    decState0);

  auto d1 = concatenate(decStates, axis=2);
  auto e2 = concatenate(outputs, axis=2);

  auto contexts = decoderGRU.getCell().getContexts();
  auto c3 = concatenate(contexts, axis=2);
  
  auto W1 = g->param("ff_logit_lstm_W", {dimDecState, dimTrgEmb},
                     init=glorot_uniform);
  auto b1 = g->param("ff_logit_lstm_b", {1, dimTrgEmb},
                     init=glorot_uniform);

  auto W2 = g->param("ff_logit_prev_W", {dimTrgEmb, dimTrgEmb},
                     init=glorot_uniform);
  auto b2 = g->param("ff_logit_prev_b", {1, dimTrgEmb},
                     init=glorot_uniform);

  auto W3 = g->param("ff_logit_ctx_W", {2 * dimEncState, dimTrgEmb},
                     init=glorot_uniform);
  auto b3 = g->param("ff_logit_ctx_b", {1, dimTrgEmb},
                     init=glorot_uniform);

  auto W4 = g->param("ff_logit_W", {dimTrgEmb, dimTrgVoc},
                     init=glorot_uniform);
  auto b4 = g->param("ff_logit_b", {1, dimTrgVoc},
                     init=glorot_uniform);

  auto t = tanh(affine(d1, W1, b1) + affine(e2, W2, b2) + affine(c3, W3, b3));
  
  auto aff = affine(t, W4, b4);
  //auto s = debug(softmax(aff), "softmax");

  auto p = g->constant(shape={(int)picks.size(), 1},
                       init=from_vector(picks));
  
  auto xe = cross_entropy(aff, p);
  auto cost = debug(name(mean(sum(xe, axis=2), axis=0), "cost"), "cost");
}

SentBatch generateSrcBatch(size_t batchSize) {
  size_t length = rand() % 40 + 10;
  //size_t length = 50;
  return SentBatch(length, WordBatch(batchSize));

  //// das ist ein Test . </s>
  //SentBatch srcBatch({
  //  WordBatch(batchSize, 13),
  //  WordBatch(batchSize, 15),
  //  WordBatch(batchSize, 20),
  //  WordBatch(batchSize, 2548),
  //  WordBatch(batchSize, 4),
  //  WordBatch(batchSize, 0)
  //});

  //if(batchSize > 2) {
  //  srcBatch[0][1] = 109; // dies
  //  srcBatch[0][2] = 19;  // es
  //}

  //return srcBatch;
}

SentBatch generateTrgBatch(size_t batchSize) {
  size_t length = rand() % 40 + 10;
  //size_t length = 50;
  return SentBatch(length, WordBatch(batchSize));

  // this is a test . </s>
  //SentBatch trgBatch({
  //  WordBatch(batchSize, 21),
  //  WordBatch(batchSize, 11),
  //  WordBatch(batchSize, 10),
  //  WordBatch(batchSize, 1078),
  //  WordBatch(batchSize, 5),
  //  WordBatch(batchSize, 0)
  //});
  //
  //if(batchSize > 2) {
  //  trgBatch[0][1] = 12; // that
  //  trgBatch[1][1] = 17; // 's
  //  trgBatch[0][2] = 12;  // that
  //}
  //
  //return trgBatch;
}

int main(int argc, char** argv) {
  cudaSetDevice(3);

  auto g = New<ExpressionGraph>();
  load(g, "../test/model.npz");
  
  size_t batchSize = 1;

  auto srcBatch = generateSrcBatch(batchSize);
  auto trgBatch = generateTrgBatch(batchSize);
  
  g->reserveWorkspaceMB(1024);
  auto opt = Optimizer<Adam>(0.0001);
  
  float sum = 0;
  boost::timer::cpu_timer timer;
  for(int i = 1; i <= 1000; ++i) {
    
    // fake batch
    auto srcBatch = generateSrcBatch(batchSize);
    auto trgBatch = generateTrgBatch(batchSize);
    construct(g, srcBatch, trgBatch);
    
    opt->update(g);
    
    float cost = g->get("cost")->val()->scalar();
    sum += cost;
    
    //if(i % 1 == 0)
    //  std::cerr << ".";
    if(i % 1 == 0)
      std::cout << "[" << i << "]" << std::fixed << std::setfill(' ') << std::setw(9)
                << " - cost: " << cost << "/" << sum / i
                << " - time: " << timer.format(5, "%ws") << std::endl;
  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
