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
    "decoder_Wc_att", "decoder_U_att"
  };

  for(auto name : parameters) {
    Shape shape;
    if(numpy[name].shape.size() == 2) {
      shape[0] = numpy[name].shape[0];
      shape[1] = numpy[name].shape[1];
    }
    else if(numpy[name].shape.size() == 1) {
      shape[0] = 1;
      shape[1] = numpy[name].shape[0];
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
  auto encContext = debug(concatenate(biStates, axis=2), "concat");
  auto meanContext = mean(encContext, axis=2);

  /*** decoder layer 1 ****************************************************/

  auto Wi = g->param("ff_state_W", {2 * dimEncState, dimDecState}, init=glorot_uniform);
  auto bi = g->param("ff_state_b", {1, dimDecState}, init=zeros);

  auto decState0 = debug(tanh(dot(meanContext, Wi) + bi), "decState0");

  auto Wemb_dec = g->param("Wemb_dec", {dimTrgVoc, dimTrgEmb}, init=glorot_uniform);

  std::vector<Expr> outputs;
  auto emptyEmbedding = name(g->zeros(shape={dimBatch, dimTrgEmb}), "emptyEmbedding");
  outputs.push_back(emptyEmbedding);

  i = 0;
  for(auto& trgWordBatch : trgSentenceBatch) {
    auto y = name(rows(Wemb_dec, trgWordBatch), "y_" + std::to_string(i++));
    outputs.push_back(y);
  }

  auto decoderGRULayer1 = [=]() {
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

    ParametersGRUFast encParams;
    encParams.U = concatenate({U, Ux}, axis=1);
    encParams.W = concatenate({W, Wx}, axis=1);
    encParams.b = concatenate({b, bx}, axis=1);

    return RNN<GRUFast>(encParams);
  };

  auto statesLayer1 = decoderGRULayer1().apply(outputs.begin(),
                                               outputs.end(),
                                               decState0);

  i = 0;
  for(auto s : statesLayer1)
    debug(s, "decoder state layer1 : " + std::to_string(i++));

  /*** attention **********************************************************/

  auto Wa = g->param("decoder_W_comb_att", {dimDecState, 2 * dimDecState},
                     init=glorot_uniform);

  //auto ba = g->param("decoder_b_att", {1, 2 * dimDecState},
  //                   init=zeros);

  auto Ua = g->param("decoder_Wc_comb_att", {2 * dimEncState, 2 * dimDecState},
                     init=glorot_uniform);

  //auto va = g->param("decoder_U_att", {1, 2 * dimDecState}, // ?
  //                   init=glorot_uniform);


  int src = srcSentenceBatch.size();
  int trg = trgSentenceBatch.size();

  auto statesConcat = concatenate(statesLayer1, axis=2);

  auto E1 = debug(reshape(dot(reshape(statesConcat, {dimBatch * trg, dimDecState}), Wa),
                  {dimBatch, 2 * dimDecState, 1, trg}), "Reshape dot");

  auto E2 = debug(reshape(dot(reshape(encContext, {dimBatch * src, 2 * dimEncState}), Ua),
                  {dimBatch, 2 * dimDecState, src, 1}), "Reshape dot");


  // batch x 2*dimDec x 1 x trg
  // batch x 2*dimDec x src x 1
  // -> batch x 2*dimDec x src x trg
  // temp -> (batch * src * trg) x 2*dimDec

  //auto temp = reshape(tanh(E1 + E2 + ba), {dimBatch * src * trg, 2 * dimDec});

  // c = dot(Va, temp^T, shape={...}) -> batch * src * trg -> batch x src x trg


  /*** decoder layer 2 ****************************************************/

}

SentBatch generateSrcBatch(size_t batchSize) {
  //size_t length = rand() % 40 + 10;
  //return SentBatch(length, WordBatch(batchSize));

  // das ist ein Test . </s>
  return SentBatch({
    WordBatch(batchSize, 13),
    WordBatch(batchSize, 15),
    WordBatch(batchSize, 20),
    WordBatch(batchSize, 2548),
    WordBatch(batchSize, 4),
    WordBatch(batchSize, 0)
  });
}

SentBatch generateTrgBatch(size_t batchSize) {
  //size_t length = rand() % 40 + 10;
  //return SentBatch(length, WordBatch(batchSize));

  // this is a test . </s>
  return SentBatch({
    WordBatch(batchSize, 21),
    WordBatch(batchSize, 11),
    WordBatch(batchSize, 10),
    WordBatch(batchSize, 1078),
    WordBatch(batchSize, 5),
    WordBatch(batchSize, 0)
  });
}

int main(int argc, char** argv) {
  cudaSetDevice(0);

  auto g = New<ExpressionGraph>();
  load(g, "/home/marcinj/Badania/amunmt/test2/model.npz");

  size_t batchSize = 1;

  boost::timer::cpu_timer timer;
  for(int i = 1; i <= 1000; ++i) {
    g->clear();

    // fake batch
    auto srcBatch = generateSrcBatch(batchSize);
    auto trgBatch = generateTrgBatch(batchSize);
    construct(g, srcBatch, trgBatch);

    g->forward();
    g->graphviz("nematus.dot");
    exit(1);

    g->backward();
    if(i % 100 == 0)
      std::cout << i << std::endl;
  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
