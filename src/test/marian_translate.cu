#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>

#include "marian.h"
#include "training/config.h"
#include "optimizers/optimizers.h"
#include "optimizers/clippers.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "models/dl4mt.h"
#include "translator/nth_element.h"

int main(int argc, char** argv) {
  using namespace marian;
  using namespace data;

  auto options = New<Config>(argc, argv, false);

  std::vector<std::string> files =
    {"../test/mini.en"};

  std::vector<std::string> vocab =
    {"../benchmark/marian32K/train.tok.true.bpe.en.json"};

  YAML::Node& c = options->get();
  c["train-sets"] = files;
  c["vocabs"] = vocab;

  auto corpus = DataSet<Corpus>(options);
  BatchGenerator<Corpus> bg(corpus, options);

  auto graph = New<ExpressionGraph>();
  graph->setDevice(0);

  auto dl4mt = New<DL4MT>();
  dl4mt->load(graph, "../benchmark/marian32K/modelBN.90000.npz");

  graph->reserveWorkspaceMB(128);

  boost::timer::cpu_timer timer;
  bg.prepare(false);
  while(bg) {
    auto batch = bg.next();
    batch->debug();

    size_t beamSize = 1;
    int batchSize = batch->size();

    cudaStream_t stream(0);
    auto nth = New<NthElement>(beamSize, batchSize, stream);

    Expr hyps, probs;
    std::tie(hyps, probs) = dl4mt->initTranslator(graph, batch, beamSize);
    auto it = graph->forward();

    std::cerr << probs->val()->debug() << std::endl;

    std::vector<float> outCosts;
    std::vector<unsigned> outKeys;
    std::vector<size_t> beamSizes;
    for(int i = 0; i < batchSize; ++i)
      beamSizes.push_back(beamSize);

    nth->getNBestList(beamSizes, probs->val(), outCosts, outKeys, true);

    for(int i = 0; i < outKeys.size(); ++i)
      std::cerr << i << " " << outKeys[i] << " " << outCosts[i] << std::endl;

    auto selected = rows(hyps, {0, 0, 0});
    it = graph->forward(it);

    std::cerr << selected->val()->debug() << std::endl;


    /*
    std::vector<size_t> bestHypIndeces;
    std::vector<size_t> bestEmbIndeces;
    do {
      std::vector<std::pair<size_t, float>>
      bestIndecesProbs = Argmax(probs->val(), beamSize);

      bestHypIndeces = toHyps(bestIndecesProbs);
      bestEmbIndeces = toEmbs(bestIndecesProbs);

      if(!bestHypIndeces.empty()) {
        //auto nextEmbs = rows(yEmb, bestEmbIndeces);
        //auto nextHyps = rows(states, bestHypIndeces);
        std::tie(hyps, probs) = dl4mt->step(hyps, bestHypIndeces, bestEmbIndeces);
        it = graph->forward(it);
      }

    } while(!bestHypsIndeces.empty());
    */
  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
