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

namespace marian {

template <class Builder>
class BeamSearch {
  private:
    Ptr<Builder> builder_;
    size_t beamSize_;
    cudaStream_t stream_{0};

  public:
    BeamSearch(Ptr<Builder> builder)
     : builder_(builder),
       beamSize_(3)
    {}

    void search(Ptr<ExpressionGraph> graph,
                Ptr<data::CorpusBatch> batch) {

      auto nth = New<NthElement>(beamSize_, batch->size(), stream_);

      Expr startState, hyps, probs;
      startState = builder_->buildEncoder(graph, batch);
      std::tie(hyps, probs) = builder_->step(startState);
      size_t pos = graph->forward();

      std::vector<size_t> beamSizes(batch->size(), beamSize_);
      std::vector<unsigned> outKeys;
      std::vector<float> outCosts;

      nth->getNBestList(beamSizes, probs->val(),
                        outCosts, outKeys, true);
      size_t dimTrgVoc = probs->shape()[1];

      std::vector<int> beam(beamSize_, 0);
      while(outKeys[0] != 0) {
        std::vector<size_t> hypIdx;
        std::vector<size_t> embIdx;
        for(int i = 0; i < outKeys.size(); ++i) {
          int k = outKeys[i];

          hypIdx.push_back(k / dimTrgVoc);
          embIdx.push_back(k % dimTrgVoc);

          std::cerr << hypIdx.back() << " "
            << embIdx.back() << " "
            << outCosts[i] << std::endl;
        }
        std::cerr << std::endl;

        std::tie(hyps, probs) = builder_->step(hyps, hypIdx, embIdx);
        pos = graph->forward(pos);

        outKeys.clear();
        outCosts.clear();
        nth->getNBestList(beamSizes, probs->val(), outCosts, outKeys, false);
      }
    }
};

}

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

  auto dl4mt = New<DL4MT>(options);
  dl4mt->load(graph, "../benchmark/marian32K/modelBN.90000.npz");

  graph->reserveWorkspaceMB(128);

  boost::timer::cpu_timer timer;
  bg.prepare(false);
  while(bg) {
    auto batch = bg.next();
    batch->debug();

    auto search = New<BeamSearch<DL4MT>>(dl4mt);
    search->search(graph, batch);

    exit(0);
  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
