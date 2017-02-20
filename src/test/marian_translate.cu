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
#include "common/history.h"

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
       beamSize_(12)
    {}

    Beam toHyps(const std::vector<uint> keys,
                  const std::vector<float> costs,
                  size_t vocabSize,
                  const Beam& beam) {
      Beam newBeam;
      for(int i = 0; i < keys.size(); ++i) {
        int embIdx = keys[i] % vocabSize;
        int hypIdx = keys[i] / vocabSize;
        newBeam.push_back(
          New<Hypothesis>(beam[hypIdx], embIdx, hypIdx, costs[i]));
      }
      return newBeam;
    }

    Beam pruneBeam(const Beam& beam) {
      Beam newBeam;
      for(auto hyp : beam) {
        if(hyp->GetWord() > 0) {
          newBeam.push_back(hyp);
        }
      }
      return newBeam;
    }

    std::tuple<Expr, Expr> step(Expr hyps, const Beam& beam) {
      std::vector<size_t> hypIndeces;
      std::vector<size_t> embIndeces;
      std::vector<float> beamCosts;

      for(auto hyp : beam) {
        hypIndeces.push_back(hyp->GetPrevStateIndex());
        embIndeces.push_back(hyp->GetWord());
        beamCosts.push_back(hyp->GetCost());
      }

      auto graph = hyps->graph();
      auto costs = graph->constant(keywords::shape={(int)beamCosts.size(), 1},
                                   keywords::init=inits::from_vector(beamCosts));
      Expr probs;
      std::tie(hyps, probs) = builder_->step(hyps, hypIndeces, embIndeces);
      probs = probs + costs;
      return std::make_tuple(hyps, probs);
    }

    Ptr<History> search(Ptr<ExpressionGraph> graph,
                        Ptr<data::CorpusBatch> batch) {

      Expr startState, hyps, probs;
      startState = builder_->buildEncoder(graph, batch);

      size_t pos = 0;
      auto history = New<History>(0);
      Beam beam(1, New<Hypothesis>());
      bool first = true;
      bool final = false;
      std::vector<size_t> beamSizes(1, beamSize_);
      auto nth = New<NthElement>(beamSize_, batch->size(), stream_);

      do {

        if(first) {
          std::tie(hyps, probs) = builder_->step(startState);
          pos = graph->forward();
        }
        else {
          std::tie(hyps, probs) = step(hyps, beam);
          beamSizes.clear();
          beamSizes.resize(1, beam.size());
          pos = graph->forward(pos);
        }

        size_t dimTrgVoc = probs->shape()[1];

        std::vector<unsigned> outKeys;
        std::vector<float> outCosts;
        nth->getNBestList(beamSizes, probs->val(),
                          outCosts, outKeys, first);
        first = false;

        beam = toHyps(outKeys, outCosts, dimTrgVoc, beam);
        final = history->size() >= 3 * batch->words();
        history->Add(beam, final);
        beam = pruneBeam(beam);

      } while(!beam.empty() && !final);

      return history;
    }
};

}

int main(int argc, char** argv) {
  using namespace marian;
  using namespace data;

  auto options = New<Config>(argc, argv, false);

  std::vector<std::string> files =
    {"../benchmark/marian32K/newstest2016.tok.true.bpe.en"};

  std::vector<std::string> vocab =
    {"../benchmark/marian32K/train.tok.true.bpe.en.json"};

  YAML::Node& c = options->get();
  c["train-sets"] = files;
  c["vocabs"] = vocab;

  auto corpus = DataSet<Corpus>(options);
  BatchGenerator<Corpus> bg(corpus, options);

  auto graph = New<ExpressionGraph>();
  graph->setDevice(0);

  auto target = New<Vocab>();
  target->load("../benchmark/marian32K/train.tok.true.bpe.de.json", 50000);

  auto encdec = New<DL4MT>(options);
  encdec->load(graph, "../benchmark/marian32K/modelML.10000.npz");

  graph->reserveWorkspaceMB(128);

  boost::timer::cpu_timer timer;
  bg.prepare(false);
  while(bg) {
    auto batch = bg.next();
    auto search = New<BeamSearch<DL4MT>>(encdec);
    auto history = search->search(graph, batch);

    auto r = history->Top();
    for(auto w : r.first)
      if(w != 0)
        std::cout << (*target)[w] << " ";
    std::cout << std::endl;

  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
