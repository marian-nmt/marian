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
#include "models/gnmt.h"
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
        float cost = costs[i];

        newBeam.push_back(
          New<Hypothesis>(beam[hypIdx], embIdx, hypIdx, cost));
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

    std::tuple<std::vector<Expr>, Expr>
    step(std::vector<Expr> hyps,
         Expr srcContext,
         Expr srcMask,
         const std::vector<size_t> hypIdx = {},
         const std::vector<size_t> embIdx = {}) {
      using namespace keywords;
      auto graph = hyps[0]->graph();

      // @TODO: not hard-coded!
      int dimTrgEmb_ = 512;
      int dimTrgVoc_ = 50000;

      std::vector<Expr> selectedHyps;
      Expr selectedEmbs;
      if(embIdx.empty()) {
        selectedHyps = hyps;
        selectedEmbs = graph->constant(shape={1, dimTrgEmb_},
                                       init=inits::zeros);
      }
      else {
        // @TODO : solve this better than reshaping!
        for(auto h : hyps)
          selectedHyps.push_back(
            reshape(rows(h, hypIdx), {1, h->shape()[1], 1, (int)hypIdx.size()}));

        auto yEmb = Embedding("Wemb_dec", dimTrgVoc_, dimTrgEmb_)(graph);
        selectedEmbs = reshape(rows(yEmb, embIdx),
                               {1, yEmb->shape()[1], 1, (int)embIdx.size()});
      }

      Expr logits;
      std::vector<Expr> newHyps;
      std::tie(logits, newHyps) = builder_->step(selectedEmbs,
                                                 selectedHyps,
                                                 srcContext,
                                                 srcMask,
                                                 true);
      return std::make_tuple(newHyps, logsoftmax(logits));
    }

    std::tuple<std::vector<Expr>, Expr>
    step(std::vector<Expr> hyps,
         Expr srcContext,
         Expr srcMask,
         const Beam& beam) {

      std::vector<size_t> hypIndeces;
      std::vector<size_t> embIndeces;
      std::vector<float> beamCosts;

      for(auto hyp : beam) {
        hypIndeces.push_back(hyp->GetPrevStateIndex());
        embIndeces.push_back(hyp->GetWord());
        beamCosts.push_back(hyp->GetCost());
      }

      auto graph = hyps[0]->graph();
      auto costs = graph->constant(keywords::shape={1, 1, 1, (int)beamCosts.size()},
                                   keywords::init=inits::from_vector(beamCosts));

      std::vector<Expr> newHyps;
      Expr probs;
      std::tie(newHyps, probs) = step(hyps,
                                      srcContext,
                                      srcMask,
                                      hypIndeces,
                                      embIndeces);
      probs = probs + costs;
      return std::make_tuple(newHyps, probs);
    }

    Ptr<History> search(Ptr<ExpressionGraph> graph,
                        Ptr<data::CorpusBatch> batch) {

      std::vector<Expr> startStates;
      Expr srcContext, srcMask;
      std::tie(startStates, srcContext, srcMask)
        = builder_->buildEncoder(graph, batch);

      size_t pos = 0;
      auto history = New<History>(0);
      Beam beam(1, New<Hypothesis>());
      bool first = true;
      bool final = false;
      std::vector<size_t> beamSizes(1, beamSize_);
      auto nth = New<NthElement>(beamSize_, batch->size(), stream_);

      history->Add(beam);

      std::vector<Expr> hyps;
      Expr probs;
      do {

        if(first) {
          std::tie(hyps, probs) = step(startStates,
                                       srcContext,
                                       srcMask);
          pos = graph->forward();
        }
        else {
          std::tie(hyps, probs) = step(hyps,
                                       srcContext,
                                       srcMask,
                                       beam);
          beamSizes[0] = beam.size();
          pos = graph->forward(pos);
        }

        size_t dimTrgVoc = probs->shape()[1];

        std::vector<unsigned> outKeys;
        std::vector<float> outCosts;

        for(int i = 0; i < probs->shape()[3]; i++) {
          probs->val()->set(i * dimTrgVoc + 1, std::numeric_limits<float>::lowest());
        }

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
    //{"../benchmark/marian32K/test.txt"};

  std::vector<std::string> vocab =
    {"../benchmark/marian32K/train.tok.true.bpe.en.json"};

  YAML::Node& c = options->get();
  c["train-sets"] = files;
  c["vocabs"] = vocab;

  auto corpus = DataSet<Corpus>(options);
  BatchGenerator<Corpus> bg(corpus, options);

  auto graph = New<ExpressionGraph>();
  graph->setDevice(1);

  auto target = New<Vocab>();
  target->load("../benchmark/marian32K/train.tok.true.bpe.de.json", 50000);

  auto encdec = New<GNMT>(options);
  encdec->load(graph, "../benchmark/marian32K/modelML6.200000.npz");

  graph->reserveWorkspaceMB(128);

  boost::timer::cpu_timer timer;
  bg.prepare(false);
  while(bg) {
    auto batch = bg.next();
    auto search = New<BeamSearch<GNMT>>(encdec);
    auto history = search->search(graph, batch);

    auto results = history->NBest(1);
    for(auto r : results) {
        for(auto w : r.first)
        if(w != 0)
          std::cout << (*target)[w] << " ";
      //std::cout << r.second->GetCost() << std::endl;
      std::cout << std::endl;
    }
  }
  std::cerr << timer.format(5, "%ws") << std::endl;

  return 0;

}
