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
#include "models/multi_gnmt.h"
#include "models/gnmt.h"
#include "models/dl4mt.h"

#include "translator/nth_element.h"
#include "common/history.h"


namespace marian {

template <class Builder>
class BeamSearch {
  private:
    Ptr<Config> options_;
    Ptr<Builder> builder_;
    size_t beamSize_;
    cudaStream_t stream_{0};

  public:
    BeamSearch(Ptr<Config> options)
     : options_(options),
       builder_(New<Builder>(options, keywords::inference=true)),
       beamSize_(options_->get<size_t>("beam-size"))
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
         Ptr<EncoderState> encState,
         const std::vector<size_t> hypIdx = {},
         const std::vector<size_t> embIdx = {}) {
      using namespace keywords;
      auto graph = hyps[0]->graph();

      // @TODO: not hard-coded!
      int dimTrgEmb = options_->get<int>("dim-emb");
      int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();

      std::vector<Expr> selectedHyps;
      Expr selectedEmbs;
      if(embIdx.empty()) {
        selectedHyps = hyps;
        selectedEmbs = graph->constant(shape={1, dimTrgEmb},
                                       init=inits::zeros);
      }
      else {
        // @TODO : solve this better than reshaping!
        for(auto h : hyps)
          selectedHyps.push_back(
            reshape(rows(h, hypIdx), {1, h->shape()[1], 1, (int)hypIdx.size()}));

        auto yEmb = Embedding("Wemb_dec", dimTrgVoc, dimTrgEmb)(graph);
        selectedEmbs = reshape(rows(yEmb, embIdx),
                               {1, yEmb->shape()[1], 1, (int)embIdx.size()});
      }

      Expr logits;
      std::vector<Expr> newHyps;
      std::tie(logits, newHyps) = builder_->step(selectedEmbs,
                                                 selectedHyps,
                                                 encState,
                                                 true);
      return std::make_tuple(newHyps, logsoftmax(logits));
    }

    std::tuple<std::vector<Expr>, Expr>
    step(std::vector<Expr> hyps,
         Ptr<EncoderState> encState,
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
                                      encState,
                                      hypIndeces,
                                      embIndeces);
      probs = probs + costs;
      return std::make_tuple(newHyps, probs);
    }

    Ptr<History> search(Ptr<ExpressionGraph> graph,
                        Ptr<data::CorpusBatch> batch) {

      std::vector<Expr> startStates;
      Ptr<EncoderState> encState;
      std::tie(startStates, encState)
        = builder_->buildEncoder(graph, batch);
        
      size_t pos = 0;
      auto history = New<History>(0, options_->get<bool>("normalize"));
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
          std::tie(hyps, probs) = step(startStates, encState);
          pos = graph->forward();
        }
        else {
          std::tie(hyps, probs) = step(hyps, encState, beam);
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

class TranslatorBase {
  public:
    virtual Ptr<History> translate(Ptr<data::CorpusBatch>) = 0;
};

template <class Model>
class Translator : public TranslatorBase {
  private:
    Ptr<Config> options_;
    Ptr<ExpressionGraph> graph_;


  public:
    Translator(Ptr<Config> options)
    : options_(options),
    graph_(New<ExpressionGraph>()) {
      auto devices = options_->get<std::vector<int>>("devices");
      graph_->setDevice(devices[0]);
      graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      auto model = New<Model>(options, keywords::inference=true);
      model->load(graph_, options_->get<std::string>("model"));
    }

    Ptr<History> translate(Ptr<data::CorpusBatch> batch) {
      auto search = New<BeamSearch<Model>>(options_);
      return search->search(graph_, batch);
    }

};

}

int main(int argc, char** argv) {
  using namespace marian;
  using namespace data;

  auto options = New<Config>(argc, argv, true, true);

  Ptr<TranslatorBase> translator;
  auto type = options->get<std::string>("type");
  if(type == "gnmt")
    translator = New<Translator<GNMT>>(options);
  else if(type == "multi-gnmt")
    translator = New<Translator<MultiGNMT>>(options);
  else
    translator = New<Translator<DL4MT>>(options);

  auto corpus = DataSet<Corpus>(options, true);
  BatchGenerator<Corpus> bg(corpus, options);

  auto target = New<Vocab>();
  auto vocabs = options->get<std::vector<std::string>>("vocabs");
  target->load(vocabs.back());

  boost::timer::cpu_timer timer;
  bg.prepare(false);
  while(bg) {
    auto batch = bg.next();
    auto history = translator->translate(batch);

    //********************************
    auto results = history->NBest(1);
    std::stringstream ss;
    for(auto r : results) {
      for(auto w : r.first)
        if(w != 0)
          ss << (*target)[w] << " ";
    }
    std::cout << ss.str() << std::endl;
    //********************************

  }
  std::cerr << timer.format(5, "%ws") << std::endl;

  return 0;

}
