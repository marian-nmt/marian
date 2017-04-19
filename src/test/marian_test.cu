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

#include "models/amun.h"
#include "models/s2s.h"
//#include "models/multi_s2s.h"

int main(int argc, char** argv) {
  using namespace marian;
  using namespace data;

  auto options = New<Config>(argc, argv, false);

  auto corpus = New<Corpus>(options);
  BatchGenerator<Corpus> bg(corpus, options);

  auto graph = New<ExpressionGraph>();
  graph->setDevice(0);

  Ptr<LexProbs> lexProbs;
  if(options->has("lexical-table"))
    lexProbs = New<LexProbs>(options,
                             corpus->getVocabs().front(),
                             corpus->getVocabs().back());
  
  auto type = options->get<std::string>("type");
  Ptr<EncoderDecoderBase> encdec;
  if(type == "s2s")
    encdec = New<S2S>(options);
  else if(type == "multi-s2s")
    encdec = New<MultiS2S>(options);
  else
    encdec = New<Amun>(options, keywords::lex_probs=lexProbs);

  //encdec->load(graph, options->get<std::string>("model"));

  graph->reserveWorkspaceMB(options->get<size_t>("workspace"));

  boost::timer::cpu_timer timer;
  //size_t batches = 1;
  for(int i = 0; i < 1; ++i) {
    bg.prepare(false);
    while(bg) {
      auto batch = bg.next();
      batch->debug();

      auto costNode = encdec->build(graph, batch);
      //for(auto p : graph->params())
      //  debug(p, p->name());
      
      debug(costNode, "cost");

      //graph->graphviz("debug.dot");

      graph->forward();
      graph->backward();
      break;
    }
  }

  //encdec->save(graph, "test.npz", true);

  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
