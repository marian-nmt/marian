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

int main(int argc, char** argv) {
  using namespace marian;
  using namespace data;

  auto options = New<Config>(argc, argv, false);

  std::vector<std::string> files =
    {"../test/mini.de",
     "../test/mini.en"};

  std::vector<std::string> vocab =
    {"../test/vocab.de.json",
     "../test/vocab.en.json"};

  YAML::Node& c = options->get();
  c["trainsets"] = files;
  c["vocabs"] = vocab;

  auto corpus = DataSet<Corpus>(options);
  BatchGenerator<Corpus> bg(corpus, options);

  auto graph = New<ExpressionGraph>();
  graph->setDevice(0);

  auto dl4mt = New<DL4MT>();
  dl4mt->load(graph, "../test/model.npz");

  graph->reserveWorkspaceMB(128);

  float sum = 0;
  boost::timer::cpu_timer timer;
  size_t batches = 1;
  for(int i = 0; i < 1; ++i) {
    bg.prepare(false);
    while(bg) {
      auto batch = bg.next();
      batch->debug();

      auto costNode = dl4mt->build(graph, batch);
      for(auto p : graph->params())
        debug(p, p->name());
      debug(costNode, "cost");

      graph->graphviz("debug.dot");

      graph->forward();
      graph->backward();

      float cost = costNode->val()->scalar();
      sum += cost;

      if(batches % 100 == 0) {
        std::cout << std::setfill(' ')
                  << "Epoch " << i
                  << " Update " << batches
                  << " Cost "   << std::setw(7) << std::setprecision(6) << cost
                  << " UD " << timer.format(2, "%ws");

        float seconds = std::stof(timer.format(5, "%w"));
        float sentences = 100 * batch->size() / seconds;

        std::cout << " " << std::setw(5)
                  << std::setprecision(4)
                  << sentences
                  << " sentences/s" << std::endl;
        timer.start();
      }

      if(batches % 10000 == 0)
        dl4mt->save(graph, "../test/model.marian." + std::to_string(batches) + ".npz");

      batches++;
    }
  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
