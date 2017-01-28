#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>

#include "marian.h"
#include "optimizers/optimizers.h"
#include "optimizers/clippers.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "models/nematus.h"

int main(int argc, char** argv) {
  using namespace marian;
  using namespace data;

  std::vector<std::string> files =
    {"../test/mini.de",
     "../test/mini.en"};

  std::vector<std::string> vocab =
    {"../test/vocab.de.json",
     "../test/vocab.en.json"};

  std::vector<int> maxVocab = { 50000, 50000 };

  auto corpus = DataSet<Corpus>(files, vocab, maxVocab, 50);
  BatchGenerator<Corpus> bg(corpus, 10, 20);

  auto graph = New<ExpressionGraph>();
  graph->setDevice(std::atoi(argv[1]));

  auto nematus = New<Nematus>();
  nematus->load(graph, "../test/model.npz");

  graph->reserveWorkspaceMB(128);

  float sum = 0;
  boost::timer::cpu_timer timer;
  size_t batches = 1;
  for(int i = 0; i < 1; ++i) {
    bg.prepare(false);
    while(bg) {
      auto batch = bg.next();
      batch->debug();

      auto costNode = nematus->construct(graph, batch);
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
        nematus->save(graph, "../test/model.marian." + std::to_string(batches) + ".npz");

      batches++;
    }
  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
