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

  cudaSetDevice(0);

  std::vector<std::string> files =
    {"../test/mini.de",
     "../test/mini.en"};

  std::vector<std::string> vocab =
    {"../test/vocab.de.json",
     "../test/vocab.en.json"};

  auto corpus = DataSet<Corpus>(files, vocab, 50);
  BatchGenerator<Corpus> bg(corpus, 10, 20);

  auto nematus = New<Nematus>();
  nematus->load("../test/model.npz");
  nematus->reserveWorkspaceMB(128);


  auto opt = Optimizer<Adam>(0.0001 /*, clip=norm(1)*/);

  float sum = 0;
  boost::timer::cpu_timer timer;
  size_t batches = 1;
  for(int i = 0; i < 1; ++i) {
    bg.prepare(false);
    while(bg) {
      auto batch = bg.next();
      batch->debug();

      nematus->construct(*batch);
      debug(nematus->get("Wemb"), "Wemb");
      debug(nematus->get("cost"), "cost");
      debug(nematus->get("decoder_bx"), "decoder_bx");
      debug(nematus->get("encoder_r_bx"), "encoder_r_bx");
      debug(nematus->get("encoder_bx"), "encoder_bx");
      debug(nematus->get("decoder_bx_nl"), "encoder_bx_nl");

      nematus->graphviz("debug.dot");

      nematus->forward();
      nematus->backward();

      //opt->update(nematus);

      float cost = nematus->cost();
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
        nematus->save("../test/model.marian." + std::to_string(batches) + ".npz");

      batches++;
    }
  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
