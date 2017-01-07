#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "nematus.h"
#include "batch_generator.h"
#include "optimizers.h"

#include "corpus.h"

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
  BatchGenerator<Corpus> bg(corpus, 3, 1000);

  auto nematus = New<Nematus>();
  nematus->load("../test/model.npz");
  nematus->reserveWorkspaceMB(1024);

  //auto opt = Optimizer<Adam>(0.001, clip=norm(1));

  float sum = 0;
  boost::timer::cpu_timer timer;
  size_t batches = 1;
  for(int i = 1; i <= 1; ++i) {
    bg.prepare(false);
    while(bg) {
      auto batch = bg.next();
      batch->debug();
      
      nematus->construct(*batch);
      nematus->forward();

      //opt->update(nematus);

      //float cost = nematus->cost();
      //sum += cost;

      //std::cerr << cost << std::endl;

      //if(batches % 1 == 0)
      //  std::cerr << ".";
      //if(batches % 100 == 0)
      //  std::cout << "[" << batches << "]" << std::fixed << std::setfill(' ') << std::setw(9)
      //            << " - cost: " << cost << "/" << sum / batches
      //            << " - time: " << timer.format(5, "%ws") << std::endl;
      //
      //if(batches % 10000 == 0)
      //  nematus->save("../test/model.marian.npz");

      batches++;
    }
  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
