#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>

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
    {"../train.src-pe.gpu0/train.all.src",
     "../train.src-pe.gpu0/train.all.pe"};

  std::vector<std::string> vocab =
    {"../train.src-pe.gpu0/src.json",
     "../train.src-pe.gpu0/pe.json"};

  /*
  std::vector<std::string> files =
    {"/work/wmt16/work/unbabel/wmt2015/APE/train.mt-pe.gpu0/train.all.mt",
     "/work/wmt16/work/unbabel/wmt2015/APE/train.mt-pe.gpu0/train.all.pe"};

  std::vector<std::string> vocab =
    {"/work/wmt16/work/unbabel/wmt2015/APE/train.mt-pe.gpu0/mt.json",
     "/work/wmt16/work/unbabel/wmt2015/APE/train.mt-pe.gpu0/pe.json"};
  */

  auto corpus = DataSet<Corpus>(files, vocab, 50);
  BatchGenerator<Corpus> bg(corpus, 40, 1000);

  auto nematus = New<Nematus>();
  nematus->load("../train.src-pe.gpu0/model.iter10000.npz");
  nematus->reserveWorkspaceMB(8000);

  auto opt = Optimizer<Adam>(0.0001 /*, clip=norm(1)*/);

  float sum = 0;
  boost::timer::cpu_timer timer;
  size_t batches = 1;
  for(int i = 0; i < 20; ++i) {
    bg.prepare();
    while(bg) {
      auto batch = bg.next();

      nematus->construct(*batch);

      opt->update(nematus);

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
  std::cout << timer.format(2, "%ws") << std::endl;

  return 0;
}
