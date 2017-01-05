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
    {"/work/wmt16/work/unbabel/wmt2015/APE/train.mt-pe.gpu0/train.all.mt",
     "/work/wmt16/work/unbabel/wmt2015/APE/train.mt-pe.gpu0/train.all.pe"};

  std::vector<std::string> vocab =
    {"/work/wmt16/work/unbabel/wmt2015/APE/train.mt-pe.gpu0/mt.json",
     "/work/wmt16/work/unbabel/wmt2015/APE/train.mt-pe.gpu0/pe.json"};

  auto corpus = DataSet<Corpus>(files, vocab, 50);
  BatchGenerator<Corpus> bg(corpus, 40, 1000);

  auto nematus = New<Nematus>();
  nematus->reserveWorkspaceMB(6144);
  auto opt = Optimizer<Adam>(0.0001
                             /*,clip=norm(1)*/);

  float sum = 0;
  boost::timer::cpu_timer timer;
  size_t batches = 1;
  for(int i = 1; i <= 20; ++i) {
    bg.prepare();
    while(bg) {
      auto batch = bg.next();

      nematus->construct(*batch);
      opt->update(nematus);

      float cost = nematus->cost();
      sum += cost;

      if(batches % 1 == 0)
        std::cerr << ".";
      if(batches % 100 == 0)
        std::cout << "[" << batches << "]" << std::fixed << std::setfill(' ') << std::setw(9)
                  << " - cost: " << cost << "/" << sum / batches
                  << " - time: " << timer.format(5, "%ws") << std::endl;

      if(batches % 10000 == 0)
        nematus->save("../test/model.marian.npz");

      batches++;
    }
  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
