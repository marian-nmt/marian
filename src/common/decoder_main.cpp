#include <cstdlib>
#include <iostream>
#include <string>
#include <memory>
#include <boost/timer/timer.hpp>
#include <boost/thread/tss.hpp>

#include "common/god.h"
#include "common/logging.h"
#include "common/search.h"
#include "common/threadpool.h"
#include "common/printer.h"
#include "common/sentence.h"
#include "common/exception.h"
#include "common/translation_task.h"

int main(int argc, char* argv[]) {
  God god;
  god.Init(argc, argv);
  std::setvbuf(stdout, NULL, _IONBF, 0);
  std::setvbuf(stdin, NULL, _IONBF, 0);
  boost::timer::cpu_timer timer;

  std::string in;
  std::size_t lineNum = 0;
  std::size_t taskCounter = 0;

  size_t bunchSize = god.Get<size_t>("bunch-size");
  size_t maxBatchSize = god.Get<size_t>("mini-batch");
  std::cerr << "mode=" << god.Get("mode") << std::endl;

  if (god.Get<bool>("wipo") || god.Get<size_t>("cpu-threads")) {
    bunchSize = 1;
    maxBatchSize = 1;
  }

  size_t cpuThreads = god.Get<size_t>("cpu-threads");
  LOG(info) << "Setting CPU thread count to " << cpuThreads;

  size_t totalThreads = cpuThreads;
#ifdef CUDA
  size_t gpuThreads = god.Get<size_t>("gpu-threads");
  auto devices = god.Get<std::vector<size_t>>("devices");
  LOG(info) << "Setting GPU thread count to " << gpuThreads;
  totalThreads += gpuThreads * devices.size();
#endif

  LOG(info) << "Total number of threads: " << totalThreads;
  UTIL_THROW_IF2(totalThreads == 0, "Total number of threads is 0");

  ThreadPool *pool = new ThreadPool(totalThreads);
  LOG(info) << "Reading input";

  std::shared_ptr<Sentences> sentences(new Sentences());

  while (std::getline(god.GetInputStream(), in)) {
    sentences->push_back(SentencePtr(new Sentence(god, lineNum++, in)));

    if (sentences->size() >= maxBatchSize * bunchSize) {
      pool->enqueue(
          [=,&god]{ return TranslationTask(god, sentences, taskCounter, maxBatchSize); }
      );

      sentences.reset(new Sentences());
      taskCounter++;
    }

  }

  if (sentences->size()) {
    pool->enqueue(
        [=,&god]{ return TranslationTask(god, sentences, taskCounter, maxBatchSize); }
    );
  }

  delete pool;

  LOG(info) << "Total time: " << timer.format();
  god.CleanUp();

  return 0;
}
