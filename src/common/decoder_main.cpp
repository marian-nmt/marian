#include <cstdlib>
#include <iostream>
#include <string>
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
  God::Init(argc, argv);
  std::setvbuf(stdout, NULL, _IONBF, 0);
  std::setvbuf(stdin, NULL, _IONBF, 0);
  boost::timer::cpu_timer timer;

  std::string in;
  std::size_t lineNum = 0;
  std::size_t taskCounter = 0;

  size_t bunchSize = God::Summon().Get<size_t>("bunch-size");
  size_t maxBatchSize = God::Summon().Get<size_t>("batch-size");
  std::cerr << "mode=" << God::Summon().Get("mode") << std::endl;

  if (God::Summon().Get<bool>("wipo") || God::Summon().Get<size_t>("cpu-threads")) {
    bunchSize = 1;
    maxBatchSize = 1;
  }

  size_t cpuThreads = God::Summon().Get<size_t>("cpu-threads");
  LOG(info) << "Setting CPU thread count to " << cpuThreads;

  size_t totalThreads = cpuThreads;
#ifdef CUDA
  size_t gpuThreads = God::Summon().Get<size_t>("gpu-threads");
  auto devices = God::Summon().Get<std::vector<size_t>>("devices");
  LOG(info) << "Setting GPU thread count to " << gpuThreads;
  totalThreads += gpuThreads * devices.size();
#endif

  LOG(info) << "Total number of threads: " << totalThreads;
  UTIL_THROW_IF2(totalThreads == 0, "Total number of threads is 0");

  ThreadPool *pool = new ThreadPool(totalThreads);
  LOG(info) << "Reading input";

  boost::shared_ptr<Sentences> sentences(new Sentences());

  while(std::getline(God::Summon().GetInputStream(), in)) {
    Sentence *sentence = new Sentence(lineNum++, in);
    sentences->push_back(boost::shared_ptr<const Sentence>(sentence));

    if (sentences->size() >= maxBatchSize * bunchSize) {

      pool->enqueue(
          [=]{ return TranslationTask(sentences, taskCounter, maxBatchSize); }
      );

      sentences.reset(new Sentences());
      taskCounter++;
    }

  }

  if (sentences->size()) {
    pool->enqueue(
        [=]{ return TranslationTask(sentences, taskCounter, maxBatchSize); }
    );
  }

  delete pool;

  LOG(info) << "Total time: " << timer.format();
  God::Summon().CleanUp();

  return 0;
}
