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

  size_t bunchSize = God::Get<size_t>("bunch-size");
  size_t maxBatchSize = God::Get<size_t>("batch-size");

  if (God::Get<bool>("wipo")) {
    bunchSize = 1;
    maxBatchSize = 1;
  }
  std::cerr << "wipo=" << God::Get<bool>("wipo") << std::endl;

  size_t cpuThreads = God::Get<size_t>("cpu-threads");
  LOG(info) << "Setting CPU thread count to " << cpuThreads;

  size_t totalThreads = cpuThreads;
#ifdef CUDA
  size_t gpuThreads = God::Get<size_t>("gpu-threads");
  auto devices = God::Get<std::vector<size_t>>("devices");
  LOG(info) << "Setting GPU thread count to " << gpuThreads;
  totalThreads += gpuThreads * devices.size();
#endif

  LOG(info) << "Total number of threads: " << totalThreads;
  UTIL_THROW_IF2(totalThreads == 0, "Total number of threads is 0");

  ThreadPool *pool = new ThreadPool(totalThreads);
  LOG(info) << "Reading input";

  boost::shared_ptr<Sentences> sentences(new Sentences());

  while(std::getline(God::GetInputStream(), in)) {
    std::cerr << "Main1" << std::endl;
    Sentence *sentence = new Sentence(lineNum++, in);
    sentences->push_back(boost::shared_ptr<const Sentence>(sentence));
    std::cerr << "Main2:" << lineNum << std::endl;

    if (sentences->size() >= maxBatchSize * bunchSize) {
      std::cerr << "Main3" << std::endl;

      pool->enqueue(
          [=]{ return TranslationTask(sentences, taskCounter, maxBatchSize); }
      );
      std::cerr << "Main4" << std::endl;

      sentences.reset(new Sentences());
      std::cerr << "Main5" << std::endl;
      taskCounter++;
    }

  }
  std::cerr << "Main6" << std::endl;

  if (sentences->size()) {
    std::cerr << "Main7" << std::endl;
    pool->enqueue(
        [=]{ return TranslationTask(sentences, taskCounter, maxBatchSize); }
    );
  }
  std::cerr << "Main8:" << pool->getNumTasks() << std::endl;

  delete pool;
  std::cerr << "Main9" << std::endl;

  LOG(info) << "Total time: " << timer.format();
  God::CleanUp();

  std::cerr << "Main10" << std::endl;
  return 0;
}
