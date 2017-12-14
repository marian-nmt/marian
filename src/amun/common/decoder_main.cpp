#include <cstdlib>
#include <iostream>
#include <string>
#include <memory>
#include <boost/timer/timer.hpp>

#include "common/god.h"
#include "common/logging.h"
#include "common/search.h"
#include "common/threadpool.h"
#include "common/printer.h"
#include "common/sentence.h"
#include "common/sentences.h"
#include "common/exception.h"
#include "common/translation_task.h"

using namespace amunmt;
using namespace std;

int main(int argc, char* argv[])
{
  God god;
  god.Init(argc, argv);

  std::setvbuf(stdout, NULL, _IONBF, 0);
  std::setvbuf(stdin, NULL, _IONBF, 0);
  boost::timer::cpu_timer timer;


  size_t miniSize = (god.Get<size_t>("cpu-threads") == 0) ? god.Get<size_t>("mini-batch") : 1;
  size_t maxiSize = (god.Get<size_t>("cpu-threads") == 0) ? god.Get<size_t>("maxi-batch") : 1;
  int miniWords = god.Get<int>("mini-batch-words");

  LOG(info)->info("Reading input");

  SentencesPtr maxiBatch(new Sentences());

  TranslationTask task;
  std::string line;
  std::size_t lineNum = 0;

  while (std::getline(god.GetInputStream(), line)) {
    maxiBatch->push_back(SentencePtr(new Sentence(god, lineNum++, line)));

    if (maxiBatch->size() >= maxiSize) {
      task.Run(god, maxiBatch, miniSize, miniWords);

      maxiBatch.reset(new Sentences());
    }

  }

  // last batch
  task.Run(god, maxiBatch, miniSize, miniWords);

  // empty batch to indicate end - async
  task.Exit(god);

  god.Cleanup();
  LOG(info)->info("Total time: {}", timer.format());

  return 0;
}
