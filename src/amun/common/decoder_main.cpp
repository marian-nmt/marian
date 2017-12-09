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

  std::string line;
  std::size_t lineNum = 0;

  while (std::getline(god.GetInputStream(), line)) {
    maxiBatch->push_back(SentencePtr(new Sentence(god, lineNum++, line)));

    if (maxiBatch->size() >= maxiSize) {

      maxiBatch->SortByLength();
      while (maxiBatch->size()) {
        SentencesPtr miniBatch = maxiBatch->NextMiniBatch(miniSize, miniWords);
        //cerr << "miniBatch=" << miniBatch->size() << " maxiBatch=" << maxiBatch->size() << endl;

        god.GetThreadPool().enqueue(
            [&god,miniBatch]{ return TranslationTaskAndOutput(god, miniBatch); }
            );
      }

      maxiBatch.reset(new Sentences());
    }

  }

  // last batch
  if (maxiBatch->size()) {
    maxiBatch->SortByLength();
    while (maxiBatch->size()) {
      SentencesPtr miniBatch = maxiBatch->NextMiniBatch(miniSize, miniWords);
      god.GetThreadPool().enqueue(
          [&god,miniBatch]{ return TranslationTaskAndOutput(god, miniBatch); }
          );
    }
  }

  god.Cleanup();
  LOG(info)->info("Total time: {}", timer.format());

  return 0;
}
