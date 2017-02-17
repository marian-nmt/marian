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

int main(int argc, char* argv[]) {
  God god;
  god.Init(argc, argv);
  std::setvbuf(stdout, NULL, _IONBF, 0);
  std::setvbuf(stdin, NULL, _IONBF, 0);
  boost::timer::cpu_timer timer;

  std::string in;
  std::size_t lineNum = 0;

  size_t maxiSize = god.Get<size_t>("maxi-batch");

  LOG(info) << "Reading input";

  SentencesPtr maxiBatch(new Sentences());

  while (std::getline(god.GetInputStream(), in)) {
    maxiBatch->push_back(SentencePtr(new Sentence(god, lineNum++, in)));

    if (maxiBatch->size() >= maxiSize) {
      god.Enqueue(*maxiBatch);
      maxiBatch.reset(new Sentences());
    }

  }

  // last batch
  god.Enqueue(*maxiBatch);

  LOG(info) << "Total time: " << timer.format();
  //sleep(10);
  return 0;
}
