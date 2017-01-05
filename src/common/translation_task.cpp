#include <boost/thread/tss.hpp>
#include "translation_task.h"
#include "search.h"

Histories TranslationTask(Sentences *sentences, size_t taskCounter) {
  thread_local std::unique_ptr<Search> search;
  if(!search) {
    LOG(info) << "Created Search for thread " << std::this_thread::get_id();
    search.reset(new Search(taskCounter));
  }

  size_t maxBatchSize = God::Get<size_t>("batch-size");

  sentences->SortByLength();

  Sentences *decodeSentences = new Sentences();
  for (size_t i = 0; i < sentences->size(); ++i) {
    decodeSentences->push_back(sentences->at(i));
  }

  assert(decodeSentences->size());
  Histories histories = search->Decode(*decodeSentences);
  histories.SortByLineNum();

  delete decodeSentences;

  delete sentences;

  return histories;
}

