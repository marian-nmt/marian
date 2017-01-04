#include <boost/thread/tss.hpp>
#include "translation_task.h"
#include "search.h"

Histories TranslationTask(const Sentences *sentences, size_t taskCounter) {
  thread_local std::unique_ptr<Search> search;
  if(!search) {
    LOG(info) << "Created Search for thread " << std::this_thread::get_id();
    search.reset(new Search(taskCounter));
  }

  std::vector<Sentences> vecSentences;
  std::vector<Histories> vecHistories;

  size_t batchSize = God::Get<size_t>("batch-size");

  assert(sentences->size());
  Histories histories = search->Decode(*sentences);
  delete sentences;

  return histories;
}

