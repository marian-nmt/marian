#include <boost/thread/tss.hpp>
#include "translation_task.h"
#include "search.h"
#include "output_collector.h"
#include "printer.h"

void TranslationTask(boost::shared_ptr<Sentences> sentences, size_t taskCounter, size_t maxBatchSize) {
  thread_local std::unique_ptr<Search> search;
  if(!search) {
    LOG(info) << "Created Search for thread " << std::this_thread::get_id();
    search.reset(new Search(taskCounter));
  }

  Histories allHistories;

  sentences->SortByLength();

  boost::shared_ptr<Sentences> decodeSentences(new Sentences());
  for (size_t i = 0; i < sentences->size(); ++i) {
    decodeSentences->push_back(sentences->at(i));

    if (decodeSentences->size() >= maxBatchSize) {
      assert(decodeSentences->size());
      Histories histories = search->Decode(*decodeSentences);
      allHistories.Append(histories);

      decodeSentences.reset(new Sentences());
    }
  }

  if (decodeSentences->size()) {
    Histories histories = search->Decode(*decodeSentences);
    allHistories.Append(histories);
  }

  allHistories.SortByLineNum();

  std::stringstream strm;
  Printer(allHistories, taskCounter, strm);

  OutputCollector &outputCollector = God::GetOutputCollector();
  outputCollector.Write(taskCounter, strm.str());
}

