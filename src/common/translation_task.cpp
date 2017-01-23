#include <boost/thread/tss.hpp>
#include "translation_task.h"
#include "search.h"
#include "output_collector.h"
#include "printer.h"

void TranslationTask(const God &god, std::shared_ptr<Sentences> sentences, size_t taskCounter, size_t miniBatch) {
  Search &search = god.GetSearch();

  try {
    Histories allHistories;
    sentences->SortByLength();

    size_t bunchId = 0;
    std::shared_ptr<Sentences> decodeSentences(new Sentences(taskCounter, bunchId++));

    for (size_t i = 0; i < sentences->size(); ++i) {
      decodeSentences->push_back(sentences->at(i));

      if (decodeSentences->size() >= miniBatch) {
        assert(decodeSentences->size());
        std::shared_ptr<Histories> histories = search.Decode(god, *decodeSentences);
        allHistories.Append(*histories.get());

        decodeSentences.reset(new Sentences(taskCounter, bunchId++));
      }
    }

    if (decodeSentences->size()) {
      std::shared_ptr<Histories> histories = search.Decode(god, *decodeSentences);
      allHistories.Append(*histories.get());
    }

    allHistories.SortByLineNum();

    std::stringstream strm;
    Printer(god, allHistories, strm);

    OutputCollector &outputCollector = god.GetOutputCollector();
    outputCollector.Write(taskCounter, strm.str());
  }
#ifdef CUDA
  catch(thrust::system_error &e)
  {
    std::cerr << "CUDA error during some_function: " << e.what() << std::endl;
    abort();
  }
#endif
  catch(std::bad_alloc &e)
  {
    std::cerr << "Bad memory allocation during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(std::runtime_error &e)
  {
    std::cerr << "Runtime error during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(...)
  {
    std::cerr << "Some other kind of error during some_function" << std::endl;
    abort();
  }

}

