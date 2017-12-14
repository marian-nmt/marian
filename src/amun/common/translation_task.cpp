#include <string>

#ifdef CUDA
#include <thrust/system_error.h>
#endif

#include "translation_task.h"
#include "search.h"
#include "output_collector.h"
#include "history.h"
#include "god.h"

using namespace std;

namespace amunmt {

void TranslationTask::Run(God &god, SentencesPtr maxiBatch, size_t miniSize, int miniWords)
{
  maxiBatch->SortByLength();
  while (maxiBatch->size()) {
    SentencesPtr miniBatch = maxiBatch->NextMiniBatch(miniSize, miniWords);
    //cerr << "miniBatch=" << miniBatch->size() << " maxiBatch=" << maxiBatch->size() << endl;

    god.GetThreadPool().enqueue(
        [&,miniBatch]{ return Run(god, miniBatch); }
        );
  }

}

void TranslationTask::Exit(God &god)
{
  god.GetThreadPool().enqueue(
      [&]{ return Run(god, SentencesPtr(new Sentences())); }
      );
}

void TranslationTask::Run(const God &god, SentencesPtr sentences) {
  try {
    Search& search = god.GetSearch();
    search.Translate(sentences);

  }
#ifdef CUDA
  catch(thrust::system_error &e)
  {
    std::cerr << "TranslationTask: CUDA error during some_function: " << e.what() << std::endl;
    abort();
  }
#endif
  catch(std::bad_alloc &e)
  {
    std::cerr << "TranslationTask: Bad memory allocation during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(std::runtime_error &e)
  {
    std::cerr << "TranslationTask: Runtime error during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(...)
  {
    std::cerr << "TranslationTask: Some other kind of error during some_function" << std::endl;
    abort();
  }

}

}
