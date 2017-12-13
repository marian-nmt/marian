#include "translation_task.h"

#include <string>

#ifdef CUDA
#include <thrust/system_error.h>
#endif

#include "search.h"
#include "output_collector.h"
#include "printer.h"
#include "history.h"

using namespace std;

namespace amunmt {

void TranslationTask(const God &god, SentencesPtr sentences) {
  try {
    Search& search = god.GetSearch();
    search.Translate(sentences);
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

}

