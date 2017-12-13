#pragma once

#include <memory>
#include "sentences.h"
#include "history.h"

namespace amunmt {

class God;
class Histories;

class TranslationTask
{
public:
  void Run(God &god, SentencesPtr maxiBatch, size_t miniSize, int miniWords);
  void Exit(God &god);

protected:
  void Run(const God &god, SentencesPtr sentences);

};

}  // namespace amunmt
