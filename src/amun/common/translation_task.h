#pragma once

#include <memory>

namespace amunmt {

class God;
class Histories;
class Sentences;
using SentencesPtr = std::shared_ptr<Sentences>;

void TranslationTask(const God &god, SentencesPtr sentences);

}  // namespace amunmt
