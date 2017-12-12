#pragma once

#include <memory>

namespace amunmt {

class God;
class Histories;
class Sentences;
using SentencesPtr = std::shared_ptr<Sentences>;

void TranslationTaskAndOutput(const God &god, SentencesPtr sentences);
std::shared_ptr<Histories> TranslationTask(const God &god, SentencesPtr sentences);

}  // namespace amunmt
