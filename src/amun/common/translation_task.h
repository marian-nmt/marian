#pragma once

#include <memory>

namespace amunmt {

class God;
class Histories;
class Sentences;

void TranslationTaskAndOutput(const God &god, std::shared_ptr<Sentences> sentences);
std::shared_ptr<Histories> TranslationTask(const God &god, std::shared_ptr<Sentences> sentences);

}  // namespace amunmt
