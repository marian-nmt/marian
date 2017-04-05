#pragma once
#include <string>
#include "history.h"

namespace amunmt {

class God;

void TranslationTaskAndOutput(const God &god, std::shared_ptr<Sentences> sentences);
std::shared_ptr<Histories> TranslationTask(const God &god, std::shared_ptr<Sentences> sentences);


}

