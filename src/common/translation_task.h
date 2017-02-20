#pragma once
#include <string>
#include "history.h"

namespace amunmt {

class God;

void TranslationTask(const God &god, std::shared_ptr<Sentences> sentences);
std::shared_ptr<Histories> TranslationTaskSync(const God &god, std::shared_ptr<Sentences> sentences);


}

