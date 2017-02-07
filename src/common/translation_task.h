#pragma once
#include <string>
#include "history.h"

namespace amunmt {

class God;

void TranslationTask(const God &god, std::shared_ptr<Sentences> sentences, size_t taskCounter);

}

