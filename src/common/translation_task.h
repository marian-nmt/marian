#pragma once
#include <string>
#include "history.h"

class God;

void TranslationTask(const God &god, std::shared_ptr<Sentences> sentences, size_t taskCounter, size_t maxBatchSize);
