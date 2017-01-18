#pragma once
#include <string>
#include <boost/shared_ptr.hpp>
#include "history.h"

class God;

void TranslationTask(God &god, boost::shared_ptr<Sentences> sentences, size_t taskCounter, size_t maxBatchSize);
