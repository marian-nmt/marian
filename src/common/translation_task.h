#pragma once
#include <string>
#include <boost/shared_ptr.hpp>
#include "history.h"

void TranslationTask(boost::shared_ptr<Sentences> sentences, size_t taskCounter, size_t maxBatchSize);
