#pragma once
#include <string>
#include "history.h"

//History TranslationTask(const Sentence *sentence, size_t taskCounter);
Histories TranslationTask(const Sentences *sentences, size_t taskCounter);
Histories TranslationTask(const Sentences&& sentences, size_t taskCounter);
Histories TranslationTask(const Sentences& sentences, size_t taskCounter);

