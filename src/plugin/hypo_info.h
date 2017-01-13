/*
 * rescore_moses.h
 *
 *  Created on: 12 Jan 2017
 *      Author: hieu
 */
#pragma once
#include <string>
#include "common/scorer.h"

struct HypoInfo
{
  std::vector<size_t> words;
  size_t lastWord;
  States prevStates;
  States nextStates;
  float score;

  std::string Debug() const;

};


