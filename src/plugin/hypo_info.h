/*
 * rescore_moses.h
 *
 *  Created on: 12 Jan 2017
 *      Author: hieu
 */
#pragma once
#include <string>
#include "common/scorer.h"

namespace amunmt {

struct HypoInfo
{
  States prevStates;
  States nextStates;
  Beam prevHyps;

  float score;

  std::string Debug() const;

};

}

