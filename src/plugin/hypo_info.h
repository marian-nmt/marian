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

struct AmunOutput
{
  States states;
  Beam prevHyps;

  float score;

  std::string Debug() const;

};

typedef std::vector<AmunOutput> AmunOutputs;

////////////////////////////////////////////////////////////////
struct AmunInput
{
  States prevStates;
  States nextStates;
  Beam prevHyps;

  Words phrase;
};

typedef std::vector<AmunInput> AmunInputs;

}

