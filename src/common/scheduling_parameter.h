#pragma once

#include "common/logging.h"
#include "common/utils.h"

#include <string>

namespace marian {

// support for scheduling parameters that can be expressed with a unit, such as --lr-decay-inv-sqrt
enum class SchedulingUnit {
  trgLabels, // "t": number of target labels seen so far
  updates,   // "u": number of updates so far (batches)
  epochs     // "e": number of epochs begun so far (very first epoch is 1)
};

struct SchedulingParameter {
  size_t n{0};                                  // number of steps measured in 'unit'
  SchedulingUnit unit{SchedulingUnit::updates}; // unit of value

  // parses scheduling parameters of the form NU where N=unsigned int and U=unit
  // Examples of valid inputs: "16000u" (16000 updates), "32000000t" (32 million target labels),
  // "100e" (100 epochs).
  static SchedulingParameter parse(std::string param) {
    SchedulingParameter res;
    if(!param.empty() && param.back() >= 'a') {
      switch(param.back()) {
        case 't': res.unit = SchedulingUnit::trgLabels; break;
        case 'u': res.unit = SchedulingUnit::updates;   break;
        case 'e': res.unit = SchedulingUnit::epochs;    break;
        default: ABORT("invalid unit '{}' in {}", param.back(), param);
      }
      param.pop_back();
    }
    double number = utils::parseNumber(param);
    res.n = (size_t)number;
    ABORT_IF(number != (double)res.n, "Scheduling parameters must be whole numbers"); // @TODO: do they?
    return res;
  }

  operator bool() const { return n > 0; } // check whether it is specified

  operator std::string() const { // convert back for storing in config
    switch(unit) {
      case SchedulingUnit::trgLabels: return std::to_string(n) + "t";
      case SchedulingUnit::updates  : return std::to_string(n) + "u";
      case SchedulingUnit::epochs   : return std::to_string(n) + "e";
      default: ABORT("corrupt enum value for scheduling unit");
    }
  }
};

}