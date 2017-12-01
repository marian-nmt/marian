#pragma once

#include "gpu/mblas/thrust_functions.h"

#define __fp16 half

__fp16 float2half_rn (float a);

float half2float(__fp16 a);
