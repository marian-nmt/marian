// Copyright 2013 by Chris Dyer
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef _DA_H_
#define _DA_H_

#include <cmath>
#include <cassert>

// m = trg len
// n = src len
// i = trg index
// j = src index
struct DiagonalAlignment {

  static double UnnormalizedProb(const unsigned i, const unsigned j, const unsigned m, const unsigned n, const double alpha) {
#if 0
    assert(i > 0);
    assert(n > 0);
    assert(m >= i);
    assert(n >= j);
#endif
    return exp(Feature(i, j, m, n) * alpha);
  }

  static double ComputeZ(const unsigned i, const unsigned m, const unsigned n, const double alpha) {
#if 0
    assert(i > 0);
    assert(n > 0);
    assert(m >= i);
#endif
    const double split = double(i) * n / m;
    const unsigned floor = static_cast<unsigned>(split);
    unsigned ceil = floor + 1;
    const double ratio = exp(-alpha / n);
    const unsigned num_top = n - floor;
    double ezt = 0;
    double ezb = 0;
    if (num_top)
      ezt = UnnormalizedProb(i, ceil, m, n, alpha) * (1.0 - pow(ratio, num_top)) / (1.0 - ratio);
    if (floor)
      ezb = UnnormalizedProb(i, floor, m, n, alpha) * (1.0 - pow(ratio, floor)) / (1.0 - ratio);
    return ezb + ezt;
  }

  static double ComputeDLogZ(const unsigned i, const unsigned m, const unsigned n, const double alpha) {
    const double z = ComputeZ(i, n, m, alpha);
    const double split = double(i) * n / m;
    const unsigned floor = static_cast<unsigned>(split);
    const unsigned ceil = floor + 1;
    const double ratio = exp(-alpha / n);
    const double d = -1.0 / n;
    const unsigned num_top = n - floor;
    double pct = 0;
    double pcb = 0;
    if (num_top) {
      pct = arithmetico_geometric_series(Feature(i, ceil, m, n), UnnormalizedProb(i, ceil, m, n, alpha), ratio, d, num_top);
      //cerr << "PCT = " << pct << endl;
    }
    if (floor) {
      pcb = arithmetico_geometric_series(Feature(i, floor, m, n), UnnormalizedProb(i, floor, m, n, alpha), ratio, d, floor);
      //cerr << "PCB = " << pcb << endl;
    }
    return (pct + pcb) / z;
  }

  inline static double Feature(const unsigned i, const unsigned j, const unsigned m, const unsigned n) {
    return -fabs(double(j) / n - double(i) / m);
  }

 private:
  inline static double arithmetico_geometric_series(const double a_1, const double g_1, const double r, const double d, const unsigned n) {
    const double g_np1 = g_1 * pow(r, n);
    const double a_n = d * (n - 1) + a_1;
    const double x_1 = a_1 * g_1;
    const double g_2 = g_1 * r;
    const double rm1 = r - 1;
    return (a_n * g_np1 - x_1) / rm1 - d*(g_np1 - g_2) / (rm1 * rm1);
  }
};

#endif
