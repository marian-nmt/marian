#pragma once

// this header is meant to be included for all operations from the "functional" namespace.

#include "functional/operands.h"
#include "functional/predicates.h"
#include "functional/operators.h"

namespace marian {
namespace functional {

template <int N>
using ref = Assignee<N>;

static ref<1> _1;
static ref<2> _2;
static ref<3> _3;
static ref<4> _4;
static ref<5> _5;
static ref<6> _6;
static ref<7> _7;
static ref<8> _8;
static ref<9> _9;

const C<0> _0c;
const C<1> _1c;
const C<2> _2c;
const C<-1> _1cneg;
const C<-2> _2cneg;
}  // namespace functional
}  // namespace marian