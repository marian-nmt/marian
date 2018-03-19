#pragma once

#include "functional/operands.h"
#include "functional/predicates.h"

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

static C<0> _0c;
static C<1> _1c;
static C<2> _2c;
static C<-1> _1cneg;
static C<-2> _2cneg;
}
}