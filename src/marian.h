#pragma once

// clang-format off
#include "common/config.h"
#include "common/definitions.h"
#include "common/logging.h"
#include "common/options.h"
#include "common/version.h"

#include "data/batch_generator.h"
#include "data/corpus.h"

#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "graph/node_initializers.h"

#include "optimizers/optimizers.h"


// TODO: The following are front-end includes that should not be included here.
//#include "layers/constructors.h"
//#include "layers/generic.h"
//#include "layers/guided_alignment.h"
//
//#include "models/model_base.h"
//#include "models/states.h"
//#include "models/encdec.h"
//
//#include "rnn/attention.h"
//#include "rnn/constructors.h"
//#include "rnn/rnn.h"
// clang-format on
