#include "history.h"

namespace marian {

History::History(size_t lineNo, bool normalize)
    : normalize_(normalize), lineNo_(lineNo) {}
}
