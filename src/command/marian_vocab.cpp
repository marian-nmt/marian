#include "marian.h"

#include "common/logging.h"
#include "data/vocab.h"

int main(int argc, char** argv) {
  using namespace marian;

  ABORT_IF(argc != 3,
           "wrong number of arguments.\nUsage: {} <corpus-path> <vocab-path>",
           argv[0]);

  auto vocab = New<Vocab>();
  vocab->create(argv[2], argv[1]);

  return 0;
}
