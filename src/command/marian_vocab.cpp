#include "marian.h"

#include "common/cli_wrapper.h"
#include "common/logging.h"
#include "data/vocab.h"

int main(int argc, char** argv) {
  using namespace marian;

  createLoggers();

  cli::CLIWrapper cli("Allowed options");
  cli.add<size_t>(
      "--max-size,-m", "Generate only  arg  most common vocabulary items", 0);

  cli.parse(argc, argv);

  LOG(info, "Creating vocabulary...");

  auto vocab = New<Vocab>();
  InputFileStream corpusStrm(std::cin);
  OutputFileStream vocabStrm(std::cout);
  vocab->create(corpusStrm, vocabStrm, cli.get<size_t>("max-size"));

  LOG(info, "Finished");

  return 0;
}
