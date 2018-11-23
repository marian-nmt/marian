#include "marian.h"

#include "common/cli_wrapper.h"
#include "common/logging.h"
#include "data/vocab.h"

int main(int argc, char** argv) {
  using namespace marian;

  createLoggers();

  auto options = New<Options>();
  {
    auto cli = New<cli::CLIWrapper>(
        options,
        "Create a vocabulary from text corpora given on STDIN",
        "Allowed options",
        "Examples:\n"
        "  ./marian-vocab < text.src > vocab.yml\n"
        "  cat text.src text.trg | ./marian-vocab > vocab.yml");
    cli->add<size_t>("--max-size,-m", "Generate only UINT most common vocabulary items", 0);
    cli->parse(argc, argv);
  }

  LOG(info, "Creating vocabulary...");

  auto vocab = New<Vocab>(options, 0);
  vocab->create("stdout", "stdin", options->get<size_t>("max-size"));

  LOG(info, "Finished");

  return 0;
}
