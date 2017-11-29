#include "marian.h"

#include <boost/program_options.hpp>

#include "common/logging.h"
#include "data/vocab.h"

int main(int argc, char** argv) {
  using namespace marian;

  createLoggers();

  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("max-size,m", po::value<size_t>()->default_value(0),
     "Generate only  arg  most common vocabulary items")
    ("help,h", "Print this message and exit")
    ;
  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch(std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl << std::endl;
    std::cerr << "Usage: " << argv[0] << " [options]" << std::endl << std::endl;
    std::cerr << desc << std::endl;
    exit(1);
  }

  if(vm.count("help")) {
    std::cerr << "Usage: " << argv[0] << " [options]" << std::endl << std::endl;
    std::cerr << desc << std::endl;
    exit(0);
  }

  LOG(info, "Creating vocabulary...");

  auto vocab = New<Vocab>();
  InputFileStream corpusStrm(std::cin);
  OutputFileStream vocabStrm(std::cout);
  vocab->create(corpusStrm, vocabStrm, vm["max-size"].as<size_t>());

  LOG(info, "Finished");

  return 0;
}
