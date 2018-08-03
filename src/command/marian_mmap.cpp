#include "marian.h"

#include <boost/program_options.hpp>

//#include "common/logging.h"

int main(int argc, char** argv) {
  using namespace marian;

  createLoggers();

  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("model,m", po::value<std::string>()->default_value("model.npz"),
     "Input non-mappable model")
    ("bin,b", po::value<std::string>()->default_value("model.npz.bin"),
     "Output mappable model")
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

  LOG(info, "Outputting {}", vm["bin"].as<std::string>());

  auto graph = New<ExpressionGraph>(true, false);
  graph->setDevice(CPU0);

  graph->load(vm["model"].as<std::string>());
  graph->forward();
  graph->saveBinary(vm["bin"].as<std::string>());
  
  LOG(info, "Finished");

  return 0;
}
