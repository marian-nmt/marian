#include "marian.h"

#include <boost/program_options.hpp>
#include <sstream>

int main(int argc, char** argv) {
  using namespace marian;

  createLoggers();

  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("from,f", po::value<std::string>()->default_value("model.npz"),
     "Input model")
    ("to,t", po::value<std::string>()->default_value("model.bin"),
     "Output model")
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

  LOG(info, "Outputting {}", vm["to"].as<std::string>());

  YAML::Node config;
  std::stringstream configStr;
  marian::io::getYamlFromModel(
      config, "special:model.yml", vm["from"].as<std::string>());
  configStr << config;

  auto graph = New<ExpressionGraph>(true, false);
  graph->setDevice(CPU0);

  graph->load(vm["from"].as<std::string>());
  graph->forward();
  graph->save(vm["to"].as<std::string>(), configStr.str());

  // graph->saveBinary(vm["bin"].as<std::string>());

  LOG(info, "Finished");

  return 0;
}
