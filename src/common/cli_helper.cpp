#include "common/cli_helper.h"
#include "common/filesystem.h"

namespace marian {
namespace cli {

void makeAbsolutePaths(YAML::Node& config,
                       const std::vector<std::string>& configPaths,
                       const std::set<std::string>& PATHS) {
  ABORT_IF(configPaths.empty(),
           "--relative-paths option requires at least one config file provided "
           "with --config");
  // TODO: expand paths relative to EACH config file
  // expand relative paths w.r.t to the first config file
  auto configDir = filesystem::Path{configPaths.front()}.parentPath();

  for(const auto& configPath : configPaths)
    ABORT_IF(filesystem::Path{configPath}.parentPath() != configDir,
             "--relative-paths option requires all config files to be in the "
             "same directory");

  auto transformFunc = [&](const std::string& nodePath) -> std::string {
    // Catch stdin/stdout and do not process
    if(nodePath == "stdin" || nodePath == "stdout")
      return nodePath;

    // replace relative path w.r.t. configDir
    try {
      return canonical(filesystem::Path{nodePath}, configDir).string();
    } catch(filesystem::FilesystemError& e) {
      // will fail if file does not exist; use parent in that case
      std::cerr << e.what() << std::endl;
      auto parentPath = filesystem::Path{nodePath}.parentPath();
      return (canonical(parentPath, configDir)
              / filesystem::Path{nodePath}.filename())
          .string();
    }
  };

  processPaths(config, transformFunc, PATHS);
}

}  // namespace cli
}  // namespace marian
