#include "common/cli_helper.h"
#include "common/filesystem.h"

namespace marian {
namespace cli {

void makeAbsolutePaths(YAML::Node& config,
                       const std::string& configPath,
                       const std::set<std::string>& PATHS) {
  auto configDir = filesystem::Path{configPath}.parentPath();

  auto transformFunc = [&](const std::string& nodePath) -> std::string {
    // Catch stdin/stdout and do not process
    if(nodePath == "stdin" || nodePath == "stdout")
      return nodePath;

    // replace relative path w.r.t. config directory
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
