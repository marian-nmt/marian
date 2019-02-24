#include "common/version.h"
#include "common/git_revision.h"     // make-generated file, contains git commit info
#include "common/project_version.h"  // cmake-generated file, major/minor/tweak versions

namespace marian {

std::string buildVersion() {
  return std::string(PROJECT_VERSION) + " " + GIT_REVISION;
}
}
