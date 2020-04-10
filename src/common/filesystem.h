#pragma once

// This is a shallow wrapper around a filesystem path library.
// We used this to wrap boost::filesystem, now we are wrapping
// Pathie, a small open source lib.

// @TODO: go back to canonical names for functions and objects
// as specified in C++17 so it becomes easy to move in the future

// Even when compiling with clang, __GNUC__ may be defined, so
// we need to add some extra checks to avoid compile errors with
// respect to -Wsuggest-override.
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-value"
#  if defined(__has_warning)
#    if __has_warning("-Wsuggest-override")
#      pragma GCC diagnostic ignored "-Wsuggest-override"
#    endif
#  else
#    pragma GCC diagnostic ignored "-Wsuggest-override"
#  endif
#endif

#include "3rd_party/pathie-cpp/include/path.hpp"  // @TODO: update to latest Pathie
#include "3rd_party/pathie-cpp/include/errors.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace marian {
namespace filesystem {

  bool is_fifo(char const* path);
  bool is_fifo(std::string const& path);

  class Path {
    private:
      Pathie::Path path;

    public:
      Path() {}
      Path(const Path& p) : path{p.path} {}
      Path& operator=(const Path& p) = default;
      Path(const std::string& s) : path{s} {}
      Path(const Pathie::Path& p) : path{p} {}

      Path parentPath() const {
        return Path(path.parent());
      }

      Path filename() const {
        return Path(path.basename());
      }

      Path extension() const {
        return Path(path.extension());
      }

      bool empty() const {
        return path.str().empty();
      }

      const Pathie::Path& getImpl() const {
        return path;
      }

      operator std::string() const {
        return path.str();
      }

      std::string string() const {
        return path.str();
      }

      bool operator==(const Path& p) const {
        return path == p.path;
      }

      bool operator!=(const Path& p) const {
        return path != p.path;
      }
  };

  static inline Path currentPath() {
    return Path(Pathie::Path::pwd());
  }

  static inline Path canonical(const Path& p, const Path& base) {
    // create absolute base path
    return p.getImpl().absolute(base.getImpl()).expand();
  }

  static inline Path relative(const Path& p, const Path& base) {
    // create a path relative to the base path
    return p.getImpl().absolute().expand().relative(base.getImpl().absolute().expand());
  }

  static inline bool exists(const Path& p) {
    return p.getImpl().exists();
  }

  static inline size_t fileSize(const Path& p) {
    return p.getImpl().size();
  }

  static inline bool isDirectory(const Path& p) {
    return p.getImpl().is_directory();
  }

  static inline Path operator/ (const Path& lhs, const Path& rhs) {
    return Path(lhs.getImpl() / rhs.getImpl());
  }

  using FilesystemError = Pathie::PathieError;

}  // namespace filesystem
}  // namespace marian
