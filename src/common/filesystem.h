#pragma once

// @TODO: This is a temporary file to move every function from boost::filesystem used in Marian
// into one place. Marian should call functions only from this file. boost::filesystem will
// be removed. This needs to be portable to Windows too.

#include <boost/filesystem.hpp>

namespace marian {
namespace filesystem {

  struct Path {
    private:
      boost::filesystem::path path;

    public:
      Path() {}
      Path(const Path& p) : path{p.path} {}
      Path(const std::string& s) : path{s} {}
      Path(const boost::filesystem::path& p) : path{p} {}

      Path parentPath() const {
        return Path{path.parent_path()};
      }

      Path filename() const {
        return Path{path.filename()};
      }

      Path extension() const {
        return Path{path.extension()};
      }

      bool empty() const {
        return path.empty();
      }

      const boost::filesystem::path& getBoost() const {
        return path;
      }

      operator std::string&() {
        return (std::string&)path;
      }

      operator std::string() const {
        return path.string();
      }

      std::string string() const {
        return path.string();
      }

      bool operator==(const Path& p) const {
        return path == p.path;
      }

      bool operator!=(const Path& p) const {
        return path != p.path;
      }
  };

  static inline Path currentPath() {
    return Path{boost::filesystem::current_path()};
  }

  static inline Path canonical(const Path& p, const Path& dir) {
    return Path{ boost::filesystem::canonical(p.getBoost(), dir.getBoost()) };
  }

  static inline bool exists(const Path& p) {
    return boost::filesystem::exists(p.getBoost());
  }

  static inline size_t fileSize(const Path& p) {
    return boost::filesystem::file_size(p.getBoost());
  }

  static inline bool isDirectory(const Path& p) {
    return boost::filesystem::is_directory(p.getBoost());
  }

  static inline bool canWrite(const Path& p) {
    return (boost::filesystem::status(p.getBoost()).permissions() & boost::filesystem::owner_write) != 0;
  }

  // concatenation?
  static inline Path operator/ (const Path& lhs, const Path& rhs) {
    return lhs.getBoost() / rhs.getBoost();
  }

  using FilesystemError = boost::filesystem::filesystem_error;

}
}