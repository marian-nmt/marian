#pragma once

#include "data/types.h"
#include "common/definitions.h"
#include "common/utils.h"
#include "common/file_stream.h"

namespace marian {

class VocabBase {
public:
  virtual int load(const std::string& vocabPath, int max = 0) = 0;

  void addCounts(std::unordered_map<std::string, size_t>& counter,
                 const std::string& trainPath) {
    std::unique_ptr<io::InputFileStream> trainStrm(
      trainPath == "stdin" ? new io::InputFileStream(std::cin)
                           : new io::InputFileStream(trainPath)
    );

    std::string line;
    while(getline(*trainStrm, line)) {
      std::vector<std::string> toks;
      utils::split(line, toks, " ");

      for(const std::string& tok : toks) {
        auto iter = counter.find(tok);
        if(iter == counter.end())
          counter[tok] = 1;
        else
          iter->second++;
      }
    }
  }

  virtual void create(const std::string& vocabPath,
                      const std::unordered_map<std::string, size_t>& counter,
                      size_t maxSize = 0) = 0;

  virtual void create(const std::string& vocabPath,
                      const std::vector<std::string>& trainPaths,
                      size_t maxSize = 0) {

    LOG(info, "[data] Creating vocabulary {} from {}",
              vocabPath,
              utils::join(trainPaths, ", "));

    if(vocabPath != "stdout") {
      filesystem::Path path(vocabPath);
      auto dir = path.parentPath();
      if(dir.empty())
        dir = filesystem::currentPath();

      ABORT_IF(!dir.empty() && !filesystem::isDirectory(dir),
              "Specified vocab directory {} does not exist",
              dir.string());

      ABORT_IF(filesystem::exists(vocabPath),
              "Vocabulary file '{}' exists. Not overwriting",
              path.string());
    }

    std::unordered_map<std::string, size_t> counter;
    for(const auto& trainPath : trainPaths)
      addCounts(counter, trainPath);
    create(vocabPath, counter, maxSize);
  }

  virtual void create(const std::string& vocabPath,
                      const std::string& trainPath,
                      size_t maxSize = 0) {
    create(vocabPath, std::vector<std::string>({trainPath}), maxSize);
  }

  // return canonical suffix for given type of vocabulary
  virtual const std::string& canonicalExtension() const = 0;
  virtual const std::vector<std::string>& suffixes() const = 0;

  int findAndLoad(const std::string& path, int max) {
    for(auto suffix : suffixes())
      if(filesystem::exists(path + suffix))
        return load(path + suffix, max);
    return 0;
  }

  virtual Word operator[](const std::string& word) const = 0;

  virtual Words encode(const std::string& line,
                       bool addEOS = true,
                       bool inference = false) const = 0;

  virtual std::string decode(const Words& sentence,
                             bool ignoreEos = true) const = 0;

  virtual const std::string& operator[](Word id) const = 0;

  virtual size_t size() const = 0;
  virtual std::string type() const = 0;

  virtual Word getEosId() const = 0;
  virtual Word getUnkId() const = 0;

  virtual void createFake() = 0;
};

}