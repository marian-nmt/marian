#pragma once

#include "common/definitions.h"
#include "common/file_stream.h"
#include "data/types.h"

namespace marian {

class BaseVocab {
protected:
  // size_t batchIndex_;
  // Ptr<Options> options_;

public:
  // BaseVocab(size_t batchIndex, Ptr<Options> options)
  //  : batchIndex_(batchIndex), options_(options) {}

  virtual int loadOrCreate(const std::string& /*vocabPath*/,
                           const std::string& /*textPath*/,
                           int /*max*/ = 0) = 0;

  virtual int load(const std::string& /*vocabPath*/, int /*max*/ = 0) = 0;
  virtual void create(const std::string& /*vocabPath*/, const std::string& /*trainPath*/) = 0;

  virtual void create(io::InputFileStream& /*trainStrm*/,
                      io::OutputFileStream& /*vocabStrm*/,
                      size_t /*maxSize*/ = 0) = 0;

  virtual Word operator[](const std::string& /*word*/) const = 0;

  virtual Words operator()(const std::vector<std::string>& /*lineTokens*/,
                           bool /*addEOS*/ = true) const = 0;

  virtual std::vector<std::string> operator()(const Words& /*sentence*/,
                                              bool /*ignoreEOS*/ = true) const = 0;

  virtual Words encode(const std::string& /*line*/,
                       bool /*addEOS*/ = true,
                       bool /*inference*/ = false) const = 0;

  virtual std::string decode(const Words& /*sentence*/,
                             bool /*ignoreEos*/ = true) const = 0;

  virtual const std::string& operator[](Word /*id*/) const = 0;

  virtual size_t size() const = 0;

  virtual Word getEosId() const = 0;
  virtual Word getUnkId() const = 0;

  virtual void createFake() = 0;
};

}