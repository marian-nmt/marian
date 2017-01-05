#pragma once

#include "dataset.h"
#include "batch_generator.h"

namespace marian {
namespace data {

class Corpus : public DataBase {
  public:
    Corpus(const std::vector<std::string>& paths,
           const std::vector<std::string>& vocabs)
      : paths_(paths),
        vocabs_(vocabs)
    {
      std::vector<std::ifstream> files;
      for(auto path : paths) {
        files.emplace_back();
        files.back().open(path.c_str());
      }

      bool cont = true;
      while(cont) {
        std::vector<DataPtr> sentences;
        for(auto& file : files) {
          std::string line;
          if(std::getline(file, line)) {
            std::cerr << line << std::endl;
            sentences.emplace_back();
          }
        }

        cont = sentences.size() == files.size();
        if(cont)
          examples_.emplace_back(new Example(sentences));
      };
    }

    ExampleIterator begin() const {
      return ExampleIterator(examples_.begin());
    }

    ExampleIterator end() const {
      return ExampleIterator(examples_.end());
    }

    void shuffle() {
      std::random_shuffle(examples_.begin(), examples_.end());
    }

    BatchPtr toBatch(const Examples& batchVector) {
      int batchSize = batchVector.size();
      BatchPtr batch;
      return batch;
    }

  private:
    Examples examples_;

    std::vector<std::string> paths_;
    std::vector<std::string> vocabs_;
};

}
}
