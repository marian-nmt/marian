#pragma once

#include "dataset.h"
#include "batch_generator.h"
#include "common/vocab.h"

namespace marian {
namespace data {

class CorpusBatch {
  public:
    void test() {
      std::cerr << "Hello" << std::endl;
    }
};

class Corpus : public DataBase {
  public:
    typedef CorpusBatch batch_type;
    typedef std::shared_ptr<batch_type> batch_ptr;

    Corpus(const std::vector<std::string> textPaths,
           const std::vector<std::string> vocabPaths)
    {
      UTIL_THROW_IF2(textPaths.size() != vocabPaths.size(),
                     "Number of corpus files and vocab files does not agree");

      std::vector<std::ifstream> files;
      for(auto path : textPaths) {
        files.emplace_back();
        files.back().open(path.c_str());
      }

      std::vector<Vocab> vocabs;
      for(auto path : vocabPaths) {
        vocabs.emplace_back(path);
      }

      bool cont = true;
      while(cont) {
        std::vector<DataPtr> sentences;
        for(int i = 0; i < files.size(); ++i) {
          std::string line;
          if(std::getline(files[i], line)) {
            Words words = vocabs[i](line);
            sentences.emplace_back(new Data());
            for(auto w : words)
              sentences.back()->push_back((float)w);
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

    batch_ptr toBatch(const Examples& batchVector) {
      int batchSize = batchVector.size();
      std::cerr << batchSize << std::endl;

      std::vector<int> maxDims;
      for(auto& ex : batchVector) {
        if(maxDims.size() < ex->size())
          maxDims.resize(ex->size(), 0);
        for(int i = 0; i < ex->size(); ++i) {
          if((*ex)[i]->size() > maxDims[i])
          maxDims[i] = (*ex)[i]->size();
        }
      }

      for(auto m : maxDims)
        std::cerr << m << " ";
      std::cerr << std::endl;

      batch_ptr batch(new batch_type());

      return batch;
    }

  private:
    Examples examples_;
};

}
}
