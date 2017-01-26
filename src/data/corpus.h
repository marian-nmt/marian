#pragma once

#include "data/dataset.h"
#include "data/batch_generator.h"
#include "data/vocab.h"

namespace marian {
namespace data {

typedef std::vector<size_t> WordBatch;
typedef std::vector<float> MaskBatch;
typedef std::pair<WordBatch, MaskBatch> WordMask;
typedef std::vector<WordMask> SentBatch;

class CorpusBatch {
  public:
    CorpusBatch(const std::vector<SentBatch>& batches, size_t words = 0)
    : batches_(batches), words_(words) {}

    const SentBatch& operator[](size_t i) const {
      return batches_[i];
    }

    void debug() {
      size_t i = 0;
      for(auto l : batches_) {
        std::cerr << "input " << i++ << ": " << std::endl;
        for(auto b : l) {
          std::cerr << "\t w: ";
          for(auto w : b.first) {
            std::cerr << w << " ";
          }
          std::cerr << std::endl;

          std::cerr << "\t m: ";
          for(auto w : b.second) {
            std::cerr << w << " ";
          }
          std::cerr << std::endl;
        }
      }
    }

    size_t size() const {
      return batches_[0][0].first.size();
    }

    size_t words() const {
      return words_;
    }

  private:
    std::vector<SentBatch> batches_;
    size_t words_;
};

class Corpus : public DataBase {
  public:
    typedef CorpusBatch batch_type;
    typedef std::shared_ptr<batch_type> batch_ptr;

    Corpus(const std::vector<std::string>& textPaths,
           const std::vector<std::string>& vocabPaths,
           const std::vector<int>& maxVocabs,
           size_t maxLength = 50)
    {
      UTIL_THROW_IF2(textPaths.size() != vocabPaths.size(),
                     "Number of corpus files and vocab files does not agree");

      std::vector<std::ifstream> files;
      for(auto path : textPaths) {
        files.emplace_back();
        files.back().open(path.c_str());
      }

      std::vector<Vocab> vocabs;
      for(int i = 0; i < vocabPaths.size(); ++i) {
        vocabs.emplace_back(vocabPaths[i], maxVocabs[i]);
      }

      bool cont = true;
      while(cont) {
        std::vector<DataPtr> sentences;
        for(int i = 0; i < files.size(); ++i) {
          std::string line;
          if(std::getline(files[i], line)) {
            Words words = vocabs[i](line);
            if(words.empty())
              words.push_back(0);

            sentences.emplace_back(new Data());
            for(auto w : words)
              sentences.back()->push_back((float)w);
          }
        }

        cont = sentences.size() == files.size();
        if(cont && std::all_of(sentences.begin(), sentences.end(),
                               [=](DataPtr d) { return d->size() > 0 && d->size() <= maxLength; }))
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
      std::random_device rd;
      std::mt19937 g(rd());

      std::shuffle(examples_.begin(), examples_.end(), g);
    }

    batch_ptr toBatch(const Examples& batchVector) {
      int batchSize = batchVector.size();
      size_t words = 0;

      std::vector<int> maxDims;
      for(auto& ex : batchVector) {
        if(maxDims.size() < ex->size())
          maxDims.resize(ex->size(), 0);
        for(int i = 0; i < ex->size(); ++i) {
          if((*ex)[i]->size() > maxDims[i])
          maxDims[i] = (*ex)[i]->size();
        }
      }

      std::vector<SentBatch> langs;
      for(auto m : maxDims) {
        langs.push_back(SentBatch(m,
                                  { WordBatch(batchSize, 0),
                                    MaskBatch(batchSize, 0) } ));
      }

      for(int i = 0; i < batchSize; ++i) {
        for(int j = 0; j < maxDims.size(); ++j) {
          for(int k = 0; k < (*batchVector[i])[j]->size(); ++k) {
            langs[j][k].first[i] = (*(*batchVector[i])[j])[k];
            langs[j][k].second[i] = 1.f;
            if(j == 0)
              words++;
          }
        }
      }

      return batch_ptr(new batch_type(langs, words));
    }

  private:
    Examples examples_;
};

}
}
