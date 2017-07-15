#pragma once

#include "common/config.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "data/vocab.h"
#include "kernels/sparse.h"
#include "layers/attention.h"

namespace marian {

class LexProbs {
private:
  static thread_local Ptr<sparse::CSR> lexProbs_;
  static thread_local Ptr<sparse::CSR> lf_;

  size_t srcDim_;
  size_t trgDim_;

  std::vector<float> values_;
  std::vector<int> rowIndices_;
  std::vector<int> colIndices_;

public:
  LexProbs(const std::string& fname,
           Ptr<Vocab> srcVocab,
           Ptr<Vocab> trgVocab,
           size_t srcDim,
           size_t trgDim)
      : srcDim_(srcDim), trgDim_(trgDim) {
    InputFileStream in(fname);

    std::vector<std::map<size_t, float>> data;
    size_t nonzeros = 0;
    std::string src, trg;
    float prob;
    while(in >> trg >> src >> prob) {
      if(src == "NULL" || trg == "NULL")
        continue;

      size_t sid = (*srcVocab)[src];
      size_t tid = (*trgVocab)[trg];

      if(sid < srcDim_ && tid < trgDim_ && prob > 0.001) {
        if(data.size() <= sid)
          data.resize(sid + 1);
        if(data[sid].count(tid) == 0) {
          data[sid][tid] = prob;
          nonzeros++;
        }
      }
    }
    // EOS symbol
    if(data.size() <= 1)
      data.resize(1);
    data[0][0] = 1;
    nonzeros++;

    values_.resize(nonzeros);
    rowIndices_.resize(nonzeros);
    colIndices_.resize(nonzeros);

    LOG(info)->info("Building dictionary of {} pairs from lexical table", nonzeros);

    size_t ind = 0;
    for(size_t sid = 0; sid < data.size() && sid < srcDim; ++sid) {
      for(auto& it : data[sid]) {
        size_t tid = it.first;
        float val = it.second;

        if(tid >= trgDim_)
          break;

        rowIndices_[ind] = sid;
        colIndices_[ind] = tid;
        values_[ind] = val;
        ind++;
      }
    }
  }

  LexProbs(Ptr<Config> options, Ptr<Vocab> srcVocab, Ptr<Vocab> trgVocab)
      : LexProbs(options->get<std::string>("lexical-table"),
                 srcVocab,
                 trgVocab,
                 options->get<std::vector<int>>("dim-vocabs").front(),
                 options->get<std::vector<int>>("dim-vocabs").back()) {}

  void buildProbs(size_t device) {
    if(!lexProbs_) {
      LOG(info)->info("Building sparse matrix for lexical probabilities");
      lexProbs_ = New<sparse::CSR>(
          srcDim_, trgDim_, values_, rowIndices_, colIndices_, device);
    }
  }

  void resetLf(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {
    buildProbs(graph->getDevice());

    auto srcBatch = batch->front();
    auto& indices = srcBatch->indeces();

    size_t rows = indices.size();

    std::vector<float> values(rows);
    std::vector<int> rowIndices(rows);
    std::vector<int> colIndices(rows);

    for(size_t i = 0; i < rows; ++i) {
      UTIL_THROW_IF2(indices[i] >= srcDim_, "column index to large");
      values[i] = 1;
      rowIndices[i] = i;
      colIndices[i] = indices[i];
    }

    auto lookup = New<sparse::CSR>(
        rows, srcDim_, values, rowIndices, colIndices, graph->getDevice());
    lf_ = New<sparse::CSR>(rows, lexProbs_->cols(), graph->getDevice());
    sparse::multiply(lf_, lookup, lexProbs_);
  }

  Ptr<sparse::CSR> getProbs() { return lexProbs_; }

  Ptr<sparse::CSR> getLf() { return lf_; }
};

class LexicalBias {
private:
  Ptr<sparse::CSR> sentLexProbs_;
  Ptr<GlobalAttention> attention_;
  float eps_;
  bool single_;

public:
  LexicalBias(Ptr<sparse::CSR> sentLexProbs,
              Ptr<GlobalAttention> attention,
              float eps,
              bool single)
      : sentLexProbs_(sentLexProbs),
        attention_(attention),
        eps_(eps),
        single_(single) {}

  Expr operator()(Expr logits);
};
}
