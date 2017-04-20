#pragma once

#include "common/definitions.h"
#include "common/file_stream.h"
#include "data/vocab.h"
#include "kernels/sparse.h"
#include "training/config.h"

namespace marian {

class LexProbs {
  private:
    Ptr<sparse::CSR> lexProbs_;
    size_t srcDim_;
    size_t trgDim_;
    size_t device_;
    
  public:    
    LexProbs(const std::string& fname,
             Ptr<Vocab> srcVocab,
             Ptr<Vocab> trgVocab,
             size_t srcDim,
             size_t trgDim,
             size_t device)
    : srcDim_(srcDim), trgDim_(trgDim), device_(device) {
      
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
            
        if(sid < srcDim && tid < trgDim && prob > 0.001) {
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
      
      std::vector<float> values(nonzeros);
      std::vector<int> rowIndices(nonzeros);
      std::vector<int> colIndices(nonzeros);
      
      std::cerr << "nnz: " << nonzeros << std::endl;
      
      size_t ind = 0;
      for(size_t sid = 0; sid < data.size() && sid < srcDim; ++sid) {
        for(auto& it : data[sid]) {
          size_t tid = it.first;
          float val = it.second;
          
          if(tid >= trgDim)
            break;
          
          rowIndices[ind] = sid;
          colIndices[ind] = tid;
          values[ind] = val;
          ind++;
        }
      }
      
      lexProbs_ = New<sparse::CSR>(srcDim, trgDim, values, rowIndices, colIndices, device_);
    }
    
    LexProbs(Ptr<Config> options,
             Ptr<Vocab> srcVocab,
             Ptr<Vocab> trgVocab,
             size_t device)
    : LexProbs(
        options->get<std::string>("lexical-table"),
        srcVocab, trgVocab, 
        options->get<std::vector<int>>("dim-vocabs").front(),
        options->get<std::vector<int>>("dim-vocabs").back(),
        device)
    {}
    
    Ptr<sparse::CSR> Lf(Ptr<data::CorpusBatch> batch) {
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
      
      auto lookup = New<sparse::CSR>(rows, srcDim_,
                                     values,
                                     rowIndices,
                                     colIndices,
                                     device_);
      auto sent = New<sparse::CSR>(rows, lexProbs_->cols(), device_);
      sparse::multiply(sent, lookup, lexProbs_);  
      return sent;
    }
    
    Ptr<sparse::CSR> getProbs() {
      return lexProbs_;
    }
};

}
