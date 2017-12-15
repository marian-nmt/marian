#pragma once
#include "enc_out_gpu.h"
#include "buffer.h"

namespace amunmt {
namespace GPU {

class EncOutBuffer
{
public:
  EncOutBuffer(unsigned int maxSize);
  virtual ~EncOutBuffer();

  void Add(EncOutPtr obj);
  EncOutPtr Get();

  void Get(size_t num, std::vector<EncOut::SentenceElement> &ret);

  size_t size() const
  { return buffer_.size(); }

protected:
  Buffer<EncOutPtr> buffer_;

  EncOutPtr unfinishedEncOutPtr_;
  size_t unfinishedInd_;
};


}
}
