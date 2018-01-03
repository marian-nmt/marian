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

  void Get(unsigned num, std::vector<BufferOutput> &ret);

  unsigned size() const
  { return buffer_.size(); }

protected:
  Buffer<EncOutPtr> buffer_;

  EncOutPtr unfinishedEncOutPtr_;
  unsigned unfinishedInd_;

  EncOutPtr Get();
};


}
}
