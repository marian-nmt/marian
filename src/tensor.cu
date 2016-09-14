#include <fstream>
#include "tensor.h"

using namespace std;

namespace marian {

void Tensor::set(const std::vector<float>& data)
{
	pimpl_->set(data.begin(), data.end());
}

void Tensor::set(const std::vector<float>::const_iterator &begin, const std::vector<float>::const_iterator &end)
{
	pimpl_->set(begin, end);
}

Tensor& operator<<(Tensor& t, const std::vector<float> &vec) {
  t.set(vec);
  return t;
}

std::vector<float>& operator<<(std::vector<float> &vec, const Tensor& t) {
  t.get(vec);
  return vec;
}


}

