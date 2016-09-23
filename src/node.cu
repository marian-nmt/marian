#include "node.h"

namespace marian {

// for backward_numeric
void Node::calc_numeric_grad(
		  Float delta,
		  Tensor input,
		  Tensor grad
		  )
{
  using namespace std;

  size_t inputSize = GetTotalSize(input.shape());
  size_t valSize = GetTotalSize(val_.shape());

  UTIL_THROW_IF2(inputSize != GetTotalSize(grad.shape()),
			  "inputSize != gradSize:" << inputSize << "!=" << GetTotalSize(grad.shape()));
  UTIL_THROW_IF2(valSize != GetTotalSize(adj_.shape()),
			  "valSize != adjSize :" << valSize << "!=" << GetTotalSize(adj_.shape()));

  cerr	<< "inputSize=grad=" << Debug(input.shape())<< "=" << inputSize << " "
		<< "valSize=adj_=" << Debug(val_.shape()) << "=" << valSize
		<< endl;

  //cerr << "input=" << input.Debug() << endl;
  //cerr << "adj_=" << adj_.Debug() << endl;

  std::vector<float> prevCalcGrad(inputSize);
  thrust::copy(grad.begin(), grad.end(), prevCalcGrad.begin());
  //cerr << "origGrad=" << grad.Debug() << endl;
  //output("diffGrad", diffGrad);

  //output("prevCalcGrad", prevCalcGrad.begin(), prevCalcGrad.end());

  std::vector<float> inputVec(inputSize);
  thrust::copy(input.begin(), input.end(), inputVec.begin());
  //output("inputVec", inputVec);

  std::vector<float> newVal(inputSize, 0);

  // LOOP thru each element in input & add delta
  for (size_t inputInd = 0; inputInd < inputSize; ++inputInd) {
	  inputVec[inputInd] += delta;
	  thrust::copy(inputVec.begin(), inputVec.end(), input.begin());
	  //output("input", input.begin(), input.end());

	  forward();

	  for (size_t i = 0; i < valSize; ++i) {
		  newVal[inputInd] += val_[i];
	  }
	  //output("val_", val_.begin(), val_.end());

	  inputVec[inputInd] -= delta;
  }

  // orig value
  thrust::copy(inputVec.begin(), inputVec.end(), input.begin());
  forward();

  float sumValOrig = 0;
  for (size_t i = 0; i < valSize; ++i) {
	  sumValOrig += val_[i];
  }

  //output("newVal", newVal.begin(), newVal.end());

  // calc gradient
  //cerr << "adj_=" << adj_.Debug() << endl;
  std::vector<float> adjVec(valSize);
  thrust::copy(adj_.begin(), adj_.end(), adjVec.begin());

  std::vector<float> numericalGrad(inputSize);
  for (size_t i = 0; i < numericalGrad.size(); ++i) {
	  numericalGrad[i] = (newVal[i] - sumValOrig) / delta;
  }

  broadcast(numericalGrad, adjVec);
  //std::cerr << "broadcast size=" << numericalGrad.size() << " " << adjVec.size() << std::endl;
  //output("adjVec=", adjVec.begin(), adjVec.end());

  for (size_t i = 0; i < numericalGrad.size(); ++i) {
	  numericalGrad[i] *= adjVec[i];
	  numericalGrad[i] += prevCalcGrad[i];
  }

  //output("prevCalcGrad=", prevCalcGrad.begin(), prevCalcGrad.end());
  //output("adjVec=", adjVec.begin(), adjVec.end());

  // set grad results
  thrust::copy(numericalGrad.begin(), numericalGrad.end(), grad.begin());
  //output("numericalGrad", numericalGrad);
}

void Node::broadcast(const std::vector<float> &largeVec, std::vector<float> &smallVec)
{
	size_t largeSize = largeVec.size();
	size_t smallSize = smallVec.size();

    UTIL_THROW_IF2(largeSize < smallSize,
    		"largeSize < smallSize:" << largeSize << "<" << smallSize);
    UTIL_THROW_IF2(largeSize % smallSize,
    		"largeSize % smallSize != 0:" << largeSize << " " << smallSize);

    smallVec.resize(largeSize);
    for (size_t i = smallSize; i < largeSize; i += smallSize) {
    	std::copy(smallVec.begin(), smallVec.begin() + smallSize, smallVec.begin() + i);
    }
}

void Node::outputL2Norm(const std::vector<float> &x, const std::vector<float> &y) const
{
  using namespace std;
  // print out diff between diffGradA and numericalGrad
  if(x.size() != y.size()) {
	cerr << "size error: " << x.size() << "!=" << y.size() << endl;
	exit(1);
  }

  std::vector<float> diff(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
	  diff[i] = x[i] - y[i];
  }
  cerr << "L2-norm of difference=" << L2Norm(diff) << endl << endl;
}

float Node::L2Norm(const std::vector<float> &vec) const
{
  float ret = 0;
  for (size_t i = 0; i < vec.size(); ++i) {
	  ret += vec[i] * vec[i];
  }
  return sqrt(ret);
}

}

