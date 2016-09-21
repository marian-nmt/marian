#include "node.h"

namespace marian {

// for backward_numeric
void Node::calc_numeric_grad(
		  Float delta,
		  Tensor input,
		  Tensor grad,
		  const std::vector<float> &prevCalcGrad
		  )
{
  using namespace std;

	  size_t inputSize = GetTotalSize(input.shape());
	  size_t valSize = GetTotalSize(val_.shape());

	  UTIL_THROW_IF2(inputSize != GetTotalSize(grad.shape()),
			  	  "inputSize != gradSize:" << inputSize << "!=" << GetTotalSize(grad.shape()));
	  UTIL_THROW_IF2(valSize != GetTotalSize(adj_.shape()),
			  	  "valSize != adjSize :" << valSize << "!=" << GetTotalSize(adj_.shape()));

	  cerr << "sizes: "
			  << Debug(input.shape())<< "=" << inputSize << " "
			  << Debug(val_.shape()) << "=" << valSize
			  << endl;

	  //cerr << "input=" << input.Debug() << endl;

	  std::vector<float> origGrad(inputSize);
	  thrust::copy(grad.begin(), grad.end(), origGrad.begin());
	  cerr << "origGrad=" << grad.Debug() << endl;
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

		  forward();

		  for (size_t i = 0; i < valSize; ++i) {
			  newVal[inputInd] += val_[i];
		  }

		  inputVec[inputInd] -= delta;
	  }

	  // orig value
	  thrust::copy(inputVec.begin(), inputVec.end(), input.begin());
	  forward();

	  Float sumValOrig = 0;
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
		  numericalGrad[i] = (adjVec[i] * (newVal[i] - sumValOrig) / delta);
		  numericalGrad[i] += prevCalcGrad[i];
	  }

	  // set grad results
	  thrust::copy(numericalGrad.begin(), numericalGrad.end(), grad.begin());
	  cerr << "numericalGrad=" << grad.Debug() << endl;
	  //output("numericalGrad", numericalGrad);

	  // print out diff between origGrad and numericalGrad
	  std::vector<float> diff(inputSize);

	  for (size_t i = 0; i < diff.size(); ++i) {
		  diff[i] = (origGrad[i] - numericalGrad[i]) ;
	  }
	  output("diff", diff.begin(), diff.end());

	  // put back origGrad
	  thrust::copy(origGrad.begin(), origGrad.end(), grad.begin());

}

std::vector<float> Node::StoreTensorInVec(Tensor tensor)
{
	  size_t totSize = GetTotalSize(tensor.shape());
	  std::vector<float> vec(totSize);
	  thrust::copy(tensor.begin(), tensor.end(), vec.begin());
	  return vec;
}


}

