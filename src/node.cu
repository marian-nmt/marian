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
	  size_t gradSize = GetTotalSize(grad.shape());
	  size_t adjSize = GetTotalSize(adj_.shape());
	  cerr << "sizes: "
			  << Debug(input.shape())<< "=" << inputSize << " "
			  << Debug(grad.shape()) << "=" << gradSize << " "
			  << Debug(adj_.shape()) << "=" << adjSize
			  << endl;

	  std::vector<float> diffGrad(gradSize);
	  thrust::copy(grad.begin(), grad.end(), diffGrad.begin());
	  cerr << "diffGrad=" << grad.Debug() << endl;
	  //output("diffGrad", diffGrad);

	  // reset grad
	  thrust::copy(prevCalcGrad.begin(), prevCalcGrad.end(), grad.begin());
	  //cerr << "reset a_->grad()=" << a_->grad().Debug() << endl;

	  // START CALC of numerical gradient
	  // new values
	  input.incr(delta);

	  forward();
	  //cerr << "input=" << input.Debug() << endl;
	  //cerr << "val_=" << val_.Debug() << endl;

	  std::vector<float> newVal(inputSize);
	  thrust::copy(val_.begin(), val_.end(), newVal.begin());
	  //output("newVal", newVal);

	  // old values
	  input.incr(-delta);

	  forward();
	  //cerr << "input=" << input.Debug() << endl;
	  //cerr << "val_=" << val_.Debug() << endl;

	  std::vector<float> origVal(inputSize);
	  thrust::copy(val_.begin(), val_.end(), origVal.begin());
	  //output("origVal", origVal);

	  // calc gradient
	  //cerr << "adj_=" << adj_.Debug() << endl;
	  std::vector<float> adjVec(adjSize);
	  thrust::copy(adj_.begin(), adj_.end(), adjVec.begin());

	  std::vector<float> numericalGrad(gradSize);
	  for (size_t i = 0; i < numericalGrad.size(); ++i) {
		  numericalGrad[i] = prevCalcGrad[i] + (adjVec[i] * (newVal[i] - origVal[i]) / delta);
	  }

	  // set grad results
	  thrust::copy(numericalGrad.begin(), numericalGrad.end(), grad.begin());
	  cerr << "numericalGrad=" << grad.Debug() << endl;
	  //output("numericalGrad", numericalGrad);

	  // print out diff between diffGrad and numericalGrad
	  std::vector<float> origGrad(gradSize);
	  std::vector<float> diff(gradSize);

	  thrust::copy(grad.begin(), grad.end(), origGrad.begin());
	  for (size_t i = 0; i < diff.size(); ++i) {
		  diff[i] = (diffGrad[i] - numericalGrad[i]) ;
	  }
	  output("diff", diff);

}

std::vector<float> Node::StoreTensorInVec(Tensor tensor)
{
	  size_t totSize = GetTotalSize(tensor.shape());
	  std::vector<float> vec(totSize);
	  thrust::copy(tensor.begin(), tensor.end(), vec.begin());
	  return vec;
}

void Node::output(const std::string &title, const std::vector<float> &vec)
{
	  std::cerr << title << "(" << vec.size() << "): ";
	  for (size_t i = 0; i < vec.size(); ++i) {
		  std::cerr << vec[i] << " ";
	  }
	  std::cerr << std::endl;
}


}

