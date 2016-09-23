#include "node.h"
#include "tensor_operators.h"

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

  std::vector<float> prevCalcGrad;
  prevCalcGrad << grad;
  //cerr << "origGrad=" << grad.Debug() << endl;
  //output("diffGrad", diffGrad);

  //output("prevCalcGrad", prevCalcGrad.begin(), prevCalcGrad.end());

  Tensor newValTensor(input.shape());

  // LOOP thru each element in input & add delta
  for (size_t inputInd = 0; inputInd < inputSize; ++inputInd) {
	  input.incr(inputInd, delta);
	  //output("input", input.begin(), input.end());

	  forward();

	  val_.sum(newValTensor, inputInd);
	  //cudaDeviceSynchronize();

	  input.incr(inputInd, -delta);
  }

  std::vector<float> newVal;
  newVal << newValTensor;
  //cudaDeviceSynchronize();

  // orig value
  forward();

  float sumValOrig = val_.sum();
  //float sumValOrig = thrust::reduce(val_.begin(), val_.end(), (float) 0.0f, thrust::plus<float>());
  //cudaDeviceSynchronize();

  //output("newVal", newVal.begin(), newVal.end());

  // calc gradient
  Tensor prevGradTensor(input.shape());
  thrust::copy(grad.begin(), grad.end(), prevGradTensor.begin());

  Tensor gradTensor(input.shape());
  Element(_1 = (_2 - sumValOrig) / delta, gradTensor, newValTensor);
  Element(_1 = _2 * _3 + _4, grad, adj_, gradTensor, prevGradTensor);
}

void Node::outputL2Norm(const std::string &str, const std::vector<float> &x, const std::vector<float> &y) const
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
  cerr << "L2-norm of difference " << typeid(*this).name() << ":" << str << "=" << L2Norm(diff) << endl << endl;
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

