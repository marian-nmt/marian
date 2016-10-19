#include "node.h"
#include "tensor_operators.h"
#include "expression_graph.h"

namespace marian {

void Node::skip_training() {
  skipTraining_ = true;
  graph_->remove_top_node(shared_from_this());
}

void Node::allocate(size_t batchSize) {
  auto it1 = shape_.begin();
  auto it2 = givenShape_.begin();
  while(it1 != shape_.end()) {
    if(*it2 == whatevs)
      *it1 = batchSize;
    it1++; it2++;
  }

  val_ = graph_->tensor(shape_);
  if(Has(keywords::value))
    val_->set(Get(keywords::value, 0));
}

void Node::init_dependent() {
  if(!adj_)
    adj_ = graph_->tensor(shape_);
  adj_->set(1);
}

void Node::set_zero_adjoint() {
  if(!adj_)
    adj_ = graph_->tensor(shape_);
  adj_->set(0);
}


// GPU
void Node::calc_numeric_grad(Float delta, Tensor input, Tensor grad) {
  using namespace std;

//  size_t inputSize = GetTotalSize(input->shape());
//  size_t valSize = GetTotalSize(val_->shape());
//
//  UTIL_THROW_IF2(inputSize != GetTotalSize(grad->shape()),
//			  "inputSize != gradSize:" << inputSize << "!=" << grad->shape()->elements());
//  UTIL_THROW_IF2(valSize != GetTotalSize(adj_->shape()),
//			  "valSize != adjSize :" << valSize << "!=" << adj_->shape()->elements());
//
//  cerr	<< "inputSize=grad=" << Debug(input->shape())<< "=" << inputSize << " "
//		<< "valSize=adj_=" << Debug(val_->shape()) << "=" << valSize
//		<< endl;
//
//  //cerr << "input=" << input.Debug() << endl;
//  //cerr << "adj_=" << adj_.Debug() << endl;
//
//  std::vector<float> prevCalcGrad;
//  prevCalcGrad << grad;
//  //cerr << "origGrad=" << grad.Debug() << endl;
//  //output("diffGrad", diffGrad);
//
//  //output("prevCalcGrad", prevCalcGrad.begin(), prevCalcGrad.end());
//
//  Tensor newValTensor(input.shape());
//
//  // LOOP thru each element in input & add delta
//  for (size_t inputInd = 0; inputInd < inputSize; ++inputInd) {
//	  input.incr(inputInd, delta);
//	  //output("input", input.begin(), input.end());
//
//	  forward();
//
//	  val_.sum(newValTensor, inputInd);
//	  //cudaDeviceSynchronize();
//
//	  input.incr(inputInd, -delta);
//  }
//
//  std::vector<float> newVal;
//  newVal << newValTensor;
//  //cudaDeviceSynchronize();
//
//  // orig value
//  forward();
//
//  float sumValOrig = val_.sum();
//  //float sumValOrig = thrust::reduce(val_.begin(), val_.end(), (float) 0.0f, thrust::plus<float>());
//  //cudaDeviceSynchronize();
//
//  //output("newVal", newVal.begin(), newVal.end());
//
//  // calc gradient
//  Tensor prevGradTensor(input.shape());
//  thrust::copy(grad.begin(), grad.end(), prevGradTensor.begin());
//
//  Tensor gradTensor(input.shape());
//  Element(_1 = (_2 - sumValOrig) / delta, gradTensor, newValTensor);
//  Element(_1 = _2 * _3 + _4, grad, adj_, gradTensor, prevGradTensor);
}

/*
// CPU
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

  std::vector<float> inputVec;
  inputVec << input;
  //output("inputVec", inputVec);

  std::vector<float> newVal(inputSize, 0);

  // LOOP thru each element in input & add delta
  for (size_t inputInd = 0; inputInd < inputSize; ++inputInd) {
	  inputVec[inputInd] += delta;
	  input << inputVec;
	  //output("input", input.begin(), input.end());

	  forward();

	  for (size_t i = 0; i < valSize; ++i) {
		  newVal[inputInd] += val_[i];
	  }
	  //output("val_", val_.begin(), val_.end());

	  inputVec[inputInd] -= delta;
  }

  // orig value
  input << inputVec;
  forward();

  float sumValOrig = 0;
  for (size_t i = 0; i < valSize; ++i) {
	  sumValOrig += val_[i];
  }

  //output("newVal", newVal.begin(), newVal.end());

  // calc gradient
  //cerr << "adj_=" << adj_.Debug() << endl;
  std::vector<float> adjVec;
  adjVec << adj_;

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
  grad << numericalGrad;
  //output("numericalGrad", numericalGrad);
}
*/

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

}
