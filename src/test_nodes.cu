#include <vector>
#include <random>
#include "marian.h"
#include "expression_graph.h"
#include "keywords.h"
#include "definitions.h"


float Rand()
{
	float LO = -10;
	float HI = +20;
	float r3 = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
	return r3;
}

int main(int argc, char** argv)
{
  using namespace std;
  using namespace marian;
  using namespace keywords;

  int input_size = 10;
  int output_size = 10;
  int batch_size = 25;

  // define graph
  ExpressionGraph g;
  Expr inExpr = g.input(shape={batch_size, input_size});
  Expr labelExpr = g.input(shape={batch_size, output_size});

  //Expr outExpr = softmax(inExpr);
  Expr outExpr = tanh(inExpr);
  //Expr outExpr = - inExpr;
  Expr ceExpr = cross_entropy(outExpr, labelExpr);
  Expr cost = mean(ceExpr, axis=0);

  // create data
  srand(0);
  std::vector<float> values(batch_size * input_size);
  generate(begin(values), end(values), Rand);

  std::vector<float> labels(batch_size * input_size);
  generate(begin(labels), end(labels), Rand);

  Tensor inTensor({batch_size, input_size});
  thrust::copy(values.begin(), values.end(), inTensor.begin());

  Tensor labelTensor({batch_size, input_size});
  thrust::copy(labels.begin(), labels.end(), labelTensor.begin());

  inExpr = inTensor;
  labelExpr = labelTensor;

  // train
  g.forward(batch_size);
  //g.backward();
  g.backward_debug(0.00001);

  std::cout << g.graphviz() << std::endl;

  std::cerr << "inTensor=" << inTensor.Debug() << std::endl;

  Tensor outTensor = outExpr.val();
  std::cerr << "outTensor=" << outTensor.Debug() << std::endl;

  Tensor outGrad = outExpr.grad();
  std::cerr << "outGrad=" << outGrad.Debug() << std::endl;


}
