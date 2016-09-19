#include <vector>
#include <random>
#include "marian.h"
#include "expression_graph.h"
#include "keywords.h"
#include "definitions.h"

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

  // create data
  random_device rnd_device;
  mt19937 mersenne_engine(rnd_device());
  uniform_real_distribution<float> dist(-1, 1);
  auto gen = std::bind(dist, mersenne_engine);

  std::vector<float> values(batch_size * input_size);
  generate(begin(values), end(values), gen);

  std::vector<float> labels(batch_size * input_size);
  generate(begin(labels), end(labels), gen);

  Tensor inTensor({batch_size, input_size});
  thrust::copy(values.begin(), values.end(), inTensor.begin());

  Tensor labelTensor({batch_size, input_size});
  thrust::copy(labels.begin(), labels.end(), labelTensor.begin());

  inExpr = inTensor;
  labelExpr = labelTensor;

  //Expr outExpr = softmax(inExpr);
  Expr outExpr = tanh(inExpr);

  g.forward(batch_size);
  g.backward();

  std::cout << g.graphviz() << std::endl;

  std::cerr << "inTensor=" << inTensor.Debug() << std::endl;

  Tensor outTensor = outExpr.val();
  std::cerr << "outTensor=" << outTensor.Debug() << std::endl;

  Tensor outGrad = outExpr.grad();
  std::cerr << "outGrad=" << outGrad.Debug() << std::endl;

}
