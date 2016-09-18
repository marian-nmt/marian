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
  int batch_size = 25;

  // define graph
  ExpressionGraph g;
  Expr inputExpr = g.input(shape={batch_size, input_size});

  // create data
  random_device rnd_device;
  mt19937 mersenne_engine(rnd_device());
  uniform_real_distribution<float> dist(1, 52);
  auto gen = std::bind(dist, mersenne_engine);

  std::vector<float> values(batch_size * input_size);
  generate(begin(values), end(values), gen);


  Tensor inputTensor({batch_size, input_size});
  thrust::copy(values.begin(), values.end(), inputTensor.begin());

  inputExpr = inputTensor;
  Expr softMaxExpr = softmax(inputExpr);

  g.forward(batch_size);
  g.backward();

  std::cout << g.graphviz() << std::endl;

  std::cerr << "inputTensor=" << inputTensor.Debug() << std::endl;

  Tensor softMaxTensor = softMaxExpr.val();
  std::cerr << "softMaxTensor=" << softMaxTensor.Debug() << std::endl;

  Tensor softMaxGrad = softMaxExpr.grad();
  std::cerr << "softMaxGrad=" << softMaxGrad.Debug() << std::endl;

}
