#include <vector>
#include <random>
#include "marian.h"
#include "expression_graph.h"
#include "keywords.h"
#include "definitions.h"
#include "batch_generator.h"

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
  using namespace data;

  int input_size = 30;
  int output_size = 30;
  int batch_size = 25;

  // define graph
  ExpressionGraph g;
  Expr inExpr = g.input(shape={batch_size, input_size});
  Expr labelExpr = g.input(shape={batch_size, output_size});

  Expr inExpr2 = g.input(shape={batch_size, input_size});

  vector<Expr> expr;

  expr.emplace_back(inExpr + inExpr2);
  expr.emplace_back(inExpr - expr.back());
  expr.emplace_back(inExpr * expr.back());
  expr.emplace_back(inExpr / expr.back());
  expr.emplace_back(reluplus(inExpr, expr.back()));

  //expr.emplace_back(dot(inExpr, inExpr3));

  expr.emplace_back(tanh(expr.back()));
  expr.emplace_back(-expr.back());
  expr.emplace_back(logit(expr.back()));
  expr.emplace_back(relu(expr.back()));
  expr.emplace_back(log(expr.back()));
  expr.emplace_back(exp(expr.back()));
  expr.emplace_back(dropout(expr.back()));
  //expr.emplace_back(softmax_slow(expr.back()));
  expr.emplace_back(softmax(expr.back()));

  Expr ceExpr = cross_entropy(expr.back(), labelExpr);
  Expr cost = mean(ceExpr, axis=0);

  std::cout << g.graphviz() << std::endl;

  // create data
  srand(0);

  BatchPtr batch(new data::Batch());

  Input values1({batch_size, input_size});
  Input labels({batch_size, input_size});
  Input values2({batch_size, input_size});

  generate(begin(values1), end(values1), Rand);
  generate(begin(labels), end(labels), Rand);
  generate(begin(values2), end(values2), Rand);

  batch->push_back(values1);
  batch->push_back(labels);
  batch->push_back(values2);

  g.forward(batch);
  //g.backward();
  g.backward_debug(0.001);

  /*
  std::cerr << "inTensor=" << inTensor.Debug() << std::endl;

  Tensor outTensor = outExpr.val();
  std::cerr << "outTensor=" << outTensor.Debug() << std::endl;

  Tensor outGrad = outExpr.grad();
  std::cerr << "outGrad=" << outGrad.Debug() << std::endl;
  */

}
