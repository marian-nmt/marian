
#include "marian.h"
#include "mnist.h"

using namespace marian;
using namespace keywords;

const int input_size = 10;
const int output_size = 15;
const int batch_size = 25;
const int hidden_size = 5;
const int num_inputs = 8;
const int num_outputs = 6;

ExpressionGraph build_graph(int cuda_device) {
  std::cerr << "Building computation graph..." << std::endl;

  ExpressionGraph g(cuda_device);
  std::vector<Expr> X, Y, H, S;

  // For the stop symbol.
  for (int t = 0; t <= num_inputs; ++t) {
    std::stringstream ss;
    ss << "X" << t;
    X.emplace_back(named(g.input(shape={batch_size, input_size}), ss.str()));
  }

  // For the stop symbol.
  for (int t = 0; t <= num_outputs; ++t) {
    std::stringstream ss;
    ss << "Y" << t;
    Y.emplace_back(named(g.input(shape={batch_size, output_size}), ss.str()));
  }

  Expr Wxh = g.param(shape={input_size, hidden_size}, init=uniform(), name="Wxh");
  Expr Whh = g.param(shape={hidden_size, hidden_size}, init=uniform(), name="Whh");
  Expr bh = g.param(shape={1, hidden_size}, init=uniform(), name="bh");
  Expr h0 = g.param(shape={1, hidden_size}, init=uniform(), name="h0");

  std::cerr << "Building encoder RNN..." << std::endl;
  H.emplace_back(tanh(dot(X[0], Wxh) + dot(h0, Whh) + bh));
  for (int t = 1; t <= num_inputs; ++t) {
    H.emplace_back(tanh(dot(X[t], Wxh) + dot(H[t-1], Whh) + bh));
  }

  Expr Wxh_d = g.param(shape={output_size, hidden_size}, init=uniform(), name="Wxh_d");
  Expr Whh_d = g.param(shape={hidden_size, hidden_size}, init=uniform(), name="Whh_d");
  Expr bh_d = g.param(shape={1, hidden_size}, init=uniform(), name="bh_d");

  std::cerr << "Building decoder RNN..." << std::endl;
  auto h0_d = H[num_inputs];
  S.emplace_back(tanh(dot(Y[0], Wxh_d) + dot(h0_d, Whh_d) + bh_d));
  for (int t = 1; t < num_outputs; ++t) {
    S.emplace_back(tanh(dot(Y[t], Wxh_d) + dot(S[t-1], Whh_d) + bh_d));
  }

  Expr Why = g.param(shape={hidden_size, output_size}, init=uniform(), name="Why");
  Expr by = g.param(shape={1, output_size}, init=uniform(), name="by");

  std::cerr << "Building output layer..." << std::endl;
  std::vector<Expr> Yp;

  Yp.emplace_back(named(softmax_fast(dot(h0_d, Why) + by), "pred"));
  Expr cross_entropy = sum(Y[0] * log(Yp[0]), axis=1);
  for (int t = 1; t <= num_outputs; ++t) {
    Yp.emplace_back(named(softmax_fast(dot(S[t-1], Why) + by), "pred"));
    cross_entropy = cross_entropy + sum(Y[t] * log(Yp[t]), axis=1);
  }
  auto graph = -mean(cross_entropy, axis=0, name="cost");

  std::cerr << "Done." << std::endl;

  return g;
}

#if 0
  // read parallel corpus from file
  std::fstream sourceFile("../examples/mt/dev/newstest2013.de");
  std::fstream targetFile("../examples/mt/dev/newstest2013.en");

  std::string sourceLine, targetLine;
  while (getline(sourceFile, sourceLine)) {
    getline(targetFile, targetLine);
    std::vector<size_t> sourceIds = sourceVocab.ProcessSentence(sourceLine);
    std::vector<size_t> targetIds = sourceVocab.ProcessSentence(targetLine);
  }
#endif


int main(int argc, char** argv) {

  ExpressionGraph g = build_graph(0);

  // For the stop symbol.
  for (int t = 0; t <= num_inputs; ++t) {
    Tensor Xt({batch_size, input_size});

    float max = 1.;
    std::vector<float> values(batch_size * input_size);
    std::vector<float> classes(batch_size * output_size, 0.0);
    int k = 0;
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < input_size; ++j, ++k) {
         values[k] = max * (2.0*static_cast<float>(rand()) / RAND_MAX - 1.0);
      }
    }

    thrust::copy(values.begin(), values.end(), Xt.begin());

    std::stringstream ss;
    ss << "X" << t;
    g[ss.str()] = Xt;

  }

  for (int t = 0; t <= num_outputs; ++t) {
    Tensor Yt({batch_size, output_size});

    std::vector<float> classes(batch_size * output_size, 0.0);
    int l = 0;
    for (int i = 0; i < batch_size; ++i) {
      int gold = output_size * static_cast<float>(rand()) / RAND_MAX;
      classes[l + gold] = 1.0;
      l += output_size;
    }

    thrust::copy(classes.begin(), classes.end(), Yt.begin());

    std::stringstream ss;
    ss << "Y" << t;
    g[ss.str()] = Yt;
  }

  std::cerr << "Graphviz step" << std::endl;
  std::cout << g.graphviz() << std::endl;

  std::cerr << "Forward step" << std::endl;
  g.forward(batch_size);
  std::cerr << "Backward step" << std::endl;
  g.backward();
  std::cerr << "Done" << std::endl;

  std::cerr << g["graph"].val().Debug() << std::endl;

  std::cerr << g["X0"].val().Debug() << std::endl;
  std::cerr << g["Y0"].val().Debug() << std::endl;

#if 0
  std::cerr << Whh.grad().Debug() << std::endl;
  std::cerr << bh.grad().Debug() << std::endl;
  std::cerr << Why.grad().Debug() << std::endl;
  std::cerr << by.grad().Debug() << std::endl;
  std::cerr << Wxh.grad().Debug() << std::endl;
  std::cerr << h0.grad().Debug() << std::endl;
#endif

  return 0;
}
