
#include "marian.h"
#include "mnist.h"

#if 0
ExpressionGraph build_graph() {
  std::cerr << "Loading model params...";
}

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
  cudaSetDevice(0);

  using namespace marian;
  using namespace keywords;

  int input_size = 10;
  int output_size = 15;
  int batch_size = 25;
  int hidden_size = 5;
  int num_inputs = 8;
  int num_outputs = 6;

  ExpressionGraph g;
  std::vector<Expr*> X(num_inputs+1); // For the stop symbol.
  std::vector<Expr*> Y(num_outputs);
  std::vector<Expr*> H(num_inputs+1); // For the stop symbol.
  std::vector<Expr*> S(num_outputs);

  // For the stop symbol.
  for (int t = 0; t <= num_inputs; ++t) {
    X[t] = new Expr(g.input(shape={batch_size, input_size}));
  }

  // For the stop symbol.
  for (int t = 0; t <= num_outputs; ++t) {
    Y[t] = new Expr(g.input(shape={batch_size, output_size}));
  }

  Expr Wxh = g.param(shape={input_size, hidden_size}, init=uniform(), name="Wxh");
  Expr Whh = g.param(shape={hidden_size, hidden_size}, init=uniform(), name="Whh");
  Expr bh = g.param(shape={1, hidden_size}, init=uniform(), name="bh");
  Expr h0 = g.param(shape={1, hidden_size}, init=uniform(), name="h0");

  std::cerr << "Building encoder RNN..." << std::endl;
  H[0] = new Expr(tanh(dot(*X[0], Wxh) + dot(h0, Whh) + bh));
  for (int t = 1; t <= num_inputs; ++t) {
    H[t] = new Expr(tanh(dot(*X[t], Wxh) + dot(*H[t-1], Whh) + bh));
  }

  Expr Wxh_d = g.param(shape={output_size, hidden_size}, init=uniform(), name="Wxh_d");
  Expr Whh_d = g.param(shape={hidden_size, hidden_size}, init=uniform(), name="Whh_d");
  Expr bh_d = g.param(shape={1, hidden_size}, init=uniform(), name="bh_d");

  std::cerr << "Building decoder RNN..." << std::endl;
  auto h0_d = *H[num_inputs];
  S[0] = new Expr(tanh(dot(*Y[0], Wxh_d) + dot(h0_d, Whh_d) + bh_d));
  for (int t = 1; t < num_outputs; ++t) {
    S[t] = new Expr(tanh(dot(*Y[t], Wxh_d) + dot(*S[t-1], Whh_d) + bh_d));
  }

  Expr Why = g.param(shape={hidden_size, output_size}, init=uniform(), name="Why");
  Expr by = g.param(shape={1, output_size}, init=uniform(), name="by");

  std::cerr << "Building output layer..." << std::endl;
  std::vector<Expr*> Yp(num_outputs+1); // For the stop symbol.

  Expr* cross_entropy = NULL;
  for (int t = 0; t <= num_outputs; ++t) {
    if (t == 0) {
      Yp[t] = new Expr(named(softmax_fast(dot(h0_d, Why) + by), "pred"));
      cross_entropy = new Expr(sum(*Y[t] * log(*Yp[t]), axis=1));
    } else {
      Yp[t] = new Expr(named(softmax_fast(dot(*S[t-1], Why) + by), "pred"));
      *cross_entropy = *cross_entropy + sum(*Y[t] * log(*Yp[t]), axis=1);
    }
  }
  auto graph = -mean(*cross_entropy, axis=0, name="cost");

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

    *X[t] = Xt;
  }

  for (int t = 0; t < num_outputs; ++t) {
    Tensor Yt({batch_size, output_size});

    std::vector<float> classes(batch_size * output_size, 0.0);
    int l = 0;
    for (int i = 0; i < batch_size; ++i) {
      int gold = output_size * static_cast<float>(rand()) / RAND_MAX;
      classes[l + gold] = 1.0;
      l += output_size;
    }

    thrust::copy(classes.begin(), classes.end(), Yt.begin());

    *Y[t] = Yt;
  }

  g.forward(batch_size);
  g.backward();

  std::cerr << graph.val().Debug() << std::endl;

  std::cerr << X[0]->val().Debug() << std::endl;
  std::cerr << Y[0]->val().Debug() << std::endl;

  std::cerr << Whh.grad().Debug() << std::endl;
  std::cerr << bh.grad().Debug() << std::endl;
  std::cerr << Why.grad().Debug() << std::endl;
  std::cerr << by.grad().Debug() << std::endl;
  std::cerr << Wxh.grad().Debug() << std::endl;
  std::cerr << h0.grad().Debug() << std::endl;

  return 0;
}
