
#include "marian.h"
#include "mnist.h"

int main(int argc, char** argv) {
  cudaSetDevice(0);

  using namespace marian;
  using namespace keywords;

  int input_size = 10;
  int output_size = 2;
  int batch_size = 25;
  int hidden_size = 5;
  int num_inputs = 8;

  std::vector<Expr*> X(num_inputs);
  std::vector<Expr*> Y(num_inputs);
  std::vector<Expr*> H(num_inputs);

  for (int t = 0; t < num_inputs; ++t) {
    X[t] = new Expr(input(shape={batch_size, input_size}));
    Y[t] = new Expr(input(shape={batch_size, output_size}));
  }

  Expr Wxh = param(shape={input_size, hidden_size}, init=uniform(), name="Wxh");
  Expr Whh = param(shape={hidden_size, hidden_size}, init=uniform(), name="Whh");
  Expr bh = param(shape={1, hidden_size}, init=uniform(), name="bh");
  Expr h0 = param(shape={1, hidden_size}, init=uniform(), name="h0");

  std::cerr << "Building RNN..." << std::endl;
  H[0] = new Expr(tanh(dot(*X[0], Wxh) + dot(h0, Whh) + bh));
  for (int t = 1; t < num_inputs; ++t) {
    H[t] = new Expr(tanh(dot(*X[t], Wxh) + dot(*H[t-1], Whh) + bh));
  }

  Expr Why = param(shape={hidden_size, output_size}, init=uniform(), name="Why");
  Expr by = param(shape={1, output_size}, init=uniform(), name="by");

  std::cerr << "Building output layer..." << std::endl;
  std::vector<Expr*> Yp(num_inputs);

  Expr* cross_entropy = NULL;
  for (int t = 0; t < num_inputs; ++t) {
    Yp[t] = new Expr(softmax_fast(dot(*H[t], Why) + by, name="pred"));
    if (!cross_entropy) {
      cross_entropy = new Expr(sum(*Y[t] * log(*Yp[t]), axis=1));
    } else {
      *cross_entropy = *cross_entropy + sum(*Y[t] * log(*Yp[t]), axis=1);
    }
  }
  auto graph = -mean(*cross_entropy, axis=0, name="cost");

  for (int t = 0; t < num_inputs; ++t) {
    Tensor Xt({batch_size, input_size});
    Tensor Yt({batch_size, output_size});

    float max = 1.;
    std::vector<float> values(batch_size * input_size);
    std::vector<float> classes(batch_size * output_size, 0.0);
    int k = 0;
    int l = 0;
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < input_size; ++j, ++k) {
         values[k] = max * (2.0*static_cast<float>(rand()) / RAND_MAX - 1.0);
      }
      int gold = output_size * static_cast<float>(rand()) / RAND_MAX;
      classes[l + gold] = 1.0;
      l += output_size;
    }

    thrust::copy(values.begin(), values.end(), Xt.begin());
    thrust::copy(classes.begin(), classes.end(), Yt.begin());

    *X[t] = Xt;
    *Y[t] = Yt;
  }

  graph.forward(batch_size);
  graph.backward();

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
