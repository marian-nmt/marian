
#include "marian.h"
#include "mnist.h"
#include "vocab.h"
#include <assert.h>

using namespace marian;
using namespace keywords;

const int input_size = 10;
const int output_size = 15;
const int embedding_size = 8;
const int hidden_size = 5;
const int batch_size = 25;
const int num_inputs = 8;
const int num_outputs = 6;

ExpressionGraph build_graph() {
  std::cerr << "Building computation graph..." << std::endl;

  ExpressionGraph g;
  std::vector<Expr> X, Y, H, S;

  // We're including the stop symbol here.
  for (int t = 0; t <= num_inputs; ++t) {
    std::stringstream ss;
    ss << "X" << t;
    X.emplace_back(named(g.input(shape={batch_size, input_size}), ss.str()));
  }

  // We're including the stop symbol here.
  for (int t = 0; t <= num_outputs; ++t) {
    std::stringstream ss;
    ss << "Y" << t;
    Y.emplace_back(named(g.input(shape={batch_size, output_size}), ss.str()));
  }

  // Source embeddings.
  Expr E = named(g.param(shape={input_size, embedding_size},
                         init=uniform()), "E");

  // Source RNN parameters.
  Expr Wxh = named(g.param(shape={embedding_size, hidden_size},
                   init=uniform()), "Wxh");
  Expr Whh = named(g.param(shape={hidden_size, hidden_size},
                   init=uniform()), "Whh");
  Expr bh = named(g.param(shape={1, hidden_size},
                  init=uniform()), "bh");
  Expr h0 = named(g.param(shape={1, hidden_size},
                  init=uniform()), "h0");

  std::cerr << "Building encoder RNN..." << std::endl;
  H.emplace_back(tanh(dot(dot(X[0], E), Wxh) + dot(h0, Whh) + bh));
  for (int t = 1; t <= num_inputs; ++t) {
    H.emplace_back(tanh(dot(dot(X[t], E), Wxh) + dot(H[t-1], Whh) + bh));
  }

  // Target RNN parameters.
  Expr Wxh_d = named(g.param(shape={output_size, hidden_size},
                     init=uniform()), "Wxh_d");
  Expr Whh_d = named(g.param(shape={hidden_size, hidden_size},
                     init=uniform()), "Whh_d");
  Expr bh_d = named(g.param(shape={1, hidden_size},
                    init=uniform()), "bh_d");

  std::cerr << "Building decoder RNN..." << std::endl;
  auto h0_d = H[num_inputs];
  S.emplace_back(tanh(dot(Y[0], Wxh_d) + dot(h0_d, Whh_d) + bh_d));
  for (int t = 1; t < num_outputs; ++t) {
    S.emplace_back(tanh(dot(Y[t], Wxh_d) + dot(S[t-1], Whh_d) + bh_d));
  }

  // Output linear layer before softmax.
  Expr Why = named(g.param(shape={hidden_size, output_size},
                           init=uniform()), "Why");
  Expr by = named(g.param(shape={1, output_size},
                          init=uniform()), "by");

  std::cerr << "Building output layer..." << std::endl;

  // Softmax layer and cost function.
  std::vector<Expr> Yp;
  Yp.emplace_back(named(softmax(dot(h0_d, Why) + by), "pred"));
  Expr cross_entropy = sum(Y[0] * log(Yp[0]), axis=1);
  for (int t = 1; t <= num_outputs; ++t) {
    Yp.emplace_back(named(softmax(dot(S[t-1], Why) + by), "pred"));
    cross_entropy = cross_entropy + sum(Y[t] * log(Yp[t]), axis=1);
  }
  auto cost = named(-mean(cross_entropy, axis=0), "cost");

  std::cerr << "Done." << std::endl;

  return g;
}

int main(int argc, char** argv) {
#if 1
  std::cerr << "Loading the data... ";
  Vocab sourceVocab, targetVocab;

  // read parallel corpus from file
  std::fstream sourceFile("../examples/mt/dev/newstest2013.de");
  std::fstream targetFile("../examples/mt/dev/newstest2013.en");

  std::vector<std::vector<size_t> > source_sentences, target_sentences;
  std::string sourceLine, targetLine;
  while (getline(sourceFile, sourceLine)) {
    getline(targetFile, targetLine);
    std::vector<size_t> sourceIds = sourceVocab.ProcessSentence(sourceLine);
    std::vector<size_t> targetIds = targetVocab.ProcessSentence(targetLine);
    source_sentences.push_back(sourceIds);
    target_sentences.push_back(targetIds);
  }
  std::cerr << "Done." << std::endl;
  std::cerr << source_sentences.size()
            << " sentence pairs read." << std::endl;
  std::cerr << "Source vocabulary size: " << sourceVocab.Size() << std::endl;
  std::cerr << "Target vocabulary size: " << targetVocab.Size() << std::endl;
#endif

  // Build the encoder-decoder computation graph.
  ExpressionGraph g = build_graph();

  // Generate input data (include the stop symbol).
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

  // Generate output data (include the stop symbol).
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

  std::cerr << "Printing the computation graph..." << std::endl;
  std::cout << g.graphviz() << std::endl;

  std::cerr << "Running the forward step..." << std::endl;
  g.forward(batch_size);
  std::cerr << "Running the backward step..." << std::endl;
  g.backward();
  std::cerr << "Done." << std::endl;

  std::cerr << g["cost"].val().Debug() << std::endl;

  std::cerr << g["X0"].val().Debug() << std::endl;
  std::cerr << g["Y0"].val().Debug() << std::endl;
  std::cerr << g["Whh"].grad().Debug() << std::endl;
  std::cerr << g["bh"].grad().Debug() << std::endl;
  std::cerr << g["Why"].grad().Debug() << std::endl;
  std::cerr << g["by"].grad().Debug() << std::endl;
  std::cerr << g["Wxh"].grad().Debug() << std::endl;
  std::cerr << g["h0"].grad().Debug() << std::endl;

  return 0;
}
