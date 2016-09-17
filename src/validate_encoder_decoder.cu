
#include "marian.h"
#include "mnist.h"
#include "vocab.h"
#include <assert.h>

using namespace marian;
using namespace keywords;

ExpressionGraph build_graph(int source_vocabulary_size,
                            int target_vocabulary_size,
                            int embedding_size,
                            int hidden_size,
                            int num_source_tokens,
                            int num_target_tokens) {
  std::cerr << "Building computation graph..." << std::endl;

  int input_size = source_vocabulary_size;
  int output_size = target_vocabulary_size;
  int num_inputs = num_source_tokens;
  int num_outputs = num_target_tokens;

  ExpressionGraph g;
  std::vector<Expr> X, Y, H, S;

  // We're including the stop symbol here.
  for (int t = 0; t <= num_inputs; ++t) {
    std::stringstream ss;
    ss << "X" << t;
    X.emplace_back(named(g.input(shape={whatevs, input_size}), ss.str()));
  }

  // We're including the stop symbol here.
  for (int t = 0; t <= num_outputs; ++t) {
    std::stringstream ss;
    ss << "Y" << t;
    Y.emplace_back(named(g.input(shape={whatevs, output_size}), ss.str()));
  }

  // Source embeddings.
  Expr E = named(g.param(shape={input_size, embedding_size},
                         init=uniform()), "E");

  // Source RNN parameters.
  Expr Wxh = named(g.param(shape={embedding_size, hidden_size},
                   init=uniform(-0.1, 0.1)), "Wxh");
  Expr Whh = named(g.param(shape={hidden_size, hidden_size},
                   init=uniform(-0.1, 0.1)), "Whh");
  Expr bh = named(g.param(shape={1, hidden_size},
                  init=uniform(-0.1, 0.1)), "bh");
  Expr h0 = named(g.param(shape={1, hidden_size},
                  init=uniform(-0.1, 0.1)), "h0");

  std::cerr << "Building encoder RNN..." << std::endl;
  H.emplace_back(tanh(dot(dot(X[0], E), Wxh) + dot(h0, Whh) + bh));
  for (int t = 1; t <= num_inputs; ++t) {
    H.emplace_back(tanh(dot(dot(X[t], E), Wxh) + dot(H[t-1], Whh) + bh));
  }

  // Target RNN parameters.
  Expr Wxh_d = named(g.param(shape={output_size, hidden_size},
                     init=uniform(-0.1, 0.1)), "Wxh_d");
  Expr Whh_d = named(g.param(shape={hidden_size, hidden_size},
                     init=uniform(-0.1, 0.1)), "Whh_d");
  Expr bh_d = named(g.param(shape={1, hidden_size},
                    init=uniform(-0.1, 0.1)), "bh_d");

  std::cerr << "Building decoder RNN..." << std::endl;
  auto h0_d = H[num_inputs];
  S.emplace_back(tanh(dot(Y[0], Wxh_d) + dot(h0_d, Whh_d) + bh_d));
  for (int t = 1; t < num_outputs; ++t) {
    S.emplace_back(tanh(dot(Y[t], Wxh_d) + dot(S[t-1], Whh_d) + bh_d));
  }

  // Output linear layer before softmax.
  Expr Why = named(g.param(shape={hidden_size, output_size},
                           init=uniform(-0.1, 0.1)), "Why");
  Expr by = named(g.param(shape={1, output_size},
                          init=uniform(-0.1, 0.1)), "by");

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
  Vocab source_vocab, target_vocab;

  // read parallel corpus from file
  std::fstream source_file("../examples/mt/dev/newstest2013.de");
  std::fstream target_file("../examples/mt/dev/newstest2013.en");

  // Right now we're only reading the first few sentence pairs, and defining
  // that as the step size.
  int batch_size = 64;
  int num_source_tokens = -1;
  int num_target_tokens = -1;
  std::vector<std::vector<size_t> > source_sentences, target_sentences;
  std::string source_line, target_line;
  while (getline(source_file, source_line)) {
    getline(target_file, target_line);
    std::vector<size_t> source_ids = source_vocab.ProcessSentence(source_line);
    source_ids.push_back(source_vocab.GetEOS()); // Append EOS token.
    std::vector<size_t> target_ids = target_vocab.ProcessSentence(target_line);
    target_ids.push_back(target_vocab.GetEOS()); // Append EOS token.
    source_sentences.push_back(source_ids);
    target_sentences.push_back(target_ids);
    if (num_source_tokens < 0 || source_ids.size() > num_source_tokens) {
      num_source_tokens = source_ids.size();
    }
    if (num_target_tokens < 0 || target_ids.size() > num_target_tokens) {
      num_target_tokens = target_ids.size();
    }
    if (source_sentences.size() == batch_size) break;
  }
  std::cerr << "Done." << std::endl;
  std::cerr << source_sentences.size()
            << " sentence pairs read." << std::endl;
  std::cerr << "Source vocabulary size: " << source_vocab.Size() << std::endl;
  std::cerr << "Target vocabulary size: " << target_vocab.Size() << std::endl;
  std::cerr << "Max source tokens: " << num_source_tokens << std::endl;
  std::cerr << "Max target tokens: " << num_target_tokens << std::endl;

  // Padding the source and target sentences.
  for (auto &sentence : source_sentences) {
    for (int i = sentence.size(); i < num_source_tokens; ++i) {
      sentence.push_back(source_vocab.GetPAD());
    }
  }
  for (auto &sentence : target_sentences) {
    for (int i = sentence.size(); i < num_target_tokens; ++i) {
      sentence.push_back(target_vocab.GetPAD());
    }
  }

  std::cerr << "Building the encoder-decoder computation graph..." << std::endl;

  // Build the encoder-decoder computation graph.
  int embedding_size = 50;
  int hidden_size = 100;
  ExpressionGraph g = build_graph(source_vocab.Size(),
                                  target_vocab.Size(),
                                  embedding_size,
                                  hidden_size,
                                  num_source_tokens-1,
                                  num_target_tokens-1);

  std::cerr << "Attaching the data to the computation graph..." << std::endl;

  // Convert the data to dense one-hot vectors.
  // TODO: make the graph handle sparse indices with a proper lookup layer.
  for (int t = 0; t < num_source_tokens; ++t) {
    Tensor Xt({batch_size, static_cast<int>(source_vocab.Size())});
    std::vector<float> values(batch_size * source_vocab.Size(), 0.0);
    int k = 0;
    for (int i = 0; i < batch_size; ++i) {
      values[k + source_sentences[i][t]] = 1.0;
      k += source_vocab.Size();
    }
    thrust::copy(values.begin(), values.end(), Xt.begin());
    // Attach this slice to the graph.
    std::stringstream ss;
    ss << "X" << t;
    g[ss.str()] = Xt;
  }

  for (int t = 0; t < num_target_tokens; ++t) {
    Tensor Yt({batch_size, static_cast<int>(target_vocab.Size())});
    std::vector<float> values(batch_size * target_vocab.Size(), 0.0);
    int k = 0;
    for (int i = 0; i < batch_size; ++i) {
      values[k + target_sentences[i][t]] = 1.0;
      k += target_vocab.Size();
    }
    thrust::copy(values.begin(), values.end(), Yt.begin());
    // Attach this slice to the graph.
    std::stringstream ss;
    ss << "Y" << t;
    g[ss.str()] = Yt;
  }

#else

  int source_vocabulary_size = 10;
  int target_vocabulary_size = 15;
  int embedding_size = 8;
  int hidden_size = 5;
  int batch_size = 25;
  int num_source_tokens = 8;
  int num_target_tokens = 6;

  // Build the encoder-decoder computation graph.
  ExpressionGraph g = build_graph(0, // cuda device.
                                  source_vocabulary_size,
                                  target_vocabulary_size,
                                  embedding_size,
                                  hidden_size,
                                  num_source_tokens,
                                  num_target_tokens);

  int input_size = source_vocabulary_size;
  int output_size = target_vocabulary_size;
  int num_inputs = num_source_tokens;
  int num_outputs = num_target_tokens;

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
#endif

  std::cerr << "Printing the computation graph..." << std::endl;
  std::cout << g.graphviz() << std::endl;

  std::cerr << "Running the forward step..." << std::endl;
  g.forward(batch_size);
  std::cerr << "Running the backward step..." << std::endl;
  g.backward();
  std::cerr << "Done." << std::endl;

  std::cerr << g["cost"].val().Debug() << std::endl;

#if 0
  std::cerr << g["X0"].val().Debug() << std::endl;
  std::cerr << g["Y0"].val().Debug() << std::endl;
  std::cerr << g["Whh"].grad().Debug() << std::endl;
  std::cerr << g["bh"].grad().Debug() << std::endl;
  std::cerr << g["Why"].grad().Debug() << std::endl;
  std::cerr << g["by"].grad().Debug() << std::endl;
  std::cerr << g["Wxh"].grad().Debug() << std::endl;
  std::cerr << g["h0"].grad().Debug() << std::endl;
#endif

  return 0;
}
