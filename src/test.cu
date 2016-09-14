
#include "marian.h"
#include "mnist.h"

using namespace std;

int main(int argc, char** argv) {
  /*auto images = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte");*/
  /*auto labels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte");*/
  /*std::cerr << images.size() << " " << images[0].size() << std::endl;*/

  using namespace marian;
  using namespace keywords;
  

  Expr x = input(shape={whatevs, 784}, name="X");
  Expr y = input(shape={whatevs, 10}, name="Y");
  
  Expr w = param(shape={784, 10}, name="W0");
  Expr b = param(shape={1, 10}, name="b0");
  
  auto scores = dot(x, w) + b;
  auto lr = softmax(scores, axis=1, name="pred");
  auto graph = -mean(sum(y * log(lr), axis=1), axis=0, name="cost");
  cerr << "lr=" << lr.Debug() << endl;

  
  Tensor tx({500, 784}, 1);
  Tensor ty({500, 10}, 1);

  int numImg, imgSize;
  vector<float> images = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte", numImg, imgSize);
  vector<float> labels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte");
  cerr << "images=" << images.size() << " labels=" << labels.size() << endl;
  tx.Load(images);
  //ty.Load(labels);

  cerr << "tx=" << tx.Debug() << endl;
  cerr << "ty=" << ty.Debug() << endl;

  x = tx;
  y = ty;

  graph.forward(500);

  std::cerr << "Result: ";
  for (auto val : scores.val().shape()) {
    std::cerr << val << " ";
  }
  std::cerr << std::endl;
  std::cerr << "Result: ";
  for (auto val : lr.val().shape()) {
    std::cerr << val << " ";
  }
  std::cerr << std::endl;
  std::cerr << "Log-likelihood: ";
  for (auto val : graph.val().shape()) {
    std::cerr << val << " ";
  }
  std::cerr << std::endl;

  graph.backward();
  
  //std::cerr << graph["pred"].val()[0] << std::endl;
  

   // XOR
  /*
  Expr x = input(shape={whatevs, 2}, name="X");
  Expr y = input(shape={whatevs, 2}, name="Y");

  Expr w = param(shape={2, 1}, name="W0");
  Expr b = param(shape={1, 1}, name="b0");

  Expr n5 = dot(x, w);
  Expr n6 = n5 + b;
  Expr lr = softmax(n6, axis=1, name="pred");
  cerr << "lr=" << lr.Debug() << endl;

  Expr graph = -mean(sum(y * log(lr), axis=1), axis=0, name="cost");

  Tensor tx({4, 2}, 1);
  Tensor ty({4, 1}, 1);
  cerr << "tx=" << tx.Debug() << endl;
  cerr << "ty=" << ty.Debug() << endl;

  tx.Load("../examples/xor/train.txt");
  ty.Load("../examples/xor/label.txt");
  */

#if 0  
  hook0(graph);
  graph.autodiff();
  std::cerr << graph["cost"].val()[0] << std::endl;
  //hook1(graph);
  for(auto p : graph.params()) {
    auto update = _1 = _1 - alpha * _2;
    Element(update, p.val(), p.grad());
  }
  hook2(graph);
  
  auto opt = adadelta(cost_function=cost,
                      eta=0.9, gamma=0.1,
                      set_batch=set,
                      before_update=before,
                      after_update=after,
                      set_valid=valid,
                      validation_freq=100,
                      verbose=1, epochs=3, early_stopping=10);
  opt.run();
#endif  
  return 0;
}
