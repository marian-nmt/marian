
#include "marian.h"

int main(int argc, char** argv) {

  using namespace marian;
  using namespace keywords;
  
  auto x = input(shape={whatevs, 784}, name="X");
  auto y = input(shape={whatevs, 10}, name="Y");
  
  auto w = param(shape={784, 10}, name="W0");
  auto b = param(shape={1, 10}, name="b0");
  
  auto scores = dot(x, w) + b;
  auto lr = softmax(scores, axis=1, name="pred");
  auto graph = -mean(sum(y * log(lr), axis=1), axis=0, name="cost");
  
  Tensor tx({500, 784}, 1);
  Tensor ty({500, 10}, 1);
  
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
  
  //std::cerr << graph["pred"].val()[0] << std::endl;
  
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