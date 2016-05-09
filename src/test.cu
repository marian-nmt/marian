
#include "marian.h"

int main(int argc, char** argv) {

  using namespace marian;
  using namespace keywords;
  
  auto x = data(shape={whatevs, 784}, name="X");
  auto y = data(shape={whatevs, 10}, name="Y");

  auto w = param(shape={784, 10}, name="W0");
  auto b = param(shape={1, 10}, name="b0");
  
  auto lr = softmax(dot(x, w) + b, axis=1);
  auto cost = -mean(sum(y * log(lr), axis=1), axis=0);
    
  cost.forward();
  
  //auto set = [](size_t i, Expr c) {
  //  size_t bid = (i + 1) % batches;
  //  Tensor x = c["X"].val();
  //  thrust::copy(XBatches[bid].begin(), XBatches[bid].end(),
  //               x.begin());
  //  Tensor y = c["Y"].val();
  //  thrust::copy(YBatches[bid].begin(), YBatches[bid].end(),
  //               y.begin());
  //};
  //
  //auto before = [](size_t i, Expr c) {
  //  for(auto&& p : c.params())
  //    clip(p.grad(), type=norm, max=10);
  //};
  //
  //
  //float sum;
  //auto after = [&sum](size_t i, Expr c) {
  //  sum += c.val()[0];
  //  
  //  if(i % 100 == 0) {
  //    std::cerr << sum / i << std::endl;
  //    std::cerr << i << " : " << c.val()[0] << std::endl;
  //  }
  //    
  //  if(i % 10000 == 0) {
  //    std::cerr << "Saving model " << i << std::endl;
  //    std::stringstream name;
  //    name << "model.iter" << i << ".yml.gz"; 
  //    dump(c, name.str());
  //  }
  //  
  //};
  //
  //auto opt = adadelta(cost_function=cost,
  //                    eta=0.9, gamma=0.1,
  //                    set_batch=set,
  //                    before_update=before,
  //                    after_update=after,
  //                    set_valid=valid,
  //                    validation_freq=100,
  //                    verbose=1, epochs=3, early_stopping=10);
  //opt.run();
  
  return 0;
}