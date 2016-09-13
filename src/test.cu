
#include "marian.h"

using namespace std;

int main(int argc, char** argv) {

  using namespace marian;
  using namespace keywords;
  
  Expr x = input(shape={whatevs, 784}, name="X");
  Expr y = input(shape={whatevs, 10}, name="Y");
  
  Expr w = param(shape={784, 10}, name="W0");
  Expr b = param(shape={1, 10}, name="b0");
  
  Expr n5 = dot(x, w);
  Expr n6 = n5 + b;
  Expr lr = softmax(n6, axis=1, name="pred");
  cerr << "lr=" << lr.Debug() << endl;

  Expr graph = -mean(sum(y * log(lr), axis=1), axis=0, name="cost");
  
  Tensor tx({500, 784}, 1);
  Tensor ty({500, 10}, 1);
  cerr << "tx=" << tx.Debug() << endl;
  cerr << "ty=" << ty.Debug() << endl;

  x = tx;
  y = ty;

  graph.forward(500);
  graph.backward();
  //std::cerr << graph["pred"].val()[0] << std::endl;
  
  
  //hook0(graph);
  //graph.autodiff();
  //std::cerr << graph["cost"].val()[0] << std::endl;
  //hook1(graph);
  //for(auto p : graph.params()) {
  //  auto update = _1 = _1 - alpha * _2;
  //  Element(update, p.val(), p.grad());
  //}
  //hook2(graph);
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

  Expr x2 = input(shape={whatevs, 2}, name="X2");
  Expr y2 = input(shape={whatevs, 2}, name="Y2");
  
  Expr w2 = param(shape={2, 1}, name="W02");
  Expr b2 = param(shape={1, 1}, name="b02");

  Expr n52 = dot(x2, w2);
  Expr n62 = n52 + b2;
  Expr lr2 = softmax(n62, axis=1, name="pred2");
  cerr << "lr=" << lr2.Debug() << endl;

  return 0;
}
