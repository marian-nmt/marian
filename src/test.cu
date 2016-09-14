
#include "marian.h"

using namespace std;

int main(int argc, char** argv) {

  using namespace marian;
  using namespace keywords;
  
  Expr x = input(name="X");
  Expr y = input(name="Y");
  
  Expr w = param(shape={784, 10}, name="W0");
  Expr b = param(shape={1, 10}, name="b0");
  
  Expr pred = softmax(dot(x, w) + b, axis=1);
  cerr << "lr=" << pred.Debug() << endl;

  Expr graph = -mean(sum(y * log(pred), axis=1),
                     axis=0, name="cost");
  
  Tensor tx({500, 784}, 1);
  Tensor ty({500, 10}, 1);
  
  cerr << "tx=" << tx.Debug() << endl;
  cerr << "ty=" << ty.Debug() << endl;

  x = tx;
  y = ty;

  graph.forward(500);
  graph.backward();
  
  return 0;
}
