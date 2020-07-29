#include "graph/expression_graph.h"
#include <memory>

namespace faiss {
  struct IndexLSH;
}

namespace marian {

class LSH {  
public:
  LSH(int k, int nbits) : k_{k}, nbits_{nbits} {}
  Expr apply(Expr query, Expr values, Expr bias);

private:
  Ptr<faiss::IndexLSH> index_;
  size_t indexHash_{0};

  int k_{100};
  int nbits_{1024};

  Expr search(Expr query, Expr values);
  Expr affine(Expr idx, Expr query, Expr values, Expr bias);
};

}