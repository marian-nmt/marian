#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "mnist.h"
#include "trainer.h"
#include "expression_graph.h"

using std::make_shared;

namespace marian {
namespace data {

class XorTruthTable : public DataBase {

  private:

	Examples examples_;

  public:
  
  	XorTruthTable() {
  		examples_.emplace_back(new Example({ make_shared<Data>(Data{0.0, 0.0}), make_shared<Data>(Data{0.0}) }));
  		examples_.emplace_back(new Example({ make_shared<Data>(Data{1.0, 0.0}), make_shared<Data>(Data{1.0}) }));
  		examples_.emplace_back(new Example({ make_shared<Data>(Data{0.0, 1.0}), make_shared<Data>(Data{1.0}) }));
  		examples_.emplace_back(new Example({ make_shared<Data>(Data{1.0, 1.0}), make_shared<Data>(Data{0.0}) }));
  	}

    ExampleIterator begin() const {
      return ExampleIterator(examples_.begin());
    }

    ExampleIterator end() const {
      return ExampleIterator(examples_.end());
    }

    void shuffle() {
      std::random_shuffle(examples_.begin(), examples_.end());
    }

};

}
}


using namespace marian;
using namespace keywords;
using namespace data;


int main(int argc, char** argv) {

  using namespace keywords;
  std::cerr << "Building single-layer Feedforward network" << std::endl;
  std::cerr << "\tLayer dimensions: 2 2 1" << std::endl;
  boost::timer::cpu_timer timer;

  // Construct a shared pointer to an empty expression graph
  auto g = New<ExpressionGraph>();

  // There are 4 input examples. Each input example has two values. Hence: shape={4, 2}
  // There are 4 labels, one per input example. Each label has one value. Hence shape={4, 1}
  
  auto input   = name(g->input(shape={4, 2}                ), "x" );
  auto weight1 = name(g->param(shape={2, 2}, init=uniform()), "W1");
  auto biasWeight1 = name(g->param(shape={1, 2}, init=zeros), "b1");

  auto hidden  = logit(dot(input, weight1) + biasWeight1);
  auto weight2 = name(g->param(shape={2, 1}, init=uniform()), "W2");

  auto result  = dot(hidden, weight2);
  auto labels  = name(g->input(shape={4, 1}),                  "y");


  auto cost    = name(training(mean(sum((result - labels) * (result - labels), axis=1), axis=0)), "cost");

  auto scores  = name(inference(softmax(result)), "scores");

  g->graphviz("xor.dot");

  auto trainSet = DataSet<XorTruthTable>();
  auto validSet = DataSet<XorTruthTable>();


  auto trainer =
    Run<Trainer>(g, trainSet,
                 optimizer=Optimizer<Adam>(0.0002),
                 batch_size=4,
                 max_epochs=350);
  trainer->run();

  auto validator =
    Run<Validator>(g, validSet,
                   batch_size=4);
  validator->run();



  return 0;
}
