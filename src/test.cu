
#include "marian.h"
#include "mnist.h"

int main(int argc, char** argv) {
  cudaSetDevice(0);

  /*int numImg = 0;*/
  /*auto images = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte", numImg);*/
  /*auto labels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte", numImg);*/

#if 1
  using namespace marian;
  using namespace keywords;

  Expr x = input(shape={1, 2});
  Expr y = input(shape={1, 2});
  
  Expr w = param(shape={2, 2}, name="W0");
  //Expr b = param(shape={1, 2}, name="b0");

  std::cerr << "Building model...";
  auto predict = softmax_fast(dot(x, w),
                         axis=1, name="pred");
  auto graph = -mean(sum(y * log(predict), axis=1),
                     axis=0, name="cost");
  
  Tensor x1t({1, 2});
  std::vector<float> xv = { 0.6, 0.1 }; 
  thrust::copy(xv.begin(), xv.end(), x1t.begin());
  
  Tensor x2t({1, 2});
  std::vector<float> yv = { 0, 1 }; 
  thrust::copy(yv.begin(), yv.end(), x2t.begin());
  
  x = x1t;
  y = x2t;
  
  graph.forward(1);
  graph.backward();
  
  std::cerr << graph.val().Debug() << std::endl;
  std::cerr << w.grad().Debug() << std::endl;
  //std::cerr << b.grad().Debug() << std::endl;
#else

  
  using namespace marian;
  using namespace keywords;
  using namespace std;
  
  const size_t BATCH_SIZE = 500;
  const size_t IMAGE_SIZE = 784;
  const size_t LABEL_SIZE = 10;

  Expr x = input(shape={whatevs, IMAGE_SIZE}, name="X");
  Expr y = input(shape={whatevs, LABEL_SIZE}, name="Y");
  
  Expr w = param(shape={IMAGE_SIZE, LABEL_SIZE}, name="W0");
  Expr b = param(shape={1, LABEL_SIZE}, name="b0");
    
  Expr z = dot(x, w) + b;
  Expr lr = softmax(z, axis=1, name="pred");
  Expr graph = -mean(sum(y * log(lr), axis=1), axis=0, name="cost");
  //cerr << "x=" << Debug(lr.val().shape()) << endl;

  int numofdata;
  //vector<float> images = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte", numofdata, IMAGE_SIZE);
  //vector<float> labels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte", numofdata, LABEL_SIZE);
  vector<float> images = datasets::mnist::ReadImages("../examples/mnist/train-images-idx3-ubyte", numofdata, IMAGE_SIZE);
  vector<float> labels = datasets::mnist::ReadLabels("../examples/mnist/train-labels-idx1-ubyte", numofdata, LABEL_SIZE);
  cerr << "images=" << images.size() << " labels=" << labels.size() << endl;
  cerr << "numofdata=" << numofdata << endl;

  size_t startInd = 0;
  size_t startIndData = 0;
  while (startInd < numofdata) {
	  size_t batchSize = (startInd + BATCH_SIZE < numofdata) ? BATCH_SIZE : numofdata - startInd;
	  cerr << "startInd=" << startInd
			  << " startIndData=" << startIndData
			  << " batchSize=" << batchSize << endl;

	  Tensor tx({numofdata, IMAGE_SIZE}, 1);
	  Tensor ty({numofdata, LABEL_SIZE}, 1);

	  tx.set(images.begin() + startIndData, images.begin() + startIndData + batchSize * IMAGE_SIZE);
	  ty.set(labels.begin() + startInd, labels.begin() + startInd + batchSize);

	  //cerr << "tx=" << Debug(tx.shape()) << endl;
	  //cerr << "ty=" << Debug(ty.shape()) << endl;

	  x = tx;
	  y = ty;

	  cerr << "x=" << Debug(x.val().shape()) << endl;
	  cerr << "y=" << Debug(y.val().shape()) << endl;


	  graph.forward(batchSize);

	  cerr << "w=" << Debug(w.val().shape()) << endl;
	  cerr << "b=" << Debug(b.val().shape()) << endl;
	  std::cerr << "z: " << Debug(z.val().shape()) << endl;
	  std::cerr << "lr: " << Debug(lr.val().shape()) << endl;
	  std::cerr << "Log-likelihood: " << graph.val().Debug() << endl ;

	  //std::cerr << "scores=" << scores.val().Debug() << endl;
	  //std::cerr << "lr=" << lr.val().Debug() << endl;

	  graph.backward();
          std::cerr << w.grad().Debug() << std::endl;

	  //std::cerr << graph["pred"].val()[0] << std::endl;

	  startInd += batchSize;
	  startIndData += batchSize * IMAGE_SIZE;
  }
#endif

  return 0;
}
