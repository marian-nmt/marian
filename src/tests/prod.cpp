#include "marian.h"

#include "tensors/tensor_operators.h"
#include "tensors/cpu/sharp/sse_gemm.h"

#include <boost/timer/timer.hpp>

int main(int argc, char** argv) {
    using namespace marian;

    auto backend = BackendByDevice({0, DeviceType::cpu}, 0);
    auto alloc = New<TensorAllocator>(backend);
    alloc->reserveExact(1024ul * 1024ul * 1024ul * 20ul);

    Tensor a, c, cq;
    Tensor bt, bq;

    int dimA = 4;
    int dimK = 1024;
    int dimB = 256;
    int N = 10000;

    std::vector<Tensor> bs(N);
    std::vector<Tensor> bqs(N);

    std::vector<float> va;
    for(int i = 0; i < dimA * dimK; ++i)
        va.push_back(i / 10000.f);

    alloc->allocate(a, {dimA, dimK});
    a->set(va);

    alloc->allocate(c,  {dimA, dimB});
    alloc->allocate(cq, {dimA, dimB});

    std::vector<float> vb;
    for(int i = 0; i < dimK * dimB; ++i)
        vb.push_back(i / 10000.f);

    for(auto& b : bs) {
        alloc->allocate(b, {dimK, dimB});
        b->set(vb);
    }

    int num_rows = bs[0]->shape()[-2];
    int width = bs[0]->shape()[-1];
    double quant_mult = pow(2.0, 10.0);
    assert(width % 8 == 0);

    alloc->allocate(bt, bs[0]->shape());
    TransposeND(bt, bs[0], {1, 0});

    for(auto& b : bqs) {
        alloc->allocate(b, {dimK, dimB}, Type::int16);
        Quantize(bt->data(),
                 b->data<__m128i>(),
                 (float)quant_mult,
                 num_rows,
                 width);
    }

    c->set(0.f);
    cq->set(0.f);

    {
        boost::timer::auto_cpu_timer t;
        for(auto b : bs)
            Prod(c, a, b, false, false, 0, 1);
    }

    {
        boost::timer::auto_cpu_timer t;
        for(auto b : bqs)
            ProdInt(cq, a, b, false, false, 0, 1);
    }

    return 0;
}
