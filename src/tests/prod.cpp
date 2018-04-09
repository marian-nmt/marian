#include "marian.h"

#include "tensors/tensor_operators.h"
#include "tensors/cpu/sharp/sse_gemm.h"

int main(int argc, char** argv) {
    using namespace marian;

    auto backend = BackendByDevice({0, DeviceType::cpu}, 0);
    auto alloc = New<TensorAllocator>(backend);

    Tensor a, b, c;

    alloc->allocate(a, {14, 256});
    alloc->allocate(b, {256, 256});
    alloc->allocate(c, {14, 256});

    std::vector<float> va;
    for(int i = 0; i < 14 * 256; ++i)
        va.push_back(i / 10000.f);
    std::vector<float> vb;
    for(int i = 0; i < 256 * 256; ++i)
        vb.push_back(i / 10000.f);

    a->set(va);
    b->set(vb);
    c->set(0);

    Prod(c, a, b, false, false, 0, 1);

    std::cout << a->debug() << std::endl;
    std::cout << b->debug() << std::endl;
    std::cout << c->debug() << std::endl;

    ProdInt(c, a, b, false, false, 0, 1);

    std::cout << c->debug() << std::endl;

    return 0;
}
