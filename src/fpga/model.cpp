#include "model.h"

namespace amunmt {
namespace FPGA {

Weights::Weights(cl_context &context, const std::string& npzFile, size_t device)
: Weights(context, NpzConverter(npzFile), device)
{}

Weights::Weights(cl_context &context, const NpzConverter& model, size_t device)
: encEmbeddings_(context, model)
, encForwardGRU_(context, model)
, encBackwardGRU_(context, model)
, decEmbeddings_(context, model)
, decInit_(context, model)
, decGru1_(context, model)
, decGru2_(context, model)
, decAlignment_(context, model)
, decSoftmax_(context, model)
, device_(device)
{

}

}
}
