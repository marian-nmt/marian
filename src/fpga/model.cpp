#include "model.h"

namespace amunmt {
namespace FPGA {

Weights::Weights(cl_context &context, const cl_device_id &device, const std::string& npzFile)
: Weights(context, device, NpzConverter(npzFile))
{}

Weights::Weights(cl_context &context, const cl_device_id &device, const NpzConverter& model)
: encEmbeddings_(context, device, model)
, encForwardGRU_(context, device, model)
, encBackwardGRU_(context, device, model)
, decEmbeddings_(context, device, model)
, decInit_(context, device, model)
, decGru1_(context, device, model)
, decGru2_(context, device, model)
, decAlignment_(context, device, model)
, decSoftmax_(context, device, model)
, device_(device)
{

}

}
}
