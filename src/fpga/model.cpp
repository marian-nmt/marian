#include "model.h"

namespace amunmt {
namespace FPGA {

Weights::Weights(const OpenCLInfo &openCLInfo, const std::string& npzFile)
: Weights(openCLInfo, NpzConverter(npzFile))
{}

Weights::Weights(const OpenCLInfo &openCLInfo, const NpzConverter& model)
: encEmbeddings_(openCLInfo.context, openCLInfo.device, model)
, encForwardGRU_(openCLInfo.context, openCLInfo.device, model)
, encBackwardGRU_(openCLInfo.context, openCLInfo.device, model)
, decEmbeddings_(openCLInfo.context, openCLInfo.device, model)
, decInit_(openCLInfo.context, openCLInfo.device, model)
, decGru1_(openCLInfo.context, openCLInfo.device, model)
, decGru2_(openCLInfo.context, openCLInfo.device, model)
, decAlignment_(openCLInfo.context, openCLInfo.device, model)
, decSoftmax_(openCLInfo.context, openCLInfo.device, model)
, openCLInfo_(openCLInfo)
{

}

}
}
