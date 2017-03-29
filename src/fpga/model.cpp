#include "model.h"

namespace amunmt {
namespace FPGA {

Weights::Weights(const OpenCLInfo &openCLInfo, const std::string& npzFile)
: Weights(openCLInfo, NpzConverter(npzFile))
{}

Weights::Weights(const OpenCLInfo &openCLInfo, const NpzConverter& model)
: encEmbeddings_(openCLInfo, model)
, encForwardGRU_(openCLInfo, model)
, encBackwardGRU_(openCLInfo, model)
, decEmbeddings_(openCLInfo, model)
, decInit_(openCLInfo, model)
, decGru1_(openCLInfo, model)
, decGru2_(openCLInfo, model)
, decAlignment_(openCLInfo, model)
, decSoftmax_(openCLInfo, model)
, openCLInfo_(openCLInfo)
{

}

}
}
