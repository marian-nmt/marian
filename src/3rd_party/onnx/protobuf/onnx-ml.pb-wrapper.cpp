// protobuf-generated files don't compile clean. This compiles them with warnings
// disabled, without having to disable it for the entire project whole-sale.

#ifdef USE_ONNX

// Get protobuf this way:
//   sudo apt-get install cmake pkg-config libprotobuf9v5 protobuf-compiler libprotobuf-dev libgoogle-perftools-dev 

// Since we don't develop the ONNX .proto file, I just hand-created the .pb. files.
// The automatic process that CMake would invoke fails because protobuf generates
// source code that is not warning-free. So let's use this manual process for now,
// and just version-control the resulting files. The command is simple enough:
//   cd src/3rd_party/onnx/protobuf
//   protoc -I=. --cpp_out=. onnx-ml.proto

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4125 4127 4244 4267 4512 4456 4510 4610 4800)
#endif
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-variable"  // note: GCC <6.0 ignores this when inside push/pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#define AuxillaryParseTableField AuxiliaryParseTableField  // in protobuf 3.12, the generated source has a spelling error

#include "onnx-ml.pb.cc" // this is the actual file we compile

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif // USE_ONNX
