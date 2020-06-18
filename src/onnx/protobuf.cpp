// This builds the runtime library for protobuf on Windows (on Linux it is installed in the OS).
// We include all CPP files from this CPP file. This way, we can find them via the include-path mechanism.
// You will need to set an environment variable PROTOBUF_RUNTIME_INC so that %PROTOBUF_RUNTIME_INC%\google\protobuf exists.

#ifdef _MSC_VER
#ifdef USE_ONNX
// note: some of the below is the result of trial-and-error, not necessarily the minimal set
#include "google/protobuf/stubs/common.cc"
#include "google/protobuf/port_undef.inc"
#undef max
#include "google/protobuf/stubs/bytestream.cc"
#include "google/protobuf/stubs/int128.cc"
#include "google/protobuf/stubs/status.cc"
#include "google/protobuf/port_undef.inc"
#undef max
#include "google/protobuf/stubs/statusor.cc"
#undef min
#include "google/protobuf/stubs/stringpiece.cc"
#include "google/protobuf/stubs/stringprintf.cc"
#include "google/protobuf/stubs/structurally_valid.cc"
namespace google { namespace protobuf { const auto LOGLEVEL_0 = LogLevel::LOGLEVEL_INFO; } }
#include "google/protobuf/stubs/strutil.cc"
#include "google/protobuf/stubs/substitute.cc"
#undef GetCurrentTime
#include "google/protobuf/stubs/time.cc"

#include "google/protobuf/io/coded_stream.cc"
#include "google/protobuf/io/gzip_stream.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/io/io_win32.cc"
#include "google/protobuf/io/printer.cc"
#include "google/protobuf/io/strtod.cc"
#include "google/protobuf/io/tokenizer.cc"
#include "google/protobuf/io/zero_copy_stream.cc"
#include "google/protobuf/io/zero_copy_stream_impl.cc"
#include "google/protobuf/io/zero_copy_stream_impl_lite.cc"

#include "google/protobuf/any.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/any.pb.cc"
#include "google/protobuf/any_lite.cc"
#define schemas schemas1
#define file_default_instances file_default_instances1
#include "google/protobuf/api.pb.cc"
#include "google/protobuf/arena.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/descriptor.cc"
#include "google/protobuf/port_undef.inc"
#define schemas schemas2
#define file_default_instances file_default_instances2
#include "google/protobuf/descriptor.pb.cc"
#include "google/protobuf/descriptor_database.cc"
#define schemas schemas3
#define file_default_instances file_default_instances3
#include "google/protobuf/duration.pb.cc"
#include "google/protobuf/dynamic_message.cc"
#define schemas schemas4
#define file_default_instances file_default_instances4
#include "google/protobuf/empty.pb.cc"
#include "google/protobuf/extension_set_heavy.cc"
#include "google/protobuf/port_undef.inc"
#define cpp_type cpp_type1
#define real_type real_type1
#include "google/protobuf/extension_set.cc"
#undef real_type1
#undef cpp_type
#include "google/protobuf/port_undef.inc"
#define schemas schemas5
#define file_default_instances file_default_instances5
#include "google/protobuf/field_mask.pb.cc"
#include "google/protobuf/generated_enum_util.cc"
#define IsMapFieldInApi IsMapFieldInApi1
#undef schemas
#include "google/protobuf/generated_message_reflection.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/generated_message_table_driven.cc"
#define MutableUnknownFields MutableUnknownFields1
#include "google/protobuf/generated_message_table_driven_lite.cc"
#undef MutableUnknownFields
#include "google/protobuf/generated_message_util.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/implicit_weak_message.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/map_field.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/message.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/message_lite.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/parse_context.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/reflection_ops.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/repeated_field.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/service.cc"
#define schemas schemas6
#define file_default_instances file_default_instances6
#define schemas schemas7
#define file_default_instances file_default_instances7
#include "google/protobuf/source_context.pb.cc"
#define schemas schemas8
#define file_default_instances file_default_instances8
#include "google/protobuf/struct.pb.cc"
#include "google/protobuf/text_format.cc"
#include "google/protobuf/port_undef.inc"
#define schemas schemas9
#define file_default_instances file_default_instances9
#include "google/protobuf/timestamp.pb.cc"
#define schemas schemasa
#define file_default_instances file_default_instancesa
#include "google/protobuf/type.pb.cc"
#include "google/protobuf/unknown_field_set.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/wire_format.cc"
#include "google/protobuf/port_undef.inc"
#include "google/protobuf/wire_format_lite.cc"
#include "google/protobuf/port_undef.inc"
#define schemas schemasb
#define file_default_instances file_default_instancesb
#include "google/protobuf/wrappers.pb.cc"
#endif
#endif
