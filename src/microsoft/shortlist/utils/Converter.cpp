#include "microsoft/shortlist/utils/Converter.h"

namespace marian {
namespace quicksand {

#include "microsoft/shortlist/logging/LoggerMacros.h"


int64_t Converter::ToInt64(const std::string& str) {
    return ConvertSingleInternal<int64_t>(str, "int64_t");
}

uint64_t Converter::ToUInt64(const std::string& str) {
    return ConvertSingleInternal<uint64_t>(str, "int64_t");
}
    
int32_t Converter::ToInt32(const std::string& str) {
    return ConvertSingleInternal<int32_t>(str, "int32_t");
}

float Converter::ToFloat(const std::string& str) {
    // In case the value is out of range of a 32-bit float, but in range of a 64-bit double,
    // it's better to convert as a double and then do the conersion.
    return (float)ConvertSingleInternal<double>(str, "float");
}

double Converter::ToDouble(const std::string& str) {
    return ConvertSingleInternal<double>(str, "double");
}

bool Converter::ToBool(const std::string& str) {
    bool value = false;
    if (!TryConvert(str, /* out */ value)) {
        LOG_ERROR_AND_THROW("The string '%s' is not interpretable as the type 'bool'", str.c_str());        
    }
    return value;
}

std::vector<int32_t> Converter::ToInt32Vector(const std::vector<std::string>& items) {
    return ConvertVectorInternal<int32_t, std::vector<std::string>::const_iterator>(items.begin(), items.end(), "int32_t");
}

std::vector<int64_t> Converter::ToInt64Vector(const std::vector<std::string>& items) {
    return ConvertVectorInternal<int64_t, std::vector<std::string>::const_iterator>(items.begin(), items.end(), "int64_t");
}

std::vector<float> Converter::ToFloatVector(const std::vector<std::string>& items) {
    return ConvertVectorInternal<float, std::vector<std::string>::const_iterator>(items.begin(), items.end(), "float");
}

std::vector<double> Converter::ToDoubleVector(const std::vector<std::string>& items) {
    return ConvertVectorInternal<double, std::vector<std::string>::const_iterator>(items.begin(), items.end(), "double");
}

void Converter::HandleConversionError(const std::string& str, const char * type_name) {
    str; type_name; // make compiler happy
    LOG_ERROR_AND_THROW("The string '%s' is not interpretable as the type '%s'", str.c_str(), type_name);
}

} // namespace quicksand
} // namespace marian