#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <sstream>

namespace marian {
namespace quicksand {

class Converter {
public:
    static int32_t ToInt32(const std::string& str);

    static int64_t ToInt64(const std::string& str);
    
    static uint64_t ToUInt64(const std::string& str);
        
    static float ToFloat(const std::string& str);
    
    static double ToDouble(const std::string& str);
    
    static bool ToBool(const std::string& str);
    
    static std::vector<int32_t> ToInt32Vector(const std::vector<std::string>& items);

    static std::vector<int64_t> ToInt64Vector(const std::vector<std::string>& items);
    
    static std::vector<float> ToFloatVector(const std::vector<std::string>& items);
    
    static std::vector<double> ToDoubleVector(const std::vector<std::string>& items);
    
    static bool TryConvert(const std::string& str, /* out*/ bool& obj) {
        if (str == "True" || str == "true" || str == "TRUE" || str == "Yes" || str == "yes" || str == "1") {
            obj = true;
            return true;
        }
        else if (str == "False" || str == "false" || str == "FALSE" || str == "No" || str == "no" || str == "0") {
            obj = false;
            return true;
        }
        return false;
    }

    template <typename T>
    static bool TryConvert(const std::string& str, /* out*/ T& value) {
        std::istringstream ss(str);
        value = T();
        if (!(ss >> value)) {
            return false;
        }
        return true;
    }
    
private:
    template <typename T>
    static T ConvertSingleInternal(const std::string& str, const char * type_name);
    
    template <typename T, typename I>
    static std::vector<T> ConvertVectorInternal(I begin, I end, const char * type_name);
    
    static void HandleConversionError(const std::string& str, const char * type_name);
};

template <typename T>
T Converter::ConvertSingleInternal(const std::string& str, const char * type_name) {
    std::istringstream ss(str);
    T value = T();
    if (!(ss >> value)) {
        HandleConversionError(str, type_name);
    }
    return value;
}

template <typename T, typename I>
std::vector<T> Converter::ConvertVectorInternal(I begin, I end, const char * type_name) {
    std::vector<T> items;
    for (I it = begin; it != end; it++) {
        items.push_back(ConvertSingleInternal<T>(*it, type_name));
    }
    return items;
}

} // namespace quicksand
} // namespace marian