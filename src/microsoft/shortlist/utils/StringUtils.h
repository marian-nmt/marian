#pragma once
 
#include <string>
#include <sstream>
#include <stdarg.h>
#include <vector>
#include <stdint.h>

#include "microsoft/shortlist/utils/PrintTypes.h"

namespace marian {
namespace quicksand {

class StringUtils {
public:
    template <typename T>
    static std::string Join(const std::string& joiner, const T& items);
        
    template <typename T>
    static std::string Join(const std::string& joiner, const T * items, int32_t length);

    static std::string Join(const std::string& joiner, const uint8_t * items, int32_t length);
    
    static std::string Join(const std::string& joiner, const int8_t * items, int32_t length);
    
    static std::vector<std::string> Split(const std::string& input, char splitter);
    
    static std::vector<std::string> Split(const std::string& input, const std::string& splitter);
    
    static std::vector<std::string> SplitFileList(const std::string& input);
    
    static std::string PrintString(const char * format, ...);
    
    static std::string VarArgsToString(const char * format, va_list args);
    
    static std::vector<std::string> WhitespaceTokenize(const std::string& input);
    
    static std::string CleanupWhitespace(const std::string& input);

    static std::string ToString(const std::string& str);
    
    static std::string ToString(bool obj);
    
    template <typename T>
    static std::string ToString(const T& obj);
    
    static std::string XmlEscape(const std::string& str);
    
    static std::vector<std::string> SplitIntoLines(const std::string& input);

    static bool StartsWith(const std::string& str, const std::string& prefix);

    static bool EndsWith(const std::string& str, const std::string& suffix);
    
    inline static bool IsWhitespace(char c) {
        return (c == ' ' || c == '\t' || c == '\n' || c == '\r');
    }
    
    // This should only be used for ASCII, e.g., filenames, NOT for language data
    static std::string ToLower(const std::string& str);
    
    // This should only be used for ASCII, e.g., filenames, NOT for language data
    static std::string ToUpper(const std::string& str);
};

template <typename T>
std::string StringUtils::Join(const std::string& joiner, const T& items) {
    std::ostringstream ss;
    bool first = true;
    for (auto it = items.begin(); it != items.end(); it++) {
        if (!first) {
            ss << joiner;
        }
        ss << (*it);
        first = false;
    }
    return ss.str();
}

template <typename T>
std::string StringUtils::Join(const std::string& joiner, const T * items, int32_t length) {
    std::ostringstream ss;
    for (int32_t i = 0; i < length; i++) {
        if (i != 0) {
            ss << joiner;
        }
        ss << items[i];
    }
    return ss.str();
}

template <typename T>
std::string StringUtils::ToString(const T& obj) {
    std::ostringstream ss;
    ss << obj;
    return ss.str();
}

} // namespace quicksand
} // namespace marian