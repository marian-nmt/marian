#include "microsoft/shortlist/utils/StringUtils.h"

#include <stdio.h>
#include <algorithm>
#include <string> 

namespace marian {
namespace quicksand {

#include "microsoft/shortlist/logging/LoggerMacros.h"

std::string StringUtils::VarArgsToString(const char * format, va_list args) {
    if (format == nullptr) {
        LOG_ERROR_AND_THROW("'format' cannot be null in StringUtils::VarArgsToString");
    }
    
    std::string output;
    // Most of the time the stack buffer (5000 chars) will be sufficient.
    // In cases where this is insufficient, dynamically allocate an appropriately sized buffer
    char buffer[5000];
#ifdef QUICKSAND_WINDOWS_BUILD
    va_list copy;
    va_copy(copy, args);
    int ret = vsnprintf_s(buffer, sizeof(buffer), _TRUNCATE, format, copy);
    va_end(copy);
    if (ret >= 0) {
        output = std::string(buffer, buffer + ret);
    }
    else {
    	va_list copy2;
    	va_copy(copy2, args);
    	int needed_size = _vscprintf(format, copy2);
    	va_end(copy2);
    	 
        if (needed_size < 0) {
            LOG_ERROR_AND_THROW("A call to vsnprintf_s() failed. This should never happen");
        }
        char * dynamic_buffer = new char[needed_size+1];
        int ret2 = vsnprintf_s(dynamic_buffer, needed_size+1, _TRUNCATE, format, args);
        if (ret2 >= 0) {
            output = std::string(dynamic_buffer, dynamic_buffer + ret2);
            delete[] dynamic_buffer;
        }
        else {
            output = "";
            delete[] dynamic_buffer;
            LOG_ERROR_AND_THROW("A call to vsnprintf_s() failed. This should never happen, "
                "since we made a call to _vscprintf() to check the dynamic buffer size. The call to _vscprintf() "
                "returned %d bytes, but apparently that was not enough. This would imply a bug in MSVC's vsnprintf_s implementation.", needed_size);
        }
    }
#else
    va_list copy;
    va_copy(copy, args);
    int needed_size = vsnprintf(buffer, sizeof(buffer), format, copy);
    va_end(copy);
    if (needed_size < (int)sizeof(buffer)) {
        output = std::string(buffer, buffer + needed_size);
    }
    else {
        char * dynamic_buffer = new char[needed_size+1];
    	int ret = vsnprintf(dynamic_buffer, needed_size + 1, format, args);
    	if (ret >= 0 && ret < needed_size + 1) {
            output = std::string(dynamic_buffer);
            delete[] dynamic_buffer;
        }
        else {
            output = "";
            delete[] dynamic_buffer;
            LOG_ERROR_AND_THROW("A call to vsnprintf() failed. Return value: %d.",
                ret);
        }
    }
#endif
    return output;
}

std::vector<std::string> StringUtils::SplitIntoLines(const std::string& input) {
    std::vector<std::string> output;
    if (input.size() == 0) {
        return output;
    }
    std::size_t start = 0;
    for (std::size_t i = 0; i < input.size(); i++) {
        char c = input[i];
        if (c == '\r' || c == '\n') {
            output.push_back(std::string(input.begin() + start, input.begin() + i));
            start = i+1;
        }
        if (c == '\r' && i + 1 < input.size() && input[i+1] == '\n') {
            i++;
            start = i+1;
        }
    }
    // do NOT put an empty length trailing line (but empty length intermediate lines are fine)
    if (input.begin() + start != input.end()) {
        output.push_back(std::string(input.begin() + start, input.end()));
    }
    return output;
}

bool StringUtils::StartsWith(const std::string& str, const std::string& prefix) {
    if (str.length() < prefix.length())
        return false;

    return std::equal(prefix.begin(), prefix.end(), str.begin());
}

bool StringUtils::EndsWith(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length())
        return false;

    return std::equal(suffix.begin(), suffix.end(), str.end() - suffix.length());
}

std::vector<std::string> StringUtils::SplitFileList(const std::string& input) {
    std::vector<std::string> output;
    for (const std::string& s : SplitIntoLines(input)) {
        for (const std::string& t : Split(s, ";")) {
            std::string f = CleanupWhitespace(t);
            output.push_back(f);
        }
    }
    return output;
}

std::vector<std::string> StringUtils::Split(const std::string& input, char splitter) {
    std::vector<std::string> output;
    if (input.size() == 0) {
        return output;
    }
    std::size_t start = 0;
    for (std::size_t i = 0; i < input.size(); i++) {
        if (input[i] == splitter) {
            output.push_back(std::string(input.begin() + start, input.begin() + i));
            start = i+1;
        }
    }
    output.push_back(std::string(input.begin() + start, input.end()));
    return output;
}

std::vector<std::string> StringUtils::Split(const std::string& input, const std::string& splitter) {
    std::vector<std::string> output;
    if (input.size() == 0) {
        return output;
    }
    std::size_t pos = 0;
    while (true) {
        std::size_t next_pos = input.find(splitter, pos);
        if (next_pos == std::string::npos) {
            output.push_back(std::string(input.begin() + pos, input.end()));
            break;
        }
        else {
            output.push_back(std::string(input.begin() + pos, input.begin() + next_pos));
        }
        pos = next_pos + splitter.size();
    }
    return output;
}

std::string StringUtils::Join(const std::string& joiner, const uint8_t * items, int32_t length) {
    std::ostringstream ss;
    for (int32_t i = 0; i < length; i++) {
        if (i != 0) {
            ss << joiner;
        }
        ss << (int32_t)(items[i]);
    }
    return ss.str();
}

std::string StringUtils::Join(const std::string& joiner, const int8_t * items, int32_t length) {
    std::ostringstream ss;
    for (int32_t i = 0; i < length; i++) {
        if (i != 0) {
            ss << joiner;
        }
        ss << (int32_t)(items[i]);
    }
    return ss.str();
}

std::string StringUtils::PrintString(const char * format, ...) {
   va_list args;
   va_start(args, format);
   std::string output = StringUtils::VarArgsToString(format, args);
   va_end(args);
   
   return output;
}

std::vector<std::string> StringUtils::WhitespaceTokenize(const std::string& input) {
    std::vector<std::string> output;
    if (input.size() == 0) {
        return output;
    }
    std::size_t size = input.size();
    std::size_t start = 0;
    std::size_t end = size;
    for (std::size_t i = 0; i < size; i++) {
        char c = input[i];
        if (IsWhitespace(c)) {
            start++;
        }
        else {
            break;
        }
    }
    for (std::size_t i = 0; i < size; i++) {
        char c = input[size-1-i];
        if (IsWhitespace(c)) {
            end--;
        }
        else {
            break;
        }
    }
    if (end <= start) {
        return output;
    }
    bool prev_is_whitespace = false;
    std::size_t token_start = start;
    for (std::size_t i = start; i < end; i++) {
        char c = input[i];
        if (IsWhitespace(c)) {
            if (!prev_is_whitespace) {
                output.push_back(std::string(input.begin() + token_start, input.begin() + i));
            }
            prev_is_whitespace = true;
            token_start = i+1;
        }
        else {
            prev_is_whitespace = false;
        }
    }
    output.push_back(std::string(input.begin() + token_start, input.begin() + end));
    return output;
}

std::string StringUtils::CleanupWhitespace(const std::string& input) {
    if (input.size() == 0) {
        return std::string("");
    }
    std::size_t size = input.size();
    std::size_t start = 0;
    std::size_t end = size;
    for (std::size_t i = 0; i < size; i++) {
        char c = input[i];
        if (IsWhitespace(c)) {
            start++;
        }
        else {
            break;
        }
    }
    for (std::size_t i = 0; i < size; i++) {
        char c = input[size-1-i];
        if (IsWhitespace(c)) {
            end--;
        }
        else {
            break;
        }
    }
    if (end <= start) {
        return std::string("");
    }
    std::ostringstream ss;
    bool prev_is_whitespace = false;
    for (std::size_t i = start; i < end; i++) {
        char c = input[i];
        if (IsWhitespace(c)) {
            if (!prev_is_whitespace) {
                ss << ' ';
            }
            prev_is_whitespace = true;
        }
        else {
            ss << c;
            prev_is_whitespace = false;
        }
    }
    return ss.str();
}

std::string StringUtils::XmlEscape(const std::string& str) {
    std::ostringstream ss;
    for (std::size_t i = 0; i < str.size(); i++) {
        char c = str[i];
        if (c == '&') {
            ss << "&amp;";
        }
        else if (c == '"') {
            ss << "&quot;";
        }
        else if (c == '\'') {
            ss << "&apos;";
        }
        else if (c == '<') {
            ss << "&lt;";
        }
        else if (c == '>') {
            ss << "&gt;";
        }
        else {
            ss << c;
        }
    }
    return ss.str();
}

std::string StringUtils::ToString(const std::string& str) {
    return str;
}

std::string StringUtils::ToString(bool obj) {
    return (obj)?"true":"false";
}

std::string StringUtils::ToUpper(const std::string& str) {
    std::vector<char> output;
    output.reserve(str.size());
    for (char c : str) {
        output.push_back((char)toupper((int)c));
    }
    return std::string(output.begin(), output.end());
}

std::string StringUtils::ToLower(const std::string& str) {
    std::ostringstream ss;
    for (char c : str) {
        ss << c;
    }
    return ss.str();
}

} // namespace quicksand
} // namespace marian