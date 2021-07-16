#include "microsoft/shortlist/utils/ParameterTree.h"

#include <string>

#include "microsoft/shortlist/utils/StringUtils.h"
#include "microsoft/shortlist/utils/Converter.h"

namespace marian {
namespace quicksand {

#include "microsoft/shortlist/logging/LoggerMacros.h"

std::shared_ptr<ParameterTree> ParameterTree::m_empty_tree = std::make_shared<ParameterTree>("params");

ParameterTree::ParameterTree() {
    m_name = "root";
}

ParameterTree::ParameterTree(const std::string& name) {
    m_name = name;
}

ParameterTree::~ParameterTree() {
}

void ParameterTree::Clear() {
    
}

void ParameterTree::ReplaceVariables(
    const std::unordered_map<std::string, std::string>& vars,
    bool error_on_unknown_vars)
{
    ReplaceVariablesInternal(vars, error_on_unknown_vars);
}

void ParameterTree::RegisterInt32(const std::string& name, int32_t * param) {
    RegisterItemInternal(name, PARAM_TYPE_INT32, (void *)param);
}

void ParameterTree::RegisterInt64(const std::string& name, int64_t * param) {
    RegisterItemInternal(name, PARAM_TYPE_INT64, (void *)param);
}

void ParameterTree::RegisterFloat(const std::string& name, float * param) {
    RegisterItemInternal(name, PARAM_TYPE_FLOAT, (void *)param);
}

void ParameterTree::RegisterDouble(const std::string& name, double * param) {
    RegisterItemInternal(name, PARAM_TYPE_DOUBLE, (void *)param);
}

void ParameterTree::RegisterBool(const std::string& name, bool * param) {
    RegisterItemInternal(name, PARAM_TYPE_BOOL, (void *)param);
}

void ParameterTree::RegisterString(const std::string& name, std::string * param) {
    RegisterItemInternal(name, PARAM_TYPE_STRING, (void *)param);
}

std::shared_ptr<ParameterTree> ParameterTree::FromBinaryReader(const void*& current) {
    std::shared_ptr<ParameterTree> root = std::make_shared<ParameterTree>();
    root->ReadBinary(current);
    return root;
}

void ParameterTree::SetRegisteredParams() {
    for (std::size_t i = 0; i < m_registered_params.size(); i++) {
        const RegisteredParam& rp = m_registered_params[i];
        switch (rp.Type()) {
            case PARAM_TYPE_INT32:
                (*(int32_t *)rp.Data()) = GetInt32Req(rp.Name());
                break;
            case PARAM_TYPE_INT64:
                (*(int64_t *)rp.Data()) = GetInt64Req(rp.Name());
                break;
            default:
                LOG_ERROR_AND_THROW("Unknown ParameterType: %d", (int)rp.Type());
        }
    }
}

int32_t ParameterTree::GetInt32Or(const std::string& name, int32_t defaultValue) const {
    const std::string * value = GetParamInternal(name);
    if (value == nullptr) {
        return defaultValue;
    }
    return Converter::ToInt32(*value);
}

int64_t ParameterTree::GetInt64Or(const std::string& name, int64_t defaultValue) const {
    const std::string * value = GetParamInternal(name);
    if (value == nullptr) {
        return defaultValue;
    }
    return Converter::ToInt64(*value);
}

uint64_t ParameterTree::GetUInt64Or(const std::string& name, uint64_t defaultValue) const {
    const std::string * value = GetParamInternal(name);
    if (value == nullptr) {
        return defaultValue;
    }
    return Converter::ToUInt64(*value);
}

double ParameterTree::GetDoubleOr(const std::string& name, double defaultValue) const {
    const std::string * value = GetParamInternal(name);
    if (value == nullptr) {
        return defaultValue;
    }
    return Converter::ToDouble(*value);
}

float ParameterTree::GetFloatOr(const std::string& name, float defaultValue) const {
    const std::string * value = GetParamInternal(name);
    if (value == nullptr) {
        return defaultValue;
    }
    return Converter::ToFloat(*value);
}

std::string ParameterTree::GetStringOr(const std::string& name, const std::string& defaultValue) const {
    const std::string * value = GetParamInternal(name);
    if (value == nullptr) {
        return defaultValue;
    }
    return (*value);
}

bool ParameterTree::GetBoolOr(const std::string& name, bool defaultValue) const {
    const std::string * value = GetParamInternal(name);
    if (value == nullptr) {
        return defaultValue;
    }
    return Converter::ToBool(*value);
}

int32_t ParameterTree::GetInt32Req(const std::string& name) const {
    std::string value = GetStringReq(name);
    return Converter::ToInt32(value);
}

uint64_t ParameterTree::GetUInt64Req(const std::string& name) const {
    std::string value = GetStringReq(name);
    return Converter::ToUInt64(value);
}

int64_t ParameterTree::GetInt64Req(const std::string& name) const {
    std::string value = GetStringReq(name);
    return Converter::ToInt64(value);
}

double ParameterTree::GetDoubleReq(const std::string& name) const {
    std::string value = GetStringReq(name);
    return Converter::ToDouble(value);
}

float ParameterTree::GetFloatReq(const std::string& name) const {
    std::string value = GetStringReq(name);
    return Converter::ToFloat(value);
}

bool ParameterTree::GetBoolReq(const std::string& name) const {
    std::string value = GetStringReq(name);
    return Converter::ToBool(value);
}

std::string ParameterTree::GetStringReq(const std::string& name) const {
    const std::string * value = GetParamInternal(name);
    if (value == nullptr) {
        LOG_ERROR_AND_THROW("Required parameter <%s> not found in ParameterTree:\n%s", name.c_str(), ToString().c_str());
    }
    return (*value);
}

std::vector<std::string> ParameterTree::GetFileListReq(const std::string& name) const {
    std::vector<std::string> output = GetFileListOptional(name);
    if (output.size() == 0) {
        LOG_ERROR_AND_THROW("No files were found for parameter: %s", name.c_str());
    }
    return output;
}

std::vector<std::string> ParameterTree::GetFileListOptional(const std::string& name) const {
    const std::string * value = GetParamInternal(name);
    if (value == nullptr || (*value).size() == 0) {
        return std::vector<std::string>();
    }
    std::vector<std::string> all_files = StringUtils::Split(*value, ";");
    return all_files;
}

std::vector<std::string> ParameterTree::GetStringListReq(const std::string& name, const std::string& sep) const {
    std::string value = GetStringReq(name);
    std::vector<std::string> output = StringUtils::Split(value, sep);
    return output;
}

std::vector<std::string> ParameterTree::GetStringListOptional(const std::string& name, const std::string& sep) const {
    std::string value = GetStringOr(name, "");
    std::vector<std::string> output = StringUtils::Split(value, sep);
    return output;
}

std::shared_ptr<ParameterTree> ParameterTree::GetChildReq(const std::string& name) const {
    for (const auto& child : m_children) {
        if (child->Name() == name) {
            return child;
        }
    }
    LOG_ERROR_AND_THROW("Unable to find child ParameterTree with name '%s'", name.c_str());
    return nullptr; // never happens
}


std::shared_ptr<ParameterTree> ParameterTree::GetChildOrEmpty(const std::string& name) const {
    for (const auto& child : m_children) {
        if (child->Name() == name) {
            return child;
        }
    }
    return std::make_shared<ParameterTree>();
}

// cast current void pointer to T pointer and move forward by num elements 
template <typename T>
const T* get(const void*& current, size_t num = 1) {
  const T* ptr = (const T*)current;
  current = (const T*)current + num;
  return ptr;
}

void ParameterTree::ReadBinary(const void*& current) {
    auto nameLength = *get<int32_t>(current);
    auto nameBytes  =  get<char>(current, nameLength);
    m_name = std::string(nameBytes, nameBytes + nameLength);
    
    auto textLength = *get<int32_t>(current);
    auto textBytes  =  get<char>(current, textLength);
    m_text = std::string(textBytes, textBytes + textLength);

    int32_t num_children = *get<int32_t>(current);
    m_children.resize(num_children);
    for (int32_t i = 0; i < num_children; i++) {
        m_children[i].reset(new ParameterTree());
        m_children[i]->ReadBinary(current);
    }
}

std::vector< std::shared_ptr<ParameterTree> > ParameterTree::GetChildren(const std::string& name) const {
    std::vector< std::shared_ptr<ParameterTree> > children;
    for (std::shared_ptr<ParameterTree> child : m_children) {
        if (child->Name() == name) {
            children.push_back(child);
        }
    }
    return children;
}

void ParameterTree::AddParam(const std::string& name, const std::string& text) {
    std::shared_ptr<ParameterTree> child = std::make_shared<ParameterTree>(name);
    child->SetText(text);
    m_children.push_back(child);
}

void ParameterTree::SetParam(const std::string& name, const std::string& text) {
    for (const auto& child : m_children) {
        if (child->Name() == name) {
            child->SetText(text);
            return;
        }
    }
    std::shared_ptr<ParameterTree> child = std::make_shared<ParameterTree>(name);
    child->SetText(text);
    m_children.push_back(child);
}

void ParameterTree::AddChild(std::shared_ptr<ParameterTree> child) {
    m_children.push_back(child);
}    

bool ParameterTree::HasParam(const std::string& name) const {
    const std::string * value = GetParamInternal(name);
    if (value == nullptr) {
    	return false;
    }
    return true;
}

bool ParameterTree::HasChild(const std::string& name) const {
    for (const auto& child : m_children) {
    	if (child->Name() == name) {
    		return true;
    	}
    }
    return false;
}

std::string ParameterTree::ToString() const {
    std::ostringstream ss;
    ToStringInternal(0, ss);
    return ss.str();
}

const std::string * ParameterTree::GetParamInternal(const std::string& name) const {
    for (const auto& child : m_children) {
        if (child->Name() == name) {
            return &(child->Text());
        }
    }
    return nullptr;
}


void ParameterTree::RegisterItemInternal(const std::string& name, ParameterType type, void * param) {
    if (m_registered_param_names.find(name) != m_registered_param_names.end()) {
        LOG_ERROR_AND_THROW("Unable to register duplicate parameter name: '%s'", name.c_str());
    }
    m_registered_params.push_back(RegisteredParam(name, type, param));
    m_registered_param_names.insert(name);
}

void ParameterTree::ToStringInternal(int32_t depth, std::ostream& ss) const {
    for (int32_t i = 0; i < 2*depth; i++) {
        ss << " ";
    }
    ss << "<" << m_name << ">";
    if (m_children.size() > 0) {
        ss << "\n";
        for (const std::shared_ptr<ParameterTree>& child : m_children) {
            child->ToStringInternal(depth+1, ss);
        }
    	for (int32_t i = 0; i < 2 * depth; i++) {
    		ss << " ";
    	}
    	ss << "</" << m_name << ">\n";
    }
    else {
    	ss << m_text << "</" << m_name << ">\n";
    }
}

std::shared_ptr<ParameterTree> ParameterTree::Clone() const {
    std::shared_ptr<ParameterTree> node = std::make_shared<ParameterTree>(m_name);
    node->m_text = m_text;
    for (auto& child : m_children) {
        node->m_children.push_back(child->Clone());
    }
    return node;
}

void ParameterTree::Merge(const ParameterTree& other) {
    m_name = other.m_name;
    m_text = other.m_text;
    for (auto& other_child : other.m_children) {
        if (HasChild(other_child->Name())) {
            auto my_child = GetChildReq(other_child->Name());
            if (other_child->Text() != "" && my_child->Text() != "") {
                my_child->SetText(other_child->Text()); 
            }
            else {
                my_child->Merge(*other_child);
            }
        }
        else {
            m_children.push_back(other_child->Clone());
        }
    }
}

void ParameterTree::ReplaceVariablesInternal(
    const std::unordered_map<std::string, std::string>& vars,
    bool error_on_unknown_vars)
{
    std::size_t offset = 0;
    std::ostringstream ss;
    while (true) {
        std::size_t s_pos = m_text.find("$$", offset);
        if (s_pos == std::string::npos) {
            break;
        }
        std::size_t e_pos = m_text.find("$$", s_pos + 2);
        if (e_pos == std::string::npos) {
            break;
        }
        
        if (offset != s_pos) {
            ss << m_text.substr(offset, s_pos-offset);
        }
        
        std::string var_name = m_text.substr(s_pos+2, e_pos - (s_pos+2));
        auto it = vars.find(var_name);
        if (it != vars.end()) {
            std::string value = it->second;
            ss << value;
        }
        else {
            if (error_on_unknown_vars) {
                LOG_ERROR_AND_THROW("The variable $$%s$$ was not found", var_name.c_str());
            }
            else {
                ss << "$$" << var_name << "$$";
            }
        }
        offset = e_pos + 2;
    }
    ss << m_text.substr(offset);
    
    m_text = ss.str();
    
    for (auto& child : m_children) {
        child->ReplaceVariablesInternal(vars, error_on_unknown_vars);
    }
}

} // namespace quicksand
} // namespace marian