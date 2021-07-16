#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>

#include "microsoft/shortlist/utils/StringUtils.h"

namespace marian {
namespace quicksand {

class ParameterTree {
private:
    enum ParameterType {
        PARAM_TYPE_INT32,
        PARAM_TYPE_INT64,
        PARAM_TYPE_UINT64,
        PARAM_TYPE_FLOAT,
        PARAM_TYPE_DOUBLE,
        PARAM_TYPE_BOOL,
        PARAM_TYPE_STRING
    };
    
    class RegisteredParam {
    private:
        std::string m_name;
        ParameterType m_type;
        void * m_data;
        
    public:
        RegisteredParam() {}
        
        RegisteredParam(const std::string& name,
            ParameterType type,
            void * data)
        {
            m_name = name;
            m_type = type;
            m_data = data;
        }
        
        const std::string& Name() const {return m_name;}
        const ParameterType& Type() const {return m_type;}
        void * Data() const {return m_data;}
    };
    
    static std::shared_ptr<ParameterTree> m_empty_tree;
    
    std::string m_name;
    
    std::string m_text;

    std::vector< std::shared_ptr<ParameterTree> > m_children;
    
    std::unordered_set<std::string> m_registered_param_names;
    
    std::vector<RegisteredParam> m_registered_params;
   
public:
    ParameterTree();
    
    ParameterTree(const std::string& name);
    
    ~ParameterTree();
        
    inline const std::string& Text() const { return m_text; }
    inline void SetText(const std::string& text) { m_text = text; }

    inline const std::string& Name() const { return m_name; }
    inline void SetName(const std::string& name) { m_name = name; }
    
    void Clear();
    
    void ReplaceVariables(
        const std::unordered_map<std::string, std::string>& vars,
        bool error_on_unknown_vars = true);
    
    void RegisterInt32(const std::string& name, int32_t * param);

    void RegisterInt64(const std::string& name, int64_t * param);
    
    void RegisterFloat(const std::string& name, float * param);
    
    void RegisterDouble(const std::string& name, double * param);
    
    void RegisterBool(const std::string& name, bool * param);
    
    void RegisterString(const std::string& name, std::string * param);

    static std::shared_ptr<ParameterTree> FromBinaryReader(const void*& current);

    void SetRegisteredParams();
    
    int32_t GetInt32Req(const std::string& name) const;
    
    int64_t GetInt64Req(const std::string& name) const;
    
    uint64_t GetUInt64Req(const std::string& name) const;
    
    double GetDoubleReq(const std::string& name) const;
    
    float GetFloatReq(const std::string& name) const;
        
    std::string GetStringReq(const std::string& name) const;
    
    bool GetBoolReq(const std::string& name) const;
    
    int32_t GetInt32Or(const std::string& name, int32_t defaultValue) const;
    
    int64_t GetInt64Or(const std::string& name, int64_t defaultValue) const;
    
    uint64_t GetUInt64Or(const std::string& name, uint64_t defaultValue) const;

    std::string GetStringOr(const std::string& name, const std::string& defaultValue) const;
        
    double GetDoubleOr(const std::string& name, double defaultValue) const;
    
    float GetFloatOr(const std::string& name, float defaultValue) const;

    bool GetBoolOr(const std::string& name, bool defaultValue) const;
    
    std::vector<std::string> GetFileListReq(const std::string& name) const;
    
    std::vector<std::string> GetFileListOptional(const std::string& name) const;

    std::vector<std::string> GetStringListReq(const std::string& name, const std::string& sep = " ") const;

    std::vector<std::string> GetStringListOptional(const std::string& name, const std::string& sep = " ") const;
    
    std::shared_ptr<ParameterTree> GetChildReq(const std::string& name) const;
    
    std::shared_ptr<ParameterTree> GetChildOrEmpty(const std::string& name) const;
    
    std::vector< std::shared_ptr<ParameterTree> > GetChildren(const std::string& name) const;
    
    inline const std::vector< std::shared_ptr<ParameterTree> >& GetChildren() const { return m_children; }
    
    void ReadBinary(const void*& current);
    
    void AddParam(const std::string& name, const std::string& text);

    template <typename T>
    void AddParam(const std::string& name, const T& obj);
        
    void SetParam(const std::string& name, const std::string& text);

    template <typename T>
    void SetParam(const std::string& name, const T& obj);
    
    void AddChild(std::shared_ptr<ParameterTree> child);

    std::string ToString() const;
    
    bool HasChild(const std::string& name) const;

    bool HasParam(const std::string& name) const;

    std::shared_ptr<ParameterTree> Clone() const;
    
    void Merge(const ParameterTree& other);
    
private:
    void ReplaceVariablesInternal(
        const std::unordered_map<std::string, std::string>& vars,
        bool error_on_unknown_vars);

    void RegisterItemInternal(const std::string& name, ParameterType type, void * param);
    
    const std::string * GetParamInternal(const std::string& name) const;
       
    void ToStringInternal(int32_t depth, std::ostream& ss) const;
};

template <typename T>
void ParameterTree::AddParam(const std::string& name, const T& obj) {
    AddParam(name, StringUtils::ToString(obj));
}

template <typename T>
void ParameterTree::SetParam(const std::string& name, const T& obj) {
    SetParam(name, StringUtils::ToString(obj));
}

} // namespace quicksand
} // namespace marian