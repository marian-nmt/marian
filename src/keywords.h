#pragma once

#include <iostream>
#include <vector>
#include <typeinfo>
#include <typeindex>
#include <unordered_map>
#include <boost/any.hpp>

#include "compile_time_crc32.h"

namespace marian {
namespace keywords {

  template <int key, typename Value>
  class Keyword {
    public:
      typedef Value value_type;
      
      struct pair {
          Keyword<key, Value> first;
          Value second;
      };
      
      Keyword(const std::string& name)
      : name_(name) {}
      
      pair operator=(Value value) {
        return pair{*this, value};
      }
    
      const std::string& operator()() const {
        return name_;
      }
      
    private:
      std::string name_;
  };
  
  struct Keywords {
    Keywords() {}  
      
    template <typename ...Args>
    Keywords(Args ...args) {
      add(args...); 
    }
    
    template <typename Head>
    void add(Head head) {
      map_[std::type_index(typeid(head.first))] = head.second;
    }
    
    template <typename Head, typename ...Tail>
    void add(Head head, Tail ...tail) {
      map_[std::type_index(typeid(head.first))] = head.second;
      add(tail...);
    }
    
    template <typename Value, typename Key>
    Value Get(Key key, Value default_value) {
      auto it = map_.find(std::type_index(typeid(key)));
      if(it != map_.end())
          return boost::any_cast<Value>(map_[std::type_index(typeid(key))]);
      else
          return default_value;
    }
    
    private:
      std::unordered_map<std::type_index, boost::any> map_;
  };
  
  #define KEY(name, value_type) \
  typedef Keyword<COMPILE_TIME_CRC32_STR(#name),value_type> name ## _k; \
  name ## _k name(#name);

  KEY(shape, std::vector<int>)
  KEY(prefix, std::string)
  KEY(axis, size_t);
}

class demo : public keywords::Keywords {
  public:
    template <typename ...Args>
    demo(size_t size, Args ...args)
    : Keywords(args...),
      size_(size),
      prefix_(Get<std::string>(keywords::prefix, std::string("_"))),
      shape_(Get<std::vector<int>>(keywords::shape, std::vector<int>()))
      {}
    
  private:
    size_t size_;
    std::string prefix_;
    std::vector<int> shape_;
};

void demo_main() {
  using namespace keywords;
  
  demo(300, shape={1,3}, prefix="layer1_", axis=0);
}

}