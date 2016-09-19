#pragma once

// This file is part of the Marian toolkit.
// Marian is copyright (c) 2016 Marcin Junczys-Dowmunt.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <typeinfo>
#include <typeindex>
#include <unordered_map>
#include <boost/any.hpp>

#include "compile_time_crc32.h"

namespace marian {
namespace keywords {

  template <unsigned key, typename Value>
  class Keyword {
    public:
      typedef Value value_type;
      
      Keyword(const std::string& name, Value value)
      : name_(name), value_(value) {}
      
      Keyword(const std::string& name)
      : name_(name), value_() {}
      
      Keyword<key, Value> operator=(Value value) const {
        return Keyword<key, Value>(name_, value);
      }
    
      const Value& operator()() const {
        return value_;
      }
      
    private:
      const std::string name_;
      const Value value_;
  };
  
  struct Keywords {
    Keywords() {}  
      
    template <typename ...Args>
    Keywords(Args ...args) {
      add(args...); 
    }
    
    template <typename Head>
    void add(Head head) {
      map_[std::type_index(typeid(head))] = head();
    }
    
    template <typename Head, typename ...Tail>
    void add(Head head, Tail ...tail) {
      map_[std::type_index(typeid(head))] = head();
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
    
    template <typename Key>
    bool Has(Key key) {
      auto it = map_.find(std::type_index(typeid(key)));
      return it != map_.end();
    }
    
    private:
      std::unordered_map<std::type_index, boost::any> map_;
  };
  
  #include <type_traits>

//template <typename...>
//struct is_one_of {
//    static constexpr bool value = false;
//};
//
//template <typename F, typename S, typename... T>
//struct is_one_of<F, S, T...> {
//    static constexpr bool value =
//        std::is_same<F, S>::value || is_one_of<F, T...>::value;
//};
//
//template <class T, class Tuple>
//struct Index;
//
//template <class T, class... Types>
//struct Index<T, std::tuple<T, Types...>> {
//    static constexpr std::size_t value = 0;
//};
//
//template <class T, class U, class... Types>
//struct Index<T, std::tuple<U, Types...>> {
//    static constexpr std::size_t value = 1 + Index<T, std::tuple<Types...>>::value;
//};
//
//struct True {};
//struct False {};
//
//template <typename Match, typename ...Args>
//typename Match::value_type opt(True foo, Args... args) {
//    std::tuple<const Args...> t(args...);
//    return std::get<Index<Match, std::tuple<const Args...>>::value>(t)();    
//}
//
//template <typename Match, typename ...Args>
//typename Match::value_type opt(False foo, Args... args) {
//    return typename Match::value_type();
//}
//
//template <typename Match, typename ...Args>
//typename Match::value_type Get(Args ...args) {
//    constexpr bool match = is_one_of<Match, const Args...>::value;
//    typename std::conditional<match, True, False>::type condition;
//    return opt<Match>(condition, args...);
//}

  
  #define KEY(name, value_type) \
  typedef const Keyword<COMPILE_TIME_CRC32_STR(#name),value_type> name ## _k; \
  name ## _k name(#name);
}

}
