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

#include <memory>
#include <iostream>
#include <thread>

#include "keywords.h"
#include "tensors/tensor.h"
#include "tensors/tensor_gpu.h"
#include "chainable.h"

namespace marian {

class ExpressionGraph;
typedef std::shared_ptr<ExpressionGraph> ExpressionGraphPtr;

class Node : public Chainable<Tensor>,
             public keywords::Keywords,
             public std::enable_shared_from_this<Node> {
  public:
    template <typename ...Args>
    Node(ExpressionGraphPtr graph, Args ...args)
     : Keywords(args...),
       graph_(graph),
       shape_(Get(keywords::shape, {1, 1})),
       givenShape_(shape_),
       name_("none"),
       markedForDebug_(false)
    {}

    virtual ~Node() {}

    virtual ExpressionGraphPtr graph() {
      return graph_;
    }

    virtual void debug(const std::string& message) {
      debugMessage_ = message;
      markedForDebug_ = true;
    }
    virtual bool marked_for_debug() { return markedForDebug_; }
    virtual const std::string& debug_message() { return debugMessage_; }

    virtual size_t allocate();

    virtual void init() {};

    virtual void init_dependent();

    virtual void set_zero_adjoint();

    virtual Tensor& val()  {
      return val_;
    };

    virtual Tensor& grad() {
      return adj_;
    };

    virtual const Shape& shape() {
      return shape_;
    }

    void set_name(const std::string& name) {
      name_ = name;
    }

    const std::string &name() const { return name_; }

    virtual const std::string label(const std::string& type) {
      std::stringstream label;
      label << "<" << type;
      if(name_ != "none") {
        label << "<br/>" << "\"" << name_ << "\"";
      }
      label << ">";
      return label.str();
    }

  protected:
    ExpressionGraphPtr graph_;
    Shape shape_;
    const Shape givenShape_;
    std::string name_;

    Tensor val_;
    Tensor adj_;

    bool markedForDebug_;
    std::string debugMessage_;
};

}
