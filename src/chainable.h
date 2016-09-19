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

#include <vector>
#include <memory>

#include "exception.h"

namespace marian {

template <class DataType>
struct Chainable {
    Chainable() { }
    virtual ~Chainable() { }
    virtual void forward() { }
    virtual void backward() { }
    virtual void backward_numeric() { }

    virtual void check() { }
    virtual void init_dependent() { }
    virtual void set_zero_adjoint() { }

    virtual void allocate(size_t) = 0;
    virtual std::string graphviz() = 0;
    virtual const std::string &name() const = 0;
    
    virtual const Shape& shape() = 0;
    virtual DataType &val() = 0;
    virtual DataType grad() = 0;
    virtual void setVal(DataType t) {
      UTIL_THROW2("Tensors can only be assigned to input nodes"); 
    };
};

// XXX Marcin, is ChainableStack the most appropriate name?
//     AFAIK, this is never used as a FILO data structure.
//     If so, perhaps "Tape" or "ChainLinks" or "ChainableList" might be more apropos?
//
//     Naumann (2012) uses "tape" to refer to this data structure.
//     -- The Art of Differentiating Computer Programs: An Introduction to Algorithmic Differentiation, Naumann (2012)
typedef std::vector<Chainable<Tensor>*> ChainableStack;
typedef std::shared_ptr<ChainableStack> ChainableStackPtr;    
typedef std::shared_ptr<Chainable<Tensor>> ChainPtr;


}
