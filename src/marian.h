#pragma once

#include <memory>
#include <functional>
#include <vector>
#include <cmath>

#include <boost/pool/pool.hpp>

namespace marian {

typedef float Tensor; // Now do this for cuDNN tensors!
struct Chainable;

boost::pool<> p(sizeof(char));
std::vector<Chainable*> stack;

struct Chainable {
    Chainable() { }
    virtual ~Chainable() { }
    
    virtual void chain() { }
    virtual void init_dependent() { }
    virtual void set_zero_adjoint() { }
    
    static inline void* operator new(size_t nbytes) {
      // thread_local variable
      return p.ordered_malloc(nbytes);
    }
};

class Vimpl : public Chainable {
  public:
    Vimpl(const Tensor& t) : val_{std::move(t)}, adj_{0} {
      stack.push_back(this);
    }
    
    ~Vimpl() {};
    
    virtual void init_dependent() { adj_ = 1; }
    virtual void set_zero_adjoint() { adj_ = 0; }
    
    const Tensor& val() const { return val_; };
    Tensor& grad() { return adj_; };
        
  protected:
    const Tensor val_;
    Tensor adj_; 
};

typedef Vimpl* VimplPtr;

static void set_zero_all_adjoints() {
  for(auto&& v : stack)
    v->set_zero_adjoint();
}

static void grad(Chainable* v) {
  typedef std::vector<Chainable*>::reverse_iterator It;
  v->init_dependent();
  for(It it = stack.rbegin(); it != stack.rend(); ++it) {
    (*it)->chain();
  }
}

class Var {
  public:
    Var() : vimpl_{nullptr} {}
    Var(const Tensor& t) : vimpl_{new Vimpl{t}} {}
    Var(const VimplPtr& vimpl) : vimpl_{vimpl} {}
    
    const Tensor& val() const {
      return vimpl_->val();
    }
    
    Tensor& grad() {
        return vimpl_->grad();
    }
    
    VimplPtr vimpl() const {
        return vimpl_;
    }
    
    void calc_gradients() {
      marian::grad(vimpl_);
    }
    
  private:
    VimplPtr vimpl_; 
};

///////////////////////////////////////////////////

struct OpVimpl : public Vimpl {
  OpVimpl(const Tensor& t, VimplPtr a) : Vimpl(t), a_(a) { }
  
  VimplPtr a_;
};


struct LogVimpl : public OpVimpl {
  LogVimpl(VimplPtr a) : OpVimpl(std::log(a->val()), a) { }
  
  void chain() {
    a_->grad() += adj_ / a_->val();
  }
};

inline Var log(const Var& a) {
  return Var(VimplPtr(new LogVimpl(a.vimpl())));
}

struct ExpVimpl : public OpVimpl {
  ExpVimpl(VimplPtr a) : OpVimpl(std::exp(a->val()), a) { }
  
  void chain() {
    a_->grad() += adj_ * std::exp(a_->val());
  }
};

inline Var exp(const Var& a) {
  return Var(VimplPtr(new ExpVimpl(a.vimpl())));
}

struct NegVimpl : public OpVimpl {
  NegVimpl(VimplPtr a) : OpVimpl(-a->val(), a) { }
  
  void chain() {
    a_->grad() -= adj_;
  }
};

inline Var operator-(const Var& a) {
  return Var(VimplPtr(new NegVimpl(a.vimpl())));
}

// @TODO: take care of large exponents
struct SigmaVimpl : public OpVimpl {
  SigmaVimpl(VimplPtr a) : OpVimpl(1.f / (1.f + std::exp(-a->val())), a) { }
  
  void chain() {
    Tensor l = 1.f / (1.f + std::exp(-a_->val()));
    a_->grad() += adj_ * l * (1 - l);
  }
};

inline Var sigma(const Var& a) {
  return Var(VimplPtr(new SigmaVimpl(a.vimpl())));
}

///////////////////////////////////////////////////


struct OpVimplVV : public Vimpl {
    VimplPtr a_;
    VimplPtr b_;
    
    OpVimplVV(Tensor t, VimplPtr a, VimplPtr b)
    : Vimpl(t), a_(a), b_(b) { }
};

struct PlusVimplVV : public OpVimplVV {
  PlusVimplVV(VimplPtr a, VimplPtr b) : OpVimplVV(a->val() + b->val(), a, b) { }
  
  void chain() {
    a_->grad() += adj_;
    b_->grad() += adj_;
  }
};

inline Var operator+(const Var& a, const Var& b) {
  return Var(VimplPtr(new PlusVimplVV(a.vimpl(), b.vimpl())));
}

struct MinusVimplVV : public OpVimplVV {
  MinusVimplVV(VimplPtr a, VimplPtr b) : OpVimplVV(a->val() - b->val(), a, b) { }
  
  void chain() {
    a_->grad() -= adj_;
    b_->grad() -= adj_;
  }
};

inline Var operator-(const Var& a, const Var& b) {
  return Var(VimplPtr(new MinusVimplVV(a.vimpl(), b.vimpl())));
}

struct MultVimplVV : public OpVimplVV {
  MultVimplVV(VimplPtr a, VimplPtr b) : OpVimplVV(a->val() * b->val(), a, b) { }
  
  void chain() {
    a_->grad() += adj_ * b_->val();
    b_->grad() += adj_ * a_->val();
  }
};

inline Var operator*(const Var& a, const Var& b) {
  return Var(VimplPtr(new MultVimplVV(a.vimpl(), b.vimpl())));
}

struct DivVimplVV : public OpVimplVV {
  DivVimplVV(VimplPtr a, VimplPtr b) : OpVimplVV(a->val() / b->val(), a, b) { }
  
  void chain() {
    a_->grad() += adj_ / b_->val();
    b_->grad() += adj_ * (a_->val() / (b_->val() * b_->val()));
  }
};

inline Var operator/(const Var& a, const Var& b) {
  return Var(VimplPtr(new DivVimplVV(a.vimpl(), b.vimpl())));
}


}