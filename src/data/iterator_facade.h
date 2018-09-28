#pragma once

// simplistic replacement for boost::iterator_facade
template <class Iterator, class Item>
struct IteratorFacade {
  // to create DummyIterator inherit from public IteratorFacade<DummyIterator, ValueType>
  // and implement these three functions
  virtual bool equal(const Iterator& other) const = 0;
  virtual const Item& dereference() const = 0;
  virtual void increment() = 0;

  bool operator==(const Iterator& other) const {
    return equal(other);
  }

  bool operator!=(const Iterator& other) const {
    return !equal(other);
  }

  const Item& operator*() const {
    return dereference();
  }

  // prefix ++
  Iterator& operator++() {
    increment();
    return dynamic_cast<Iterator&>(*this);
  }

  // postfix ++
  Iterator operator++(int) {
    auto ret = dynamic_cast<Iterator&>(*this);
    increment();
    return ret;
  }

  const Item* operator->() const {
    return &dereference();
  }
};
