#pragma once


namespace marian {
namespace data {

class Batch {
  public:
    virtual size_t size() const = 0;
    virtual size_t words() const {
      return 0;
    };
};

}
}
