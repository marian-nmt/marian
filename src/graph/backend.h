#pragma once

namespace marian {

class Backend {
public:
  virtual void setDevice(size_t device) = 0;
};
}
