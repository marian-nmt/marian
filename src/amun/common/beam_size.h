#pragma once

namespace amunmt {

class BeamSize
{
  BeamSize(size_t size, uint val)
  :sentences_(size, val)
  {

  }

protected:
  std::vector<uint> sentences_;

};

}
