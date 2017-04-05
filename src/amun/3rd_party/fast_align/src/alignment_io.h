#ifndef ALIGNMENT_IO_H_
#define ALIGNMENT_IO_H_

#include <string>
#include <iostream>
#include <memory>
#include "array2d.h"

struct AlignmentIO {
  enum AlignmentType { kNONE = 0, kTRANSLATION = 1, kTRANSLITERATION = 2 };

  static std::shared_ptr<Array2D<bool> > ReadPharaohAlignmentGrid(const std::string& al);
  static void SerializePharaohFormat(const Array2D<bool>& alignment, std::ostream* out);
  static void SerializeTypedAlignment(const Array2D<AlignmentType>& alignment, std::ostream* out);
};

inline std::ostream& operator<<(std::ostream& os, const Array2D<AlignmentIO::AlignmentType>& m) {
  os << ' ';
  for (unsigned j=0; j<m.height(); ++j)
    os << (j%10);
  os << "\n";
  for (unsigned i=0; i<m.width(); ++i) {
    os << (i%10);
    for (unsigned j=0; j<m.height(); ++j) {
      switch (m(i,j)) {
        case AlignmentIO::kNONE:            os << '.'; break;
        case AlignmentIO::kTRANSLATION:     os << '*'; break;
        case AlignmentIO::kTRANSLITERATION: os << '#'; break;
        default:                            os << '?'; break;
      }
    }
    os << (i%10) << "\n";
  }
  os << ' ';
  for (unsigned j=0; j<m.height(); ++j)
    os << (j%10);
  os << "\n";
  return os;
}


#endif
