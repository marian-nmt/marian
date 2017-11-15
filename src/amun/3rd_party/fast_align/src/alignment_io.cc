#include "src/alignment_io.h"

using namespace std;

static bool is_digit(char x) { return x >= '0' && x <= '9'; }

std::shared_ptr<Array2D<bool> > AlignmentIO::ReadPharaohAlignmentGrid(const string& al) {
  int max_x = 0;
  int max_y = 0;
  unsigned i = 0;
  size_t pos = al.rfind(" ||| ");
  if (pos != string::npos) { i = pos + 5; }
  while (i < al.size()) {
    if (al[i] == '\n' || al[i] == '\r') break;
    int x = 0;
    while(i < al.size() && is_digit(al[i])) {
      x *= 10;
      x += al[i] - '0';
      ++i;
    }
    if (x > max_x) max_x = x;
    assert(i < al.size());
    if(al[i] != '-') {
      cerr << "BAD ALIGNMENT: " << al << endl;
      abort();
    }
    ++i;
    int y = 0;
    while(i < al.size() && is_digit(al[i])) {
      y *= 10;
      y += al[i] - '0';
      ++i;
    }
    if (y > max_y) max_y = y;
    while(i < al.size() && al[i] == ' ') { ++i; }
  }

  std::shared_ptr<Array2D<bool> > grid(new Array2D<bool>(max_x + 1, max_y + 1));
  i = 0;
  if (pos != string::npos) { i = pos + 5; }
  while (i < al.size()) {
    if (al[i] == '\n' || al[i] == '\r') break;
    int x = 0;
    while(i < al.size() && is_digit(al[i])) {
      x *= 10;
      x += al[i] - '0';
      ++i;
    }
    assert(i < al.size());
    assert(al[i] == '-');
    ++i;
    int y = 0;
    while(i < al.size() && is_digit(al[i])) {
      y *= 10;
      y += al[i] - '0';
      ++i;
    }
    (*grid)(x, y) = true;
    while(i < al.size() && al[i] == ' ') { ++i; }
  }
  // cerr << *grid << endl;
  return grid;
}

void AlignmentIO::SerializePharaohFormat(const Array2D<bool>& alignment, ostream* o) {
  ostream& out = *o;
  bool need_space = false;
  for (unsigned i = 0; i < alignment.width(); ++i)
    for (unsigned j = 0; j < alignment.height(); ++j)
      if (alignment(i,j)) {
        if (need_space) out << ' '; else need_space = true;
        out << i << '-' << j;
      }
  out << endl;
}

void AlignmentIO::SerializeTypedAlignment(const Array2D<AlignmentType>& alignment, ostream* o) {
  ostream& out = *o;
  bool need_space = false;
  for (unsigned i = 0; i < alignment.width(); ++i)
    for (unsigned j = 0; j < alignment.height(); ++j) {
      const AlignmentType& aij = alignment(i,j);
      if (aij != kNONE) {
        if (need_space) out << ' '; else need_space = true;
        if (aij == kTRANSLATION) {}
        else if (aij == kTRANSLITERATION) {
          out << 'T' << ':';
        } else {
          cerr << "\nUnexpected alignment point type: " << static_cast<int>(aij) << endl;
          abort();
        }
        out << i << '-' << j;
      }
    }
  out << endl;
}

