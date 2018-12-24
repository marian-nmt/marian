#ifndef PATHIE_TEMPDIR_HPP
#define PATHIE_TEMPDIR_HPP
#include "path.hpp"

namespace Pathie {

  /**
   * A class for working with temporary entries; this is the
   * superclass of Tempdir and Tempfile that encapsulates the common
   * logic between the two. This class cannot be instanciated
   * directly, instead use Tempdir and Tempfile.
   *
   * This class relies on `rand()` when generating the temporary
   * path name.  Therefore, it is recommended to initialise the
   * random number generator before creating instances of this class
   * by calling the `srand()` function.
   *
   * In a multithreaded environment, this class generates conflicting
   * directory names if the C random number generator is in the same state
   * in two threads and an instance of Tempdir is constructed in these two
   * threads in the very same second. You should not use an instance of
   * this class in multiple threads.
   */
  class TempEntry
  {
  public:
    TempEntry(std::string namepart);
    virtual ~TempEntry();

    virtual void remove() const = 0;
    void keep(bool k = true);

    Path path() const;
    bool is_kept() const;
  protected:
    bool m_keep;
    Path m_path;
  };

  /**
   * Class for working with temporary directories. Creating
   * an instance of this class creates a temporary directory,
   * which is removed again when the object is destroyed.
   * If you want to keep the directory for whatever reason,
   * call TempEntry::keep().
   *
   * Call TempEntry::path() to retrieve the path of the
   * generated directory.
   *
   * See the docs for the TempEntry class for information
   * on how the temporary names are generated.
   */
  class Tempdir: public TempEntry
  {
  public:
    Tempdir(std::string namepart);
    virtual ~Tempdir();
    virtual void remove() const;
  };

  /**
   * Class for working with temporary files. Creating
   * an instance of this class creates a temporary file,
   * which is removed again when the object is destroyed.
   * If you want to keep the file for whatever reason,
   * call TempEntry::keep().
   *
   * Call TempEntry::path() to retrieve the path of the
   * generated directory.
   *
   * See the docs for the TempEntry class for information
   * on how the temporary names are generated.
   */
  class Tempfile: public TempEntry
  {
  public:
    Tempfile(std::string namepart);
    virtual ~Tempfile();
    virtual void remove() const;
  };
}

#endif /* PATHIE_TEMPDIR_HPP */
