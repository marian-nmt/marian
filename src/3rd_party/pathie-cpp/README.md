PATHIE.
=======

This is the Pathie project. It aims to provide a C++ library that covers
all needs of pathname manipulation and filename fiddling, without
having to worry about the underlying platform. That is, it is a glue
library that allows you to create platform-independent filename
handling code with special regard to Unicode path names.

Supported systems
-----------------

Currently supported platforms are Linux and Windows, the latter via
MSYS2 GCC. Any other compiler or system might or might not work. Mac
OS should work as well, but I cannot test this due to lack of a Mac. I
gladly accept contributions for any system or compiler.

Pathie's source code itself is written conforming to C++98. On UNIX
systems, it assumes the system supports POSIX.1-2001. On Windows
systems, the minimum supported Windows version is Windows Vista.

Installation
------------

See INSTALL.md.

The library
-----------

The entire world is using UTF-8 as the primary Unicode encoding. The
entire world? No, a little company from Redmond resists the temptation
and instead uses UTF-16LE, causing cross-platform handling of Unicode
paths to be a nightmare.

One of the main problems the author ran into was compiler-dependant
code that was not marked as such. Many sites on the Internet claim
Unicode path handling on Windows is easy, but in fact, it only is if
you define “development for Windows” as “development with MSVC”,
Microsoft’s proprietary C/C++ compiler, which provides nonstandard
interfaces to allow for handling UTF-16LE filenames. The Pathie
library has been developed with a focus on MinGW and crosscompilation
from Linux to Windows and thus does not suffer from this problem.

The Pathie library has been developed to release the programmer from
the burden of handling the different encodings in use for filenames,
and does so by focusing its API on UTF-8 regardless of the platform in
use. Thus, if you use UTF-8 as your preferred encoding inside your
program (take a look at the [UTF8 Everywhere
website](http://www.utf8everywhere.org) for reasons why you should do
that), Pathie will be of the most use for you, since it transparently
converts whatever filesystem encoding is encountered to UTF-8 in its
public interface. Likewise, any pathname you pass to the library is
assumed to be UTF-8 and is transcoded transparently to the filesystem
encoding before invoking the respective OS' filesystem access
methods. Of course, explicit conversion functions are also provided,
in case you do need a string in the native encoding or need to
construct a path from a string in the native encoding.

General Usage
-------------

First thing is to include the main header:

~~~~~~~~~~~~~~~~~~{.cpp}
#include <pathie/path.hpp>
~~~~~~~~~~~~~~~~~~

Now consider the simple task to get all children of a directory, which
have Unicode filenames. Doing that manually will result in you having
to convert between UTF-8 and UTF-16 all the time. With pathie, you can
just do this:

~~~~~~~~~~~~~~~~~~~{.cpp}
std::vector<Pathie::Path> children = your_path.children();
~~~~~~~~~~~~~~~~~~~

Done. Retrieving the parent directory of your directory is pretty easy:

~~~~~~~~~~~~~~~~~~~{.cpp}
Pathie::Path yourpath("foo/bar/baz");
Pathie::Path parent = yourpath.parent();
~~~~~~~~~~~~~~~~~~~

But Pathie is much more than just an abstraction of different filepath
encodings. It is a utility library for pathname manipulation, i.e. it
allows you to do things like finding the parent directory, expanding
relative to absolute paths, decomposing a filename into basename,
dirname, and extension, and so on. See the documentation of the
central Pathie::Path class on what you can do.

~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// Assume current directory is /tmp
Pathie::Path p("foo/bar/../baz");
p.expand(); // => /tmp/foo/baz
~~~~~~~~~~~~~~~~~~~~~~

Or my personal favourite:

~~~~~~~~~~~~~~~~~~~{.cpp}
Pathie::Path p1("/tmp/foo/bar");
Pathie::Path p2("/tmp/bar/foo");
Pathie::Path p3 = p1.relative(p2); // => ../../foo/bar
~~~~~~~~~~~~~~~~~~~

It also provides you with commonly used paths like the user’s
configuration directory or the path to the running executable.

~~~~~~~~~~~~~~~~~~~~{.cpp}
Pathie::Path configdir  = Pathie::Path::config_dir();
Pathie::Path exepath    = Pathie::Path::exe();
~~~~~~~~~~~~~~~~~~~~

Pathie assumes that all string arguments passed are in UTF-8 and
transparently converts to the native filesystem encoding internally.

Still, if you interface directly with the Windows API or other external
libraries, you might want to retrieve the native representation from a
Path or construct a Path from the native representation. Pathie
doesn’t want to be in your way then. The following example constructs
from and converts to the native representation on Windows, which is
UTF-16LE:

~~~~~~~~~~~~~~~~~~~~{.cpp}
// Contruct from native
wchar_t* utf16 = Win32ApiCall();
Path mypath = Path::from_native(utf16); // also accepts std::wstring

// Retrieve native (Note C++’ish std::wstring rather than
// raw wchar_t* on Windows)
std::wstring native_utf16 = mypath.native();
~~~~~~~~~~~~~~~~~~~~

On UNIX, these methods work with normal strings (std::string instead
of std::wstring) in the underlying filesystem encoding. In most cases,
that will be UTF-8, but some legacy systems may still use something
like ISO-8859-1 in which case that will differ.

### Temporary files and directories

There are two classes `Pathie::Tempdir` and `Pathie::Tempfile` that
you can use if you need to work with temporary files or directories,
respectively. Constructing instances of these classes creates a
temporary entry, which is removed (recursively in case of directories)
when the instance is destroyed again. Use TempEntry::path() to get
access to the Path instance pointing to the created entry.

~~~~~~~~~~~~~~~~~~~~{.cpp}
#include <pathie/tempdir.hpp>

//...

{
  srand(time(NULL)); // Needs random number generator
  Pathie::Tempdir tmpdir("foo"); // Pass a fragment to use as part of filename
  std::cout << "Temporary dir is: " << tmpdir.path() << std::endl;
}
// When `tmpdir' is destroyed, the destructor recursively
// deletes the directory that was created.
~~~~~~~~~~~~~~~~~~~~

### Opening a file with a Unicode path name

On Windows with GCC, it is [not possible to open a file with Unicode
pathname](https://stackoverflow.com/questions/821873) via C++'s usual
`std::ifstream` and `std::ofstream` mechanism. There's a nonstandard
extension provided by Microsoft's proprietary compiler that does this,
but GCC does not have this extension. Consequently, code that is
intended to compile on GCC (like Pathie) has to avoid it.

There *is* however a function in the Win32API that allows to open a
file with a Unicode pathname *and* that returns a standard C `FILE*`
handle,
[_wfopen()](http://msdn.microsoft.com/en-us/library/yeby3zcb.aspx). The
method Path::fopen() uses this function on Windows and a regular C
`fopen()` on all other platforms, thus allowing you to just deal with
your Unicode filename via the regular C I/O interface. If you urgently
need C++ I/O streams, read on.

### Stream replacements

Pathie mainly provides you with the means to handle paths, compose,
and decompose them. There is an experimental feature however that
provides replacements for C++ file streams that work with instances of
Pathie::Path instead of strings for opening a file. These replacements
are neither elegant nor portable, because they don't nicely honour the
template concept the STL is based on by directly subclassing the
standard streams in the matter needed most frequently and additionally
relying on vendor-specific details. For GCC, an internal (but at least
documented) interface is used to exchange the file descriptor inside a
stream, and for MSVC, a nonstandard (but documented) constructor is
used. Other compilers are not supported by this feature (which most
notably affects clang, where I have no idea on the interfaces I need
to use for such a trick).

In one word, these replacements are hacky and I consider them
experimental. If that does not strike you as problematic, you can
enable this feature by passing `-DPATHIE_BUILD_STREAM_REPLACEMENTS=ON`
when invoking `cmake` during the build process.

In order to use the replacements, include the respective header
(either `pathie_ifstream` or `pathie_ofstream`) and use the
`Pathie::ifstream` and `Pathie::ofstream` classes just like you would
use `std::ifstream` and `std::ofstream`, with the only difference
being that you construct them from a Pathie::Path instance instead of
a string. See the documentation of Pathie::ofstream for more
information.

~~~~~~~~~~~~~~~~~{.cpp}
#include <pathie/pathie_ofstream>

// ...

Pathie::Path p("Bärenstark.txt");
Pathie::ofstream file(p);
file << "Some content" << std::endl;
file.close()
~~~~~~~~~~~~~~~~~

There's also the inofficial
[boost::nowide](http://cppcms.com/files/nowide/html/), which is
similar to this feature and maybe more reliable. It has [recently been
accepted into
boost](https://lists.boost.org/boost-announce/2017/06/0516.php).

Dependencies and linking
------------------------

Pathie is standalone, that is, it requires no other libraries except
for those provided by your operating system. Note that there’s a
caveat with this on Windows, which does provide the `Shlwapi` library
by default, but MinGW's GCC does not automatically link it in. Be sure
to link to this library explicitely when compiling for MinGW Windows
by appending `-lShlwapi` to the end of your linking command line.

It is recommended to link in pathie as a dynamic library, because
there are some problems with it when linked statically on certain
operating systems (see _Caveats_ below). If you are sure you aren’t
affected by those problems, it is possible to link in pathie
statically.

Caveats
-------

This library assumes that under all UNIX systems out there (I also
consider Mac OSX to be a UNIX system) the file system root always is
`/` and the directory separator also always is `/`. This structure is
mandatory as per POSIX -- in POSIX.1-2008, it’s specified in section
10.1. Systems which do neither follow POSIX directory structure, nor
are Windows, are unsupported.

On POSIX-compliant systems other than Mac OS X, the filesystem
encoding [generally is
unspecified](https://unix.stackexchange.com/questions/2089/what-charset-encoding-is-used-for-filenames-and-paths-on-linux).
Pathnames are merely byte blobs which do not contain NUL bytes, and
components are separated by `/`. It’s up to the applications,
including utilities like a shell or the ls(1) program, to make
something of those byte streams. Therefore, it is perfectly possible
that on one system, user A uses ISO-8859-1 filenames and user B uses
UTF-8 filenames. Even the same user could use differently encoded
filenames. Programs that have to interpret the byte blobs in pathnames
on these systems look at the locale environment variables, namely
`LANG` and `LC_ALL`, see section 7 of POSIX.1-2008. As a consequence,
it may happen you want to create filenames with characters not
supported in the user’s pathname encoding. For example, if you want to
create a file with a hebrew filename and the user’s pathname encoding
is ISO-8859-1, there’s a problem, because ISO-8859-1 has no hebrew
characters in it, but in UTF-8, which is the encoding you are advised
to use and which is what Pathie’s API expects from you, they are
available. There is no sensible solution to this problem that the
Pathie library could dictate; the `iconv()` function used by pathie
just replaces characters that are unavailable in the target encoding
with a system-defined default (probably “?”). Note that on systems
which have a Unicode pathname encoding, especially modern Linuxes with
UTF-8, such a situation can’t ever arise, because the Unicode
encodings (UTF-*) cover all characters you can ever use.

At least on FreeBSD, calling the POSIX `iconv()` function fails with
the cryptic error message “Service unavailable” if a program is linked
statically. I’ve reported [a bug on
this](https://bugs.freebsd.org/bugzilla/show_bug.cgi?id=196567). This
means that you currently can’t link in pathie statically on FreeBSD
and systems which don’t allow statically linked executables to call
`iconv()`.

On Linux systems, it is recommended to set your program’s locale to the
environment’s locale before you call any functions the Pathie library
provides, because this will allow Pathie to use the correct encoding
for filenames. This is relevant where the environment’s encoding is
not UTF-8, e.g. with $LANG set to `de_DE.ISO-8859-1`. You can do this
as follows (the `""` locale always refers to the locale of the
environment):

~~~~~~~~~~~~~~~~~~~~~{.cpp}
#include <locale>
std::locale::global(std::locale(""));
~~~~~~~~~~~~~~~~~~~~~

This is not required on Windows nor on Mac OS X, because these
operating systems always use UTF-16LE (Windows) or UTF-8 (Mac OS X) as
the filesystem encoding, regardless of the user's locale. It however
does not hurt to call this either, it simply makes no difference for
Pathie on these systems. If you urgently need to avoid this call on
Linux, you need to compile pathie with the special build option
PATHIE_ASSUME_UTF8_ON_UNIX, which will force Pathie to assume that
UTF-8 is used as the filesystem encoding under any UNIX-based system.

Links
-----

* Project page: https://www.guelkerdev.de/projects/pathie/
* GitHub mirror: https://github.com/Quintus/pathie-cpp
* Issue tracker: https://github.com/Quintus/pathie-cpp/issues

Contributing
------------

Feel free to submit any contributions you deem useful. Try to make
separate branches for your new features, give a description on what
you changed, etc.

Don’t you duplicate boost::filesystem?
-------------------------------------

Yes and
no. [boost::filesystem](http://www.boost.org/doc/libs/1_56_0/libs/filesystem/doc/index.htm)
provides many methods pathie provides, but has a major problem with
Unicode path handling if you are not willing to do the UTF-8/UTF-16
conversion manually. boost::filesystem always uses UTF-8 to store the
paths on UNIX, and, which is the problem, always uses UTF-16LE to
store the paths on a Windows system. There is no way to override
this, although there is a [hidden documentation
page](http://www.boost.org/doc/libs/1_51_0/libs/locale/doc/html/default_encoding_under_windows.html)
that claims to solve the problem. I have wasted a great amount of time
to persuade boost::filesystem to automatically convert all
`std::string` input it receives into UTF-16LE, but failed to
succeed. Each time I wanted to create a file with a Unicode filename,
the test failed on Windows by producing garbage filenames. Finally I
found out that the neat trick shown in the documentation above indeed
does work -- but only if you use the Microsoft Visual C++ compiler
(MSVC) to compile your code. I don’t, I generally use g++ via the
[MinGW](http://www.mingw.org) toolchain. boost::filesystem fails with
g++ via MinGW with regard to Unicode filenames on Windows as of this
writing (September 2014).

Apart from that, pathie provides some additional methods, especially
with regard to finding out where the user’s paths are. It is modelled
after Ruby’s popular
[Pathname](http://ruby-doc.org/stdlib-2.1.2/libdoc/pathname/rdoc/Pathname.html#method-i-rmtree)
class, but it doesn’t entirely duplicate its interface (which wouldn’t
be idiomatic C++).

Also, pathie is a small library. Adding it to your project shouldn’t
hurt too much, while boost::filesystem is quite a large dependency.

License
-------

Pathie is BSD-licensed; see the file “LICENSE” for the exact license
conditions.
