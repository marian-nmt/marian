# Introduction #

This is a simple implementation of the CHD perfect hash algorithm. CHD can
generate perfect hash functions for very large key sets--on the order of
millions of keys--in a very short time. On my circa 2012 desktop and using
the default parameters (hash load factor of 80% and average displacement map
bucket load of 4.0 keys) this implementation can generate a hash function
for 1,000 keys in less than 1/100th of a second, and 1,000,000 keys in less
than a second.

For more information about the algorithm, see
http://cmph.sourceforge.net/chd.html.

# Dependencies #

* No runtime dependencies.
* Requires a modern C++ compiler to build.
* The included build requires GNU Make.

# Building #

## Make Macros ##

The typical GNU macros can be used control the build.

### Compilation ###

Note that the modules for Lua 5.1, 5.2, and 5.3 can be built simultaneously.

* CXX: C++ compiler path.
* CXXFLAGS: C++ compiler flags.
* CPPFLAGS: C preprocessor flags. Necessary if Lua API cannot be discovered
  automatically. You can specify multiple include paths if building more than
  one Lua module.
* LDFLAGS: Linker flags. Not normally needed.
* SOFLAGS: Flags needed to build dynamic library.
* LOFLAGS: Flags needed to build loadable module. Normally should be the
  same as SOFLAGS, except on OS X.
* LIBS: Library dependencies. Normally empty, but see the section Avoiding
  C++ Dependencies.

#### Avoiding C++ Dependencies

Defining the preprocessor macro PHF_NO_LIBCXX to 1 will prevent usage of C++
interfaces such as std::string that would require a dependency on libc++ or
libstdc++. This allows using platform-dependent flags in CXXFLAGS, LDFLAGS,
and SOFLAGS to prevent a dependency on the system C++ library.

For example, on OS X you can do:
```sh
$ make CPPFLAGS="-DPHF_NO_LIBCXX" \
CXXFLAGS="-std=c++11 -fno-rtti -fno-exceptions -O3 -march=native" \
LDFLAGS="-nostdlib" \
LIBS="-lSystem"
```

### Installation ####
* prefix
* includedir
* libdir
* luacpath: Lua C module install path. Can be used for one-shot installation
  of a particular Lua version module.
* lua51cpath: Lua 5.1 C module install path.
* lua52cpath: Same as above, for 5.2.
* lua53cpath: Same as above, for 5.3.

## Make Targets ##

* phf: Builds command-line utility (development)
* libphf.so: Builds dynamic library for non-OS X
* libphf.dylib: Builds dynamic library for OS X
* lua5.1: Builds Lua 5.1 module at 5.1/phf.so. Lua 5.1 headers should be
  specified using CPPFLAGS if not in normal locations.
* lua5.2: Same as above, for Lua 5.2.
* lua5.3: Same as above, for Lua 5.3.

# Usage #

## Lua ##

## API ###

### phf.new(keys[, lambda][, alpha][, seed][, nodiv]) ###

* keys: array of keys in order from 1..#keys. They should be all
  numbers or all strings.

* lambda: number of keys per bucket when generating the g() function mapping.

* alpha: output hash space loading factor as percentage from
  1..100. 100% generates a *minimal* perfect hash function. But note that
  the implementation does *not* implement the necessary optimizations to
  ensure timely generation of minimal perfect hash functions. Normally you
  want a loading factor of 80% to 90% for large key sets.

* seed: random integer seed.

* nodiv: if true rounds r and m to powers of 2, and performs modular
  reduction using bitwise AND. Otherwise, r and m are rounded up to the
  nearest primes and modulo division used when indexing tables. Note that
  the rounding occurs after calculation of the intermediate and output hash
  table loading.

  This is more important when building small hash tables with the C
  interface. The optimization is substantial when the compiler can inline
  the code, but isn't substantial from Lua.

Returns a callable object.

### phf:hash(key)

* Returns an integer hash in the range 1..phf:m(). The returned integer will
  be unique for all keys in the original set. Otherwise the result is
  unspecified.

### Example ###

```Lua
local phf = require"phf"

local lambda = 4 -- how many keys per intermediate bucket
local alpha = 80 -- output hash space loading in percentage.

local keys = { "apple", "banana", "cherry", "date", "eggplant", "fig",
               "guava", "honeydew", "jackfruit", "kiwi", "lemon", "mango" }

local F = phf.new(keys, lambda, alpha)

for i=1,#keys do
	print(keys[i], F(keys[i]))
end

```

## C++ ##

## API ##

### PHF::uniq<T>(T k[], size_t n); ###

Similar to the shell command `sort | uniq`. Sorts, deduplicates, and shifts
down the keys in the array k. Returns the number of unique keys, which will
have been moved to the beginning of the array. If necessary do this before
calling PHF::init, as PHF::init does not tolerate duplicate keys.

### int PHF::init<T, nodiv>(struct phf *f, const T k[], size_t n, size_t l, size_t a, phf_seed_t s);

Generate a perfect hash function for the n keys in array k and store the
results in f. Returns a system error number on failure, or 0 on success. f
is unmodified on failure.

### void PHF::destroy(struct phf *);

Deallocates internal tables, but not the struct object itself.

### void PHF::compact<T, nodiv>(struct phf *);

By default the displacement map is an array of uint32_t integers. This
function will select the smallest type necessary to hold the largest
displacement value and update the internal state accordingly. For a loading
factor of 80% (0.8) in the output hash space, and displacement map loading
factor of 4 (400%), the smallest primitive type will often be uint8_t.

### phf_hash_t PHF::hash<T>(struct phf *f, T k);

Returns an integer hash value, h, where 0 <= h < f->m. h will be unique for
each unique key provided when generating the function. f->m will be larger
than the number of unique keys and is based on the specified loading factor
(alpha), rounded up to the nearest prime or nearest power of 2, depending on
the mode of modular reduction selected. For example, for a loading factor of
80% m will be 127: 100 is 80% of 125, and 127 is the closest prime greater
than or equal to 125. With the nodiv option, m would be 128: 100 is 80% of
125, and 128 is the closest power of 2 greater than or equal to 125.

## C ##

The C API is nearly identical to the C++ API, except the prefix is phf_
instead of PHF::. phf_uniq, phf_init, and phf_hash are macros which utilize
C11's _Generic or GCC's __builtin_types_compatible_p interfaces to overload
the interfaces by key type. The explicit suffixes _uint32, _uint64, and
_string may be used directly.

