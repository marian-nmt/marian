#pragma once

#include <cstdlib>
#include <cstdint>
#include <vector>

typedef size_t Word;
typedef std::vector<Word> Words;

const Word EOS = 0;
const Word UNK = 1;

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>

template<class T>
using DeviceVector = thrust::device_vector<T>;

template<class T>
using HostVector = thrust::host_vector<T>;

namespace algo = thrust;
namespace iteralgo = thrust;