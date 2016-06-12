#pragma once

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cstddef> 

typedef size_t Word;
typedef std::vector<Word> Words;

const Word EOS = 0;
const Word UNK = 1;

template<class T>
using DeviceVector = std::vector<T>;

template<class T>
using HostVector = std::vector<T>;
