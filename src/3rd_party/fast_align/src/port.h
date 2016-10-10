// Copyright 2013 by Tetsuo Kiso
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FAST_ALIGN_PORT_H_
#define FAST_ALIGN_PORT_H_

// As of OS X 10.9, it looks like C++ TR1 headers are removed from the
// search paths. Instead, we can include C++11 headers.
#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#endif

#if defined(__APPLE__) && defined(MAC_OS_X_VERSION_10_9) && \
  MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9
#include <unordered_map>
#include <functional>
#else // Assuming older OS X, Linux or similar platforms
#include <unordered_map>
#include <functional>
//#include <tr1/unordered_map>
//#include <tr1/functional>
//namespace std {
//using tr1::unordered_map;
//using tr1::hash;
//} // namespace std
#endif

#endif // FAST_ALIGN_PORT_H_
