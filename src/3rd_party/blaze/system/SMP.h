//=================================================================================================
/*!
//  \file blaze/system/SMP.h
//  \brief System settings for the shared-memory parallelization
//
//  Copyright (C) 2013 Klaus Iglberger - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its contributors
//     may be used to endorse or promote products derived from this software without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================

#ifndef _BLAZE_SYSTEM_SMP_H_
#define _BLAZE_SYSTEM_SMP_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/config/SMP.h>




//=================================================================================================
//
//  OPENMP MODE CONFIGURATION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compilation switch for the OpenMP parallelization.
// \ingroup system
//
// This compilation switch enables/disables the OpenMP parallelization. In case OpenMP is enabled
// during compilation the Blaze library attempts to parallelize all matrix and vector computations.
// Note that the OpenMP-based parallelization has priority over the C++11 and Boost thread-based
// parallelization and will be preferred in case several parallelizations are activated. In case
// no parallelization is not enabled, all computations are performed on a single compute core.
*/
#if BLAZE_USE_SHARED_MEMORY_PARALLELIZATION && defined(_OPENMP)
#define BLAZE_OPENMP_PARALLEL_MODE 1
#else
#define BLAZE_OPENMP_PARALLEL_MODE 0
#endif
//*************************************************************************************************




//=================================================================================================
//
//  C++11 THREAD PARALLEL MODE CONFIGURATION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compilation switch for the C++11 parallelization.
// \ingroup system
//
// This compilation switch enables/disables the parallelization based on C++11 threads. In case
// the \c BLAZE_USE_CPP_THREADS command line argument is specified during compilation the Blaze
// library attempts to parallelize all matrix and vector computations. Note however that the
// OpenMP-based parallelization has priority over the C++11 thread parallelization and will
// be preferred in case both parallelizations are activated. On the other hand, the C++11
// thread parallelization has priority over the Boost thread-based parallelization. In case
// no parallelization is enabled, all computations are performed on a single compute core.
*/
#if BLAZE_USE_SHARED_MEMORY_PARALLELIZATION && defined(BLAZE_USE_CPP_THREADS)
#define BLAZE_CPP_THREADS_PARALLEL_MODE 1
#else
#define BLAZE_CPP_THREADS_PARALLEL_MODE 0
#endif
//*************************************************************************************************




//=================================================================================================
//
//  BOOST THREAD PARALLEL MODE CONFIGURATION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compilation switch for the Boost parallelization.
// \ingroup system
//
// This compilation switch enables/disables the parallelization based on Boost threads. In case
// the \c BLAZE_USE_BOOST_THREADS command line argument is specified during compilation the Blaze
// library attempts to parallelize all matrix and vector computations. Note however that the
// OpenMP-based and the C++11 thread-based parallelizations have priority over the Boost thread
// parallelization and will be preferred in case several parallelizations are activated. In case
// no parallelization is enabled, all computations are performed on a single compute core.
*/
#if BLAZE_USE_SHARED_MEMORY_PARALLELIZATION && defined(BLAZE_USE_BOOST_THREADS)
#define BLAZE_BOOST_THREADS_PARALLEL_MODE 1
#else
#define BLAZE_BOOST_THREADS_PARALLEL_MODE 0
#endif
//*************************************************************************************************

#endif
