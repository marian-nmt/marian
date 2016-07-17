//=================================================================================================
/*!
//  \file blaze/system/Vectorization.h
//  \brief System settings for the SSE mode
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

#ifndef _BLAZE_SYSTEM_VECTORIZATION_H_
#define _BLAZE_SYSTEM_VECTORIZATION_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/config/Vectorization.h>
#include <blaze/util/StaticAssert.h>




//=================================================================================================
//
//  AVX2 ENFORCEMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
#ifdef BLAZE_ENFORCE_AVX2
#  ifndef BLAZE_ENFORCE_AVX
#    define BLAZE_ENFORCE_AVX
#  endif
#  ifndef __AVX2__
#    define __AVX2__
#  endif
#endif
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  AVX ENFORCEMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
#ifdef BLAZE_ENFORCE_AVX
#  ifndef __MMX__
#    define __MMX__
#  endif
#  ifndef __SSE__
#    define __SSE__
#  endif
#  ifndef __SSE2__
#    define __SSE2__
#  endif
#  ifndef __SSE3__
#    define __SSE3__
#  endif
#  ifndef __SSSE3__
#    define __SSSE3__
#  endif
#  ifndef __SSE4_1__
#    define __SSE4_1__
#  endif
#  ifndef __SSE4_2__
#    define __SSE4_2__
#  endif
#  ifndef __AVX__
#    define __AVX__
#  endif
#endif
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SSE/AVX/MIC MODE CONFIGURATION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compilation switch for the SSE mode.
// \ingroup system
//
// This compilation switch enables/disables the SSE mode. In case the SSE mode is enabled
// (i.e. in case SSE functionality is available) the Blaze library attempts to vectorize
// the linear algebra operations by SSE intrinsics. In case the SSE mode is disabled, the
// Blaze library chooses default, non-vectorized functionality for the operations.
*/
#if BLAZE_USE_VECTORIZATION && ( defined(__SSE__) || ( _M_IX86_FP > 0 ) )
#  define BLAZE_SSE_MODE 1
#else
#  define BLAZE_SSE_MODE 0
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the SSE2 mode.
// \ingroup system
//
// This compilation switch enables/disables the SSE2 mode. In case the SSE2 mode is enabled
// (i.e. in case SSE2 functionality is available) the Blaze library attempts to vectorize
// the linear algebra operations by SSE2 intrinsics. In case the SSE2 mode is disabled, the
// Blaze library chooses default, non-vectorized functionality for the operations.
*/
#if BLAZE_USE_VECTORIZATION && ( defined(__SSE2__) || ( _M_IX86_FP > 1 ) )
#  define BLAZE_SSE2_MODE 1
#else
#  define BLAZE_SSE2_MODE 0
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the SSE3 mode.
// \ingroup system
//
// This compilation switch enables/disables the SSE3 mode. In case the SSE3 mode is enabled
// (i.e. in case SSE3 functionality is available) the Blaze library attempts to vectorize
// the linear algebra operations by SSE3 intrinsics. In case the SSE3 mode is disabled, the
// Blaze library chooses default, non-vectorized functionality for the operations.
*/
#if BLAZE_USE_VECTORIZATION && defined(__SSE3__)
#  define BLAZE_SSE3_MODE 1
#else
#  define BLAZE_SSE3_MODE 0
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the SSSE3 mode.
// \ingroup system
//
// This compilation switch enables/disables the SSSE3 mode. In case the SSSE3 mode is enabled
// (i.e. in case SSSE3 functionality is available) the Blaze library attempts to vectorize
// the linear algebra operations by SSSE3 intrinsics. In case the SSSE3 mode is disabled, the
// Blaze library chooses default, non-vectorized functionality for the operations.
*/
#if BLAZE_USE_VECTORIZATION && defined(__SSSE3__)
#  define BLAZE_SSSE3_MODE 1
#else
#  define BLAZE_SSSE3_MODE 0
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the SSE4 mode.
// \ingroup system
//
// This compilation switch enables/disables the SSE4 mode. In case the SSE4 mode is enabled
// (i.e. in case SSE4 functionality is available) the Blaze library attempts to vectorize
// the linear algebra operations by SSE4 intrinsics. In case the SSE4 mode is disabled,
// the Blaze library chooses default, non-vectorized functionality for the operations.
*/
#if BLAZE_USE_VECTORIZATION && ( defined(__SSE4_2__) || defined(__SSE4_1__) )
#  define BLAZE_SSE4_MODE 1
#else
#  define BLAZE_SSE4_MODE 0
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the AVX mode.
// \ingroup system
//
// This compilation switch enables/disables the AVX mode. In case the AVX mode is enabled
// (i.e. in case AVX functionality is available) the Blaze library attempts to vectorize
// the linear algebra operations by AVX intrinsics. In case the AVX mode is disabled,
// the Blaze library chooses default, non-vectorized functionality for the operations.
*/
#if BLAZE_USE_VECTORIZATION && defined(__AVX__)
#  define BLAZE_AVX_MODE 1
#else
#  define BLAZE_AVX_MODE 0
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the AVX2 mode.
// \ingroup system
//
// This compilation switch enables/disables the AVX2 mode. In case the AVX2 mode is enabled
// (i.e. in case AVX2 functionality is available) the Blaze library attempts to vectorize
// the linear algebra operations by AVX2 intrinsics. In case the AVX2 mode is disabled,
// the Blaze library chooses default, non-vectorized functionality for the operations.
*/
#if BLAZE_USE_VECTORIZATION && defined(__AVX2__)
#  define BLAZE_AVX2_MODE 1
#else
#  define BLAZE_AVX2_MODE 0
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the MIC mode.
// \ingroup system
//
// This compilation switch enables/disables the MIC mode. In case the MIC mode is enabled
// (i.e. in case MIC functionality is available) the Blaze library attempts to vectorize
// the linear algebra operations by MIC intrinsics. In case the MIC mode is disabled,
// the Blaze library chooses default, non-vectorized functionality for the operations.
*/
#if BLAZE_USE_VECTORIZATION && defined(__MIC__)
#  define BLAZE_MIC_MODE 1
#else
#  define BLAZE_MIC_MODE 0
#endif
//*************************************************************************************************




//=================================================================================================
//
//  FMA MODE CONFIGURATION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compilation switch for the FMA mode.
// \ingroup system
//
// This compilation switch enables/disables the FMA mode. In case the FMA mode is enabled
// (i.e. in case FMA functionality is available) the Blaze library attempts to vectorize
// the linear algebra operations by FMA intrinsics. In case the FMA mode is disabled,
// the Blaze library chooses default, non-vectorized functionality for the operations.
*/
#if BLAZE_USE_VECTORIZATION && defined(__FMA__)
#  define BLAZE_FMA_MODE 1
#else
#  define BLAZE_FMA_MODE 0
#endif
//*************************************************************************************************




//=================================================================================================
//
//  SVML MODE CONFIGURATION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compilation switch for the SVML mode.
// \ingroup system
//
// This compilation switch enables/disables the SVML mode. In case the SVML mode is enabled
// (i.e. in case an Intel compiler is used) the Blaze library attempts to vectorize several
// linear algebra operations by SVML intrinsics. In case the SVML mode is disabled, the
// Blaze library chooses default, non-vectorized functionality for the operations.
*/
#if BLAZE_USE_VECTORIZATION && ( defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) || defined(__ECC) )
#  define BLAZE_SVML_MODE 1
#else
#  define BLAZE_SVML_MODE 0
#endif
//*************************************************************************************************




//=================================================================================================
//
//  COMPILE TIME CONSTRAINTS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
namespace {

BLAZE_STATIC_ASSERT( !BLAZE_SSE2_MODE  || BLAZE_SSE_MODE   );
BLAZE_STATIC_ASSERT( !BLAZE_SSE3_MODE  || BLAZE_SSE2_MODE  );
BLAZE_STATIC_ASSERT( !BLAZE_SSSE3_MODE || BLAZE_SSE3_MODE  );
BLAZE_STATIC_ASSERT( !BLAZE_SSE4_MODE  || BLAZE_SSSE3_MODE );
BLAZE_STATIC_ASSERT( !BLAZE_AVX_MODE   || BLAZE_SSE4_MODE  );
BLAZE_STATIC_ASSERT( !BLAZE_AVX2_MODE  || BLAZE_AVX_MODE   );

}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SSE/AVX/MIC INCLUDE FILE CONFIGURATION
//
//=================================================================================================

#if BLAZE_MIC_MODE || BLAZE_AVX_MODE || BLAZE_AVX2_MODE
#  include <immintrin.h>
#elif BLAZE_SSE4_MODE
#  include <smmintrin.h>
#elif BLAZE_SSSE3_MODE
#  include <tmmintrin.h>
#elif BLAZE_SSE3_MODE
#  include <pmmintrin.h>
#elif BLAZE_SSE2_MODE
#  include <emmintrin.h>
#elif BLAZE_SSE_MODE
#  include <xmmintrin.h>
#endif

#endif
