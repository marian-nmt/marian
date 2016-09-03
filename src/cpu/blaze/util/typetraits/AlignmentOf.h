//=================================================================================================
/*!
//  \file blaze/util/typetraits/AlignmentOf.h
//  \brief Header file for the AlignmentOf type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_ALIGNMENTOF_H_
#define _BLAZE_UTIL_TYPETRAITS_ALIGNMENTOF_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <type_traits>
#include <blaze/system/Vectorization.h>
#include <blaze/util/Complex.h>
#include <blaze/util/typetraits/IsVectorizable.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the AlignmentOf type trait.
// \ingroup type_traits
*/
template< typename T >
struct AlignmentOfHelper
{
 private:
   //**********************************************************************************************
   static constexpr size_t defaultAlignment = std::alignment_of<T>::value;
   //**********************************************************************************************

 public:
   //**********************************************************************************************
#if BLAZE_MIC_MODE
   static constexpr size_t value = ( IsVectorizable<T>::value )?( 64UL ):( defaultAlignment );
#elif BLAZE_AVX2_MODE
   static constexpr size_t value = ( IsVectorizable<T>::value )?( 32UL ):( defaultAlignment );
#elif BLAZE_SSE2_MODE
   static constexpr size_t value = ( IsVectorizable<T>::value )?( 16UL ):( defaultAlignment );
#else
   static constexpr size_t value = defaultAlignment;
#endif
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of \c AlignmentOfHelper for 'float'.
// \ingroup type_traits
*/
template<>
struct AlignmentOfHelper<float>
{
 public:
   //**********************************************************************************************
#if BLAZE_MIC_MODE
   static constexpr size_t value = 64UL;
#elif BLAZE_AVX_MODE
   static constexpr size_t value = 32UL;
#elif BLAZE_SSE_MODE
   static constexpr size_t value = 16UL;
#else
   static constexpr size_t value = std::alignment_of<float>::value;
#endif
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of \c AlignmentOfHelper for 'double'.
// \ingroup type_traits
*/
template<>
struct AlignmentOfHelper<double>
{
 public:
   //**********************************************************************************************
#if BLAZE_MIC_MODE
   static constexpr size_t value = 64UL;
#elif BLAZE_AVX_MODE
   static constexpr size_t value = 32UL;
#elif BLAZE_SSE_MODE
   static constexpr size_t value = 16UL;
#else
   static constexpr size_t value = std::alignment_of<double>::value;
#endif
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of \c AlignmentOfHelper for 'complex<float>'.
// \ingroup type_traits
*/
template<>
struct AlignmentOfHelper< complex<float> >
{
 public:
   //**********************************************************************************************
#if BLAZE_MIC_MODE
   static constexpr size_t value = 64UL;
#elif BLAZE_AVX_MODE
   static constexpr size_t value = 32UL;
#elif BLAZE_SSE_MODE
   static constexpr size_t value = 16UL;
#else
   static constexpr size_t value = std::alignment_of< complex<float> >::value;
#endif
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of \c AlignmentOfHelper for 'complex<double>'.
// \ingroup type_traits
*/
template<>
struct AlignmentOfHelper< complex<double> >
{
 public:
   //**********************************************************************************************
#if BLAZE_MIC_MODE
   static constexpr size_t value = 64UL;
#elif BLAZE_AVX_MODE
   static constexpr size_t value = 32UL;
#elif BLAZE_SSE_MODE
   static constexpr size_t value = 16UL;
#else
   static constexpr size_t value = std::alignment_of< complex<double> >::value;
#endif
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Evaluation of the required alignment of the given data type.
// \ingroup type_traits
//
// The AlignmentOf type trait template evaluates the required alignment for the given data type.
// For instance, for fundamental data types that can be vectorized via SSE or AVX instructions,
// the proper alignment is 16 or 32 bytes, respectively. For all other data types, a multiple
// of the alignment chosen by the compiler is returned. The evaluated alignment can be queried
// via the nested \a value member.

   \code
   AlignmentOf<unsigned int>::value  // Evaluates to 32 if AVX2 is available, to 16 if only
                                     // SSE2 is available, and a multiple of the alignment
                                     // chosen by the compiler otherwise.
   AlignmentOf<double>::value        // Evaluates to 32 if AVX is available, to 16 if only
                                     // SSE is available, and a multiple of the alignment
                                     // chosen by the compiler otherwise.
   \endcode
*/
template< typename T >
struct AlignmentOf : IntegralConstant<size_t,AlignmentOfHelper<T>::value>
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Partial specialization of \c AlignmentOf for 'const' data types.
// \ingroup type_traits
*/
template< typename T >
struct AlignmentOf< const T > : IntegralConstant<size_t,AlignmentOfHelper<T>::value>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Partial specialization of \c AlignmentOf for 'volatile' data types.
// \ingroup type_traits
*/
template< typename T >
struct AlignmentOf< volatile T > : IntegralConstant<size_t,AlignmentOfHelper<T>::value>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Partial specialization of \c AlignmentOf for 'const volatile' data types.
// \ingroup type_traits
*/
template< typename T >
struct AlignmentOf< const volatile T > : IntegralConstant<size_t,AlignmentOfHelper<T>::value>
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
