//=================================================================================================
/*!
//  \file blaze/util/typetraits/Rank.h
//  \brief Header file for the Rank type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_RANK_H_
#define _BLAZE_UTIL_TYPETRAITS_RANK_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/IntegralConstant.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time check for array ranks.
// \ingroup type_traits
//
// This type trait determines the rank of the given template argument. In case the given type
// is an array type, the nested \a value member constant is set to the number of dimensions
// of \a T. Otherwise \a value is set to 0.

   \code
   blaze::Rank< int[] >::value               // Evaluates to 1
   blaze::Rank< int[3] >::value              // Evaluates to 1
   blaze::Rank< const int[2][3][4] >::value  // Evaluates to 3
   blaze::Rank< int[][3] >::value            // Evaluates to 2
   blaze::Rank< int const* >::value          // Evaluates to 0
   blaze::Rank< std::vector<int> >::value    // Evaluates to 0
   \endcode
*/
template< typename T >
struct Rank : public IntegralConstant<size_t,0UL>
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the Rank type trait for empty arrays.
template< typename T >
struct Rank<T[]> : public IntegralConstant<size_t,1UL+Rank<T>::value>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the Rank type trait for non-empty arrays.
template< typename T, unsigned int N >
struct Rank<T[N]> : public IntegralConstant<size_t,1UL+Rank<T>::value>
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
