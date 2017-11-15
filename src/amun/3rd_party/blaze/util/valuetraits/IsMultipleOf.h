//=================================================================================================
/*!
//  \file blaze/util/valuetraits/IsMultipleOf.h
//  \brief Header file for the IsMultipleOf value trait
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

#ifndef _BLAZE_UTIL_VALUETRAITS_ISMULTIPLEOF_H_
#define _BLAZE_UTIL_VALUETRAITS_ISMULTIPLEOF_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/FalseType.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/TrueType.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time check for a multiplicative relationship of two integral values.
// \ingroup value_traits
//
// This value trait tests whether the first given integral value \a M is a multiple of the second
// integral value \a N (i.e. if \f$ M = x*N \f$, where x is any positive integer in the range
// \f$ [0..\infty) \f$). In case the value is a multiple of \a N, the \a value member enumeration
// is set to \a true, the nested type definition \a Type is \a TrueType, and the class derives
// from \a TrueType. Otherwise \a value is set to \a false, \a Type is \a FalseType, and the
// class derives from \a FalseType.

   \code
   blaze::IsMultipleOf<8,2>::value  // Evaluates to 1 (x*2 = 8 for x = 4)
   blaze::IsMultipleOf<2,2>::value  // Evaluates to 1 (x*2 = 2 for x = 1)
   blaze::IsMultipleOf<0,2>::Type   // Results in TrueType (x*2 = 0 for x = 0)
   blaze::IsMultipleOf<0,0>         // Is derived from TrueType (x*0 = 0 for any x)

   blaze::IsMultipleOf<5,3>::value  // Evaluates to 0 (5 is no integral multiple of 3)
   blaze::IsMultipleOf<2,3>::Type   // Results in TrueType (2 is no integral multiple of 3)
   blaze::IsMultipleOf<2,0>         // Is derived from TrueType (2 is no multiple of 0)
   \endcode
*/
template< size_t M, size_t N >
struct IsMultipleOf : public BoolConstant< M % N == 0UL >
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Partial specialization of the IsMultipleOf value trait for M > 0 and N = 0.
// \ingroup type_traits
*/
template< size_t M >
struct IsMultipleOf<M,0UL> : public FalseType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Partial specialization of the IsMultipleOf value trait for M = 0 and N = 0.
// \ingroup type_traits
*/
template<>
struct IsMultipleOf<0,0> : public TrueType
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
