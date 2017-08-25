//=================================================================================================
/*!
//  \file blaze/util/constraints/SameType.h
//  \brief Data type constraint
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

#ifndef _BLAZE_UTIL_CONSTRAINTS_SAMETYPE_H_
#define _BLAZE_UTIL_CONSTRAINTS_SAMETYPE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/typetraits/IsSame.h>


namespace blaze {

//=================================================================================================
//
//  MUST_BE_SAME_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Data type constraint.
// \ingroup constraints
//
// In case the two types \a A and \a B are not the same (ignoring all cv-qualifiers of both data
// types), a compilation error is created. The following example illustrates the behavior of this
// constraint:

   \code
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( double, double );        // No compilation error
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( double, const double );  // No compilation error (only cv-qualifiers differ)
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( double, float );         // Compilation error, different data types!
   \endcode

// In case the cv-qualifiers should not be ignored (e.g. 'double' and 'const double' should be
// considered to be unequal), use the blaze::BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE constraint.
*/
#define BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(A,B) \
   static_assert( ::blaze::IsSame<A,B>::value, "Non-matching types detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_NOT_BE_SAME_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Data type constraint.
// \ingroup constraints
//
// In case the two types \a A and \a B are the same (ignoring all cv-qualifiers of both data
// types), a compilation error is created. The following example illustrates the behavior of
// this constraint:

   \code
   BLAZE_CONSTRAINT_MUST_NOT_BE_SAME_TYPE( double, float );         // No compilation error, different data types
   BLAZE_CONSTRAINT_MUST_NOT_BE_SAME_TYPE( double, const double );  // Compilation error (only cv-qualifiers differ)
   BLAZE_CONSTRAINT_MUST_NOT_BE_SAME_TYPE( double, double );        // Compilation error, same data type!
   \endcode

// In case the cv-qualifiers should not be ignored (e.g. 'double' and 'const double' should
// be considered to be unequal), use the blaze::BLAZE_CONSTRAINT_MUST_NOT_BE_STRICTLY_SAME_TYPE
// constraint.
*/
#define BLAZE_CONSTRAINT_MUST_NOT_BE_SAME_TYPE(A,B) \
   static_assert( !::blaze::IsSame<A,B>::value, "Matching types detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_BE_STRICTLY_SAME_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Data type constraint.
// \ingroup constraints
//
// In case the two types \a A and \a B are not the same, a compilation error is created. Note
// that this constraint even considers two types as unequal if the cv-qualifiers differ, e.g.

   \code
   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( double, double );        // No compilation error
   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( double, const double );  // Compilation error, different cv-qualifiers!
   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( double, float );         // Compilation error, different data types!
   \endcode

// In case the cv-qualifiers should be ignored (e.g. 'double' and 'const double' should be
// considered to be equal), use the blaze::BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE constraint.
*/
#define BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE(A,B) \
   static_assert( ::blaze::IsStrictlySame<A,B>::value, "Non-matching types detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_NOT_BE_STRICTLY_SAME_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Data type constraint.
// \ingroup constraints
//
// In case the two types \a A and \a B are the same, a compilation error is created. Note that
// this constraint even considers two types as unequal if the cv-qualifiers differ, e.g.

   \code
   BLAZE_CONSTRAINT_MUST_NOT_BE_STRICTLY_SAME_TYPE( double, float );         // No compilation error, different data types
   BLAZE_CONSTRAINT_MUST_NOT_BE_STRICTLY_SAME_TYPE( double, const double );  // No compilation error, different cv-qualifiers!
   BLAZE_CONSTRAINT_MUST_NOT_BE_STRICTLY_SAME_TYPE( double, double );        // Compilation error, same data type!
   \endcode

// In case the cv-qualifiers should be ignored (e.g. 'double' and 'const double' should be
// considered to be equal), use the blaze::BLAZE_CONSTRAINT_MUST_NOT_BE_SAME_TYPE constraint.
*/
#define BLAZE_CONSTRAINT_MUST_NOT_BE_STRICTLY_SAME_TYPE(A,B) \
   static_assert( !::blaze::IsStrictlySame<A,B>::value, "Matching types detected" )
//*************************************************************************************************

} // namespace blaze

#endif
