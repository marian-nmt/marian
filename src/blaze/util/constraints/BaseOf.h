//=================================================================================================
/*!
//  \file blaze/util/constraints/BaseOf.h
//  \brief Constraint on the inheritance relationship of a data type
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

#ifndef _BLAZE_UTIL_CONSTRAINTS_BASEOF_H_
#define _BLAZE_UTIL_CONSTRAINTS_BASEOF_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/typetraits/IsBaseOf.h>


namespace blaze {

//=================================================================================================
//
//  MUST_BE_BASE_OF CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the inheritance relationship.
// \ingroup constraints
//
// In case \a B is not a base class of \a D, a compilation error is created.
*/
#define BLAZE_CONSTRAINT_MUST_BE_BASE_OF(B,D) \
   static_assert( ::blaze::IsBaseOf<B,D>::value, "Broken inheritance relationship detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_NOT_BE_BASE_OF CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the inheritance relationship.
// \ingroup constraints
//
// In case \a B is a base class of \a D or in case \a B is the same type as \a D, a
// compilation error is created.
*/
#define BLAZE_CONSTRAINT_MUST_NOT_BE_BASE_OF(B,D) \
   static_assert( !::blaze::IsBaseOf<B,D>::value, "Unexpected inheritance relationship detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_BE_STRICTLY_BASE_OF CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the inheritance relationship.
// \ingroup constraints
//
// In case \a B is not a base class of \a D, a compilation error is created. In contrast to
// the BLAZE_CONSTRAINT_MUST_BE_BASE_OF constraint, a compilation error is also created in case
// \a B and \a D are the same type.
*/
#define BLAZE_CONSTRAINT_MUST_BE_STRICTLY_BASE_OF(B,D) \
   static_assert( ::blaze::IsBaseOf<B,D>::value && !::blaze::IsBaseOf<D,B>::value , "Broken inheritance relationship detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_NOT_BE_STRICTLY_BASE_OF CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the inheritance relationship.
// \ingroup constraints
//
// In case \a B is a base class of \a D, a compilation error is created. In contrast to the
// BLAZE_CONSTRAINT_MUST_NOT_BE_BASE_OF constraint, no compilation error is created in case \a B
// and \a D are the same type.
*/
#define BLAZE_CONSTRAINT_MUST_NOT_BE_STRICTLY_BASE_OF(B,D) \
   static_assert( !::blaze::IsBaseOf<B,D>::value || ::blaze::IsBaseOf<D,B>::value , "Unexpected inheritance relationship detected" )
//*************************************************************************************************

} // namespace blaze

#endif
