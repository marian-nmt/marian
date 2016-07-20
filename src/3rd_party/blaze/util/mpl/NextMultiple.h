//=================================================================================================
/*!
//  \file blaze/util/mpl/NextMultiple.h
//  \brief Header file for the NextMultiple class template
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

#ifndef _BLAZE_UTIL_MPL_NEXTMULTIPLE_H_
#define _BLAZE_UTIL_MPL_NEXTMULTIPLE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/IntegralConstant.h>
#include <blaze/util/mpl/Minus.h>
#include <blaze/util/mpl/Modulus.h>
#include <blaze/util/mpl/Plus.h>
#include <blaze/util/StaticAssert.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time integral round up operation.
// \ingroup mpl
//
// The NextMultiple class template rounds up the given template argument of type \a T1 to the
// next multiple of the given template argument of type \a T2. In case \a T1 already represents
// a multiple of \a T2, the result is \a T1. In order for NextMultiple to be able to perform the
// round up operation, both arguments are required to have a nested member \a value. The result
// of the operation can be accessed via the nested member \a value, the resulting type is
// available via the nested type \a ValueType.

   \code
   blaze::NextMultiple< Int<3> , Int<2>  >::value      // Results in 4
   blaze::NextMultiple< Long<3>, Int<2>  >::ValueType  // Results in long
   blaze::NextMultiple< Int<3> , Long<2> >::ValueType  // Results in long
   \endcode

// Note that both \a T1 and \a T2 are expected to represent positive integrals. The attempt to
// use NextMultiple with a negative or zero integral results in a compilation error!
*/
template< typename T1    // Type of the first compile time value
        , typename T2 >  // Type of the second compile time value
struct NextMultiple
   : public Plus< T1, Modulus< Minus< T2, Modulus< T1, T2 > >, T2 > >
{
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_STATIC_ASSERT( T1::value > 0 && T2::value > 0 );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blaze

#endif
