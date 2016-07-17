//=================================================================================================
/*!
//  \file blaze/util/mpl/Min.h
//  \brief Header file for the Min class template
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

#ifndef _BLAZE_UTIL_MPL_MIN_H_
#define _BLAZE_UTIL_MPL_MIN_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Less.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time value evaluation.
// \ingroup mpl
//
// The Min class template selects the smaller of the two given template arguments \a T1 and \a T2.
// In order for Min to be able to determine the smaller type, both arguments are required to have
// a nested member \a value. In case \a T1::value is smaller than \a T2::value, the nested type
// definition \a Type is set to \a T1, otherwise it is set to \a T2.

   \code
   blaze::Min< Int<3> , Int<2>  >::Type  // Results in Int<2>
   blaze::Min< Long<3>, Int<2>  >::Type  // Results in Int<2>
   blaze::Min< Int<3> , Long<2> >::Type  // Results in Long<2>
   \endcode
*/
template< typename T1    // Type of the first compile time value
        , typename T2 >  // Type of the second compile time value
struct Min : public If_< Less<T1,T2>, T1, T2 >
{};
//*************************************************************************************************

} // namespace blaze

#endif
