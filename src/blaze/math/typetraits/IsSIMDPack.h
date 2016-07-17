//=================================================================================================
/*!
//  \file blaze/math/typetraits/IsSIMDPack.h
//  \brief Header file for the IsSIMDPack type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_ISSIMDPACK_H_
#define _BLAZE_MATH_TYPETRAITS_ISSIMDPACK_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/simd/SIMDPack.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/typetraits/IsBaseOf.h>
#include <blaze/util/typetraits/RemoveCV.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time check for SIMD data types.
// \ingroup math_type_traits
//
// This type trait tests whether the given data type is a Blaze SIMD packed data type. The
// following types are considered valid SIMD packed types:
//
// <ul>
//    <li>Basic SIMD packed data types:</li>
//    <ul>
//       <li>SIMDint8</li>
//       <li>SIMDint16</li>
//       <li>SIMDint32</li>
//       <li>SIMDint64</li>
//       <li>SIMDfloat</li>
//       <li>SIMDdouble</li>
//       <li>SIMDcint8</li>
//       <li>SIMDcint16</li>
//       <li>SIMDcint32</li>
//       <li>SIMDcint64</li>
//       <li>SIMDcfloat</li>
//       <li>SIMDcdouble</li>
//    </ul>
//    <li>Derived SIMD packed data types:</li>
//    <ul>
//       <li>SIMDshort</li>
//       <li>SIMDushort</li>
//       <li>SIMDint</li>
//       <li>SIMDuint</li>
//       <li>SIMDlong</li>
//       <li>SIMDulong</li>
//       <li>SIMDcshort</li>
//       <li>SIMDcushort</li>
//       <li>SIMDcint</li>
//       <li>SIMDcuint</li>
//       <li>SIMDclong</li>
//       <li>SIMDculong</li>
//    </ul>
// </ul>
//
// In case the data type is a SIMD data type, the \a value member constant is set to \a true,
// the nested type definition \a Type is \a TrueType, and the class derives from \a TrueType.
// Otherwise \a value is set to \a false, \a Type is \a FalseType, and the class derives from
// \a FalseType. Examples:

   \code
   blaze::IsSIMDPack< SIMDint32 >::value        // Evaluates to 1
   blaze::IsSIMDPack< const SIMDdouble >::Type  // Results in TrueType
   blaze::IsSIMDPack< volatile SIMDint >        // Is derived from TrueType
   blaze::IsSIMDPack< int >::value                 // Evaluates to 0
   blaze::IsSIMDPack< const double >::Type         // Results in FalseType
   blaze::IsSIMDPack< volatile complex<double> >   // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsSIMDPack
   : public BoolConstant< Or< IsBaseOf<SIMDPack< RemoveCV_<T> >,T>
                            , IsBaseOf<SIMDPack< RemoveCV_<T> >,T> >::value >
{};
//*************************************************************************************************

} // namespace blaze

#endif
