//=================================================================================================
/*!
//  \file blaze/util/typetraits/IsInteger.h
//  \brief Header file for the IsInteger type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_ISINTEGER_H_
#define _BLAZE_UTIL_TYPETRAITS_ISINTEGER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/FalseType.h>
#include <blaze/util/TrueType.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time check for integer types.
// \ingroup type_traits
//
// This type trait tests whether or not the given template parameter is an integer type (i.e.,
// either (signed) int or unsigned int, possibly cv-qualified). In case the type is an integer
// type (ignoring the cv-qualifiers), the \a value member constant is set to \a true, the nested
// type definition \a Type is \a TrueType, and the class derives from \a TrueType. Otherwise
// \a value is set to \a false, \a Type is \a FalseType, and the class derives from \a FalseType.

   \code
   blaze::IsInteger<int>::value                 // Evaluates to 'true'
   blaze::IsInteger<const unsigned int>::Type   // Results in TrueType
   blaze::IsInteger<const volatile signed int>  // Is derived from TrueType
   blaze::IsInteger<unsigned short>::value      // Evaluates to 'false'
   blaze::IsInteger<const long>::Type           // Results in FalseType
   blaze::IsInteger<volatile float>             // Is derived from FalseType
   \endcode

// Note the difference between the IsInteger and IsIntegral type traits: Whereas the IsInteger
// type trait specifically tests whether the given data type is either int or unsigned int
// (possibly cv-qualified), the IsIntegral type trait tests whether the given template argument
// is an integral data type (char, short, int, long, etc.).
*/
template< typename T >
struct IsInteger : public FalseType
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsInteger type trait for the plain 'int' type.
template<>
struct IsInteger<int> : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsInteger type trait for 'const int'.
template<>
struct IsInteger<const int> : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsInteger type trait for 'volatile int'.
template<>
struct IsInteger<volatile int> : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsInteger type trait for 'const volatile int'.
template<>
struct IsInteger<const volatile int> : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsInteger type trait for the plain 'unsigned int' type.
template<>
struct IsInteger<unsigned int> : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsInteger type trait for 'const unsigned int'.
template<>
struct IsInteger<const unsigned int> : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsInteger type trait for 'volatile unsigned int'.
template<>
struct IsInteger<volatile unsigned int> : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsInteger type trait for 'const volatile unsigned int'.
template<>
struct IsInteger<const volatile unsigned int> : public TrueType
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
