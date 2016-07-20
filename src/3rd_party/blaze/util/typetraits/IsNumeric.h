//=================================================================================================
/*!
//  \file blaze/util/typetraits/IsNumeric.h
//  \brief Header file for the IsNumeric type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_ISNUMERIC_H_
#define _BLAZE_UTIL_TYPETRAITS_ISNUMERIC_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/Complex.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/typetraits/IsBoolean.h>
#include <blaze/util/typetraits/IsBuiltin.h>
#include <blaze/util/typetraits/IsVoid.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time check for numeric types.
// \ingroup type_traits
//
// This type trait tests whether or not the given template parameter is a numeric data type.
// Blaze considers all integral (except \a bool), floating point, and complex data types as
// numeric data types. In case the type is a numeric type, the \a value member constant is
// set to \a true, the nested type definition \a Type is \a TrueType, and the class derives
// from \a TrueType. Otherwise \a value is set to \a false, \a Type is \a FalseType, and the
// class derives from \a FalseType.

   \code
   blaze::IsNumeric<int>::value                // Evaluates to 'true' (int is a numeric data type)
   blaze::IsNumeric<const double>::Type        // Results in TrueType (float is a numeric data type)
   blaze::IsNumeric<volatile complex<float> >  // Is derived from TrueType (complex<float> is a numeric data type)
   blaze::IsNumeric<void>::value               // Evaluates to 'false' (void is not a numeric data type)
   blaze::IsNumeric<bool>::Type                // Results in FalseType (bool is not a numeric data type)
   blaze::IsNumeric<const bool>                // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsNumeric : public BoolConstant< IsBuiltin<T>::value && !IsBoolean<T>::value && !IsVoid<T>::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsNumeric type trait for the plain 'complex' type.
template< typename T >
struct IsNumeric< complex<T> > : public IsNumeric<T>::Type
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsNumeric type trait for 'const complex'.
template< typename T >
struct IsNumeric< const complex<T> > : public IsNumeric<T>::Type
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsNumeric type trait for 'volatile complex'.
template< typename T >
struct IsNumeric< volatile complex<T> > : public IsNumeric<T>::Type
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsNumeric type trait for 'const volatile complex'.
template< typename T >
struct IsNumeric< const volatile complex<T> > : public IsNumeric<T>::Type
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
