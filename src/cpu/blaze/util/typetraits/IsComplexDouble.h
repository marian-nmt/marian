//=================================================================================================
/*!
//  \file blaze/util/typetraits/IsComplexDouble.h
//  \brief Header file for the IsComplexDouble type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_ISCOMPLEXDOUBLE_H_
#define _BLAZE_UTIL_TYPETRAITS_ISCOMPLEXDOUBLE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/Complex.h>
#include <blaze/util/FalseType.h>
#include <blaze/util/TrueType.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time check for single precision complex types.
// \ingroup type_traits
//
// This type trait tests whether or not the given template parameter is of type \c complex<double>.
// In case the type is \c complex<double> (ignoring the cv-qualifiers), the \a value member
// constant is set to \a true, the nested type definition \a Type is \a TrueType, and the class
// derives from \a TrueType. Otherwise \a value is set to \a false, \a Type is \a FalseType, and
// the class derives from \a FalseType.

   \code
   blaze::IsComplexDouble< complex<double> >::value       // Evaluates to 'true'
   blaze::IsComplexDouble< const complex<double> >::Type  // Results in TrueType
   blaze::IsComplexDouble< volatile complex<double> >     // Is derived from TrueType
   blaze::IsComplexDouble< double >::value                // Evaluates to 'false'
   blaze::IsComplexDouble< const complex<float> >::Type   // Results in FalseType
   blaze::IsComplexDouble< const volatile complex<int> >  // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsComplexDouble : public FalseType
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsComplexDouble type trait for the plain 'complex<double>' type.
template<>
struct IsComplexDouble< complex<double> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsComplexDouble type trait for 'const complex<double>'.
template<>
struct IsComplexDouble< const complex<double> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsComplexDouble type trait for 'volatile complex<double>'.
template<>
struct IsComplexDouble< volatile complex<double> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsComplexDouble type trait for 'const volatile complex<double>'
template<>
struct IsComplexDouble< const volatile complex<double> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
