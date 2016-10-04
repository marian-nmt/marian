//=================================================================================================
/*!
//  \file blaze/math/typetraits/HasConstDataAccess.h
//  \brief Header file for the HasConstDataAccess type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_HASCONSTDATAACCESS_H_
#define _BLAZE_MATH_TYPETRAITS_HASCONSTDATAACCESS_H_


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
/*!\brief Compile time check for low-level access to constant data.
// \ingroup math_type_traits
//
// This type trait tests whether the given data type provides a low-level access to constant
// data via a const 'data' member function. In case the according member function is provided,
// the \a value member constant is set to \a true, the nested type definition \a Type is
// \a TrueType, and the class derives from \a TrueType. Otherwise \a value is set to \a false,
// \a Type is \a FalseType, and the class derives from \a FalseType. Examples:

   \code
   blaze::HasConstDataAccess< StaticVector<float,3U> >::value      // Evaluates to 1
   blaze::HasConstDataAccess< const DynamicVector<double> >::Type  // Results in TrueType
   blaze::HasConstDataAccess< volatile DynamicMatrix<int> >        // Is derived from TrueType
   blaze::HasConstDataAccess< int >::value                         // Evaluates to 0
   blaze::HasConstDataAccess< const CompressedVector<int> >::Type  // Results in FalseType
   blaze::HasConstDataAccess< volatile CompressedMatrix<int> >     // Is derived from FalseType
   \endcode
*/
template< typename T >
struct HasConstDataAccess : public FalseType
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasConstDataAccess type trait for const types.
// \ingroup math_type_traits
*/
template< typename T >
struct HasConstDataAccess< const T > : public HasConstDataAccess<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasConstDataAccess type trait for volatile types.
// \ingroup math_type_traits
*/
template< typename T >
struct HasConstDataAccess< volatile T > : public HasConstDataAccess<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasConstDataAccess type trait for cv qualified types.
// \ingroup math_type_traits
*/
template< typename T >
struct HasConstDataAccess< const volatile T > : public HasConstDataAccess<T>
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
