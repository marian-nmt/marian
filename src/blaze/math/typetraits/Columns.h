//=================================================================================================
/*!
//  \file blaze/math/typetraits/Columns.h
//  \brief Header file for the Columns type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_COLUMNS_H_
#define _BLAZE_MATH_TYPETRAITS_COLUMNS_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/mpl/SizeT.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time evaluation of the number of columns of a matrix.
// \ingroup math_type_traits
//
// The Columns type trait evaluates the number of columns of the given matrix type at compile time.
// In case the given type \a T is a matrix type with a fixed number of columns (e.g. StaticMatrix),
// the \a value member constant is set to the according number of columns. In all other cases,
// \a value is set to 0.

   \code
   using blaze::StaticMatrix;
   using blaze::HybridMatrix;
   using blaze::DynamicMatrix;

   blaze::Columns< StaticMatrix<int,3UL,2UL> >::value  // Evaluates to 2
   blaze::Columns< HybridMatrix<int,3UL,2UL> >::value  // Evaluates to 0; Only maximum number of columns is fixed!
   blaze::Columns< DynamicMatrix<int> >::value         // Evaluates to 0; Number of columns not fixed at compile time!
   blaze::Columns< int >::value                        // Evaluates to 0
   \endcode
*/
template< typename T >
struct Columns : public SizeT<0UL>
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Columns type trait for const types.
// \ingroup math_type_traits
*/
template< typename T >
struct Columns< const T > : public SizeT< Columns<T>::value >
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Columns type trait for volatile types.
// \ingroup math_type_traits
*/
template< typename T >
struct Columns< volatile T > : public SizeT< Columns<T>::value >
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Columns type trait for cv qualified types.
// \ingroup math_type_traits
*/
template< typename T >
struct Columns< const volatile T > : public SizeT< Columns<T>::value >
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
