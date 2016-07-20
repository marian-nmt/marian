//=================================================================================================
/*!
//  \file blaze/math/typetraits/IsLower.h
//  \brief Header file for the IsLower type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_ISLOWER_H_
#define _BLAZE_MATH_TYPETRAITS_ISLOWER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/mpl/Or.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time check for lower triangular matrices.
// \ingroup math_type_traits
//
// This type trait tests whether or not the given template parameter is a lower triangular matrix
// type (i.e. a matrix type that is guaranteed to be lower triangular at compile time). This also
// includes lower unitriangular and strictly lower triangular matrices. In case the type is a
// lower triangular matrix type, the \a value member constant is set to \a true, the nested type
// definition \a Type is \a TrueType, and the class derives from \a TrueType. Otherwise \a value
// is set to \a false, \a Type is \a FalseType, and the class derives from \a FalseType.

   \code
   using blaze::rowMajor;

   typedef blaze::StaticMatrix<double,3UL,3UL,rowMajor>  StaticMatrixType;
   typedef blaze::DynamicMatrix<float,rowMajor>          DynamicMatrixType;
   typedef blaze::CompressedMatrix<int,rowMajor>         CompressedMatrixType;

   typedef blaze::LowerMatrix<StaticMatrixType>         LowerStaticType;
   typedef blaze::LowerMatrix<DynamicMatrixType>        LowerDynamicType;
   typedef blaze::UniLowerMatrix<CompressedMatrixType>  UniLowerCompressedType;

   blaze::IsLower< LowerStaticType >::value           // Evaluates to 1
   blaze::IsLower< const LowerDynamicType >::Type     // Results in TrueType
   blaze::IsLower< volatile UniLowerCompressedType >  // Is derived from TrueType
   blaze::IsLower< StaticMatrixType >::value          // Evaluates to 0
   blaze::IsLower< const DynamicMatrixType >::Type    // Results in FalseType
   blaze::IsLower< volatile CompressedMatrixType >    // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsLower : public BoolConstant< Or< IsUniLower<T>, IsStrictlyLower<T> >::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsLower type trait for const types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsLower< const T > : public IsLower<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsLower type trait for volatile types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsLower< volatile T > : public IsLower<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsLower type trait for cv qualified types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsLower< const volatile T > : public IsLower<T>
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
