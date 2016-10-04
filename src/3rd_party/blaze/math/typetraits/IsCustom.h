//=================================================================================================
/*!
//  \file blaze/math/typetraits/IsCustom.h
//  \brief Header file for the IsCustom type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_ISCUSTOM_H_
#define _BLAZE_MATH_TYPETRAITS_ISCUSTOM_H_


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
/*!\brief Compile time check for custom data types.
// \ingroup math_type_traits
//
// This type trait tests whether the given data type is a custom data type, i.e. a custom vector
// or a custom matrix. In case the data type a custom data type, the \a value member constant is
// set to \a true, the nested type definition \a Type is \a TrueType, and the class derives from
// \a TrueType. Otherwise \a value is set to \a false, \a Type is \a FalseType, and the class
// derives from \a FalseType. Examples:

   \code
   using blaze::CustomVector;
   using blaze::aligned;
   using blaze::unpadded;
   using blaze::columnVector;
   using blaze::rowMajor;

   typedef CustomVector<int,aligned,unpadded,columnVector>  CustomVectorType;
   typedef CustomMatrix<double,aligned,unpadded,rowMajor>   CustomMatrixType;

   blaze::IsCustom< CustomVectorType >::value                        // Evaluates to 1
   blaze::IsCustom< const CustomVectorType >::Type                   // Results in TrueType
   blaze::IsCustom< volatile CustomMatrixType >                      // Is derived from TrueType
   blaze::IsCustom< int >::value                                     // Evaluates to 0
   blaze::IsCustom< const DynamicVector<float,columnVector> >::Type  // Results in FalseType
   blaze::IsCustom< volatile DynamicMatrix<int,rowMajor> >           // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsCustom : public FalseType
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsCustom type trait for const types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsCustom< const T > : public IsCustom<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsCustom type trait for volatile types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsCustom< volatile T > : public IsCustom<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsCustom type trait for cv qualified types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsCustom< const volatile T > : public IsCustom<T>
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
