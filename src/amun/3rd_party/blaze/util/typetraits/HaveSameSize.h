//=================================================================================================
/*!
//  \file blaze/util/typetraits/HaveSameSize.h
//  \brief Header file for the HaveSameSize type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_HAVESAMESIZE_H_
#define _BLAZE_UTIL_TYPETRAITS_HAVESAMESIZE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/FalseType.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/TrueType.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time size check.
// \ingroup type_traits
//
// This class offers the possibility to test the size of two types at compile time. If an object
// of type \a T1 has the same size as an object of type \a T2, the \a value member constant is
// set to \a true, the nested type definition \a Type is \a TrueType, and the class derives from
// \a TrueType. Otherwise \a value  is set to \a false, \a Type is \a FalseType, and the class
// derives from \a FalseType.

   \code
   blaze::HaveSameSize<int,unsigned int>::value  // Evaluates to 'true'
   blaze::HaveSameSize<int,unsigned int>::Type   // Results in TrueType
   blaze::HaveSameSize<int,unsigned int>         // Is derived from TrueType
   blaze::HaveSameSize<char,wchar_t>::value      // Evalutes to 'false'
   blaze::HaveSameSize<char,wchar_t>::Type       // Results in FalseType
   blaze::HaveSameSize<char,wchar_t>             // Is derived from FalseType
   \endcode

// One example for the application of this type trait is a compile time check if the compiler
// supports the 'Empty Derived class Optimization (EDO)':

   \code
   // Definition of the base class A
   struct A {
      int i_;
   };

   // Definition of the derived class B
   struct B : public A {};

   // Testing whether or not an object of type B has the same size as the
   //   base class A and whether the compiler supports EDO
   blaze::HaveSameSize( A, B );
   \endcode
*/
template< typename T1, typename T2 >
class HaveSameSize : public BoolConstant< sizeof(T1) == sizeof(T2) >
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Partial specialization of the compile time size constraint.
// \ingroup type_traits
//
// This class is a partial specialization of the HaveSameSize template for the type \a void
// as first template argument. The \a value member constant is automatically set to \a false,
// the nested type definition \a Type is \a FalseType, and the class derives from \a FalseType
// for any given type \a T since the \a void type has no size.
*/
template< typename T >
class HaveSameSize<void,T> : public FalseType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Partial specialization of the compile time size constraint.
// \ingroup type_traits
//
// This class is a partial specialization of the HaveSameSize template for the type \a void
// as second template argument. The \a value member constant is automatically set to \a false,
// the nested type definition \a Type is \a FalseType, and the class derives from \a FalseType
// for any given type \a T since the \a void type has no size.
*/
template< typename T >
class HaveSameSize<T,void> : public FalseType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Full specialization of the compile time size constraint.
// \ingroup type_traits
//
// This class is a full specialization of the HaveSameSize template for the type \a void
// as first and second template argument. The \a value member constant is automatically set
// to \a true, the nested type definition \a Type is \a TrueType, and the class derives from
// \a TrueType since both arguments are \a void.
*/
template<>
class HaveSameSize<void,void> : public TrueType
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
