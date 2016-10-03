//=================================================================================================
/*!
//  \file blaze/math/typetraits/RemoveAdaptor.h
//  \brief Header file for the RemoveAdaptor type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_REMOVEADAPTOR_H_
#define _BLAZE_MATH_TYPETRAITS_REMOVEADAPTOR_H_


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Removal of top level adaptor types.
// \ingroup math_type_traits
//
// In case the given type is an adaptor type (SymmetricMatrix, LowerMatrix, UpperMatrix, ...),
// the RemoveAdaptor type trait removes the adaptor and extracts the contained general matrix
// type. Else the given type is returned as is. Note that cv-qualifiers are preserved.

   \code
   using blaze::DynamicVector;
   using blaze::DynamicMatrix;
   using blaze::CompressedMatrix;
   using blaze::SymmetricMatrix;
   using blaze::LowerMatrix;
   using blaze::UpperMatrix;

   typedef SymmetricMatrix< DynamicMatrix<int> >   SymmetricDynamic;
   typedef LowerMatrix< CompressedMatrix<float> >  LowerCompressed;
   typedef UpperMatrix< DynamicMatrix<double> >    UpperDynamic;

   blaze::RemoveAdaptor< SymmetricDynamic >::Type             // Results in 'DynamicMatrix<int>'
   blaze::RemoveAdaptor< const LowerCompressed >::Type        // Results in 'const CompressedMatrix<float>'
   blaze::RemoveAdaptor< volatile UpperDynamic >::Type        // Results in 'volatile DynamicMatrix<double>'
   blaze::RemoveAdaptor< int >::Type                          // Results in 'int'
   blaze::RemoveAdaptor< const DynamicVector<int> >::Type     // Results in 'const DynamicVector<int>'
   blaze::RemoveAdaptor< volatile DynamicMatrix<int> >::Type  // Results in 'volatile DynamicMatrix<int>'
   \endcode
*/
template< typename T >
struct RemoveAdaptor
{
 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   typedef T  Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the RemoveAdaptor type trait for const types.
// \ingroup math_type_traits
*/
template< typename T >
struct RemoveAdaptor< const T >
{
 public:
   //**********************************************************************************************
   typedef const typename RemoveAdaptor<T>::Type  Type;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the RemoveAdaptor type trait for volatile types.
// \ingroup math_type_traits
*/
template< typename T >
struct RemoveAdaptor< volatile T >
{
 public:
   //**********************************************************************************************
   typedef volatile typename RemoveAdaptor<T>::Type  Type;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the RemoveAdaptor type trait for cv qualified types.
// \ingroup math_type_traits
*/
template< typename T >
struct RemoveAdaptor< const volatile T >
{
 public:
   //**********************************************************************************************
   typedef const volatile typename RemoveAdaptor<T>::Type  Type;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the RemoveAdaptor type trait.
// \ingroup type_traits
//
// The RemoveAdaptor_ alias declaration provides a convenient shortcut to access the nested
// \a Type of the RemoveAdaptor class template. For instance, given the type \a T the following
// two type definitions are identical:

   \code
   using Type1 = typename RemoveAdaptor<T>::Type;
   using Type2 = RemoveAdaptor_<T>;
   \endcode
*/
template< typename T >
using RemoveAdaptor_ = typename RemoveAdaptor<T>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
