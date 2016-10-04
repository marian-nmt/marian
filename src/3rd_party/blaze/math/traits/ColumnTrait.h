//=================================================================================================
/*!
//  \file blaze/math/traits/ColumnTrait.h
//  \brief Header file for the column trait
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

#ifndef _BLAZE_MATH_TRAITS_COLUMNTRAIT_H_
#define _BLAZE_MATH_TRAITS_COLUMNTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/InvalidType.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/typetraits/Decay.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/typetraits/IsReference.h>
#include <blaze/util/typetraits/IsVolatile.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Base template for the ColumnTrait class.
// \ingroup math_traits
//
// \section columntrait_general General
//
// The ColumnTrait class template offers the possibility to select the resulting data type when
// creating a view on a specific column of a dense or sparse matrix. ColumnTrait defines the nested
// type \a Type, which represents the resulting data type of the column operation. In case the
// given data type is not a dense or sparse matrix type, the resulting data type \a Type is
// set to \a INVALID_TYPE. Note that \a const and \a volatile qualifiers and reference modifiers
// are generally ignored.
//
// Per default, the ColumnTrait template only supports the following matrix types:
//
// <ul>
//    <li>blaze::StaticMatrix</li>
//    <li>blaze::HybridMatrix</li>
//    <li>blaze::DynamicMatrix</li>
//    <li>blaze::CustomMatrix</li>
//    <li>blaze::CompressedMatrix</li>
//    <li>blaze::Submatrix</li>
// </ul>
//
//
// \section columntrait_specializations Creating custom specializations
//
// It is possible to specialize the ColumnTrait template for additional user-defined matrix types.
// The following example shows the according specialization for the DynamicMatrix class template:

   \code
   template< typename T1, bool SO >
   struct ColumnTrait< DynamicMatrix<T1,SO> >
   {
      typedef DynamicVector<T1,true>  Type;
   };
   \endcode

// \n \section columntrait_examples Examples
//
// The following example demonstrates the use of the ColumnTrait template, where depending on
// the given matrix type the resulting column type is selected:

   \code
   using blaze::rowMajor;
   using blaze::columnMajor;

   // Definition of the column type of a column-major dynamic matrix
   typedef blaze::DynamicMatrix<int,columnMajor>    MatrixType1;
   typedef typename ColumnTrait<MatrixType1>::Type  ColumnType1;

   // Definition of the column type of the row-major static matrix
   typedef blaze::StaticMatrix<int,3UL,3UL,rowMajor>  MatrixType2;
   typedef typename ColumnTrait<MatrixType2>::Type    ColumnType2;
   \endcode
*/
template< typename MT >  // Type of the matrix
struct ColumnTrait
{
 private:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   struct Failure { using Type = INVALID_TYPE; };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = typename If_< Or< IsConst<MT>, IsVolatile<MT>, IsReference<MT> >
                            , ColumnTrait< Decay_<MT> >
                            , Failure >::Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the ColumnTrait type trait.
// \ingroup math_traits
//
// The ColumnTrait_ alias declaration provides a convenient shortcut to access the nested
// \a Type of the ColumnTrait class template. For instance, given the matrix type \a MT the
// following two type definitions are identical:

   \code
   using Type1 = typename ColumnTrait<MT>::Type;
   using Type2 = ColumnTrait_<MT>;
   \endcode
*/
template< typename MT >  // Type of the matrix
using ColumnTrait_ = typename ColumnTrait<MT>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
