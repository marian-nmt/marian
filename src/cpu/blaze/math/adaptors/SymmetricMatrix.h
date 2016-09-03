//=================================================================================================
/*!
//  \file blaze/math/adaptors/SymmetricMatrix.h
//  \brief Header file for the implementation of a symmetric matrix adaptor
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

#ifndef _BLAZE_MATH_ADAPTORS_SYMMETRICMATRIX_H_
#define _BLAZE_MATH_ADAPTORS_SYMMETRICMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/adaptors/symmetricmatrix/BaseTemplate.h>
#include <blaze/math/adaptors/symmetricmatrix/DenseNonNumeric.h>
#include <blaze/math/adaptors/symmetricmatrix/DenseNumeric.h>
#include <blaze/math/adaptors/symmetricmatrix/SparseNonNumeric.h>
#include <blaze/math/adaptors/symmetricmatrix/SparseNumeric.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/dense/StaticMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/Forward.h>
#include <blaze/math/Functions.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsDivisor.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/ColumnTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/ForEachTrait.h>
#include <blaze/math/traits/MathTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/RowTrait.h>
#include <blaze/math/traits/SubmatrixTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/Columns.h>
#include <blaze/math/typetraits/IsAdaptor.h>
#include <blaze/math/typetraits/IsHermitian.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsRestricted.h>
#include <blaze/math/typetraits/IsSquare.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/RemoveAdaptor.h>
#include <blaze/math/typetraits/Rows.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/typetraits/IsBuiltin.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  SYMMETRICMATRIX OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name SymmetricMatrix operators */
//@{
template< typename MT, bool SO, bool DF, bool NF >
inline void reset( SymmetricMatrix<MT,SO,DF,NF>& m );

template< typename MT, bool SO, bool DF, bool NF >
inline void reset( SymmetricMatrix<MT,SO,DF,NF>& m, size_t i );

template< typename MT, bool SO, bool DF, bool NF >
inline void clear( SymmetricMatrix<MT,SO,DF,NF>& m );

template< typename MT, bool SO, bool DF, bool NF >
inline bool isDefault( const SymmetricMatrix<MT,SO,DF,NF>& m );

template< typename MT, bool SO, bool DF, bool NF >
inline bool isIntact( const SymmetricMatrix<MT,SO,DF,NF>& m );

template< typename MT, bool SO, bool DF, bool NF >
inline void swap( SymmetricMatrix<MT,SO,DF,NF>& a, SymmetricMatrix<MT,SO,DF,NF>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given symmetric matrix.
// \ingroup symmetric_matrix
//
// \param m The symmetric matrix to be resetted.
// \return void
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF      // Density flag
        , bool NF >    // Numeric flag
inline void reset( SymmetricMatrix<MT,SO,DF,NF>& m )
{
   m.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the specified row/column of the given symmetric matrix.
// \ingroup symmetric_matrix
//
// \param m The symmetric matrix to be resetted.
// \param i The index of the row/column to be resetted.
// \return void
//
// This function resets the values in the specified row/column of the given symmetric matrix to
// their default value. In case the given matrix is a \a rowMajor matrix the function resets the
// values in row \a i, if it is a \a columnMajor matrix the function resets the values in column
// \a i. Note that the capacity of the row/column remains unchanged.
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF      // Density flag
        , bool NF >    // Numeric flag
inline void reset( SymmetricMatrix<MT,SO,DF,NF>& m, size_t i )
{
   m.reset( i );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given symmetric matrix.
// \ingroup symmetric_matrix
//
// \param m The symmetric matrix to be cleared.
// \return void
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF      // Density flag
        , bool NF >    // Numeric flag
inline void clear( SymmetricMatrix<MT,SO,DF,NF>& m )
{
   m.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given symmetric matrix is in default state.
// \ingroup symmetric_matrix
//
// \param m The symmetric matrix to be tested for its default state.
// \return \a true in case the given matrix is component-wise zero, \a false otherwise.
//
// This function checks whether the matrix is in default state. For instance, in case the
// matrix is instantiated for a built-in integral or floating point data type, the function
// returns \a true in case all matrix elements are 0 and \a false in case any matrix element
// is not 0. The following example demonstrates the use of the \a isDefault function:

   \code
   blaze::SymmetricMatrix<int> A;
   // ... Resizing and initialization
   if( isDefault( A ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF      // Density flag
        , bool NF >    // Numeric flag
inline bool isDefault( const SymmetricMatrix<MT,SO,DF,NF>& m )
{
   return isDefault( m.matrix_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given symmetric matrix are intact.
// \ingroup symmetric_matrix
//
// \param m The symmetric matrix to be tested.
// \return \a true in case the given matrix's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the symmetric matrix are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   using blaze::DynamicMatrix;
   using blaze::SymmetricMatrix;

   SymmetricMatrix< DynamicMatrix<int> > A;
   // ... Resizing and initialization
   if( isIntact( A ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF      // Density flag
        , bool NF >    // Numeric flag
inline bool isIntact( const SymmetricMatrix<MT,SO,DF,NF>& m )
{
   return m.isIntact();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two matrices.
// \ingroup symmetric_matrix
//
// \param a The first matrix to be swapped.
// \param b The second matrix to be swapped.
// \return void
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF      // Density flag
        , bool NF >    // Numeric flag
inline void swap( SymmetricMatrix<MT,SO,DF,NF>& a, SymmetricMatrix<MT,SO,DF,NF>& b ) noexcept
{
   a.swap( b );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given symmetric dense \f$ 2 \times 2 \f$ matrix.
// \ingroup symmetric_matrix
//
// \param m The symmetric dense matrix to be inverted.
// \return void
// \exception std::invalid_argument Inversion of singular matrix failed.
//
// This function inverts the given symmetric dense \f$ 2 \times 2 \f$ matrix via the rule of
// Sarrus. The matrix inversion fails if the given matrix is singular and not invertible. In
// this case a \a std::invalid_argument exception is thrown.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invert2x2( SymmetricMatrix<MT,SO,true,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   BLAZE_INTERNAL_ASSERT( m.rows()    == 2UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( m.columns() == 2UL, "Invalid number of columns detected" );

   typedef ElementType_<MT>  ET;

   const MT& A( m.matrix_ );
   MT& B( m.matrix_ );

   const ET det( A(0,0)*A(1,1) - A(0,1)*A(1,0) );

   if( !isDivisor( det ) ) {
      BLAZE_THROW_DIVISION_BY_ZERO( "Inversion of singular matrix failed" );
   }

   const ET idet( ET(1) / det );
   const ET a11( A(0,0) * idet );

   B(0,0) =  A(1,1) * idet;
   B(1,0) = -A(1,0) * idet;
   B(0,1) =  B(1,0);
   B(1,1) =  a11;

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given symmetric dense \f$ 3 \times 3 \f$ matrix.
// \ingroup symmetric_matrix
//
// \param m The symmetric dense matrix to be inverted.
// \return void
// \exception std::invalid_argument Inversion of singular matrix failed.
//
// This function inverts the given symmetric dense \f$ 3 \times 3 \f$ matrix via the rule of
// Sarrus. The matrix inversion fails if the given matrix is singular and not invertible. In
// this case a \a std::invalid_argument exception is thrown.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invert3x3( SymmetricMatrix<MT,SO,true,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   BLAZE_INTERNAL_ASSERT( m.rows()    == 3UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( m.columns() == 3UL, "Invalid number of columns detected" );

   typedef ElementType_<MT>  ET;

   const StaticMatrix<ET,3UL,3UL,SO> A( m.matrix_ );
   MT& B( m.matrix_ );

   B(0,0) = A(1,1)*A(2,2) - A(1,2)*A(2,1);
   B(1,0) = A(1,2)*A(2,0) - A(1,0)*A(2,2);
   B(2,0) = A(1,0)*A(2,1) - A(1,1)*A(2,0);

   const ET det( A(0,0)*B(0,0) + A(0,1)*B(1,0) + A(0,2)*B(2,0) );

   if( !isDivisor( det ) ) {
      BLAZE_THROW_DIVISION_BY_ZERO( "Inversion of singular matrix failed" );
   }

   B(0,1) = B(1,0);
   B(1,1) = A(0,0)*A(2,2) - A(0,2)*A(2,0);
   B(2,1) = A(0,1)*A(2,0) - A(0,0)*A(2,1);
   B(0,2) = B(2,0);
   B(1,2) = B(2,1);
   B(2,2) = A(0,0)*A(1,1) - A(0,1)*A(1,0);

   B /= det;

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given symmetric dense \f$ 4 \times 4 \f$ matrix.
// \ingroup symmetric_matrix
//
// \param m The symmetric dense matrix to be inverted.
// \return void
// \exception std::invalid_argument Inversion of singular matrix failed.
//
// This function inverts the given symmetric dense \f$ 4 \times 4 \f$ matrix via the rule of
// Sarrus. The matrix inversion fails if the given matrix is singular and not invertible. In
// this case a \a std::invalid_argument exception is thrown.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invert4x4( SymmetricMatrix<MT,SO,true,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   BLAZE_INTERNAL_ASSERT( m.rows()    == 4UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( m.columns() == 4UL, "Invalid number of columns detected" );

   typedef ElementType_<MT>  ET;

   const StaticMatrix<ET,4UL,4UL,SO> A( m.matrix_ );
   MT& B( m.matrix_ );

   ET tmp1( A(2,2)*A(3,3) - A(2,3)*A(3,2) );
   ET tmp2( A(2,1)*A(3,3) - A(2,3)*A(3,1) );
   ET tmp3( A(2,1)*A(3,2) - A(2,2)*A(3,1) );

   B(0,0) = A(1,1)*tmp1 - A(1,2)*tmp2 + A(1,3)*tmp3;
   B(0,1) = A(0,2)*tmp2 - A(0,1)*tmp1 - A(0,3)*tmp3;

   ET tmp4( A(2,0)*A(3,3) - A(2,3)*A(3,0) );
   ET tmp5( A(2,0)*A(3,2) - A(2,2)*A(3,0) );

   B(1,1) = A(0,0)*tmp1 - A(0,2)*tmp4 + A(0,3)*tmp5;

   tmp1 = A(2,0)*A(3,1) - A(2,1)*A(3,0);

   B(2,0) = A(1,0)*tmp2 - A(1,1)*tmp4 + A(1,3)*tmp1;
   B(2,1) = A(0,1)*tmp4 - A(0,0)*tmp2 - A(0,3)*tmp1;
   B(3,0) = A(1,1)*tmp5 - A(1,0)*tmp3 - A(1,2)*tmp1;
   B(3,1) = A(0,0)*tmp3 - A(0,1)*tmp5 + A(0,2)*tmp1;

   tmp1 = A(0,1)*A(1,3) - A(0,3)*A(1,1);
   tmp2 = A(0,1)*A(1,2) - A(0,2)*A(1,1);
   tmp3 = A(0,0)*A(1,3) - A(0,3)*A(1,0);
   tmp4 = A(0,0)*A(1,2) - A(0,2)*A(1,0);
   tmp5 = A(0,0)*A(1,1) - A(0,1)*A(1,0);

   B(2,2) = A(3,0)*tmp1 - A(3,1)*tmp3 + A(3,3)*tmp5;
   B(2,3) = A(2,1)*tmp3 - A(2,0)*tmp1 - A(2,3)*tmp5;
   B(3,3) = A(2,0)*tmp2 - A(2,1)*tmp4 + A(2,2)*tmp5;

   B(0,2) = B(2,0);
   B(0,3) = B(3,0);
   B(1,0) = B(0,1);
   B(1,2) = B(2,1);
   B(1,3) = B(3,1);
   B(3,2) = B(2,3);

   const ET det( A(0,0)*B(0,0) + A(0,1)*B(1,0) + A(0,2)*B(2,0) + A(0,3)*B(3,0) );

   if( !isDivisor( det ) ) {
      BLAZE_THROW_DIVISION_BY_ZERO( "Inversion of singular matrix failed" );
   }

   B /= det;

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given symmetric/Hermitian dense \f$ 5 \times 5 \f$ matrix.
// \ingroup symmetric_matrix
//
// \param m The symmetric/Hermitian dense matrix to be inverted.
// \return void
// \exception std::invalid_argument Inversion of singular matrix failed.
//
// This function inverts the given symmetric dense \f$ 5 \times 5 \f$ matrix via the rule of
// Sarrus. The matrix inversion fails if the given matrix is singular and not invertible. In
// this case a \a std::invalid_argument exception is thrown.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invert5x5( SymmetricMatrix<MT,SO,true,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   BLAZE_INTERNAL_ASSERT( m.rows()    == 5UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( m.columns() == 5UL, "Invalid number of columns detected" );

   typedef ElementType_<MT>  ET;

   const StaticMatrix<ET,5UL,5UL,SO> A( m.matrix_ );
   MT& B( m.matrix_ );

   ET tmp1 ( A(3,3)*A(4,4) - A(3,4)*A(4,3) );
   ET tmp2 ( A(3,2)*A(4,4) - A(3,4)*A(4,2) );
   ET tmp3 ( A(3,2)*A(4,3) - A(3,3)*A(4,2) );
   ET tmp4 ( A(3,1)*A(4,4) - A(3,4)*A(4,1) );
   ET tmp5 ( A(3,1)*A(4,3) - A(3,3)*A(4,1) );
   ET tmp6 ( A(3,1)*A(4,2) - A(3,2)*A(4,1) );
   ET tmp7 ( A(3,0)*A(4,4) - A(3,4)*A(4,0) );
   ET tmp8 ( A(3,0)*A(4,3) - A(3,3)*A(4,0) );
   ET tmp9 ( A(3,0)*A(4,2) - A(3,2)*A(4,0) );
   ET tmp10( A(3,0)*A(4,1) - A(3,1)*A(4,0) );

   ET tmp11( A(2,2)*tmp1 - A(2,3)*tmp2 + A(2,4)*tmp3  );
   ET tmp12( A(2,1)*tmp1 - A(2,3)*tmp4 + A(2,4)*tmp5  );
   ET tmp13( A(2,1)*tmp2 - A(2,2)*tmp4 + A(2,4)*tmp6  );
   ET tmp14( A(2,1)*tmp3 - A(2,2)*tmp5 + A(2,3)*tmp6  );
   ET tmp15( A(2,0)*tmp1 - A(2,3)*tmp7 + A(2,4)*tmp8  );
   ET tmp16( A(2,0)*tmp2 - A(2,2)*tmp7 + A(2,4)*tmp9  );
   ET tmp17( A(2,0)*tmp3 - A(2,2)*tmp8 + A(2,3)*tmp9  );

   B(0,0) =   A(1,1)*tmp11 - A(1,2)*tmp12 + A(1,3)*tmp13 - A(1,4)*tmp14;
   B(0,1) = - A(0,1)*tmp11 + A(0,2)*tmp12 - A(0,3)*tmp13 + A(0,4)*tmp14;
   B(1,1) =   A(0,0)*tmp11 - A(0,2)*tmp15 + A(0,3)*tmp16 - A(0,4)*tmp17;

   ET tmp18( A(2,0)*tmp4 - A(2,1)*tmp7 + A(2,4)*tmp10 );
   ET tmp19( A(2,0)*tmp5 - A(2,1)*tmp8 + A(2,3)*tmp10 );
   ET tmp20( A(2,0)*tmp6 - A(2,1)*tmp9 + A(2,2)*tmp10 );

   B(2,0) =   A(1,0)*tmp12 - A(1,1)*tmp15 + A(1,3)*tmp18 - A(1,4)*tmp19;
   B(2,1) = - A(0,0)*tmp12 + A(0,1)*tmp15 - A(0,3)*tmp18 + A(0,4)*tmp19;
   B(3,0) = - A(1,0)*tmp13 + A(1,1)*tmp16 - A(1,2)*tmp18 + A(1,4)*tmp20;
   B(3,1) =   A(0,0)*tmp13 - A(0,1)*tmp16 + A(0,2)*tmp18 - A(0,4)*tmp20;
   B(4,0) =   A(1,0)*tmp14 - A(1,1)*tmp17 + A(1,2)*tmp19 - A(1,3)*tmp20;
   B(4,1) = - A(0,0)*tmp14 + A(0,1)*tmp17 - A(0,2)*tmp19 + A(0,3)*tmp20;

   tmp11 = A(1,1)*tmp1 - A(1,3)*tmp4 + A(1,4)*tmp5;
   tmp12 = A(1,0)*tmp1 - A(1,3)*tmp7 + A(1,4)*tmp8;
   tmp13 = A(1,0)*tmp4 - A(1,1)*tmp7 + A(1,4)*tmp10;
   tmp14 = A(1,0)*tmp5 - A(1,1)*tmp8 + A(1,3)*tmp10;

   B(2,2) = A(0,0)*tmp11 - A(0,1)*tmp12 + A(0,3)*tmp13 - A(0,4)*tmp14;

   tmp1  = A(0,2)*A(1,3) - A(0,3)*A(1,2);
   tmp2  = A(0,1)*A(1,3) - A(0,3)*A(1,1);
   tmp3  = A(0,1)*A(1,2) - A(0,2)*A(1,1);
   tmp4  = A(0,0)*A(1,3) - A(0,3)*A(1,0);
   tmp5  = A(0,0)*A(1,2) - A(0,2)*A(1,0);
   tmp6  = A(0,0)*A(1,1) - A(0,1)*A(1,0);
   tmp7  = A(0,2)*A(1,4) - A(0,4)*A(1,2);
   tmp8  = A(0,1)*A(1,4) - A(0,4)*A(1,1);
   tmp9  = A(0,0)*A(1,4) - A(0,4)*A(1,0);
   tmp10 = A(0,3)*A(1,4) - A(0,4)*A(1,3);

   tmp11 = A(2,1)*tmp10 - A(2,3)*tmp8 + A(2,4)*tmp2;
   tmp12 = A(2,1)*tmp7  - A(2,2)*tmp8 + A(2,4)*tmp3;
   tmp13 = A(2,1)*tmp1  - A(2,2)*tmp2 + A(2,3)*tmp3;
   tmp14 = A(2,0)*tmp10 - A(2,3)*tmp9 + A(2,4)*tmp4;
   tmp15 = A(2,0)*tmp7  - A(2,2)*tmp9 + A(2,4)*tmp5;
   tmp16 = A(2,0)*tmp1  - A(2,2)*tmp4 + A(2,3)*tmp5;
   tmp17 = A(2,0)*tmp8  - A(2,1)*tmp9 + A(2,4)*tmp6;
   tmp18 = A(2,0)*tmp2  - A(2,1)*tmp4 + A(2,3)*tmp6;
   tmp19 = A(2,0)*tmp3  - A(2,1)*tmp5 + A(2,2)*tmp6;

   B(2,3) =   A(4,0)*tmp11 - A(4,1)*tmp14 + A(4,3)*tmp17 - A(4,4)*tmp18;
   B(2,4) = - A(3,0)*tmp11 + A(3,1)*tmp14 - A(3,3)*tmp17 + A(3,4)*tmp18;
   B(3,3) = - A(4,0)*tmp12 + A(4,1)*tmp15 - A(4,2)*tmp17 + A(4,4)*tmp19;
   B(3,4) =   A(3,0)*tmp12 - A(3,1)*tmp15 + A(3,2)*tmp17 - A(3,4)*tmp19;
   B(4,4) = - A(3,0)*tmp13 + A(3,1)*tmp16 - A(3,2)*tmp18 + A(3,3)*tmp19;

   B(0,2) = B(2,0);
   B(0,3) = B(3,0);
   B(0,4) = B(4,0);
   B(1,0) = B(0,1);
   B(1,2) = B(2,1);
   B(1,3) = B(3,1);
   B(1,4) = B(4,1);
   B(3,2) = B(2,3);
   B(4,2) = B(2,4);
   B(4,3) = B(3,4);

   const ET det( A(0,0)*B(0,0) + A(0,1)*B(1,0) + A(0,2)*B(2,0) + A(0,3)*B(3,0) + A(0,4)*B(4,0) );

   if( !isDivisor( det ) ) {
      BLAZE_THROW_DIVISION_BY_ZERO( "Inversion of singular matrix failed" );
   }

   B /= det;

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given symmetric dense \f$ 6 \times 6 \f$ matrix.
// \ingroup symmetric_matrix
//
// \param m The symmetric dense matrix to be inverted.
// \return void
// \exception std::invalid_argument Inversion of singular matrix failed.
//
// This function inverts the given symmetric dense \f$ 6 \times 6 \f$ matrix via the rule of
// Sarrus. The matrix inversion fails if the given matrix is singular and not invertible. In
// this case a \a std::invalid_argument exception is thrown.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invert6x6( SymmetricMatrix<MT,SO,true,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   BLAZE_INTERNAL_ASSERT( m.rows()    == 6UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( m.columns() == 6UL, "Invalid number of columns detected" );

   typedef ElementType_<MT>  ET;

   const StaticMatrix<ET,6UL,6UL,SO> A( m.matrix_ );
   MT& B( m.matrix_ );

   ET tmp1 ( A(4,4)*A(5,5) - A(4,5)*A(5,4) );
   ET tmp2 ( A(4,3)*A(5,5) - A(4,5)*A(5,3) );
   ET tmp3 ( A(4,3)*A(5,4) - A(4,4)*A(5,3) );
   ET tmp4 ( A(4,2)*A(5,5) - A(4,5)*A(5,2) );
   ET tmp5 ( A(4,2)*A(5,4) - A(4,4)*A(5,2) );
   ET tmp6 ( A(4,2)*A(5,3) - A(4,3)*A(5,2) );
   ET tmp7 ( A(4,1)*A(5,5) - A(4,5)*A(5,1) );
   ET tmp8 ( A(4,1)*A(5,4) - A(4,4)*A(5,1) );
   ET tmp9 ( A(4,1)*A(5,3) - A(4,3)*A(5,1) );
   ET tmp10( A(4,1)*A(5,2) - A(4,2)*A(5,1) );
   ET tmp11( A(4,0)*A(5,5) - A(4,5)*A(5,0) );
   ET tmp12( A(4,0)*A(5,4) - A(4,4)*A(5,0) );
   ET tmp13( A(4,0)*A(5,3) - A(4,3)*A(5,0) );
   ET tmp14( A(4,0)*A(5,2) - A(4,2)*A(5,0) );
   ET tmp15( A(4,0)*A(5,1) - A(4,1)*A(5,0) );

   ET tmp16( A(3,3)*tmp1  - A(3,4)*tmp2  + A(3,5)*tmp3  );
   ET tmp17( A(3,2)*tmp1  - A(3,4)*tmp4  + A(3,5)*tmp5  );
   ET tmp18( A(3,2)*tmp2  - A(3,3)*tmp4  + A(3,5)*tmp6  );
   ET tmp19( A(3,2)*tmp3  - A(3,3)*tmp5  + A(3,4)*tmp6  );
   ET tmp20( A(3,1)*tmp1  - A(3,4)*tmp7  + A(3,5)*tmp8  );
   ET tmp21( A(3,1)*tmp2  - A(3,3)*tmp7  + A(3,5)*tmp9  );
   ET tmp22( A(3,1)*tmp3  - A(3,3)*tmp8  + A(3,4)*tmp9  );
   ET tmp23( A(3,1)*tmp4  - A(3,2)*tmp7  + A(3,5)*tmp10 );
   ET tmp24( A(3,1)*tmp5  - A(3,2)*tmp8  + A(3,4)*tmp10 );
   ET tmp25( A(3,1)*tmp6  - A(3,2)*tmp9  + A(3,3)*tmp10 );
   ET tmp26( A(3,0)*tmp1  - A(3,4)*tmp11 + A(3,5)*tmp12 );
   ET tmp27( A(3,0)*tmp2  - A(3,3)*tmp11 + A(3,5)*tmp13 );
   ET tmp28( A(3,0)*tmp3  - A(3,3)*tmp12 + A(3,4)*tmp13 );
   ET tmp29( A(3,0)*tmp4  - A(3,2)*tmp11 + A(3,5)*tmp14 );
   ET tmp30( A(3,0)*tmp5  - A(3,2)*tmp12 + A(3,4)*tmp14 );
   ET tmp31( A(3,0)*tmp6  - A(3,2)*tmp13 + A(3,3)*tmp14 );
   ET tmp32( A(3,0)*tmp7  - A(3,1)*tmp11 + A(3,5)*tmp15 );
   ET tmp33( A(3,0)*tmp8  - A(3,1)*tmp12 + A(3,4)*tmp15 );
   ET tmp34( A(3,0)*tmp9  - A(3,1)*tmp13 + A(3,3)*tmp15 );
   ET tmp35( A(3,0)*tmp10 - A(3,1)*tmp14 + A(3,2)*tmp15 );

   ET tmp36( A(2,2)*tmp16 - A(2,3)*tmp17 + A(2,4)*tmp18 - A(2,5)*tmp19 );
   ET tmp37( A(2,1)*tmp16 - A(2,3)*tmp20 + A(2,4)*tmp21 - A(2,5)*tmp22 );
   ET tmp38( A(2,1)*tmp17 - A(2,2)*tmp20 + A(2,4)*tmp23 - A(2,5)*tmp24 );
   ET tmp39( A(2,1)*tmp18 - A(2,2)*tmp21 + A(2,3)*tmp23 - A(2,5)*tmp25 );
   ET tmp40( A(2,1)*tmp19 - A(2,2)*tmp22 + A(2,3)*tmp24 - A(2,4)*tmp25 );
   ET tmp41( A(2,0)*tmp16 - A(2,3)*tmp26 + A(2,4)*tmp27 - A(2,5)*tmp28 );
   ET tmp42( A(2,0)*tmp17 - A(2,2)*tmp26 + A(2,4)*tmp29 - A(2,5)*tmp30 );
   ET tmp43( A(2,0)*tmp18 - A(2,2)*tmp27 + A(2,3)*tmp29 - A(2,5)*tmp31 );
   ET tmp44( A(2,0)*tmp19 - A(2,2)*tmp28 + A(2,3)*tmp30 - A(2,4)*tmp31 );

   B(0,0) =   A(1,1)*tmp36 - A(1,2)*tmp37 + A(1,3)*tmp38 - A(1,4)*tmp39 + A(1,5)*tmp40;
   B(0,1) = - A(0,1)*tmp36 + A(0,2)*tmp37 - A(0,3)*tmp38 + A(0,4)*tmp39 - A(0,5)*tmp40;
   B(1,1) =   A(0,0)*tmp36 - A(0,2)*tmp41 + A(0,3)*tmp42 - A(0,4)*tmp43 + A(0,5)*tmp44;

   ET tmp45( A(2,0)*tmp20 - A(2,1)*tmp26 + A(2,4)*tmp32 - A(2,5)*tmp33 );
   ET tmp46( A(2,0)*tmp21 - A(2,1)*tmp27 + A(2,3)*tmp32 - A(2,5)*tmp34 );
   ET tmp47( A(2,0)*tmp22 - A(2,1)*tmp28 + A(2,3)*tmp33 - A(2,4)*tmp34 );
   ET tmp48( A(2,0)*tmp23 - A(2,1)*tmp29 + A(2,2)*tmp32 - A(2,5)*tmp35 );
   ET tmp49( A(2,0)*tmp24 - A(2,1)*tmp30 + A(2,2)*tmp33 - A(2,4)*tmp35 );

   B(2,0) =   A(1,0)*tmp37 - A(1,1)*tmp41 + A(1,3)*tmp45 - A(1,4)*tmp46 + A(1,5)*tmp47;
   B(2,1) = - A(0,0)*tmp37 + A(0,1)*tmp41 - A(0,3)*tmp45 + A(0,4)*tmp46 - A(0,5)*tmp47;
   B(3,0) = - A(1,0)*tmp38 + A(1,1)*tmp42 - A(1,2)*tmp45 + A(1,4)*tmp48 - A(1,5)*tmp49;
   B(3,1) =   A(0,0)*tmp38 - A(0,1)*tmp42 + A(0,2)*tmp45 - A(0,4)*tmp48 + A(0,5)*tmp49;

   ET tmp50( A(2,0)*tmp25 - A(2,1)*tmp31 + A(2,2)*tmp34 - A(2,3)*tmp35 );

   B(4,0) =   A(1,0)*tmp39 - A(1,1)*tmp43 + A(1,2)*tmp46 - A(1,3)*tmp48 + A(1,5)*tmp50;
   B(4,1) = - A(0,0)*tmp39 + A(0,1)*tmp43 - A(0,2)*tmp46 + A(0,3)*tmp48 - A(0,5)*tmp50;
   B(5,0) = - A(1,0)*tmp40 + A(1,1)*tmp44 - A(1,2)*tmp47 + A(1,3)*tmp49 - A(1,4)*tmp50;
   B(5,1) =   A(0,0)*tmp40 - A(0,1)*tmp44 + A(0,2)*tmp47 - A(0,3)*tmp49 + A(0,4)*tmp50;

   tmp36 = A(1,1)*tmp16 - A(1,3)*tmp20 + A(1,4)*tmp21 - A(1,5)*tmp22;
   tmp37 = A(1,1)*tmp17 - A(1,2)*tmp20 + A(1,4)*tmp23 - A(1,5)*tmp24;
   tmp38 = A(1,0)*tmp16 - A(1,3)*tmp26 + A(1,4)*tmp27 - A(1,5)*tmp28;
   tmp39 = A(1,0)*tmp17 - A(1,2)*tmp26 + A(1,4)*tmp29 - A(1,5)*tmp30;
   tmp40 = A(1,0)*tmp20 - A(1,1)*tmp26 + A(1,4)*tmp32 - A(1,5)*tmp33;
   tmp41 = A(1,0)*tmp21 - A(1,1)*tmp27 + A(1,3)*tmp32 - A(1,5)*tmp34;
   tmp42 = A(1,0)*tmp22 - A(1,1)*tmp28 + A(1,3)*tmp33 - A(1,4)*tmp34;
   tmp43 = A(1,0)*tmp23 - A(1,1)*tmp29 + A(1,2)*tmp32 - A(1,5)*tmp35;
   tmp44 = A(1,0)*tmp24 - A(1,1)*tmp30 + A(1,2)*tmp33 - A(1,4)*tmp35;

   B(2,2) =   A(0,0)*tmp36 - A(0,1)*tmp38 + A(0,3)*tmp40 - A(0,4)*tmp41 + A(0,5)*tmp42;
   B(3,2) = - A(0,0)*tmp37 + A(0,1)*tmp39 - A(0,2)*tmp40 + A(0,4)*tmp43 - A(0,5)*tmp44;

   tmp1  = A(0,3)*A(1,4) - A(0,4)*A(1,3);
   tmp2  = A(0,2)*A(1,4) - A(0,4)*A(1,2);
   tmp3  = A(0,2)*A(1,3) - A(0,3)*A(1,2);
   tmp4  = A(0,1)*A(1,4) - A(0,4)*A(1,1);
   tmp5  = A(0,1)*A(1,3) - A(0,3)*A(1,1);
   tmp6  = A(0,1)*A(1,2) - A(0,2)*A(1,1);
   tmp7  = A(0,0)*A(1,4) - A(0,4)*A(1,0);
   tmp8  = A(0,0)*A(1,3) - A(0,3)*A(1,0);
   tmp9  = A(0,0)*A(1,2) - A(0,2)*A(1,0);
   tmp10 = A(0,0)*A(1,1) - A(0,1)*A(1,0);
   tmp11 = A(0,3)*A(1,5) - A(0,5)*A(1,3);
   tmp12 = A(0,2)*A(1,5) - A(0,5)*A(1,2);
   tmp13 = A(0,1)*A(1,5) - A(0,5)*A(1,1);
   tmp14 = A(0,0)*A(1,5) - A(0,5)*A(1,0);
   tmp15 = A(0,4)*A(1,5) - A(0,5)*A(1,4);

   tmp16 = A(2,3)*tmp15 - A(2,4)*tmp11 + A(2,5)*tmp1;
   tmp17 = A(2,2)*tmp15 - A(2,4)*tmp12 + A(2,5)*tmp2;
   tmp18 = A(2,2)*tmp11 - A(2,3)*tmp12 + A(2,5)*tmp3;
   tmp19 = A(2,2)*tmp1  - A(2,3)*tmp2  + A(2,4)*tmp3;
   tmp20 = A(2,1)*tmp15 - A(2,4)*tmp13 + A(2,5)*tmp4;
   tmp21 = A(2,1)*tmp11 - A(2,3)*tmp13 + A(2,5)*tmp5;
   tmp22 = A(2,1)*tmp1  - A(2,3)*tmp4  + A(2,4)*tmp5;
   tmp23 = A(2,1)*tmp12 - A(2,2)*tmp13 + A(2,5)*tmp6;
   tmp24 = A(2,1)*tmp2  - A(2,2)*tmp4  + A(2,4)*tmp6;
   tmp25 = A(2,1)*tmp3  - A(2,2)*tmp5  + A(2,3)*tmp6;
   tmp26 = A(2,0)*tmp15 - A(2,4)*tmp14 + A(2,5)*tmp7;
   tmp27 = A(2,0)*tmp11 - A(2,3)*tmp14 + A(2,5)*tmp8;
   tmp28 = A(2,0)*tmp1  - A(2,3)*tmp7  + A(2,4)*tmp8;
   tmp29 = A(2,0)*tmp12 - A(2,2)*tmp14 + A(2,5)*tmp9;
   tmp30 = A(2,0)*tmp2  - A(2,2)*tmp7  + A(2,4)*tmp9;
   tmp31 = A(2,0)*tmp3  - A(2,2)*tmp8  + A(2,3)*tmp9;
   tmp32 = A(2,0)*tmp13 - A(2,1)*tmp14 + A(2,5)*tmp10;
   tmp33 = A(2,0)*tmp4  - A(2,1)*tmp7  + A(2,4)*tmp10;
   tmp34 = A(2,0)*tmp5  - A(2,1)*tmp8  + A(2,3)*tmp10;
   tmp35 = A(2,0)*tmp6  - A(2,1)*tmp9  + A(2,2)*tmp10;

   tmp36 = A(3,1)*tmp16 - A(3,3)*tmp20 + A(3,4)*tmp21 - A(3,5)*tmp22;
   tmp37 = A(3,1)*tmp17 - A(3,2)*tmp20 + A(3,4)*tmp23 - A(3,5)*tmp24;
   tmp38 = A(3,0)*tmp16 - A(3,3)*tmp26 + A(3,4)*tmp27 - A(3,5)*tmp28;
   tmp39 = A(3,0)*tmp17 - A(3,2)*tmp26 + A(3,4)*tmp29 - A(3,5)*tmp30;
   tmp40 = A(3,0)*tmp20 - A(3,1)*tmp26 + A(3,4)*tmp32 - A(3,5)*tmp33;
   tmp41 = A(3,0)*tmp21 - A(3,1)*tmp27 + A(3,3)*tmp32 - A(3,5)*tmp34;
   tmp42 = A(3,0)*tmp22 - A(3,1)*tmp28 + A(3,3)*tmp33 - A(3,4)*tmp34;
   tmp43 = A(3,0)*tmp23 - A(3,1)*tmp29 + A(3,2)*tmp32 - A(3,5)*tmp35;
   tmp44 = A(3,0)*tmp24 - A(3,1)*tmp30 + A(3,2)*tmp33 - A(3,4)*tmp35;

   B(2,4) = - A(5,0)*tmp36 + A(5,1)*tmp38 - A(5,3)*tmp40 + A(5,4)*tmp41 - A(5,5)*tmp42;
   B(2,5) =   A(4,0)*tmp36 - A(4,1)*tmp38 + A(4,3)*tmp40 - A(4,4)*tmp41 + A(4,5)*tmp42;
   B(3,4) =   A(5,0)*tmp37 - A(5,1)*tmp39 + A(5,2)*tmp40 - A(5,4)*tmp43 + A(5,5)*tmp44;
   B(3,5) = - A(4,0)*tmp37 + A(4,1)*tmp39 - A(4,2)*tmp40 + A(4,4)*tmp43 - A(4,5)*tmp44;

   tmp36 = A(3,1)*tmp18 - A(3,2)*tmp21 + A(3,3)*tmp23 - A(3,5)*tmp25;
   tmp37 = A(3,1)*tmp19 - A(3,2)*tmp22 + A(3,3)*tmp24 - A(3,4)*tmp25;
   tmp38 = A(3,0)*tmp18 - A(3,2)*tmp27 + A(3,3)*tmp29 - A(3,5)*tmp31;
   tmp39 = A(3,0)*tmp19 - A(3,2)*tmp28 + A(3,3)*tmp30 - A(3,4)*tmp31;
   tmp40 = A(3,0)*tmp25 - A(3,1)*tmp31 + A(3,2)*tmp34 - A(3,3)*tmp35;

   B(4,4) = - A(5,0)*tmp36 + A(5,1)*tmp38 - A(5,2)*tmp41 + A(5,3)*tmp43 - A(5,5)*tmp40;
   B(4,5) =   A(4,0)*tmp36 - A(4,1)*tmp38 + A(4,2)*tmp41 - A(4,3)*tmp43 + A(4,5)*tmp40;
   B(5,5) = - A(4,0)*tmp37 + A(4,1)*tmp39 - A(4,2)*tmp42 + A(4,3)*tmp44 - A(4,4)*tmp40;

   tmp36 = A(4,1)*tmp17 - A(4,2)*tmp20 + A(4,4)*tmp23 - A(4,5)*tmp24;
   tmp37 = A(4,0)*tmp17 - A(4,2)*tmp26 + A(4,4)*tmp29 - A(4,5)*tmp30;
   tmp38 = A(4,0)*tmp20 - A(4,1)*tmp26 + A(4,4)*tmp32 - A(4,5)*tmp33;
   tmp39 = A(4,0)*tmp23 - A(4,1)*tmp29 + A(4,2)*tmp32 - A(4,5)*tmp35;
   tmp40 = A(4,0)*tmp24 - A(4,1)*tmp30 + A(4,2)*tmp33 - A(4,4)*tmp35;

   B(3,3) = - A(5,0)*tmp36 + A(5,1)*tmp37 - A(5,2)*tmp38 + A(5,4)*tmp39 - A(5,5)*tmp40;

   B(0,2) = B(2,0);
   B(0,3) = B(3,0);
   B(0,4) = B(4,0);
   B(0,5) = B(5,0);
   B(1,0) = B(0,1);
   B(1,2) = B(2,1);
   B(1,3) = B(3,1);
   B(1,4) = B(4,1);
   B(1,5) = B(5,1);
   B(2,3) = B(3,2);
   B(4,2) = B(2,4);
   B(4,3) = B(3,4);
   B(5,2) = B(2,5);
   B(5,3) = B(3,5);
   B(5,4) = B(4,5);

   const ET det( A(0,0)*B(0,0) + A(0,1)*B(1,0) + A(0,2)*B(2,0) +
                 A(0,3)*B(3,0) + A(0,4)*B(4,0) + A(0,5)*B(5,0) );

   if( !isDivisor( det ) ) {
      BLAZE_THROW_DIVISION_BY_ZERO( "Inversion of singular matrix failed" );
   }

   B /= det;

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given symmetric dense matrix.
// \ingroup symmetric_matrix
//
// \param m The symmetric dense matrix to be inverted.
// \return void
// \exception std::invalid_argument Inversion of singular matrix failed.
//
// This function inverts the given symmetric dense matrix by means of the most suited matrix
// inversion algorithm. The matrix inversion fails if the given matrix is singular and not
// invertible. In this case a \a std::invalid_argument exception is thrown.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a linker error will be created.
//
// \note This function does only provide the basic exception safety guarantee, i.e. in case of an
// exception \a m may already have been modified.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invertByDefault( SymmetricMatrix<MT,SO,true,true>& m )
{
   invertByLDLT( m );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place LU-based inversion of the given symmetric dense matrix.
// \ingroup symmetric_matrix
//
// \param m The symmetric dense matrix to be inverted.
// \return void
// \exception std::invalid_argument Inversion of singular matrix failed.
//
// This function inverts the given symmetric dense matrix by means of an LU decomposition.
// The inversion fails if the given matrix is singular and not invertible. In this case a
// \a std::invalid_argument exception is thrown.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a linker error will be created.
//
// \note This function does only provide the basic exception safety guarantee, i.e. in case of an
// exception \a m may already have been modified.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invertByLU( SymmetricMatrix<MT,SO,true,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   MT tmp( m.matrix_ );
   invertByLU( tmp );
   m.matrix_ = std::move( tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place Bunch-Kaufman-based inversion of the given symmetric dense matrix.
// \ingroup symmetric_matrix
//
// \param m The symmetric dense matrix to be inverted.
// \return void
// \exception std::invalid_argument Inversion of singular matrix failed.
//
// This function inverts the given symmetric dense matrix by means of a Bunch-Kaufman-based
// decomposition. The inversion fails if the given matrix is singular and not invertible. In
// this case a \a std::invalid_argument exception is thrown.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a linker error will be created.
//
// \note This function does only provide the basic exception safety guarantee, i.e. in case of an
// exception \a m may already have been modified.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invertByLDLT( SymmetricMatrix<MT,SO,true,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   MT tmp( m.matrix_ );
   invertByLDLT( tmp );
   m.matrix_ = std::move( tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place Bunch-Kaufman-based inversion of the given symmetric dense matrix.
// \ingroup symmetric_matrix
//
// \param m The symmetric dense matrix to be inverted.
// \return void
// \exception std::invalid_argument Inversion of singular matrix failed.
//
// This function inverts the given symmetric dense matrix by means of a Bunch-Kaufman-based
// decomposition. The inversion fails if the given matrix is singular and not invertible. In
// this case a \a std::invalid_argument exception is thrown.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a linker error will be created.
//
// \note This function does only provide the basic exception safety guarantee, i.e. in case of an
// exception \a m may already have been modified.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invertByLDLH( SymmetricMatrix<MT,SO,true,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   MT tmp( m.matrix_ );
   invertByLDLH( tmp );
   m.matrix_ = std::move( tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place Cholesky-based inversion of the given symmetric dense matrix.
// \ingroup symmetric_matrix
//
// \param m The symmetric dense matrix to be inverted.
// \return void
// \exception std::invalid_argument Inversion of singular matrix failed.
//
// This function inverts the given symmetric dense matrix by means of a Cholesky-based
// decomposition. The inversion fails if the given matrix is singular and not invertible. In
// this case a \a std::invalid_argument exception is thrown.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a linker error will be created.
//
// \note This function does only provide the basic exception safety guarantee, i.e. in case of an
// exception \a m may already have been modified.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invertByLLH( SymmetricMatrix<MT,SO,true,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   MT tmp( m.matrix_ );
   invertByLLH( tmp );
   m.matrix_ = std::move( tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a matrix to a symmetric matrix.
// \ingroup symmetric_matrix
//
// \param lhs The target left-hand side symmetric matrix.
// \param rhs The right-hand side matrix to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the adapted matrix
        , bool SO1      // Storage order of the adapted matrix
        , bool DF       // Density flag
        , bool NF       // Numeric flag
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline bool tryAssign( const SymmetricMatrix<MT1,SO1,DF,NF>& lhs,
                       const Matrix<MT2,SO2>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT2 );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= lhs.rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   const size_t M( (~rhs).rows()    );
   const size_t N( (~rhs).columns() );

   if( ( row + M <= column ) || ( column + N <= row ) )
      return true;

   const bool   lower( row > column );
   const size_t size ( min( row + M, column + N ) - ( lower ? row : column ) );

   if( size < 2UL )
      return true;

   const size_t subrow( lower ? 0UL : column - row );
   const size_t subcol( lower ? row - column : 0UL );

   return isSymmetric( submatrix( ~rhs, subrow, subcol, size, size ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a matrix to a symmetric matrix.
// \ingroup symmetric_matrix
//
// \param lhs The target left-hand side symmetric matrix.
// \param rhs The right-hand side matrix to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the adapted matrix
        , bool SO1      // Storage order of the adapted matrix
        , bool DF       // Density flag
        , bool NF       // Numeric flag
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline bool tryAddAssign( const SymmetricMatrix<MT1,SO1,DF,NF>& lhs,
                          const Matrix<MT2,SO2>& rhs, size_t row, size_t column )
{
   return tryAssign( lhs, ~rhs, row, column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a matrix to a symmetric
//        matrix.
// \ingroup symmetric_matrix
//
// \param lhs The target left-hand side symmetric matrix.
// \param rhs The right-hand side matrix to be subtracted.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the adapted matrix
        , bool SO1      // Storage order of the adapted matrix
        , bool DF       // Density flag
        , bool NF       // Numeric flag
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline bool trySubAssign( const SymmetricMatrix<MT1,SO1,DF,NF>& lhs,
                          const Matrix<MT2,SO2>& rhs, size_t row, size_t column )
{
   return tryAssign( lhs, ~rhs, row, column );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ROWS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct Rows< SymmetricMatrix<MT,SO,DF,NF> > : public Rows<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COLUMNS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct Columns< SymmetricMatrix<MT,SO,DF,NF> > : public Columns<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSQUARE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct IsSquare< SymmetricMatrix<MT,SO,DF,NF> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSYMMETRIC SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct IsSymmetric< SymmetricMatrix<MT,SO,DF,NF> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISHERMITIAN SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct IsHermitian< SymmetricMatrix<MT,SO,DF,NF> >
   : public BoolConstant< IsBuiltin< ElementType_<MT> >::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISADAPTOR SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct IsAdaptor< SymmetricMatrix<MT,SO,DF,NF> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISRESTRICTED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct IsRestricted< SymmetricMatrix<MT,SO,DF,NF> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HASCONSTDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool NF >
struct HasConstDataAccess< SymmetricMatrix<MT,SO,true,NF> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISALIGNED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct IsAligned< SymmetricMatrix<MT,SO,DF,NF> > : public BoolConstant< IsAligned<MT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISPADDED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct IsPadded< SymmetricMatrix<MT,SO,DF,NF> > : public BoolConstant< IsPadded<MT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISRESIZABLE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct IsResizable< SymmetricMatrix<MT,SO,DF,NF> > : public BoolConstant< IsResizable<MT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  REMOVEADAPTOR SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct RemoveAdaptor< SymmetricMatrix<MT,SO,DF,NF> >
{
   using Type = MT;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ADDTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO1, bool DF, bool NF, typename T, size_t M, size_t N, bool SO2 >
struct AddTrait< SymmetricMatrix<MT,SO1,DF,NF>, StaticMatrix<T,M,N,SO2> >
{
   using Type = AddTrait_< MT, StaticMatrix<T,M,N,SO2> >;
};

template< typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct AddTrait< StaticMatrix<T,M,N,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = AddTrait_< StaticMatrix<T,M,N,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, bool NF, typename T, size_t M, size_t N, bool SO2 >
struct AddTrait< SymmetricMatrix<MT,SO1,DF,NF>, HybridMatrix<T,M,N,SO2> >
{
   using Type = AddTrait_< MT, HybridMatrix<T,M,N,SO2> >;
};

template< typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct AddTrait< HybridMatrix<T,M,N,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = AddTrait_< HybridMatrix<T,M,N,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, bool NF, typename T, bool SO2 >
struct AddTrait< SymmetricMatrix<MT,SO1,DF,NF>, DynamicMatrix<T,SO2> >
{
   using Type = AddTrait_< MT, DynamicMatrix<T,SO2> >;
};

template< typename T, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct AddTrait< DynamicMatrix<T,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = AddTrait_< DynamicMatrix<T,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, bool NF, typename T, bool AF, bool PF, bool SO2 >
struct AddTrait< SymmetricMatrix<MT,SO1,DF,NF>, CustomMatrix<T,AF,PF,SO2> >
{
   using Type = AddTrait_< MT, CustomMatrix<T,AF,PF,SO2> >;
};

template< typename T, bool AF, bool PF, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct AddTrait< CustomMatrix<T,AF,PF,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = AddTrait_< CustomMatrix<T,AF,PF,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, bool NF, typename T, bool SO2 >
struct AddTrait< SymmetricMatrix<MT,SO1,DF,NF>, CompressedMatrix<T,SO2> >
{
   using Type = AddTrait_< MT, CompressedMatrix<T,SO2> >;
};

template< typename T, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct AddTrait< CompressedMatrix<T,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = AddTrait_< CompressedMatrix<T,SO1>, MT >;
};

template< typename MT1, bool SO1, bool DF1, bool NF1, typename MT2, bool SO2, bool DF2, bool NF2 >
struct AddTrait< SymmetricMatrix<MT1,SO1,DF1,NF1>, SymmetricMatrix<MT2,SO2,DF2,NF2> >
{
   using Type = SymmetricMatrix< AddTrait_<MT1,MT2> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO1, bool DF, bool NF, typename T, size_t M, size_t N, bool SO2 >
struct SubTrait< SymmetricMatrix<MT,SO1,DF,NF>, StaticMatrix<T,M,N,SO2> >
{
   using Type = SubTrait_< MT, StaticMatrix<T,M,N,SO2> >;
};

template< typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct SubTrait< StaticMatrix<T,M,N,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = SubTrait_< StaticMatrix<T,M,N,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, bool NF, typename T, size_t M, size_t N, bool SO2 >
struct SubTrait< SymmetricMatrix<MT,SO1,DF,NF>, HybridMatrix<T,M,N,SO2> >
{
   using Type = SubTrait_< MT, HybridMatrix<T,M,N,SO2> >;
};

template< typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct SubTrait< HybridMatrix<T,M,N,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = SubTrait_< HybridMatrix<T,M,N,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, bool NF, typename T, bool SO2 >
struct SubTrait< SymmetricMatrix<MT,SO1,DF,NF>, DynamicMatrix<T,SO2> >
{
   using Type = SubTrait_< MT, DynamicMatrix<T,SO2> >;
};

template< typename T, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct SubTrait< DynamicMatrix<T,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = SubTrait_< DynamicMatrix<T,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, bool NF, typename T, bool AF, bool PF, bool SO2 >
struct SubTrait< SymmetricMatrix<MT,SO1,DF,NF>, CustomMatrix<T,AF,PF,SO2> >
{
   using Type = SubTrait_< MT, CustomMatrix<T,AF,PF,SO2> >;
};

template< typename T, bool AF, bool PF, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct SubTrait< CustomMatrix<T,AF,PF,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = SubTrait_< CustomMatrix<T,AF,PF,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, bool NF, typename T, bool SO2 >
struct SubTrait< SymmetricMatrix<MT,SO1,DF,NF>, CompressedMatrix<T,SO2> >
{
   using Type = SubTrait_< MT, CompressedMatrix<T,SO2> >;
};

template< typename T, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct SubTrait< CompressedMatrix<T,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = SubTrait_< CompressedMatrix<T,SO1>, MT >;
};

template< typename MT1, bool SO1, bool DF1, bool NF1, typename MT2, bool SO2, bool DF2, bool NF2 >
struct SubTrait< SymmetricMatrix<MT1,SO1,DF1,NF1>, SymmetricMatrix<MT2,SO2,DF2,NF2> >
{
   using Type = SymmetricMatrix< SubTrait_<MT1,MT2> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MULTTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF, typename T >
struct MultTrait< SymmetricMatrix<MT,SO,DF,NF>, T, EnableIf_< IsNumeric<T> > >
{
   using Type = SymmetricMatrix< MultTrait_<MT,T> >;
};

template< typename T, typename MT, bool SO, bool DF, bool NF >
struct MultTrait< T, SymmetricMatrix<MT,SO,DF,NF>, EnableIf_< IsNumeric<T> > >
{
   using Type = SymmetricMatrix< MultTrait_<T,MT> >;
};

template< typename MT, bool SO, bool DF, bool NF, typename T, size_t N >
struct MultTrait< SymmetricMatrix<MT,SO,DF,NF>, StaticVector<T,N,false> >
{
   using Type = MultTrait_< MT, StaticVector<T,N,false> >;
};

template< typename T, size_t N, typename MT, bool SO, bool DF, bool NF >
struct MultTrait< StaticVector<T,N,true>, SymmetricMatrix<MT,SO,DF,NF> >
{
   using Type = MultTrait_< StaticVector<T,N,true>, MT >;
};

template< typename MT, bool SO, bool DF, bool NF, typename T, size_t N >
struct MultTrait< SymmetricMatrix<MT,SO,DF,NF>, HybridVector<T,N,false> >
{
   using Type = MultTrait_< MT, HybridVector<T,N,false> >;
};

template< typename T, size_t N, typename MT, bool SO, bool DF, bool NF >
struct MultTrait< HybridVector<T,N,true>, SymmetricMatrix<MT,SO,DF,NF> >
{
   using Type = MultTrait_< HybridVector<T,N,true>, MT >;
};

template< typename MT, bool SO, bool DF, bool NF, typename T >
struct MultTrait< SymmetricMatrix<MT,SO,DF,NF>, DynamicVector<T,false> >
{
   using Type = MultTrait_< MT, DynamicVector<T,false> >;
};

template< typename T, typename MT, bool SO, bool DF, bool NF >
struct MultTrait< DynamicVector<T,true>, SymmetricMatrix<MT,SO,DF,NF> >
{
   using Type = MultTrait_< DynamicVector<T,true>, MT >;
};

template< typename MT, bool SO, bool DF, bool NF, typename T, bool AF, bool PF >
struct MultTrait< SymmetricMatrix<MT,SO,DF,NF>, CustomVector<T,AF,PF,false> >
{
   using Type = MultTrait_< MT, CustomVector<T,AF,PF,false> >;
};

template< typename T, bool AF, bool PF, typename MT, bool SO, bool DF, bool NF >
struct MultTrait< CustomVector<T,AF,PF,true>, SymmetricMatrix<MT,SO,DF,NF> >
{
   using Type = MultTrait_< CustomVector<T,AF,PF,true>, MT >;
};

template< typename MT, bool SO, bool DF, bool NF, typename T >
struct MultTrait< SymmetricMatrix<MT,SO,DF,NF>, CompressedVector<T,false> >
{
   using Type = MultTrait_< MT, CompressedVector<T,false> >;
};

template< typename T, typename MT, bool SO, bool DF, bool NF >
struct MultTrait< CompressedVector<T,true>, SymmetricMatrix<MT,SO,DF,NF> >
{
   using Type = MultTrait_< CompressedVector<T,true>, MT >;
};

template< typename MT, bool SO1, bool DF, bool NF, typename T, size_t M, size_t N, bool SO2 >
struct MultTrait< SymmetricMatrix<MT,SO1,DF,NF>, StaticMatrix<T,M,N,SO2> >
{
   using Type = MultTrait_< MT, StaticMatrix<T,M,N,SO2> >;
};

template< typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct MultTrait< StaticMatrix<T,M,N,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = MultTrait_< StaticMatrix<T,M,N,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, bool NF, typename T, size_t M, size_t N, bool SO2 >
struct MultTrait< SymmetricMatrix<MT,SO1,DF,NF>, HybridMatrix<T,M,N,SO2> >
{
   using Type = MultTrait_< MT, HybridMatrix<T,M,N,SO2> >;
};

template< typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct MultTrait< HybridMatrix<T,M,N,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = MultTrait_< HybridMatrix<T,M,N,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, bool NF, typename T, bool SO2 >
struct MultTrait< SymmetricMatrix<MT,SO1,DF,NF>, DynamicMatrix<T,SO2> >
{
   using Type = MultTrait_< MT, DynamicMatrix<T,SO2> >;
};

template< typename T, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct MultTrait< DynamicMatrix<T,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = MultTrait_< DynamicMatrix<T,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, bool NF, typename T, bool AF, bool PF, bool SO2 >
struct MultTrait< SymmetricMatrix<MT,SO1,DF,NF>, CustomMatrix<T,AF,PF,SO2> >
{
   using Type = MultTrait_< MT, CustomMatrix<T,AF,PF,SO2> >;
};

template< typename T, bool AF, bool PF, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct MultTrait< CustomMatrix<T,AF,PF,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = MultTrait_< DynamicMatrix<T,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, bool NF, typename T, bool SO2 >
struct MultTrait< SymmetricMatrix<MT,SO1,DF,NF>, CompressedMatrix<T,SO2> >
{
   using Type = MultTrait_< MT, CompressedMatrix<T,SO2> >;
};

template< typename T, bool SO1, typename MT, bool SO2, bool DF, bool NF >
struct MultTrait< CompressedMatrix<T,SO1>, SymmetricMatrix<MT,SO2,DF,NF> >
{
   using Type = MultTrait_< CompressedMatrix<T,SO1>, MT >;
};

template< typename MT1, bool SO1, bool DF1, bool NF1, typename MT2, bool SO2, bool DF2, bool NF2 >
struct MultTrait< SymmetricMatrix<MT1,SO1,DF1,NF1>, SymmetricMatrix<MT2,SO2,DF2,NF2> >
{
   using Type = MultTrait_<MT1,MT2>;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DIVTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF, typename T >
struct DivTrait< SymmetricMatrix<MT,SO,DF,NF>, T, EnableIf_< IsNumeric<T> > >
{
   using Type = SymmetricMatrix< DivTrait_<MT,T> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  FOREACHTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Abs >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Abs> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Floor >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Floor> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Ceil >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Ceil> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Conj >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Conj> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Real >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Real> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Imag >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Imag> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Sqrt >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,InvSqrt> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Cbrt >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Cbrt> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, InvCbrt >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,InvCbrt> >;
};

template< typename MT, bool SO, bool DF, bool NF, typename ET >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Pow<ET> >
{
   using Type = SymmetricMatrix< ForEachTrait_< MT, Pow<ET> > >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Exp >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Exp> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Log >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Log> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Log10 >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Log10> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Sin >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Sin> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Asin >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Asin> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Sinh >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Sinh> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Asinh >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Asinh> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Cos >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Cos> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Acos >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Acos> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Cosh >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Cosh> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Acosh >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Acosh> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Tan >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Tan> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Atan >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Atan> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Tanh >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Tanh> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Atanh >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Atanh> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Erf >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Erf> >;
};

template< typename MT, bool SO, bool DF, bool NF >
struct ForEachTrait< SymmetricMatrix<MT,SO,DF,NF>, Erfc >
{
   using Type = SymmetricMatrix< ForEachTrait_<MT,Erfc> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MATHTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, bool SO1, bool DF1, bool NF1, typename MT2, bool SO2, bool DF2, bool NF2 >
struct MathTrait< SymmetricMatrix<MT1,SO1,DF1,NF1>, SymmetricMatrix<MT2,SO2,DF2,NF2> >
{
   typedef SymmetricMatrix< typename MathTrait<MT1,MT2>::HighType >  HighType;
   typedef SymmetricMatrix< typename MathTrait<MT1,MT2>::LowType  >  LowType;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBMATRIXTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct SubmatrixTrait< SymmetricMatrix<MT,SO,DF,NF> >
{
   using Type = SubmatrixTrait_<MT>;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ROWTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct RowTrait< SymmetricMatrix<MT,SO,DF,NF> >
{
   using Type = RowTrait_<MT>;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COLUMNTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, bool NF >
struct ColumnTrait< SymmetricMatrix<MT,SO,DF,NF> >
{
   using Type = ColumnTrait_<MT>;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
