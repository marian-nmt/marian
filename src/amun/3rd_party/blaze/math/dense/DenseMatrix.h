//=================================================================================================
/*!
//  \file blaze/math/dense/DenseMatrix.h
//  \brief Header file for utility functions for dense matrices
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

#ifndef _BLAZE_MATH_DENSE_DENSEMATRIX_H_
#define _BLAZE_MATH_DENSE_DENSEMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/Triangular.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/Functions.h>
#include <blaze/math/shims/Conjugate.h>
#include <blaze/math/shims/Equal.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsNaN.h>
#include <blaze/math/shims/IsOne.h>
#include <blaze/math/shims/IsReal.h>
#include <blaze/math/shims/IsZero.h>
#include <blaze/math/StorageOrder.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsDiagonal.h>
#include <blaze/math/typetraits/IsHermitian.h>
#include <blaze/math/typetraits/IsIdentity.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsTriangular.h>
#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/math/typetraits/IsUniTriangular.h>
#include <blaze/math/typetraits/IsUniUpper.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FalseType.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/RemoveReference.h>


namespace blaze {

//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DenseMatrix operators */
//@{
template< typename T1, typename T2 >
inline bool operator==( const DenseMatrix<T1,false>& lhs, const DenseMatrix<T2,false>& rhs );

template< typename T1, typename T2 >
inline bool operator==( const DenseMatrix<T1,true>& lhs, const DenseMatrix<T2,true>& rhs );

template< typename T1, typename T2, bool SO >
inline bool operator==( const DenseMatrix<T1,SO>& lhs, const DenseMatrix<T2,!SO>& rhs );

template< typename T1, typename T2, bool SO >
inline bool operator==( const DenseMatrix<T1,SO>& lhs, const SparseMatrix<T2,false>& rhs );

template< typename T1, typename T2, bool SO >
inline bool operator==( const DenseMatrix<T1,SO>& lhs, const SparseMatrix<T2,true>& rhs );

template< typename T1, bool SO1, typename T2, bool SO2 >
inline bool operator==( const SparseMatrix<T1,SO1>& lhs, const DenseMatrix<T2,SO2>& rhs );

template< typename T1, typename T2 >
inline EnableIf_<IsNumeric<T2>, bool > operator==( const DenseMatrix<T1,false>& mat, T2 scalar );

template< typename T1, typename T2 >
inline EnableIf_<IsNumeric<T2>, bool > operator==( const DenseMatrix<T1,true>& mat, T2 scalar );

template< typename T1, typename T2, bool SO >
inline EnableIf_<IsNumeric<T2>, bool > operator==( T1 scalar, const DenseMatrix<T2,SO>& mat );

template< typename T1, bool SO1, typename T2, bool SO2 >
inline bool operator!=( const DenseMatrix<T1,SO1>& lhs, const DenseMatrix<T2,SO2>& rhs );

template< typename T1, bool SO1, typename T2, bool SO2 >
inline bool operator!=( const DenseMatrix<T1,SO1>& lhs, const SparseMatrix<T2,SO2>& rhs );

template< typename T1, bool SO1, typename T2, bool SO2 >
inline bool operator!=( const SparseMatrix<T1,SO1>& lhs, const DenseMatrix<T2,SO2>& rhs );

template< typename T1, typename T2, bool SO >
inline EnableIf_<IsNumeric<T2>, bool > operator!=( const DenseMatrix<T1,SO>& mat, T2 scalar );

template< typename T1, typename T2, bool SO >
inline EnableIf_<IsNumeric<T2>, bool > operator!=( T1 scalar, const DenseMatrix<T2,SO>& mat );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of two rwo-major dense matrices.
// \ingroup dense_matrix
//
// \param lhs The left-hand side matrix for the comparison.
// \param rhs The right-hand side matrix for the comparison.
// \return \a true if the two matrices are equal, \a false if not.
*/
template< typename T1    // Type of the left-hand side dense matrix
        , typename T2 >  // Type of the right-hand side dense matrix
inline bool operator==( const DenseMatrix<T1,false>& lhs, const DenseMatrix<T2,false>& rhs )
{
   typedef CompositeType_<T1>  CT1;
   typedef CompositeType_<T2>  CT2;

   // Early exit in case the matrix sizes don't match
   if( (~lhs).rows() != (~rhs).rows() || (~lhs).columns() != (~rhs).columns() )
      return false;

   // Evaluation of the two dense matrix operands
   CT1 A( ~lhs );
   CT2 B( ~rhs );

   // In order to compare the two matrices, the data values of the lower-order data
   // type are converted to the higher-order data type within the equal function.
   for( size_t i=0; i<A.rows(); ++i ) {
      for( size_t j=0; j<A.columns(); ++j ) {
         if( !equal( A(i,j), B(i,j) ) ) return false;
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of two column-major dense matrices.
// \ingroup dense_matrix
//
// \param lhs The left-hand side matrix for the comparison.
// \param rhs The right-hand side matrix for the comparison.
// \return \a true if the two matrices are equal, \a false if not.
*/
template< typename T1    // Type of the left-hand side dense matrix
        , typename T2 >  // Type of the right-hand side dense matrix
inline bool operator==( const DenseMatrix<T1,true>& lhs, const DenseMatrix<T2,true>& rhs )
{
   typedef CompositeType_<T1>  CT1;
   typedef CompositeType_<T2>  CT2;

   // Early exit in case the matrix sizes don't match
   if( (~lhs).rows() != (~rhs).rows() || (~lhs).columns() != (~rhs).columns() )
      return false;

   // Evaluation of the two dense matrix operands
   CT1 A( ~lhs );
   CT2 B( ~rhs );

   // In order to compare the two matrices, the data values of the lower-order data
   // type are converted to the higher-order data type within the equal function.
   for( size_t j=0; j<A.columns(); ++j ) {
      for( size_t i=0; i<A.rows(); ++i ) {
         if( !equal( A(i,j), B(i,j) ) ) return false;
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of two dense matrices with different storage order.
// \ingroup dense_matrix
//
// \param lhs The left-hand side matrix for the comparison.
// \param rhs The right-hand side matrix for the comparison.
// \return \a true if the two matrices are equal, \a false if not.
*/
template< typename T1  // Type of the left-hand side dense matrix
        , typename T2  // Type of the right-hand side dense matrix
        , bool SO >    // Storage order
inline bool operator==( const DenseMatrix<T1,SO>& lhs, const DenseMatrix<T2,!SO>& rhs )
{
   typedef CompositeType_<T1>  CT1;
   typedef CompositeType_<T2>  CT2;

   // Early exit in case the matrix sizes don't match
   if( (~lhs).rows() != (~rhs).rows() || (~lhs).columns() != (~rhs).columns() )
      return false;

   // Evaluation of the two dense matrix operands
   CT1 A( ~lhs );
   CT2 B( ~rhs );

   // In order to compare the two matrices, the data values of the lower-order data
   // type are converted to the higher-order data type within the equal function.
   const size_t rows   ( A.rows() );
   const size_t columns( A.columns() );
   const size_t block  ( 16 );

   for( size_t ii=0; ii<rows; ii+=block ) {
      const size_t iend( ( rows < ii+block )?( rows ):( ii+block ) );
      for( size_t jj=0; jj<columns; jj+=block ) {
         const size_t jend( ( columns < jj+block )?( columns ):( jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               if( !equal( A(i,j), B(i,j) ) ) return false;
            }
         }
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a dense matrix and a row-major sparse matrix.
// \ingroup dense_matrix
//
// \param lhs The left-hand side dense matrix for the comparison.
// \param rhs The right-hand side row-major sparse matrix for the comparison.
// \return \a true if the two matrices are equal, \a false if not.
*/
template< typename T1  // Type of the left-hand side dense matrix
        , typename T2  // Type of the right-hand side sparse matrix
        , bool SO >    // Storage order of the left-hand side dense matrix
inline bool operator==( const DenseMatrix<T1,SO>& lhs, const SparseMatrix<T2,false>& rhs )
{
   typedef CompositeType_<T1>  CT1;
   typedef CompositeType_<T2>  CT2;
   typedef ConstIterator_< RemoveReference_<CT2> >  ConstIterator;

   // Early exit in case the matrix sizes don't match
   if( (~lhs).rows() != (~rhs).rows() || (~lhs).columns() != (~rhs).columns() )
      return false;

   // Evaluation of the dense matrix and sparse matrix operand
   CT1 A( ~lhs );
   CT2 B( ~rhs );

   // In order to compare the two matrices, the data values of the lower-order data
   // type are converted to the higher-order data type within the equal function.
   size_t j( 0 );

   for( size_t i=0; i<B.rows(); ++i ) {
      j = 0;
      for( ConstIterator element=B.begin(i); element!=B.end(i); ++element, ++j ) {
         for( ; j<element->index(); ++j ) {
            if( !isDefault( A(i,j) ) ) return false;
         }
         if( !equal( element->value(), A(i,j) ) ) return false;
      }
      for( ; j<A.columns(); ++j ) {
         if( !isDefault( A(i,j) ) ) return false;
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a dense matrix and a column-major sparse matrix.
// \ingroup dense_matrix
//
// \param lhs The left-hand side dense matrix for the comparison.
// \param rhs The right-hand side column-major sparse matrix for the comparison.
// \return \a true if the two matrices are equal, \a false if not.
*/
template< typename T1  // Type of the left-hand side dense matrix
        , typename T2  // Type of the right-hand side sparse matrix
        , bool SO >    // Storage order of the left-hand side dense matrix
inline bool operator==( const DenseMatrix<T1,SO>& lhs, const SparseMatrix<T2,true>& rhs )
{
   typedef CompositeType_<T1>  CT1;
   typedef CompositeType_<T2>  CT2;
   typedef ConstIterator_< RemoveReference_<CT2> >  ConstIterator;

   // Early exit in case the matrix sizes don't match
   if( (~lhs).rows() != (~rhs).rows() || (~lhs).columns() != (~rhs).columns() )
      return false;

   // Evaluation of the dense matrix and sparse matrix operand
   CT1 A( ~lhs );
   CT2 B( ~rhs );

   // In order to compare the two matrices, the data values of the lower-order data
   // type are converted to the higher-order data type within the equal function.
   size_t i( 0 );

   for( size_t j=0; j<B.columns(); ++j ) {
      i = 0;
      for( ConstIterator element=B.begin(j); element!=B.end(j); ++element, ++i ) {
         for( ; i<element->index(); ++i ) {
            if( !isDefault( A(i,j) ) ) return false;
         }
         if( !equal( element->value(), A(i,j) ) ) return false;
      }
      for( ; i<A.rows(); ++i ) {
         if( !isDefault( A(i,j) ) ) return false;
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a sparse matrix and a dense matrix.
// \ingroup dense_matrix
//
// \param lhs The left-hand side sparse matrix for the comparison.
// \param rhs The right-hand side dense matrix for the comparison.
// \return \a true if the two matrices are equal, \a false if not.
*/
template< typename T1  // Type of the left-hand side sparse matrix
        , bool SO1     // Storage order of the left-hand side sparse matrix
        , typename T2  // Type of the right-hand side dense matrix
        , bool SO2 >   // Storage order of the right-hand side sparse matrix
inline bool operator==( const SparseMatrix<T1,SO1>& lhs, const DenseMatrix<T2,SO2>& rhs )
{
   return ( rhs == lhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a row-major dense matrix and a scalar value.
// \ingroup dense_matrix
//
// \param mat The left-hand side row-major dense matrix for the comparison.
// \param scalar The right-hand side scalar value for the comparison.
// \return \a true if all elements of the matrix are equal to the scalar, \a false if not.
//
// If all values of the matrix are equal to the scalar value, the equality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1    // Type of the left-hand side dense matrix
        , typename T2 >  // Type of the right-hand side scalar
inline EnableIf_<IsNumeric<T2>, bool > operator==( const DenseMatrix<T1,false>& mat, T2 scalar )
{
   typedef CompositeType_<T1>  CT1;

   // Evaluation of the dense matrix operand
   CT1 A( ~mat );

   // In order to compare the matrix and the scalar value, the data values of the lower-order
   // data type are converted to the higher-order data type within the equal function.
   for( size_t i=0; i<A.rows(); ++i ) {
      for( size_t j=0; j<A.columns(); ++j ) {
         if( !equal( A(i,j), scalar ) ) return false;
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a column-major dense matrix and a scalar value.
// \ingroup dense_matrix
//
// \param mat The left-hand side column-major dense matrix for the comparison.
// \param scalar The right-hand side scalar value for the comparison.
// \return \a true if all elements of the matrix are equal to the scalar, \a false if not.
//
// If all values of the matrix are equal to the scalar value, the equality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1    // Type of the left-hand side dense matrix
        , typename T2 >  // Type of the right-hand side scalar
inline EnableIf_<IsNumeric<T2>, bool > operator==( const DenseMatrix<T1,true>& mat, T2 scalar )
{
   typedef CompositeType_<T1>  CT1;

   // Evaluation of the dense matrix operand
   CT1 A( ~mat );

   // In order to compare the matrix and the scalar value, the data values of the lower-order
   // data type are converted to the higher-order data type within the equal function.
   for( size_t j=0; j<A.columns(); ++j ) {
      for( size_t i=0; i<A.rows(); ++i ) {
         if( !equal( A(i,j), scalar ) ) return false;
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a scalar value and a dense matrix.
// \ingroup dense_matrix
//
// \param scalar The left-hand side scalar value for the comparison.
// \param mat The right-hand side dense matrix for the comparison.
// \return \a true if all elements of the matrix are equal to the scalar, \a false if not.
//
// If all values of the matrix are equal to the scalar value, the equality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1  // Type of the left-hand side scalar
        , typename T2  // Type of the right-hand side dense matrix
        , bool SO >    // Storage order
inline EnableIf_<IsNumeric<T1>, bool > operator==( T1 scalar, const DenseMatrix<T2,SO>& mat )
{
   return ( mat == scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of two dense matrices.
// \ingroup dense_matrix
//
// \param lhs The left-hand side dense matrix for the comparison.
// \param rhs The right-hand side dense matrix for the comparison.
// \return \a true if the two matrices are not equal, \a false if they are equal.
*/
template< typename T1  // Type of the left-hand side dense matrix
        , bool SO1     // Storage order of the left-hand side dense matrix
        , typename T2  // Type of the right-hand side dense matrix
        , bool SO2 >   // Storage order of the right-hand side dense matrix
inline bool operator!=( const DenseMatrix<T1,SO1>& lhs, const DenseMatrix<T2,SO2>& rhs )
{
   return !( lhs == rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of a dense matrix and a sparse matrix.
// \ingroup dense_matrix
//
// \param lhs The left-hand side dense matrix for the comparison.
// \param rhs The right-hand side sparse matrix for the comparison.
// \return \a true if the two matrices are not equal, \a false if they are equal.
*/
template< typename T1  // Type of the left-hand side dense matrix
        , bool SO1     // Storage order of the left-hand side dense matrix
        , typename T2  // Type of the right-hand side sparse matrix
        , bool SO2 >   // Storage order of the right-hand side sparse matrix
inline bool operator!=( const DenseMatrix<T1,SO1>& lhs, const SparseMatrix<T2,SO2>& rhs )
{
   return !( lhs == rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of a sparse matrix and a dense matrix.
// \ingroup dense_matrix
//
// \param lhs The left-hand side sparse matrix for the comparison.
// \param rhs The right-hand side dense matrix for the comparison.
// \return \a true if the two matrices are not equal, \a false if they are equal.
*/
template< typename T1  // Type of the left-hand side sparse matrix
        , bool SO1     // Storage order of the left-hand side sparse matrix
        , typename T2  // Type of the right-hand side dense matrix
        , bool SO2 >   // Storage order right-hand side dense matrix
inline bool operator!=( const SparseMatrix<T1,SO1>& lhs, const DenseMatrix<T2,SO2>& rhs )
{
   return !( rhs == lhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of a dense matrix and a scalar value.
// \ingroup dense_matrix
//
// \param mat The left-hand side dense matrix for the comparison.
// \param scalar The right-hand side scalar value for the comparison.
// \return \a true if at least one element of the matrix is different from the scalar, \a false if not.
//
// If one value of the matrix is inequal to the scalar value, the inequality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1  // Type of the left-hand side dense matrix
        , typename T2  // Type of the right-hand side scalar
        , bool SO >    // Storage order
inline EnableIf_<IsNumeric<T2>, bool > operator!=( const DenseMatrix<T1,SO>& mat, T2 scalar )
{
   return !( mat == scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of a scalar value and a dense matrix.
// \ingroup dense_matrix
//
// \param scalar The left-hand side scalar value for the comparison.
// \param mat The right-hand side dense matrix for the comparison.
// \return \a true if at least one element of the matrix is different from the scalar, \a false if not.
//
// If one value of the matrix is inequal to the scalar value, the inequality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1  // Type of the left-hand side scalar
        , typename T2  // Type of the right-hand side dense matrix
        , bool SO >    // Storage order
inline EnableIf_<IsNumeric<T1>, bool > operator!=( T1 scalar, const DenseMatrix<T2,SO>& mat )
{
   return !( mat == scalar );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DenseMatrix functions */
//@{
template< typename MT, bool SO >
bool isnan( const DenseMatrix<MT,SO>& dm );

template< typename MT, bool SO >
bool isSymmetric( const DenseMatrix<MT,SO>& dm );

template< typename MT, bool SO >
bool isHermitian( const DenseMatrix<MT,SO>& dm );

template< typename MT, bool SO >
bool isUniform( const DenseMatrix<MT,SO>& dm );

template< typename MT, bool SO >
bool isLower( const DenseMatrix<MT,SO>& dm );

template< typename MT, bool SO >
bool isUniLower( const DenseMatrix<MT,SO>& dm );

template< typename MT, bool SO >
bool isStrictlyLower( const DenseMatrix<MT,SO>& dm );

template< typename MT, bool SO >
bool isUpper( const DenseMatrix<MT,SO>& dm );

template< typename MT, bool SO >
bool isUniUpper( const DenseMatrix<MT,SO>& dm );

template< typename MT, bool SO >
bool isStrictlyUpper( const DenseMatrix<MT,SO>& dm );

template< typename MT, bool SO >
bool isDiagonal( const DenseMatrix<MT,SO>& dm );

template< typename MT, bool SO >
bool isIdentity( const DenseMatrix<MT,SO>& dm );

template< typename MT, bool SO >
const ElementType_<MT> min( const DenseMatrix<MT,SO>& dm );

template< typename MT, bool SO >
const ElementType_<MT> max( const DenseMatrix<MT,SO>& dm );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks the given dense matrix for not-a-number elements.
// \ingroup dense_matrix
//
// \param dm The matrix to be checked for not-a-number elements.
// \return \a true if at least one element of the matrix is not-a-number, \a false otherwise.
//
// This function checks the dense matrix for not-a-number (NaN) elements. If at least one
// element of the matrix is not-a-number, the function returns \a true, otherwise it returns
// \a false.

   \code
   blaze::DynamicMatrix<double> A( 3UL, 4UL );
   // ... Initialization
   if( isnan( A ) ) { ... }
   \endcode

// Note that this function only works for matrices with floating point elements. The attempt to
// use it for a matrix with a non-floating point element type results in a compile time error.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
bool isnan( const DenseMatrix<MT,SO>& dm )
{
   typedef CompositeType_<MT>  CT;

   CT A( ~dm );  // Evaluation of the dense matrix operand

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<A.rows(); ++i ) {
         for( size_t j=0UL; j<A.columns(); ++j )
            if( isnan( A(i,j) ) ) return true;
      }
   }
   else {
      for( size_t j=0UL; j<A.columns(); ++j ) {
         for( size_t i=0UL; i<A.rows(); ++i )
            if( isnan( A(i,j) ) ) return true;
      }
   }

   return false;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given dense matrix is symmetric.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is symmetric, \a false if not.
//
// This function checks if the given dense matrix is symmetric. The matrix is considered to be
// symmetric if it is a square matrix whose transpose is equal to itself (\f$ A = A^T \f$). The
// following code example demonstrates the use of the function:

   \code
   blaze::DynamicMatrix<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isSymmetric( A ) ) { ... }
   \endcode

// It is also possible to check if a matrix expression results in a symmetric matrix:

   \code
   if( isSymmetric( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary matrix.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
bool isSymmetric( const DenseMatrix<MT,SO>& dm )
{
   typedef CompositeType_<MT>  CT;

   if( IsSymmetric<MT>::value )
      return true;

   if( !isSquare( ~dm ) )
      return false;

   if( (~dm).rows() < 2UL )
      return true;

   if( IsTriangular<MT>::value )
      return isDiagonal( ~dm );

   CT A( ~dm );  // Evaluation of the dense matrix operand

   if( SO == rowMajor ) {
      for( size_t i=1UL; i<A.rows(); ++i ) {
         for( size_t j=0UL; j<i; ++j ) {
            if( !equal( A(i,j), A(j,i) ) )
               return false;
         }
      }
   }
   else {
      for( size_t j=1UL; j<A.columns(); ++j ) {
         for( size_t i=0UL; i<j; ++i ) {
            if( !equal( A(i,j), A(j,i) ) )
               return false;
         }
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given dense matrix is Hermitian.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is Hermitian, \a false if not.
//
// This function checks if the given dense matrix is an Hermitian matrix. The matrix is considered
// to be an Hermitian matrix if it is a square matrix whose conjugate transpose is equal to itself
// (\f$ A = \overline{A^T} \f$), i.e. each matrix element \f$ a_{ij} \f$ is equal to the complex
// conjugate of the element \f$ a_{ji} \f$. The following code example demonstrates the use of the
// function:

   \code
   blaze::DynamicMatrix<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isHermitian( A ) ) { ... }
   \endcode

// It is also possible to check if a matrix expression results in an Hermitian matrix:

   \code
   if( isHermitian( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary matrix.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
bool isHermitian( const DenseMatrix<MT,SO>& dm )
{
   typedef ElementType_<MT>    ET;
   typedef CompositeType_<MT>  CT;

   if( IsHermitian<MT>::value )
      return true;

   if( !IsNumeric<ET>::value || !isSquare( ~dm ) )
      return false;

   if( (~dm).rows() < 2UL )
      return true;

   if( IsTriangular<MT>::value )
      return isDiagonal( ~dm );

   CT A( ~dm );  // Evaluation of the dense matrix operand

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<A.rows(); ++i ) {
         for( size_t j=0UL; j<i; ++j ) {
            if( !equal( A(i,j), conj( A(j,i) ) ) )
               return false;
         }
         if( !isReal( A(i,i) ) )
            return false;
      }
   }
   else {
      for( size_t j=0UL; j<A.columns(); ++j ) {
         for( size_t i=0UL; i<j; ++i ) {
            if( !equal( A(i,j), conj( A(j,i) ) ) )
               return false;
         }
         if( !isReal( A(j,j) ) )
            return false;
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given row-major triangular dense matrix is a uniform matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is a uniform matrix, \a false if not.
*/
template< typename MT >  // Type of the dense matrix
bool isUniform_backend( const DenseMatrix<MT,false>& dm, TrueType )
{
   BLAZE_CONSTRAINT_MUST_BE_TRIANGULAR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT );

   BLAZE_INTERNAL_ASSERT( (~dm).rows()    != 0UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( (~dm).columns() != 0UL, "Invalid number of columns detected" );

   const size_t ibegin( ( IsStrictlyLower<MT>::value )?( 1UL ):( 0UL ) );
   const size_t iend  ( ( IsStrictlyUpper<MT>::value )?( (~dm).rows()-1UL ):( (~dm).rows() ) );

   for( size_t i=ibegin; i<iend; ++i ) {
      if( !IsUpper<MT>::value ) {
         for( size_t j=0UL; j<i; ++j ) {
            if( !isDefault( (~dm)(i,j) ) )
               return false;
         }
      }
      if( !isDefault( (~dm)(i,i) ) )
         return false;
      if( !IsLower<MT>::value ) {
         for( size_t j=i+1UL; j<(~dm).columns(); ++j ) {
            if( !isDefault( (~dm)(i,j) ) )
               return false;
         }
      }
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given column-major triangular dense matrix is a uniform matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is a uniform matrix, \a false if not.
*/
template< typename MT >  // Type of the dense matrix
bool isUniform_backend( const DenseMatrix<MT,true>& dm, TrueType )
{
   BLAZE_CONSTRAINT_MUST_BE_TRIANGULAR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT );

   BLAZE_INTERNAL_ASSERT( (~dm).rows()    != 0UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( (~dm).columns() != 0UL, "Invalid number of columns detected" );

   const size_t jbegin( ( IsStrictlyUpper<MT>::value )?( 1UL ):( 0UL ) );
   const size_t jend  ( ( IsStrictlyLower<MT>::value )?( (~dm).columns()-1UL ):( (~dm).columns() ) );

   for( size_t j=jbegin; j<jend; ++j ) {
      if( !IsLower<MT>::value ) {
         for( size_t i=0UL; i<j; ++i ) {
            if( !isDefault( (~dm)(i,j) ) )
               return false;
         }
      }
      if( !isDefault( (~dm)(j,j) ) )
         return false;
      if( !IsUpper<MT>::value ) {
         for( size_t i=j+1UL; i<(~dm).rows(); ++i ) {
            if( !isDefault( (~dm)(i,j) ) )
               return false;
         }
      }
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given row-major general dense matrix is a uniform matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is a uniform matrix, \a false if not.
*/
template< typename MT >  // Type of the dense matrix
bool isUniform_backend( const DenseMatrix<MT,false>& dm, FalseType )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRIANGULAR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT );

   BLAZE_INTERNAL_ASSERT( (~dm).rows()    != 0UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( (~dm).columns() != 0UL, "Invalid number of columns detected" );

   ConstReference_<MT> cmp( (~dm)(0UL,0UL) );

   for( size_t i=0UL; i<(~dm).rows(); ++i ) {
      for( size_t j=0UL; j<(~dm).columns(); ++j ) {
         if( (~dm)(i,j) != cmp )
            return false;
      }
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checks if the given column-major general dense matrix is a uniform matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is a uniform matrix, \a false if not.
*/
template< typename MT >  // Type of the dense matrix
bool isUniform_backend( const DenseMatrix<MT,true>& dm, FalseType )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRIANGULAR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT );

   BLAZE_INTERNAL_ASSERT( (~dm).rows()    != 0UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( (~dm).columns() != 0UL, "Invalid number of columns detected" );

   ConstReference_<MT> cmp( (~dm)(0UL,0UL) );

   for( size_t j=0UL; j<(~dm).columns(); ++j ) {
      for( size_t i=0UL; i<(~dm).rows(); ++i ) {
         if( (~dm)(i,j) != cmp )
            return false;
      }
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given dense matrix is a uniform matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is a uniform matrix, \a false if not.
//
// This function checks if the given dense matrix is a uniform matrix. The matrix is considered
// to be uniform if all its elements are identical. The following code example demonstrates the
// use of the function:

   \code
   blaze::DynamicMatrix<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isUniform( A ) ) { ... }
   \endcode

// It is also possible to check if a matrix expression results in a uniform matrix:

   \code
   if( isUniform( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary matrix.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
bool isUniform( const DenseMatrix<MT,SO>& dm )
{
   if( IsUniTriangular<MT>::value )
      return false;

   if( (~dm).rows() == 0UL || (~dm).columns() == 0UL ||
       ( (~dm).rows() == 1UL && (~dm).columns() == 1UL ) )
      return true;

   CompositeType_<MT> A( ~dm );  // Evaluation of the dense matrix operand

   return isUniform_backend( A, typename IsTriangular<MT>::Type() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given dense matrix is a lower triangular matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is a lower triangular matrix, \a false if not.
//
// This function checks if the given dense matrix is a lower triangular matrix. The matrix is
// considered to be lower triangular if it is a square matrix of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        l_{0,0} & 0       & 0       & \cdots & 0       \\
                        l_{1,0} & l_{1,1} & 0       & \cdots & 0       \\
                        l_{2,0} & l_{2,1} & l_{2,2} & \cdots & 0       \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots  \\
                        l_{N,0} & l_{N,1} & l_{N,2} & \cdots & l_{N,N} \\
                        \end{array}\right).\f]

// \f$ 0 \times 0 \f$ or \f$ 1 \times 1 \f$ matrices are considered as trivially lower triangular.
// The following code example demonstrates the use of the function:

   \code
   blaze::DynamicMatrix<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isLower( A ) ) { ... }
   \endcode

// It is also possible to check if a matrix expression results in a lower triangular matrix:

   \code
   if( isLower( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary matrix.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
bool isLower( const DenseMatrix<MT,SO>& dm )
{
   typedef ResultType_<MT>     RT;
   typedef ReturnType_<MT>     RN;
   typedef CompositeType_<MT>  CT;
   typedef If_< IsExpression<RN>, const RT, CT >  Tmp;

   if( IsLower<MT>::value )
      return true;

   if( !isSquare( ~dm ) )
      return false;

   if( (~dm).rows() < 2UL )
      return true;

   Tmp A( ~dm );  // Evaluation of the dense matrix operand

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<A.rows()-1UL; ++i ) {
         for( size_t j=i+1UL; j<A.columns(); ++j ) {
            if( !isDefault( A(i,j) ) )
               return false;
         }
      }
   }
   else {
      for( size_t j=1UL; j<A.columns(); ++j ) {
         for( size_t i=0UL; i<j; ++i ) {
            if( !isDefault( A(i,j) ) )
               return false;
         }
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given dense matrix is a lower unitriangular matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is a lower unitriangular matrix, \a false if not.
//
// This function checks if the given dense matrix is a lower unitriangular matrix. The matrix is
// considered to be lower unitriangular if it is a square matrix of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        1       & 0       & 0       & \cdots & 0      \\
                        l_{1,0} & 1       & 0       & \cdots & 0      \\
                        l_{2,0} & l_{2,1} & 1       & \cdots & 0      \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots \\
                        l_{N,0} & l_{N,1} & l_{N,2} & \cdots & 1      \\
                        \end{array}\right).\f]

// The following code example demonstrates the use of the function:

   \code
   blaze::DynamicMatrix<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isUniLower( A ) ) { ... }
   \endcode

// It is also possible to check if a matrix expression results in a lower unitriangular matrix:

   \code
   if( isUniLower( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary matrix.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
bool isUniLower( const DenseMatrix<MT,SO>& dm )
{
   typedef ResultType_<MT>     RT;
   typedef ReturnType_<MT>     RN;
   typedef CompositeType_<MT>  CT;
   typedef If_< IsExpression<RN>, const RT, CT >  Tmp;

   if( IsUniLower<MT>::value )
      return true;

   if( !isSquare( ~dm ) )
      return false;

   Tmp A( ~dm );  // Evaluation of the dense matrix operand

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<A.rows(); ++i ) {
         if( !isOne( A(i,i) ) )
            return false;
         for( size_t j=i+1UL; j<A.columns(); ++j ) {
            if( !isZero( A(i,j) ) )
               return false;
         }
      }
   }
   else {
      for( size_t j=0UL; j<A.columns(); ++j ) {
         for( size_t i=0UL; i<j; ++i ) {
            if( !isZero( A(i,j) ) )
               return false;
         }
         if( !isOne( A(j,j) ) )
            return false;
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given dense matrix is a strictly lower triangular matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is a strictly lower triangular matrix, \a false if not.
//
// This function checks if the given dense matrix is a strictly lower triangular matrix. The
// matrix is considered to be strictly lower triangular if it is a square matrix of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        0       & 0       & 0       & \cdots & 0      \\
                        l_{1,0} & 0       & 0       & \cdots & 0      \\
                        l_{2,0} & l_{2,1} & 0       & \cdots & 0      \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots \\
                        l_{N,0} & l_{N,1} & l_{N,2} & \cdots & 0      \\
                        \end{array}\right).\f]

// The following code example demonstrates the use of the function:

   \code
   blaze::DynamicMatrix<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isStrictlyLower( A ) ) { ... }
   \endcode

// It is also possible to check if a matrix expression results in a strictly lower triangular
// matrix:

   \code
   if( isStrictlyLower( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary matrix.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
bool isStrictlyLower( const DenseMatrix<MT,SO>& dm )
{
   typedef ResultType_<MT>     RT;
   typedef ReturnType_<MT>     RN;
   typedef CompositeType_<MT>  CT;
   typedef If_< IsExpression<RN>, const RT, CT >  Tmp;

   if( IsStrictlyLower<MT>::value )
      return true;

   if( IsUniLower<MT>::value || IsUniUpper<MT>::value || !isSquare( ~dm ) )
      return false;

   Tmp A( ~dm );  // Evaluation of the dense matrix operand

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<A.rows(); ++i ) {
         for( size_t j=i; j<A.columns(); ++j ) {
            if( !isDefault( A(i,j) ) )
               return false;
         }
      }
   }
   else {
      for( size_t j=0UL; j<A.columns(); ++j ) {
         for( size_t i=0UL; i<=j; ++i ) {
            if( !isDefault( A(i,j) ) )
               return false;
         }
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given dense matrix is an upper triangular matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is an upper triangular matrix, \a false if not.
//
// This function checks if the given dense matrix is an upper triangular matrix. The matrix is
// considered to be upper triangular if it is a square matrix of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        u_{0,0} & u_{0,1} & u_{0,2} & \cdots & u_{0,N} \\
                        0       & u_{1,1} & u_{1,2} & \cdots & u_{1,N} \\
                        0       & 0       & u_{2,2} & \cdots & u_{2,N} \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots  \\
                        0       & 0       & 0       & \cdots & u_{N,N} \\
                        \end{array}\right).\f]

// \f$ 0 \times 0 \f$ or \f$ 1 \times 1 \f$ matrices are considered as trivially upper triangular.
// The following code example demonstrates the use of the function:

   \code
   blaze::DynamicMatrix<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isUpper( A ) ) { ... }
   \endcode

// It is also possible to check if a matrix expression results in an upper triangular matrix:

   \code
   if( isUpper( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary matrix.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
bool isUpper( const DenseMatrix<MT,SO>& dm )
{
   typedef ResultType_<MT>     RT;
   typedef ReturnType_<MT>     RN;
   typedef CompositeType_<MT>  CT;
   typedef If_< IsExpression<RN>, const RT, CT >  Tmp;

   if( IsUpper<MT>::value )
      return true;

   if( !isSquare( ~dm ) )
      return false;

   if( (~dm).rows() < 2UL )
      return true;

   Tmp A( ~dm );  // Evaluation of the dense matrix operand

   if( SO == rowMajor ) {
      for( size_t i=1UL; i<A.rows(); ++i ) {
         for( size_t j=0UL; j<i; ++j ) {
            if( !isDefault( A(i,j) ) )
               return false;
         }
      }
   }
   else {
      for( size_t j=0UL; j<A.columns()-1UL; ++j ) {
         for( size_t i=j+1UL; i<A.rows(); ++i ) {
            if( !isDefault( A(i,j) ) )
               return false;
         }
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given dense matrix is an upper unitriangular matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is an upper unitriangular matrix, \a false if not.
//
// This function checks if the given dense matrix is an upper unitriangular matrix. The matrix is
// considered to be upper unitriangular if it is a square matrix of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        1      & u_{0,1} & u_{0,2} & \cdots & u_{0,N} \\
                        0      & 1       & u_{1,2} & \cdots & u_{1,N} \\
                        0      & 0       & 1       & \cdots & u_{2,N} \\
                        \vdots & \vdots  & \vdots  & \ddots & \vdots  \\
                        0      & 0       & 0       & \cdots & 1       \\
                        \end{array}\right).\f]

// The following code example demonstrates the use of the function:

   \code
   blaze::DynamicMatrix<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isUniUpper( A ) ) { ... }
   \endcode

// It is also possible to check if a matrix expression results in an upper unitriangular matrix:

   \code
   if( isUniUpper( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary matrix.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
bool isUniUpper( const DenseMatrix<MT,SO>& dm )
{
   typedef ResultType_<MT>     RT;
   typedef ReturnType_<MT>     RN;
   typedef CompositeType_<MT>  CT;
   typedef If_< IsExpression<RN>, const RT, CT >  Tmp;

   if( IsUniUpper<MT>::value )
      return true;

   if( !isSquare( ~dm ) )
      return false;

   Tmp A( ~dm );  // Evaluation of the dense matrix operand

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<A.rows(); ++i ) {
         for( size_t j=0UL; j<i; ++j ) {
            if( !isZero( A(i,j) ) )
               return false;
         }
         if( !isOne( A(i,i) ) )
            return false;
      }
   }
   else {
      for( size_t j=0UL; j<A.columns(); ++j ) {
         if( !isOne( A(j,j) ) )
            return false;
         for( size_t i=j+1UL; i<A.rows(); ++i ) {
            if( !isZero( A(i,j) ) )
               return false;
         }
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given dense matrix is a strictly upper triangular matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is a strictly upper triangular matrix, \a false if not.
//
// This function checks if the given dense matrix is a strictly upper triangular matrix. The
// matrix is considered to be strictly upper triangular if it is a square matrix of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        0      & u_{0,1} & u_{0,2} & \cdots & u_{0,N} \\
                        0      & 0       & u_{1,2} & \cdots & u_{1,N} \\
                        0      & 0       & 0       & \cdots & u_{2,N} \\
                        \vdots & \vdots  & \vdots  & \ddots & \vdots  \\
                        0      & 0       & 0       & \cdots & 0       \\
                        \end{array}\right).\f]

// The following code example demonstrates the use of the function:

   \code
   blaze::DynamicMatrix<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isStrictlyUpper( A ) ) { ... }
   \endcode

// It is also possible to check if a matrix expression results in a strictly upper triangular
// matrix:

   \code
   if( isStrictlyUpper( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary matrix.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
bool isStrictlyUpper( const DenseMatrix<MT,SO>& dm )
{
   typedef ResultType_<MT>     RT;
   typedef ReturnType_<MT>     RN;
   typedef CompositeType_<MT>  CT;
   typedef If_< IsExpression<RN>, const RT, CT >  Tmp;

   if( IsStrictlyUpper<MT>::value )
      return true;

   if( IsUniLower<MT>::value || IsUniUpper<MT>::value || !isSquare( ~dm ) )
      return false;

   Tmp A( ~dm );  // Evaluation of the dense matrix operand

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<A.rows(); ++i ) {
         for( size_t j=0UL; j<=i; ++j ) {
            if( !isDefault( A(i,j) ) )
               return false;
         }
      }
   }
   else {
      for( size_t j=0UL; j<A.columns(); ++j ) {
         for( size_t i=j; i<A.rows(); ++i ) {
            if( !isDefault( A(i,j) ) )
               return false;
         }
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the give dense matrix is diagonal.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is diagonal, \a false if not.
//
// This function tests whether the matrix is diagonal, i.e. if the non-diagonal elements are
// default elements. In case of integral or floating point data types, a diagonal matrix has
// the form

                        \f[\left(\begin{array}{*{5}{c}}
                        aa     & 0      & 0      & \cdots & 0  \\
                        0      & bb     & 0      & \cdots & 0  \\
                        0      & 0      & cc     & \cdots & 0  \\
                        \vdots & \vdots & \vdots & \ddots & 0  \\
                        0      & 0      & 0      & 0      & xx \\
                        \end{array}\right)\f]

// \f$ 0 \times 0 \f$ or \f$ 1 \times 1 \f$ matrices are considered as trivially diagonal. The
// following example demonstrates the use of the function:

   \code
   blaze::DynamicMatrix<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isDiagonal( A ) ) { ... }
   \endcode

// It is also possible to check if a matrix expression results in a diagonal matrix:

   \code
   if( isDiagonal( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary matrix.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
bool isDiagonal( const DenseMatrix<MT,SO>& dm )
{
   typedef ResultType_<MT>     RT;
   typedef ReturnType_<MT>     RN;
   typedef CompositeType_<MT>  CT;
   typedef If_< IsExpression<RN>, const RT, CT >  Tmp;

   if( IsDiagonal<MT>::value )
      return true;

   if( !isSquare( ~dm ) )
      return false;

   if( (~dm).rows() < 2UL )
      return true;

   Tmp A( ~dm );  // Evaluation of the dense matrix operand

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<A.rows(); ++i ) {
         if( !IsUpper<MT>::value ) {
            for( size_t j=0UL; j<i; ++j ) {
               if( !isDefault( A(i,j) ) )
                  return false;
            }
         }
         if( !IsLower<MT>::value ) {
            for( size_t j=i+1UL; j<A.columns(); ++j ) {
               if( !isDefault( A(i,j) ) )
                  return false;
            }
         }
      }
   }
   else {
      for( size_t j=0UL; j<A.columns(); ++j ) {
         if( !IsLower<MT>::value ) {
            for( size_t i=0UL; i<j; ++i ) {
               if( !isDefault( A(i,j) ) )
                  return false;
            }
         }
         if( !IsUpper<MT>::value ) {
            for( size_t i=j+1UL; i<A.rows(); ++i ) {
               if( !isDefault( A(i,j) ) )
                  return false;
            }
         }
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the give dense matrix is an identity matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be checked.
// \return \a true if the matrix is an identity matrix, \a false if not.
//
// This function tests whether the matrix is an identity matrix, i.e. if the diagonal elements
// are 1 and the non-diagonal elements are 0. In case of integral or floating point data types,
// an identity matrix has the form

                        \f[\left(\begin{array}{*{5}{c}}
                        1      & 0      & 0      & \cdots & 0 \\
                        0      & 1      & 0      & \cdots & 0 \\
                        0      & 0      & 1      & \cdots & 0 \\
                        \vdots & \vdots & \vdots & \ddots & 0 \\
                        0      & 0      & 0      & 0      & 1 \\
                        \end{array}\right)\f]

// The following example demonstrates the use of the function:

   \code
   blaze::DynamicMatrix<int,blaze::rowMajor> A, B;
   // ... Initialization
   if( isIdentity( A ) ) { ... }
   \endcode

// It is also possible to check if a matrix expression results in an identity matrix:

   \code
   if( isIdentity( A * B ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary matrix.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
bool isIdentity( const DenseMatrix<MT,SO>& dm )
{
   typedef ResultType_<MT>     RT;
   typedef ReturnType_<MT>     RN;
   typedef CompositeType_<MT>  CT;
   typedef If_< IsExpression<RN>, const RT, CT >  Tmp;

   if( IsIdentity<MT>::value )
      return true;

   if( !isSquare( ~dm ) )
      return false;

   if( (~dm).rows() == 0UL )
      return true;

   Tmp A( ~dm );  // Evaluation of the dense matrix operand

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<A.rows(); ++i ) {
         if( !IsUpper<MT>::value ) {
            for( size_t j=0UL; j<i; ++j ) {
               if( !isZero( A(i,j) ) )
                  return false;
            }
         }
         if( !IsUniLower<MT>::value && !IsUniUpper<MT>::value && !isOne( A(i,i) ) ) {
            return false;
         }
         if( !IsLower<MT>::value ) {
            for( size_t j=i+1UL; j<A.columns(); ++j ) {
               if( !isZero( A(i,j) ) )
                  return false;
            }
         }
      }
   }
   else {
      for( size_t j=0UL; j<A.columns(); ++j ) {
         if( !IsLower<MT>::value ) {
            for( size_t i=0UL; i<j; ++i ) {
               if( !isZero( A(i,j) ) )
                  return false;
            }
         }
         if( !IsUniLower<MT>::value && !IsUniUpper<MT>::value && !isOne( A(j,j) ) ) {
            return false;
         }
         if( !IsUpper<MT>::value ) {
            for( size_t i=j+1UL; i<A.rows(); ++i ) {
               if( !isZero( A(i,j) ) )
                  return false;
            }
         }
      }
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the smallest element of the dense matrix.
// \ingroup dense_matrix
//
// \param dm The given dense matrix.
// \return The smallest dense matrix element.
//
// This function returns the smallest element of the given dense matrix. This function can
// only be used for element types that support the smaller-than relationship. In case the
// matrix currently has either 0 rows or 0 columns, the returned value is the default value
// (e.g. 0 in case of fundamental data types).
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
const ElementType_<MT> min( const DenseMatrix<MT,SO>& dm )
{
   using blaze::min;

   typedef ElementType_<MT>    ET;
   typedef CompositeType_<MT>  CT;

   CT A( ~dm );  // Evaluation of the dense matrix operand

   if( A.rows() == 0UL || A.columns() == 0UL ) return ET();

   ET minimum( A(0,0) );

   if( SO == rowMajor ) {
      for( size_t j=1UL; j<A.columns(); ++j )
         minimum = min( minimum, A(0UL,j) );
      for( size_t i=1UL; i<A.rows(); ++i )
         for( size_t j=0UL; j<A.columns(); ++j )
            minimum = min( minimum, A(i,j) );
   }
   else {
      for( size_t i=1UL; i<A.rows(); ++i )
         minimum = min( minimum, A(i,0UL) );
      for( size_t j=1UL; j<A.columns(); ++j )
         for( size_t i=0UL; i<A.rows(); ++i )
            minimum = min( minimum, A(i,j) );
   }

   return minimum;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the largest element of the dense matrix.
// \ingroup dense_matrix
//
// \param dm The given dense matrix.
// \return The largest dense matrix element.
//
// This function returns the largest element of the given dense matrix. This function can
// only be used for element types that support the smaller-than relationship. In case the
// matrix currently has either 0 rows or 0 columns, the returned value is the default value
// (e.g. 0 in case of fundamental data types).
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Transpose flag
const ElementType_<MT> max( const DenseMatrix<MT,SO>& dm )
{
   using blaze::max;

   typedef ElementType_<MT>    ET;
   typedef CompositeType_<MT>  CT;

   CT A( ~dm );  // Evaluation of the dense matrix operand

   if( A.rows() == 0UL || A.columns() == 0UL ) return ET();

   ET maximum( A(0,0) );

   if( SO == rowMajor ) {
      for( size_t j=1UL; j<A.columns(); ++j )
         maximum = max( maximum, A(0UL,j) );
      for( size_t i=1UL; i<A.rows(); ++i )
         for( size_t j=0UL; j<A.columns(); ++j )
            maximum = max( maximum, A(i,j) );
   }
   else {
      for( size_t i=1UL; i<A.rows(); ++i )
         maximum = max( maximum, A(i,0UL) );
      for( size_t j=1UL; j<A.columns(); ++j )
         for( size_t i=0UL; i<A.rows(); ++i )
            maximum = max( maximum, A(i,j) );
   }

   return maximum;
}
//*************************************************************************************************

} // namespace blaze

#endif
