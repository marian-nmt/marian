//=================================================================================================
/*!
//  \file blaze/math/dense/LU.h
//  \brief Header file for the dense matrix in-place LU decomposition
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

#ifndef _BLAZE_MATH_DENSE_LU_H_
#define _BLAZE_MATH_DENSE_LU_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <memory>
#include <utility>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Adaptor.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Hermitian.h>
#include <blaze/math/constraints/Lower.h>
#include <blaze/math/constraints/StrictlyTriangular.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/UniTriangular.h>
#include <blaze/math/constraints/Upper.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/Functions.h>
#include <blaze/math/lapack/getrf.h>
#include <blaze/math/traits/DerestrictTrait.h>
#include <blaze/math/typetraits/IsResizable.h>


namespace blaze {

//=================================================================================================
//
//  LU DECOMPOSITION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name LU decomposition functions */
//@{
template< typename MT1, bool SO1, typename MT2, typename MT3, typename MT4, bool SO2 >
void lu( const DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO1>& L,
         DenseMatrix<MT3,SO1>& U, Matrix<MT4,SO2>& P );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary function for the LU decomposition of the given dense matrix.
// \ingroup dense_matrix
//
// \param A The matrix to be decomposed.
// \param P The resulting permutation matrix.
// \return void
//
// This function is an auxiliary helper for the dense matrix LU decomposition. It performs an
// in-place LU decomposition on the given matrix \c A and reconstructs the permutation matrix
// \c P.
*/
template< typename MT1  // Type of matrix A
        , bool SO1      // Storage order of dense matrix A
        , typename MT2  // Type of matrix P
        , bool SO2 >    // Storage order of matrix P
void lu( DenseMatrix<MT1,SO1>& A, Matrix<MT2,SO2>& P )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT1> );
   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT2 );

   typedef ElementType_<MT2>  ET;

   const size_t m( (~A).rows()    );
   const size_t n( (~A).columns() );
   const size_t mindim( min( m, n ) );
   const size_t size( SO1 ? m : n );

   const std::unique_ptr<int[]> helper( new int[mindim + size] );
   int* ipiv  ( helper.get() );
   int* permut( ipiv + mindim );

   getrf( ~A, ipiv );

   for( size_t i=0UL; i<size; ++i ) {
      permut[i] = i;
   }

   for( int i=0; i<mindim; ++i ) {
      --ipiv[i];
      if( ipiv[i] != i ) {
         std::swap( permut[ipiv[i]], permut[i] );
      }
   }

   resize( ~P, size, size );
   reset( ~P );
   for( size_t i=0UL; i<size; ++i ) {
      (~P)( ( SO1 ? permut[i] : i ), ( SO1 ? i : permut[i] ) ) = ET(1);
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LU decomposition of the given dense matrix.
// \ingroup dense_matrix
//
// \param A The matrix to be decomposed.
// \param L The resulting lower triangular matrix.
// \param U The resulting upper triangular matrix.
// \param P The resulting permutation matrix.
// \return void
// \exception std::invalid_argument Dimensions of fixed size matrix do not match.
// \exception std::invalid_argument Square matrix cannot be resized to \a m-by-\a n.
//
// This function performs the dense matrix (P)LU decomposition of a general \a m-by-\a n matrix.
// The resulting decomposition is written to the three distinct matrices \c L, \c U, and \c P,
// which are resized to the correct dimensions (if possible and necessary).
//
// In case of a column-major matrix the algorithm performs the decomposition using partial pivoting
// with row interchanges. The resulting decomposition has the form

                          \f[ A = P \cdot L \cdot U, \f]

// where \c P is an m-by-m permutation matrix, which represents the pivoting indices for the applied
// row interchanges, \c L is a lower triangular matrix (lower trapezoidal if \a m > \a n), and \c U
// is an upper triangular matrix (upper trapezoidal if \a m < \a n).
//
// In case of a row-major matrix the algorithm performs the decomposition using partial pivoting
// with column interchanges. The resulting decomposition has the form

                          \f[ A = L \cdot U \cdot P, \f]

// where \c L is a lower triangular matrix (lower trapezoidal if \a m > \a n), \c U is an upper
// triangular matrix (upper trapezoidal if \a m < \a n), and \c P is an n-by-n permutation matrix,
// which represents the pivoting indices for the applied column interchanges.
//
// The function fails if ...
//
//  - ... either \a L, \a U, and \a P are fixed size matrices and the dimensions don't match;
//  - ... \a A is a non-square m-by-n matrix, but \a L or \a U is a compile time square matrix.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// Examples:

   \code
   blaze::DynamicMatrix<double,blaze::rowMajor> A;
   // ... Resizing and initialization

   blaze::DynamicMatrix<double,blaze::rowMajor> L, U, P;

   lu( A, L, U, P );  // LU decomposition of a row-major matrix

   assert( A == L * U * P );
   \endcode

   \code
   blaze::DynamicMatrix<double,blaze::columnMajor> A;
   // ... Resizing and initialization

   blaze::DynamicMatrix<double,blaze::columnMajor> L, U, P;

   lu( A, L, U, P );  // LU decomposition of a column-major matrix

   assert( A == P * L * U );
   \endcode

// \note This function only works for matrices with \c float, \c double, \c complex<float>, or
// \c complex<double> element type. The attempt to call the function with matrices of any other
// element type results in a compile time error!
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
//
// \note The LU decomposition will never fail, even for singular matrices. However, in case of a
// singular matrix the resulting decomposition cannot be used for a matrix inversion or solving
// a linear system of equations.
*/
template< typename MT1  // Type of matrix A
        , bool SO1      // Storage order of matrix A, L and U
        , typename MT2  // Type of matrix L
        , typename MT3  // Type of matrix U
        , typename MT4  // Type of matrix P
        , bool SO2 >    // Storage order of matrix P
void lu( const DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO1>& L,
         DenseMatrix<MT3,SO1>& U, Matrix<MT4,SO2>& P )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_STRICTLY_TRIANGULAR_MATRIX_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT1> );

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_HERMITIAN_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_UPPER_MATRIX_TYPE( MT2 );

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT3 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_HERMITIAN_MATRIX_TYPE( MT3 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT3 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_LOWER_MATRIX_TYPE( MT3 );

   typedef ElementType_<MT2>  ET2;
   typedef ElementType_<MT3>  ET3;

   const size_t m( (~A).rows()    );
   const size_t n( (~A).columns() );
   const size_t mindim( min( m, n ) );
   const size_t size( SO1 ? m : n );

   if( ( !IsResizable<MT2>::value && ( (~L).rows() != m      || (~L).columns() != mindim ) ) ||
       ( !IsResizable<MT3>::value && ( (~U).rows() != mindim || (~U).columns() != n      ) ) ||
       ( !IsResizable<MT4>::value && ( (~P).rows() != size   || (~P).columns() != size   ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Dimensions of fixed size matrix do not match" );
   }

   if( ( IsSquare<MT2>::value && n < m ) || ( IsSquare<MT3>::value && m < n ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Square matrix cannot be resized to m-by-n" );
   }

   DerestrictTrait_<MT2> l( derestrict( ~L ) );
   DerestrictTrait_<MT3> u( derestrict( ~U ) );

   if( m < n )
   {
      u = (~A);
      lu( u, ~P );

      resize( ~L, m, m );
      reset( l );

      if( SO1 == rowMajor )
      {
         for( size_t i=0UL; i<m; ++i )
         {
            for( size_t j=0UL; j<i; ++j ) {
               l(i,j) = u(i,j);
               reset( u(i,j) );
            }

            l(i,i) = u(i,i);
            u(i,i) = ET3(1);
         }
      }
      else
      {
         for( size_t j=0UL; j<m; ++j )
         {
            l(j,j) = ET2(1);

            for( size_t i=j+1UL; i<m; ++i ) {
               l(i,j) = u(i,j);
               reset( u(i,j) );
            }
         }
      }
   }
   else
   {
      l = (~A);
      lu( l, ~P );

      resize( ~U, n, n );
      reset( u );

      if( SO1 == rowMajor )
      {
         for( size_t i=0UL; i<n; ++i )
         {
            u(i,i) = ET3(1);

            for( size_t j=i+1UL; j<n; ++j ) {
               u(i,j) = l(i,j);
               reset( l(i,j) );
            }
         }
      }
      else
      {
         for( size_t j=0UL; j<n; ++j )
         {
            for( size_t i=0UL; i<j; ++i ) {
               u(i,j) = l(i,j);
               reset( l(i,j) );
            }

            u(j,j) = l(j,j);
            l(j,j) = ET2(1);
         }
      }
   }
}
//*************************************************************************************************

} // namespace blaze

#endif
