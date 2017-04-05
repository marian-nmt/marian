//=================================================================================================
/*!
//  \file blaze/math/dense/LLH.h
//  \brief Header file for the dense matrix in-place Cholesky (LLH) decomposition
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

#ifndef _BLAZE_MATH_DENSE_LLH_H_
#define _BLAZE_MATH_DENSE_LLH_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Hermitian.h>
#include <blaze/math/constraints/StrictlyTriangular.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/UniTriangular.h>
#include <blaze/math/constraints/Upper.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/lapack/potrf.h>
#include <blaze/math/traits/DerestrictTrait.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>


namespace blaze {

//=================================================================================================
//
//  LLH DECOMPOSITION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name LLH decomposition functions */
//@{
template< typename MT1, bool SO1, typename MT2, bool SO2 >
void llh( const DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& L );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Cholesky (LLH) decomposition of the given dense matrix.
// \ingroup dense_matrix
//
// \param A The matrix to be decomposed.
// \param L The resulting lower triangular matrix.
// \return void
// \exception std::invalid_argument Invalid non-square matrix provided.
// \exception std::invalid_argument Dimensions of fixed size matrix do not match.
// \exception std::invalid_argument Decomposition of singular matrix failed.
//
// This function performs the dense matrix Cholesky (LLH) decomposition of a positive definite
// n-by-n matrix. The resulting decomposition has the form

                              \f[ A = L \cdot L^{H}, \f]

// where \c L is a lower triangular n-by-n matrix. The decomposition is written to the matrix
// \c L, which is resized to the correct dimensions (if possible and necessary).
//
// The function fails if ...
//
//  - ... \a A is not a square matrix;
//  - ... \a L is a fixed size matrix and the dimensions don't match \a A.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// Example:

   \code
   blaze::DynamicMatrix<double,blaze::columnMajor> A( 32, 32 );
   // ... Initialization of A as positive definite matrix

   blaze::DynamicMatrix<double,blaze::columnMajor> L( 32, 32 );

   llh( A, L );

   assert( A == L * trans( L ) );
   \endcode

// \note This function only works for matrices with \c float, \c double, \c complex<float>, or
// \c complex<double> element type. The attempt to call the function with matrices of any other
// element type results in a compile time error!
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
//
// \note This function does only provide the basic exception safety guarantee, i.e. in case of an
// exception \a L may already have been modified.
*/
template< typename MT1  // Type of matrix A
        , bool SO1      // Storage order of matrix A
        , typename MT2  // Type of matrix L
        , bool SO2 >    // Storage order of matrix L
void llh( const DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& L )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_STRICTLY_TRIANGULAR_MATRIX_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT1> );

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_HERMITIAN_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_UPPER_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT2> );

   if( !isSquare( ~A ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid non-square matrix provided" );
   }

   const size_t n( (~A).rows() );

   if( ( !IsResizable<MT2>::value && ( (~L).rows() != n || (~L).columns() != n ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Dimensions of fixed size matrix do not match" );
   }

   DerestrictTrait_<MT2> l( derestrict( ~L ) );

   resize( ~L, n, n );
   reset( l );

   if( IsRowMajorMatrix<MT2>::value ) {
      for( size_t i=0UL; i<n; ++i ) {
         for( size_t j=0UL; j<=i; ++j ) {
            l(i,j) = (~A)(i,j);
         }
      }
   }
   else {
      for( size_t j=0UL; j<n; ++j ) {
         for( size_t i=j; i<n; ++i ) {
            l(i,j) = (~A)(i,j);
         }
      }
   }

   potrf( l, 'L' );
}
//*************************************************************************************************

} // namespace blaze

#endif
