//=================================================================================================
/*!
//  \file blaze/math/lapack/trsv.h
//  \brief Header file for the LAPACK triangular linear system solver functions (trsv)
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

#ifndef _BLAZE_MATH_LAPACK_TRSV_H_
#define _BLAZE_MATH_LAPACK_TRSV_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <boost/cast.hpp>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Adaptor.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/MutableDataAccess.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/lapack/clapack/trsv.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/SameType.h>


namespace blaze {

//=================================================================================================
//
//  LAPACK TRIANGULAR LINEAR SYSTEM FUNCTIONS (TRSV)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK triangular linear system functions (trsv) */
//@{
template< typename MT, bool SO, typename VT, bool TF >
inline void trsv( const DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b,
                  char uplo, char trans, char diag );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for solving a triangular linear system of equations (\f$ A*x=b \f$).
//        (\f$ A*x=b \f$).
// \ingroup lapack_solver
//
// \param A The system matrix.
// \param b The right-hand side vector.
// \param uplo \c 'L' in case of a lower matrix, \c 'U' in case of an upper matrix.
// \param trans \c 'N' for \f$ A*x=b \f$, \c 'T' for \f$ A^T*x=b \f$, and \c C for \f$ A^H*x=b \f$.
// \param diag \c 'U' in case of a unitriangular matrix, \c 'N' otherwise.
// \return void
// \exception std::invalid_argument Invalid non-square matrix provided.
// \exception std::invalid_argument Invalid uplo argument provided.
// \exception std::invalid_argument Invalid trans argument provided.
// \exception std::invalid_argument Invalid diag argument provided.
//
// This function uses the LAPACK trsv() functions to compute the solution to the triangular system
// of linear equations:
//
//  - \f$ A  *x=b \f$ if \a A is column-major
//  - \f$ A^T*x=b \f$ if \a A is row-major
//
// In this context the positive definite system matrix \a A is a n-by-n matrix and \a x and \a b
// are n-dimensional vectors. Note that the function only works for general, non-adapted matrices
// with \c float, \c double, \c complex<float>, or \c complex<double> element type. The attempt
// to call the function with adaptors or matrices of any other element type results in a compile
// time error!
//
// If the function exits successfully, the vector \a x contains the solution of the linear system
// of equations. The function fails if ...
//
//  - ... the given system matrix is not a square matrix;
//  - ... the given \a uplo argument is neither \c 'L' nor \c 'U';
//  - ... the given \a trans argument is neither \c 'N' nor \c 'T' nor \c 'C';
//  - ... the given \a diag argument is neither \c 'U' nor \c 'N'.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// Examples:

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::columnMajor;
   using blaze::columnVector;

   DynamicMatrix<double,columnMajor>  A( 2UL, 2UL );  // The system matrix A
   DynamicVector<double,columnVector> b( 2UL );       // The right-hand side vector b
   // ... Initialization

   DynamicMatrix<double,columnMajor>  D( A );  // Temporary matrix to be decomposed
   DynamicVector<double,columnVector> x( b );  // Temporary vector for the solution

   trsv( D, x, 'L', 'N', 'N' );

   assert( A * x == b );
   \endcode

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::rowMajor;
   using blaze::columnVector;

   DynamicMatrix<double,rowMajor>  A( 2UL, 2UL );  // The system matrix A
   DynamicVector<double,columnVector> b( 2UL );    // The right-hand side vector b
   // ... Initialization

   DynamicMatrix<double,rowMajor>  D( A );     // Temporary matrix to be decomposed
   DynamicVector<double,columnVector> x( b );  // Temporary vector for the solution

   trsv( D, x, 'L', 'N', 'N' );

   assert( trans( A ) * x == b );
   \endcode

// For more information on the trsv() functions (i.e. strsv(), dtrsv(), ctrsv(), and ztrsv()), see
// the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
//
// \note The function does not perform any test for singularity or near-singularity. Such tests
// must be performed prior to calling this function!
*/
template< typename MT  // Type of the system matrix
        , bool SO      // Storage order of the system matrix
        , typename VT  // Type of the right-hand side vector
        , bool TF >    // Transpose flag of the right-hand side vector
inline void trsv( const DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b, char uplo, char trans, char diag )
{
   using boost::numeric_cast;

   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   if( !isSquare( ~A ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid non-square matrix provided" );
   }

   if( uplo != 'L' && uplo != 'U' ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid uplo argument provided" );
   }

   if( trans != 'N' && trans != 'T' && trans != 'C' ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid trans argument provided" );
   }

   if( diag != 'U' && diag != 'N' ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid diag argument provided" );
   }

   int n   ( numeric_cast<int>( (~A).rows() ) );
   int lda ( numeric_cast<int>( (~A).spacing() ) );
   int incX( 1 );

   if( n == 0 ) {
      return;
   }

   if( IsRowMajorMatrix<MT>::value ) {
      ( uplo == 'L' )?( uplo = 'U' ):( uplo = 'L' );
   }

   trsv( uplo, trans, diag, n, (~A).data(), lda, (~b).data(), incX );
}
//*************************************************************************************************

} // namespace blaze

#endif
