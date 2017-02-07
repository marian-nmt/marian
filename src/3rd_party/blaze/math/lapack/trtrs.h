//=================================================================================================
/*!
//  \file blaze/math/lapack/trtrs.h
//  \brief Header file for the LAPACK triangular backward substitution functions (trtrs)
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

#ifndef _BLAZE_MATH_LAPACK_TRTRS_H_
#define _BLAZE_MATH_LAPACK_TRTRS_H_


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
#include <blaze/math/lapack/clapack/trtrs.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/SameType.h>


namespace blaze {

//=================================================================================================
//
//  LAPACK TRIANGULAR SUBSTITUTION FUNCTIONS (TRTRS)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK triangular substitution functions (trtrs) */
//@{
template< typename MT, bool SO, typename VT, bool TF >
inline void trtrs( const DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b,
                   char uplo, char trans, char diag );

template< typename MT1, bool SO1, typename MT2, bool SO2 >
inline void trtrs( const DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& B,
                   char uplo, char trans, char diag );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the substitution step of solving a triangular linear system of equations
//        (\f$ A*x=b \f$).
// \ingroup lapack_substitution
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
// This function uses the LAPACK trtrs() functions to perform the substitution step to compute
// the solution to the triangular system of linear equations:
//
//  - \f$ A  *x=b \f$ if \a A is column-major
//  - \f$ A^T*x=b \f$ if \a A is row-major
//
// In this context the triangular system matrix \a A is a n-by-n matrix and \a x and \a b are
// n-dimensional vectors. Note that the function only works for general, non-adapted matrices with
// \c float, \c double, \c complex<float>, or \c complex<double> element type. The attempt to call
// the function with adaptors or matrices of any other element type results in a compile time error!
//
// If the function exits successfully, the vector \a b contains the solution of the linear system
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

   trtrs( D, x, 'L', 'N', 'N' );

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

   trtrs( D, x, 'L', 'N', 'N' );

   assert( trans( A ) * x == b );
   \endcode

// For more information on the trtrs() functions (i.e. strtrs(), dtrtrs(), ctrtrs(), and ztrtrs()),
// see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
template< typename MT  // Type of the system matrix
        , bool SO      // Storage order of the system matrix
        , typename VT  // Type of the right-hand side vector
        , bool TF >    // Transpose flag of the right-hand side vector
inline void trtrs( const DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b, char uplo, char trans, char diag )
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
   int nrhs( 1 );
   int lda ( numeric_cast<int>( (~A).spacing() ) );
   int ldb ( numeric_cast<int>( (~b).size() ) );
   int info( 0 );

   if( n == 0 ) {
      return;
   }

   if( IsRowMajorMatrix<MT>::value ) {
      ( uplo == 'L' )?( uplo = 'U' ):( uplo = 'L' );
   }

   trtrs( uplo, trans, diag, n, nrhs, (~A).data(), lda, (~b).data(), ldb, &info );

   BLAZE_INTERNAL_ASSERT( info == 0, "Invalid function argument" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the substitution step of solving a triangular linear system of equations
//        (\f$ A*X=B \f$).
// \ingroup lapack_substitution
//
// \param A The system matrix.
// \param B The matrix of right-hand sides.
// \param uplo \c 'L' in case of a lower matrix, \c 'U' in case of an upper matrix.
// \param trans \c 'N' for \f$ A*x=b \f$, \c 'T' for \f$ A^T*x=b \f$, and \c C for \f$ A^H*x=b \f$.
// \param diag \c 'U' in case of a unitriangular matrix, \c 'N' otherwise.
// \return void
// \exception std::invalid_argument Invalid non-square matrix provided.
// \exception std::invalid_argument Invalid uplo argument provided.
// \exception std::invalid_argument Invalid trans argument provided.
// \exception std::invalid_argument Invalid diag argument provided.
// \exception std::invalid_argument Matrix sizes do not match.
//
// This function uses the LAPACK getrs() functions to perform the substitution step to compute
// the solution to the triangular system of linear equations:
//
//  - \f$ A  *X  =B   \f$ if both \a A and \a B are column-major
//  - \f$ A^T*X  =B   \f$ if \a A is row-major and \a B is column-major
//  - \f$ A  *X^T=B^T \f$ if \a A is column-major and \a B is row-major
//  - \f$ A^T*X^T=B^T \f$ if both \a A and \a B are row-major
//
// In this context the triangular system matrix \a A is a n-by-n matrix and \a X and \a B are
// either row-major m-by-n matrices or column-major n-by-m matrices. Note that the function only
// works for general, non-adapted matrices with \c float, \c double, \c complex<float>, or
// \c complex<double> element type. The attempt to call the function with adaptors or matrices
// of any other element type results in a compile time error!
//
// If the function exits successfully, the matrix \a B contains the solutions of the linear system
// of equations. The function fails if ...
//
//  - ... the given system matrix is not a square matrix;
//  - ... the given \a uplo argument is neither \c 'L' nor \c 'U';
//  - ... the given \a trans argument is neither \c 'N' nor \c 'T' nor \c 'C';
//  - ... the given \a diag argument is neither \c 'U' nor \c 'N';
//  - ... the sizes of the two given matrices do not match.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// Examples:

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::columnMajor;
   using blaze::columnVector;

   DynamicMatrix<double,columnMajor> A( 2UL, 2UL );  // The system matrix A
   DynamicMatrix<double,columnMajor> B( 2UL, 4UL );  // The right-hand side matrix B
   // ... Initialization

   DynamicMatrix<double,columnMajor> D( A );  // Temporary matrix to be decomposed
   DynamicMatrix<double,columnMajor> X( B );  // Temporary matrix for the solution

   trtrs( D, X, 'L', 'N', 'N' );

   assert( A * X == B );
   \endcode

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::rowMajor;
   using blaze::columnVector;

   DynamicMatrix<double,rowMajor> A( 2UL, 2UL );  // The system matrix A
   DynamicMatrix<double,rowMajor> B( 2UL, 4UL );  // The right-hand side matrix B
   // ... Initialization

   DynamicMatrix<double,rowMajor> D( A );  // Temporary matrix to be decomposed
   DynamicMatrix<double,rowMajor> X( B );  // Temporary matrix for the solution

   trtrs( D, X, 'L', 'N', 'N' );

   assert( trans( A ) * trans( X ) == trans( B ) );
   \endcode

// For more information on the trtrs() functions (i.e. strtrs(), dtrtrs(), ctrtrs(), and ztrtrs()),
// see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
template< typename MT1  // Type of the system matrix
        , bool SO1      // Storage order of the system matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline void trtrs( const DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& B,
                   char uplo, char trans, char diag )
{
   using boost::numeric_cast;

   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT1 );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT2 );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT1> );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ElementType_<MT1>, ElementType_<MT2> );

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

   int n   ( numeric_cast<int>( (~A).rows()    ) );
   int mrhs( numeric_cast<int>( SO2 ? (~B).rows() : (~B).columns() ) );
   int nrhs( numeric_cast<int>( SO2 ? (~B).columns() : (~B).rows() ) );
   int lda ( numeric_cast<int>( (~A).spacing() ) );
   int ldb ( numeric_cast<int>( (~B).spacing() ) );
   int info( 0 );

   if( n != mrhs ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( n == 0 ) {
      return;
   }

   if( IsRowMajorMatrix<MT1>::value ) {
      ( uplo == 'L' )?( uplo = 'U' ):( uplo = 'L' );
   }

   trtrs( uplo, trans, diag, n, nrhs, (~A).data(), lda, (~B).data(), ldb, &info );

   BLAZE_INTERNAL_ASSERT( info == 0, "Invalid function argument" );
}
//*************************************************************************************************

} // namespace blaze

#endif
