//=================================================================================================
/*!
//  \file blaze/math/blas/trmm.h
//  \brief Header file for BLAS triangular matrix/matrix multiplication functions (trmm)
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

#ifndef _BLAZE_MATH_BLAS_TRMM_H_
#define _BLAZE_MATH_BLAS_TRMM_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <boost/cast.hpp>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/ConstDataAccess.h>
#include <blaze/math/constraints/MutableDataAccess.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/system/BLAS.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Complex.h>


namespace blaze {

//=================================================================================================
//
//  BLAS WRAPPER FUNCTIONS (TRMM)
//
//=================================================================================================

//*************************************************************************************************
/*!\name BLAS wrapper functions (trmm) */
//@{
#if BLAZE_BLAS_MODE

BLAZE_ALWAYS_INLINE void trmm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo,
                               CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n,
                               float alpha, const float* A, int lda, float* B, int ldb );

BLAZE_ALWAYS_INLINE void trmm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo,
                               CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n,
                               double alpha, const double* A, int lda, double* B, int ldb );

BLAZE_ALWAYS_INLINE void trmm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo,
                               CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n,
                               complex<float> alpha, const complex<float>* A, int lda,
                               complex<float>* B, int ldb );

BLAZE_ALWAYS_INLINE void trmm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo,
                               CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n,
                               complex<double> alpha, const complex<double>* A, int lda,
                               complex<double>* B, int ldb );

template< typename MT1, bool SO1, typename MT2, bool SO2, typename ST >
BLAZE_ALWAYS_INLINE void trmm( DenseMatrix<MT1,SO1>& B, const DenseMatrix<MT2,SO2>& A,
                               CBLAS_SIDE side, CBLAS_UPLO uplo, ST alpha );

#endif
//@}
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_BLAS_MODE
/*!\brief BLAS kernel for a triangular dense matrix/dense matrix multiplication with single
//        precision matrices (\f$ B=\alpha*A*B \f$ or \f$ B=\alpha*B*A \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasRowMajor or \a CblasColMajor).
// \param side \a CblasLeft to compute \f$ B=\alpha*A*B \f$, \a CblasRight to compute \f$ B=\alpha*B*A \f$.
// \param uplo \a CblasLower to use the lower triangle from \a A, \a CblasUpper to use the upper triangle.
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param diag Specifies whether \a A is unitriangular (\a CblasNonUnit or \a CblasUnit).
// \param m The number of rows of matrix \a B \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a B \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*B \f$ or \f$ B*A \f$.
// \param A Pointer to the first element of the triangular matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of matrix \a B.
// \param ldb The total number of elements between two rows/columns of matrix \a B \f$[0..\infty)\f$.
// \return void
//
// This function performs the scaling and multiplication of a triangular matrix by a matrix
// based on the cblas_strmm() function. Note that matrix \a A is expected to be a square matrix.
*/
BLAZE_ALWAYS_INLINE void trmm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo,
                               CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n,
                               float alpha, const float* A, int lda, float* B, int ldb )
{
   cblas_strmm( order, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_BLAS_MODE
/*!\brief BLAS kernel for a triangular dense matrix/dense matrix multiplication with double
//        precision matrices (\f$ B=\alpha*A*B \f$ or \f$ B=\alpha*B*A \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasRowMajor or \a CblasColMajor).
// \param side \a CblasLeft to compute \f$ B=\alpha*A*B \f$, \a CblasRight to compute \f$ B=\alpha*B*A \f$.
// \param uplo \a CblasLower to use the lower triangle from \a A, \a CblasUpper to use the upper triangle.
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param diag Specifies whether \a A is unitriangular (\a CblasNonUnit or \a CblasUnit).
// \param m The number of rows of matrix \a B \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a B \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*B \f$ or \f$ B*A \f$.
// \param A Pointer to the first element of the triangular matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of matrix \a B.
// \param ldb The total number of elements between two rows/columns of matrix \a B \f$[0..\infty)\f$.
// \return void
//
// This function performs the scaling and multiplication of a triangular matrix by a matrix
// based on the cblas_dtrmm() function. Note that matrix \a A is expected to be a square matrix.
*/
BLAZE_ALWAYS_INLINE void trmm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo,
                               CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n,
                               double alpha, const double* A, int lda, double* B, int ldb )
{
   cblas_dtrmm( order, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_BLAS_MODE
/*!\brief BLAS kernel for a triangular dense matrix/dense matrix multiplication with single
//        precision complex matrices (\f$ B=\alpha*A*B \f$ or \f$ B=\alpha*B*A \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasRowMajor or \a CblasColMajor).
// \param side \a CblasLeft to compute \f$ B=\alpha*A*B \f$, \a CblasRight to compute \f$ B=\alpha*B*A \f$.
// \param uplo \a CblasLower to use the lower triangle from \a A, \a CblasUpper to use the upper triangle.
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param diag Specifies whether \a A is unitriangular (\a CblasNonUnit or \a CblasUnit).
// \param m The number of rows of matrix \a B \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a B \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*B \f$ or \f$ B*A \f$.
// \param A Pointer to the first element of the triangular matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of matrix \a B.
// \param ldb The total number of elements between two rows/columns of matrix \a B \f$[0..\infty)\f$.
// \return void
//
// This function performs the scaling and multiplication of a triangular matrix by a matrix
// based on the cblas_ctrmm() function. Note that matrix \a A is expected to be a square matrix.
*/
BLAZE_ALWAYS_INLINE void trmm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo,
                               CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n,
                               complex<float> alpha, const complex<float>* A, int lda,
                               complex<float>* B, int ldb )
{
   cblas_ctrmm( order, side, uplo, transA, diag, m, n, &alpha, A, lda, B, ldb );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_BLAS_MODE
/*!\brief BLAS kernel for a triangular dense matrix/dense matrix multiplication with double
//        precision complex matrices (\f$ B=\alpha*A*B \f$ or \f$ B=\alpha*B*A \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasRowMajor or \a CblasColMajor).
// \param side \a CblasLeft to compute \f$ B=\alpha*A*B \f$, \a CblasRight to compute \f$ B=\alpha*B*A \f$.
// \param uplo \a CblasLower to use the lower triangle from \a A, \a CblasUpper to use the upper triangle.
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param diag Specifies whether \a A is unitriangular (\a CblasNonUnit or \a CblasUnit).
// \param m The number of rows of matrix \a B \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a B \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*B \f$ or \f$ B*A \f$.
// \param A Pointer to the first element of the triangular matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of matrix \a B.
// \param ldb The total number of elements between two rows/columns of matrix \a B \f$[0..\infty)\f$.
// \return void
//
// This function performs the scaling and multiplication of a triangular matrix by a matrix
// based on the cblas_ztrmm() function. Note that matrix \a A is expected to be a square matrix.
*/
BLAZE_ALWAYS_INLINE void trmm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo,
                               CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, int m, int n,
                               complex<double> alpha, const complex<double>* A, int lda,
                               complex<double>* B, int ldb )
{
   cblas_ztrmm( order, side, uplo, transA, diag, m, n, &alpha, A, lda, B, ldb );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_BLAS_MODE
/*!\brief BLAS kernel for a triangular dense matrix/dense matrix multiplication
//        (\f$ B=\alpha*A*B \f$ or \f$ B=\alpha*B*A \f$).
// \ingroup blas
//
// \param B The target dense matrix.
// \param A The dense matrix multiplication operand.
// \param side \a CblasLeft to compute \f$ B=\alpha*A*B \f$, \a CblasRight to compute \f$ B=\alpha*B*A \f$.
// \param uplo \a CblasLower to use the lower triangle from \a A, \a CblasUpper to use the upper triangle.
// \param alpha The scaling factor for \f$ A*B \f$ or \f$ B*A \f$.
// \return void
//
// This function performs the scaling and multiplication of a triangular matrix by a matrix
// based on the BLAS trmm() functions. Note that the function only works for matrices with
// \c float, \c double, \c complex<float>, and \c complex<double> element type. The attempt to
// call the function with matrices of any other element type results in a compile time error.
// Also note that matrix \a A is expected to be a square matrix.
*/
template< typename MT1   // Type of the left-hand side target matrix
        , bool SO1       // Storage order of the left-hand side target matrix
        , typename MT2   // Type of the left-hand side matrix operand
        , bool SO2       // Storage order of the left-hand side matrix operand
        , typename ST >  // Type of the scalar factor
BLAZE_ALWAYS_INLINE void trmm( DenseMatrix<MT1,SO1>& B, const DenseMatrix<MT2,SO2>& A,
                               CBLAS_SIDE side, CBLAS_UPLO uplo, ST alpha )
{
   using boost::numeric_cast;

   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT2 );

   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT1 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS  ( MT2 );

   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT1> );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT2> );

   BLAZE_INTERNAL_ASSERT( (~A).rows() == (~A).columns(), "Non-square triangular matrix detected" );
   BLAZE_INTERNAL_ASSERT( side == CblasLeft  || side == CblasRight, "Invalid side argument detected" );
   BLAZE_INTERNAL_ASSERT( uplo == CblasLower || uplo == CblasUpper, "Invalid uplo argument detected" );

   const int m  ( numeric_cast<int>( (~B).rows() )    );
   const int n  ( numeric_cast<int>( (~B).columns() ) );
   const int lda( numeric_cast<int>( (~A).spacing() ) );
   const int ldb( numeric_cast<int>( (~B).spacing() ) );

   trmm( ( IsRowMajorMatrix<MT1>::value )?( CblasRowMajor ):( CblasColMajor ),
         side,
         ( SO1 == SO2 )?( uplo ):( ( uplo == CblasLower )?( CblasUpper ):( CblasLower ) ),
         ( SO1 == SO2 )?( CblasNoTrans ):( CblasTrans ),
         CblasNonUnit,
         m, n, alpha, (~A).data(), lda, (~B).data(), ldb );
}
#endif
//*************************************************************************************************

} // namespace blaze

#endif
