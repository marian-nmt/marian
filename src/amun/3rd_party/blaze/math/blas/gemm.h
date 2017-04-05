//=================================================================================================
/*!
//  \file blaze/math/blas/gemm.h
//  \brief Header file for BLAS general matrix/matrix multiplication functions (gemm)
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

#ifndef _BLAZE_MATH_BLAS_GEMM_H_
#define _BLAZE_MATH_BLAS_GEMM_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <boost/cast.hpp>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/ConstDataAccess.h>
#include <blaze/math/constraints/MutableDataAccess.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/system/BLAS.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Complex.h>


namespace blaze {

//=================================================================================================
//
//  BLAS WRAPPER FUNCTIONS (GEMM)
//
//=================================================================================================

//*************************************************************************************************
/*!\name BLAS wrapper functions (gemm) */
//@{
#if BLAZE_BLAS_MODE

BLAZE_ALWAYS_INLINE void gemm( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                               int m, int n, int k, float alpha, const float* A, int lda,
                               const float* B, int ldb, float beta, float* C, int ldc );

BLAZE_ALWAYS_INLINE void gemm( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                               int m, int n, int k, double alpha, const double* A, int lda,
                               const double* B, int ldb, double beta, float* C, int ldc );

BLAZE_ALWAYS_INLINE void gemm( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                               int m, int n, int k, complex<float> alpha, const complex<float>* A,
                               int lda, const complex<float>* B, int ldb, complex<float> beta,
                               float* C, int ldc );

BLAZE_ALWAYS_INLINE void gemm( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                               int m, int n, int k, complex<double> alpha, const complex<double>* A,
                               int lda, const complex<double>* B, int ldb, complex<double> beta,
                               float* C, int ldc );

template< typename MT1, bool SO1, typename MT2, bool SO2, typename MT3, bool SO3, typename ST >
BLAZE_ALWAYS_INLINE void gemm( DenseMatrix<MT1,SO1>& C, const DenseMatrix<MT2,SO2>& A,
                               const DenseMatrix<MT3,SO3>& B, ST alpha, ST beta );

#endif
//@}
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_BLAS_MODE
/*!\brief BLAS kernel for a dense matrix/dense matrix multiplication with single precision
//        matrices (\f$ C=\alpha*A*B+\beta*C \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param transB Specifies whether to transpose matrix \a B (\a CblasNoTrans or \a CblasTrans).
// \param m The number of rows of matrix \a A and \a C \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a B and \a C \f$[0..\infty)\f$.
// \param k The number of columns of matrix \a A and rows in matrix \a B \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*B \f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of matrix \a B.
// \param ldb The total number of elements between two rows/columns of matrix \a B \f$[0..\infty)\f$.
// \param beta The scaling factor for \f$ C \f$.
// \param C Pointer to the first element of matrix \a C.
// \param ldc The total number of elements between two rows/columns of matrix \a C \f$[0..\infty)\f$.
// \return void
//
// This function performs the dense matrix/dense matrix multiplication for single precision
// matrices based on the BLAS cblas_sgemm() function.
*/
BLAZE_ALWAYS_INLINE void gemm( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                               int m, int n, int k, float alpha, const float* A, int lda,
                               const float* B, int ldb, float beta, float* C, int ldc )
{
   cblas_sgemm( order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_BLAS_MODE
/*!\brief BLAS kernel for a dense matrix/dense matrix multiplication with double precision
//        matrices (\f$ C=\alpha*A*B+\beta*C \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param transB Specifies whether to transpose matrix \a B (\a CblasNoTrans or \a CblasTrans).
// \param m The number of rows of matrix \a A and \a C \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a B and \a C \f$[0..\infty)\f$.
// \param k The number of columns of matrix \a A and rows in matrix \a B \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*B \f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of matrix \a B.
// \param ldb The total number of elements between two rows/columns of matrix \a B \f$[0..\infty)\f$.
// \param beta The scaling factor for \f$ C \f$.
// \param C Pointer to the first element of matrix \a C.
// \param ldc The total number of elements between two rows/columns of matrix \a C \f$[0..\infty)\f$.
// \return void
//
// This function performs the dense matrix/dense matrix multiplication for double precision
// matrices based on the BLAS cblas_dgemm() function.
*/
BLAZE_ALWAYS_INLINE void gemm( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                               int m, int n, int k, double alpha, const double* A, int lda,
                               const double* B, int ldb, double beta, double* C, int ldc )
{
   cblas_dgemm( order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_BLAS_MODE
/*!\brief BLAS kernel for a dense matrix/dense matrix multiplication with single precision
//        matrices (\f$ C=\alpha*A*B+\beta*C \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param transB Specifies whether to transpose matrix \a B (\a CblasNoTrans or \a CblasTrans).
// \param m The number of rows of matrix \a A and \a C \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a B and \a C \f$[0..\infty)\f$.
// \param k The number of columns of matrix \a A and rows in matrix \a B \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*B \f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of matrix \a B.
// \param ldb The total number of elements between two rows/columns of matrix \a B \f$[0..\infty)\f$.
// \param beta The scaling factor for \f$ C \f$.
// \param C Pointer to the first element of matrix \a C.
// \param ldc The total number of elements between two rows/columns of matrix \a C \f$[0..\infty)\f$.
// \return void
//
// This function performs the dense matrix/dense matrix multiplication for single precision
// complex matrices based on the BLAS cblas_cgemm() function.
*/
BLAZE_ALWAYS_INLINE void gemm( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                               int m, int n, int k, complex<float> alpha, const complex<float>* A,
                               int lda, const complex<float>* B, int ldb, complex<float> beta,
                               complex<float>* C, int ldc )
{
   cblas_cgemm( order, transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_BLAS_MODE
/*!\brief BLAS kernel for a dense matrix/dense matrix multiplication with double precision
//        matrices (\f$ C=\alpha*A*B+\beta*C \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param transB Specifies whether to transpose matrix \a B (\a CblasNoTrans or \a CblasTrans).
// \param m The number of rows of matrix \a A and \a C \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a B and \a C \f$[0..\infty)\f$.
// \param k The number of columns of matrix \a A and rows in matrix \a B \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*B \f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of matrix \a B.
// \param ldb The total number of elements between two rows/columns of matrix \a B \f$[0..\infty)\f$.
// \param beta The scaling factor for \f$ C \f$.
// \param C Pointer to the first element of matrix \a C.
// \param ldc The total number of elements between two rows/columns of matrix \a C \f$[0..\infty)\f$.
// \return void
//
// This function performs the dense matrix/dense matrix multiplication for double precision
// complex matrices based on the BLAS cblas_zgemm() function.
*/
BLAZE_ALWAYS_INLINE void gemm( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                               int m, int n, int k, complex<double> alpha, const complex<double>* A,
                               int lda, const complex<double>* B, int ldb, complex<double> beta,
                               complex<double>* C, int ldc )
{
   cblas_zgemm( order, transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_BLAS_MODE
/*!\brief BLAS kernel for a dense matrix/dense matrix multiplication (\f$ C=\alpha*A*B+\beta*C \f$).
// \ingroup blas
//
// \param C The target left-hand side dense matrix.
// \param A The left-hand side multiplication operand.
// \param B The right-hand side multiplication operand.
// \param alpha The scaling factor for \f$ A*B \f$.
// \param beta The scaling factor for \f$ C \f$.
// \return void
//
// This function performs the dense matrix/dense matrix multiplication based on the BLAS
// gemm() functions. Note that the function only works for matrices with \c float, \c double,
// \c complex<float>, and \c complex<double> element type. The attempt to call the function
// with matrices of any other element type results in a compile time error.
*/
template< typename MT1   // Type of the left-hand side target matrix
        , bool SO1       // Storage order of the left-hand side target matrix
        , typename MT2   // Type of the left-hand side matrix operand
        , bool SO2       // Storage order of the left-hand side matrix operand
        , typename MT3   // Type of the right-hand side matrix operand
        , bool SO3       // Storage order of the right-hand side matrix operand
        , typename ST >  // Type of the scalar factors
BLAZE_ALWAYS_INLINE void gemm( DenseMatrix<MT1,SO1>& C, const DenseMatrix<MT2,SO2>& A,
                               const DenseMatrix<MT3,SO3>& B, ST alpha, ST beta )
{
   using boost::numeric_cast;

   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT3 );

   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT1 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS  ( MT2 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS  ( MT3 );

   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT1> );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT2> );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT3> );

   const int m  ( numeric_cast<int>( (~A).rows() )    );
   const int n  ( numeric_cast<int>( (~B).columns() ) );
   const int k  ( numeric_cast<int>( (~A).columns() ) );
   const int lda( numeric_cast<int>( (~A).spacing() ) );
   const int ldb( numeric_cast<int>( (~B).spacing() ) );
   const int ldc( numeric_cast<int>( (~C).spacing() ) );

   gemm( ( IsRowMajorMatrix<MT1>::value )?( CblasRowMajor ):( CblasColMajor ),
         ( SO1 == SO2 )?( CblasNoTrans ):( CblasTrans ),
         ( SO1 == SO3 )?( CblasNoTrans ):( CblasTrans ),
         m, n, k, alpha, (~A).data(), lda, (~B).data(), ldb, beta, (~C).data(), ldc );
}
#endif
//*************************************************************************************************

} // namespace blaze

#endif
