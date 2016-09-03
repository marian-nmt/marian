//=================================================================================================
/*!
//  \file blaze/math/lapack/clapack/getri.h
//  \brief Header file for the CLAPACK getri wrapper functions
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

#ifndef _BLAZE_MATH_LAPACK_CLAPACK_GETRI_H_
#define _BLAZE_MATH_LAPACK_CLAPACK_GETRI_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/Complex.h>
#include <blaze/util/StaticAssert.h>


namespace blaze {

//=================================================================================================
//
//  LAPACK FORWARD DECLARATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
extern "C" {

void sgetri_( int* n, float*  A, int* lda, int* ipiv, float*  work, int* lwork, int* info );
void dgetri_( int* n, double* A, int* lda, int* ipiv, double* work, int* lwork, int* info );
void cgetri_( int* n, float*  A, int* lda, int* ipiv, float*  work, int* lwork, int* info );
void zgetri_( int* n, double* A, int* lda, int* ipiv, double* work, int* lwork, int* info );

}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LAPACK LU-BASED INVERSION FUNCTIONS (GETRI)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK LU-based inversion functions (getri) */
//@{
inline void getri( int n, float* A, int lda, const int* ipiv, float* work, int lwork, int* info );

inline void getri( int n, double* A, int lda, const int* ipiv, double* work, int lwork, int* info );

inline void getri( int n, complex<float>* A, int lda, const int* ipiv,
                   complex<float>* work, int lwork, int* info );

inline void getri( int n, complex<double>* A, int lda, const int* ipiv,
                   complex<double>* work, int lwork, int* info );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense general single precision column-major
//        square matrix.
// \ingroup lapack_inversion
//
// \param n The number of rows/columns of the matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision column-major square matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array for the pivot indices; size >= min( \a m, \a n ).
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK sgetri() function for
// single precision column-major matrices that have already been factorized by the sgetrf()
// function. The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the inversion could not be computed since U(i,i) is exactly zero.
//
// If the function exits successfully (i.e. \a info = 0) then the first element of the \a work
// array returns the optimal \a lwork. For optimal performance \a lwork >= N*NB, where NB is the
// optimal blocksize returned by the LAPACK function ilaenv(). If \a lwork = -1 then a workspace
// query is assumed. The function only calculates the optimal size of the \a work array and
// returns this value as the first entry of the \a work array.
//
// For more information on the sgetri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void getri( int n, float* A, int lda, const int* ipiv, float* work, int lwork, int* info )
{
   sgetri_( &n, A, &lda, const_cast<int*>( ipiv ), work, &lwork, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense general double precision column-major
//        square matrix.
// \ingroup lapack_inversion
//
// \param n The number of rows/columns of the matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision column-major square matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array for the pivot indices; size >= min( \a m, \a n ).
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK dgetri() function for
// double precision column-major matrices that have already been factorized by the dgetrf()
// function. The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the inversion could not be computed since U(i,i) is exactly zero.
//
// If the function exits successfully (i.e. \a info = 0) then the first element of the \a work
// array returns the optimal \a lwork. For optimal performance \a lwork >= N*NB, where NB is the
// optimal blocksize returned by the LAPACK function ilaenv(). If \a lwork = -1 then a workspace
// query is assumed. The function only calculates the optimal size of the \a work array and
// returns this value as the first entry of the \a work array.
//
// For more information on the sgetri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void getri( int n, double* A, int lda, const int* ipiv, double* work, int lwork, int* info )
{
   dgetri_( &n, A, &lda, const_cast<int*>( ipiv ), work, &lwork, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense general single precision complex
//        column-major square matrix.
// \ingroup lapack_inversion
//
// \param n The number of rows/columns of the matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision complex column-major square matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array for the pivot indices; size >= min( \a m, \a n ).
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK cgetri() function for
// single precision complex column-major matrices that have already been factorized by the
// cgetrf() function. The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the inversion could not be computed since U(i,i) is exactly zero.
//
// If the function exits successfully (i.e. \a info = 0) then the first element of the \a work
// array returns the optimal \a lwork. For optimal performance \a lwork >= N*NB, where NB is the
// optimal blocksize returned by the LAPACK function ilaenv(). If \a lwork = -1 then a workspace
// query is assumed. The function only calculates the optimal size of the \a work array and
// returns this value as the first entry of the \a work array.
//
// For more information on the sgetri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void getri( int n, complex<float>* A, int lda, const int* ipiv,
                   complex<float>* work, int lwork, int* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

   cgetri_( &n, reinterpret_cast<float*>( A ), &lda, const_cast<int*>( ipiv ),
            reinterpret_cast<float*>( work ), &lwork, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense general double precision complex
//        column-major square matrix.
// \ingroup lapack_inversion
//
// \param n The number of rows/columns of the matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision complex column-major square matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array for the pivot indices; size >= min( \a m, \a n ).
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK cgetri() function for
// double precision complex column-major matrices that have already been factorized by the
// zgetrf() function. The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the inversion could not be computed since U(i,i) is exactly zero.
//
// If the function exits successfully (i.e. \a info = 0) then the first element of the \a work
// array returns the optimal \a lwork. For optimal performance \a lwork >= N*NB, where NB is the
// optimal blocksize returned by the LAPACK function ilaenv(). If \a lwork = -1 then a workspace
// query is assumed. The function only calculates the optimal size of the \a work array and
// returns this value as the first entry of the \a work array.
//
// For more information on the sgetri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void getri( int n, complex<double>* A, int lda, const int* ipiv,
                   complex<double>* work, int lwork, int* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

   zgetri_( &n, reinterpret_cast<double*>( A ), &lda, const_cast<int*>( ipiv ),
            reinterpret_cast<double*>( work ), &lwork, info );
}
//*************************************************************************************************

} // namespace blaze

#endif
