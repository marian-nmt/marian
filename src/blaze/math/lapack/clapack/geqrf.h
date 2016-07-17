//=================================================================================================
/*!
//  \file blaze/math/lapack/clapack/geqrf.h
//  \brief Header file for the CLAPACK geqrf wrapper functions
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

#ifndef _BLAZE_MATH_LAPACK_CLAPACK_GEQRF_H_
#define _BLAZE_MATH_LAPACK_CLAPACK_GEQRF_H_


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

void sgeqrf_( int* m, int* n, float*  A, int* lda, float*  tau, float*  work, int* lwork, int* info );
void dgeqrf_( int* m, int* n, double* A, int* lda, double* tau, double* work, int* lwork, int* info );
void cgeqrf_( int* m, int* n, float*  A, int* lda, float*  tau, float*  work, int* lwork, int* info );
void zgeqrf_( int* m, int* n, double* A, int* lda, double* tau, double* work, int* lwork, int* info );

}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LAPACK QR DECOMPOSITION FUNCTIONS (GEQRF)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK QR decomposition functions (geqrf) */
//@{
inline void geqrf( int m, int n, float* A, int lda, float* tau,
                   float* work, int lwork, int* info );

inline void geqrf( int m, int n, double* A, int lda, double* tau,
                   double* work, int lwork, int* info );

inline void geqrf( int m, int n, complex<float>* A, int lda, complex<float>* tau,
                   complex<float>* work, int lwork, int* info );

inline void geqrf( int m, int n, complex<double>* A, int lda, complex<double>* tau,
                   complex<double>* work, int lwork, int* info );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the QR decomposition of the given dense single precision column-major
//        matrix.
// \ingroup lapack_decomposition
//
// \param m The number of rows of the given matrix \f$[0..\infty)\f$.
// \param n The number of columns of the given matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param tau Array for the scalar factors of the elementary reflectors; size >= min( \a m, \a n ).
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix QR decomposition of a general \a m-by-\a n single
// precision column-major matrix based on the LAPACK sgeqrf() function. The resulting decomposition
// has the form

                              \f[ A = Q \cdot R, \f]

// where the \c Q is represented as a product of elementary reflectors

               \f[ Q = H(1) H(2) . . . H(k) \texttt{, with k = min(m,n).} \f]

// Each H(i) has the form

                      \f[ H(i) = I - tau \cdot v \cdot v^T, \f]

// where \c tau is a real scalar, and \c v is a real vector with <tt>v(0:i-1) = 0</tt> and
// <tt>v(i) = 1</tt>. <tt>v(i+1:m)</tt> is stored on exit in <tt>A(i+1:m,i)</tt>, and \c tau
// in \c tau(i). Thus on exit the elements on and above the diagonal of the matrix contain the
// min(\a m,\a n)-by-\a n upper trapezoidal matrix \c R (\c R is upper triangular if \a m >= \a n);
// the elements below the diagonal, with the array \c tau, represent the orthogonal matrix \c Q
// as a product of min(\a m,\a n) elementary reflectors.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: The i-th argument had an illegal value.
//
// For more information on the sgeqrf() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void geqrf( int m, int n, float* A, int lda, float* tau,
                   float* work, int lwork, int* info )
{
   sgeqrf_( &m, &n, A, &lda, tau, work, &lwork, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the QR decomposition of the given dense double precision column-major
//        matrix.
// \ingroup lapack_decomposition
//
// \param m The number of rows of the given matrix \f$[0..\infty)\f$.
// \param n The number of columns of the given matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param tau Array for the scalar factors of the elementary reflectors; size >= min( \a m, \a n ).
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix QR decomposition of a general \a m-by-\a n double
// precision column-major matrix based on the LAPACK dgeqrf() function. The resulting decomposition
// has the form

                              \f[ A = Q \cdot R, \f]

// where the \c Q is represented as a product of elementary reflectors

               \f[ Q = H(1) H(2) . . . H(k) \texttt{, with k = min(m,n).} \f]

// Each H(i) has the form

                      \f[ H(i) = I - tau \cdot v \cdot v^T, \f]

// where \c tau is a real scalar, and \c v is a real vector with <tt>v(0:i-1) = 0</tt> and
// <tt>v(i) = 1</tt>. <tt>v(i+1:m)</tt> is stored on exit in <tt>A(i+1:m,i)</tt>, and \c tau
// in \c tau(i). Thus on exit the elements on and above the diagonal of the matrix contain the
// min(\a m,\a n)-by-\a n upper trapezoidal matrix \c R (\c R is upper triangular if \a m >= \a n);
// the elements below the diagonal, with the array \c tau, represent the orthogonal matrix \c Q
// as a product of min(\a m,\a n) elementary reflectors.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: The i-th argument had an illegal value.
//
// For more information on the dgeqrf() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void geqrf( int m, int n, double* A, int lda, double* tau,
                   double* work, int lwork, int* info )
{
   dgeqrf_( &m, &n, A, &lda, tau, work, &lwork, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the QR decomposition of the given dense single precision complex
//        column-major matrix.
// \ingroup lapack_decomposition
//
// \param m The number of rows of the given matrix \f$[0..\infty)\f$.
// \param n The number of columns of the given matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision complex column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param tau Array for the scalar factors of the elementary reflectors; size >= min( \a m, \a n ).
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix QR decomposition of a general \a m-by-\a n single
// precision complex column-major matrix based on the LAPACK cgeqrf() function. The resulting
// decomposition has the form

                              \f[ A = Q \cdot R, \f]

// where the \c Q is represented as a product of elementary reflectors

               \f[ Q = H(1) H(2) . . . H(k) \texttt{, with k = min(m,n).} \f]

// Each H(i) has the form

                      \f[ H(i) = I - tau \cdot v \cdot v^T, \f]

// where \c tau is a real scalar, and \c v is a real vector with <tt>v(0:i-1) = 0</tt> and
// <tt>v(i) = 1</tt>. <tt>v(i+1:m)</tt> is stored on exit in <tt>A(i+1:m,i)</tt>, and \c tau
// in \c tau(i). Thus on exit the elements on and above the diagonal of the matrix contain the
// min(\a m,\a n)-by-\a n upper trapezoidal matrix \c R (\c R is upper triangular if \a m >= \a n);
// the elements below the diagonal, with the array \c tau, represent the orthogonal matrix \c Q
// as a product of min(\a m,\a n) elementary reflectors.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: The i-th argument had an illegal value.
//
// For more information on the cgeqrf() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void geqrf( int m, int n, complex<float>* A, int lda, complex<float>* tau,
                   complex<float>* work, int lwork, int* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

   cgeqrf_( &m, &n, reinterpret_cast<float*>( A ), &lda, reinterpret_cast<float*>( tau ),
            reinterpret_cast<float*>( work ), &lwork, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the QR decomposition of the given dense double precision complex
//        column-major matrix.
// \ingroup lapack_decomposition
//
// \param m The number of rows of the given matrix \f$[0..\infty)\f$.
// \param n The number of columns of the given matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision complex column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param tau Array for the scalar factors of the elementary reflectors; size >= min( \a m, \a n ).
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix QR decomposition of a general \a m-by-\a n double
// precision complex column-major matrix based on the LAPACK zgeqrf() function. The resulting
// decomposition has the form

                              \f[ A = Q \cdot R, \f]

// where the \c Q is represented as a product of elementary reflectors

               \f[ Q = H(1) H(2) . . . H(k) \texttt{, with k = min(m,n).} \f]

// Each H(i) has the form

                      \f[ H(i) = I - tau \cdot v \cdot v^T, \f]

// where \c tau is a real scalar, and \c v is a real vector with <tt>v(0:i-1) = 0</tt> and
// <tt>v(i) = 1</tt>. <tt>v(i+1:m)</tt> is stored on exit in <tt>A(i+1:m,i)</tt>, and \c tau
// in \c tau(i). Thus on exit the elements on and above the diagonal of the matrix contain the
// min(\a m,\a n)-by-\a n upper trapezoidal matrix \c R (\c R is upper triangular if \a m >= \a n);
// the elements below the diagonal, with the array \c tau, represent the orthogonal matrix \c Q
// as a product of min(\a m,\a n) elementary reflectors.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: The i-th argument had an illegal value.
//
// For more information on the zgeqrf() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void geqrf( int m, int n, complex<double>* A, int lda, complex<double>* tau,
                   complex<double>* work, int lwork, int* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

   zgeqrf_( &m, &n, reinterpret_cast<double*>( A ), &lda, reinterpret_cast<double*>( tau ),
            reinterpret_cast<double*>( work ), &lwork, info );
}
//*************************************************************************************************

} // namespace blaze

#endif
