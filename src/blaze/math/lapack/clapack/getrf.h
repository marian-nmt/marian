//=================================================================================================
/*!
//  \file blaze/math/lapack/clapack/getrf.h
//  \brief Header file for the CLAPACK getrf wrapper functions
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

#ifndef _BLAZE_MATH_LAPACK_CLAPACK_GETRF_H_
#define _BLAZE_MATH_LAPACK_CLAPACK_GETRF_H_


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

void sgetrf_( int* m, int* n, float*  A, int* lda, int* ipiv, int* info );
void dgetrf_( int* m, int* n, double* A, int* lda, int* ipiv, int* info );
void cgetrf_( int* m, int* n, float*  A, int* lda, int* ipiv, int* info );
void zgetrf_( int* m, int* n, double* A, int* lda, int* ipiv, int* info );

}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LAPACK LU DECOMPOSITION FUNCTIONS (GETRF)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK LU decomposition functions (getrf) */
//@{
inline void getrf( int m, int n, float* A, int lda, int* ipiv, int* info );

inline void getrf( int m, int n, double* A, int lda, int* ipiv, int* info );

inline void getrf( int m, int n, complex<float>* A, int lda, int* ipiv, int* info );

inline void getrf( int m, int n, complex<double>* A, int lda, int* ipiv, int* info );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the LU decomposition of the given dense general single precision
//        column-major matrix.
// \ingroup lapack_decomposition
//
// \param m The number of rows of the given matrix \f$[0..\infty)\f$.
// \param n The number of columns of the given matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array for the pivot indices; size >= min( \a m, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix LU decomposition of a general m-by-n single precision
// column-major matrix based on the LAPACK sgetrf() function, which uses partial pivoting with row
// interchanges. The resulting decomposition has the form

                          \f[ A = P \cdot L \cdot U, \f]

// where \c P is a permutation matrix, \c L is a lower unitriangular matrix (lower trapezoidal if
// \a m > \a n), and \c U is an upper triangular matrix (upper trapezoidal if \a m < \a n). The
// resulting decomposition is stored within the matrix \a A: \c L is stored in the lower part of
// \a A and \c U is stored in the upper part. The unit diagonal elements of \c L are not stored.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the decomposition has been completed, but the factor U(i,i) is singular.
//
// For more information on the sgetrf() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void getrf( int m, int n, float* A, int lda, int* ipiv, int* info )
{
   sgetrf_( &m, &n, A, &lda, ipiv, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the LU decomposition of the given dense general double precision
//        column-major matrix.
// \ingroup lapack_decomposition
//
// \param m The number of rows of the given matrix \f$[0..\infty)\f$.
// \param n The number of columns of the given matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array for the pivot indices; size >= min( \a m, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix LU decomposition of a general m-by-n double precision
// column-major matrix based on the LAPACK dgetrf() function, which uses partial pivoting with row
// interchanges. The resulting decomposition has the form

                          \f[ A = P \cdot L \cdot U, \f]

// where \c P is a permutation matrix, \c L is a lower unitriangular matrix (lower trapezoidal if
// \a m > \a n), and \c U is an upper triangular matrix (upper trapezoidal if \a m < \a n). The
// resulting decomposition is stored within the matrix \a A: \c L is stored in the lower part of
// \a A and \c U is stored in the upper part. The unit diagonal elements of \c L are not stored.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the decomposition has been completed, but the factor U(i,i) is singular.
//
// For more information on the dgetrf() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void getrf( int m, int n, double* A, int lda, int* ipiv, int* info )
{
   dgetrf_( &m, &n, A, &lda, ipiv, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the LU decomposition of the given dense general single precision
//        complex column-major matrix.
// \ingroup lapack_decomposition
//
// \param m The number of rows of the given matrix \f$[0..\infty)\f$.
// \param n The number of columns of the given matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision complex column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array for the pivot indices; size >= min( \a m, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix LU decomposition of a general m-by-n single precision
// complex column-major matrix based on the LAPACK cgetrf() function, which uses partial pivoting
// with row interchanges. The resulting decomposition has the form

                          \f[ A = P \cdot L \cdot U, \f]

// where \c P is a permutation matrix, \c L is a lower unitriangular matrix (lower trapezoidal if
// \a m > \a n), and \c U is an upper triangular matrix (upper trapezoidal if \a m < \a n). The
// resulting decomposition is stored within the matrix \a A: \c L is stored in the lower part of
// \a A and \c U is stored in the upper part. The unit diagonal elements of \c L are not stored.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the decomposition has been completed, but the factor U(i,i) is singular.
//
// For more information on the cgetrf() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void getrf( int m, int n, complex<float>* A, int lda, int* ipiv, int* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

   cgetrf_( &m, &n, reinterpret_cast<float*>( A ), &lda, ipiv, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the LU decomposition of the given dense general double precision
//        complex column-major matrix.
// \ingroup lapack_decomposition
//
// \param m The number of rows of the given matrix \f$[0..\infty)\f$.
// \param n The number of columns of the given matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision complex column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array for the pivot indices; size >= min( \a m, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix LU decomposition of a general m-by-n double precision
// complex column-major matrix based on the LAPACK zgetrf() function, which uses partial pivoting
// with row interchanges. The resulting decomposition has the form

                          \f[ A = P \cdot L \cdot U, \f]

// where \c P is a permutation matrix, \c L is a lower unitriangular matrix (lower trapezoidal if
// \a m > \a n), and \c U is an upper triangular matrix (upper trapezoidal if \a m < \a n). The
// resulting decomposition is stored within the matrix \a A: \c L is stored in the lower part of
// \a A and \c U is stored in the upper part. The unit diagonal elements of \c L are not stored.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the decomposition has been completed, but the factor U(i,i) is singular.
//
// For more information on the zgetrf() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void getrf( int m, int n, complex<double>* A, int lda, int* ipiv, int* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

   zgetrf_( &m, &n, reinterpret_cast<double*>( A ), &lda, ipiv, info );
}
//*************************************************************************************************

} // namespace blaze

#endif
