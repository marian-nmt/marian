//=================================================================================================
/*!
//  \file blaze/math/lapack/clapack/getrs.h
//  \brief Header file for the CLAPACK getrs wrapper functions
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

#ifndef _BLAZE_MATH_LAPACK_CLAPACK_GETRS_H_
#define _BLAZE_MATH_LAPACK_CLAPACK_GETRS_H_


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

void sgetrs_( char* trans, int* n, int* nrhs, float*  A, int* lda, int* ipiv, float*  B, int* ldb, int* info );
void dgetrs_( char* trans, int* n, int* nrhs, double* A, int* lda, int* ipiv, double* B, int* ldb, int* info );
void cgetrs_( char* trans, int* n, int* nrhs, float*  A, int* lda, int* ipiv, float*  B, int* ldb, int* info );
void zgetrs_( char* trans, int* n, int* nrhs, double* A, int* lda, int* ipiv, double* B, int* ldb, int* info );

}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LAPACK LU-BASED SUBSTITUTION FUNCTIONS (GETRS)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK LU-based substitution functions (getrs) */
//@{
inline void getrs( char trans, int n, int nrhs, const float* A, int lda, const int* ipiv,
                   float* B, int ldb, int* info );

inline void getrs( char trans, int n, int nrhs, const double* A, int lda, const int* ipiv,
                   double* B, int ldb, int* info );

inline void getrs( char trans, int n, int nrhs, const complex<float>* A, int lda,
                   const int* ipiv, complex<float>* B, int ldb, int* info );

inline void getrs( char trans, int n, int nrhs, const complex<double>* A, int lda,
                   const int* ipiv, complex<double>* B, int ldb, int* info );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the substitution step of solving a general single precision linear
//        system of equations (\f$ A*X=B \f$).
// \ingroup lapack_substitution
//
// \param trans \c 'N' for \f$ A*X=B \f$, \c 'T' for \f$ A^T*X=B \f$, and \c C for \f$ A^H*X=B \f$.
// \param n The number of rows/columns of the column-major matrix \f$[0..\infty)\f$.
// \param nrhs The number of right-hand side vectors \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision column-major square matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param B Pointer to the first element of the column-major matrix.
// \param ldb The total number of elements between two columns of matrix \a B \f$[0..\infty)\f$.
// \param info Return code of the function call.
// \return void
//
// This function uses the LAPACK sgetrs() function to perform the substitution step to compute
// the solution to the general system of linear equations \f$ A*X=B \f$, \f$ A^{T}*X=B \f$, or
// \f$ A^{H}*X=B \f$, where \a A is a n-by-n matrix that has already been factorized by the
// sgetrf() function and \a X and \a B are column-major n-by-nrhs matrices. The \a trans argument
// specifies the form of the linear system of equations:
//
//   - 'N': \f$ A*X=B \f$ (no transpose)
//   - 'T': \f$ A^{T}*X=B \f$ (transpose)
//   - 'C': \f$ A^{H}*X=B \f$ (conjugate transpose)
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The function finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//
// For more information on the sgetrs() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void getrs( char trans, int n, int nrhs, const float* A, int lda,
                   const int* ipiv, float* B, int ldb, int* info )
{
   sgetrs_( &trans, &n, &nrhs, const_cast<float*>( A ), &lda,
            const_cast<int*>( ipiv ), B, &ldb, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the substitution step of solving a general double precision linear
//        system of equations (\f$ A*X=B \f$).
// \ingroup lapack_substitution
//
// \param trans \c 'N' for \f$ A*X=B \f$, \c 'T' for \f$ A^T*X=B \f$, and \c C for \f$ A^H*X=B \f$.
// \param n The number of rows/columns of the column-major matrix \f$[0..\infty)\f$.
// \param nrhs The number of right-hand side vectors \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision column-major square matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param B Pointer to the first element of the column-major matrix.
// \param ldb The total number of elements between two columns of matrix \a B \f$[0..\infty)\f$.
// \param info Return code of the function call.
// \return void
//
// This function uses the LAPACK dgetrs() function to perform the substitution step to compute
// the solution to the general system of linear equations \f$ A*X=B \f$, \f$ A^{T}*X=B \f$, or
// \f$ A^{H}*X=B \f$, where \a A is a n-by-n matrix that has already been factorized by the
// dgetrf() function and \a X and \a B are column-major n-by-nrhs matrices. The \a trans argument
// specifies the form of the linear system of equations:
//
//   - 'N': \f$ A*X=B \f$ (no transpose)
//   - 'T': \f$ A^{T}*X=B \f$ (transpose)
//   - 'C': \f$ A^{H}*X=B \f$ (conjugate transpose)
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The function finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//
// For more information on the dgetrs() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void getrs( char trans, int n, int nrhs, const double* A, int lda,
                   const int* ipiv, double* B, int ldb, int* info )
{
   dgetrs_( &trans, &n, &nrhs, const_cast<double*>( A ), &lda,
            const_cast<int*>( ipiv ), B, &ldb, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the substitution step of solving a general single precision complex
//        linear system of equations (\f$ A*X=B \f$).
// \ingroup lapack_substitution
//
// \param trans \c 'N' for \f$ A*X=B \f$, \c 'T' for \f$ A^T*X=B \f$, and \c C for \f$ A^H*X=B \f$.
// \param n The number of rows/columns of the column-major matrix \f$[0..\infty)\f$.
// \param nrhs The number of right-hand side vectors \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision complex column-major square matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param B Pointer to the first element of the column-major matrix.
// \param ldb The total number of elements between two columns of matrix \a B \f$[0..\infty)\f$.
// \param info Return code of the function call.
// \return void
//
// This function uses the LAPACK cgetrs() function to perform the substitution step to compute
// the solution to the general system of linear equations \f$ A*X=B \f$, \f$ A^{T}*X=B \f$, or
// \f$ A^{H}*X=B \f$, where \a A is a n-by-n matrix that has already been factorized by the
// cgetrf() function and \a X and \a B are column-major n-by-nrhs matrices. The \a trans argument
// specifies the form of the linear system of equations:
//
//   - 'N': \f$ A*X=B \f$ (no transpose)
//   - 'T': \f$ A^{T}*X=B \f$ (transpose)
//   - 'C': \f$ A^{H}*X=B \f$ (conjugate transpose)
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The function finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//
// For more information on the cgetrs() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void getrs( char trans, int n, int nrhs, const complex<float>* A, int lda,
                   const int* ipiv, complex<float>* B, int ldb, int* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

   cgetrs_( &trans, &n, &nrhs, const_cast<float*>( reinterpret_cast<const float*>( A ) ),
            &lda, const_cast<int*>( ipiv ), reinterpret_cast<float*>( B ), &ldb, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the substitution step of solving a general double precision complex
//        linear system of equations (\f$ A*X=B \f$).
// \ingroup lapack_substitution
//
// \param trans \c 'N' for \f$ A*X=B \f$, \c 'T' for \f$ A^T*X=B \f$, and \c C for \f$ A^H*X=B \f$.
// \param n The number of rows/columns of the column-major matrix \f$[0..\infty)\f$.
// \param nrhs The number of right-hand side vectors \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision complex column-major square matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param B Pointer to the first element of the column-major matrix.
// \param ldb The total number of elements between two columns of matrix \a B \f$[0..\infty)\f$.
// \param info Return code of the function call.
// \return void
//
// This function uses the LAPACK zgetrs() function to perform the substitution step to compute
// the solution to the general system of linear equations \f$ A*X=B \f$, \f$ A^{T}*X=B \f$, or
// \f$ A^{H}*X=B \f$, where \a A is a n-by-n matrix that has already been factorized by the
// zgetrf() function and \a X and \a B are column-major n-by-nrhs matrices. The \a trans argument
// specifies the form of the linear system of equations:
//
//   - 'N': \f$ A*X=B \f$ (no transpose)
//   - 'T': \f$ A^{T}*X=B \f$ (transpose)
//   - 'C': \f$ A^{H}*X=B \f$ (conjugate transpose)
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The function finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//
// For more information on the zgetrs() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void getrs( char trans, int n, int nrhs, const complex<double>* A, int lda,
                   const int* ipiv, complex<double>* B, int ldb, int* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

   zgetrs_( &trans, &n, &nrhs, const_cast<double*>( reinterpret_cast<const double*>( A ) ),
            &lda, const_cast<int*>( ipiv ), reinterpret_cast<double*>( B ), &ldb, info );
}
//*************************************************************************************************

} // namespace blaze

#endif
