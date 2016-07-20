//=================================================================================================
/*!
//  \file blaze/math/lapack/clapack/trtri.h
//  \brief Header file for the CLAPACK trtri wrapper functions
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

#ifndef _BLAZE_MATH_LAPACK_CLAPACK_TRTRI_H_
#define _BLAZE_MATH_LAPACK_CLAPACK_TRTRI_H_


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

void strtri_( char* uplo, char* diag, int* n, float*  A, int* lda, int* info );
void dtrtri_( char* uplo, char* diag, int* n, double* A, int* lda, int* info );
void ctrtri_( char* uplo, char* diag, int* n, float*  A, int* lda, int* info );
void ztrtri_( char* uplo, char* diag, int* n, double* A, int* lda, int* info );

}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LAPACK TRIANGULAR MATRIX INVERSION FUNCTIONS (TRTRI)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK triangular matrix inversion functions (trtri) */
//@{
inline void trtri( char uplo, char diag, int n, float* A, int lda, int* info );

inline void trtri( char uplo, char diag, int n, double* A, int lda, int* info );

inline void trtri( char uplo, char diag, int n, complex<float>* A, int lda, int* info );

inline void trtri( char uplo, char diag, int n, complex<double>* A, int lda, int* info );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense triangular single precision
//        column-major matrix.
// \ingroup lapack_inversion
//
// \param uplo \c 'L' in case of a lower matrix, \c 'U' in case of an upper matrix.
// \param diag \c 'U' in case of a unitriangular matrix, \c 'N' otherwise.
// \param n The number of rows/columns of the triangular matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK strtri() function for
// lower triangular (\a uplo = \c 'L') or upper triangular (\a uplo = \a 'U') single precision
// column-major matrices.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If \a info = -i, the i-th argument had an illegal value.
//   - > 0: If \a info = i, element A(i,i) is exactly zero and the inverse could not be computed.
//
// For more information on the strtri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void trtri( char uplo, char diag, int n, float* A, int lda, int* info )
{
   strtri_( &uplo, &diag, &n, A, &lda, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense triangular double precision
//        column-major matrix.
// \ingroup lapack_inversion
//
// \param uplo \c 'L' in case of a lower matrix, \c 'U' in case of an upper matrix.
// \param diag \c 'U' in case of a unitriangular matrix, \c 'N' otherwise.
// \param n The number of rows/columns of the triangular matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK dtrtri() function for
// lower triangular (\a uplo = \c 'L') or upper triangular (\a uplo = \a 'U') double precision
// column-major matrices.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If \a info = -i, the i-th argument had an illegal value.
//   - > 0: If \a info = i, element A(i,i) is exactly zero and the inverse could not be computed.
//
// For more information on the dtrtri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void trtri( char uplo, char diag, int n, double* A, int lda, int* info )
{
   dtrtri_( &uplo, &diag, &n, A, &lda, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense triangular single precision complex
//        column-major matrix.
// \ingroup lapack_inversion
//
// \param uplo \c 'L' in case of a lower matrix, \c 'U' in case of an upper matrix.
// \param diag \c 'U' in case of a unitriangular matrix, \c 'N' otherwise.
// \param n The number of rows/columns of the triangular matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision complex column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK ctrtri() function for
// lower triangular (\a uplo = \c 'L') or upper triangular (\a uplo = \a 'U') single precision
// complex column-major matrices.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If \a info = -i, the i-th argument had an illegal value.
//   - > 0: If \a info = i, element A(i,i) is exactly zero and the inverse could not be computed.
//
// For more information on the ctrtri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void trtri( char uplo, char diag, int n, complex<float>* A, int lda, int* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

   ctrtri_( &uplo, &diag, &n, reinterpret_cast<float*>( A ), &lda, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense triangular double precision complex
//        column-major matrix.
// \ingroup lapack_inversion
//
// \param uplo \c 'L' in case of a lower matrix, \c 'U' in case of an upper matrix.
// \param diag \c 'U' in case of a unitriangular matrix, \c 'N' otherwise.
// \param n The number of rows/columns of the triangular matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision complex column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK ztrtri() function for
// lower triangular (\a uplo = \c 'L') or upper triangular (\a uplo = \a 'U') double precision
// complex column-major matrices.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If \a info = -i, the i-th argument had an illegal value.
//   - > 0: If \a info = i, element A(i,i) is exactly zero and the inverse could not be computed.
//
// For more information on the ztrtri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void trtri( char uplo, char diag, int n, complex<double>* A, int lda, int* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

   ztrtri_( &uplo, &diag, &n, reinterpret_cast<double*>( A ), &lda, info );
}
//*************************************************************************************************

} // namespace blaze

#endif
