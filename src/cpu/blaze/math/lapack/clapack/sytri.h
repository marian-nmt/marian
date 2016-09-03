//=================================================================================================
/*!
//  \file blaze/math/lapack/clapack/sytri.h
//  \brief Header file for the CLAPACK sytri wrapper functions
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

#ifndef _BLAZE_MATH_LAPACK_CLAPACK_SYTRI_H_
#define _BLAZE_MATH_LAPACK_CLAPACK_SYTRI_H_


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

void ssytri_( char* uplo, int* n, float*  A, int* lda, int* ipiv, float*  work, int* info );
void dsytri_( char* uplo, int* n, double* A, int* lda, int* ipiv, double* work, int* info );
void csytri_( char* uplo, int* n, float*  A, int* lda, int* ipiv, float*  work, int* info );
void zsytri_( char* uplo, int* n, double* A, int* lda, int* ipiv, double* work, int* info );

}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LAPACK LDLT-BASED INVERSION FUNCTIONS (SYTRI)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK LDLT-based inversion functions (sytri) */
//@{
inline void sytri( char uplo, int n, float* A, int lda, const int* ipiv, float* work, int* info );

inline void sytri( char uplo, int n, double* A, int lda, const int* ipiv, double* work, int* info );

inline void sytri( char uplo, int n, complex<float>* A, int lda,
                   const int* ipiv, complex<float>* work, int* info );

inline void sytri( char uplo, int n, complex<double>* A, int lda,
                   const int* ipiv, complex<double>* work, int* info );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense symmetric indefinite single precision
//        column-major square matrix.
// \ingroup lapack_inversion
//
// \param uplo \c 'L' in case of a lower matrix, \c 'U' in case of an upper matrix.
// \param n The number of rows/columns of the symmetric matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param work Auxiliary array of size \a n.
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK ssytri() function for
// symmetric indefinite single precision column-major matrices that have already been factorized
// by the ssytrf() function.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If \a info = -i, the i-th argument had an illegal value.
//   - > 0: If \a info = i, element D(i,i) is exactly zero and the inverse could not be computed.
//
// For more information on the ssytri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void sytri( char uplo, int n, float* A, int lda, const int* ipiv, float* work, int* info )
{
   ssytri_( &uplo, &n, A, &lda, const_cast<int*>( ipiv ), work, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense symmetric indefinite double precision
//        column-major square matrix.
// \ingroup lapack_inversion
//
// \param uplo \c 'L' in case of a lower matrix, \c 'U' in case of an upper matrix.
// \param n The number of rows/columns of the symmetric matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param work Auxiliary array of size \a n.
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK dsytri() function for
// symmetric indefinite double precision column-major matrices that have already been factorized
// by the dsytrf() function.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If \a info = -i, the i-th argument had an illegal value.
//   - > 0: If \a info = i, element D(i,i) is exactly zero and the inverse could not be computed.
//
// For more information on the dsytri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void sytri( char uplo, int n, double* A, int lda, const int* ipiv, double* work, int* info )
{
   dsytri_( &uplo, &n, A, &lda, const_cast<int*>( ipiv ), work, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense symmetric indefinite single precision
//        complex column-major square matrix.
// \ingroup lapack_inversion
//
// \param uplo \c 'L' in case of a lower matrix, \c 'U' in case of an upper matrix.
// \param n The number of rows/columns of the symmetric matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision complex column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param work Auxiliary array of size \a n.
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK csytri() function for
// symmetric indefinite single precision complex column-major matrices that have already been
// factorized by the csytrf() function.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If \a info = -i, the i-th argument had an illegal value.
//   - > 0: If \a info = i, element D(i,i) is exactly zero and the inverse could not be computed.
//
// For more information on the csytri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void sytri( char uplo, int n, complex<float>* A, int lda,
                   const int* ipiv, complex<float>* work, int* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

   csytri_( &uplo, &n, reinterpret_cast<float*>( A ), &lda,
            const_cast<int*>( ipiv ), reinterpret_cast<float*>( work ), info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense symmetric indefinite double precision
//        complex column-major square matrix.
// \ingroup lapack_inversion
//
// \param uplo \c 'L' in case of a lower matrix, \c 'U' in case of an upper matrix.
// \param n The number of rows/columns of the symmetric matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision complex column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param work Auxiliary array of size \a n.
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK zsytri() function for
// symmetric indefinite double precision complex column-major matrices that have already been
// factorized by the zsytrf() function.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If \a info = -i, the i-th argument had an illegal value.
//   - > 0: If \a info = i, element D(i,i) is exactly zero and the inverse could not be computed.
//
// For more information on the zsytri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
inline void sytri( char uplo, int n, complex<double>* A, int lda,
                   const int* ipiv, complex<double>* work, int* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

   zsytri_( &uplo, &n, reinterpret_cast<double*>( A ), &lda,
            const_cast<int*>( ipiv ), reinterpret_cast<double*>( work ), info );
}
//*************************************************************************************************

} // namespace blaze

#endif
