//=================================================================================================
/*!
//  \file blaze/math/lapack/hetri.h
//  \brief Header file for the LAPACK Hermitian matrix inversion functionality (hetri)
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

#ifndef _BLAZE_MATH_LAPACK_HETRI_H_
#define _BLAZE_MATH_LAPACK_HETRI_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <memory>
#include <boost/cast.hpp>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Adaptor.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/MutableDataAccess.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/lapack/clapack/hetri.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/util/Assert.h>



namespace blaze {

//=================================================================================================
//
//  LAPACK LDLH-BASED INVERSION FUNCTIONS (HETRI)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK LDLH-based inversion functions (hetri) */
//@{
template< typename MT, bool SO >
inline void hetri( DenseMatrix<MT,SO>& A, char uplo, const int* ipiv );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense Hermitian indefinite matrix.
// \ingroup lapack_inversion
//
// \param A The triangular matrix to be inverted.
// \param uplo \c 'L' in case of a lower matrix, \c 'U' in case of an upper matrix.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \return void
// \exception std::invalid_argument Invalid non-square matrix provided.
// \exception std::invalid_argument Invalid uplo argument provided.
// \exception std::runtime_error Inversion of singular matrix failed.
//
// This function performs the dense matrix inversion based on the LAPACK hetri() functions for
// Hermitian indefinite matrices that have already been factorized by the hetrf() functions.
// Note that the function only works for general, non-adapted matrices with \c float, \c double,
// \c complex<float>, or \c complex<double> element type. The attempt to call the function with
// adaptors or matrices of any other element type results in a compile time error!
//
// The function fails if ...
//
//  - ... the given matrix is not a square matrix;
//  - ... the given \a uplo argument is neither \c 'L' nor \c 'U';
//  - ... the given matrix is singular and not invertible.
//
// In all failure cases an exception is thrown.
//
// For more information on the hetri() functions (i.e. chetri() and zhetri()) see the LAPACK
// online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
//
// \note This function does only provide the basic exception safety guarantee, i.e. in case of an
// exception \a A may already have been modified.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void hetri( DenseMatrix<MT,SO>& A, char uplo, const int* ipiv )
{
   using boost::numeric_cast;

   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   typedef ElementType_<MT>  ET;

   if( !isSquare( ~A ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid non-square matrix provided" );
   }

   if( uplo != 'L' && uplo != 'U' ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid uplo argument provided" );
   }

   int n   ( numeric_cast<int>( (~A).columns() ) );
   int lda ( numeric_cast<int>( (~A).spacing() ) );
   int info( 0 );

   if( n == 0 ) {
      return;
   }

   if( IsRowMajorMatrix<MT>::value ) {
      ( uplo == 'L' )?( uplo = 'U' ):( uplo = 'L' );
   }

   const std::unique_ptr<ET[]> work( new ET[n] );

   hetri( uplo, n, (~A).data(), lda, ipiv, work.get(), &info );

   BLAZE_INTERNAL_ASSERT( info >= 0, "Invalid argument for matrix inversion" );

   if( info > 0 ) {
      BLAZE_THROW_LAPACK_ERROR( "Inversion of singular matrix failed" );
   }
}
//*************************************************************************************************

} // namespace blaze

#endif
