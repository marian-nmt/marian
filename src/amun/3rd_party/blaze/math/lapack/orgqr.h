//=================================================================================================
/*!
//  \file blaze/math/lapack/orgqr.h
//  \brief Header file for the LAPACK functions to reconstruct Q from a QR decomposition (orgqr)
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

#ifndef _BLAZE_MATH_LAPACK_ORGQR_H_
#define _BLAZE_MATH_LAPACK_ORGQR_H_


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
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/lapack/clapack/orglq.h>
#include <blaze/math/lapack/clapack/orgqr.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Builtin.h>


namespace blaze {

//=================================================================================================
//
//  LAPACK FUNCTIONS TO RECONSTRUCT Q FROM A QR DECOMPOSITION (ORGQR)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK functions to reconstruct Q from a QR decomposition (orgqr) */
//@{
template< typename MT, bool SO >
inline void orgqr( DenseMatrix<MT,SO>& A, const ElementType_<MT>* tau );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the reconstruction of the orthogonal matrix Q from a QR decomposition.
// \ingroup lapack_decomposition
//
// \param A The decomposed matrix.
// \param tau Array for the scalar factors of the elementary reflectors; size >= min( \a m, \a n ).
// \return void
//
// This function reconstructs the orthogonal matrix \a Q of a QR decomposition based on the LAPACK
// orgqr() functions from matrices that have already been QR factorized by the geqrf() functions.
// Note that this function can only be used for general, non-adapted matrices with \c float or
// \c double element type. The attempt to call the function with any adapted matrix or matrices
// of any other element type results in a compile time error!\n
//
// The \a m-by-min(\a m,\a n) \a Q matrix is stored within the given matrix \a A:

   \code
   using blaze::DynamicMatrix;
   using blaze::columnMajor;

   DynamicMatrix<double,columnMajor> A;
   DynamicVector<double> tau;
   // ... Resizing and initialization

   geqrf( A, tau.data() );  // Performing the QR decomposition
   orgqr( A, tau.data() );  // Reconstructing the Q matrix

   const int m( A.rows() );
   const int n( A.columns() );

   DynamicMatrix<double,columnMajor> Q( submatrix( A, 0UL, 0UL, m, min(m,n) ) );
   \endcode

// For more information on the orgqr() functions (i.e. sorgqr() and dorgqr()) see the LAPACK
// online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
template< typename MT, bool SO >
inline void orgqr( DenseMatrix<MT,SO>& A, const ElementType_<MT>* tau )
{
   using boost::numeric_cast;

   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );
   BLAZE_CONSTRAINT_MUST_BE_BUILTIN_TYPE( ElementType_<MT> );

   typedef ElementType_<MT>  ET;

   int m   ( numeric_cast<int>( SO ? (~A).rows() : (~A).columns() ) );
   int n   ( numeric_cast<int>( SO ? (~A).columns() : (~A).rows() ) );
   int k   ( min( m, n ) );
   int lda ( numeric_cast<int>( (~A).spacing() ) );
   int info( 0 );

   if( k == 0 ) {
      return;
   }

   int lwork( k*lda );
   const std::unique_ptr<ET[]> work( new ET[lwork] );

   if( SO ) {
      orgqr( m, k, k, (~A).data(), lda, tau, work.get(), lwork, &info );
   }
   else {
      orglq( k, n, k, (~A).data(), lda, tau, work.get(), lwork, &info );
   }

   BLAZE_INTERNAL_ASSERT( info == 0, "Invalid argument for Q reconstruction" );
}
//*************************************************************************************************

} // namespace blaze

#endif
