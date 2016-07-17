//=================================================================================================
/*!
//  \file blaze/math/lapack/ormlq.h
//  \brief Header file for the LAPACK functions to multiply Q from a LQ decomposition with a matrix (ormlq)
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

#ifndef _BLAZE_MATH_LAPACK_ORMLQ_H_
#define _BLAZE_MATH_LAPACK_ORMLQ_H_


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
#include <blaze/math/lapack/clapack/ormlq.h>
#include <blaze/math/lapack/clapack/ormqr.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/FloatingPoint.h>
#include <blaze/util/mpl/Xor.h>


namespace blaze {

//=================================================================================================
//
//  LAPACK FUNCTIONS TO MULTIPLY Q FROM A LQ DECOMPOSITION WITH A MATRIX (ORMLQ)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK functions to multiply Q from a LQ decomposition with a matrix (ormlq) */
//@{
template< typename MT1, bool SO1, typename MT2, bool SO2 >
inline void ormlq( DenseMatrix<MT1,SO1>& C, const DenseMatrix<MT2,SO2>& A,
                   char side, char trans, const ElementType_<MT2>* tau );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the multiplication of the double precision Q from a LQ decomposition
//        with another matrix.
// \ingroup lapack_decomposition
//
// \param C The matrix multiplier.
// \param A The decomposed matrix.
// \param side \c 'L' to apply \f$ Q \f$ or \f$ Q^T \f$ from the left, \c 'R' to apply from the right.
// \param trans \c 'N' for \f$ Q \f$, \c 'T' for \f$ Q^T \f$.
// \param tau Array for the scalar factors of the elementary reflectors.
// \return void
// \exception std::invalid_argument Invalid size of Q matrix.
// \exception std::invalid_argument Invalid side argument provided.
// \exception std::invalid_argument Invalid trans argument provided.
//
// This function multiplies a square \a Q matrix resulting from the LQ decomposition of the
// gelqf() functions with the given general \a m-by-\a n matrix \a C. Depending on the settings
// of \a side and \a trans it overwrites \a C with

   \code
                | side = 'L'   | side = 'R'
   -------------|--------------|--------------
   trans = 'N': | Q * C        | C * Q
   trans = 'T': | trans(Q) * C | C * trans(Q)
   \endcode

// Note that the size of matrix \c C is preserved, which means that the function does not work for
// non-square Q matrices. Therefore in case the number of rows of \a A is larger than the number
// of columns, a \a std::invalid_argument exception is thrown. Also Note that this function can
// only be used for general, non-adapted matrices with \c float or \c double element type. The
// attempt to call the function with any adapted matrix or matrices of any other element type
// results in a compile time error!
//
// The following code example demonstrates the use of the ormlq() function:

   \code
   using blaze::DynamicMatrix;
   using blaze::columnMajor;

   DynamicMatrix<double,columnMajor> A;
   DynamicMatrix<double,columnMajor> C;
   DynamicVector<double> tau;
   // ... Resizing and initialization

   gelqf( A, tau.data() );               // Performing the LQ decomposition
   ormlq( C, A, 'R', 'N', tau.data() );  // Computing C = C * Q
   \endcode

// For more information on the ormlq() functions (i.e. sormlq() and dormlq()) see the LAPACK
// online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a call to this function will result in a linker error.
*/
template< typename MT1, bool SO1, typename MT2, bool SO2 >
inline void ormlq( DenseMatrix<MT1,SO1>& C, const DenseMatrix<MT2,SO2>& A,
                   char side, char trans, const ElementType_<MT2>* tau )
{
   using boost::numeric_cast;

   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT1> );
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( ElementType_<MT1> );

   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS( MT2 );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT2> );
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( ElementType_<MT2> );

   typedef ElementType_<MT1>  ET;

   if( (~A).rows() > (~A).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid size of Q matrix" );
   }

   if( side != 'L' && side != 'R' ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid side argument provided" );
   }

   if( trans != 'N' && trans != 'T' ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid trans argument provided" );
   }

   int m   ( numeric_cast<int>( SO1 ? (~C).rows() : (~C).columns() ) );
   int n   ( numeric_cast<int>( SO1 ? (~C).columns() : (~C).rows() ) );
   int k   ( numeric_cast<int>( min( (~A).rows(), (~A).columns() ) ) );
   int lda ( numeric_cast<int>( (~A).spacing() ) );
   int ldc ( numeric_cast<int>( (~C).spacing() ) );
   int info( 0 );

   if( m == 0 || n == 0 || k == 0 ) {
      return;
   }

   if( IsRowMajorMatrix<MT1>::value ) {
      ( side  == 'L' )?( side  = 'R' ):( side  = 'L' );
   }

   if( Xor< IsRowMajorMatrix<MT1>, IsRowMajorMatrix<MT2> >::value ) {
      ( trans == 'N' )?( trans = 'T' ):( trans = 'N' );
   }

   int lwork( k*ldc );
   const std::unique_ptr<ET[]> work( new ET[lwork] );

   if( SO2 ) {
      ormlq( side, trans, m, n, k, (~A).data(), lda, tau, (~C).data(), ldc, work.get(), lwork, &info );
   }
   else {
      ormqr( side, trans, m, n, k, (~A).data(), lda, tau, (~C).data(), ldc, work.get(), lwork, &info );
   }

   BLAZE_INTERNAL_ASSERT( info == 0, "Invalid argument for Q multiplication" );
}
//*************************************************************************************************

} // namespace blaze

#endif
