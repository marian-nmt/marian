//=================================================================================================
/*!
//  \file blaze/math/StaticMatrix.h
//  \brief Header file for the complete StaticMatrix implementation
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

#ifndef _BLAZE_MATH_STATICMATRIX_H_
#define _BLAZE_MATH_STATICMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/dense/StaticMatrix.h>
#include <blaze/math/DenseMatrix.h>
#include <blaze/math/HybridMatrix.h>
#include <blaze/math/shims/Conjugate.h>
#include <blaze/math/shims/Real.h>
#include <blaze/math/StaticVector.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>
#include <blaze/system/Precision.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Numeric.h>
#include <blaze/util/Random.h>
#include <blaze/util/StaticAssert.h>


namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for StaticMatrix.
// \ingroup random
//
// This specialization of the Rand class creates random instances of StaticMatrix.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
class Rand< StaticMatrix<Type,M,N,SO> >
{
 public:
   //**Generate functions**************************************************************************
   /*!\name Generate functions */
   //@{
   inline const StaticMatrix<Type,M,N,SO> generate() const;

   template< typename Arg >
   inline const StaticMatrix<Type,M,N,SO> generate( const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************

   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( StaticMatrix<Type,M,N,SO>& matrix ) const;

   template< typename Arg >
   inline void randomize( StaticMatrix<Type,M,N,SO>& matrix, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random StaticMatrix.
//
// \return The generated random matrix.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline const StaticMatrix<Type,M,N,SO> Rand< StaticMatrix<Type,M,N,SO> >::generate() const
{
   StaticMatrix<Type,M,N,SO> matrix;
   randomize( matrix );
   return matrix;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random StaticMatrix.
//
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return The generated random matrix.
*/
template< typename Type   // Data type of the matrix
        , size_t M        // Number of rows
        , size_t N        // Number of columns
        , bool SO >       // Storage order
template< typename Arg >  // Min/max argument type
inline const StaticMatrix<Type,M,N,SO>
   Rand< StaticMatrix<Type,M,N,SO> >::generate( const Arg& min, const Arg& max ) const
{
   StaticMatrix<Type,M,N,SO> matrix;
   randomize( matrix, min, max );
   return matrix;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a StaticMatrix.
//
// \param matrix The matrix to be randomized.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void Rand< StaticMatrix<Type,M,N,SO> >::randomize( StaticMatrix<Type,M,N,SO>& matrix ) const
{
   using blaze::randomize;

   for( size_t i=0UL; i<M; ++i ) {
      for( size_t j=0UL; j<N; ++j ) {
         randomize( matrix(i,j) );
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a StaticMatrix.
//
// \param matrix The matrix to be randomized.
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return void
*/
template< typename Type   // Data type of the matrix
        , size_t M        // Number of rows
        , size_t N        // Number of columns
        , bool SO >       // Storage order
template< typename Arg >  // Min/max argument type
inline void Rand< StaticMatrix<Type,M,N,SO> >::randomize( StaticMatrix<Type,M,N,SO>& matrix,
                                                          const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   for( size_t i=0UL; i<M; ++i ) {
      for( size_t j=0UL; j<N; ++j ) {
         randomize( matrix(i,j), min, max );
      }
   }
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MAKE FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random symmetric StaticMatrix.
//
// \param matrix The matrix to be randomized.
// \return void
// \exception std::invalid_argument Invalid non-square matrix provided.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
void makeSymmetric( StaticMatrix<Type,M,N,SO>& matrix )
{
   using blaze::randomize;

   BLAZE_STATIC_ASSERT( M == N );

   for( size_t i=0UL; i<N; ++i ) {
      for( size_t j=0UL; j<i; ++j ) {
         randomize( matrix(i,j) );
         matrix(j,i) = matrix(i,j);
      }
      randomize( matrix(i,i) );
   }

   BLAZE_INTERNAL_ASSERT( isSymmetric( matrix ), "Non-symmetric matrix detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random symmetric StaticMatrix.
//
// \param matrix The matrix to be randomized.
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return void
// \exception std::invalid_argument Invalid non-square matrix provided.
*/
template< typename Type   // Data type of the matrix
        , size_t M        // Number of rows
        , size_t N        // Number of columns
        , bool SO         // Storage order
        , typename Arg >  // Min/max argument type
void makeSymmetric( StaticMatrix<Type,M,N,SO>& matrix, const Arg& min, const Arg& max )
{
   using blaze::randomize;

   BLAZE_STATIC_ASSERT( M == N );

   for( size_t i=0UL; i<N; ++i ) {
      for( size_t j=0UL; j<i; ++j ) {
         randomize( matrix(i,j), min, max );
         matrix(j,i) = matrix(i,j);
      }
      randomize( matrix(i,i), min, max );
   }

   BLAZE_INTERNAL_ASSERT( isSymmetric( matrix ), "Non-symmetric matrix detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random Hermitian StaticMatrix.
//
// \param matrix The matrix to be randomized.
// \return void
// \exception std::invalid_argument Invalid non-square matrix provided.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
void makeHermitian( StaticMatrix<Type,M,N,SO>& matrix )
{
   using blaze::randomize;

   BLAZE_STATIC_ASSERT( M == N );
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );

   typedef UnderlyingBuiltin_<Type>  BT;

   for( size_t i=0UL; i<N; ++i ) {
      for( size_t j=0UL; j<i; ++j ) {
         randomize( matrix(i,j) );
         matrix(j,i) = conj( matrix(i,j) );
      }
      matrix(i,i) = rand<BT>();
   }

   BLAZE_INTERNAL_ASSERT( isHermitian( matrix ), "Non-Hermitian matrix detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random Hermitian StaticMatrix.
//
// \param matrix The matrix to be randomized.
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return void
// \exception std::invalid_argument Invalid non-square matrix provided.
*/
template< typename Type   // Data type of the matrix
        , size_t M        // Number of rows
        , size_t N        // Number of columns
        , bool SO         // Storage order
        , typename Arg >  // Min/max argument type
void makeHermitian( StaticMatrix<Type,M,N,SO>& matrix, const Arg& min, const Arg& max )
{
   using blaze::randomize;

   BLAZE_STATIC_ASSERT( M == N );
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );

   typedef UnderlyingBuiltin_<Type>  BT;

   for( size_t i=0UL; i<N; ++i ) {
      for( size_t j=0UL; j<i; ++j ) {
         randomize( matrix(i,j), min, max );
         matrix(j,i) = conj( matrix(i,j) );
      }
      matrix(i,i) = rand<BT>( real( min ), real( max ) );
   }

   BLAZE_INTERNAL_ASSERT( isHermitian( matrix ), "Non-Hermitian matrix detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random (Hermitian) positive definite StaticMatrix.
//
// \param matrix The matrix to be randomized.
// \return void
// \exception std::invalid_argument Invalid non-square matrix provided.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
void makePositiveDefinite( StaticMatrix<Type,M,N,SO>& matrix )
{
   using blaze::randomize;

   BLAZE_STATIC_ASSERT( M == N );
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( Type );

   randomize( matrix );
   matrix *= ctrans( matrix );

   for( size_t i=0UL; i<N; ++i ) {
      matrix(i,i) += Type(N);
   }

   BLAZE_INTERNAL_ASSERT( isHermitian( matrix ), "Non-symmetric matrix detected" );
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
