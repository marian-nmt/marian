//=================================================================================================
/*!
//  \file blaze/math/HermitianMatrix.h
//  \brief Header file for the complete HermitianMatrix implementation
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

#ifndef _BLAZE_MATH_HERMITIANMATRIX_H_
#define _BLAZE_MATH_HERMITIANMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cmath>
#include <stdexcept>
#include <blaze/math/Aliases.h>
#include <blaze/math/adaptors/HermitianMatrix.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/Resizable.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/DenseMatrix.h>
#include <blaze/math/shims/Real.h>
#include <blaze/math/SparseMatrix.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>
#include <blaze/util/Assert.h>
#include <blaze/util/FalseType.h>
#include <blaze/util/Random.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for HermitianMatrix.
// \ingroup random
//
// This specialization of the Rand class creates random instances of HermitianMatrix.
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
class Rand< HermitianMatrix<MT,SO,DF> >
{
 public:
   //**Generate functions**************************************************************************
   /*!\name Generate functions */
   //@{
   inline const HermitianMatrix<MT,SO,DF> generate() const;
   inline const HermitianMatrix<MT,SO,DF> generate( size_t n ) const;
   inline const HermitianMatrix<MT,SO,DF> generate( size_t n, size_t nonzeros ) const;

   template< typename Arg >
   inline const HermitianMatrix<MT,SO,DF> generate( const Arg& min, const Arg& max ) const;

   template< typename Arg >
   inline const HermitianMatrix<MT,SO,DF> generate( size_t n, const Arg& min, const Arg& max ) const;

   template< typename Arg >
   inline const HermitianMatrix<MT,SO,DF> generate( size_t n, size_t nonzeros,
                                                    const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************

   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( HermitianMatrix<MT,SO,DF>& matrix ) const;
   inline void randomize( HermitianMatrix<MT,SO,DF>& matrix, size_t nonzeros ) const;

   template< typename Arg >
   inline void randomize( HermitianMatrix<MT,SO,DF>& matrix, const Arg& min, const Arg& max ) const;

   template< typename Arg >
   inline void randomize( HermitianMatrix<MT,SO,DF>& matrix, size_t nonzeros,
                          const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************

 private:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( HermitianMatrix<MT,SO,DF>& matrix, TrueType  ) const;
   inline void randomize( HermitianMatrix<MT,SO,DF>& matrix, FalseType ) const;

   template< typename Arg >
   inline void randomize( HermitianMatrix<MT,SO,DF>& matrix, const Arg& min, const Arg& max, TrueType ) const;

   template< typename Arg >
   inline void randomize( HermitianMatrix<MT,SO,DF>& matrix, const Arg& min, const Arg& max, FalseType ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random HermitianMatrix.
//
// \return The generated random matrix.
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline const HermitianMatrix<MT,SO,DF> Rand< HermitianMatrix<MT,SO,DF> >::generate() const
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_RESIZABLE( MT );

   HermitianMatrix<MT,SO,DF> matrix;
   randomize( matrix );
   return matrix;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random HermitianMatrix.
//
// \param n The number of rows and columns of the random matrix.
// \return The generated random matrix.
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline const HermitianMatrix<MT,SO,DF>
   Rand< HermitianMatrix<MT,SO,DF> >::generate( size_t n ) const
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE( MT );

   HermitianMatrix<MT,SO,DF> matrix( n );
   randomize( matrix );
   return matrix;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random HermitianMatrix.
//
// \param n The number of rows and columns of the random matrix.
// \param nonzeros The number of non-zero elements of the random matrix.
// \return The generated random matrix.
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline const HermitianMatrix<MT,SO,DF>
   Rand< HermitianMatrix<MT,SO,DF> >::generate( size_t n, size_t nonzeros ) const
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE         ( MT );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( MT );

   if( nonzeros > n*n )
      throw std::invalid_argument( "Invalid number of non-zero elements" );

   HermitianMatrix<MT,SO,DF> matrix( n );
   randomize( matrix, nonzeros );

   return matrix;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random HermitianMatrix.
//
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return The generated random matrix.
*/
template< typename MT     // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF >       // Density flag
template< typename Arg >  // Min/max argument type
inline const HermitianMatrix<MT,SO,DF>
   Rand< HermitianMatrix<MT,SO,DF> >::generate( const Arg& min, const Arg& max ) const
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_RESIZABLE( MT );

   HermitianMatrix<MT,SO,DF> matrix;
   randomize( matrix, min, max );
   return matrix;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random HermitianMatrix.
//
// \param n The number of rows and columns of the random matrix.
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return The generated random matrix.
*/
template< typename MT     // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF >       // Density flag
template< typename Arg >  // Min/max argument type
inline const HermitianMatrix<MT,SO,DF>
   Rand< HermitianMatrix<MT,SO,DF> >::generate( size_t n, const Arg& min, const Arg& max ) const
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE( MT );

   HermitianMatrix<MT,SO,DF> matrix( n );
   randomize( matrix, min, max );
   return matrix;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random HermitianMatrix.
//
// \param n The number of rows and columns of the random matrix.
// \param nonzeros The number of non-zero elements of the random matrix.
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return The generated random matrix.
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename MT     // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF >       // Density flag
template< typename Arg >  // Min/max argument type
inline const HermitianMatrix<MT,SO,DF>
   Rand< HermitianMatrix<MT,SO,DF> >::generate( size_t n, size_t nonzeros,
                                                const Arg& min, const Arg& max ) const
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE         ( MT );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( MT );

   if( nonzeros > n*n )
      throw std::invalid_argument( "Invalid number of non-zero elements" );

   HermitianMatrix<MT,SO,DF> matrix( n );
   randomize( matrix, nonzeros, min, max );

   return matrix;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a HermitianMatrix.
//
// \param matrix The matrix to be randomized.
// \return void
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline void Rand< HermitianMatrix<MT,SO,DF> >::randomize( HermitianMatrix<MT,SO,DF>& matrix ) const
{
   randomize( matrix, typename IsDenseMatrix<MT>::Type() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense HermitianMatrix.
//
// \param matrix The matrix to be randomized.
// \return void
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline void Rand< HermitianMatrix<MT,SO,DF> >::randomize( HermitianMatrix<MT,SO,DF>& matrix, TrueType ) const
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT );

   typedef ElementType_<MT>        ET;
   typedef UnderlyingBuiltin_<ET>  BT;

   const size_t n( matrix.rows() );

   for( size_t i=0UL; i<n; ++i ) {
      for( size_t j=0UL; j<i; ++j ) {
         matrix(i,j) = rand<ET>();
      }
      matrix(i,i) = rand<BT>();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse HermitianMatrix.
//
// \param matrix The matrix to be randomized.
// \return void
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline void Rand< HermitianMatrix<MT,SO,DF> >::randomize( HermitianMatrix<MT,SO,DF>& matrix, FalseType ) const
{
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( MT );

   const size_t n( matrix.rows() );

   if( n == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*n*n ) ) );

   randomize( matrix, nonzeros );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse HermitianMatrix.
//
// \param matrix The matrix to be randomized.
// \param nonzeros The number of non-zero elements of the random matrix.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline void Rand< HermitianMatrix<MT,SO,DF> >::randomize( HermitianMatrix<MT,SO,DF>& matrix, size_t nonzeros ) const
{
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( MT );

   typedef ElementType_<MT>        ET;
   typedef UnderlyingBuiltin_<ET>  BT;

   const size_t n( matrix.rows() );

   if( nonzeros > n*n )
      throw std::invalid_argument( "Invalid number of non-zero elements" );

   if( n == 0UL ) return;

   matrix.reset();
   matrix.reserve( nonzeros );

   while( matrix.nonZeros() < nonzeros )
   {
      const size_t row   ( rand<size_t>( 0UL, n-1UL ) );
      const size_t column( rand<size_t>( 0UL, n-1UL ) );

      if( row == column )
         matrix(row,column) = rand<BT>();
      else
         matrix(row,column) = rand<ET>();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a HermitianMatrix.
//
// \param matrix The matrix to be randomized.
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return void
*/
template< typename MT     // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF >       // Density flag
template< typename Arg >  // Min/max argument type
inline void Rand< HermitianMatrix<MT,SO,DF> >::randomize( HermitianMatrix<MT,SO,DF>& matrix,
                                                          const Arg& min, const Arg& max ) const
{
   randomize( matrix, min, max, typename IsDenseMatrix<MT>::Type() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense HermitianMatrix.
//
// \param matrix The matrix to be randomized.
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return void
*/
template< typename MT     // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF >       // Density flag
template< typename Arg >  // Min/max argument type
inline void Rand< HermitianMatrix<MT,SO,DF> >::randomize( HermitianMatrix<MT,SO,DF>& matrix,
                                                          const Arg& min, const Arg& max, TrueType ) const
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT );

   typedef ElementType_<MT>        ET;
   typedef UnderlyingBuiltin_<ET>  BT;

   const size_t n( matrix.rows() );

   for( size_t i=0UL; i<n; ++i ) {
      for( size_t j=0UL; j<i; ++j ) {
         matrix(i,j) = rand<ET>( min, max );
      }
      matrix(i,i) = rand<BT>( real( min ), real( max ) );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse HermitianMatrix.
//
// \param matrix The matrix to be randomized.
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return void
*/
template< typename MT     // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF >       // Density flag
template< typename Arg >  // Min/max argument type
inline void Rand< HermitianMatrix<MT,SO,DF> >::randomize( HermitianMatrix<MT,SO,DF>& matrix,
                                                          const Arg& min, const Arg& max, FalseType ) const
{
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( MT );

   const size_t n( matrix.rows() );

   if( n == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*n*n ) ) );

   randomize( matrix, nonzeros, min, max );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse HermitianMatrix.
//
// \param matrix The matrix to be randomized.
// \param nonzeros The number of non-zero elements of the random matrix.
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename MT     // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF >       // Density flag
template< typename Arg >  // Min/max argument type
inline void Rand< HermitianMatrix<MT,SO,DF> >::randomize( HermitianMatrix<MT,SO,DF>& matrix,
                                                          size_t nonzeros, const Arg& min, const Arg& max ) const
{
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( MT );

   typedef ElementType_<MT>        ET;
   typedef UnderlyingBuiltin_<ET>  BT;

   const size_t n( matrix.rows() );

   if( nonzeros > n*n )
      throw std::invalid_argument( "Invalid number of non-zero elements" );

   if( n == 0UL ) return;

   matrix.reset();
   matrix.reserve( nonzeros );

   while( matrix.nonZeros() < nonzeros )
   {
      const size_t row   ( rand<size_t>( 0UL, n-1UL ) );
      const size_t column( rand<size_t>( 0UL, n-1UL ) );

      if( row == column )
         matrix(row,column) = rand<BT>( real( min ), real( max ) );
      else
         matrix(row,column) = rand<ET>( min, max );
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
/*!\brief Setup of a random symmetric HermitianMatrix.
//
// \param matrix The matrix to be randomized.
// \return void
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
void makeSymmetric( HermitianMatrix<MT,SO,DF>& matrix )
{
   typedef UnderlyingBuiltin_< ElementType_<MT> >  BT;

   const size_t n( matrix.rows() );

   for( size_t i=0UL; i<n; ++i ) {
      for( size_t j=0UL; j<=i; ++j ) {
         matrix(i,j) = rand<BT>();
      }
   }

   BLAZE_INTERNAL_ASSERT( isSymmetric( matrix ), "Non-symmetric matrix detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random symmetric HermitianMatrix.
//
// \param matrix The matrix to be randomized.
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return void
*/
template< typename MT     // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF         // Density flag
        , typename Arg >  // Min/max argument type
void makeSymmetric( HermitianMatrix<MT,SO,DF>& matrix, const Arg& min, const Arg& max )
{
   typedef UnderlyingBuiltin_< ElementType_<MT> >  BT;

   const size_t n( matrix.rows() );

   for( size_t i=0UL; i<n; ++i ) {
      for( size_t j=0UL; j<=i; ++j ) {
         matrix(i,j) = rand<BT>( real( min ), real( max ) );
      }
   }

   BLAZE_INTERNAL_ASSERT( isSymmetric( matrix ), "Non-symmetric matrix detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random Hermitian HermitianMatrix.
//
// \param matrix The matrix to be randomized.
// \return void
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
void makeHermitian( HermitianMatrix<MT,SO,DF>& matrix )
{
   using blaze::randomize;

   randomize( matrix );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random Hermitian HermitianMatrix.
//
// \param matrix The matrix to be randomized.
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return void
*/
template< typename MT     // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF         // Density flag
        , typename Arg >  // Min/max argument type
void makeHermitian( HermitianMatrix<MT,SO,DF>& matrix, const Arg& min, const Arg& max )
{
   using blaze::randomize;

   randomize( matrix, min, max );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setup of a random (Hermitian) positive definite HermitianMatrix.
//
// \param matrix The matrix to be randomized.
// \return void
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
void makePositiveDefinite( HermitianMatrix<MT,SO,DF>& matrix )
{
   using blaze::randomize;

   typedef UnderlyingBuiltin_< ElementType_<MT> >  BT;

   const size_t n( matrix.rows() );

   randomize( matrix );
   matrix *= matrix;

   for( size_t i=0UL; i<n; ++i ) {
      matrix(i,i) += BT(n);
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
