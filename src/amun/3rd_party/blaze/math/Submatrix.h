//=================================================================================================
/*!
//  \file blaze/math/Submatrix.h
//  \brief Header file for the complete Submatrix implementation
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

#ifndef _BLAZE_MATH_SUBMATRIX_H_
#define _BLAZE_MATH_SUBMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/Exception.h>
#include <blaze/math/smp/DenseMatrix.h>
#include <blaze/math/smp/SparseMatrix.h>
#include <blaze/math/views/Submatrix.h>
#include <blaze/math/views/Subvector.h>
#include <blaze/util/Random.h>


namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION FOR DENSE SUBMATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for dense submatrices.
// \ingroup random
//
// This specialization of the Rand class randomizes dense submatrices.
*/
template< typename MT  // Type of the dense matrix
        , bool AF      // Alignment flag
        , bool SO >    // Storage order
class Rand< Submatrix<MT,AF,SO,true> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( Submatrix<MT,AF,SO,true>& submatrix ) const;

   template< typename Arg >
   inline void randomize( Submatrix<MT,AF,SO,true>& submatrix, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense submatrix.
//
// \param submatrix The submatrix to be randomized.
// \return void
*/
template< typename MT  // Type of the dense matrix
        , bool AF      // Alignment flag
        , bool SO >    // Storage order
inline void Rand< Submatrix<MT,AF,SO,true> >::randomize( Submatrix<MT,AF,SO,true>& submatrix ) const
{
   using blaze::randomize;

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<submatrix.rows(); ++i ) {
         for( size_t j=0UL; j<submatrix.columns(); ++j ) {
            randomize( submatrix(i,j) );
         }
      }
   }
   else {
      for( size_t j=0UL; j<submatrix.columns(); ++j ) {
         for( size_t i=0UL; i<submatrix.rows(); ++i ) {
            randomize( submatrix(i,j) );
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense submatrix.
//
// \param submatrix The submatrix to be randomized.
// \param min The smallest possible value for a matrix element.
// \param max The largest possible value for a matrix element.
// \return void
*/
template< typename MT     // Type of the dense matrix
        , bool AF         // Alignment flag
        , bool SO >       // Storage order
template< typename Arg >  // Min/max argument type
inline void Rand< Submatrix<MT,AF,SO,true> >::randomize( Submatrix<MT,AF,SO,true>& submatrix,
                                                         const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<submatrix.rows(); ++i ) {
         for( size_t j=0UL; j<submatrix.columns(); ++j ) {
            randomize( submatrix(i,j), min, max );
         }
      }
   }
   else {
      for( size_t j=0UL; j<submatrix.columns(); ++j ) {
         for( size_t i=0UL; i<submatrix.rows(); ++i ) {
            randomize( submatrix(i,j), min, max );
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RAND SPECIALIZATION FOR SPARSE SUBMATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for sparse submatrices.
// \ingroup random
//
// This specialization of the Rand class randomizes sparse submatrices.
*/
template< typename MT  // Type of the sparse matrix
        , bool AF      // Alignment flag
        , bool SO >    // Storage order
class Rand< Submatrix<MT,AF,SO,false> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( Submatrix<MT,AF,SO,false>& submatrix ) const;
   inline void randomize( Submatrix<MT,AF,SO,false>& submatrix, size_t nonzeros ) const;

   template< typename Arg >
   inline void randomize( Submatrix<MT,AF,SO,false>& submatrix, const Arg& min, const Arg& max ) const;

   template< typename Arg >
   inline void randomize( Submatrix<MT,AF,SO,false>& submatrix, size_t nonzeros,
                          const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse submatrix.
//
// \param submatrix The submatrix to be randomized.
// \return void
*/
template< typename MT  // Type of the sparse matrix
        , bool AF      // Alignment flag
        , bool SO >    // Storage order
inline void Rand< Submatrix<MT,AF,SO,false> >::randomize( Submatrix<MT,AF,SO,false>& submatrix ) const
{
   typedef ElementType_< Submatrix<MT,AF,SO,false> >  ElementType;

   const size_t m( submatrix.rows()    );
   const size_t n( submatrix.columns() );

   if( m == 0UL || n == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*m*n ) ) );

   submatrix.reset();
   submatrix.reserve( nonzeros );

   while( submatrix.nonZeros() < nonzeros ) {
      submatrix( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse submatrix.
//
// \param submatrix The submatrix to be randomized.
// \param nonzeros The number of non-zero elements of the random submatrix.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename MT  // Type of the sparse matrix
        , bool AF      // Alignment flag
        , bool SO >    // Storage order
inline void Rand< Submatrix<MT,AF,SO,false> >::randomize( Submatrix<MT,AF,SO,false>& submatrix, size_t nonzeros ) const
{
   typedef ElementType_< Submatrix<MT,AF,SO,false> >  ElementType;

   const size_t m( submatrix.rows()    );
   const size_t n( submatrix.columns() );

   if( nonzeros > m*n ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   if( m == 0UL || n == 0UL ) return;

   submatrix.reset();
   submatrix.reserve( nonzeros );

   while( submatrix.nonZeros() < nonzeros ) {
      submatrix( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse submatrix.
//
// \param submatrix The submatrix to be randomized.
// \param min The smallest possible value for a submatrix element.
// \param max The largest possible value for a submatrix element.
// \return void
*/
template< typename MT     // Type of the sparse matrix
        , bool AF         // Alignment flag
        , bool SO >       // Storage order
template< typename Arg >  // Min/max argument type
inline void Rand< Submatrix<MT,AF,SO,false> >::randomize( Submatrix<MT,AF,SO,false>& submatrix,
                                                          const Arg& min, const Arg& max ) const
{
   typedef ElementType_< Submatrix<MT,AF,SO,false> >  ElementType;

   const size_t m( submatrix.rows()    );
   const size_t n( submatrix.columns() );

   if( m == 0UL || n == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*m*n ) ) );

   submatrix.reset();
   submatrix.reserve( nonzeros );

   while( submatrix.nonZeros() < nonzeros ) {
      submatrix( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>( min, max );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse submatrix.
//
// \param submatrix The submatrix to be randomized.
// \param nonzeros The number of non-zero elements of the random submatrix.
// \param min The smallest possible value for a submatrix element.
// \param max The largest possible value for a submatrix element.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename MT     // Type of the sparse matrix
        , bool AF         // Alignment flag
        , bool SO >       // Storage order
template< typename Arg >  // Min/max argument type
inline void Rand< Submatrix<MT,AF,SO,false> >::randomize( Submatrix<MT,AF,SO,false>& submatrix,
                                                          size_t nonzeros, const Arg& min, const Arg& max ) const
{
   typedef ElementType_< Submatrix<MT,AF,SO,false> >  ElementType;

   const size_t m( submatrix.rows()    );
   const size_t n( submatrix.columns() );

   if( nonzeros > m*n ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   if( m == 0UL || n == 0UL ) return;

   submatrix.reset();
   submatrix.reserve( nonzeros );

   while( submatrix.nonZeros() < nonzeros ) {
      submatrix( rand<size_t>( 0UL, m-1UL ), rand<size_t>( 0UL, n-1UL ) ) = rand<ElementType>( min, max );
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
