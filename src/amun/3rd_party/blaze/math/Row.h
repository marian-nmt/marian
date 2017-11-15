//=================================================================================================
/*!
//  \file blaze/math/Row.h
//  \brief Header file for the complete Row implementation
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

#ifndef _BLAZE_MATH_ROW_H_
#define _BLAZE_MATH_ROW_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/Exception.h>
#include <blaze/math/smp/DenseVector.h>
#include <blaze/math/smp/SparseVector.h>
#include <blaze/math/views/Column.h>
#include <blaze/math/views/Row.h>
#include <blaze/util/Random.h>


namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION FOR DENSE ROWS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for dense rows.
// \ingroup random
//
// This specialization of the Rand class randomizes dense rows.
*/
template< typename MT  // Type of the dense matrix
        , bool SO      // Storage order
        , bool SF >    // Symmetry flag
class Rand< Row<MT,SO,true,SF> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( Row<MT,SO,true,SF>& row ) const;

   template< typename Arg >
   inline void randomize( Row<MT,SO,true,SF>& row, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense row.
//
// \param row The row to be randomized.
// \return void
*/
template< typename MT  // Type of the dense matrix
        , bool SO      // Storage order
        , bool SF >    // Symmetry flag
inline void Rand< Row<MT,SO,true,SF> >::randomize( Row<MT,SO,true,SF>& row ) const
{
   using blaze::randomize;

   for( size_t i=0UL; i<row.size(); ++i ) {
      randomize( row[i] );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a dense row.
//
// \param row The row to be randomized.
// \param min The smallest possible value for a row element.
// \param max The largest possible value for a row element.
// \return void
*/
template< typename MT     // Type of the dense matrix
        , bool SO         // Storage order
        , bool SF >       // Symmetry flag
template< typename Arg >  // Min/max argument type
inline void Rand< Row<MT,SO,true,SF> >::randomize( Row<MT,SO,true,SF>& row,
                                                   const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   for( size_t i=0UL; i<row.size(); ++i ) {
      randomize( row[i], min, max );
   }
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RAND SPECIALIZATION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for sparse rows.
// \ingroup random
//
// This specialization of the Rand class randomizes sparse rows.
*/
template< typename MT  // Type of the sparse matrix
        , bool SO      // Storage order
        , bool SF >    // Symmetry flag
class Rand< Row<MT,SO,false,SF> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( Row<MT,SO,false,SF>& row ) const;
   inline void randomize( Row<MT,SO,false,SF>& row, size_t nonzeros ) const;

   template< typename Arg >
   inline void randomize( Row<MT,SO,false,SF>& row, const Arg& min, const Arg& max ) const;

   template< typename Arg >
   inline void randomize( Row<MT,SO,false,SF>& row, size_t nonzeros, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse row.
//
// \param row The row to be randomized.
// \return void
*/
template< typename MT  // Type of the sparse matrix
        , bool SO      // Storage order
        , bool SF >    // Symmetry flag
inline void Rand< Row<MT,SO,false,SF> >::randomize( Row<MT,SO,false,SF>& row ) const
{
   typedef ElementType_< Row<MT,SO,false,SF> >  ElementType;

   const size_t size( row.size() );

   if( size == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*size ) ) );

   row.reset();
   row.reserve( nonzeros );

   while( row.nonZeros() < nonzeros ) {
      row[ rand<size_t>( 0UL, size-1UL ) ] = rand<ElementType>();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse row.
//
// \param row The row to be randomized.
// \param nonzeros The number of non-zero elements of the random row.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename MT  // Type of the sparse matrix
        , bool SO      // Storage order
        , bool SF >    // Symmetry flag
inline void Rand< Row<MT,SO,false,SF> >::randomize( Row<MT,SO,false,SF>& row, size_t nonzeros ) const
{
   typedef ElementType_< Row<MT,SO,false,SF> >  ElementType;

   const size_t size( row.size() );

   if( nonzeros > size ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   if( size == 0UL ) return;

   row.reset();
   row.reserve( nonzeros );

   while( row.nonZeros() < nonzeros ) {
      row[ rand<size_t>( 0UL, size-1UL ) ] = rand<ElementType>();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse row.
//
// \param row The row to be randomized.
// \param min The smallest possible value for a row element.
// \param max The largest possible value for a row element.
// \return void
*/
template< typename MT     // Type of the sparse matrix
        , bool SO         // Storage order
        , bool SF >       // Symmetry flag
template< typename Arg >  // Min/max argument type
inline void Rand< Row<MT,SO,false,SF> >::randomize( Row<MT,SO,false,SF>& row,
                                                   const Arg& min, const Arg& max ) const
{
   typedef ElementType_< Row<MT,SO,false,SF> >  ElementType;

   const size_t size( row.size() );

   if( size == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*size ) ) );

   row.reset();
   row.reserve( nonzeros );

   while( row.nonZeros() < nonzeros ) {
      row[ rand<size_t>( 0UL, size-1UL ) ] = rand<ElementType>( min, max );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a sparse row.
//
// \param row The row to be randomized.
// \param nonzeros The number of non-zero elements of the random row.
// \param min The smallest possible value for a row element.
// \param max The largest possible value for a row element.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename MT     // Type of the sparse matrix
        , bool SO         // Storage order
        , bool SF >       // Symmetry flag
template< typename Arg >  // Min/max argument type
inline void Rand< Row<MT,SO,false,SF> >::randomize( Row<MT,SO,false,SF>& row, size_t nonzeros,
                                                   const Arg& min, const Arg& max ) const
{
   typedef ElementType_< Row<MT,SO,false,SF> >  ElementType;

   const size_t size( row.size() );

   if( nonzeros > size ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   if( size == 0UL ) return;

   row.reset();
   row.reserve( nonzeros );

   while( row.nonZeros() < nonzeros ) {
      row[ rand<size_t>( 0UL, size-1UL ) ] = rand<ElementType>( min, max );
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
