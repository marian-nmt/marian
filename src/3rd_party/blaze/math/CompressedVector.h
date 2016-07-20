//=================================================================================================
/*!
//  \file blaze/math/CompressedVector.h
//  \brief Header file for the complete CompressedVector implementation
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

#ifndef _BLAZE_MATH_COMPRESSEDVECTOR_H_
#define _BLAZE_MATH_COMPRESSEDVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/dense/StaticVector.h>
#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/sparse/CompressedVector.h>
#include <blaze/math/SparseVector.h>
#include <blaze/system/Precision.h>
#include <blaze/util/Indices.h>
#include <blaze/util/Random.h>


namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for CompressedVector.
// \ingroup random
//
// This specialization of the Rand class creates random instances of CompressedVector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
class Rand< CompressedVector<Type,TF> >
{
 public:
   //**Generate functions**************************************************************************
   /*!\name Generate functions */
   //@{
   inline const CompressedVector<Type,TF> generate( size_t size ) const;
   inline const CompressedVector<Type,TF> generate( size_t size, size_t nonzeros ) const;

   template< typename Arg >
   inline const CompressedVector<Type,TF> generate( size_t size, const Arg& min, const Arg& max ) const;

   template< typename Arg >
   inline const CompressedVector<Type,TF> generate( size_t size, size_t nonzeros, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************

   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( CompressedVector<Type,TF>& vector ) const;
   inline void randomize( CompressedVector<Type,TF>& vector, size_t nonzeros ) const;

   template< typename Arg >
   inline void randomize( CompressedVector<Type,TF>& vector, const Arg& min, const Arg& max ) const;

   template< typename Arg >
   inline void randomize( CompressedVector<Type,TF>& vector, size_t nonzeros, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random CompressedVector.
//
// \param size The size of the random vector.
// \return The generated random vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline const CompressedVector<Type,TF>
   Rand< CompressedVector<Type,TF> >::generate( size_t size ) const
{
   CompressedVector<Type,TF> vector( size );
   randomize( vector );

   return vector;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random CompressedVector.
//
// \param size The size of the random vector.
// \param nonzeros The number of non-zero elements of the random vector.
// \return The generated random vector.
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline const CompressedVector<Type,TF>
   Rand< CompressedVector<Type,TF> >::generate( size_t size, size_t nonzeros ) const
{
   if( nonzeros > size ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   CompressedVector<Type,TF> vector( size, nonzeros );
   randomize( vector, nonzeros );

   return vector;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random CompressedVector.
//
// \param size The size of the random vector.
// \param min The smallest possible value for a vector element.
// \param max The largest possible value for a vector element.
// \return The generated random vector.
*/
template< typename Type   // Data type of the vector
        , bool TF >       // Transpose flag
template< typename Arg >  // Min/max argument type
inline const CompressedVector<Type,TF>
   Rand< CompressedVector<Type,TF> >::generate( size_t size, const Arg& min, const Arg& max ) const
{
   CompressedVector<Type,TF> vector( size );
   randomize( vector, min, max );

   return vector;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random CompressedVector.
//
// \param size The size of the random vector.
// \param nonzeros The number of non-zero elements of the random vector.
// \param min The smallest possible value for a vector element.
// \param max The largest possible value for a vector element.
// \return The generated random vector.
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename Type   // Data type of the vector
        , bool TF >       // Transpose flag
template< typename Arg >  // Min/max argument type
inline const CompressedVector<Type,TF>
   Rand< CompressedVector<Type,TF> >::generate( size_t size, size_t nonzeros, const Arg& min, const Arg& max ) const
{
   if( nonzeros > size ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   CompressedVector<Type,TF> vector( size, nonzeros );
   randomize( vector, nonzeros, min, max );

   return vector;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a CompressedVector.
//
// \param vector The vector to be randomized.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void Rand< CompressedVector<Type,TF> >::randomize( CompressedVector<Type,TF>& vector ) const
{
   const size_t size( vector.size() );

   if( size == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*size ) ) );

   randomize( vector, nonzeros );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a CompressedVector.
//
// \param vector The vector to be randomized.
// \param nonzeros The number of non-zero elements of the random vector.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void Rand< CompressedVector<Type,TF> >::randomize( CompressedVector<Type,TF>& vector, size_t nonzeros ) const
{
   const size_t size( vector.size() );

   if( nonzeros > size ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   if( size == 0UL ) return;

   vector.reset();
   vector.reserve( nonzeros );

   const Indices indices( 0UL, vector.size()-1UL, nonzeros );

   for( size_t index : indices ) {
      vector.append( index, rand<Type>() );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a CompressedVector.
//
// \param vector The vector to be randomized.
// \param min The smallest possible value for a vector element.
// \param max The largest possible value for a vector element.
// \return void
*/
template< typename Type   // Data type of the vector
        , bool TF >       // Transpose flag
template< typename Arg >  // Min/max argument type
inline void Rand< CompressedVector<Type,TF> >::randomize( CompressedVector<Type,TF>& vector,
                                                          const Arg& min, const Arg& max ) const
{
   const size_t size( vector.size() );

   if( size == 0UL ) return;

   const size_t nonzeros( rand<size_t>( 1UL, std::ceil( 0.5*size ) ) );

   randomize( vector, nonzeros, min, max );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a CompressedVector.
//
// \param vector The vector to be randomized.
// \param nonzeros The number of non-zero elements of the random vector.
// \param min The smallest possible value for a vector element.
// \param max The largest possible value for a vector element.
// \return void
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename Type   // Data type of the vector
        , bool TF >       // Transpose flag
template< typename Arg >  // Min/max argument type
inline void Rand< CompressedVector<Type,TF> >::randomize( CompressedVector<Type,TF>& vector,
                                                          size_t nonzeros, const Arg& min, const Arg& max ) const
{
   const size_t size( vector.size() );

   if( nonzeros > size ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of non-zero elements" );
   }

   if( size == 0UL ) return;

   vector.reset();
   vector.reserve( nonzeros );

   const Indices indices( 0UL, vector.size()-1UL, nonzeros );

   for( size_t index : indices ) {
      vector.append( index, rand<Type>( min, max ) );
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
