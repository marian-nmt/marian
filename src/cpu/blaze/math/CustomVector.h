//=================================================================================================
/*!
//  \file blaze/math/CustomVector.h
//  \brief Header file for the complete CustomVector implementation
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

#ifndef _BLAZE_MATH_CUSTOMVECTOR_H_
#define _BLAZE_MATH_CUSTOMVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/dense/CustomVector.h>
#include <blaze/math/dense/DynamicVector.h>
#include <blaze/math/dense/StaticVector.h>
#include <blaze/math/DenseVector.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/util/Random.h>


namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for CustomVector.
// \ingroup random
//
// This specialization of the Rand class randomizes instances of CustomVector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
class Rand< CustomVector<Type,AF,PF,TF> >
{
 public:
   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( CustomVector<Type,AF,PF,TF>& vector ) const;

   template< typename Arg >
   inline void randomize( CustomVector<Type,AF,PF,TF>& vector, const Arg& min, const Arg& max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a CustomVector.
//
// \param vector The vector to be randomized.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline void Rand< CustomVector<Type,AF,PF,TF> >::randomize( CustomVector<Type,AF,PF,TF>& vector ) const
{
   using blaze::randomize;

   const size_t size( vector.size() );
   for( size_t i=0UL; i<size; ++i ) {
      randomize( vector[i] );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a CustomVector.
//
// \param vector The vector to be randomized.
// \param min The smallest possible value for a vector element.
// \param max The largest possible value for a vector element.
// \return void
*/
template< typename Type   // Data type of the vector
        , bool AF         // Alignment flag
        , bool PF         // Padding flag
        , bool TF >       // Transpose flag
template< typename Arg >  // Min/max argument type
inline void Rand< CustomVector<Type,AF,PF,TF> >::randomize( CustomVector<Type,AF,PF,TF>& vector,
                                                            const Arg& min, const Arg& max ) const
{
   using blaze::randomize;

   const size_t size( vector.size() );
   for( size_t i=0UL; i<size; ++i ) {
      randomize( vector[i], min, max );
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
