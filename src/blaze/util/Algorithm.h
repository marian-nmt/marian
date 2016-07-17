//=================================================================================================
/*!
//  \file blaze/util/Algorithm.h
//  \brief Headerfile for generic algorithms
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

#ifndef _BLAZE_UTIL_ALGORITHM_H_
#define _BLAZE_UTIL_ALGORITHM_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <blaze/util/constraints/DerivedFrom.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsAssignable.h>


namespace blaze {

//=================================================================================================
//
//  TRANSFER
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Transfers the elements from the given source range to the destination range.
//
// \param first Iterator to the first element of the source range.
// \param last Iterator to the element one past the last element of the source range.
// \param dest Iterator to the first element of the destination range.
// \return Output iterator to the element one past the last copied element.
//
// This function transfers the elements in the range \f$ [first,last) \f$ to the specified
// destination range. In case the elements provide a no-throw move assignment, the transfer
// operation is handled via move. Else the elements are copied.
*/
template< typename InputIterator
        , typename OutputIterator >
OutputIterator transfer( InputIterator first, InputIterator last, OutputIterator dest )
{
   using ValueType = typename std::iterator_traits<InputIterator>::value_type;

   if( IsNothrowMoveAssignable<ValueType>::value ) {
      return std::move( first, last, dest );
   }
   else {
      return std::copy( first, last, dest );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  POLYMORPHIC COUNT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Counts the pointer to objects with dynamic type \a D.
//
// \param first Iterator to the first pointer of the pointer range.
// \param last Iterator to the pointer one past the last pointer of the pointer range.
// \return The number of objects with dynamic type \a D.
//
// This function traverses the range \f$ [first,last) \f$ of pointers to objects with static
// type \a S and counts all polymorphic pointers to objects of dynamic type \a D. Note that
// in case \a D is not a type derived from \a S, a compile time error is created!
*/
template< typename D    // Dynamic type of the objects
        , typename S >  // Static type of the objects
inline size_t polymorphicCount( S *const * first, S *const * last )
{
   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_DERIVED_FROM( D, S );

   size_t count( 0 );
   for( S *const * it=first; it!=last; ++it )
      if( dynamic_cast<D*>( *it ) ) ++count;
   return count;
}
//*************************************************************************************************




//=================================================================================================
//
//  POLYMORPHIC FIND
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Finds the next pointer to an object with dynamic type \a D.
//
// \param first Iterator to the first pointer of the pointer range.
// \param last Iterator to the pointer one past the last pointer of the pointer range.
// \return The next pointer to an object with dynamic type \a D.
//
// This function traverses the range \f$ [first,last) \f$ of pointers to objects with static
// type \a S until it finds the next polymorphic pointer to an object of dynamic type \a D.
// Note that in case \a D is not a type derived from \a S, a compile time error is created!
*/
template< typename D    // Dynamic type of the objects
        , typename S >  // Static type of the objects
inline S *const * polymorphicFind( S *const * first, S *const * last )
{
   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_DERIVED_FROM( D, S );

   while( first != last && !dynamic_cast<D*>( *first ) ) ++first;
   return first;
}
//*************************************************************************************************

} // namespace blaze

#endif
