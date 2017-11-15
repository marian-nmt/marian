//=================================================================================================
/*!
//  \file blaze/math/proxy/SparseVectorProxy.h
//  \brief Header file for the SparseVectorProxy class
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

#ifndef _BLAZE_MATH_PROXY_SPARSEVECTORPROXY_H_
#define _BLAZE_MATH_PROXY_SPARSEVECTORPROXY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/SparseVector.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/SparseVector.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/typetraits/IsRowVector.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Types.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Proxy backend for sparse vector types.
// \ingroup math
//
// The SparseVectorProxy class serves as a backend for the Proxy class. It is used in case the
// data type represented by the proxy is a sparse vector and augments the Proxy interface by
// the complete interface required of sparse vectors.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
class SparseVectorProxy : public SparseVector< PT, IsRowVector<VT>::value >
{
 public:
   //**Type definitions****************************************************************************
   typedef ResultType_<VT>      ResultType;      //!< Result type for expression template evaluations.
   typedef TransposeType_<VT>   TransposeType;   //!< Transpose type for expression template evaluations.
   typedef ElementType_<VT>     ElementType;     //!< Type of the sparse vector elements.
   typedef ReturnType_<VT>      ReturnType;      //!< Return type for expression template evaluations.
   typedef CompositeType_<VT>   CompositeType;   //!< Data type for composite expression templates.
   typedef Reference_<VT>       Reference;       //!< Reference to a non-constant vector value.
   typedef ConstReference_<VT>  ConstReference;  //!< Reference to a constant vector value.
   typedef Iterator_<VT>        Iterator;        //!< Iterator over non-constant elements.
   typedef ConstIterator_<VT>   ConstIterator;   //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SMP assignments.
   enum : bool { smpAssignable = VT::smpAssignable };
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference operator[]( size_t index ) const;
   inline Reference at( size_t index ) const;

   inline Iterator      begin () const;
   inline ConstIterator cbegin() const;
   inline Iterator      end   () const;
   inline ConstIterator cend  () const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t   size() const;
   inline size_t   capacity() const;
   inline size_t   nonZeros() const;
   inline void     reset() const;
   inline void     clear() const;
   inline Iterator set( size_t index, const ElementType& value ) const;
   inline Iterator insert( size_t index, const ElementType& value ) const;
   inline void     append( size_t index, const ElementType& value, bool check=false ) const;
   inline void     erase( size_t index ) const;
   inline Iterator erase( Iterator pos ) const;
   inline Iterator erase( Iterator first, Iterator last ) const;
   inline void     resize( size_t n, bool preserve=true ) const;
   inline void     reserve( size_t n ) const;

   template< typename Other > inline void scale( const Other& scalar ) const;
   //@}
   //**********************************************************************************************

   //**Lookup functions****************************************************************************
   /*!\name Lookup functions */
   //@{
   inline Iterator find      ( size_t index ) const;
   inline Iterator find      ( size_t i, size_t j ) const;
   inline Iterator lowerBound( size_t index ) const;
   inline Iterator lowerBound( size_t i, size_t j ) const;
   inline Iterator upperBound( size_t index ) const;
   inline Iterator upperBound( size_t i, size_t j ) const;
   //@}
   //**********************************************************************************************

 private:
   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( VT );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Subscript operator for the direct access to vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function returns a reference to the accessed value at position \a index. In case the
// sparse vector does not yet store an element for index \a index, a new element is inserted
// into the sparse vector. A more efficient alternative for traversing the non-zero elements
// of the sparse vector are the begin() and end() functions.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline typename SparseVectorProxy<PT,VT>::Reference
   SparseVectorProxy<PT,VT>::operator[]( size_t index ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get()[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::invalid_argument Invalid access to restricted element.
// \exception std::out_of_range Invalid vector access index.
//
// This function returns a reference to the accessed value at position \a index. In case the
// sparse vector does not yet store an element for index \a index, a new element is inserted
// into the sparse vector. In contrast to the subscript operator this function always performs
// a check of the given access index.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline typename SparseVectorProxy<PT,VT>::Reference
   SparseVectorProxy<PT,VT>::at( size_t index ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().at( index );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the represented vector.
//
// \return Iterator to the first element of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline typename SparseVectorProxy<PT,VT>::Iterator SparseVectorProxy<PT,VT>::begin() const
{
   return (~*this).get().begin();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the represented vector.
//
// \return Iterator to the first element of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline typename SparseVectorProxy<PT,VT>::ConstIterator SparseVectorProxy<PT,VT>::cbegin() const
{
   return (~*this).get().cbegin();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the represented vector.
//
// \return Iterator just past the last element of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline typename SparseVectorProxy<PT,VT>::Iterator SparseVectorProxy<PT,VT>::end() const
{
   return (~*this).get().end();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the represented vector.
//
// \return Iterator just past the last element of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline typename SparseVectorProxy<PT,VT>::ConstIterator SparseVectorProxy<PT,VT>::cend() const
{
   return (~*this).get().cend();
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the current size/dimension of the represented vector.
//
// \return The size of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline size_t SparseVectorProxy<PT,VT>::size() const
{
   return (~*this).get().size();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the represented vector.
//
// \return The capacity of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline size_t SparseVectorProxy<PT,VT>::capacity() const
{
   return (~*this).get().capacity();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the represented vector.
//
// \return The number of non-zero elements in the vector.
//
// Note that the number of non-zero elements is always smaller than the current size of the
// sparse vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline size_t SparseVectorProxy<PT,VT>::nonZeros() const
{
   return (~*this).get().nonZeros();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset to the default initial value.
//
// \return void
//
// This function resets all elements of the vector to the default initial values.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline void SparseVectorProxy<PT,VT>::reset() const
{
   using blaze::reset;

   reset( (~*this).get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the represented vector.
//
// \return void
//
// This function clears the vector to its default initial state.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline void SparseVectorProxy<PT,VT>::clear() const
{
   using blaze::clear;

   clear( (~*this).get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting an element of the represented sparse vector.
//
// \param index The index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be set.
// \return Reference to the set value.
// \exception std::invalid_argument Invalid access to restricted element.
// \exception std::invalid_argument Invalid compressed vector access index.
//
// This function sets the value of an element of the sparse vector. In case the sparse vector
// already contains an element with index \a index its value is modified, else a new element
// with the given \a value is inserted.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline typename SparseVectorProxy<PT,VT>::Iterator
   SparseVectorProxy<PT,VT>::set( size_t index, const ElementType& value ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().set( index, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inserting an element into the represented sparse vector.
//
// \param index The index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be inserted.
// \return Reference to the inserted value.
// \exception std::invalid_argument Invalid access to restricted element.
// \exception std::invalid_argument Invalid compressed vector access index.
//
// This function inserts a new element into the sparse vector. However, duplicate elements are
// not allowed. In case the sparse vector already contains an element with index \a index, a
// \a std::invalid_argument exception is thrown.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline typename SparseVectorProxy<PT,VT>::Iterator
   SparseVectorProxy<PT,VT>::insert( size_t index, const ElementType& value ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().insert( index, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Appending an element to the represented sparse vector.
//
// \param index The index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be appended.
// \param check \a true if the new value should be checked for default values, \a false if not.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function provides a very efficient way to fill a compressed vector with elements. It
// appends a new element to the end of the compressed vector without any memory allocation.
// Therefore it is strictly necessary to keep the following preconditions in mind:
//
//  - the index of the new element must be strictly larger than the largest index of non-zero
//    elements in the compressed vector
//  - the current number of non-zero elements must be smaller than the capacity of the vector
//
// Ignoring these preconditions might result in undefined behavior! The optional \a check
// parameter specifies whether the new value should be tested for a default value. If the new
// value is a default value (for instance 0 in case of an integral element type) the value is
// not appended. Per default the values are not tested.
//
// \note Although append() does not allocate new memory, it still invalidates all iterators
// returned by the end() functions!
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline void SparseVectorProxy<PT,VT>::append( size_t index, const ElementType& value, bool check ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().append( index, value, check );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Erasing an element from the sparse vector.
//
// \param index The index of the element to be erased. The index has to be in the range \f$[0..N-1]\f$.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function erases an element from the sparse vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline void SparseVectorProxy<PT,VT>::erase( size_t index ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().erase( index );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Erasing an element from the sparse vector.
//
// \param pos Iterator to the element to be erased.
// \return Iterator to the element after the erased element.
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function erases an element from the sparse vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline typename SparseVectorProxy<PT,VT>::Iterator SparseVectorProxy<PT,VT>::erase( Iterator pos ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().erase( pos );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Erasing a range of elements from the compressed vector.
//
// \param first Iterator to first element to be erased.
// \param last Iterator just past the last element to be erased.
// \return Iterator to the element after the erased element.
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function erases a range of elements from the sparse vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline typename SparseVectorProxy<PT,VT>::Iterator
   SparseVectorProxy<PT,VT>::erase( Iterator first, Iterator last ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().erase( first, last );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the represented vector.
//
// \param n The new size of the vector.
// \param preserve \a true if the old values of the vector should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function changes the size of the vector. Depending on the type of the vector, during this
// operation new dynamic memory may be allocated in case the capacity of the vector is too small.
// Note that this function may invalidate all existing views (subvectors, ...) on the vector if
// it is used to shrink the vector. Additionally, the resize() operation potentially
// changes all vector elements. In order to preserve the old vector values, the \a preserve flag
// can be set to \a true.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline void SparseVectorProxy<PT,VT>::resize( size_t n, bool preserve ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().resize( n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the minimum capacity of the represented vector.
//
// \param n The new minimum capacity of the vector.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function increases the capacity of the compressed vector to at least \a n elements. The
// current values of the vector elements are preserved.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline void SparseVectorProxy<PT,VT>::reserve( size_t n ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().reserve( n );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the sparse vector by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the vector scaling.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
*/
template< typename PT       // Type of the proxy
        , typename VT >     // Type of the sparse vector
template< typename Other >  // Data type of the scalar value
inline void SparseVectorProxy<PT,VT>::scale( const Other& scalar ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().scale( scalar );
}
//*************************************************************************************************




//=================================================================================================
//
//  LOOKUP FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Searches for a specific vector element.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the element in case the index is found, end() iterator otherwise.
//
// This function can be used to check whether a specific element is contained in the sparse
// vector. It specifically searches for the element with index \a index. In case the element
// is found, the function returns an iterator to the element. Otherwise an iterator just past
// the last non-zero element of the compressed vector (the end() iterator) is returned. Note
// that the returned compressed vector iterator is subject to invalidation due to inserting
// operations via the subscript operator or the insert() function!
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline typename SparseVectorProxy<PT,VT>::Iterator
   SparseVectorProxy<PT,VT>::find( size_t index ) const
{
   return (~*this).get().find( index );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first index not less then the given index.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index not less then the given index, end() iterator otherwise.
//
// This function returns an iterator to the first element with an index not less then the given
// index. In combination with the upperBound() function this function can be used to create a
// pair of iterators specifying a range of indices. Note that the returned compressed vector
// iterator is subject to invalidation due to inserting operations via the subscript operator
// or the insert() function!
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline typename SparseVectorProxy<PT,VT>::Iterator
   SparseVectorProxy<PT,VT>::lowerBound( size_t index ) const
{
   return (~*this).get().lowerBound( index );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first index greater then the given index.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index greater then the given index, end() iterator otherwise.
//
// This function returns an iterator to the first element with an index greater then the given
// index. In combination with the lowerBound() function this function can be used to create a
// pair of iterators specifying a range of indices. Note that the returned compressed vector
// iterator is subject to invalidation due to inserting operations via the subscript operator
// or the insert() function!
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
inline typename SparseVectorProxy<PT,VT>::Iterator
   SparseVectorProxy<PT,VT>::upperBound( size_t index ) const
{
   return (~*this).get().upperBound( index );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name SparseVectorProxy global functions */
//@{
template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE typename SparseVectorProxy<PT,VT>::Iterator
   begin( const SparseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE typename SparseVectorProxy<PT,VT>::ConstIterator
   cbegin( const SparseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE typename SparseVectorProxy<PT,VT>::Iterator
   end( const SparseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE typename SparseVectorProxy<PT,VT>::ConstIterator
   cend( const SparseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE size_t size( const SparseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE size_t capacity( const SparseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE size_t nonZeros( const SparseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE void resize( const SparseVectorProxy<PT,VT>& proxy, size_t n, bool preserve=true );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE void reset( const SparseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE void clear( const SparseVectorProxy<PT,VT>& proxy );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the represented vector.
// \ingroup math
//
// \param proxy The given access proxy.
// \return Iterator to the first element of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
BLAZE_ALWAYS_INLINE typename SparseVectorProxy<PT,VT>::Iterator
   begin( const SparseVectorProxy<PT,VT>& proxy )
{
   return proxy.begin();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the represented vector.
// \ingroup math
//
// \param proxy The given access proxy.
// \return Iterator to the first element of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
BLAZE_ALWAYS_INLINE typename SparseVectorProxy<PT,VT>::ConstIterator
   cbegin( const SparseVectorProxy<PT,VT>& proxy )
{
   return proxy.cbegin();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the represented vector.
// \ingroup math
//
// \param proxy The given access proxy.
// \return Iterator just past the last element of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
BLAZE_ALWAYS_INLINE typename SparseVectorProxy<PT,VT>::Iterator
   end( const SparseVectorProxy<PT,VT>& proxy )
{
   return proxy.end();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the represented vector.
// \ingroup math
//
// \param proxy The given access proxy.
// \return Iterator just past the last element of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
BLAZE_ALWAYS_INLINE typename SparseVectorProxy<PT,VT>::ConstIterator
   cend( const SparseVectorProxy<PT,VT>& proxy )
{
   return proxy.cend();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current size/dimension of the represented vector.
// \ingroup math
//
// \param proxy The given access proxy.
// \return The size of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
BLAZE_ALWAYS_INLINE size_t size( const SparseVectorProxy<PT,VT>& proxy )
{
   return proxy.size();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the represented vector.
// \ingroup math
//
// \param proxy The given access proxy.
// \return The capacity of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
BLAZE_ALWAYS_INLINE size_t capacity( const SparseVectorProxy<PT,VT>& proxy )
{
   return proxy.capacity();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the represented vector.
// \ingroup math
//
// \param proxy The given access proxy.
// \return The number of non-zero elements in the vector.
//
// Note that the number of non-zero elements is always less than or equal to the current size
// of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
BLAZE_ALWAYS_INLINE size_t nonZeros( const SparseVectorProxy<PT,VT>& proxy )
{
   return proxy.nonZeros();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the represented vector.
// \ingroup math
//
// \param proxy The given access proxy.
// \param n The new size of the vector.
// \param preserve \a true if the old values of the vector should be preserved, \a false if not.
// \return void
//
// This function resizes the represented vector to the specified \a size.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
BLAZE_ALWAYS_INLINE void resize( const SparseVectorProxy<PT,VT>& proxy, size_t n, bool preserve )
{
   proxy.resize( n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the represented element to the default initial values.
// \ingroup math
//
// \param proxy The given access proxy.
// \return void
//
// This function resets all elements of the vector to the default initial values.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
BLAZE_ALWAYS_INLINE void reset( const SparseVectorProxy<PT,VT>& proxy )
{
   proxy.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the represented element.
// \ingroup math
//
// \param proxy The given access proxy.
// \return void
//
// This function clears the vector to its default initial state.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the sparse vector
BLAZE_ALWAYS_INLINE void clear( const SparseVectorProxy<PT,VT>& proxy )
{
   proxy.clear();
}
//*************************************************************************************************

} // namespace blaze

#endif
