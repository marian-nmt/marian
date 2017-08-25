//=================================================================================================
/*!
//  \file blaze/math/proxy/DenseVectorProxy.h
//  \brief Header file for the DenseVectorProxy class
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

#ifndef _BLAZE_MATH_PROXY_DENSEVECTORPROXY_H_
#define _BLAZE_MATH_PROXY_DENSEVECTORPROXY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsRowVector.h>
#include <blaze/system/Inline.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/Types.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Proxy backend for dense vector types.
// \ingroup math
//
// The DenseVectorProxy class serves as a backend for the Proxy class. It is used in case the
// data type represented by the proxy is a dense vector and augments the Proxy interface by
// the complete interface required of dense vectors.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
class DenseVectorProxy : public DenseVector< PT, IsRowVector<VT>::value >
{
 public:
   //**Type definitions****************************************************************************
   typedef ResultType_<VT>      ResultType;      //!< Result type for expression template evaluations.
   typedef TransposeType_<VT>   TransposeType;   //!< Transpose type for expression template evaluations.
   typedef ElementType_<VT>     ElementType;     //!< Type of the vector elements.
   typedef ReturnType_<VT>      ReturnType;      //!< Return type for expression template evaluations
   typedef CompositeType_<VT>   CompositeType;   //!< Data type for composite expression templates.
   typedef Reference_<VT>       Reference;       //!< Reference to a non-constant vector value.
   typedef ConstReference_<VT>  ConstReference;  //!< Reference to a constant vector value.
   typedef Pointer_<VT>         Pointer;         //!< Pointer to a non-constant vector value.
   typedef ConstPointer_<VT>    ConstPointer;    //!< Pointer to a constant vector value.
   typedef Iterator_<VT>        Iterator;        //!< Iterator over non-constant elements.
   typedef ConstIterator_<VT>   ConstIterator;   //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SIMD optimization.
   enum : bool { simdEnabled = VT::simdEnabled };

   //! Compilation flag for SMP assignments.
   enum : bool { smpAssignable = VT::smpAssignable };
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference operator[]( size_t index ) const;
   inline Reference at( size_t index ) const;

   inline Pointer       data  () const;
   inline Iterator      begin () const;
   inline ConstIterator cbegin() const;
   inline Iterator      end   () const;
   inline ConstIterator cend  () const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t size() const;
   inline size_t capacity() const;
   inline size_t nonZeros() const;
   inline void   reset() const;
   inline void   clear() const;
   inline void   resize( size_t n, bool preserve=true ) const;
   inline void   extend( size_t n, bool preserve=true ) const;
   inline void   reserve( size_t n ) const;

   template< typename Other > inline void scale( const Other& scalar ) const;
   //@}
   //**********************************************************************************************

 private:
   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( VT );
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
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
inline typename DenseVectorProxy<PT,VT>::Reference
   DenseVectorProxy<PT,VT>::operator[]( size_t index ) const
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
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
inline typename DenseVectorProxy<PT,VT>::Reference
   DenseVectorProxy<PT,VT>::at( size_t index ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().at( index );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to vector elements.
//
// \return Pointer to the internal element storage.
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function returns a pointer to the internal storage of the dynamic vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
inline typename DenseVectorProxy<PT,VT>::Pointer DenseVectorProxy<PT,VT>::data() const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().data();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the represented vector.
//
// \return Iterator to the first element of the vector.
// \exception std::invalid_argument Invalid access to restricted element.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
inline typename DenseVectorProxy<PT,VT>::Iterator DenseVectorProxy<PT,VT>::begin() const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().begin();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the represented vector.
//
// \return Iterator to the first element of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
inline typename DenseVectorProxy<PT,VT>::ConstIterator DenseVectorProxy<PT,VT>::cbegin() const
{
   return (~*this).get().cbegin();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the represented vector.
//
// \return Iterator just past the last element of the vector.
// \exception std::invalid_argument Invalid access to restricted element.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
inline typename DenseVectorProxy<PT,VT>::Iterator DenseVectorProxy<PT,VT>::end() const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().end();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the represented vector.
//
// \return Iterator just past the last element of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
inline typename DenseVectorProxy<PT,VT>::ConstIterator DenseVectorProxy<PT,VT>::cend() const
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
        , typename VT >  // Type of the dense vector
inline size_t DenseVectorProxy<PT,VT>::size() const
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
        , typename VT >  // Type of the dense vector
inline size_t DenseVectorProxy<PT,VT>::capacity() const
{
   return (~*this).get().capacity();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the represented vector.
//
// \return The number of non-zero elements in the vector.
//
// Note that the number of non-zero elements is always less than or equal to the current size
// of the vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
inline size_t DenseVectorProxy<PT,VT>::nonZeros() const
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
        , typename VT >  // Type of the dense vector
inline void DenseVectorProxy<PT,VT>::reset() const
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
        , typename VT >  // Type of the dense vector
inline void DenseVectorProxy<PT,VT>::clear() const
{
   using blaze::clear;

   clear( (~*this).get() );
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
// it is used to shrink the vector. Additionally, the resize() operation potentially changes all
// vector elements. In order to preserve the old vector values, the \a preserve flag can be set
// to \a true. However, note that depending on the type of the vector new vector elements may not
// initialized!
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
inline void DenseVectorProxy<PT,VT>::resize( size_t n, bool preserve ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().resize( n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Extending the size of the represented vector.
//
// \param n Number of additional vector elements.
// \param preserve \a true if the old values of the vector should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function extends the size of the vector. Depending on the type of the vector, during this
// operation new dynamic memory may be allocated in case the capacity of the vector is too small.
// Therefore this function potentially changes all vector elements. In order to preserve the old
// vector values, the \a preserve flag can be set to \a true. However, note that depending on the
// type vector new vector elements may not initialized!
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
inline void DenseVectorProxy<PT,VT>::extend( size_t n, bool preserve ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().extend( n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the minimum capacity of the represented vector.
//
// \param n The new minimum capacity of the vector.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function increases the capacity of the vector to at least \a n elements. The current
// values of the vector elements are preserved.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
inline void DenseVectorProxy<PT,VT>::reserve( size_t n ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().reserve( n );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the vector by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the vector scaling.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
*/
template< typename PT       // Type of the proxy
        , typename VT >     // Type of the dense vector
template< typename Other >  // Data type of the scalar value
inline void DenseVectorProxy<PT,VT>::scale( const Other& scalar ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().scale( scalar );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DenseVectorProxy global functions */
//@{
template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE typename DenseVectorProxy<PT,VT>::Iterator
   begin( const DenseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE typename DenseVectorProxy<PT,VT>::ConstIterator
   cbegin( const DenseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE typename DenseVectorProxy<PT,VT>::Iterator
   end( const DenseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE typename DenseVectorProxy<PT,VT>::ConstIterator
   cend( const DenseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE size_t size( const DenseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE size_t capacity( const DenseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE size_t nonZeros( const DenseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE void resize( const DenseVectorProxy<PT,VT>& proxy, size_t n, bool preserve=true );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE void reset( const DenseVectorProxy<PT,VT>& proxy );

template< typename PT, typename VT >
BLAZE_ALWAYS_INLINE void clear( const DenseVectorProxy<PT,VT>& proxy );
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
        , typename VT >  // Type of the dense vector
BLAZE_ALWAYS_INLINE typename DenseVectorProxy<PT,VT>::Iterator
   begin( const DenseVectorProxy<PT,VT>& proxy )
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
        , typename VT >  // Type of the dense vector
BLAZE_ALWAYS_INLINE typename DenseVectorProxy<PT,VT>::ConstIterator
   cbegin( const DenseVectorProxy<PT,VT>& proxy )
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
        , typename VT >  // Type of the dense vector
BLAZE_ALWAYS_INLINE typename DenseVectorProxy<PT,VT>::Iterator
   end( const DenseVectorProxy<PT,VT>& proxy )
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
        , typename VT >  // Type of the dense vector
BLAZE_ALWAYS_INLINE typename DenseVectorProxy<PT,VT>::ConstIterator
   cend( const DenseVectorProxy<PT,VT>& proxy )
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
        , typename VT >  // Type of the dense vector
BLAZE_ALWAYS_INLINE size_t size( const DenseVectorProxy<PT,VT>& proxy )
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
        , typename VT >  // Type of the dense vector
BLAZE_ALWAYS_INLINE size_t capacity( const DenseVectorProxy<PT,VT>& proxy )
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
        , typename VT >  // Type of the dense vector
BLAZE_ALWAYS_INLINE size_t nonZeros( const DenseVectorProxy<PT,VT>& proxy )
{
   return proxy.nonZeros();
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for non-resizable vectors.
// \ingroup math
//
// \param proxy The given access proxy
// \param n The new size of the vector.
// \param preserve \a true if the old values of the vector should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Vector cannot be resized.
//
// This function tries to change the number of rows and columns of a non-resizable vector. Since
// the vector cannot be resized, in case the specified size is not identical to the current size
// of the vector, a \a std::invalid_argument exception is thrown.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
BLAZE_ALWAYS_INLINE DisableIf_< IsResizable<VT> >
   resize_backend( const DenseVectorProxy<PT,VT>& proxy, size_t n, bool preserve )
{
   UNUSED_PARAMETER( preserve );

   if( proxy.size() != n ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector cannot be resized" );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for resizable vectors.
// \ingroup math
//
// \param proxy The given access proxy
// \param n The new size of the vector.
// \param preserve \a true if the old values of the vector should be preserved, \a false if not.
// \return void
//
// This function changes the size of the given resizable vector.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
BLAZE_ALWAYS_INLINE EnableIf_< IsResizable<VT> >
   resize_backend( const DenseVectorProxy<PT,VT>& proxy, size_t n, bool preserve )
{
   proxy.resize( n, preserve );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the represented vector.
// \ingroup math
//
// \param proxy The given access proxy.
// \param n The new size of the vector.
// \param preserve \a true if the old values of the vector should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Vector cannot be resized.
//
// This function resizes the represented vector to the specified \a size. Note that in contrast
// to the \c resize() member function, which is only available on resizable vector types, this
// function can be used on both resizable and non-resizable vectors. In case the type \a VT of
// the represented vector is resizable (i.e. provides a \c resize() function), the type-specific
// \c resize() member function is called. Depending on the type \a VT, this may result in the
// allocation of new dynamic memory and the invalidation of existing views (subvectors, ...). In
// case \a VT is non-resizable (i.e. does not provide a \c resize() function) and if the specified
// size is not identical to the current size of the vector, a \a std::invalid_argument exception
// is thrown.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
BLAZE_ALWAYS_INLINE void resize( const DenseVectorProxy<PT,VT>& proxy, size_t n, bool preserve )
{
   resize_backend( proxy, n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the represented vector to the default initial values.
// \ingroup math
//
// \param proxy The given access proxy.
// \return void
//
// This function resets all elements of the vector to the default initial values.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
BLAZE_ALWAYS_INLINE void reset( const DenseVectorProxy<PT,VT>& proxy )
{
   proxy.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the represented vector.
// \ingroup math
//
// \param proxy The given access proxy.
// \return void
//
// This function clears the vector to its default initial state.
*/
template< typename PT    // Type of the proxy
        , typename VT >  // Type of the dense vector
BLAZE_ALWAYS_INLINE void clear( const DenseVectorProxy<PT,VT>& proxy )
{
   proxy.clear();
}
//*************************************************************************************************

} // namespace blaze

#endif
