//=================================================================================================
/*!
//  \file blaze/util/PtrIterator.h
//  \brief Iterator class for pointer vectors
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

#ifndef _BLAZE_UTIL_PTRITERATOR_H_
#define _BLAZE_UTIL_PTRITERATOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Implementation of an iterator for pointer vectors.
// \ingroup util
//
// The PtrIterator class follows the example of the random-access iterator classes of the STL.
// However, the focus of this iterator implementation is the use with (polymorphic) pointers.
// The implementation of the Blaze library eases the use of iterators over a range of pointers
// and improves the semantics on these pointers.\n
//
// In contrast to the STL iterators, the PtrIterator class slightly changes the meaning of the
// access operators. Consider the following example:

   \code
   // Definition of class A
   class A
   {
    public:
      A( int i=0 ):i_(i) {}

      void set( int i )       { i_ = i; }
      int  get()        const { return i_; }

    private:
      int i_;
   };

   // Definition of a pointer vector for class A
   typedef blaze::PtrVector<A>  AVector;

   AVector vector;
   AVector::Iterator it = vector.begin();

   // The subscript operator returns a handle to the underlying object
   A* a1 = it[0];

   // The dereference operator returns a handle to the underlying object
   A* a2 = *it;

   // The member access operator offers direct access to the underlying object
   it->set( 2 );
   \endcode

// The constant iterators (iterator over constant objects) prohibit the access to non-const
// member functions. Therefore the following operation results in a compile-time error:

   \code
   AVector vector;
   AVector::ConstIterator it = vector.begin();

   it->set( 2 );  // Compile-time error!
   \endcode
*/
template< typename Type >
class PtrIterator
{
 public:
   //**Type definitions****************************************************************************
   // blaze naming convention
   typedef std::random_access_iterator_tag  IteratorCategory;   //!< The iterator category.
   typedef Type*                            ValueType;          //!< Type of the underlying pointers.
   typedef Type*                            PointerType;        //!< Pointer return type.
   typedef ValueType const&                 ReferenceType;      //!< Reference return type.
   typedef ValueType const*                 IteratorType;       //!< Type of the internal pointer.
   typedef std::ptrdiff_t                   DifferenceType;     //!< Difference between two iterators.

   // STL iterator requirements
   typedef IteratorCategory                 iterator_category;  //!< The iterator category.
   typedef ValueType                        value_type;         //!< Type of the underlying pointers.
   typedef PointerType                      pointer;            //!< Pointer return type.
   typedef ReferenceType                    reference;          //!< Reference return type.
   typedef DifferenceType                   difference_type;    //!< Difference between two iterators.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
            inline PtrIterator();
   explicit inline PtrIterator( const IteratorType& it );

   template< typename Other >
   inline PtrIterator( const PtrIterator<Other>& it );

   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Copy assignment operator********************************************************************
   // No explicitly declared copy assignment operator.
   //**********************************************************************************************

   //**Operators***********************************************************************************
   /*!\name Operators */
   //@{
   inline PtrIterator&   operator++();
   inline PtrIterator    operator++( int );
   inline PtrIterator&   operator--();
   inline PtrIterator    operator--( int );
   inline PtrIterator&   operator+=( DifferenceType n );
   inline PtrIterator    operator+ ( DifferenceType n )      const;
   inline PtrIterator&   operator-=( DifferenceType n );
   inline PtrIterator    operator- ( DifferenceType n )      const;
   inline DifferenceType operator- ( const PtrIterator& it ) const;
   //@}
   //**********************************************************************************************

   //**Access operators****************************************************************************
   /*!\name Access operators */
   //@{
   inline PointerType operator[]( DifferenceType n ) const;
   inline PointerType operator*()                    const;
   inline PointerType operator->()                   const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline const IteratorType& base() const;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   IteratorType it_;  //!< Pointer to the current memory location.
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Default constructor for PtrIterator.
*/
template< typename Type >
inline PtrIterator<Type>::PtrIterator()
   : it_( nullptr )  // Pointer to the current memory location
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Standard constructor for PtrIterator.
//
// \param it The value of the iterator.
*/
template< typename Type >
inline PtrIterator<Type>::PtrIterator( const IteratorType& it )
   : it_( it )  // Pointer to the current memory location
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different PtrIterator instances.
//
// \param it The foreign PtrIterator instance to be copied.
*/
template< typename Type >
template< typename Other >
inline PtrIterator<Type>::PtrIterator( const PtrIterator<Other>& it )
   : it_( it.base() )  // Pointer to the current memory location
{}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Pre-increment operator.
//
// \return Reference to the incremented pointer iterator.
*/
template< typename Type >
inline PtrIterator<Type>& PtrIterator<Type>::operator++()
{
   ++it_;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Post-increment operator.
//
// \return The incremented pointer iterator.
*/
template< typename Type >
inline PtrIterator<Type> PtrIterator<Type>::operator++( int )
{
   PtrIterator tmp( *this );
   ++it_;
   return tmp;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Pre-decrement operator.
//
// \return Reference to the decremented pointer iterator.
*/
template< typename Type >
inline PtrIterator<Type>& PtrIterator<Type>::operator--()
{
   --it_;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Post-decrement operator.
//
// \return The decremented pointer iterator.
*/
template< typename Type >
inline PtrIterator<Type> PtrIterator<Type>::operator--( int )
{
   PtrIterator tmp( *this );
   --it_;
   return tmp;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Shifting the iterator by \a n elements to the higher elements.
//
// \param n The number of elements.
// \return Reference to the shifted pointer iterator.
*/
template< typename Type >
inline PtrIterator<Type>& PtrIterator<Type>::operator+=( DifferenceType n )
{
   it_ += n;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Shifting the iterator by \a n elements to the higher elements.
//
// \param n The number of elements.
// \return The shifted pointer iterator.
*/
template< typename Type >
inline PtrIterator<Type> PtrIterator<Type>::operator+( DifferenceType n ) const
{
   return PtrIterator( it_ + n );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Shifting the iterator by \a n elements to the lower elements.
//
// \param n The number of elements.
// \return Reference to the shifted pointer iterator.
*/
template< typename Type >
inline PtrIterator<Type>& PtrIterator<Type>::operator-=( DifferenceType n )
{
   it_ -= n;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Shifting the iterator by \a n elements to the lower elements.
//
// \param n The number of elements.
// \return The shifted pointer iterator.
*/
template< typename Type >
inline PtrIterator<Type> PtrIterator<Type>::operator-( DifferenceType n ) const
{
   return PtrIterator( it_ - n );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculating the number of elements between two pointer iterators.
//
// \param it The right hand side iterator.
// \return The number of elements between the two pointer iterators.
*/
template< typename Type >
inline typename PtrIterator<Type>::DifferenceType PtrIterator<Type>::operator-( const PtrIterator& it ) const
{
   return it_ - it.it_;
}
//*************************************************************************************************




//=================================================================================================
//
//  ACCESS OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Subscript operator for the direct element access.
//
// \param index Access index. Accesses the element \a index elements away from the current iterator position.
// \return Handle to the accessed element.
*/
template< typename Type >
inline typename PtrIterator<Type>::PointerType PtrIterator<Type>::operator[]( DifferenceType index ) const
{
   return it_[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a handle to the element at the current iterator position.
//
// \return Handle to the element at the current iterator position.
*/
template< typename Type >
inline typename PtrIterator<Type>::PointerType PtrIterator<Type>::operator*() const
{
   return *it_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Direct access to the element at the current iterator position.
//
// \return Reference to the element at the current iterator position.
*/
template< typename Type >
inline typename PtrIterator<Type>::PointerType PtrIterator<Type>::operator->() const
{
   return *it_;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Access to the underlying member of the pointer iterator.
//
// \return Pointer to the current memory location.
*/
template< typename Type >
inline const typename PtrIterator<Type>::IteratorType& PtrIterator<Type>::base() const
{
   return it_;
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name PtrIterator operators */
//@{
template< typename TypeL, typename TypeR >
inline bool operator==( const PtrIterator<TypeL>& lhs, const PtrIterator<TypeR>& rhs );

template< typename TypeL, typename TypeR >
inline bool operator!=( const PtrIterator<TypeL>& lhs, const PtrIterator<TypeR>& rhs );

template< typename TypeL, typename TypeR >
inline bool operator<( const PtrIterator<TypeL>& lhs, const PtrIterator<TypeR>& rhs );

template< typename TypeL, typename TypeR >
inline bool operator>( const PtrIterator<TypeL>& lhs, const PtrIterator<TypeR>& rhs );

template< typename TypeL, typename TypeR >
inline bool operator<=( const PtrIterator<TypeL>& lhs, const PtrIterator<TypeR>& rhs );

template< typename TypeL, typename TypeR >
inline bool operator>=( const PtrIterator<TypeL>& lhs, const PtrIterator<TypeR>& rhs );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between two PtrIterator objects.
//
// \param lhs The left-hand side pointer iterator.
// \param rhs The right-hand side pointer iterator.
// \return \a true if the iterators point to the same element, \a false if not.
*/
template< typename TypeL, typename TypeR >
inline bool operator==( const PtrIterator<TypeL>& lhs, const PtrIterator<TypeR>& rhs )
{
   return lhs.base() == rhs.base();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between two PtrIterator objects.
//
// \param lhs The left-hand side pointer iterator.
// \param rhs The right-hand side pointer iterator.
// \return \a true if the iterators don't point to the same element, \a false if they do.
*/
template< typename TypeL, typename TypeR >
inline bool operator!=( const PtrIterator<TypeL>& lhs, const PtrIterator<TypeR>& rhs )
{
   return lhs.base() != rhs.base();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-than comparison between two PtrIterator objects.
//
// \param lhs The left-hand side pointer iterator.
// \param rhs The right-hand side pointer iterator.
// \return \a true if the left-hand side iterator points to a lower element, \a false if not.
*/
template< typename TypeL, typename TypeR >
inline bool operator<( const PtrIterator<TypeL>& lhs, const PtrIterator<TypeR>& rhs )
{
   return lhs.base() < rhs.base();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-than comparison between two PtrIterator objects.
//
// \param lhs The left-hand side pointer iterator.
// \param rhs The right-hand side pointer iterator.
// \return \a true if the left-hand side iterator points to a higher element, \a false if not.
*/
template< typename TypeL, typename TypeR >
inline bool operator>( const PtrIterator<TypeL>& lhs, const PtrIterator<TypeR>& rhs )
{
   return lhs.base() > rhs.base();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between two PtrIterator objects.
//
// \param lhs The left-hand side pointer iterator.
// \param rhs The right-hand side pointer iterator.
// \return \a true if the left-hand side iterator points to a lower or the same element, \a false if not.
*/
template< typename TypeL, typename TypeR >
inline bool operator<=( const PtrIterator<TypeL>& lhs, const PtrIterator<TypeR>& rhs )
{
   return lhs.base() <= rhs.base();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-or-equal-than comparison between two PtrIterator objects.
//
// \param lhs The left-hand side pointer iterator.
// \param rhs The right-hand side pointer iterator.
// \return \a true if the left-hand side iterator points to a higher or the same element, \a false if not.
*/
template< typename TypeL, typename TypeR >
inline bool operator>=( const PtrIterator<TypeL>& lhs, const PtrIterator<TypeR>& rhs )
{
   return lhs.base() >= rhs.base();
}
//*************************************************************************************************

} // namespace blaze

#endif
