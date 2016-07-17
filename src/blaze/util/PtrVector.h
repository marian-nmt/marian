//=================================================================================================
/*!
//  \file blaze/util/PtrVector.h
//  \brief Implementation of a vector for (polymorphic) pointers
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

#ifndef _BLAZE_UTIL_PTRVECTOR_H_
#define _BLAZE_UTIL_PTRVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <blaze/util/Algorithm.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Convertible.h>
#include <blaze/util/constraints/DerivedFrom.h>
#include <blaze/util/Exception.h>
#include <blaze/util/policies/PtrDelete.h>
#include <blaze/util/policies/OptimalGrowth.h>
#include <blaze/util/PtrIterator.h>
#include <blaze/util/Template.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Implementation of a vector for (polymorphic) pointers.
// \ingroup util
//
// \section basics Basic usage
//
// The \a std::vector is one of the standard libraries most useful tools. It is the standard
// solution for a dynamically allocated, automatically growing, and memory managed array. It
// provides fast random access to its elements, since a vector guarantees that the elements
// lie adjacent in memory and manages the dynamically allocated memory according to the RAII
// idiom.\n
// Yet there are some situations, where users of \a std::vector experience several drawbacks,
// especially when \a std::vector is used in combination with pointers. For instance, a
// \a const_iterator over a range of pointers will not allow the stored pointers to change,
// but the objects behind the pointers remain changeable. The following example illustrates
// that it is possible to change the values of \a double values through an iterator-to-const:

   \code
   typedef std::vector<double*>  Doubles;

   Doubles doubles;  // Creating a vector for pointers to double values

   // Filling the vector with pointers to double values. All values are initialized with 1.
   for( size_t i=0; i<10; ++i )
      doubles.push_back( new double( 1.0 ) );

   // Accessing the first rigid body
   Doubles::const_iterator first = doubles.begin();
   **first = 2.0;  // Changes the double value through an iterator-to-const
   \endcode

// The basic reason for this behavior is that \a std::vector is unaware of the fact that it
// stores pointers instead of objects and therefore the pointer are considered constant, not
// the objects behind the pointer.\n
// Another drawback of \a std::vector is the fact that during destruction of a vector object
// the dynamically allocated bodies are not deleted. Again, \a std::vector is unaware of the
// special property of pointers and therefore does not apply any kind of deletion policy. It
// basically calls the default destructor for pointers, which in turn does nothing and
// especially does not destroy the attached objects.\n
// A different approach is taken by the Boost \a ptr_vector. A \a ptr_vector is perfectly
// aware of the fact that is stores pointers to dynamically objects (and in consequence may
// only be used with pointers to dynamically allocated objects) and takes full responsibilty
// for these resources. However, in order to accomplish this task, \a ptr_vector completely
// abstracts from the fact that it stores pointers and provides a view as if it would contain
// objects instead of pointers. Unfortunately, this strict memory management might cause
// problems, for instance in case the vector to pointers is used both internally (including
// proper resource management) and outside by the user (without any resource management).\n
// In case both \a std::vector and \a boost::ptr_vector are not suitable data structures, the
// Blaze library provides a special vector container for pointers, which is a cross of the
// functionalities of the \a std::vector and \a ptr_vector. The Blaze PtrVector is not a RAII
// class in the classic sense (as for instance the Boost \a ptr_vector) since it does not
// strictly encapsule the resource management. As in the case of \a std::vector, it still is
// the responsibility of a user of PtrVector to manage the resources accordingly. However,
// PtrVector can be used internally to store pointers to dynamically allocated objects and
// resources within RAII classes, and outside by a user as storage for handles to resources
// that are managed elsewhere. In contrast to the \a boost::ptr_vector, the PtrVector provides
// full access to the contained pointers, but its iterators work similar to the \a ptr_vector
// iterator and only provide access to the objects behind the pointers, creating the illusion
// that objects are stored instead of pointers:

   \code
   typedef blaze::PtrVector<double>  Doubles;
   Doubles doubles;  // Creating an empty PtrVector for pointers to double values

   doubles.pushBack( new double(1.0) ); // A new pointer-to-double is added to the vector

   double_vector::iterator first = doubles.begin();
   *first = 2.0;  // No indirection needed

   Doubles::ConstIterator second( first+1 );
   *second = 3.0;  // Compile time error! It is not possible to change double
                   // values via an iterator-to-const
   \endcode

// Notice the differences in the usage of the iterator in contrast to the \a std::vector and
// \a boost::ptr_vector. In contrast to them the functions of PtrVector follow the naming
// convention of the Blaze library (i.e. pushBack instead of push_back). In addition, the
// underlying iterator adds an additional dereference to all access operators, which eases
// the access to the underlying objects:

   \code
   // STL style:
   **first = 2.0;

   // pe style:
   *first = 2.0;
   \endcode

// A noteworthy difference between the STL vector and the pointer vector is the used template
// argument: instead of the pointer type, the Blaze pointer vector is only provided with the
// type of the underlying objects:

   \code
   // STL style:
   std::vector<double*> vector;

   // pe style:
   blaze::PtrVector<double> vector;
   \endcode

// Additionally, the Blaze pointer vector offers some limited possibilities to configure the
// memory management and the growth of the internal storage, and implements special features
// for polymorphic pointers, as for instance a convenient way to iterate over a subset of
// polymorphic objects contained in the pointer vector.\n\n
//
//
// \section polymorphic Polymorphic pointers
//
// For polymorphic pointers, the PtrVector class additionally offers two special iterators to
// iterate over all objects of a specific type: the CastIterator and ConstCastIterator.

   \code
   // Definition of class A and the derived type B
   class A { ... };
   class B : public A { ... };

   // Definition of function f for non-const pointer vectors
   void f( blaze::PtrVector<A>& vector )
   {
      blaze::PtrVector<A>::CastIterator<B> begin = vector.begin<B>();
      blaze::PtrVector<A>::CastIterator<B> end   = vector.end<B>();

      // Loop over all objects of type B contained in the vector
      for( ; begin!=end; ++begin )
         ...
   }

   // Definition of function f for const pointer vectors
   void f( const blaze::PtrVector<A>& vector )
   {
      blaze::PtrVector<A>::ConstCastIterator<B> begin = vector.begin<B>();
      blaze::PtrVector<A>::ConstCastIterator<B> end   = vector.end<B>();

      // Loop over all objects of type B contained in the vector
      for( ; begin!=end; ++begin )
   }
   \endcode

// In the example, the cast iterators are used to iterate over all objects of type \a B within
// the pointer vector, where \a B must be a type derived from \a A. The attempt to use these
// iterators for types that are not derived from \a A results in a compile time error. Note that
// the usage of the cast iterators is computaionally more expensive than the use of the standard
// iterators. Therefore these iterators should not be used unless a down-cast is really necessary,
// e.g. in order to access a type specific function.\n\n
//
//
// \section container Using a pointer vector within other container classes
//
// If a pointer vector is used within an other container and is used to store polymorphic pointers,
// you might face the problem of not being able to create type definitions for the cast iterators.
// Whereas it is possible to create typedefs for the standard iterators, it is unfortunately not
// possible (yet) to create type definitions for template classes. In order to create a new return
// type within the container, the following approach could be taken:

   \code
   template< typename A >
   class Container
   {
    public:
      template< typename C >
      struct CastIterator : public blaze::PtrVector<A>::CastIterator<C>
      {
         CastIterator( const blaze::PtrVector<A>::CastIterator<C>& it )
            : blaze::PtrVector<A>::CastIterator<C>( it )  // Initializing the base class
         {}
      };

      template< typename C >
      CastIterator<C> begin();

      template< typename C >
      CastIterator<C> end();

    private:
      blaze::PtrVector<A> vector_;
   };
   \endcode

// Instead of a typedef within the Container class, a new class CastIterator is derived from the
// PtrVector::CastIterator class. This approach acts similar as the typedef as a user can now
// use the Container as follows:

   \code
   class A { ... };
   class B : public A { ... };

   Container<A>::CastIterator<B> begin;
   \endcode

// This provides the same abstraction from the internal implementation as the desired typedef. The
// same approach could be taken for a ConstCastIterator definition.\n\n
//
//
// \section adaptions Adapting a pointer vector
//
// The growth and deletion behavior of the PtrVector class can be adapted to any specific task. The
// second template argument of the PtrVector specifies the growth rate. The following growth rates
// can be selected:
//
//  - ConstantGrowth
//  - LinearGrowth
//  - OptimalGrowth (the default behavior)
//
// The third template argument of the PtrVector specifies the deletion behavior for the case that
// the pointer vector is destroyed. Note that the deletion behavior has only limited effect on
// the memory management of the contained resources. For instance, copying a PtrVector always
// results in a shallow copy, i.e., the contained resources are not copied/cloned. Therefore the
// deletion policy should be considered a convenience functionality in the context of a resource
// managing class. The following policies can be selected:
//
//  - NoDelete : No deletion of the contained pointers.
//  - PtrDelete : Applies \a delete to all contained pointers (the default behavior).
//  - ArrayDelete : Applies \a delete[] to all contained pointers.\n\n
*/
template< typename T                    // Type
        , typename D = PtrDelete        // Deletion policy
        , typename G = OptimalGrowth >  // Growth policy
class PtrVector
{
 private:
   //**Friend declarations*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   template< typename T2, typename D2, typename G2 > friend class PtrVector;
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   // Blaze naming convention
   typedef T*                    ValueType;           //!< Type of the underlying values.
   typedef T*                    PointerType;         //!< Pointer to a non-const object.
   typedef const T*              ConstPointerType;    //!< Pointer to a const object.
   typedef T*&                   ReferenceType;       //!< Reference to a non-const object.
   typedef T*const&              ConstReferenceType;  //!< Reference to a const object.
   typedef size_t                SizeType;            //!< Size type of the pointer vector.
   typedef PtrIterator<T>        Iterator;            //!< Iterator over non-const objects.
   typedef PtrIterator<const T>  ConstIterator;       //!< Iterator over const objects.
   typedef D                     DeletionPolicy;      //!< Type of the deletion policy.
   typedef G                     GrowthPolicy;        //!< Type of the growth policy.

   // STL iterator requirements
   typedef ValueType             value_type;          //!< Type of the underlying values.
   typedef PointerType           pointer;             //!< Pointer to a non-const object.
   typedef ConstPointerType      const_pointer;       //!< Pointer to a const object.
   typedef ReferenceType         reference;           //!< Reference to a non-const object.
   typedef ConstReferenceType    const_reference;     //!< Reference to a const object.
   typedef SizeType              size_type;           //!< Size type of the pointer vector.
   //**********************************************************************************************

   //**Forward declarations for nested classes*****************************************************
   template< typename C > class CastIterator;
   template< typename C > class ConstCastIterator;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline PtrVector( SizeType initCapacity = 0 );
            inline PtrVector( const PtrVector& pv );

   template< typename T2, typename D2, typename G2 >
            inline PtrVector( const PtrVector<T2,D2,G2>& pv );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~PtrVector();
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   PtrVector& operator=( const PtrVector& pv );

   template< typename T2, typename D2, typename G2 >
   PtrVector& operator=( const PtrVector<T2,D2,G2>& pv );
   //@}
   //**********************************************************************************************

   //**Get functions*******************************************************************************
   /*!\name Get functions */
   //@{
                          inline SizeType maxSize()  const;
                          inline SizeType size()     const;
   template< typename C > inline SizeType size()     const;
                          inline SizeType capacity() const;
                          inline bool     isEmpty()  const;
   //@}
   //**********************************************************************************************

   //**Access functions****************************************************************************
   /*!\name Access functions */
   //@{
   inline ReferenceType      operator[]( SizeType index );
   inline ConstReferenceType operator[]( SizeType index ) const;
   inline ReferenceType      front();
   inline ConstReferenceType front() const;
   inline ReferenceType      back();
   inline ConstReferenceType back()  const;
   //@}
   //**********************************************************************************************

   //**Iterator functions**************************************************************************
   /*!\name Iterator functions */
   //@{
                          inline Iterator             begin();
                          inline ConstIterator        begin() const;
   template< typename C > inline CastIterator<C>      begin();
   template< typename C > inline ConstCastIterator<C> begin() const;

                          inline Iterator             end();
                          inline ConstIterator        end()   const;
   template< typename C > inline CastIterator<C>      end();
   template< typename C > inline ConstCastIterator<C> end()   const;
   //@}
   //**********************************************************************************************

   //**Element functions***************************************************************************
   /*!\name Element functions */
   //@{
   inline void     pushBack   ( PointerType p );
   inline void     popBack    ();
   inline void     releaseBack();

   template< typename IteratorType >
   inline void     assign( IteratorType first, IteratorType last );

   inline Iterator insert( Iterator pos, PointerType p );

   template< typename IteratorType >
   inline void     insert( Iterator pos, IteratorType first, IteratorType last );

   /*! \cond BLAZE_INTERNAL */
   template< typename IteratorType >
   inline void     insert( Iterator pos, IteratorType* first, IteratorType* last );
   /*! \endcond */

                          inline Iterator        erase  ( Iterator pos );
   template< typename C > inline CastIterator<C> erase  ( CastIterator<C> pos );
                          inline Iterator        release( Iterator pos );
   template< typename C > inline CastIterator<C> release( CastIterator<C> pos );
                          inline void            clear  ();
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
          void reserve( SizeType newCapacity );
   inline void swap( PtrVector& pv ) noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Helper functions****************************************************************************
   /*!\name Helper functions */
   //@{
   inline size_t calcCapacity ( size_t minCapacity ) const;
   inline void   deleteElement( PointerType ptr )    const;
   //@}
   //**********************************************************************************************

   //**Insertion helper functions******************************************************************
   /*!\name Insertion helper functions */
   //@{
          void insert( T**const pos, PointerType p );

   /*! \cond BLAZE_INTERNAL */
   template< typename IteratorType >
   inline void insert( Iterator pos, IteratorType first, IteratorType last, std::input_iterator_tag );

   template< typename IteratorType >
   inline void insert( Iterator pos, IteratorType first, IteratorType last, std::random_access_iterator_tag );
   /*! \endcond */

   template< typename IteratorType >
          void insert( T** pos, IteratorType first, IteratorType last, SizeType n );
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   SizeType size_;       //!< The current size of the pointer vector.
   SizeType capacity_;   //!< The capacity of the pointer vector.
   PointerType* begin_;  //!< Pointer to the first element of the pointer vector.
   PointerType* end_;    //!< Pointer to the last element of the pointer vector.
   //@}
   //**********************************************************************************************

 public:
   //**CastIterator/ConstCastIterator comparison operators*****************************************
   // The following comparison operators cannot be defined as namespace or member functions
   // but have to be injected into the surrounding scope via the Barton-Nackman trick since
   // the template arguments of nested templates cannot be deduced (C++ standard 14.8.2.4/4).
   /*!\name CastIterator/ConstCastIterator comparison operators */
   //@{

   //**********************************************************************************************
   /*!\brief Equality comparison between two CastIterator objects.
   //
   // \param lhs The left hand side cast iterator.
   // \param rhs The right hand side cast iterator.
   // \return \a true if the iterators point to the same element, \a false if not.
   */
   template< typename L, typename R >
   friend inline bool operator==( const CastIterator<L>& lhs, const CastIterator<R>& rhs )
   {
      return lhs.base() == rhs.base();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Equality comparison between a CastIterator and a ConstCastIterator.
   //
   // \param lhs The left hand side cast iterator.
   // \param rhs The right hand side constant cast iterator.
   // \return \a true if the iterators point to the same element, \a false if not.
   */
   template< typename L, typename R >
   friend inline bool operator==( const CastIterator<L>& lhs, const ConstCastIterator<R>& rhs )
   {
      return lhs.base() == rhs.base();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Equality comparison between a ConstCastIterator and a CastIterator.
   //
   // \param lhs The left hand side constant cast iterator.
   // \param rhs The right hand side cast iterator.
   // \return \a true if the iterators point to the same element, \a false if not.
   */
   template< typename L, typename R >
   friend inline bool operator==( const ConstCastIterator<L>& lhs, const CastIterator<R>& rhs )
   {
      return lhs.base() == rhs.base();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Equality comparison between two ConstCastIterator objects.
   //
   // \param lhs The left hand side constant cast iterator.
   // \param rhs The right hand side constant cast iterator.
   // \return \a true if the iterators point to the same element, \a false if not.
   */
   template< typename L, typename R >
   friend inline bool operator==( const ConstCastIterator<L>& lhs, const ConstCastIterator<R>& rhs )
   {
      return lhs.base() == rhs.base();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Inequality comparison between two CastIterator objects.
   //
   // \param lhs The left hand side cast iterator.
   // \param rhs The right hand side cast iterator.
   // \return \a true if the iterators don't point to the same element, \a false if they do.
   */
   template< typename L, typename R >
   friend inline bool operator!=( const CastIterator<L>& lhs, const CastIterator<R>& rhs )
   {
      return lhs.base() != rhs.base();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Inequality comparison between a CastIterator and a ConstCastIterator.
   //
   // \param lhs The left hand side cast iterator.
   // \param rhs The right hand side constant cast iterator.
   // \return \a true if the iterators don't point to the same element, \a false if they do.
   */
   template< typename L, typename R >
   friend inline bool operator!=( const CastIterator<L>& lhs, const ConstCastIterator<R>& rhs )
   {
      return lhs.base() != rhs.base();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Inequality comparison between a ConstCastIterator and a CastIterator.
   //
   // \param lhs The left hand side constant cast iterator.
   // \param rhs The right hand side cast iterator.
   // \return \a true if the iterators don't point to the same element, \a false if they do.
   */
   template< typename L, typename R >
   friend inline bool operator!=( const ConstCastIterator<L>& lhs, const CastIterator<R>& rhs )
   {
      return lhs.base() != rhs.base();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Inequality comparison between two ConstCastIterator objects.
   //
   // \param lhs The left hand side constant cast iterator.
   // \param rhs The right hand side constant cast iterator.
   // \return \a true if the iterators don't point to the same element, \a false if they do.
   */
   template< typename L, typename R >
   friend inline bool operator!=( const ConstCastIterator<L>& lhs, const ConstCastIterator<R>& rhs )
   {
      return lhs.base() != rhs.base();
   }
   //**********************************************************************************************

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
/*!\brief Standard constructor for PtrVector.
//
// \param initCapacity The initial capacity of the pointer vector.
//
// The default initial capacity of the pointer vector is specified by the selected growth policy.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline PtrVector<T,D,G>::PtrVector( SizeType initCapacity )
   : size_( 0 )                               // Current size of the pointer vector
   , capacity_( initCapacity )                // Capacity of the pointer vector
   , begin_( new PointerType[initCapacity] )  // Pointer to the first element
   , end_( begin_ )                           // Pointer to the last element
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Copy constructor for PtrVector.
//
// \param pv The pointer vector to be copied.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline PtrVector<T,D,G>::PtrVector( const PtrVector& pv )
   : size_( pv.size_ )                     // Current size of the pointer vector
   , capacity_( pv.size_ )                 // Capacity of the pointer vector
   , begin_( new PointerType[capacity_] )  // Pointer to the first element
   , end_( begin_+size_ )                  // Pointer to the last element
{
   for( SizeType i=0; i<size_; ++i )
      begin_[i] = pv.begin_[i];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different PtrVector instances.
//
// \param pv The pointer vector to be copied.
*/
template< typename T     // Type of the pointer vector
        , typename D     // Deletion policy of the pointer vector
        , typename G >   // Growth policy of the pointer vector
template< typename T2    // Type of the foreign pointer vector
        , typename D2    // Deletion policy of the foreign pointer vector
        , typename G2 >  // Growth policy of the foreign pointer vector
inline PtrVector<T,D,G>::PtrVector( const PtrVector<T2,D2,G2>& pv )
   : size_( pv.size_ )                     // Current size of the pointer vector
   , capacity_( pv.size_ )                 // Capacity of the pointer vector
   , begin_( new PointerType[capacity_] )  // Pointer to the first element
   , end_( begin_+size_ )                  // Pointer to the last element
{
   for( SizeType i=0; i<size_; ++i )
      begin_[i] = pv.begin_[i];
}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor for PtrVector.
//
// In the destructor, the selected deletion policy is applied to all elements of the pointer
// vector.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline PtrVector<T,D,G>::~PtrVector()
{
   for( PointerType* it=begin_; it!=end_; ++it )
      deleteElement( *it );
   delete [] begin_;
}
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Copy assignment operator for PtrVector.
//
// \param pv The pointer vector to be copied.
// \return Reference to the assigned pointer vector.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
PtrVector<T,D,G>& PtrVector<T,D,G>::operator=( const PtrVector& pv )
{
   if( &pv == this ) return *this;

   if( pv.size_ > capacity_ ) {
      PointerType* newBegin( new PointerType[pv.size_] );
      end_ = std::copy( pv.begin_, pv.end_, newBegin );
      std::swap( begin_, newBegin );
      delete [] newBegin;

      size_ = pv.size_;
      capacity_ = pv.size_;
   }
   else {
      end_ = std::copy( pv.begin_, pv.end_, begin_ );
      size_ = pv.size_;
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different PtrVector instances.
//
// \param pv The pointer vector to be copied.
// \return Reference to the assigned pointer vector.
*/
template< typename T     // Type of the pointer vector
        , typename D     // Deletion policy of the pointer vector
        , typename G >   // Growth policy of the pointer vector
template< typename T2    // Type of the foreign pointer vector
        , typename D2    // Deletion policy of the foreign pointer vector
        , typename G2 >  // Growth policy of the foreign pointer vector
PtrVector<T,D,G>& PtrVector<T,D,G>::operator=( const PtrVector<T2,D2,G2>& pv )
{
   if( pv.size_ > capacity_ ) {
      PointerType* newBegin( new PointerType[pv.size_] );
      end_ = std::copy( pv.begin_, pv.end_, newBegin );
      std::swap( begin_, newBegin );
      delete [] newBegin;

      size_ = pv.size_;
      capacity_ = pv.size_;
   }
   else {
      end_ = std::copy( pv.begin_, pv.end_, begin_ );
      size_ = pv.size_;
   }

   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  GET FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the maximum possible size of a pointer vector.
//
// \return The maximum possible size.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::SizeType PtrVector<T,D,G>::maxSize() const
{
   return SizeType(-1) / sizeof(PointerType);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current size of the pointer vector.
//
// \return The current size.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::SizeType PtrVector<T,D,G>::size() const
{
   return size_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of objects of type \a C contained in the pointer vector.
//
// \return The total number of objects of type \a C.
//
// This function calculates the total number of objects of type \a C within the pointer vector,
// where \a C is a type derived from the type \a T of objects contained in the pointer vector.
// The attempt to use this function for types that are not derived from \a T results in a
// compile time error.

   \code
   // Definition of class A and the derived type B
   class A { ... };
   class B : public A { ... };

   // Definition of a pointer vector for class A
   typedef blaze::PtrVector<A> AVector;
   AVector vector;

   AVector::SizeType total = vector.size();     // Calculating the total number of pointers
   AVector::SizeType numB  = vector.size<B>();  // Calculating the total number of B objects
   \endcode

// \note The total number of objects of type \a C is not cached inside the pointer vector
// but is calculated each time the function is called. Using the templated version of size()
// to calculate the total number objects of type \a C is therefore more expensive than using
// the non-template version of size() to get the total number of pointers in the vector!
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::SizeType PtrVector<T,D,G>::size() const
{
   // The polymorphicCount() function returns the number of objects with dynamic type 'C'
   // contained in the range [begin,end). An equivalent code might look like this:
   //
   // SizeType count( 0 );
   // for( PointerType* it=begin_; it!=end_; ++it )
   //    if( dynamic_cast<C*>( *it ) ) ++count;
   // return count;
   //
   // However, the specialization of polymorphicCount() for special type combinations is
   // much more efficient (and easier) than the specialization of this function!
   return polymorphicCount<C>( begin_, end_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the capacity of the pointer vector.
//
// \return The capacity.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::SizeType PtrVector<T,D,G>::capacity() const
{
   return capacity_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns \a true if the pointer vector has no elements.
//
// \return \a true if the pointer vector is empty, \a false if it is not.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline bool PtrVector<T,D,G>::isEmpty() const
{
   return size_ == 0;
}
//*************************************************************************************************




//=================================================================================================
//
//  ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Subscript operator for the direct access to the pointer vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..size-1]\f$.
// \return Handle to the accessed element.
//
// \note No runtime check is performed to insure the validity of the access index.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::ReferenceType PtrVector<T,D,G>::operator[]( SizeType index )
{
   return *(begin_+index);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subscript operator for the direct access to the pointer vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..size-1]\f$.
// \return Handle to the accessed element.
//
// \note No runtime check is performed to insure the validity of the access index.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::ConstReferenceType PtrVector<T,D,G>::operator[]( SizeType index ) const
{
   return *(begin_+index);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a reference to the first element of the pointer vector.
//
// \return Handle to the first element.
//
// \note No runtime check is performed if the first element exists!
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::ReferenceType PtrVector<T,D,G>::front()
{
   BLAZE_USER_ASSERT( size_ > 0, "Pointer vector is empty, invalid access to the front element" );
   return *begin_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a reference to the first element of the pointer vector.
//
// \return Handle to the first element.
//
// \note No runtime check is performed if the first element exists!
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::ConstReferenceType PtrVector<T,D,G>::front() const
{
   BLAZE_USER_ASSERT( size_ > 0, "Pointer vector is empty, invalid access to the front element" );
   return *begin_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a reference to the last element of the pointer vector.
//
// \return Handle to the last element.
//
// \note No runtime check is performed if the last element exists!
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::ReferenceType PtrVector<T,D,G>::back()
{
   BLAZE_USER_ASSERT( size_ > 0, "Pointer vector is empty, invalid access to the back element" );
   return *(end_-1);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a reference to the last element of the pointer vector.
//
// \return Handle to the last element.
//
// \note No runtime check is performed if the last element exists!
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::ConstReferenceType PtrVector<T,D,G>::back() const
{
   BLAZE_USER_ASSERT( size_ > 0, "Pointer vector is empty, invalid access to the back element" );
   return *(end_-1);
}
//*************************************************************************************************




//=================================================================================================
//
//  ITERATOR FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns an iterator to the beginning of the pointer vector.
//
// \return Iterator to the beginning of the pointer vector.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::Iterator PtrVector<T,D,G>::begin()
{
   return Iterator( begin_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the beginning of the pointer vector.
//
// \return Iterator to the beginning of the pointer vector.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::ConstIterator PtrVector<T,D,G>::begin() const
{
   return ConstIterator( begin_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of type \a C within the pointer vector.
//
// \return Iterator to the first element of type \a C.
//
// This function returns an iterator to the first element of type \a C within in the pointer
// vector, where \a C is a type derived from the type \a T of objects contained in the pointer
// vector. In case there is no element of type \a C contained in the vector, an iterator just
// past the last element of the pointer vector is returned. In combination with the according
// end function (see example), this iterator allows to iterate over all objects of type \a C
// in the range of the pointer vector. The attempt to use this function for types that are not
// derived from \a T results in a compile time error.

   \code
   // Definition of class A and the derived type B
   class A { ... };
   class B : public A { ... };

   // Definition of function f for non-const pointer vectors
   void f( blaze::PtrVector<A>& vector )
   {
      blaze::PtrVector<A>::CastIterator<B> begin = vector.begin<B>();
      blaze::PtrVector<A>::CastIterator<B> end   = vector.end<B>();

      // Loop over all objects of type B contained in the vector
      for( ; begin!=end; ++begin )
         ...
   }

   // Definition of function f for const pointer vectors
   void f( const blaze::PtrVector<A>& vector )
   {
      blaze::PtrVector<A>::ConstCastIterator<B> begin = vector.begin<B>();
      blaze::PtrVector<A>::ConstCastIterator<B> end   = vector.end<B>();

      // Loop over all objects of type B contained in the vector
      for( ; begin!=end; ++begin )
   }
   \endcode

// \note Using the templated versions of begin() and end() to traverse all elements of type
// \a C in the element range of the pointer vector is more expensive than using the non-template
// versions to traverse the entire range of elements. Use this function only if you require a
// type-specific member of type \a C.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE CastIterator<C> PtrVector<T,D,G>::begin()
{
   return CastIterator<C>( begin_, end_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of type \a C within the pointer vector.
//
// \return Iterator to the first element of type \a C.
//
// This function returns an iterator to the first element of type \a C within in the pointer
// vector, where \a C is a type derived from the type \a T of objects contained in the pointer
// vector. In case there is no element of type \a C contained in the vector, an iterator just
// past the last element of the pointer vector is returned. In combination with the according
// end function (see example), this iterator allows to iterate over all objects of type \a C
// in the range of the pointer vector. The attempt to use this function for types that are not
// derived from \a T results in a compile time error.

   \code
   // Definition of class A and the derived type B
   class A { ... };
   class B : public A { ... };

   // Definition of function f for non-const pointer vectors
   void f( blaze::PtrVector<A>& vector )
   {
      blaze::PtrVector<A>::CastIterator<B> begin = vector.begin<B>();
      blaze::PtrVector<A>::CastIterator<B> end   = vector.end<B>();

      // Loop over all objects of type B contained in the vector
      for( ; begin!=end; ++begin )
         ...
   }

   // Definition of function f for const pointer vectors
   void f( const blaze::PtrVector<A>& vector )
   {
      blaze::PtrVector<A>::ConstCastIterator<B> begin = vector.begin<B>();
      blaze::PtrVector<A>::ConstCastIterator<B> end   = vector.end<B>();

      // Loop over all objects of type B contained in the vector
      for( ; begin!=end; ++begin )
   }
   \endcode

// \note Using the templated versions of begin() and end() to traverse all elements of type
// \a C in the element range of the pointer vector is more expensive than using the non-template
// version to traverse the entire range of elements. Use this function only if you require a
// type-specific member of type \a C.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE ConstCastIterator<C> PtrVector<T,D,G>::begin() const
{
   return ConstCastIterator<C>( begin_, end_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the pointer vector.
//
// \return Iterator just past the last element of the pointer vector.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::Iterator PtrVector<T,D,G>::end()
{
   return Iterator( end_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the pointer vector.
//
// \return Iterator just past the last element of the pointer vector.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::ConstIterator PtrVector<T,D,G>::end() const
{
   return ConstIterator( end_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the pointer vector.
//
// \return Iterator just past the last element of the pointer vector.
//
// This function returns an iterator just past the last element of the pointer vector. In
// combination with the according begin function (see example), this iterator allows to iterate
// over all objects of type \a C in the range of the pointer vector. The attempt to use this
// function for types that are not derived from \a T results in a compile time error.

   \code
   // Definition of class A and the derived type B
   class A { ... };
   class B : public A { ... };

   // Definition of function f for non-const pointer vectors
   void f( blaze::PtrVector<A>& vector )
   {
      blaze::PtrVector<A>::CastIterator<B> begin = vector.begin<B>();
      blaze::PtrVector<A>::CastIterator<B> end   = vector.end<B>();

      // Loop over all objects of type B contained in the vector
      for( ; begin!=end; ++begin )
         ...
   }

   // Definition of function f for const pointer vectors
   void f( const blaze::PtrVector<A>& vector )
   {
      blaze::PtrVector<A>::ConstCastIterator<B> begin = vector.begin<B>();
      blaze::PtrVector<A>::ConstCastIterator<B> end   = vector.end<B>();

      // Loop over all objects of type B contained in the vector
      for( ; begin!=end; ++begin )
   }
   \endcode

// \note Using the templated versions of begin() and end() to traverse all elements of type
// \a C in the element range of the pointer vector is more expensive than using the non-template
// versions to traverse the entire range of elements. Use this function only if you require a
// type-specific member of type \a C.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE CastIterator<C> PtrVector<T,D,G>::end()
{
   return CastIterator<C>( end_, end_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the pointer vector.
//
// \return Iterator just past the last element of the pointer vector.
//
// This function returns an iterator just past the last element of the pointer vector. In
// combination with the according begin function (see example), this iterator allows to iterate
// over all objects of type \a C in the range of the pointer vector. The attempt to use this
// function for types that are not derived from \a T results in a compile time error.

   \code
   // Definition of class A and the derived type B
   class A { ... };
   class B : public A { ... };

   // Definition of function f for non-const pointer vectors
   void f( blaze::PtrVector<A>& vector )
   {
      blaze::PtrVector<A>::CastIterator<B> begin = vector.begin<B>();
      blaze::PtrVector<A>::CastIterator<B> end   = vector.end<B>();

      // Loop over all objects of type B contained in the vector
      for( ; begin!=end; ++begin )
         ...
   }

   // Definition of function f for const pointer vectors
   void f( const blaze::PtrVector<A>& vector )
   {
      blaze::PtrVector<A>::ConstCastIterator<B> begin = vector.begin<B>();
      blaze::PtrVector<A>::ConstCastIterator<B> end   = vector.end<B>();

      // Loop over all objects of type B contained in the vector
      for( ; begin!=end; ++begin )
   }
   \endcode

// \note Using the templated versions of begin() and end() to traverse all elements of type
// \a C in the element range of the pointer vector is more expensive than using the non-template
// version to traverse the entire range of elements. Use this function only if you require a
// type-specific member of type \a C.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE ConstCastIterator<C> PtrVector<T,D,G>::end() const
{
   return ConstCastIterator<C>( end_, end_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  ELEMENT FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Adding an element to the end of the pointer vector.
//
// \param p The pointer to be added to the end of the pointer vector.
// \return void
// \exception std::length_error Maximum pointer vector length exceeded.
//
// The pushBack function runs in constant time.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline void PtrVector<T,D,G>::pushBack( PointerType p )
{
   if( size_ != capacity_ ) {
      *end_ = p;
      ++end_;
      ++size_;
   }
   else {
      insert( end_, p );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Removing an element from the end of the pointer vector.
//
// \return void
//
// This function removes the element at the end of the pointer vector, i.e. the element
// is deleted according to the deletion policy and removed from the vector. Note that in
// case the deletion policy is NoDelete, this function is identical to the releaseBack()
// function.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline void PtrVector<T,D,G>::popBack()
{
   deleteElement( *--end_ );
   --size_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Releasing the element at the end of the pointer vector.
//
// \return void
//
// This function releases the element at the end of the pointer vector, i.e. the element is
// removed without applying the deletion policy. Therefore the responsibility to delete the
// element is passed to the function caller. Note that in case the deletion policy is NoDelete,
// this function is identical to the popBack() function.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline void PtrVector<T,D,G>::releaseBack()
{
   --end_;
   --size_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assigning a range of elements to the pointer vector.
//
// \param first Iterator to the first element of the element range.
// \param last Iterator to the element one past the last element of the element range.
// \return void
// \exception std::length_error Maximum pointer vector length exceeded.
//
// This functions assigns the elements in the range \f$ [first,last) \f$ to the pointer vector.
// All elements previously contained in the pointer vector are removed. The assign function runs
// in linear time.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename IteratorType >
inline void PtrVector<T,D,G>::assign( IteratorType first, IteratorType last )
{
   clear();
   insert( end(), first, last );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inserting an element into the pointer vector.
//
// \param pos The position before which the element is inserted.
// \param p The pointer to be inserted into the pointer vector.
// \return Iterator to the inserted element.
// \exception std::length_error Maximum pointer vector length exceeded.
//
// The insert function runs in linear time. Note however that inserting elements into a pointer
// vector can be a relatively time-intensive operation.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::Iterator PtrVector<T,D,G>::insert( Iterator pos, PointerType p )
{
   T** const base = const_cast<T**>( pos.base() );
   const SizeType diff( base - begin_ );

   if( size_ != capacity_ && base == end_ ) {
      *end_ = p;
      ++end_;
      ++size_;
   }
   else {
      insert( base, p );
   }

   return Iterator( begin_+diff );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inserting a range of elements into the pointer vector.
//
// \param pos The position before which the elements are inserted.
// \param first Iterator to the first element of the element range.
// \param last Iterator to the element one past the last element of the element range.
// \return void
// \exception std::length_error Maximum pointer vector length exceeded.
//
// This functions inserts the elements in the range \f$ [first,last) \f$ into the pointer vector.
// The insert function runs in linear time. Note however that inserting elements into a pointer
// vector can be a relatively time-intensive operation.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename IteratorType >
inline void PtrVector<T,D,G>::insert( Iterator pos, IteratorType first, IteratorType last )
{
   insert( pos, first, last, typename IteratorType::iterator_category() );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Inserting a range of elements into the pointer vector.
//
// \param pos The position before which the elements are inserted.
// \param first Pointer to the first element of the element range.
// \param last Pointer to the element one past the last element of the element range.
// \return void
// \exception std::length_error Maximum pointer vector length exceeded.
//
// This functions inserts the elements in the range \f$ [first,last) \f$ into the pointer vector.
// The insert function runs in linear time. Note however that inserting elements into a pointer
// vector can be a relatively time-intensive operation.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename IteratorType >
inline void PtrVector<T,D,G>::insert( Iterator pos, IteratorType* first, IteratorType* last )
{
   insert( pos, first, last, std::random_access_iterator_tag() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Removing an element from the pointer vector.
//
// \param pos The position of the element to be removed.
// \return Iterator to the element after the erased element.
//
// This function erases an element from the pointer vector, i.e. the element is deleted
// according to the deletion policy of the pointer vector and removed from the vector.
// Note that in case the deletion policy is NoDelete, this function is identical to the
// release() function.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::Iterator PtrVector<T,D,G>::erase( Iterator pos )
{
   T** const base = const_cast<T**>( pos.base() );
   deleteElement( *base );
   std::copy( base+1, end_, base );

   --size_;
   --end_;

   return pos;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Removing an element from the pointer vector.
//
// \param pos The position of the element to be removed.
// \return Iterator to the element after the erased element.
//
// This function erases an element from the pointer vector, i.e. the element is deleted
// according to the deletion policy of the pointer vector and removed from the vector.
// Note that in case the deletion policy is NoDelete, this function is identical to the
// release() function.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE CastIterator<C>
   PtrVector<T,D,G>::erase( CastIterator<C> pos )
{
   T** const base = const_cast<T**>( pos.base() );
   deleteElement( *base );
   std::copy( base+1, end_, base );

   --size_;
   --end_;

   return CastIterator<C>( base, end_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Releasing an element from the pointer vector.
//
// \param pos The position of the element to be released.
// \return Iterator to the element after the released element.
//
// This function releases an element from the pointer vector, i.e. the element is removed
// without applying the deletion policy. Therefore the responsibility to delete the element
// is passed to the function caller.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline typename PtrVector<T,D,G>::Iterator PtrVector<T,D,G>::release( Iterator pos )
{
   T** const base = const_cast<T**>( pos.base() );
   std::copy( base+1, end_, base );

   --size_;
   --end_;

   return pos;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Releasing an element from the pointer vector.
//
// \param pos The position of the element to be released.
// \return Iterator to the element after the released element.
//
// This function releases an element from the pointer vector, i.e. the element is removed
// without applying the deletion policy. Therefore the responsibility to delete the element
// is passed to the function caller.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE CastIterator<C>
   PtrVector<T,D,G>::release( CastIterator<C> pos )
{
   T** const base = const_cast<T**>( pos.base() );
   std::copy( base+1, end_, base );

   --size_;
   --end_;

   return CastIterator<C>( base, end_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Removing all elements from the pointer vector.
//
// \return void
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline void PtrVector<T,D,G>::clear()
{
   for( PointerType* it=begin_; it!=end_; ++it )
      deleteElement( *it );

   end_  = begin_;
   size_ = 0;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Setting the minimum capacity of the pointer vector.
//
// \param newCapacity The new minimum capacity of the pointer vector.
// \return void
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
void PtrVector<T,D,G>::reserve( SizeType newCapacity )
{
   if( newCapacity > capacity_ )
   {
      // Calculating the new capacity
      newCapacity = calcCapacity( newCapacity );

      // Allocating a new array
      PointerType* tmp = new PointerType[newCapacity];

      // Replacing the old array
      std::copy( begin_, end_, tmp );
      std::swap( tmp, begin_ );
      capacity_ = newCapacity;
      end_ = begin_ + size_;
      delete [] tmp;
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two pointer vectors.
//
// \param pv The pointer vector to be swapped.
// \return void
// \exception no-throw guarantee.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline void PtrVector<T,D,G>::swap( PtrVector& pv ) noexcept
{
   // By using the 'std::swap' function to swap all member variables,
   // the function can give the nothrow guarantee.
   std::swap( size_, pv.size_ );
   std::swap( capacity_, pv.capacity_ );
   std::swap( begin_, pv.begin_ );
   std::swap( end_, pv.end_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  HELPER FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Calculating the new capacity of the vector based on its growth policy.
//
// \param minCapacity The minimum necessary capacity.
// \return The new capacity.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline size_t PtrVector<T,D,G>::calcCapacity( size_t minCapacity ) const
{
   BLAZE_INTERNAL_ASSERT( minCapacity > capacity_, "Invalid new vector capacity" );
   const size_t newCapacity( GrowthPolicy()( capacity_, minCapacity ) );
   BLAZE_INTERNAL_ASSERT( newCapacity > capacity_, "Invalid new vector capacity" );
   return newCapacity;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deleting an element of the pointer vector according to the deletion policy.
//
// \param ptr The element to be deleted.
// \return void
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline void PtrVector<T,D,G>::deleteElement( PointerType ptr ) const
{
   DeletionPolicy()( ptr );
}
//*************************************************************************************************




//=================================================================================================
//
//  INSERTION HELPER FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Inserting an element into the pointer vector.
//
// \param pos The position before which the element is inserted.
// \param p The pointer to be inserted into the pointer vector.
// \return void
// \exception std::length_error Maximum pointer vector length exceeded.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
void PtrVector<T,D,G>::insert( T**const pos, PointerType p )
{
   if( size_ != capacity_ ) {
      std::copy_backward( pos, end_, end_+1 );
      *pos = p;
      ++end_;
      ++size_;
   }
   else if( size_ == maxSize() ) {
      BLAZE_THROW_LENGTH_ERROR( "Maximum pointer vector length exceeded!" );
   }
   else {
      SizeType newCapacity( calcCapacity( capacity_+1 ) );
      if( newCapacity > maxSize() || newCapacity < capacity_ ) newCapacity = maxSize();

      PointerType* newBegin = new PointerType[newCapacity];
      PointerType* newEnd = std::copy( begin_, pos, newBegin );
      *newEnd = p;
      ++newEnd;
      end_ = std::copy( pos, end_, newEnd );

      std::swap( newBegin, begin_ );
      delete [] newBegin;
      capacity_ = newCapacity;
      ++size_;
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Inserting a range of elements into the pointer vector.
//
// \param pos The position before which the elements are inserted.
// \param first Iterator to the first element of the element range.
// \param last Iterator to the element one past the last element of the element range.
// \return void
// \exception std::length_error Maximum pointer vector length exceeded.
//
// This functions inserts the elements in the range \f$ [first,last) \f$ into the pointer vector.
// The iterators are treated as input iterators.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename IteratorType >
inline void PtrVector<T,D,G>::insert( Iterator pos, IteratorType first, IteratorType last,
                                      std::input_iterator_tag )
{
   for( ; first!=last; ++first ) {
      pos = insert( pos, *first );
      ++pos;
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Inserting a range of elements into the pointer vector.
//
// \param pos The position before which the elements are inserted.
// \param first Iterator to the first element of the element range.
// \param last Iterator to the element one past the last element of the element range.
// \return void
// \exception std::length_error Maximum pointer vector length exceeded.
//
// This functions inserts the elements in the range \f$ [first,last) \f$ into the pointer vector.
// The iterators are treated as random access iterators.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename IteratorType >
inline void PtrVector<T,D,G>::insert( Iterator pos, IteratorType first, IteratorType last,
                                      std::random_access_iterator_tag )
{
   T** const base = const_cast<T**>( pos.base() );
   const SizeType diff( last - first );

   if( size_+diff <= capacity_ && base == end_ ) {
      for( ; first!=last; ++first, ++end_ ) {
         *end_ = *first;
      }
      size_ += diff;
   }
   else {
      insert( base, first, last, diff );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inserting a range of elements into the pointer vector.
//
// \param pos The position before which the elements are inserted.
// \param first Iterator to the first element of the element range.
// \param last Iterator to the element one past the last element of the element range.
// \param n The number of elements to be inserted.
// \return void
// \exception std::length_error Maximum pointer vector length exceeded.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename IteratorType >
void PtrVector<T,D,G>::insert( T** pos, IteratorType first, IteratorType last, SizeType n )
{
   const SizeType newSize( size_ + n );

   if( newSize <= capacity_ ) {
      std::copy_backward( pos, end_, end_+n );
      for( ; first!=last; ++first, ++pos ) {
         *pos = *first;
      }
      end_ += n;
      size_ = newSize;
   }
   else if( newSize > maxSize() || newSize < size_ ) {
      BLAZE_THROW_LENGTH_ERROR( "Maximum pointer vector length exceeded!" );
   }
   else {
      PointerType* newBegin = new PointerType[newSize];
      PointerType* newEnd = std::copy( begin_, pos, newBegin );

      for( ; first!=last; ++first, ++newEnd ) {
         *newEnd = *first;
      }

      end_ = std::copy( pos, end_, newEnd );

      std::swap( newBegin, begin_ );
      delete [] newBegin;
      capacity_ = newSize;
      size_ = newSize;
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name PtrVector operators */
//@{
template< typename T, typename D, typename G >
inline bool operator==( const PtrVector<T,D,G>& lhs, const PtrVector<T,D,G>& rhs );

template< typename T, typename D, typename G >
inline bool operator!=( const PtrVector<T,D,G>& lhs, const PtrVector<T,D,G>& rhs );

template< typename T, typename D, typename G >
inline void swap( PtrVector<T,D,G>& a, PtrVector<T,D,G>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between two pointer vectors.
//
// \param lhs The left hand side pointer vector.
// \param rhs The right hand side pointer vector.
// \return \a true if the two pointer vectors are equal, \a false if they are not.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline bool operator==( const PtrVector<T,D,G>& lhs, const PtrVector<T,D,G>& rhs )
{
   return lhs.size() == rhs.size() && std::equal( lhs.begin(), lhs.end(), rhs.begin() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between two pointer vectors.
//
// \param lhs The left hand side pointer vector.
// \param rhs The right hand side pointer vector.
// \return \a true if the two pointer vectors are inequal, \a false if they are not.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline bool operator!=( const PtrVector<T,D,G>& lhs, const PtrVector<T,D,G>& rhs )
{
   return lhs.size() != rhs.size() || !std::equal( lhs.begin(), lhs.end(), rhs.begin() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two pointer vectors.
//
// \param a The first pointer vector to be swapped.
// \param b The second pointer vector to be swapped.
// \return void
// \exception no-throw guarantee.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
inline void swap( PtrVector<T,D,G>& a, PtrVector<T,D,G>& b ) noexcept
{
   a.swap( b );
}
//*************************************************************************************************








//=================================================================================================
//
//  NESTED CLASS PTRVECTOR::CASTITERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Dynamic cast iterator for polymorphic pointer vectors.
// \ingroup util
//
// The CastIterator class is part of the PtrVector class and represent a forward iterator
// over all elements of type \a C contained in a range of elements of type \a T, where \a C
// is a type derived from \a T.

   \code
   class A { ... };
   class B : public class A { ... };

   blaze::PtrVector<A>::CastIterator<B> begin;
   blaze::PtrVector<A>::CastIterator<B> end;

   // Loop over all elements of type B within the range [begin..end)
   for( ; begin!=end; ++begin )
      ...
   \endcode

// \note Using a CastIterator is computationally more expensive than using a standard iterator
// over all elements contained in the vector.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
class PtrVector<T,D,G>::CastIterator
{
 public:
   //**Type definitions****************************************************************************
   // Blaze naming convention
   typedef std::forward_iterator_tag  IteratorCategory;   //!< The iterator category.
   typedef C*                         ValueType;          //!< Type of the underlying pointers.
   typedef C*                         PointerType;        //!< Pointer return type.
   typedef C* const&                  ReferenceType;      //!< Reference return type.
   typedef ptrdiff_t                  DifferenceType;     //!< Difference between two iterators.
   typedef T* const*                  IteratorType;       //!< Type of the internal pointer.

   // STL iterator requirements
   typedef IteratorCategory           iterator_category;  //!< The iterator category.
   typedef ValueType                  value_type;         //!< Type of the underlying pointers.
   typedef PointerType                pointer;            //!< Pointer return type.
   typedef ReferenceType              reference;          //!< Reference return type.
   typedef DifferenceType             difference_type;    //!< Difference between two iterators.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   inline CastIterator();
   inline CastIterator( IteratorType begin, IteratorType end );

   template< typename Other >
   inline CastIterator( const CastIterator<Other>& it );

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
   inline CastIterator& operator++();
   inline CastIterator  operator++( int );
   //@}
   //**********************************************************************************************

   //**Access operators****************************************************************************
   /*!\name Access operators */
   //@{
   inline PointerType operator*()  const;
   inline PointerType operator->() const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline const IteratorType& base() const;
   inline const IteratorType& stop() const;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   IteratorType cur_;  //!< Pointer to the current memory location.
   IteratorType end_;  //!< Pointer to the element one past the last element in the element range.
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Default constructor for CastIterator.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline PtrVector<T,D,G>::CastIterator<C>::CastIterator()
   : cur_( nullptr )  // Pointer to the current memory location
   , end_( nullptr )  // Pointer to the element one past the last element in the element range
{
   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_DERIVED_FROM( C, T );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Standard constructor for CastIterator.
//
// \param begin The beginning of the element range.
// \param end The end of the element range.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline PtrVector<T,D,G>::CastIterator<C>::CastIterator( IteratorType begin, IteratorType end )
   : cur_(begin)  // Pointer to the current memory location
   , end_(end)    // Pointer to the element one past the last element in the element range
{
   // The polymorphicFind() function finds the next pointer to an object with dynamic type 'C'
   // contained in the range [cur_,end). An equivalent code might look like this:
   //
   // while( cur_ != end_ && !dynamic_cast<C*>( *cur_ ) ) ++cur_;
   //
   // However, the specialization of polymorphicFind() for special type combinations is much
   // more efficient (and way easier!) than the specialization of this function!
   cur_ = polymorphicFind<C>( cur_, end_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different CastIterator instances.
//
// \param it The foreign CastIterator instance to be copied.
*/
template< typename T        // Type
        , typename D        // Deletion policy
        , typename G >      // Growth policy
template< typename C >      // Cast type
template< typename Other >  // The foreign cast iterator type
inline PtrVector<T,D,G>::CastIterator<C>::CastIterator( const CastIterator<Other>& it )
   : cur_( it.base() )  // Pointer to the current memory location
   , end_( it.stop() )  // Pointer to the element one past the last element in the element range
{
   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_DERIVED_FROM( C, T );
   BLAZE_CONSTRAINT_MUST_BE_CONVERTIBLE( Other*, C* );
}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Pre-increment operator.
//
// \return Reference to the incremented cast iterator.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE CastIterator<C>&
   PtrVector<T,D,G>::CastIterator<C>::operator++()
{
   // The polymorphicFind() function finds the next pointer to an object with dynamic type 'C'
   // contained in the range [cur_+1,end). An equivalent code might look like this:
   //
   // while( ++cur_ != end_ && !dynamic_cast<C*>( *cur_ ) ) {}
   //
   // However, the specialization of polymorphicFind() for special type combinations is much
   // more efficient (and way easier!) than the specialization of this function!
   cur_ = polymorphicFind<C>( ++cur_, end_ );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Post-increment operator.
//
// \return The incremented cast iterator.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE CastIterator<C>
   PtrVector<T,D,G>::CastIterator<C>::operator++( int )
{
   CastIterator tmp( *this );

   // The polymorphicFind() function finds the next pointer to an object with dynamic type 'C'
   // contained in the range [cur_+1,end). An equivalent code might look like this:
   //
   // while( ++cur_ != end_ && !dynamic_cast<C*>( *cur_ ) ) {}
   //
   // However, the specialization of polymorphicFind() for special type combinations is much
   // more efficient (and way easier!) than the specialization of this function!
   cur_ = polymorphicFind<C>( ++cur_, end_ );

   return tmp;
}
//*************************************************************************************************




//=================================================================================================
//
//  ACCESS OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns a handle to the element at the current iterator position.
//
// \return Handle to the element at the current iterator position.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE CastIterator<C>::PointerType
   PtrVector<T,D,G>::CastIterator<C>::operator*() const
{
   return static_cast<C*>( *cur_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Direct access to the element at the current iterator position.
//
// \return Reference to the element at the current iterator position.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE CastIterator<C>::PointerType
   PtrVector<T,D,G>::CastIterator<C>::operator->() const
{
   return static_cast<C*>( *cur_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Direct access to the current memory location of the cast iterator.
//
// \return Pointer to the current memory location.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline const typename PtrVector<T,D,G>::BLAZE_TEMPLATE CastIterator<C>::IteratorType&
   PtrVector<T,D,G>::CastIterator<C>::base() const
{
   return cur_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Direct access to the final memory location of the cast iterator.
//
// \return Pointer to the final memory location.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline const typename PtrVector<T,D,G>::BLAZE_TEMPLATE CastIterator<C>::IteratorType&
   PtrVector<T,D,G>::CastIterator<C>::stop() const
{
   return end_;
}
//*************************************************************************************************








//=================================================================================================
//
//  NESTED CLASS PTRVECTOR::CONSTCASTITERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Dynamic cast iterator for polymorphic pointer vectors.
// \ingroup util
//
// The ConstCastIterator class is part of the PtrVector class and represent a forward iterator
// over all elements of type \a C contained in a range of elements of type \a T, where \a C
// is a type derived from \a T. The ConstCastIterator is the counterpart of CastIterator for
// constant vectors.

   \code
   class A { ... };
   class B : public class A { ... };

   blaze::PtrVector<A>::ConstCastIterator<B> begin;
   blaze::PtrVector<A>::ConstCastIterator<B> end;

   // Loop over all elements of type B within the range [begin..end)
   for( ; begin!=end; ++begin )
      ...
   \endcode

// \note Using a ConstCastIterator is computationally more expensive than using a standard
// iterator over all elements contained in the vector.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
class PtrVector<T,D,G>::ConstCastIterator
{
 public:
   //**Type definitions****************************************************************************
   // Blaze naming convention
   typedef std::forward_iterator_tag  IteratorCategory;   //!< The iterator category.
   typedef const C*                   ValueType;          //!< Type of the underlying pointers.
   typedef const C*                   PointerType;        //!< Pointer return type.
   typedef const C* const&            ReferenceType;      //!< Reference return type.
   typedef ptrdiff_t                  DifferenceType;     //!< Difference between two iterators.
   typedef const T* const*            IteratorType;       //!< Type of the internal pointer.

   // STL iterator requirements
   typedef IteratorCategory           iterator_category;  //!< The iterator category.
   typedef ValueType                  value_type;         //!< Type of the underlying pointers.
   typedef PointerType                pointer;            //!< Pointer return type.
   typedef ReferenceType              reference;          //!< Reference return type.
   typedef DifferenceType             difference_type;    //!< Difference between two iterators.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   inline ConstCastIterator();
   inline ConstCastIterator( IteratorType begin, IteratorType end );

   template< typename Other >
   inline ConstCastIterator( const ConstCastIterator<Other>& it );

   template< typename Other >
   inline ConstCastIterator( const typename PtrVector<T,D,G>::BLAZE_TEMPLATE CastIterator<Other>& it );

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
   inline ConstCastIterator& operator++();
   inline ConstCastIterator  operator++( int );
   //@}
   //**********************************************************************************************

   //**Access operators****************************************************************************
   /*!\name Access operators */
   //@{
   inline PointerType operator*()  const;
   inline PointerType operator->() const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline const IteratorType& base() const;
   inline const IteratorType& stop() const;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   IteratorType cur_;  //!< Pointer to the current memory location.
   IteratorType end_;  //!< Pointer to the element one past the last element in the element range.
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Default constructor for ConstCastIterator.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline PtrVector<T,D,G>::ConstCastIterator<C>::ConstCastIterator()
   : cur_( nullptr )  // Pointer to the current memory location
   , end_( nullptr )  // Pointer to the element one past the last element in the element range
{
   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_DERIVED_FROM( C, T );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Standard constructor for ConstCastIterator.
//
// \param begin The beginning of the element range.
// \param end The end of the element range.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline PtrVector<T,D,G>::ConstCastIterator<C>::ConstCastIterator( IteratorType begin, IteratorType end )
   : cur_(begin)  // Pointer to the current memory location
   , end_(end)    // Pointer to the element one past the last element in the element range
{
   // The polymorphicFind() function finds the next pointer to an object with dynamic type 'C'
   // contained in the range [cur_,end). An equivalent code might look like this:
   //
   // while( cur_ != end_ && !dynamic_cast<C*>( *cur_ ) ) ++cur_;
   //
   // However, the specialization of polymorphicFind() for special type combinations is much
   // more efficient (and way easier!) than the specialization of this function!
   cur_ = polymorphicFind<C>( cur_, end_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different ConstCastIterator instances.
//
// \param it The foreign ConstCastIterator instance to be copied.
*/
template< typename T        // Type
        , typename D        // Deletion policy
        , typename G >      // Growth policy
template< typename C >      // Cast type
template< typename Other >  // The foreign constant cast iterator type
inline PtrVector<T,D,G>::ConstCastIterator<C>::ConstCastIterator( const ConstCastIterator<Other>& it )
   : cur_( it.base() )  // Pointer to the current memory location
   , end_( it.stop() )  // Pointer to the element one past the last element in the element range
{
   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_DERIVED_FROM( C, T );
   BLAZE_CONSTRAINT_MUST_BE_CONVERTIBLE( Other*, C* );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from CastIterator instances.
//
// \param it The foreign CastIterator instance to be copied.
*/
template< typename T        // Type
        , typename D        // Deletion policy
        , typename G >      // Growth policy
template< typename C >      // Cast type
template< typename Other >  // The foreign cast iterator type
inline PtrVector<T,D,G>::ConstCastIterator<C>::ConstCastIterator( const typename PtrVector<T,D,G>::BLAZE_TEMPLATE CastIterator<Other>& it )
   : cur_( it.base() )  // Pointer to the current memory location
   , end_( it.stop() )  // Pointer to the element one past the last element in the element range
{
   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_DERIVED_FROM( C, T );
   BLAZE_CONSTRAINT_MUST_BE_CONVERTIBLE( Other*, C* );
}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Pre-increment operator.
//
// \return Reference to the incremented cast iterator.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE ConstCastIterator<C>&
   PtrVector<T,D,G>::ConstCastIterator<C>::operator++()
{
   // The polymorphicFind() function finds the next pointer to an object with dynamic type 'C'
   // contained in the range [cur_+1,end). An equivalent code might look like this:
   //
   // while( ++cur_ != end_ && !dynamic_cast<const C*>( *cur_ ) ) {}
   //
   // However, the specialization of polymorphicFind() for special type combinations is much
   // more efficient (and way easier!) than the specialization of this function!
   cur_ = polymorphicFind<const C>( ++cur_, end_ );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Post-increment operator.
//
// \return The incremented cast iterator.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE ConstCastIterator<C>
   PtrVector<T,D,G>::ConstCastIterator<C>::operator++( int )
{
   ConstCastIterator tmp( *this );

   // The polymorphicFind() function finds the next pointer to an object with dynamic type 'C'
   // contained in the range [cur_+1,end). An equivalent code might look like this:
   //
   // while( ++cur_ != end_ && !dynamic_cast<const C*>( *cur_ ) ) {}
   //
   // However, the specialization of polymorphicFind() for special type combinations is much
   // more efficient (and way easier!) than the specialization of this function!
   cur_ = polymorphicFind<const C>( ++cur_, end_ );

   return tmp;
}
//*************************************************************************************************




//=================================================================================================
//
//  ACCESS OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns a handle to the element at the current iterator position.
//
// \return Handle to the element at the current iterator position.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE ConstCastIterator<C>::PointerType
   PtrVector<T,D,G>::ConstCastIterator<C>::operator*() const
{
   return static_cast<const C*>( *cur_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Direct access to the element at the current iterator position.
//
// \return Reference to the element at the current iterator position.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline typename PtrVector<T,D,G>::BLAZE_TEMPLATE ConstCastIterator<C>::PointerType
   PtrVector<T,D,G>::ConstCastIterator<C>::operator->() const
{
   return static_cast<const C*>( *cur_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Direct access to the current memory location of the constant cast iterator.
//
// \return Pointer to the current memory location.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline const typename PtrVector<T,D,G>::BLAZE_TEMPLATE ConstCastIterator<C>::IteratorType&
   PtrVector<T,D,G>::ConstCastIterator<C>::base() const
{
   return cur_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Direct access to the final memory location of the constant cast iterator.
//
// \return Pointer to the final memory location.
*/
template< typename T    // Type
        , typename D    // Deletion policy
        , typename G >  // Growth policy
template< typename C >  // Cast type
inline const typename PtrVector<T,D,G>::BLAZE_TEMPLATE ConstCastIterator<C>::IteratorType&
   PtrVector<T,D,G>::ConstCastIterator<C>::stop() const
{
   return end_;
}
//*************************************************************************************************

} // namespace blaze

#endif
