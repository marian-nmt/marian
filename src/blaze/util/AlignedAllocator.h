//=================================================================================================
/*!
//  \file blaze/util/AlignedAllocator.h
//  \brief Header file for the AlignedAllocator implementation
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

#ifndef _BLAZE_UTIL_ALIGNEDALLOCATOR_H_
#define _BLAZE_UTIL_ALIGNEDALLOCATOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/Memory.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Allocator for type-specific aligned memory.
// \ingroup util
//
// The AlignedAllocator class template represents an implementation of the allocator concept of
// the standard library for the allocation of type-specific, aligned, uninitialized memory. The
// allocator performs its allocation via the blaze::allocate() and blaze::deallocate() functions
// to guarantee properly aligned memory based on the alignment restrictions of the specified type
// \a Type. For instance, in case the given type is a fundamental, built-in data type and in case
// SSE vectorization is possible, the returned memory is guaranteed to be at least 16-byte aligned.
// In case AVX is active, the memory is even guaranteed to be at least 32-byte aligned.
*/
template< typename Type >
class AlignedAllocator
{
 public:
   //**Type definitions****************************************************************************
   typedef Type            ValueType;        //!< Type of the allocated values.
   typedef Type*           Pointer;          //!< Type of a pointer to the allocated values.
   typedef const Type*     ConstPointer;     //!< Type of a pointer-to-const to the allocated values.
   typedef Type&           Reference;        //!< Type of a reference to the allocated values.
   typedef const Type&     ConstReference;   //!< Type of a reference-to-const to the allocated values.
   typedef std::size_t     SizeType;         //!< Size type of the aligned allocator.
   typedef std::ptrdiff_t  DifferenceType;   //!< Difference type of the aligned allocator.

   // STL allocator requirements
   typedef ValueType       value_type;       //!< Type of the allocated values.
   typedef Pointer         pointer;          //!< Type of a pointer to the allocated values.
   typedef ConstPointer    const_pointer;    //!< Type of a pointer-to-const to the allocated values.
   typedef Reference       reference;        //!< Type of a reference to the allocated values.
   typedef ConstReference  const_reference;  //!< Type of a reference-to-const to the allocated values.
   typedef SizeType        size_type;        //!< Size type of the aligned allocator.
   typedef DifferenceType  difference_type;  //!< Difference type of the aligned allocator.
   //**********************************************************************************************

   //**rebind class definition*********************************************************************
   /*!\brief Implementation of the AlignedAllocator rebind mechanism.
   */
   template< typename Type2 >
   struct rebind
   {
      typedef AlignedAllocator<Type2>  other;  //!< Type of the other allocator.
   };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline AlignedAllocator();

   template< typename Type2 >
   inline AlignedAllocator( const AlignedAllocator<Type2>& );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline constexpr size_t max_size() const noexcept;
   inline Pointer          address( Reference x ) const noexcept;
   inline ConstPointer     address( ConstReference x ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Allocation functions************************************************************************
   /*!\name Allocation functions */
   //@{
   inline Pointer allocate  ( size_t numObjects, const void* localityHint = nullptr );
   inline void    deallocate( Pointer ptr, size_t numObjects );
   //@}
   //**********************************************************************************************

   //**Construction functions**********************************************************************
   /*!\name Construction functions */
   //@{
   inline void construct( Pointer ptr, const Type& value );
   inline void destroy  ( Pointer ptr ) noexcept;
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
/*!\brief The default constructor for AlignedAllocator.
*/
template< typename Type >
inline AlignedAllocator<Type>::AlignedAllocator()
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different AlignedAllocator instances.
//
// \param allocator The foreign aligned allocator to be copied.
*/
template< typename Type >
template< typename Type2 >
inline AlignedAllocator<Type>::AlignedAllocator( const AlignedAllocator<Type2>& allocator )
{
   UNUSED_PARAMETER( allocator );
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the maximum possible number of elements that can be allocated together.
//
// \return The maximum number of elements that can be allocated together.
*/
template< typename Type >
inline constexpr size_t AlignedAllocator<Type>::max_size() const noexcept
{
   return size_t(-1) / sizeof( Type );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the address of the given element.
//
// \return The address of the given element.
*/
template< typename Type >
inline typename AlignedAllocator<Type>::Pointer
   AlignedAllocator<Type>::address( Reference x ) const noexcept
{
   return &x;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the address of the given element.
//
// \return The address of the given element.
*/
template< typename Type >
inline typename AlignedAllocator<Type>::ConstPointer
   AlignedAllocator<Type>::address( ConstReference x ) const noexcept
{
   return &x;
}
//*************************************************************************************************




//=================================================================================================
//
//  ALLOCATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Allocates aligned memory for the specified number of objects.
//
// \param numObjects The number of objects to be allocated.
// \param localityHint Hint for improved locality.
// \return Pointer to the newly allocated memory.
//
// This function allocates a junk of memory for the specified number of objects of type \a Type.
// The returned pointer is guaranteed to be aligned according to the alignment restrictions of
// the data type \a Type. For instance, in case the type is a fundamental, built-in data type
// and in case SSE vectorization is possible, the returned memory is guaranteed to be at least
// 16-byte aligned. In case AVX is active, the memory is even guaranteed to be 32-byte aligned.
*/
template< typename Type >
inline typename AlignedAllocator<Type>::Pointer
   AlignedAllocator<Type>::allocate( size_t numObjects, const void* localityHint )
{
   UNUSED_PARAMETER( localityHint );

   const size_t alignment( AlignmentOf<Type>::value );

   if( alignment >= 8UL ) {
      return reinterpret_cast<Type*>( allocate_backend( numObjects*sizeof(Type), alignment ) );
   }
   else {
      return static_cast<Pointer>( operator new[]( numObjects * sizeof( Type ) ) );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deallocation of memory.
//
// \param ptr The address of the first element of the array to be deallocated.
// \param numObjects The number of objects to be deallocated.
// \return void
//
// This function deallocates a junk of memory that was previously allocated via the allocate()
// function. Note that the argument \a numObjects must be equal ot the first argument of the call
// to allocate() that origianlly produced \a ptr.
*/
template< typename Type >
inline void AlignedAllocator<Type>::deallocate( Pointer ptr, size_t numObjects )
{
   UNUSED_PARAMETER( numObjects );

   if( ptr == nullptr )
      return;

   const size_t alignment( AlignmentOf<Type>::value );

   if( alignment >= 8UL ) {
      deallocate_backend( ptr );
   }
   else {
      operator delete[]( ptr );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructs an object of type \a Type at the specified memory location.
//
// \param ptr Pointer to the allocated, uninitialized storage.
// \param value The initialization value.
// \return void
//
// This function constructs an object of type \a Type in the allocated, uninitialized storage
// pointed to by \a ptr. This construction is performed via placement-new.
*/
template< typename Type >
inline void AlignedAllocator<Type>::construct( Pointer ptr, ConstReference value )
{
   ::new( ptr ) Type( value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Destroys the object of type \a Type at the specified memory location.
//
// \param ptr Pointer to the object to be destroyed.
// \return void
//
// This function destroys the object at the specified memory location via a direct call to its
// destructor.
*/
template< typename Type >
inline void AlignedAllocator<Type>::destroy( Pointer ptr ) noexcept
{
   ptr->~Type();
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name AlignedAllocator operators */
//@{
template< typename T1, typename T2 >
inline bool operator==( const AlignedAllocator<T1>& lhs, const AlignedAllocator<T2>& rhs ) noexcept;

template< typename T1, typename T2 >
inline bool operator!=( const AlignedAllocator<T1>& lhs, const AlignedAllocator<T2>& rhs ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between two AlignedAllocator objects.
//
// \param lhs The left-hand side aligned allocator.
// \param rhs The right-hand side aligned allocator.
// \return \a true.
*/
template< typename T1    // Type of the left-hand side aligned allocator
        , typename T2 >  // Type of the right-hand side aligned allocator
inline bool operator==( const AlignedAllocator<T1>& lhs, const AlignedAllocator<T2>& rhs ) noexcept
{
   UNUSED_PARAMETER( lhs, rhs );
   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between two AlignedAllocator objects.
//
// \param lhs The left-hand side aligned allocator.
// \param rhs The right-hand side aligned allocator.
// \return \a false.
*/
template< typename T1    // Type of the left-hand side aligned allocator
        , typename T2 >  // Type of the right-hand side aligned allocator
inline bool operator!=( const AlignedAllocator<T1>& lhs, const AlignedAllocator<T2>& rhs ) noexcept
{
   UNUSED_PARAMETER( lhs, rhs );
   return false;
}
//*************************************************************************************************

} // namespace blaze

#endif
