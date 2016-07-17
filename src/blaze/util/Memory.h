//=================================================================================================
/*!
//  \file blaze/util/Memory.h
//  \brief Header file for memory allocation and deallocation functionality
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

#ifndef _BLAZE_UTIL_MEMORY_H_
#define _BLAZE_UTIL_MEMORY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#if defined(_MSC_VER)
#  include <malloc.h>
#endif
#include <cstdlib>
#include <new>
#include <blaze/util/Assert.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/Exception.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blaze/util/typetraits/IsBuiltin.h>


namespace blaze {

//=================================================================================================
//
//  BACKEND ALLOCATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation for aligned array allocation.
// \ingroup util
//
// \param size The number of bytes to be allocated.
// \param alignment The required minimum alignment.
// \return Byte pointer to the first element of the aligned array.
// \exception std::bad_alloc Allocation failed.
//
// This function provides the functionality to allocate memory based on the given alignment
// restrictions. For that purpose it uses the according system-specific memory allocation
// functions.
*/
inline byte_t* allocate_backend( size_t size, size_t alignment )
{
   void* raw( nullptr );

#if defined(_MSC_VER)
   raw = _aligned_malloc( size, alignment );
   if( raw == nullptr ) {
#else
   if( posix_memalign( &raw, alignment, size ) ) {
#endif
      BLAZE_THROW_BAD_ALLOC;
   }

   return reinterpret_cast<byte_t*>( raw );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation for the deallocation of aligned memory.
// \ingroup util
//
// \param address The address of the first element of the array to be deallocated.
// \return void
//
// This function deallocates the given memory that was previously allocated via the allocate()
// function. For that purpose it uses the according system-specific memory deallocation functions.
*/
inline void deallocate_backend( const void* address ) noexcept
{
#if defined(_MSC_VER)
   _aligned_free( const_cast<void*>( address ) );
#else
   free( const_cast<void*>( address ) );
#endif
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ALLOCATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Aligned array allocation for built-in data types.
// \ingroup util
//
// \param size The number of elements of the given type to allocate.
// \return Pointer to the first element of the aligned array.
// \exception std::bad_alloc Allocation failed.
//
// The allocate() function provides the functionality to allocate memory based on the alignment
// restrictions of the given built-in data type. For instance, in case SSE vectorization is
// possible, the returned memory is guaranteed to be at least 16-byte aligned. In case AVX is
// active, the memory is even guaranteed to be at least 32-byte aligned.
//
// Examples:

   \code
   // Guaranteed to be 16-byte aligned (32-byte aligned in case AVX is used)
   double* dp = allocate<double>( 10UL );
   \endcode
*/
template< typename T >
EnableIf_< IsBuiltin<T>, T* > allocate( size_t size )
{
   const size_t alignment( AlignmentOf<T>::value );

   if( alignment >= 8UL ) {
      return reinterpret_cast<T*>( allocate_backend( size*sizeof(T), alignment ) );
   }
   else return ::new T[size];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned array allocation for user-specific class types.
// \ingroup util
//
// \param size The number of elements of the given type to allocate.
// \return Pointer to the first element of the aligned array.
// \exception std::bad_alloc Allocation failed.
//
// The allocate() function provides the functionality to allocate memory based on the alignment
// restrictions of the given user-specific class type. For instance, in case the given type has
// the requirement to be 32-byte aligned, the returned pointer is guaranteed to be 32-byte
// aligned. Additionally, all elements of the array are guaranteed to be default constructed.
// Note that the allocate() function provides exception safety similar to the new operator: In
// case any element throws an exception during construction, all elements that have already been
// constructed are destroyed in reverse order and the allocated memory is deallocated again.
*/
template< typename T >
DisableIf_< IsBuiltin<T>, T* > allocate( size_t size )
{
   const size_t alignment ( AlignmentOf<T>::value );
   const size_t headersize( ( sizeof(size_t) < alignment ) ? ( alignment ) : ( sizeof( size_t ) ) );

   BLAZE_INTERNAL_ASSERT( headersize >= alignment      , "Invalid header size detected" );
   BLAZE_INTERNAL_ASSERT( headersize % alignment == 0UL, "Invalid header size detected" );

   if( alignment >= 8UL )
   {
      byte_t* const raw( allocate_backend( size*sizeof(T)+headersize, alignment ) );

      *reinterpret_cast<size_t*>( raw ) = size;

      T* const address( reinterpret_cast<T*>( raw + headersize ) );
      size_t i( 0UL );

      try {
         for( ; i<size; ++i )
            ::new (address+i) T();
      }
      catch( ... ) {
         while( i != 0UL )
            address[--i].~T();
         deallocate_backend( raw );
         throw;
      }

      return address;
   }
   else return ::new T[size];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deallocation of memory for built-in data types.
// \ingroup util
//
// \param address The address of the first element of the array to be deallocated.
// \return void
//
// This function deallocates the given memory that was previously allocated via the allocate()
// function.
*/
template< typename T >
EnableIf_< IsBuiltin<T> > deallocate( T* address ) noexcept
{
   if( address == nullptr )
      return;

   const size_t alignment( AlignmentOf<T>::value );

   if( alignment >= 8UL ) {
      deallocate_backend( address );
   }
   else delete[] address;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deallocation of memory for user-specific class types.
// \ingroup util
//
// \param address The address of the first element of the array to be deallocated.
// \return void
//
// This function deallocates the given memory that was previously allocated via the allocate()
// function.
*/
template< typename T >
DisableIf_< IsBuiltin<T> > deallocate( T* address )
{
   if( address == nullptr )
      return;

   const size_t alignment ( AlignmentOf<T>::value );
   const size_t headersize( ( sizeof(size_t) < alignment ) ? ( alignment ) : ( sizeof( size_t ) ) );

   BLAZE_INTERNAL_ASSERT( headersize >= alignment      , "Invalid header size detected" );
   BLAZE_INTERNAL_ASSERT( headersize % alignment == 0UL, "Invalid header size detected" );

   if( alignment >= 8UL )
   {
      const byte_t* const raw = reinterpret_cast<byte_t*>( address ) - headersize;

      const size_t size( *reinterpret_cast<const size_t*>( raw ) );
      for( size_t i=0UL; i<size; ++i )
         address[i].~T();

      deallocate_backend( raw );
   }
   else delete[] address;
}
//*************************************************************************************************

} // namespace blaze

#endif
