//=================================================================================================
/*!
//  \file blaze/util/serialization/Archive.h
//  \brief Header file for the Archive class
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

#ifndef _BLAZE_UTIL_SERIALIZATION_ARCHIVE_H_
#define _BLAZE_UTIL_SERIALIZATION_ARCHIVE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <memory>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/NonCopyable.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsNumeric.h>


namespace blaze {


//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Binary archive for the portable serialization of data.
// \ingroup serialization
//
// The Archive class implements the functionality to create platform independent, portable,
// representations of arbitrary C++ data structures. The resulting binary data structures can
// be used to reconstitute the data structures in a different context, on another platform,
// etc.
//
// The following example demonstrates the Archive class by means of a C-style POD data structure:

   \code
   struct POD {
      int i;
      unsigned int u;
      float f;
      double d;
   };
   \endcode

// The archive already supports the (de-)serialization of built-in data types. In order to be able
// to serialize the POD data structure, i.e. to convert it into a binary representation that can
// stored or transfered to other machines, the according \c serialize and \c deserialize function
// have to be implemented:

   \code
   template< typename Archive >
   void serialize( Archive& archive, const POD& pod )
   {
      archive << pod.i << pod.u << pod.f << pod.d;
   }

   template< typename Archive >
   void deserialize( Archive& archive, POD& pod )
   {
      archive >> pod.i >> pod.u >> pod.f >> pod.d;
   }
   \endcode

// The \c serialize function implements the conversion from the POD to a binary representation,
// the \c deserialize function implements the reverse conversion from a binary representation to
// a POD. Note that it is important to write the values to the archive in exactly the same order
// as the values are read from the archive!
//
// With these two functions it is already possible to serialize a POD object:

   \code
   int main()
   {
      // Creating a binary representation of a POD in the file 'filename'
      {
         POD pod;
         // ... Initialize the POD with values
         Archive<std::ofstream> archive( "filename", std::ofstream::trunc );
         archive << pod;
      }

      // Reconstituting the POD from the binary representation
      {
         POD pod;
         Archive<std::ifstream> archive( "filename" );
         archive >> pod;
      }
   }
   \endcode

// Each archive has to be bound to either an output or input stream. In the example, the
// archive is bound to a file output stream that is created by passing the file name and
// the according flags to the archive. Subsequently, the archive can be used like an output
// stream to write the POD data structure to the file called 'filename'.
//
// The reverse conversion from the binary representation contained in the file to the POD
// data structure can be achieved by binding an archive to a file input stream. Subsequently,
// the archive can be used as an input stream to read the POD from file.
//
// Note that the Archive class can be bound to any kind of input or output stream (or also
// iostream) that supports the standard write or read functions, respectively. Therefore
// the serialization of a C++ data structure is not restricted to binary files, but allows
// for any possible destination.
*/
template< typename Stream >  // Type of the bound stream
class Archive : private NonCopyable
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... Args >
   explicit inline Archive( Args&&... args );

   explicit inline Archive( Stream& stream );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Operators***********************************************************************************
   /*!\name Operators */
   //@{
   inline operator bool() const;
   inline bool operator!() const;
   //@}
   //**********************************************************************************************

   //**Serialization functions*********************************************************************
   /*!\name Serialization functions */
   //@{
   template< typename T >
   EnableIf_< IsNumeric<T>, Archive& > operator<<( const T& value );

   template< typename T >
   DisableIf_< IsNumeric<T>, Archive& > operator<<( const T& value );

   template< typename T >
   EnableIf_< IsNumeric<T>, Archive& > operator>>( T& value );

   template< typename T >
   DisableIf_< IsNumeric<T>, Archive& > operator>>( T& value );

   template< typename Type >
   inline EnableIf_< IsNumeric<Type>, Archive& > write( const Type* array, size_t count );

   template< typename Type >
   inline EnableIf_< IsNumeric<Type>, Archive& > read ( Type* array, size_t count );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline typename Stream::int_type peek() const;

   inline bool good() const;
   inline bool eof () const;
   inline bool fail() const;
   inline bool bad () const;

   inline std::ios_base::iostate rdstate () const;
   inline void                   setstate( std::ios_base::iostate state );
   inline void                   clear   ( std::ios_base::iostate state = std::ios_base::goodbit );
   //@}
   //**********************************************************************************************

 private:

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::unique_ptr<Stream> ptr_;  //!< The dynamically allocated stream resource.
                                  /*!< In case no stream is bound to the archive from the outside,
                                       this smart pointer handles the internally allocated stream
                                       resource. */
   Stream& stream_;               //!< Reference to the bound stream.
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
/*!\brief Creating an archive with an internal stream resource.
//
// \param args The stream arguments.
//
// This function creates a new archive with an internal stream resource, which is created based
// on the given arguments \a args.
*/
template< typename Stream >   // Type of the bound stream
template< typename... Args >  // Types of the optional arguments
inline Archive<Stream>::Archive( Args&&... args )
   : ptr_   ( new Stream( std::forward<Args>( args )... ) )  // The dynamically allocated stream resource
   , stream_( *ptr_.get() )                                  // Reference to the bound stream
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating an archive with an external stream resource.
//
// \param stream The stream to be bound to the archive.
//
// This function creates a new archive with an external stream resource, which is bound to the
// archive. Note that the stream is NOT automatically closed when the archive is destroyed.
*/
template< typename Stream >  // Type of the bound stream
inline Archive<Stream>::Archive( Stream& stream )
   : ptr_   ()          // The dynamically allocated stream resource
   , stream_( stream )  // Reference to the bound stream
{}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the current state of the archive.
//
// \return \a false in case an input/output error has occurred, \a true otherwise.
//
// This operator returns the current state of the Archive based on the state of the bound stream.
// In case an input/output error has occurred, the operator returns \a false, otherwise it returns
// \a true.
*/
template< typename Stream >  // Type of the bound stream
inline Archive<Stream>::operator bool() const
{
   return !stream_.fail();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the negated state of the archive.
//
// \return \a true in case an input/output error has occurred, \a false otherwise.
//
// This operator returns the negated state of the Archive based on the state of the bound stream.
// In case an input/output error has occurred, the operator returns \a true, otherwise it returns
// \a false.
*/
template< typename Stream >  // Type of the bound stream
inline bool Archive<Stream>::operator!() const
{
   return stream_.fail();
}
//*************************************************************************************************




//=================================================================================================
//
//  SERIALIZATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Serializes the given built-in data value and writes it to the archive.
//
// \param value The built-in data value to be serialized.
// \return Reference to the archive.
*/
template< typename Stream >  // Type of the bound stream
template< typename T >       // Type of the value to be serialized
EnableIf_< IsNumeric<T>, Archive<Stream>& > Archive<Stream>::operator<<( const T& value )
{
   typedef typename Stream::char_type  CharType;
   stream_.write( reinterpret_cast<const CharType*>( &value ), sizeof( T ) );
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Serializes the user-defined object and writes it to the archive.
//
// \param value The user-defined object to be serialized.
// \return Reference to the archive.
*/
template< typename Stream >  // Type of the bound stream
template< typename T >       // Type of the object to be serialized
DisableIf_< IsNumeric<T>, Archive<Stream>& > Archive<Stream>::operator<<( const T& value )
{
   serialize( *this, value );
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a value of built-in data type and reads it from the archive.
//
// \param value The built-in data value to be read from the archive.
// \return Reference to the archive.
*/
template< typename Stream >  // Type of the bound stream
template< typename T >       // Type of the value to be deserialized
EnableIf_< IsNumeric<T>, Archive<Stream>& > Archive<Stream>::operator>>( T& value )
{
   typedef typename Stream::char_type  CharType;
   stream_.read( reinterpret_cast<CharType*>( &value ), sizeof( T ) );
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes an object of user-defined data type and reads it from the archive.
//
// \param value The user-defined object to be read from the archive.
// \return Reference to the archive.
*/
template< typename Stream >  // Type of the bound stream
template< typename T >       // Type of the value to be deserialized
DisableIf_< IsNumeric<T>, Archive<Stream>& > Archive<Stream>::operator>>( T& value )
{
   deserialize( *this, value );
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Writing an array of values to the stream.
//
// \param array Pointer to the first element of the array.
// \param count The number of elements in the array.
//
// This function writes \a count elements of the numeric type \a Type from the given \a array
// to the archive. Note that the function can only be used for arrays of numeric type!
*/
template< typename Stream >  // Type of the bound stream
template< typename Type >    // Type of the array elements
inline EnableIf_< IsNumeric<Type>, Archive<Stream>& >
   Archive<Stream>::write( const Type* array, size_t count )
{
   typedef typename Stream::char_type  CharType;
   stream_.write( reinterpret_cast<const CharType*>( array ), count*sizeof(Type) );
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reading an array of values from the stream.
//
// \param array Pointer to the first element of the array.
// \param count The number of elements in the array.
//
// This function reads \a count elements of the numeric type \a Type from the archive and
// writes them to the \a array. Note that the function can only be used for arrays of numeric
// type. Also note that the read function does not allocate memory, but expects that \a array
// provides storage for at least \a count elements of type \a Type!
*/
template< typename Stream >  // Type of the bound stream
template< typename Type >    // Type of the array elements
inline EnableIf_< IsNumeric<Type>, Archive<Stream>& >
   Archive<Stream>::read( Type* array, size_t count )
{
   typedef typename Stream::char_type  CharType;
   stream_.read( reinterpret_cast<CharType*>( array ), count*sizeof(Type) );
   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Reads the next character from the input stream without extracting it.
//
// \return The next character contained in the input stream.
*/
template< typename Stream >  // Type of the bound stream
inline typename Stream::int_type Archive<Stream>::peek() const
{
   return stream_.peek();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if no error has occurred, i.e. I/O operations are available.
//
// \return \a true in case no error has occurred, \a false otherwise.
*/
template< typename Stream >  // Type of the bound stream
inline bool Archive<Stream>::good() const
{
   return stream_.good();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if end-of-file (EOF) has been reached
//
// \return \a true in case end-of-file has been reached, \a false otherwise.
*/
template< typename Stream >  // Type of the bound stream
inline bool Archive<Stream>::eof() const
{
   return stream_.eof();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if a recoverable error has occurred.
//
// \return \a true in case a recoverable error has occurred, \a false otherwise.
*/
template< typename Stream >  // Type of the bound stream
inline bool Archive<Stream>::fail() const
{
   return stream_.fail();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if a non-recoverable error has occurred.
//
// \return \a true in case a non-recoverable error has occurred, \a false otherwise.
*/
template< typename Stream >  // Type of the bound stream
inline bool Archive<Stream>::bad() const
{
   return stream_.bad();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current state flags settings.
//
// \return The current state flags settings.
*/
template< typename Stream >  // Type of the bound stream
inline std::ios_base::iostate Archive<Stream>::rdstate() const
{
   return stream_.rdstate();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Sets the state flags to a specific value.
//
// \param state The new error state flags setting.
// \return void
*/
template< typename Stream >  // Type of the bound stream
inline void Archive<Stream>::setstate( std::ios_base::iostate state )
{
   stream_.setstate( state );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clears error and eof flags.
//
// \param state The new error state flags setting.
// \return void
*/
template< typename Stream >  // Type of the bound stream
inline void Archive<Stream>::clear( std::ios_base::iostate state )
{
   return stream_.clear( state );
}
//*************************************************************************************************

} // namespace blaze

#endif
