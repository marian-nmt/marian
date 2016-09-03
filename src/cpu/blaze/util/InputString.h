//=================================================================================================
/*!
//  \file blaze/util/InputString.h
//  \brief String implementation for the extraction of input strings
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

#ifndef _BLAZE_UTIL_INPUTSTRING_H_
#define _BLAZE_UTIL_INPUTSTRING_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cctype>
#include <iostream>
#include <string>
#include <blaze/util/Assert.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Implementation of a string wrapper.
// \ingroup util
//
// The InputString class is a wrapper class for the purpose to read input strings delimited by
// quotations from streams, like for instance "example input". All characters between the
// leading and the trailing quotation are extracted unchanged from the input stream, including
// whitespaces. The input string has to be in one single line. In case of input errors, the
// \a std::istream::failbit of the input stream is set.
*/
class InputString
{
   //**Friend declarations*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   friend std::istream& operator>>( std::istream& is, InputString& str );
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef std::string::size_type SizeType;  //!< Size type of the InputString.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline InputString( const char* string="" );
   explicit inline InputString( const std::string& string );
            inline InputString( const InputString& s );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline InputString& operator=( const char* string );
   inline InputString& operator=( const std::string& string );
   // No explicitly declared copy assignment operator.
   //@}
   //**********************************************************************************************

   //**Access functions****************************************************************************
   /*!\name Access functions */
   //@{
   inline char&       operator[]( SizeType index );
   inline const char& operator[]( SizeType index ) const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions
   // \brief The utility functions are named in the style of \a std::string.
   */
   //@{
   inline const char*        c_str()    const;
   inline const std::string& str()      const;
   inline SizeType           size()     const;
   inline SizeType           capacity() const;
   inline bool               empty()    const;
   inline void               reserve( SizeType newSize );
   //@}
   //**********************************************************************************************

 private:
   //**Member varibales****************************************************************************
   /*!\name Member variables */
   //@{
   std::string buffer_;  //!< The character buffer.
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
/*!\brief The default constructor for InputString.
//
// \param string The initial value for the string.
*/
inline InputString::InputString( const char* string )
   : buffer_( string )  // Character buffer
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for the direct initialization with a \a std::string.
//
// \param string The initial value for the string.
*/
inline InputString::InputString( const std::string& string )
   : buffer_( string )  // Character buffer
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for InputString.
//
// \param s The string object to be copied.
*/
inline InputString::InputString( const InputString& s )
   : buffer_( s.buffer_ )  // Character buffer
{}
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Assignment operator for C-style character strings.
//
// \param string The C-style string to be copied.
// \return Reference to the assigned string.
*/
inline InputString& InputString::operator=( const char* string )
{
   buffer_ = string;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for STL strings.
//
// \param string The STL string to be copied.
// \return Reference to the assigned string.
*/
inline InputString& InputString::operator=( const std::string& string )
{
   buffer_ = string;
   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Direct access to the characters of the string.
//
// \param index Access index. The index has to be in the range \f$[0..size-1]\f$.
// \return A reference to the indexed character.
*/
inline char& InputString::operator[]( SizeType index )
{
   BLAZE_USER_ASSERT( index < size(), "Invalid access index" );
   return buffer_[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Direct access to the characters of the string.
//
// \param index Access index. The index has to be in the range \f$[0..size-1]\f$.
// \return A reference to the indexed character.
*/
inline const char& InputString::operator[]( SizeType index ) const
{
   BLAZE_USER_ASSERT( index < size(), "Invalid access index" );
   return buffer_[index];
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion to a constant character array.
//
// \return The converted constant character array.
*/
inline const char* InputString::c_str() const
{
   return buffer_.c_str();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion to a \a std::string.
//
// \return The converted \a std::string.
*/
inline const std::string& InputString::str() const
{
   return buffer_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the size of the string.
//
// \return The size of the string.
*/
inline InputString::SizeType InputString::size() const
{
   return buffer_.size();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the string.
//
// \return The capacity of the string.
*/
inline InputString::SizeType InputString::capacity() const
{
   return buffer_.capacity();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns if the string is empty.
//
// \return \a true if the string is empty, \a false if it is not.
*/
inline bool InputString::empty() const
{
   return buffer_.empty();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reserves at least \a size characters within the string.
//
// \param newSize The minimum size of the string.
// \return void
*/
inline void InputString::reserve( SizeType newSize )
{
   buffer_.reserve( newSize );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name InputString operators */
//@{
inline bool          IsFileName( const InputString& s );
inline std::ostream& operator<<( std::ostream& os, const InputString& str );
inline std::istream& operator>>( std::istream& is, InputString& str );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Tests for a valid file name.
// \ingroup util
//
// \param s The file name string.
// \return \a true if the string is a file name, \a false if it is not.
//
// In order to be a file name, the first character can only be an alphanumerical character,
// '.', '/' or '_'.
*/
inline bool IsFileName( const InputString& s )
{
   if( s.empty() ) return false;
   else if( isalnum(s[0]) || s[0] == '.' || s[0] == '/' || s[0] == '_' ) return true;
   else return false;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Global output operator for the InputString class.
// \ingroup util
//
// \param os Reference to the output stream.
// \param str Reference to a string object.
// \return The output stream.
*/
inline std::ostream& operator<<( std::ostream& os, const InputString& str )
{
   return os << str.str();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Global input operator for the InputString class.
// \ingroup util
//
// \param is Reference to the input stream.
// \param str Reference to a string object.
// \return The input stream.
//
// The input operator guarantees that the string object is not changed in the case of an input
// error.
*/
inline std::istream& operator>>( std::istream& is, InputString& str )
{
   if( !is ) return is;

   char c;
   std::string buffer;
   std::istream::pos_type pos( is.tellg() );

   buffer.reserve( 20 );

   // Extracting the leading quotation
   is >> std::ws;
   if( !is.get( c ) || c != '"' ) {
      is.clear();
      is.seekg( pos );
      is.setstate( std::istream::failbit );
      return is;
   }

   // Extracting the input string
   while( true )
   {
      if( !is.get( c ) || c == '\n' ) {
         is.clear();
         is.seekg( pos );
         is.setstate( std::istream::failbit );
         return is;
      }
      else if( c == '"' ) break;

      buffer.push_back( c );
   }

   // Replacing the old string
   swap( str.buffer_, buffer );

   return is;
}
//*************************************************************************************************

} // namespace blaze

#endif
