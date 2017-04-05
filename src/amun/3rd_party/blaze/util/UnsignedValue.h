//=================================================================================================
/*!
//  \file blaze/util/UnsignedValue.h
//  \brief Header file for the UnsignedValue class
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

#ifndef _BLAZE_UTIL_UNSIGNEDVALUE_H_
#define _BLAZE_UTIL_UNSIGNEDVALUE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <istream>
#include <ostream>
#include <blaze/util/constraints/Unsigned.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Implementation of a wrapper for built-in unsigned integral values.
// \ingroup util
//
// This class wraps a value of built-in unsigned integral type in order to be able to extract
// non-negative unsigned integral values from an input stream.
*/
template< typename T >  // Type of the unsigned value
class UnsignedValue
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline UnsignedValue( T value=0 );
   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Assignment operator*************************************************************************
   /*!\name Assignment operator */
   //@{
   inline UnsignedValue& operator=( T value );
   // No explicitly declared copy assignment operator.
   //@}
   //**********************************************************************************************

   //**Conversion operator*************************************************************************
   /*!\name Conversion operator */
   //@{
   inline operator T() const;
   //@}
   //**********************************************************************************************

   //**Access function*****************************************************************************
   /*!\name Access functions */
   //@{
   inline T get() const;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   T value_;  //!< The wrapped built-in unsigned integral value.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_UNSIGNED_TYPE( T );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The default constructor for UnsignedInt.
//
// \param value The initial value for the unsigned integer.
*/
template< typename T >  // Type of the unsigned value
inline UnsignedValue<T>::UnsignedValue( T value )
   : value_( value )  // The wrapped built-in unsigned integral value
{}
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Assignment of a built-in unsigned integral value.
//
// \param value The unsigned integral value.
// \return Reference to the assigned UnsignedValue object.
*/
template< typename T >  // Type of the unsigned value
inline UnsignedValue<T>& UnsignedValue<T>::operator=( T value )
{
   value_ = value;
   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion to the built-in unsigned integral type.
//
// \return The wrapped built-in unsigned integral value.
*/
template< typename T >  // Type of the unsigned value
inline UnsignedValue<T>::operator T() const
{
   return value_;
}
//*************************************************************************************************




//=================================================================================================
//
//  ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Access to the wrapped built-in unsigned integral value.
//
// \return The wrapped built-in unsigned integral value.
*/
template< typename T >  // Type of the unsigned value
inline T UnsignedValue<T>::get() const
{
   return value_;
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name UnsignedValue operators */
//@{
template< typename T1, typename T2 >
inline bool operator==( const UnsignedValue<T1>& lhs, const UnsignedValue<T2>& rhs );

template< typename T1, typename T2 >
inline bool operator!=( const UnsignedValue<T1>& lhs, const UnsignedValue<T2>& rhs );

template< typename T1, typename T2 >
inline bool operator< ( const UnsignedValue<T1>& lhs, const UnsignedValue<T2>& rhs );

template< typename T1, typename T2 >
inline bool operator> ( const UnsignedValue<T1>& lhs, const UnsignedValue<T2>& rhs );

template< typename T1, typename T2 >
inline bool operator<=( const UnsignedValue<T1>& lhs, const UnsignedValue<T2>& rhs );

template< typename T1, typename T2 >
inline bool operator>=( const UnsignedValue<T1>& lhs, const UnsignedValue<T2>& rhs );

template< typename T >
inline std::ostream& operator<<( std::ostream& os, const UnsignedValue<T>& uv );

template< typename T >
std::istream& operator>>( std::istream& is, UnsignedValue<T>& uv );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between two UnsignedValue objects.
//
// \param lhs The left-hand side UnsignedValue wrapper.
// \param rhs The right-hand side UnsignedValue wrapper.
// \return \a true if the two values are equal, \a false if not.
*/
template< typename T1    // Type of the left-hand side unsigned value
        , typename T2 >  // Type of the right-hand side unsigned value
inline bool operator==( const UnsignedValue<T1>& lhs, const UnsignedValue<T2>& rhs )
{
   return lhs.get() == rhs.get();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between two UnsignedValue objects.
//
// \param lhs The left-hand side UnsignedValue wrapper.
// \param rhs The right-hand side UnsignedValue wrapper.
// \return \a true if the two values are not equal, \a true if they are equal.
*/
template< typename T1    // Type of the left-hand side unsigned value
        , typename T2 >  // Type of the right-hand side unsigned value
inline bool operator!=( const UnsignedValue<T1>& lhs, const UnsignedValue<T2>& rhs )
{
   return lhs.get() != rhs.get();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-than comparison between two UnsignedValue objects.
//
// \param lhs The left-hand side UnsignedValue wrapper.
// \param rhs The right-hand side UnsignedValue wrapper.
// \return \a true if the left value is less than the right value, \a false if not.
*/
template< typename T1    // Type of the left-hand side unsigned value
        , typename T2 >  // Type of the right-hand side unsigned value
inline bool operator<( const UnsignedValue<T1>& lhs, const UnsignedValue<T2>& rhs )
{
   return lhs.get() < rhs.get();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-than comparison between two UnsignedValue objects.
//
// \param lhs The left-hand side UnsignedValue wrapper.
// \param rhs The right-hand side UnsignedValue wrapper.
// \return \a true if the left value if greater than the right value, \a false if not.
*/
template< typename T1    // Type of the left-hand side unsigned value
        , typename T2 >  // Type of the right-hand side unsigned value
inline bool operator>( const UnsignedValue<T1>& lhs, const UnsignedValue<T2>& rhs )
{
   return lhs.get() > rhs.get();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between two UnsignedValue objects.
//
// \param lhs The left-hand side UnsignedValue wrapper.
// \param rhs The right-hand side UnsignedValue wrapper.
// \return \a true if the left value is less or equal than the right value, \a false if not.
*/
template< typename T1    // Type of the left-hand side unsigned value
        , typename T2 >  // Type of the right-hand side unsigned value
inline bool operator<=( const UnsignedValue<T1>& lhs, const UnsignedValue<T2>& rhs )
{
   return lhs.get() <= rhs.get();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-or-equal-than comparison between two UnsignedValue objects.
//
// \param lhs The left-hand side UnsignedValue wrapper.
// \param rhs The right-hand side UnsignedValue wrapper.
// \return \a true if the left value is greater or equal than the right value, \a false if not.
*/
template< typename T1    // Type of the left-hand side unsigned value
        , typename T2 >  // Type of the right-hand side unsigned value
inline bool operator>=( const UnsignedValue<T1>& lhs, const UnsignedValue<T2>& rhs )
{
   return lhs.get() >= rhs.get();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Global output operator for the UnsignedValue wrapper.
//
// \param os Reference to the output stream.
// \param uv Reference to a UnsignedValue object.
// \return The output stream.
*/
template< typename T >  // Type of the unsigned value
inline std::ostream& operator<<( std::ostream& os, const UnsignedValue<T>& uv )
{
   return os << uv.get();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Global input operator for the UnsignedValue wrapper.
//
// \param is Reference to the input stream.
// \param uv Reference to a UnsignedValue object.
// \return The input stream.
//
// The input operator guarantees that this object is not changed in the case of an input error.
// Only values suitable for the according built-in unsigned integral data type \a T are allowed.
// Otherwise, the input stream's position is returned to its previous position and the
// \a std::istream::failbit is set.
*/
template< typename T >  // Type of the unsigned value
std::istream& operator>>( std::istream& is, UnsignedValue<T>& uv )
{
   T tmp;
   const std::istream::pos_type pos( is.tellg() );

   // Skipping any leading whitespaces
   is >> std::ws;

   // Extracting the value
   if( is.peek() == '-' || !(is >> tmp) )
   {
      is.clear();
      is.seekg( pos );
      is.setstate( std::istream::failbit );
      return is;
   }

   // Transfering the input to the unsigned integer value
   uv = tmp;

   return is;
}
//*************************************************************************************************

} // namespace blaze

#endif
