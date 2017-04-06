//=================================================================================================
/*!
//  \file blaze/math/Infinity.h
//  \brief Numerical infinity for built-in data types.
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

#ifndef _BLAZE_MATH_INFINITY_H_
#define _BLAZE_MATH_INFINITY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/constraints/Builtin.h>
#include <blaze/util/Limits.h>
#include <blaze/util/Types.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Negative infinity for built-in data types.
// \ingroup math
//
// The NegativeInfinity class is a wrapper class around the functionality of the blaze::Limits
// class to provide the possibility to assign negative infinity values to built-in data types.
// As negative infinity value, the largest possible negative value of the corresponding data
// type is used. In order to assign the negative infinity value, the NegativeInfinity class
// can be implicitly converted to all signed integral and floating point data types:
//
// <ul>
//    <li>integers</li>
//    <ul>
//       <li>signed char, char, wchar_t</li>
//       <li>short</li>
//       <li>int</li>
//       <li>long</li>
//       <li>ptrdiff_t (for certain 64-bit compilers)</li>
//    </ul>
//    <li>floating points</li>
//    <ul>
//       <li>float</li>
//       <li>double</li>
//       <li>long double</li>
//    </ul>
// </ul>
//
// \note The NegativeInfinity class is a helper class for the Infinity class. It cannot be
// instantiated on its own, but can only be used by the Infinity class.
*/
template< typename I >  // Positive infinity type
class NegativeInfinity
{
 public:
   //**Type definitions****************************************************************************
   typedef I  PositiveType;  //!< The positive infinity type.
   //**********************************************************************************************

 private:
   //**Constructor*********************************************************************************
   /*!\name Constructor */
   //@{
   explicit inline NegativeInfinity();
   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

 public:
   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Conversion operators************************************************************************
   /*!\name Conversion operators */
   //@{
   inline operator signed char() const;
   inline operator char()        const;
   inline operator wchar_t()     const;
   inline operator short()       const;
   inline operator int()         const;
   inline operator long()        const;
#if defined(_WIN64)
   inline operator ptrdiff_t()   const;
#endif
   inline operator float()       const;
   inline operator double()      const;
   inline operator long double() const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   template< typename T >
   inline bool equal( const T& rhs ) const;
   //@}
   //**********************************************************************************************

 private:
   //**Forbidden operations************************************************************************
   /*!\name Forbidden operations */
   //@{
   NegativeInfinity& operator=( const NegativeInfinity& ninf );  //!< Copy assignment operator (private & undefined)
   void* operator&() const;                                      //!< Address operator (private & undefined)
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   friend class Infinity;
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
/*!\brief The default constructor of the NegativeInfinity class.
*/
template< typename I >  // Positive infinity type
inline NegativeInfinity<I>::NegativeInfinity()
{}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion operator to the signed char built-in type.
//
// The conversion operator returns the smallest possible signed char value.
*/
template< typename I >  // Positive infinity type
inline NegativeInfinity<I>::operator signed char() const
{
   return Limits<signed char>::ninf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the char built-in type.
//
// The conversion operator returns the smallest possible char value.
*/
template< typename I >  // Positive infinity type
inline NegativeInfinity<I>::operator char() const
{
   return Limits<char>::ninf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the wchar_t built-in type.
//
// The conversion operator returns the smallest possible wchar_t value.
*/
template< typename I >  // Positive infinity type
inline NegativeInfinity<I>::operator wchar_t() const
{
   return Limits<wchar_t>::ninf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the short built-in type.
//
// The conversion operator returns the smallest possible short value.
*/
template< typename I >  // Positive infinity type
inline NegativeInfinity<I>::operator short() const
{
   return Limits<short>::ninf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the int built-in type.
//
// The conversion operator returns the smallest possible int value.
*/
template< typename I >  // Positive infinity type
inline NegativeInfinity<I>::operator int() const
{
   return Limits<int>::ninf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the long built-in type.
//
// The conversion operator returns the smallest possible long value.
*/
template< typename I >  // Positive infinity type
inline NegativeInfinity<I>::operator long() const
{
   return Limits<long>::ninf();
}
//*************************************************************************************************


//*************************************************************************************************
#if defined(_WIN64)
/*!\brief Conversion operator to the ptrdiff_t built-in type.
//
// The conversion operator returns the smallest possible ptrdiff_t value.
*/
template< typename I >  // Positive infinity type
inline NegativeInfinity<I>::operator ptrdiff_t() const
{
   return Limits<ptrdiff_t>::ninf();
}
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the float built-in type.
//
// The conversion operator returns the smallest possible float value.
*/
template< typename I >  // Positive infinity type
inline NegativeInfinity<I>::operator float() const
{
   return Limits<float>::ninf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the double built-in type.
//
// The conversion operator returns the smallest possible double value.
*/
template< typename I >  // Positive infinity type
inline NegativeInfinity<I>::operator double() const
{
   return Limits<double>::ninf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the long double built-in type.
//
// The conversion operator returns the smallest possible long double value.
*/
template< typename I >  // Positive infinity type
inline NegativeInfinity<I>::operator long double() const
{
   return Limits<long double>::ninf();
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Equality comparison to a built-in data type.
//
// This function compares built-in data types with their largest possible value. The function
// only works for built-in data types. The attempt to compare user-defined class types will
// result in a compile time error.
*/
template< typename I >  // Positive infinity type
template< typename T >  // Built-in data type
inline bool NegativeInfinity<I>::equal( const T& rhs ) const
{
   BLAZE_CONSTRAINT_MUST_BE_BUILTIN_TYPE( T );
   return Limits<T>::ninf() == rhs;
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name NegativeInfinity operators */
//@{
template< typename I1, typename I2 >
inline bool operator==( const NegativeInfinity<I1>& lhs, const NegativeInfinity<I2>& rhs );

template< typename I, typename T >
inline bool operator==( const NegativeInfinity<I>& lhs, const T& rhs );

template< typename I, typename T >
inline bool operator==( const T& lhs, const NegativeInfinity<I>& rhs );

template< typename I1, typename I2 >
inline bool operator!=( const NegativeInfinity<I1>& lhs, const NegativeInfinity<I2>& rhs );

template< typename I, typename T >
inline bool operator!=( const NegativeInfinity<I>& lhs, const T& rhs );

template< typename I, typename T >
inline bool operator!=( const T& lhs, const NegativeInfinity<I>& rhs );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between two NegativeInfinity objects.
// \ingroup math
//
// \param lhs The left-hand side NegativeInfinity object.
// \param rhs The right-hand side NegativeInfinity object.
// \return \a true.
*/
template< typename I1    // Left-hand side positive infinity type
        , typename I2 >  // Right-hand side positive infinity type
inline bool operator==( const NegativeInfinity<I1>& lhs, const NegativeInfinity<I2>& rhs )
{
   UNUSED_PARAMETER( lhs, rhs );
   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between an NegativeInfinity object and a built-in data type.
// \ingroup math
//
// \param lhs The left-hand side NegativeInfinity object.
// \param rhs The right-hand side built-in data value.
// \return \a true if the built-in data value is negative infinity, \a false if not.
//
// This operator works only for built-in data types. The attempt to compare user-defined class
// types will result in a compile time error.
*/
template< typename I    // Positive infinity type
        , typename T >  // Built-in data type
inline bool operator==( const NegativeInfinity<I>& lhs, const T& rhs )
{
   return lhs.equal( rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between a built-in data type and an NegativeInfinity object.
// \ingroup math
//
// \param lhs The left-hand side built-in data value.
// \param rhs The right-hand side NegativeInfinity object.
// \return \a true if the built-in data value is negative infinity, \a false if not.
//
// This operator works only for built-in data types. The attempt to compare user-defined class
// types will result in a compile time error.
*/
template< typename I    // Positive infinity type
        , typename T >  // Built-in data type
inline bool operator==( const T& lhs, const NegativeInfinity<I>& rhs )
{
   return rhs.equal( lhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between two NegativeInfinity objects.
// \ingroup math
//
// \param lhs The left-hand side NegativeInfinity object.
// \param rhs The right-hand side NegativeInfinity object.
// \return \a false.
*/
template< typename I1    // Left-hand side positive infinity type
        , typename I2 >  // Right-hand side positive infinity type
inline bool operator!=( const NegativeInfinity<I1>& lhs, const NegativeInfinity<I2>& rhs )
{
   UNUSED_PARAMETER( lhs, rhs );
   return false;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between an NegativeInfinity object and a built-in data type.
// \ingroup math
//
// \param lhs The left-hand side NegativeInfinity object.
// \param rhs The right-hand side built-in data value.
// \return \a true if the built-in data value is not negative infinity, \a false if it is.
//
// This operator works only for built-in data types. The attempt to compare user-defined class
// types will result in a compile time error.
*/
template< typename I    // Positive infinity type
        , typename T >  // Built-in data type
inline bool operator!=( const NegativeInfinity<I>& lhs, const T& rhs )
{
   return !lhs.equal( rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between a built-in data type and an NegativeInfinity object.
// \ingroup math
//
// \param lhs The left-hand side built-in data value.
// \param rhs The right-hand side NegativeInfinity object.
// \return \a true if the built-in data value is not negative infinity, \a false if it is.
//
// This operator works only for built-in data types. The attempt to compare user-defined class
// types will result in a compile time error.
*/
template< typename I    // Positive infinity type
        , typename T >  // Built-in data type
inline bool operator!=( const T& lhs, const NegativeInfinity<I>& rhs )
{
   return !rhs.equal( lhs );
}
//*************************************************************************************************








//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Positive infinity for built-in data types.
// \ingroup math
//
// The Infinity class is a wrapper class around the functionality of the blaze::Limits class
// to provide the possiblity to assign a positive infinity value to built-in data types.
// As positive infinity value, the largest possible positive value of the corresponding
// data type is used. In order to assign the positive infinity value, the Infinity class
// can be implicitly converted to the following 13 built-in integral and floating point
// data types:
//
// <ul>
//    <li>integers</li>
//    <ul>
//       <li>unsigned char, signed char, char, wchar_t</li>
//       <li>unsigned short, short</li>
//       <li>unsigned int, int</li>
//       <li>unsigned long, long</li>
//    </ul>
//    <li>floating points</li>
//    <ul>
//       <li>float</li>
//       <li>double</li>
//       <li>long double</li>
//    </ul>
// </ul>
//
// In order to be able to assign infinity values, the global Infinity instance blaze::inf
// is provided, which can be used wherever a built-in data type is required.

   \code
   int i    =  inf;  // Assigns a positive infinity value
   double d = -inf;  // Assigns a negative infinity value
   ...
   \endcode
*/
class Infinity
{
 public:
   //**Type definitions****************************************************************************
   typedef NegativeInfinity<Infinity>  NegativeType;  //!< The negative infinity type.
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\name Constructor */
   //@{
   explicit inline Infinity();
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Conversion operators************************************************************************
   /*!\name Conversion operators */
   //@{
   inline operator unsigned char()  const;
   inline operator signed char()    const;
   inline operator char()           const;
   inline operator wchar_t()        const;
   inline operator unsigned short() const;
   inline operator short()          const;
   inline operator unsigned int()   const;
   inline operator int()            const;
   inline operator unsigned long()  const;
   inline operator long()           const;
#if defined(_WIN64)
   inline operator size_t()         const;
   inline operator ptrdiff_t()      const;
#endif
   inline operator float()          const;
   inline operator double()         const;
   inline operator long double()    const;
   //@}
   //**********************************************************************************************

   //**Arithmetic operators************************************************************************
   /*!\name Arithmetic operators */
   //@{
   inline const Infinity&    operator+() const;
   inline const NegativeType operator-() const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   template< typename T >
   inline bool equal( const T& rhs ) const;
   //@}
   //**********************************************************************************************

 private:
   //**Forbidden operations************************************************************************
   /*!\name Forbidden operations */
   //@{
   Infinity& operator=( const Infinity& inf );  //!< Copy assignment operator (private & undefined)
   void* operator&() const;                     //!< Address operator (private & undefined)
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
/*!\brief The default constructor of the Infinity class.
*/
inline Infinity::Infinity()
{}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion operator to the unsigned char built-in type.
//
// The conversion operator returns the largest possible unsigned char value.
*/
inline Infinity::operator unsigned char() const
{
   return Limits<unsigned char>::inf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the char built-in type.
//
// The conversion operator returns the largest possible char value.
*/
inline Infinity::operator char() const
{
   return Limits<char>::inf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the signed char built-in type.
//
// The conversion operator returns the largest possible signed char value.
*/
inline Infinity::operator signed char() const
{
   return Limits<signed char>::inf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the wchar_t built-in type.
//
// The conversion operator returns the largest possible wchar_t value.
*/
inline Infinity::operator wchar_t() const
{
   return Limits<wchar_t>::inf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the unsigned short built-in type.
//
// The conversion operator returns the largest possible unsigned short value.
*/
inline Infinity::operator unsigned short() const
{
   return Limits<unsigned short>::inf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the short built-in type.
//
// The conversion operator returns the largest possible short value.
*/
inline Infinity::operator short() const
{
   return Limits<short>::inf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the unsigned int built-in type.
//
// The conversion operator returns the largest possible unsigned int value.
*/
inline Infinity::operator unsigned int() const
{
   return Limits<unsigned int>::inf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the int built-in type.
//
// The conversion operator returns the largest possible int value.
*/
inline Infinity::operator int() const
{
   return Limits<int>::inf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the unsigned long built-in type.
//
// The conversion operator returns the largest possible unsigned long value.
*/
inline Infinity::operator unsigned long() const
{
   return Limits<unsigned long>::inf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the long built-in type.
//
// The conversion operator returns the largest possible long value.
*/
inline Infinity::operator long() const
{
   return Limits<long>::inf();
}
//*************************************************************************************************


//*************************************************************************************************
#if defined(_WIN64)
/*!\brief Conversion operator to the size_t built-in type.
//
// The conversion operator returns the largest possible size_t value.
*/
inline Infinity::operator size_t() const
{
   return Limits<size_t>::inf();
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if defined(_WIN64)
/*!\brief Conversion operator to the ptrdiff_t built-in type.
//
// The conversion operator returns the largest possible ptrdiff_t value.
*/
inline Infinity::operator ptrdiff_t() const
{
   return Limits<ptrdiff_t>::inf();
}
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the float built-in type.
//
// The conversion operator returns the largest possible float value.
*/
inline Infinity::operator float() const
{
   return Limits<float>::inf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the double built-in type.
//
// The conversion operator returns the largest possible double value.
*/
inline Infinity::operator double() const
{
   return Limits<double>::inf();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to the long double built-in type.
//
// The conversion operator returns the largest possible long double value.
*/
inline Infinity::operator long double() const
{
   return Limits<long double>::inf();
}
//*************************************************************************************************




//=================================================================================================
//
//  ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the positive infinity value for all built-in data types.
//
// \return The positive infinity value.
*/
inline const Infinity& Infinity::operator+() const
{
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the negative infinity value for all built-in data types.
//
// \return The negative infinity value.
*/
inline const Infinity::NegativeType Infinity::operator-() const
{
   return NegativeType();
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Equality comparison to a built-in data type.
//
// This function compares built-in data types with their largest possible value. The function
// only works for built-in data types. The attempt to compare user-defined class types will
// result in a compile time error.
*/
template< typename T >
inline bool Infinity::equal( const T& rhs ) const
{
   BLAZE_CONSTRAINT_MUST_BE_BUILTIN_TYPE( T );
   return Limits<T>::inf() == rhs;
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Infinity operators */
//@{
inline bool operator==( const Infinity& lhs, const Infinity& rhs );

template< typename I >
inline bool operator==( const Infinity& lhs, const NegativeInfinity<I>& rhs );

template< typename I >
inline bool operator==( const NegativeInfinity<I>& lhs, const Infinity& rhs );

template< typename T >
inline bool operator==( const Infinity& lhs, const T& rhs );

template< typename T >
inline bool operator==( const T& lhs, const Infinity& rhs );

inline bool operator!=( const Infinity& lhs, const Infinity& rhs );

template< typename I >
inline bool operator!=( const Infinity& lhs, const NegativeInfinity<I>& rhs );

template< typename I >
inline bool operator!=( const NegativeInfinity<I>& lhs, const Infinity& rhs );

template< typename T >
inline bool operator!=( const Infinity& lhs, const T& rhs );

template< typename T >
inline bool operator!=( const T& lhs, const Infinity& rhs );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between two Infinity objects.
// \ingroup math
//
// \param lhs The left-hand side Infinity object.
// \param rhs The right-hand side Infinity object.
// \return \a true.
*/
inline bool operator==( const Infinity& lhs, const Infinity& rhs )
{
   UNUSED_PARAMETER( lhs, rhs );
   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between an Infinity object and a NegativeInfinity object.
// \ingroup math
//
// \param lhs The left-hand side Infinity object.
// \param rhs The right-hand side NegativeInfinity object.
// \return \a false.
*/
template< typename I >  // Positive infinity type
inline bool operator==( const Infinity& lhs, const NegativeInfinity<I>& rhs )
{
   UNUSED_PARAMETER( lhs, rhs );
   return false;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between a NegativeInfinity object and an Infinity object.
// \ingroup math
//
// \param rhs The left-hand side NegativeInfinity object.
// \param lhs The right-hand side Infinity object.
// \return \a false.
*/
template< typename I >  // Positive infinity type
inline bool operator==( const NegativeInfinity<I>& lhs, const Infinity& rhs )
{
   UNUSED_PARAMETER( lhs, rhs );
   return false;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between an Infinity object and a built-in data type.
// \ingroup math
//
// \param lhs The left-hand side Infinity object.
// \param rhs The right-hand side built-in data value.
// \return \a true if the built-in data value is infinity, \a false if not.
//
// This operator works only for built-in data types. The attempt to compare user-defined class
// types will result in a compile time error.
*/
template< typename T >
inline bool operator==( const Infinity& lhs, const T& rhs )
{
   return lhs.equal( rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between a built-in data type and an Infinity object.
// \ingroup math
//
// \param lhs The left-hand side built-in data value.
// \param rhs The right-hand side Infinity object.
// \return \a true if the built-in data value is infinity, \a false if not.
//
// This operator works only for built-in data types. The attempt to compare user-defined class
// types will result in a compile time error.
*/
template< typename T >
inline bool operator==( const T& lhs, const Infinity& rhs )
{
   return rhs.equal( lhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between two Infinity objects.
// \ingroup math
//
// \param lhs The left-hand side Infinity object.
// \param rhs The right-hand side Infinity object.
// \return \a false.
*/
inline bool operator!=( const Infinity& lhs, const Infinity& rhs )
{
   UNUSED_PARAMETER( lhs, rhs );
   return false;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between an Infinity object and a NegativeInfinity object.
// \ingroup math
//
// \param lhs The left-hand side Infinity object.
// \param rhs The right-hand side NegativeInfinity object.
// \return \a true.
*/
template< typename I >  // Positive infinity type
inline bool operator!=( const Infinity& lhs, const NegativeInfinity<I>& rhs )
{
   UNUSED_PARAMETER( lhs, rhs );
   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between a NegativeInfinity object and an Infinity object.
// \ingroup math
//
// \param rhs The left-hand side NegativeInfinity object.
// \param lhs The right-hand side Infinity object.
// \return \a true.
*/
template< typename I >  // Positive infinity type
inline bool operator!=( const NegativeInfinity<I>& lhs, const Infinity& rhs )
{
   UNUSED_PARAMETER( lhs, rhs );
   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between an Infinity object and a built-in data type.
// \ingroup math
//
// \param lhs The left-hand side Infinity object.
// \param rhs The right-hand side built-in data value.
// \return \a true if the built-in data value is not infinity, \a false if it is.
//
// This operator works only for built-in data types. The attempt to compare user-defined class
// types will result in a compile time error.
*/
template< typename T >
inline bool operator!=( const Infinity& lhs, const T& rhs )
{
   return !lhs.equal( rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between a built-in data type and an Infinity object.
// \ingroup math
//
// \param lhs The left-hand side built-in data value.
// \param rhs The right-hand side Infinity object.
// \return \a true if the built-in data value is not infinity, \a false if it is.
//
// This operator works only for built-in data types. The attempt to compare user-defined class
// types will result in a compile time error.
*/
template< typename T >
inline bool operator!=( const T& lhs, const Infinity& rhs )
{
   return !rhs.equal( lhs );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL INFINITY VALUE
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Global Infinity instance.
// \ingroup math
//
// The blaze::inf instance can be used wherever a built-in data type is expected. It is implicitly
// converted to the corresponding built-in data type and represents its largest possible data
// value.
*/
const Infinity inf;
//*************************************************************************************************

} // namespace blaze

#endif
