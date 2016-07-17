//=================================================================================================
/*!
//  \file blaze/math/Functions.h
//  \brief Header file for mathematical functions
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

#ifndef _BLAZE_MATH_FUNCTIONS_H_
#define _BLAZE_MATH_FUNCTIONS_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cmath>
#include <blaze/system/Inline.h>
#include <blaze/util/constraints/Builtin.h>
#include <blaze/util/constraints/Integral.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/All.h>
#include <blaze/util/typetraits/CommonType.h>
#include <blaze/util/typetraits/IsBuiltin.h>
#include <blaze/util/typetraits/IsFloatingPoint.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsSigned.h>


namespace blaze {

//=================================================================================================
//
//  MATHEMATICAL UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Mathematical utility functions */
//@{
template< typename T >
inline constexpr int sign( T a ) noexcept;

template< typename T >
inline size_t digits( T a ) noexcept;

template< typename T1, typename T2 >
BLAZE_ALWAYS_INLINE constexpr CommonType_<T1,T2>
   min( const T1& a, const T2& b ) noexcept( All<IsNumeric,T1,T2>::value );

template< typename T1, typename T2, typename... Ts >
BLAZE_ALWAYS_INLINE constexpr CommonType_<T1,T2,Ts...>
   min( const T1& a, const T2& b, const Ts&... args ) noexcept( All<IsNumeric,T1,T2,Ts...>::value );

template< typename T1, typename T2 >
BLAZE_ALWAYS_INLINE constexpr CommonType_<T1,T2>
   max( const T1& a, const T2& b ) noexcept( All<IsNumeric,T1,T2>::value );

template< typename T1, typename T2, typename... Ts >
BLAZE_ALWAYS_INLINE constexpr CommonType_<T1,T2,Ts...>
   max( const T1& a, const T2& b, const Ts&... args ) noexcept( All<IsNumeric,T1,T2,Ts...>::value );

template< typename T >
BLAZE_ALWAYS_INLINE T round( T a ) noexcept;

template< typename T >
BLAZE_ALWAYS_INLINE constexpr T nextMultiple( T value, T factor ) noexcept;

template< typename T1, typename T2 >
BLAZE_ALWAYS_INLINE constexpr bool lessThan( T1 a, T2 b )
   noexcept( IsBuiltin< CommonType_<T1,T2> >::value );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Sign function.
// \ingroup math
//
// \param a The given value.
// \return 1 if the value is greater than zero, 0 if it is zero, and -1 if it is smaller than zero.
//
// The sign function only works for built-in data types. The attempt to use any user-defined class
// type will result in a compile time error.
*/
template< typename T >
inline constexpr int sign( T a ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_BUILTIN_TYPE( T );

   return ( IsSigned<T>::value || IsFloatingPoint<T>::value )
          ?( T(0) < a ) - ( a < T(0) )
          :( T(0) < a );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of valid digits of an integral value.
// \ingroup math
//
// \param a The integral value.
// \return The number of valid digits.
//
// This function counts the number of valid digits in the given integral value.

   \code
   digits( 100   );  // Returns 3
   digits( 12345 );  // Returns 5
   digits( 0     );  // Returns 0
   \endcode

// The digits function only works for integral built-in data types. The attempt to use any
// other type will result in a compile time error.
*/
template< typename T >
inline size_t digits( T a ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_INTEGRAL_TYPE( T );

   size_t count( 0 );

   while( a != 0 ) {
      a /= 10;
      ++count;
   }

   return count;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Minimum function for two data values.
// \ingroup math
//
// \param a The first value.
// \param b The second value.
// \return The minimum of the two values.
//
// This function returns the minimum of the two given data values. The return type of the function
// is determined by the data types of the given arguments (for further detail see the CommonType
// class description).
*/
template< typename T1, typename T2 >
BLAZE_ALWAYS_INLINE constexpr CommonType_<T1,T2>
   min( const T1& a, const T2& b ) noexcept( All<IsNumeric,T1,T2>::value )
{
   return ( a < b )?( a ):( b );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Minimum function for at least three data values.
// \ingroup math
//
// \param a The first value.
// \param b The second value.
// \param args The pack of additional values.
// \return The minimum of the given values.
//
// This function returns the minimum of the given data values. The return type of the function
// is determined by the data types of the given arguments (for further detail see the CommonType
// class description).
*/
template< typename T1, typename T2, typename... Ts >
BLAZE_ALWAYS_INLINE constexpr CommonType_<T1,T2,Ts...>
   min( const T1& a, const T2& b, const Ts&... args ) noexcept( All<IsNumeric,T1,T2,Ts...>::value )
{
   return min( a, min( b, args... ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Maximum function for two data values.
// \ingroup math
//
// \param a The first value.
// \param b The second value.
// \return The maximum of the two values.
//
// This function returns the maximum of the two given data values. The return type of the function
// is determined by the data types of the given arguments (for further detail see the CommonType
// class description).
*/
template< typename T1, typename T2 >
BLAZE_ALWAYS_INLINE constexpr CommonType_<T1,T2>
   max( const T1& a, const T2& b ) noexcept( All<IsNumeric,T1,T2>::value )
{
   return ( a < b )?( b ):( a );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Maximum function for at least three data values.
// \ingroup math
//
// \param a The first value.
// \param b The second value.
// \param args The pack of additional values.
// \return The maximum of the given values.
//
// This function returns the maximum of the given data values. The return type of the function
// is determined by the data types of the given arguments (for further detail see the CommonType
// class description).
*/
template< typename T1, typename T2, typename... Ts >
BLAZE_ALWAYS_INLINE constexpr CommonType_<T1,T2,Ts...>
   max( const T1& a, const T2& b, const Ts&... args ) noexcept( All<IsNumeric,T1,T2,Ts...>::value )
{
   return max( a, max( b, args... ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Rounds the given input value.
// \ingroup math
//
// \param a Value to be rounded.
// \return The rounded value.
//
// This function rounds the given input value. In case the first digit after the comma
// is smaller than five the value is rounded down. Otherwise it is rounded up. Note that
// this function only works for integral and floating point types. The attempt to use the
// function for any other type will result in a compile time error.
*/
template< typename T >
BLAZE_ALWAYS_INLINE T round( T a ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_INTEGRAL_TYPE( T );
   return a;
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Rounds the given single precision floating point value.
// \ingroup math
//
// \param a Value to be rounded.
// \return The rounded value.
//
// This function rounds the given single precision floating point value. In case the first
// digit after the comma is smaller than five the value is rounded down. Otherwise it is
// rounded up.
*/
BLAZE_ALWAYS_INLINE float round( float a ) noexcept
{
   return std::floor( a + 0.5F );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Rounds the given double precision floating point value.
// \ingroup math
//
// \param a Value to be rounded.
// \return The rounded value.
//
// This function rounds the given double precision floating point value. In case the first
// digit after the comma is smaller than five the value is rounded down. Otherwise it is
// rounded up.
*/
BLAZE_ALWAYS_INLINE double round( double a ) noexcept
{
   return std::floor( a + 0.5 );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Rounds the given long double precision floating point value.
// \ingroup math
//
// \param a Value to be rounded.
// \return The rounded value.
//
// This function rounds the given long double precision floating point value. In case the
// first digit after the comma is smaller than five the value is rounded down. Otherwise
// it is rounded up.
*/
BLAZE_ALWAYS_INLINE long double round( long double a ) noexcept
{
   return std::floor( a + 0.5L );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Rounds up an integral value to the next multiple of a given factor.
// \ingroup math
//
// \param value The integral value to be rounded up \f$[1..\infty)\f$.
// \param factor The factor of the multiple \f$[1..\infty)\f$.
// \return The multiple of the given factor.
//
// This function rounds up the given integral value to the next multiple of the given factor.
// In case the integral value is already a multiple of the given factor, the value itself is
// returned. Note that both \a value and \a factor are expected to be positive integrals. In
// case any of them is negative, the function returns 0. Note that the attempt to use the
// function with non-integral types results in a compilation error!
*/
template< typename T >
BLAZE_ALWAYS_INLINE constexpr T nextMultiple( T value, T factor ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_INTEGRAL_TYPE( T );

   return ( value > T(0) && factor > T(0) )
          ?( value + ( factor - ( value % factor ) ) % factor )
          :( T(0) );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default less-than comparison for any data type.
// \ingroup math
//
// \param a First value.
// \param b Second value.
// \return \a true if the first value is smaller than the second, \a false if not.
//
// Default implementation of a less-than comparison of two data values.
*/
template< typename T >
BLAZE_ALWAYS_INLINE constexpr bool lessThan_backend( const T& a, const T& b )
   noexcept( IsBuiltin<T>::value )
{
   return a < b;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Less-than comparison for single precision floating point values.
// \ingroup math
//
// \param a First value.
// \param b Second value.
// \return \a true if the first value is smaller than the second, \a false if not.
//
// Less-than function for the comparison of two single precision floating point numbers. Due
// to the limited machine accuracy, a direct comparison of two floating point numbers should
// be avoided. This functions offers the possibility to compare two floating-point values with
// a certain accuracy margin.
*/
BLAZE_ALWAYS_INLINE constexpr bool lessThan_backend( float a, float b ) noexcept
{
   return ( b - a ) > 1E-8F;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Less-than comparison for double precision floating point values.
// \ingroup math
//
// \param a First value.
// \param b Second value.
// \return \a true if the first value is smaller than the second, \a false if not.
//
// Less-than function for the comparison of two double precision floating point numbers. Due
// to the limited machine accuracy, a direct comparison of two floating point numbers should
// be avoided. This functions offers the possibility to compare two floating-point values with
// a certain accuracy margin.
*/
BLAZE_ALWAYS_INLINE constexpr bool lessThan_backend( double a, double b ) noexcept
{
   return ( b - a ) > 1E-8;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Less-than comparison for long double precision floating point values.
// \ingroup math
//
// \param a First value.
// \param b Second value.
// \return \a true if the first value is smaller than the second, \a false if not.
//
// Less-than function for the comparison of two long double precision floating point numbers. Due
// to the limited machine accuracy, a direct comparison of two floating point numbers should be
// avoided. This functions offers the possibility to compare two floating-point values with a
// certain accuracy margin.
*/
BLAZE_ALWAYS_INLINE constexpr bool lessThan_backend( long double a, long double b ) noexcept
{
   return ( b - a ) > 1E-10;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Generic less-than comparison.
// \ingroup math
//
// \param a First value.
// \param b Second value.
// \return \a true if the first value is smaller than the second, \a false if not.
//
// Generic less-than comparison between to numeric values. Depending on the types of the
// two arguments, a special comparison for floating point values is selected that takes
// the limited machine accuracy into account.
*/
template< typename T1, typename T2 >
BLAZE_ALWAYS_INLINE constexpr bool lessThan( const T1& a, const T2& b )
   noexcept( IsBuiltin< CommonType_<T1,T2> >::value )
{
   return lessThan_backend< CommonType_<T1,T2> >( a, b );
}
//*************************************************************************************************

} // namespace blaze

#endif
