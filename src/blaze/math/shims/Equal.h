//=================================================================================================
/*!
//  \file blaze/math/shims/Equal.h
//  \brief Header file for the equal shim
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

#ifndef _BLAZE_MATH_SHIMS_EQUAL_H_
#define _BLAZE_MATH_SHIMS_EQUAL_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cmath>
#include <blaze/math/Accuracy.h>
#include <blaze/math/Functions.h>
#include <blaze/util/Complex.h>


namespace blaze {

//=================================================================================================
//
//  EQUAL SHIM
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Generic equality check.
// \ingroup math_shims
//
// \param a First value/object.
// \param b Second value/object.
// \return \a true if the two values/objects are equal, \a false if not.
//
// The equal shim represents an abstract interface for testing two values/objects for equality.
// In case the two values/objects are equal, the function returns \a true, otherwise it returns
// \a false. Per default, the comparison of the two values/objects uses the equality operator
// operator==(). For built-in floating point data types a special comparison is selected that
// takes the limited machine accuracy into account.
*/
template< typename T1    // Type of the left-hand side value/object
        , typename T2 >  // Type of the right-hand side value/object
inline bool equal( const T1& a, const T2& b )
{
   return a == b;
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Equality check for two single precision floating point values.
// \ingroup math_shims
//
// \param a The left-hand side single precision floating point value.
// \param b The right-hand side single precision floating point value.
// \return \a true if the two values are equal, \a false if not.
//
// Equal function for the comparison of two single precision floating point numbers. Due to the
// limited machine accuracy, a direct comparison of two floating point numbers should be avoided.
// This function offers the possibility to compare two floating-point values with a certain
// accuracy margin.
//
// For more information on comparing float point numbers, see
//
//       http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
*/
inline bool equal( float a, float b )
{
   const float acc( static_cast<float>( accuracy ) );
   return ( std::fabs( a - b ) <= max( acc, acc * std::fabs( a ) ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Equality check for a single precision and a double precision floating point value.
// \ingroup math_shims
//
// \param a The left-hand side single precision floating point value.
// \param b The right-hand side double precision floating point value.
// \return \a true if the two values are equal, \a false if not.
//
// Equal function for the comparison of a single precision and a double precision floating point
// number. Due to the limited machine accuracy, a direct comparison of two floating point numbers
// should be avoided. This function offers the possibility to compare two floating-point values
// with a certain accuracy margin.
//
// For more information on comparing float point numbers, see
//
//       http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
*/
inline bool equal( float a, double b )
{
   return equal( a, static_cast<float>( b ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Equality check for a single precision and an extended precision floating point value.
// \ingroup math_shims
//
// \param a The left-hand side single precision floating point value.
// \param b The right-hand side extended precision floating point value.
// \return \a true if the two values are equal, \a false if not.
//
// Equal function for the comparison of a single precision and an extended precision floating point
// number. Due to the limited machine accuracy, a direct comparison of two floating point numbers
// should be avoided. This function offers the possibility to compare two floating-point values
// with a certain accuracy margin.
//
// For more information on comparing float point numbers, see
//
//       http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
*/
inline bool equal( float a, long double b )
{
   return equal( a, static_cast<float>( b ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Equality check for a double precision and a single precision floating point value.
// \ingroup math_shims
//
// \param a The left-hand side double precision floating point value.
// \param b The right-hand side single precision floating point value.
// \return \a true if the two values are equal, \a false if not.
//
// Equal function for the comparison of a double precision and a single precision floating point
// number. Due to the limited machine accuracy, a direct comparison of two floating point numbers
// should be avoided. This function offers the possibility to compare two floating-point values
// with a certain accuracy margin.
*/
inline bool equal( double a, float b )
{
   return equal( static_cast<float>( a ), b );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Equality check for two double precision floating point values.
// \ingroup math_shims
//
// \param a The left-hand side double precision floating point value.
// \param b The right-hand side double precision floating point value.
// \return \a true if the two values are equal, \a false if not.
//
// Equal function for the comparison of two double precision floating point numbers. Due to the
// limited machine accuracy, a direct comparison of two floating point numbers should be avoided.
// This function offers the possibility to compare two floating-point values with a certain
// accuracy margin.
//
// For more information on comparing float point numbers, see
//
//       http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
*/
inline bool equal( double a, double b )
{
   const double acc( static_cast<double>( accuracy ) );
   return ( std::fabs( a - b ) <= max( acc, acc * std::fabs( a ) ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Equality check for a double precision and an extended precision floating point value.
// \ingroup math_shims
//
// \param a The left-hand side double precision floating point value.
// \param b The right-hand side extended precision floating point value.
// \return \a true if the two values are equal, \a false if not.
//
// Equal function for the comparison of a double precision and an extended precision floating point
// number. Due to the limited machine accuracy, a direct comparison of two floating point numbers
// should be avoided. This function offers the possibility to compare two floating-point values
// with a certain accuracy margin.
//
// For more information on comparing float point numbers, see
//
//       http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
*/
inline bool equal( double a, long double b )
{
   return equal( a, static_cast<double>( b ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Equality check for an extended precision and a single precision floating point value.
// \ingroup math_shims
//
// \param a The left-hand side extended precision floating point value.
// \param b The right-hand side single precision floating point value.
// \return \a true if the two values are equal, \a false if not.
//
// Equal function for the comparison of an extended precision and a single precision floating point
// number. Due to the limited machine accuracy, a direct comparison of two floating point numbers
// should be avoided. This function offers the possibility to compare two floating-point values
// with a certain accuracy margin.
//
// For more information on comparing float point numbers, see
//
//       http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
*/
inline bool equal( long double a, float b )
{
   return equal( static_cast<float>( a ), b );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Equality check for an extended precision and a double precision floating point value.
// \ingroup math_shims
//
// \param a The left-hand side extended precision floating point value.
// \param b The right-hand side double precision floating point value.
// \return \a true if the two values are equal, \a false if not.
//
// Equal function for the comparison of an extended precision and a double precision floating point
// number. Due to the limited machine accuracy, a direct comparison of two floating point numbers
// should be avoided. This function offers the possibility to compare two floating-point values
// with a certain accuracy margin.
//
// For more information on comparing float point numbers, see
//
//       http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
*/
inline bool equal( long double a, double b )
{
   return equal( static_cast<double>( a ), b );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Equality check for two long double precision floating point values.
// \ingroup math_shims
//
// \param a The left-hand side extended precision floating point value.
// \param b The right-hand side extended precision floating point value.
// \return \a true if the two values are equal, \a false if not.
//
// Equal function for the comparison of two long double precision floating point numbers. Due
// to the limited machine accuracy, a direct comparison of two floating point numbers should be
// avoided. This function offers the possibility to compare two floating-point values with a
// certain accuracy margin.
//
// For more information on comparing float point numbers, see
//
//       http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
*/
inline bool equal( long double a, long double b )
{
   const long double acc( static_cast<long double>( accuracy ) );
   return ( std::fabs( a - b ) <= max( acc, acc * std::fabs( a ) ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Equality check for a complex and a scalar value.
// \ingroup math_shims
//
// \param a The left-hand side complex value.
// \param b The right-hand side scalar value.
// \return \a true if the two values are equal, \a false if not.
//
// Equal function for the comparison of a complex and a scalar value. The function compares the
// real part of the complex value with the scalar. In case these two values match and in case
// the imaginary part is zero, the function returns \a true. Otherwise it returns \a false.
*/
template< typename T1    // Type of the left-hand side complex value
        , typename T2 >  // Type of the right-hand side scalar value
inline bool equal( complex<T1> a, T2 b )
{
   return equal( real( a ), b ) && equal( imag( a ), T1() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Equality check for a scalar and a complex value.
// \ingroup math_shims
//
// \param a The left-hand side scalar value.
// \param b The right-hand side complex value.
// \return \a true if the two values are equal, \a false if not.
//
// Equal function for the comparison of a scalar and a complex value. The function compares the
// scalar with the real part of the complex value. In case these two values match and in case
// the imaginary part is zero, the function returns \a true. Otherwise it returns \a false.
*/
template< typename T1    // Type of the left-hand side scalar value
        , typename T2 >  // Type of the right-hand side complex value
inline bool equal( T1 a, complex<T2> b )
{
   return equal( a, real( b ) ) && equal( imag( b ), T2() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Equality check for two complex values.
// \ingroup math_shims
//
// \param a The left-hand side complex value.
// \param b The right-hand side complex value.
// \return \a true if the two values are equal, \a false if not.
//
// Equal function for the comparison of two complex numbers. Due to the limited machine accuracy,
// a direct comparison of two floating point numbers should be avoided. This function offers the
// possibility to compare two floating-point values with a certain accuracy margin.
*/
template< typename T1    // Type of the left-hand side complex value
        , typename T2 >  // Type of the right-hand side complex value
inline bool equal( complex<T1> a, complex<T2> b )
{
   return equal( real( a ), real( b ) ) && equal( imag( a ), imag( b ) );
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
