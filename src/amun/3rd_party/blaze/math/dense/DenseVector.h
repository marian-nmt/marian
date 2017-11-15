//=================================================================================================
/*!
//  \file blaze/math/dense/DenseVector.h
//  \brief Header file for utility functions for dense vectors
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

#ifndef _BLAZE_MATH_DENSE_DENSEVECTOR_H_
#define _BLAZE_MATH_DENSE_DENSEVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cmath>
#include <blaze/math/Aliases.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/SparseVector.h>
#include <blaze/math/Functions.h>
#include <blaze/math/shims/Equal.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsDivisor.h>
#include <blaze/math/shims/IsNaN.h>
#include <blaze/math/shims/Sqrt.h>
#include <blaze/math/shims/Square.h>
#include <blaze/math/TransposeFlag.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Numeric.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/RemoveReference.h>


namespace blaze {

//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DenseVector operators */
//@{
template< typename T1, bool TF1, typename T2, bool TF2 >
inline bool operator==( const DenseVector<T1,TF1>& lhs, const DenseVector<T2,TF2>& rhs );

template< typename T1, bool TF1, typename T2, bool TF2 >
inline bool operator==( const DenseVector<T1,TF1>& lhs, const SparseVector<T2,TF2>& rhs );

template< typename T1, bool TF1, typename T2, bool TF2 >
inline bool operator==( const SparseVector<T1,TF1>& lhs, const DenseVector<T2,TF2>& rhs );

template< typename T1, typename T2, bool TF >
inline EnableIf_<IsNumeric<T2>, bool > operator==( const DenseVector<T1,TF>& vec, T2 scalar );

template< typename T1, typename T2, bool TF >
inline EnableIf_<IsNumeric<T1>, bool > operator==( T1 scalar, const DenseVector<T2,TF>& vec );

template< typename T1, bool TF1, typename T2, bool TF2 >
inline bool operator!=( const DenseVector<T1,TF1>& lhs, const DenseVector<T2,TF2>& rhs );

template< typename T1, bool TF1, typename T2, bool TF2 >
inline bool operator!=( const DenseVector<T1,TF1>& lhs, const SparseVector<T2,TF2>& rhs );

template< typename T1, bool TF1, typename T2, bool TF2 >
inline bool operator!=( const SparseVector<T1,TF1>& lhs, const DenseVector<T2,TF2>& rhs );

template< typename T1, typename T2, bool TF >
inline EnableIf_<IsNumeric<T2>, bool > operator!=( const DenseVector<T1,TF>& vec, T2 scalar );

template< typename T1, typename T2, bool TF >
inline EnableIf_<IsNumeric<T1>, bool > operator!=( T1 scalar, const DenseVector<T2,TF>& vec );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of two dense vectors.
// \ingroup dense_vector
//
// \param lhs The left-hand side dense vector for the comparison.
// \param rhs The right-hand side dense vector for the comparison.
// \return \a true if the two vectors are equal, \a false if not.
*/
template< typename T1  // Type of the left-hand side dense vector
        , bool TF1     // Transpose flag of the left-hand side dense vector
        , typename T2  // Type of the right-hand side dense vector
        , bool TF2 >   // Transpose flag of the right-hand side dense vector
inline bool operator==( const DenseVector<T1,TF1>& lhs, const DenseVector<T2,TF2>& rhs )
{
   typedef CompositeType_<T1>  CT1;
   typedef CompositeType_<T2>  CT2;

   // Early exit in case the vector sizes don't match
   if( (~lhs).size() != (~rhs).size() ) return false;

   // Evaluation of the two dense vector operands
   CT1 a( ~lhs );
   CT2 b( ~rhs );

   // In order to compare the two vectors, the data values of the lower-order data
   // type are converted to the higher-order data type within the equal function.
   for( size_t i=0; i<a.size(); ++i )
      if( !equal( a[i], b[i] ) ) return false;
   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a dense vector and a sparse vector.
// \ingroup dense_vector
//
// \param lhs The left-hand side dense vector for the comparison.
// \param rhs The right-hand side sparse vector for the comparison.
// \return \a true if the two vectors are equal, \a false if not.
*/
template< typename T1  // Type of the left-hand side dense vector
        , bool TF1     // Transpose flag of the left-hand side dense vector
        , typename T2  // Type of the right-hand side sparse vector
        , bool TF2 >   // Transpose flag of the right-hand side sparse vector
inline bool operator==( const DenseVector<T1,TF1>& lhs, const SparseVector<T2,TF2>& rhs )
{
   typedef CompositeType_<T1>  CT1;
   typedef CompositeType_<T2>  CT2;
   typedef ConstIterator_< RemoveReference_<CT2> >  ConstIterator;

   // Early exit in case the vector sizes don't match
   if( (~lhs).size() != (~rhs).size() ) return false;

   // Evaluation of the dense vector and sparse vector operand
   CT1 a( ~lhs );
   CT2 b( ~rhs );

   // In order to compare the two vectors, the data values of the lower-order data
   // type are converted to the higher-order data type within the equal function.
   size_t i( 0 );

   for( ConstIterator element=b.begin(); element!=b.end(); ++element, ++i ) {
      for( ; i<element->index(); ++i ) {
         if( !isDefault( a[i] ) ) return false;
      }
      if( !equal( element->value(), a[i] ) ) return false;
   }
   for( ; i<a.size(); ++i ) {
      if( !isDefault( a[i] ) ) return false;
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a sparse vector and a dense vector.
// \ingroup dense_vector
//
// \param lhs The left-hand side sparse vector for the comparison.
// \param rhs The right-hand side dense vector for the comparison.
// \return \a true if the two vectors are equal, \a false if not.
*/
template< typename T1  // Type of the left-hand side sparse vector
        , bool TF1     // Transpose flag of the left-hand side sparse vector
        , typename T2  // Type of the right-hand side dense vector
        , bool TF2 >   // Transpose flag of the right-hand side dense vector
inline bool operator==( const SparseVector<T1,TF1>& lhs, const DenseVector<T2,TF2>& rhs )
{
   return ( rhs == lhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a dense vector and a scalar value.
// \ingroup dense_vector
//
// \param vec The left-hand side dense vector for the comparison.
// \param scalar The right-hand side scalar value for the comparison.
// \return \a true if all elements of the vector are equal to the scalar, \a false if not.
//
// If all values of the vector are equal to the scalar value, the equality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1  // Type of the left-hand side dense vector
        , typename T2  // Type of the right-hand side scalar
        , bool TF >    // Transpose flag
inline EnableIf_<IsNumeric<T2>, bool > operator==( const DenseVector<T1,TF>& vec, T2 scalar )
{
   typedef CompositeType_<T1>  CT1;

   // Evaluation of the dense vector operand
   CT1 a( ~vec );

   // In order to compare the vector and the scalar value, the data values of the lower-order
   // data type are converted to the higher-order data type within the equal function.
   for( size_t i=0; i<a.size(); ++i )
      if( !equal( a[i], scalar ) ) return false;
   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality operator for the comparison of a scalar value and a dense vector.
// \ingroup dense_vector
//
// \param scalar The left-hand side scalar value for the comparison.
// \param vec The right-hand side dense vector for the comparison.
// \return \a true if all elements of the vector are equal to the scalar, \a false if not.
//
// If all values of the vector are equal to the scalar value, the equality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1  // Type of the left-hand side scalar
        , typename T2  // Type of the right-hand side dense vector
        , bool TF >    // Transpose flag
inline EnableIf_<IsNumeric<T1>, bool > operator==( T1 scalar, const DenseVector<T2,TF>& vec )
{
   return ( vec == scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of two dense vectors.
// \ingroup dense_vector
//
// \param lhs The left-hand side dense vector for the comparison.
// \param rhs The right-hand side dense vector for the comparison.
// \return \a true if the two vectors are not equal, \a false if they are equal.
*/
template< typename T1  // Type of the left-hand side dense vector
        , bool TF1     // Transpose flag of the left-hand side dense vector
        , typename T2  // Type of the right-hand side dense vector
        , bool TF2 >   // Transpose flag of the right-hand side dense vector
inline bool operator!=( const DenseVector<T1,TF1>& lhs, const DenseVector<T2,TF2>& rhs )
{
   return !( lhs == rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of a dense vector and a sparse vector.
// \ingroup dense_vector
//
// \param lhs The left-hand side dense vector for the comparison.
// \param rhs The right-hand side sparse vector for the comparison.
// \return \a true if the two vectors are not equal, \a false if they are equal.
*/
template< typename T1  // Type of the left-hand side dense vector
        , bool TF1     // Transpose flag of the left-hand side dense vector
        , typename T2  // Type of the right-hand side sparse vector
        , bool TF2 >   // Transpose flag of the right-hand side sparse vector
inline bool operator!=( const DenseVector<T1,TF1>& lhs, const SparseVector<T2,TF2>& rhs )
{
   return !( lhs == rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of a sparse vector and a dense vector.
// \ingroup dense_vector
//
// \param lhs The left-hand side sparse vector for the comparison.
// \param rhs The right-hand side dense vector for the comparison.
// \return \a true if the two vectors are not equal, \a false if they are equal.
*/
template< typename T1  // Type of the left-hand side sparse vector
        , bool TF1     // Transpose flag of the left-hand side sparse vector
        , typename T2  // Type of the right-hand side dense vector
        , bool TF2 >   // Transpose flag of the right-hand side dense vector
inline bool operator!=( const SparseVector<T1,TF1>& lhs, const DenseVector<T2,TF2>& rhs )
{
   return !( rhs == lhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of a dense vector and a scalar value.
// \ingroup dense_vector
//
// \param vec The left-hand side dense vector for the comparison.
// \param scalar The right-hand side scalar value for the comparison.
// \return \a true if at least one element of the vector is different from the scalar, \a false if not.
//
// If one value of the vector is inequal to the scalar value, the inequality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1  // Type of the left-hand side dense vector
        , typename T2  // Type of the right-hand side scalar
        , bool TF >    // Transpose flag
inline EnableIf_<IsNumeric<T2>, bool > operator!=( const DenseVector<T1,TF>& vec, T2 scalar )
{
   return !( vec == scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality operator for the comparison of a scalar value and a dense vector.
// \ingroup dense_vector
//
// \param scalar The left-hand side scalar value for the comparison.
// \param vec The right-hand side dense vector for the comparison.
// \return \a true if at least one element of the vector is different from the scalar, \a false if not.
//
// If one value of the vector is inequal to the scalar value, the inequality test returns \a true,
// otherwise \a false. Note that this function can only be used with built-in, numerical data
// types!
*/
template< typename T1  // Type of the left-hand side scalar
        , typename T2  // Type of the right-hand side vector
        , bool TF >    // Transpose flag
inline EnableIf_<IsNumeric<T1>, bool > operator!=( T1 scalar, const DenseVector<T2,TF>& vec )
{
   return !( vec == scalar );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DenseVector functions */
//@{
template< typename VT, bool TF >
bool isnan( const DenseVector<VT,TF>& dv );

template< typename VT, bool TF >
bool isDivisor( const DenseVector<VT,TF>& dv );

template< typename VT, bool TF >
bool isUniform( const DenseVector<VT,TF>& dv );

template< typename VT, bool TF >
const ElementType_<VT> sqrLength( const DenseVector<VT,TF>& dv );

template< typename VT, bool TF >
inline auto length( const DenseVector<VT,TF>& dv ) -> decltype( sqrt( sqrLength( ~dv ) ) );

template< typename VT, bool TF >
const ElementType_<VT> min( const DenseVector<VT,TF>& dv );

template< typename VT, bool TF >
const ElementType_<VT> max( const DenseVector<VT,TF>& dv );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks the given dense vector for not-a-number elements.
// \ingroup dense_vector
//
// \param dv The vector to be checked for not-a-number elements.
// \return \a true if at least one element of the vector is not-a-number, \a false otherwise.
//
// This function checks the N-dimensional dense vector for not-a-number (NaN) elements. If at
// least one element of the vector is not-a-number, the function returns \a true, otherwise it
// returns \a false.

   \code
   blaze::DynamicVector<double> a;
   // ... Resizing and initialization
   if( isnan( a ) ) { ... }
   \endcode

// Note that this function only works for vectors with floating point elements. The attempt to
// use it for a vector with a non-floating point element type results in a compile time error.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
bool isnan( const DenseVector<VT,TF>& dv )
{
   CompositeType_<VT> a( ~dv );  // Evaluation of the dense vector operand

   for( size_t i=0UL; i<a.size(); ++i ) {
      if( isnan( a[i] ) ) return true;
   }
   return false;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given dense vector is a valid divisor.
// \ingroup dense_vector
//
// \param dv The dense vector to be tested.
// \return \a true in case the given vector is a valid divisor, \a false otherwise.
//
// This function checks if the given dense vector is a valid divisor. If all elements of the
// vector are valid divisors the function returns \a true, if at least one element of the vector
// is not a valid divisor, the function returns \a false.

   \code
   StaticVector<int,3UL> a{ 1, -1, 2 };  // isDivisor( a ) returns true
   StaticVector<int,3UL> b{ 1, -1, 0 };  // isDivisor( b ) returns false
   \endcode
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
bool isDivisor( const DenseVector<VT,TF>& dv )
{
   CompositeType_<VT> a( ~dv );  // Evaluation of the dense vector operand

   for( size_t i=0UL; i<a.size(); ++i ) {
      if( !isDivisor( a[i] ) ) return false;
   }
   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given dense vector is a uniform vector.
// \ingroup dense_vector
//
// \param dv The dense vector to be checked.
// \return \a true if the vector is a uniform vector, \a false if not.
//
// This function checks if the given dense vector is a uniform vector. The vector is considered
// to be uniform if all its elements are identical. The following code example demonstrates the
// use of the function:

   \code
   blaze::DynamicVector<int,blaze::columnVector> a, b;
   // ... Initialization
   if( isUniform( a ) ) { ... }
   \endcode

// It is also possible to check if a vector expression results in a uniform vector:

   \code
   if( isUniform( a + b ) ) { ... }
   \endcode

// However, note that this might require the complete evaluation of the expression, including
// the generation of a temporary vector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
bool isUniform( const DenseVector<VT,TF>& dv )
{
   typedef CompositeType_<VT>  CT;
   typedef ConstReference_< RemoveReference_<CT> >  ConstReference;

   if( (~dv).size() < 2UL )
      return true;

   CT a( ~dv );  // Evaluation of the dense vector operand

   ConstReference cmp( (~dv)[0UL] );

   for( size_t i=1UL; i<(~dv).size(); ++i ) {
      if( (~dv)[i] != cmp )
         return false;
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculation of the dense vector square length \f$|\vec{a}|^2\f$.
// \ingroup dense_vector
//
// \param dv The given dense vector.
// \return The square length of the dense vector.
//
// This function calculates the actual square length of the dense vector.
//
// \note This operation is only defined for numeric data types. In case the element type is
// not a numeric data type (i.e. a user defined data type or boolean) the attempt to use the
// sqrLength() function results in a compile time error!
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
const ElementType_<VT> sqrLength( const DenseVector<VT,TF>& dv )
{
   typedef ElementType_<VT>  ElementType;

   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( ElementType );

   ElementType sum( 0 );
   for( size_t i=0UL; i<(~dv).size(); ++i )
      sum += sq( (~dv)[i] );
   return sum;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculation of the dense vector length \f$|\vec{a}|\f$.
// \ingroup dense_vector
//
// \param dv The given dense vector.
// \return The length of the dense vector.
//
// This function calculates the actual length of the dense vector. The return type of the
// length() function depends on the actual element type of the vector instance:
//
// <table border="0" cellspacing="0" cellpadding="1">
//    <tr>
//       <td width="250px"> \b Type </td>
//       <td width="100px"> \b LengthType </td>
//    </tr>
//    <tr>
//       <td>float</td>
//       <td>float</td>
//    </tr>
//    <tr>
//       <td>integral data types and double</td>
//       <td>double</td>
//    </tr>
//    <tr>
//       <td>long double</td>
//       <td>long double</td>
//    </tr>
//    <tr>
//       <td>complex<T></td>
//       <td>complex<T></td>
//    </tr>
// </table>
//
// \note This operation is only defined for numeric data types. In case the element type is
// not a numeric data type (i.e. a user defined data type or boolean) the attempt to use the
// sqrLength() function results in a compile time error!
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline auto length( const DenseVector<VT,TF>& dv ) -> decltype( sqrt( sqrLength( ~dv ) ) )
{
   return sqrt( sqrLength( ~dv ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the smallest element of the dense vector.
// \ingroup dense_vector
//
// \param dv The given dense vector.
// \return The smallest dense vector element.
//
// This function returns the smallest element of the given dense vector. This function can
// only be used for element types that support the smaller-than relationship. In case the
// vector currently has a size of 0, the returned value is the default value (e.g. 0 in case
// of fundamental data types).
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
const ElementType_<VT> min( const DenseVector<VT,TF>& dv )
{
   using blaze::min;

   typedef ElementType_<VT>    ET;
   typedef CompositeType_<VT>  CT;

   CT a( ~dv );  // Evaluation of the dense vector operand

   if( a.size() == 0UL ) return ET();

   ET minimum( a[0] );
   for( size_t i=1UL; i<a.size(); ++i )
      minimum = min( minimum, a[i] );
   return minimum;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the largest element of the dense vector.
// \ingroup dense_vector
//
// \param dv The given dense vector.
// \return The largest dense vector element.
//
// This function returns the largest element of the given dense vector. This function can
// only be used for element types that support the smaller-than relationship. In case the
// vector currently has a size of 0, the returned value is the default value (e.g. 0 in case
// of fundamental data types).
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
const ElementType_<VT> max( const DenseVector<VT,TF>& dv )
{
   using blaze::max;

   typedef ElementType_<VT>    ET;
   typedef CompositeType_<VT>  CT;

   CT a( ~dv );  // Evaluation of the dense vector operand

   if( a.size() == 0UL ) return ET();

   ET maximum( a[0] );
   for( size_t i=1UL; i<a.size(); ++i )
      maximum = max( maximum, a[i] );
   return maximum;
}
//*************************************************************************************************

} // namespace blaze

#endif
