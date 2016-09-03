//=================================================================================================
/*!
//  \file blaze/math/adaptors/unilowermatrix/UniLowerValue.h
//  \brief Header file for the UniLowerValue class
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

#ifndef _BLAZE_MATH_ADAPTORS_UNILOWERMATRIX_UNILOWERVALUE_H_
#define _BLAZE_MATH_ADAPTORS_UNILOWERMATRIX_UNILOWERVALUE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Expression.h>
#include <blaze/math/constraints/Hermitian.h>
#include <blaze/math/constraints/Lower.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/Upper.h>
#include <blaze/math/Exception.h>
#include <blaze/math/proxy/Proxy.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/Invert.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsNaN.h>
#include <blaze/math/shims/IsOne.h>
#include <blaze/math/shims/IsReal.h>
#include <blaze/math/shims/IsZero.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Numeric.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/InvalidType.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsComplex.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Representation of a value within a sparse lower unitriangular matrix.
// \ingroup unilower_matrix
//
// The UniLowerValue class represents a single value within a sparse lower unitriangular matrix.
// It guarantees that the unilower matrix invariant is not violated, i.e. that elements in the
// upper part of the matrix remain 0 and the diagonal elements remain 1. The following example
// illustrates this by means of a \f$ 3 \times 3 \f$ sparse lower unitriangular matrix:

   \code
   typedef blaze::UniLowerMatrix< blaze::CompressedMatrix<int> >  UniLower;

   // Creating a 3x3 lower unitriangular sparse matrix
   UniLower A( 3UL );

   A(1,0) = -2;  //        (  1 0 0 )
   A(2,0) =  3;  // => A = ( -2 1 0 )
   A(2,1) =  5;  //        (  3 5 1 )

   UniLower::Iterator it = A.begin( 1UL );
   it->value() = 4;  // Modification of matrix element (1,0)
   ++it;
   it->value() = 9;  // Invalid assignment to diagonal matrix element; results in an exception!
   \endcode
*/
template< typename MT >  // Type of the adapted matrix
class UniLowerValue : public Proxy< UniLowerValue<MT> >
{
 private:
   //**struct BuiltinType**************************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Auxiliary struct to determine the value type of the represented complex element.
   */
   template< typename T >
   struct BuiltinType { typedef INVALID_TYPE  Type; };
   /*! \endcond */
   //**********************************************************************************************

   //**struct ComplexType**************************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Auxiliary struct to determine the value type of the represented complex element.
   */
   template< typename T >
   struct ComplexType { typedef typename T::value_type  Type; };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef ElementType_<MT>  RepresentedType;   //!< Type of the represented matrix element.

   //! Value type of the represented complex element.
   typedef typename If_< IsComplex<RepresentedType>
                       , ComplexType<RepresentedType>
                       , BuiltinType<RepresentedType> >::Type  ValueType;

   typedef ValueType  value_type;  //!< Value type of the represented complex element.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   inline UniLowerValue( RepresentedType& value, bool diagonal );
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
                          inline UniLowerValue& operator= ( const UniLowerValue& ulv );
   template< typename T > inline UniLowerValue& operator= ( const T& value );
   template< typename T > inline UniLowerValue& operator+=( const T& value );
   template< typename T > inline UniLowerValue& operator-=( const T& value );
   template< typename T > inline UniLowerValue& operator*=( const T& value );
   template< typename T > inline UniLowerValue& operator/=( const T& value );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline void reset () const;
   inline void clear () const;
   inline void invert() const;

   inline RepresentedType get() const noexcept;
   inline bool            isRestricted() const noexcept;
   //@}
   //**********************************************************************************************

   //**Conversion operator*************************************************************************
   /*!\name Conversion operator */
   //@{
   inline operator RepresentedType() const noexcept;
   //@}
   //**********************************************************************************************

   //**Complex data access functions***************************************************************
   /*!\name Complex data access functions */
   //@{
   inline ValueType real() const;
   inline void      real( ValueType value ) const;
   inline ValueType imag() const;
   inline void      imag( ValueType value ) const;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   RepresentedType* value_;     //!< The represented value.
   bool             diagonal_;  //!< \a true in case the element is on the diagonal, \a false if not.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE       ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE       ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE         ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST                ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE             ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_EXPRESSION_TYPE      ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_HERMITIAN_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_LOWER_MATRIX_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_UPPER_MATRIX_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( RepresentedType );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the UniLowerValue class.
//
// \param value Reference to the represented value.
// \param diagonal \a true in case the element is on the diagonal, \a false if not.
*/
template< typename MT >  // Type of the adapted matrix
inline UniLowerValue<MT>::UniLowerValue( RepresentedType& value, bool diagonal )
   : value_   ( &value   )  // The represented value.
   , diagonal_( diagonal )  // true in case the element is on the diagonal, false if not
{}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Copy assignment operator for UniLowerValue.
//
// \param ulv The unilower value to be copied.
// \return Reference to the assigned unilower value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline UniLowerValue<MT>& UniLowerValue<MT>::operator=( const UniLowerValue& ulv )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *value_ = *ulv.value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment to the unilower value.
//
// \param value The new value of the unilower value.
// \return Reference to the assigned unilower value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniLowerValue<MT>& UniLowerValue<MT>::operator=( const T& value )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *value_ = value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment to the unilower value.
//
// \param value The right-hand side value to be added to the unilower value.
// \return Reference to the assigned unilower value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniLowerValue<MT>& UniLowerValue<MT>::operator+=( const T& value )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *value_ += value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment to the unilower value.
//
// \param value The right-hand side value to be subtracted from the unilower value.
// \return Reference to the assigned unilower value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniLowerValue<MT>& UniLowerValue<MT>::operator-=( const T& value )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *value_ -= value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment to the unilower value.
//
// \param value The right-hand side value for the multiplication.
// \return Reference to the assigned unilower value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniLowerValue<MT>& UniLowerValue<MT>::operator*=( const T& value )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *value_ *= value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment to the unilower value.
//
// \param value The right-hand side value for the division.
// \return Reference to the assigned unilower value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniLowerValue<MT>& UniLowerValue<MT>::operator/=( const T& value )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *value_ /= value;

   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Reset the unilower value to its default initial value.
//
// \return void
//
// This function resets the unilower value to its default initial value.
*/
template< typename MT >  // Type of the adapted matrix
inline void UniLowerValue<MT>::reset() const
{
   using blaze::reset;

   if( !diagonal_ )
      reset( *value_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the unilower value.
//
// \return void
//
// This function clears the unilower value to its default initial state.
*/
template< typename MT >  // Type of the adapted matrix
inline void UniLowerValue<MT>::clear() const
{
   using blaze::clear;

   if( !diagonal_ )
      clear( *value_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place inversion of the unilower value
//
// \return void
*/
template< typename MT >  // Type of the adapted matrix
inline void UniLowerValue<MT>::invert() const
{
   using blaze::invert;

   if( !diagonal_ )
      invert( *value_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Access to the represented value.
//
// \return Copy of the represented value.
*/
template< typename MT >  // Type of the adapted matrix
inline typename UniLowerValue<MT>::RepresentedType UniLowerValue<MT>::get() const noexcept
{
   return *value_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the value represents a restricted matrix element..
//
// \return \a true in case access to the matrix element is restricted, \a false if not.
*/
template< typename MT >  // Type of the adapted matrix
inline bool UniLowerValue<MT>::isRestricted() const noexcept
{
   return diagonal_;
}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion to the represented value.
//
// \return Copy of the represented value.
*/
template< typename MT >  // Type of the adapted matrix
inline UniLowerValue<MT>::operator RepresentedType() const noexcept
{
   return *value_;
}
//*************************************************************************************************




//=================================================================================================
//
//  COMPLEX DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the real part of the represented complex number.
//
// \return The current real part of the represented complex number.
//
// In case the proxy represents a complex number, this function returns the current value of its
// real part.
*/
template< typename MT >  // Type of the adapted matrix
inline typename UniLowerValue<MT>::ValueType UniLowerValue<MT>::real() const
{
   return value_->real();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the real part of the represented complex number.
//
// \param value The new value for the real part.
// \return void
// \exception std::invalid_argument Invalid setting for diagonal matrix element.
//
// In case the proxy represents a complex number, this function sets a new value to its real part.
*/
template< typename MT >  // Type of the adapted matrix
inline void UniLowerValue<MT>::real( ValueType value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setting for diagonal matrix element" );
   }

   value_->real( value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the imaginary part of the represented complex number.
//
// \return The current imaginary part of the represented complex number.
//
// In case the proxy represents a complex number, this function returns the current value of its
// imaginary part.
*/
template< typename MT >  // Type of the adapted matrix
inline typename UniLowerValue<MT>::ValueType UniLowerValue<MT>::imag() const
{
   return value_->imag();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the imaginary part of the represented complex number.
//
// \param value The new value for the imaginary part.
// \return void
// \exception std::invalid_argument Invalid setting for diagonal matrix element.
//
// In case the proxy represents a complex number, this function sets a new value to its imaginary
// part. In case the proxy represents a diagonal element and the given value is not zero, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
inline void UniLowerValue<MT>::imag( ValueType value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setting for diagonal matrix element" );
   }

   value_->imag( value );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name UniLowerValue global functions */
//@{
template< typename MT >
inline void reset( const UniLowerValue<MT>& value );

template< typename MT >
inline void clear( const UniLowerValue<MT>& value );

template< typename MT >
inline void invert( const UniLowerValue<MT>& value );

template< typename MT >
inline bool isDefault( const UniLowerValue<MT>& value );

template< typename MT >
inline bool isReal( const UniLowerValue<MT>& value );

template< typename MT >
inline bool isZero( const UniLowerValue<MT>& value );

template< typename MT >
inline bool isOne( const UniLowerValue<MT>& value );

template< typename MT >
inline bool isnan( const UniLowerValue<MT>& value );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the unilower value to the default initial values.
// \ingroup unilower_matrix
//
// \param value The given unilower value.
// \return void
//
// This function resets the unilower value to its default initial value.
*/
template< typename MT >
inline void reset( const UniLowerValue<MT>& value )
{
   value.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the unilower value.
// \ingroup unilower_matrix
//
// \param value The given unilower value.
// \return void
//
// This function clears the unilower value to its default initial state.
*/
template< typename MT >
inline void clear( const UniLowerValue<MT>& value )
{
   value.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place inversion of the unilower value.
// \ingroup unilower_matrix
//
// \param value The given unilower value.
// \return void
*/
template< typename MT >
inline void invert( const UniLowerValue<MT>& value )
{
   value.invert();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the unilower value is in default state.
// \ingroup unilower_matrix
//
// \param value The given unilower value.
// \return \a true in case the unilower value is in default state, \a false otherwise.
//
// This function checks whether the unilower value is in default state. In case it is in
// default state, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isDefault( const UniLowerValue<MT>& value )
{
   using blaze::isDefault;

   return isDefault( value.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the unilower value represents a real number.
// \ingroup unilower_matrix
//
// \param value The given unilower value.
// \return \a true in case the unilower value represents a real number, \a false otherwise.
//
// This function checks whether the unilower value represents the a real number. In case the
// value is of built-in type, the function returns \a true. In case the element is of complex
// type, the function returns \a true if the imaginary part is equal to 0. Otherwise it returns
// \a false.
*/
template< typename MT >
inline bool isReal( const UniLowerValue<MT>& value )
{
   using blaze::isReal;

   return isReal( value.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the unilower value is 0.
// \ingroup unilower_matrix
//
// \param value The given unilower value.
// \return \a true in case the unilower value is 0, \a false otherwise.
//
// This function checks whether the unilower value represents the numeric value 0. In case it
// is 0, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isZero( const UniLowerValue<MT>& value )
{
   using blaze::isZero;

   return isZero( value.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the unilower value is 1.
// \ingroup unilower_matrix
//
// \param value The given unilower value.
// \return \a true in case the unilower value is 1, \a false otherwise.
//
// This function checks whether the unilower value represents the numeric value 1. In case it
// is 1, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isOne( const UniLowerValue<MT>& value )
{
   using blaze::isOne;

   return isOne( value.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the unilower value is not a number.
// \ingroup unilower_matrix
//
// \param value The given unilower value.
// \return \a true in case the unilower value is in not a number, \a false otherwise.
//
// This function checks whether the unilower value is not a number (NaN). In case it is not a
// number, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isnan( const UniLowerValue<MT>& value )
{
   using blaze::isnan;

   return isnan( value.get() );
}
//*************************************************************************************************

} // namespace blaze

#endif
