//=================================================================================================
/*!
//  \file blaze/math/adaptors/unilowermatrix/UniLowerProxy.h
//  \brief Header file for the UniLowerProxy class
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

#ifndef _BLAZE_MATH_ADAPTORS_LOWERMATRIX_UNILOWERPROXY_H_
#define _BLAZE_MATH_ADAPTORS_LOWERMATRIX_UNILOWERPROXY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Expression.h>
#include <blaze/math/constraints/Hermitian.h>
#include <blaze/math/constraints/Lower.h>
#include <blaze/math/constraints/Matrix.h>
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
/*!\brief Access proxy for lower unitriangular matrices.
// \ingroup unilower_matrix
//
// The UniLowerProxy provides controlled access to the elements of a non-const lower unitriangular
// matrix. It guarantees that the unilower matrix invariant is not violated, i.e. that elements
// in the upper part of the matrix remain 0 and the diagonal elements remain 1. The following
// example illustrates this by means of a \f$ 3 \times 3 \f$ dense lower unitriangular matrix:

   \code
   // Creating a 3x3 lower unitriangular dense matrix
   blaze::UniLowerMatrix< blaze::DynamicMatrix<int> > A( 3UL );

   A(1,0) = -2;  //        (  1 0 0 )
   A(2,0) =  3;  // => A = ( -2 1 0 )
   A(2,1) =  5;  //        (  3 5 1 )

   A(1,1) =  4;  // Invalid assignment to diagonal matrix element; results in an exception!
   A(0,2) =  7;  // Invalid assignment to upper matrix element; results in an exception!
   \endcode
*/
template< typename MT >  // Type of the adapted matrix
class UniLowerProxy : public Proxy< UniLowerProxy<MT> >
{
 private:
   //**Type definitions****************************************************************************
   //! Reference type of the underlying matrix type.
   typedef typename MT::Reference  ReferenceType;
   //**********************************************************************************************

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
   //! Type of the represented matrix element.
   typedef ElementType_<MT>  RepresentedType;

   //! Value type of the represented complex element.
   typedef typename If_< IsComplex<RepresentedType>
                       , ComplexType<RepresentedType>
                       , BuiltinType<RepresentedType> >::Type  ValueType;

   typedef ValueType  value_type;  //!< Value type of the represented complex element.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline UniLowerProxy( MT& matrix, size_t row, size_t column );
            inline UniLowerProxy( const UniLowerProxy& ulp );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
                          inline const UniLowerProxy& operator= ( const UniLowerProxy& ulp ) const;
   template< typename T > inline const UniLowerProxy& operator= ( const T& value ) const;
   template< typename T > inline const UniLowerProxy& operator+=( const T& value ) const;
   template< typename T > inline const UniLowerProxy& operator-=( const T& value ) const;
   template< typename T > inline const UniLowerProxy& operator*=( const T& value ) const;
   template< typename T > inline const UniLowerProxy& operator/=( const T& value ) const;
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
   /*!\name Member variables */
   //@{
   ReferenceType value_;  //!< Reference to the accessed matrix element.
   size_t row_;           //!< Row index of the accessed matrix element.
   size_t column_;        //!< Column index of the accessed matrix element.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_MATRIX_TYPE              ( MT );
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
/*!\brief Initialization constructor for an UniLowerProxy.
//
// \param matrix Reference to the adapted matrix.
// \param row The row-index of the accessed matrix element.
// \param column The column-index of the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline UniLowerProxy<MT>::UniLowerProxy( MT& matrix, size_t row, size_t column )
   : value_ ( matrix( row, column ) )  // Reference to the accessed matrix element
   , row_   ( row    )                 // Row index of the accessed matrix element
   , column_( column )                 // Column index of the accessed matrix element
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for LowerProxy.
//
// \param ulp Proxy to be copied.
*/
template< typename MT >  // Type of the adapted matrix
inline UniLowerProxy<MT>::UniLowerProxy( const UniLowerProxy& ulp )
   : value_ ( ulp.value_  )  // Reference to the accessed matrix element
   , row_   ( ulp.row_    )  // Row index of the accessed matrix element
   , column_( ulp.column_ )  // Column index of the accessed matrix element
{}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Copy assignment operator for UniLowerProxy.
//
// \param ulp Proxy to be copied.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal or upper matrix element.
//
// In case the proxy represents an element on the diagonal or in the upper part of the matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
inline const UniLowerProxy<MT>& UniLowerProxy<MT>::operator=( const UniLowerProxy& ulp ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal or upper matrix element" );
   }

   value_ = ulp.value_;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment to the accessed matrix element.
//
// \param value The new value of the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal or upper matrix element.
//
// In case the proxy represents an element on the diagonal or in the upper part of the matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UniLowerProxy<MT>& UniLowerProxy<MT>::operator=( const T& value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal or upper matrix element" );
   }

   value_ = value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment to the accessed matrix element.
//
// \param value The right-hand side value to be added to the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal or upper matrix element.
//
// In case the proxy represents an element on the diagonal or in the upper part of the matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UniLowerProxy<MT>& UniLowerProxy<MT>::operator+=( const T& value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal or upper matrix element" );
   }

   value_ += value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment to the accessed matrix element.
//
// \param value The right-hand side value to be subtracted from the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal or upper matrix element.
//
// In case the proxy represents an element on the diagonal or in the upper part of the matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UniLowerProxy<MT>& UniLowerProxy<MT>::operator-=( const T& value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal or upper matrix element" );
   }

   value_ -= value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment to the accessed matrix element.
//
// \param value The right-hand side value for the multiplication.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal or upper matrix element.
//
// In case the proxy represents an element on the diagonal or in the upper part of the matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UniLowerProxy<MT>& UniLowerProxy<MT>::operator*=( const T& value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal or upper matrix element" );
   }

   value_ *= value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment to the accessed matrix element.
//
// \param value The right-hand side value for the division.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal or upper matrix element.
//
// In case the proxy represents an element on the diagonal or in the upper part of the matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UniLowerProxy<MT>& UniLowerProxy<MT>::operator/=( const T& value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal or upper matrix element" );
   }

   value_ /= value;

   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Reset the represented element to its default initial value.
//
// \return void
//
// This function resets the element represented by the proxy to its default initial value.
*/
template< typename MT >  // Type of the adapted matrix
inline void UniLowerProxy<MT>::reset() const
{
   using blaze::reset;

   if( column_ < row_ )
      reset( value_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the represented element.
//
// \return void
//
// This function clears the element represented by the proxy to its default initial state.
*/
template< typename MT >  // Type of the adapted matrix
inline void UniLowerProxy<MT>::clear() const
{
   using blaze::clear;

   if( column_ < row_ )
      clear( value_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place inversion of the represented element
//
// \return void
// \exception std::invalid_argument Invalid inversion of upper matrix element.
//
// In case the proxy represents an upper element, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
inline void UniLowerProxy<MT>::invert() const
{
   using blaze::invert;

   if( row_ < column_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid inversion of upper matrix element" );
   }

   if( column_ < row_ )
      invert( value_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returning the value of the accessed matrix element.
//
// \return Direct/raw reference to the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline typename UniLowerProxy<MT>::RepresentedType UniLowerProxy<MT>::get() const noexcept
{
   return value_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the proxy represents a restricted matrix element..
//
// \return \a true in case access to the matrix element is restricted, \a false if not.
*/
template< typename MT >  // Type of the adapted matrix
inline bool UniLowerProxy<MT>::isRestricted() const noexcept
{
   return row_ <= column_;
}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion to the accessed matrix element.
//
// \return Direct/raw reference to the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline UniLowerProxy<MT>::operator RepresentedType() const noexcept
{
   return get();
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
inline typename UniLowerProxy<MT>::ValueType UniLowerProxy<MT>::real() const
{
   return value_.real();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the real part of the represented complex number.
//
// \param value The new value for the real part.
// \return void
// \exception std::invalid_argument Invalid setting for diagonal or upper matrix element.
//
// In case the proxy represents a complex number, this function sets a new value to its real part.
// In case the represented value is a diagonal element or an element in the upper part of the
// matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
inline void UniLowerProxy<MT>::real( ValueType value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setting for diagonal or upper matrix element" );
   }

   value_.real( value );
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
inline typename UniLowerProxy<MT>::ValueType UniLowerProxy<MT>::imag() const
{
   return value_.imag();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the imaginary part of the represented complex number.
//
// \param value The new value for the imaginary part.
// \return void
// \exception std::invalid_argument Invalid setting for diagonal or upper matrix element.
//
// In case the proxy represents a complex number, this function sets a new value to its imaginary
// part. In case the represented value is a diagonal element or an element in the upper part of
// the matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
inline void UniLowerProxy<MT>::imag( ValueType value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setting for diagonal or upper matrix element" );
   }

   value_.imag( value );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name UniLowerProxy global functions */
//@{
template< typename MT >
inline void reset( const UniLowerProxy<MT>& proxy );

template< typename MT >
inline void clear( const UniLowerProxy<MT>& proxy );

template< typename MT >
inline void invert( const UniLowerProxy<MT>& proxy );

template< typename MT >
inline bool isDefault( const UniLowerProxy<MT>& proxy );

template< typename MT >
inline bool isReal( const UniLowerProxy<MT>& proxy );

template< typename MT >
inline bool isZero( const UniLowerProxy<MT>& proxy );

template< typename MT >
inline bool isOne( const UniLowerProxy<MT>& proxy );

template< typename MT >
inline bool isnan( const UniLowerProxy<MT>& proxy );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the represented element to the default initial values.
// \ingroup unilower_matrix
//
// \param proxy The given access proxy.
// \return void
//
// This function resets the element represented by the access proxy to its default initial
// value.
*/
template< typename MT >
inline void reset( const UniLowerProxy<MT>& proxy )
{
   proxy.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the represented element.
// \ingroup unilower_matrix
//
// \param proxy The given access proxy.
// \return void
//
// This function clears the element represented by the access proxy to its default initial
// state.
*/
template< typename MT >
inline void clear( const UniLowerProxy<MT>& proxy )
{
   proxy.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place inversion of the represented element.
// \ingroup unilower_matrix
//
// \param proxy The given access proxy.
// \return void
*/
template< typename MT >
inline void invert( const UniLowerProxy<MT>& proxy )
{
   proxy.invert();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the represented element is in default state.
// \ingroup unilower_matrix
//
// \param proxy The given access proxy
// \return \a true in case the represented element is in default state, \a false otherwise.
//
// This function checks whether the element represented by the access proxy is in default state.
// In case it is in default state, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isDefault( const UniLowerProxy<MT>& proxy )
{
   using blaze::isDefault;

   return isDefault( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the matrix element represents a real number.
// \ingroup unilower_matrix
//
// \param proxy The given access proxy.
// \return \a true in case the matrix element represents a real number, \a false otherwise.
//
// This function checks whether the element represented by the access proxy represents the a
// real number. In case the element is of built-in type, the function returns \a true. In case
// the element is of complex type, the function returns \a true if the imaginary part is equal
// to 0. Otherwise it returns \a false.
*/
template< typename MT >
inline bool isReal( const UniLowerProxy<MT>& proxy )
{
   using blaze::isReal;

   return isReal( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the represented element is 0.
// \ingroup unilower_matrix
//
// \param proxy The given access proxy.
// \return \a true in case the represented element is 0, \a false otherwise.
//
// This function checks whether the element represented by the access proxy represents the numeric
// value 0. In case it is 0, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isZero( const UniLowerProxy<MT>& proxy )
{
   using blaze::isZero;

   return isZero( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the represented element is 1.
// \ingroup unilower_matrix
//
// \param proxy The given access proxy.
// \return \a true in case the represented element is 1, \a false otherwise.
//
// This function checks whether the element represented by the access proxy represents the numeric
// value 1. In case it is 1, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isOne( const UniLowerProxy<MT>& proxy )
{
   using blaze::isOne;

   return isOne( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the represented element is not a number.
// \ingroup unilower_matrix
//
// \param proxy The given access proxy.
// \return \a true in case the represented element is in not a number, \a false otherwise.
//
// This function checks whether the element represented by the access proxy is not a number (NaN).
// In case it is not a number, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isnan( const UniLowerProxy<MT>& proxy )
{
   using blaze::isnan;

   return isnan( proxy.get() );
}
//*************************************************************************************************

} // namespace blaze

#endif
