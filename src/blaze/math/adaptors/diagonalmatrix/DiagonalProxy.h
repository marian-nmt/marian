//=================================================================================================
/*!
//  \file blaze/math/adaptors/diagonalmatrix/DiagonalProxy.h
//  \brief Header file for the DiagonalProxy class
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

#ifndef _BLAZE_MATH_ADAPTORS_DIAGONALMATRIX_DIAGONALPROXY_H_
#define _BLAZE_MATH_ADAPTORS_DIAGONALMATRIX_DIAGONALPROXY_H_


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
#include <blaze/math/InitializerList.h>
#include <blaze/math/proxy/Proxy.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsNaN.h>
#include <blaze/math/shims/IsOne.h>
#include <blaze/math/shims/IsReal.h>
#include <blaze/math/shims/IsZero.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/typetraits/AddConst.h>
#include <blaze/util/typetraits/AddReference.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Access proxy for diagonal matrices.
// \ingroup diagonal_matrix
//
// The DiagonalProxy provides controlled access to the elements of a non-const diagonal matrix.
// It guarantees that the diagonal matrix invariant is not violated, i.e. that elements in the
// lower and upper part of the matrix remain default values. The following example illustrates
// this by means of a \f$ 3 \times 3 \f$ dense diagonal matrix:

   \code
   // Creating a 3x3 dense diagonal matrix
   blaze::DiagonalMatrix< blaze::DynamicMatrix<int> > A( 3UL );

   A(0,0) = -2;  //        ( -2 0 0 )
   A(1,1) =  3;  // => A = (  0 3 0 )
   A(2,2) =  5;  //        (  0 0 5 )

   A(0,2) =  7;  // Invalid assignment to upper matrix element; results in an exception!
   \endcode
*/
template< typename MT >  // Type of the adapted matrix
class DiagonalProxy : public Proxy< DiagonalProxy<MT>, ElementType_<MT> >
{
 private:
   //**Type definitions****************************************************************************
   //! Reference type of the underlying matrix type.
   typedef AddConst_< typename MT::Reference >  ReferenceType;
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef ElementType_<MT>              RepresentedType;  //!< Type of the represented matrix element.
   typedef AddReference_<ReferenceType>  RawReference;     //!< Reference-to-non-const to the represented element.
   typedef const RepresentedType&        ConstReference;   //!< Reference-to-const to the represented element.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline DiagonalProxy( MT& matrix, size_t row, size_t column );
            inline DiagonalProxy( const DiagonalProxy& dp );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline DiagonalProxy& operator=( const DiagonalProxy& dp );

   template< typename T >
   inline DiagonalProxy& operator=( initializer_list<T> list );

   template< typename T >
   inline DiagonalProxy& operator=( initializer_list< initializer_list<T> > list );

   template< typename T > inline DiagonalProxy& operator= ( const T& value );
   template< typename T > inline DiagonalProxy& operator+=( const T& value );
   template< typename T > inline DiagonalProxy& operator-=( const T& value );
   template< typename T > inline DiagonalProxy& operator*=( const T& value );
   template< typename T > inline DiagonalProxy& operator/=( const T& value );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline RawReference get()          const noexcept;
   inline bool         isRestricted() const noexcept;
   //@}
   //**********************************************************************************************

   //**Conversion operator*************************************************************************
   /*!\name Conversion operator */
   //@{
   inline operator ConstReference() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   ReferenceType value_;    //!< Reference to the accessed matrix element.
   const bool restricted_;  //!< Access flag for the accessed matrix element.
                            /*!< The flag indicates if access to the matrix element is
                                 restricted. It is \a true in case the proxy represents
                                 an element in the lower or upper part of the matrix. */
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
/*!\brief Initialization constructor for a DiagonalProxy.
//
// \param matrix Reference to the adapted matrix.
// \param row The row-index of the accessed matrix element.
// \param column The column-index of the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline DiagonalProxy<MT>::DiagonalProxy( MT& matrix, size_t row, size_t column )
   : value_     ( matrix( row, column ) )  // Reference to the accessed matrix element
   , restricted_( row != column )          // Access flag for the accessed matrix element
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for DiagonalProxy.
//
// \param dp Diagonal proxy to be copied.
*/
template< typename MT >  // Type of the adapted matrix
inline DiagonalProxy<MT>::DiagonalProxy( const DiagonalProxy& dp )
   : value_     ( dp.value_      )  // Reference to the accessed matrix element
   , restricted_( dp.restricted_ )  // Access flag for the accessed matrix element
{}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Copy assignment operator for DiagonalProxy.
//
// \param dp Diagonal proxy to be copied.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to non-diagonal matrix element.
//
// In case the proxy represents a non-diagonal matrix element, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
inline DiagonalProxy<MT>& DiagonalProxy<MT>::operator=( const DiagonalProxy& dp )
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to non-diagonal matrix element" );
   }

   value_ = dp.value_;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initializer list assignment to the accessed matrix element.
//
// \param list The list to be assigned to the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to non-diagonal matrix element.
//
// In case the proxy represents a non-diagonal matrix element, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline DiagonalProxy<MT>& DiagonalProxy<MT>::operator=( initializer_list<T> list )
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to non-diagonal matrix element" );
   }

   value_ = list;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initializer list assignment to the accessed matrix element.
//
// \param list The list to be assigned to the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to non-diagonal matrix element.
//
// In case the proxy represents a non-diagonal matrix element, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline DiagonalProxy<MT>& DiagonalProxy<MT>::operator=( initializer_list< initializer_list<T> > list )
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to non-diagonal matrix element" );
   }

   value_ = list;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment to the accessed matrix element.
//
// \param value The new value of the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to non-diagonal matrix element.
//
// In case the proxy represents a non-diagonal matrix element, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline DiagonalProxy<MT>& DiagonalProxy<MT>::operator=( const T& value )
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to non-diagonal matrix element" );
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
// \exception std::invalid_argument Invalid assignment to non-diagonal matrix element.
//
// In case the proxy represents a non-diagonal matrix element, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline DiagonalProxy<MT>& DiagonalProxy<MT>::operator+=( const T& value )
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to non-diagonal matrix element" );
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
// \exception std::invalid_argument Invalid assignment to non-diagonal matrix element.
//
// In case the proxy represents a non-diagonal matrix element, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline DiagonalProxy<MT>& DiagonalProxy<MT>::operator-=( const T& value )
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to non-diagonal matrix element" );
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
// \exception std::invalid_argument Invalid assignment to non-diagonal matrix element.
//
// In case the proxy represents a non-diagonal matrix element, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline DiagonalProxy<MT>& DiagonalProxy<MT>::operator*=( const T& value )
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to non-diagonal matrix element" );
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
// \exception std::invalid_argument Invalid assignment to non-diagonal matrix element.
//
// In case the proxy represents a non-diagonal matrix element, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline DiagonalProxy<MT>& DiagonalProxy<MT>::operator/=( const T& value )
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to non-diagonal matrix element" );
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
/*!\brief Returning the value of the accessed matrix element.
//
// \return Direct/raw reference to the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline typename DiagonalProxy<MT>::RawReference DiagonalProxy<MT>::get() const noexcept
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
inline bool DiagonalProxy<MT>::isRestricted() const noexcept
{
   return restricted_;
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
// \return Reference-to-const to the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline DiagonalProxy<MT>::operator ConstReference() const noexcept
{
   return static_cast<ConstReference>( value_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DiagonalProxy global functions */
//@{
template< typename MT >
inline void reset( const DiagonalProxy<MT>& proxy );

template< typename MT >
inline void clear( const DiagonalProxy<MT>& proxy );

template< typename MT >
inline bool isDefault( const DiagonalProxy<MT>& proxy );

template< typename MT >
inline bool isReal( const DiagonalProxy<MT>& proxy );

template< typename MT >
inline bool isZero( const DiagonalProxy<MT>& proxy );

template< typename MT >
inline bool isOne( const DiagonalProxy<MT>& proxy );

template< typename MT >
inline bool isnan( const DiagonalProxy<MT>& proxy );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the represented element to the default initial values.
// \ingroup diagonal_matrix
//
// \param proxy The given access proxy.
// \return void
//
// This function resets the element represented by the numeric proxy to its default initial
// value.
*/
template< typename MT >
inline void reset( const DiagonalProxy<MT>& proxy )
{
   using blaze::reset;

   reset( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the represented element.
// \ingroup diagonal_matrix
//
// \param proxy The given access proxy.
// \return void
//
// This function clears the element represented by the numeric proxy to its default initial
// state.
*/
template< typename MT >
inline void clear( const DiagonalProxy<MT>& proxy )
{
   using blaze::clear;

   clear( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the represented element is in default state.
// \ingroup diagonal_matrix
//
// \param proxy The given access proxy
// \return \a true in case the represented element is in default state, \a false otherwise.
//
// This function checks whether the element represented by the access proxy is in default state.
// In case it is in default state, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isDefault( const DiagonalProxy<MT>& proxy )
{
   using blaze::isDefault;

   return isDefault( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the matrix element represents a real number.
// \ingroup diagonal_matrix
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
inline bool isReal( const DiagonalProxy<MT>& proxy )
{
   using blaze::isReal;

   return isReal( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the represented element is 0.
// \ingroup diagonal_matrix
//
// \param proxy The given access proxy.
// \return \a true in case the represented element is 0, \a false otherwise.
//
// This function checks whether the element represented by the access proxy represents the numeric
// value 0. In case it is 0, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isZero( const DiagonalProxy<MT>& proxy )
{
   using blaze::isZero;

   return isZero( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the represented element is 1.
// \ingroup diagonal_matrix
//
// \param proxy The given access proxy.
// \return \a true in case the represented element is 1, \a false otherwise.
//
// This function checks whether the element represented by the access proxy represents the numeric
// value 1. In case it is 1, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isOne( const DiagonalProxy<MT>& proxy )
{
   using blaze::isOne;

   return isOne( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the represented element is not a number.
// \ingroup diagonal_matrix
//
// \param proxy The given access proxy.
// \return \a true in case the represented element is in not a number, \a false otherwise.
//
// This function checks whether the element represented by the access proxy is not a number (NaN).
// In case it is not a number, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isnan( const DiagonalProxy<MT>& proxy )
{
   using blaze::isnan;

   return isnan( proxy.get() );
}
//*************************************************************************************************

} // namespace blaze

#endif
