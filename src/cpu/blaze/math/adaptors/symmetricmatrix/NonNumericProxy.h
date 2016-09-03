//=================================================================================================
/*!
//  \file blaze/math/adaptors/symmetricmatrix/NonNumericProxy.h
//  \brief Header file for the NonNumericProxy class
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

#ifndef _BLAZE_MATH_ADAPTORS_SYMMETRICMATRIX_NONNUMERICPROXY_H_
#define _BLAZE_MATH_ADAPTORS_SYMMETRICMATRIX_NONNUMERICPROXY_H_


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
#include <blaze/math/InitializerList.h>
#include <blaze/math/proxy/Proxy.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsNaN.h>
#include <blaze/math/shims/IsOne.h>
#include <blaze/math/shims/IsReal.h>
#include <blaze/math/shims/IsZero.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Numeric.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Access proxy for symmetric, square matrices with non-numeric element types.
// \ingroup symmetric_matrix
//
// The NonNumericProxy provides controlled access to the elements of a non-const symmetric matrix
// with non-numeric element type (e.g. vectors or matrices). It guarantees that a modification of
// element \f$ a_{ij} \f$ of the accessed matrix is also applied to element \f$ a_{ji} \f$. The
// following example illustrates this by means of a \f$ 3 \times 3 \f$ sparse symmetric matrix
// with StaticVector elements:

   \code
   using blaze::CompressedMatrix;
   using blaze::StaticVector;
   using blaze::SymmetricMatrix;

   typedef StaticVector<int,3UL>  Vector;

   // Creating a 3x3 symmetric sparses matrix
   SymmetricMatrix< CompressedMatrix< Vector > > A( 3UL );

   A(0,2) = Vector( -2,  1 );  //        ( (  0 0 ) ( 0  0 ) ( -2  1 ) )
   A(1,1) = Vector(  3,  4 );  // => A = ( (  0 0 ) ( 3  4 ) (  5 -1 ) )
   A(1,2) = Vector(  5, -1 );  //        ( ( -2 1 ) ( 5 -1 ) (  0  0 ) )
   \endcode
*/
template< typename MT >  // Type of the adapted matrix
class NonNumericProxy : public Proxy< NonNumericProxy<MT>, ValueType_< ElementType_<MT> > >
{
 private:
   //**Enumerations********************************************************************************
   //! Compile time flag indicating whether the given matrix type is a row-major matrix.
   enum : bool { rmm = IsRowMajorMatrix<MT>::value };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   /*! \cond BLAZE_INTERNAL */
   typedef ElementType_<MT>  ET;  //!< Element type of the adapted matrix.
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef ValueType_<ET>  RepresentedType;  //!< Type of the represented matrix element.
   typedef Reference_<ET>  RawReference;     //!< Raw reference to the represented element.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline NonNumericProxy( MT& sm, size_t i, size_t j );
            inline NonNumericProxy( const NonNumericProxy& nnp );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~NonNumericProxy();
   //@}
   //**********************************************************************************************

   //**Operators***********************************************************************************
   /*!\name Operators */
   //@{
   inline NonNumericProxy& operator= ( const NonNumericProxy& nnp );

   template< typename T >
   inline NonNumericProxy& operator=( initializer_list<T> list );

   template< typename T >
   inline NonNumericProxy& operator=( initializer_list< initializer_list<T> > list );

   template< typename T > inline NonNumericProxy& operator= ( const T& value );
   template< typename T > inline NonNumericProxy& operator+=( const T& value );
   template< typename T > inline NonNumericProxy& operator-=( const T& value );
   template< typename T > inline NonNumericProxy& operator*=( const T& value );
   template< typename T > inline NonNumericProxy& operator/=( const T& value );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline RawReference get() const noexcept;
   //@}
   //**********************************************************************************************

   //**Conversion operator*************************************************************************
   /*!\name Conversion operator */
   //@{
   inline operator RawReference() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   MT&    matrix_;  //!< Reference to the adapted matrix.
   size_t i_;       //!< Row-index of the accessed matrix element.
   size_t j_;       //!< Column-index of the accessed matrix element.
   //@}
   //**********************************************************************************************

   //**Forbidden operations************************************************************************
   /*!\name Forbidden operations */
   //@{
   void* operator&() const;  //!< Address operator (private & undefined)
   //@}
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
   BLAZE_CONSTRAINT_MUST_NOT_BE_NUMERIC_TYPE         ( RepresentedType );
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
/*!\brief Initialization constructor for a NonNumericProxy.
//
// \param matrix Reference to the adapted matrix.
// \param i The row-index of the accessed matrix element.
// \param j The column-index of the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline NonNumericProxy<MT>::NonNumericProxy( MT& matrix, size_t i, size_t j )
   : matrix_( matrix )  // Reference to the adapted matrix
   , i_     ( i )       // Row-index of the accessed matrix element
   , j_     ( j )       // Column-index of the accessed matrix element
{
   const typename MT::Iterator pos( matrix_.find( i_, j_ ) );
   const size_t index( rmm ? i_ : j_ );

   if( pos == matrix_.end(index) )
   {
      const ElementType_<MT> element( ( RepresentedType() ) );
      matrix_.insert( i_, j_, element );
      if( i_ != j_ )
         matrix_.insert( j_, i_, element );
   }

   BLAZE_INTERNAL_ASSERT( matrix_.find(i_,j_)->value() == matrix_.find(j_,i_)->value(), "Unbalance detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for NonNumericProxy.
//
// \param nnp Non-numeric access proxy to be copied.
*/
template< typename MT >  // Type of the adapted matrix
inline NonNumericProxy<MT>::NonNumericProxy( const NonNumericProxy& nnp )
   : matrix_( nnp.matrix_ )  // Reference to the adapted matrix
   , i_     ( nnp.i_ )       // Row-index of the accessed matrix element
   , j_     ( nnp.j_ )       // Column-index of the accessed matrix element
{
   BLAZE_INTERNAL_ASSERT( matrix_.find(i_,j_) != matrix_.end( rmm ? i_ : j_ ), "Missing matrix element detected" );
   BLAZE_INTERNAL_ASSERT( matrix_.find(j_,i_) != matrix_.end( rmm ? j_ : i_ ), "Missing matrix element detected" );
}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The destructor for NonNumericProxy.
*/
template< typename MT >  // Type of the adapted matrix
inline NonNumericProxy<MT>::~NonNumericProxy()
{
   const typename MT::Iterator pos( matrix_.find( i_, j_ ) );
   const size_t index( rmm ? i_ : j_ );

   if( pos != matrix_.end( index ) && isDefault( *pos->value() ) )
   {
      matrix_.erase( index, pos );
      if( i_ != j_ )
         matrix_.erase( ( rmm ? j_ : i_ ), matrix_.find( j_, i_ ) );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Copy assignment operator for NonNumericProxy.
//
// \param nnp Non-numeric access proxy to be copied.
// \return Reference to the assigned access proxy.
*/
template< typename MT >  // Type of the adapted matrix
inline NonNumericProxy<MT>& NonNumericProxy<MT>::operator=( const NonNumericProxy& nnp )
{
   get() = nnp.get();
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initializer list assignment to the represented matrix element.
//
// \param list The list to be assigned to the matrix element.
// \return Reference to the assigned proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonNumericProxy<MT>& NonNumericProxy<MT>::operator=( initializer_list<T> list )
{
   get() = list;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initializer list assignment to the represented matrix element.
//
// \param list The list to be assigned to the matrix element.
// \return Reference to the assigned proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonNumericProxy<MT>& NonNumericProxy<MT>::operator=( initializer_list< initializer_list<T> > list )
{
   get() = list;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment to the represented matrix element.
//
// \param value The new value of the matrix element.
// \return Reference to the assigned access proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonNumericProxy<MT>& NonNumericProxy<MT>::operator=( const T& value )
{
   get() = value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment to the represented matrix element.
//
// \param value The right-hand side value to be added to the matrix element.
// \return Reference to the assigned access proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonNumericProxy<MT>& NonNumericProxy<MT>::operator+=( const T& value )
{
   get() += value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment to the represented matrix element.
//
// \param value The right-hand side value to be subtracted from the matrix element.
// \return Reference to the assigned access proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonNumericProxy<MT>& NonNumericProxy<MT>::operator-=( const T& value )
{
   get() -= value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment to the represented matrix element.
//
// \param value The right-hand side value for the multiplication.
// \return Reference to the assigned access proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonNumericProxy<MT>& NonNumericProxy<MT>::operator*=( const T& value )
{
   get() *= value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment to the represented matrix element.
//
// \param value The right-hand side value for the division.
// \return Reference to the assigned access proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonNumericProxy<MT>& NonNumericProxy<MT>::operator/=( const T& value )
{
   get() /= value;
   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returning a reference to the accessed matrix element.
//
// \return Direct/raw reference to the accessed matrix element.
*/
template< typename MT >  // Type of the sparse matrix
inline typename NonNumericProxy<MT>::RawReference NonNumericProxy<MT>::get() const noexcept
{
   const typename MT::Iterator pos( matrix_.find( i_, j_ ) );
   BLAZE_INTERNAL_ASSERT( pos != matrix_.end( rmm ? i_ : j_ ), "Missing matrix element detected" );
   return *pos->value();
}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion to the represented matrix element.
//
// \return Direct/raw reference to the represented matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline NonNumericProxy<MT>::operator RawReference() const noexcept
{
   return get();
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name NonNumericProxy global functions */
//@{
template< typename MT >
inline void reset( const NonNumericProxy<MT>& proxy );

template< typename MT >
inline void clear( const NonNumericProxy<MT>& proxy );

template< typename MT >
inline bool isDefault( const NonNumericProxy<MT>& proxy );

template< typename MT >
inline bool isReal( const NonNumericProxy<MT>& proxy );

template< typename MT >
inline bool isZero( const NonNumericProxy<MT>& proxy );

template< typename MT >
inline bool isOne( const NonNumericProxy<MT>& proxy );

template< typename MT >
inline bool isnan( const NonNumericProxy<MT>& proxy );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the represented element to the default initial values.
// \ingroup symmetric_matrix
//
// \param proxy The given access proxy.
// \return void
//
// This function resets the element represented by the access proxy to its default initial value.
// In case the access proxy represents a vector- or matrix-like data structure that provides a
// reset() function, this function resets all elements of the vector/matrix to the default initial
// values.
*/
template< typename MT >
inline void reset( const NonNumericProxy<MT>& proxy )
{
   using blaze::reset;

   reset( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the represented element.
// \ingroup symmetric_matrix
//
// \param proxy The given access proxy.
// \return void
//
// This function clears the element represented by the access proxy to its default initial state.
// In case the access proxy represents a vector- or matrix-like data structure that provides a
// clear() function, this function clears the vector/matrix to its default initial state.
*/
template< typename MT >
inline void clear( const NonNumericProxy<MT>& proxy )
{
   using blaze::clear;

   clear( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the represented element is in default state.
// \ingroup symmetric_matrix
//
// \param proxy The given access proxy.
// \return \a true in case the represented element is in default state, \a false otherwise.
//
// This function checks whether the element represented by the access proxy is in default state.
// In case it is in default state, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isDefault( const NonNumericProxy<MT>& proxy )
{
   using blaze::isDefault;

   return isDefault( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the matrix element represents a real number.
// \ingroup symmetric_matrix
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
inline bool isReal( const NonNumericProxy<MT>& proxy )
{
   using blaze::isReal;

   return isReal( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the represented element is 0.
// \ingroup symmetric_matrix
//
// \param proxy The given access proxy.
// \return \a true in case the represented element is 0, \a false otherwise.
//
// This function checks whether the element represented by the access proxy represents the numeric
// value 0. In case it is 0, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isZero( const NonNumericProxy<MT>& proxy )
{
   using blaze::isZero;

   return isZero( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the represented element is 1.
// \ingroup symmetric_matrix
//
// \param proxy The given access proxy.
// \return \a true in case the represented element is 1, \a false otherwise.
//
// This function checks whether the element represented by the access proxy represents the numeric
// value 1. In case it is 1, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isOne( const NonNumericProxy<MT>& proxy )
{
   using blaze::isOne;

   return isOne( proxy.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the represented element is not a number.
// \ingroup symmetric_matrix
//
// \param proxy The given access proxy.
// \return \a true in case the represented element is in not a number, \a false otherwise.
//
// This function checks whether the element represented by the access proxy is not a number (NaN).
// In case it is not a number, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT >
inline bool isnan( const NonNumericProxy<MT>& proxy )
{
   using blaze::isnan;

   return isnan( proxy.get() );
}
//*************************************************************************************************

} // namespace blaze

#endif
