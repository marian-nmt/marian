//=================================================================================================
/*!
//  \file blaze/math/adaptors/unilowermatrix/UniLowerElement.h
//  \brief Header file for the UniLowerElement class
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

#ifndef _BLAZE_MATH_ADAPTORS_UNILOWERMATRIX_UNILOWERELEMENT_H_
#define _BLAZE_MATH_ADAPTORS_UNILOWERMATRIX_UNILOWERELEMENT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/adaptors/unilowermatrix/UniLowerValue.h>
#include <blaze/math/constraints/Expression.h>
#include <blaze/math/constraints/Hermitian.h>
#include <blaze/math/constraints/Lower.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/Upper.h>
#include <blaze/math/Exception.h>
#include <blaze/math/sparse/SparseElement.h>
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
/*!\brief Representation of an element within a sparse lower unitriangular matrix.
// \ingroup unilower_matrix
//
// The UniLowerElement class represents an element (i.e. value/index pair) within a sparse lower
// unitriangular matrix. It guarantees that the unilower matrix invariant is not violated, i.e.
// that elements in the upper part of the matrix remain 0 and the diagonal elements remain 1. The
// following example illustrates this by means of a \f$ 3 \times 3 \f$ sparse lower unitriangular
// matrix:

   \code
   typedef blaze::UniLowerMatrix< blaze::CompressedMatrix<int> >  UniLower;

   // Creating a 3x3 lower unitriangular sparse matrix
   UniLower A( 3UL );

   A(1,0) = -2;  //        (  1 0 0 )
   A(2,0) =  3;  // => A = ( -2 1 0 )
   A(2,1) =  5;  //        (  3 5 1 )

   UniLower::Iterator it = A.begin( 1UL );
   *it = 4;  // Modification of matrix element (1,0)
   ++it;
   *it = 9;  // Invalid assignment to diagonal matrix element; results in an exception!
   \endcode
*/
template< typename MT >  // Type of the adapted matrix
class UniLowerElement : private SparseElement
{
 private:
   //**Type definitions****************************************************************************
   typedef ElementType_<MT>  ElementType;   //!< Type of the represented matrix element.
   typedef Iterator_<MT>     IteratorType;  //!< Type of the underlying sparse matrix iterators.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef UniLowerValue<MT>        ValueType;       //!< The value type of the value-index-pair.
   typedef size_t                   IndexType;       //!< The index type of the value-index-pair.
   typedef UniLowerValue<MT>        Reference;       //!< Reference return type.
   typedef const UniLowerValue<MT>  ConstReference;  //!< Reference-to-const return type.
   typedef UniLowerElement*         Pointer;         //!< Pointer return type.
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\name Constructors */
   //@{
   inline UniLowerElement( IteratorType pos, bool diagonal );
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   template< typename T > inline UniLowerElement& operator= ( const T& v );
   template< typename T > inline UniLowerElement& operator+=( const T& v );
   template< typename T > inline UniLowerElement& operator-=( const T& v );
   template< typename T > inline UniLowerElement& operator*=( const T& v );
   template< typename T > inline UniLowerElement& operator/=( const T& v );
   //@}
   //**********************************************************************************************

   //**Access operators****************************************************************************
   /*!\name Access operators */
   //@{
   inline Pointer operator->() noexcept;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline Reference value() const;
   inline IndexType index() const;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   IteratorType pos_;  //!< Iterator to the current lower unitriangular matrix element.
   bool    diagonal_;  //!< \a true in case the element is on the diagonal, \a false if not.
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
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE             ( ElementType );
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
/*!\brief Constructor for the UniLowerElement class.
//
// \param pos Iterator to the current position with the sparse lower unitriangular matrix.
// \param diagonal \a true in case the element is on the diagonal, \a false if not.
*/
template< typename MT >  // Type of the adapted matrix
inline UniLowerElement<MT>::UniLowerElement( IteratorType pos, bool diagonal )
   : pos_     ( pos      )  // Iterator to the current lower unitriangular matrix element
   , diagonal_( diagonal )  // true in case the element is on the diagonal, false if not
{}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Assignment to the unilower element.
//
// \param v The new value of the unilower element.
// \return Reference to the assigned unilower element.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniLowerElement<MT>& UniLowerElement<MT>::operator=( const T& v )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *pos_ = v;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment to the unilower element.
//
// \param v The right-hand side value for the addition.
// \return Reference to the assigned unilower element.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniLowerElement<MT>& UniLowerElement<MT>::operator+=( const T& v )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *pos_ += v;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment to the unilower element.
//
// \param v The right-hand side value for the subtraction.
// \return Reference to the assigned unilower element.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniLowerElement<MT>& UniLowerElement<MT>::operator-=( const T& v )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *pos_ -= v;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment to the unilower element.
//
// \param v The right-hand side value for the multiplication.
// \return Reference to the assigned unilower element.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniLowerElement<MT>& UniLowerElement<MT>::operator*=( const T& v )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *pos_ *= v;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment to the unilower element.
//
// \param v The right-hand side value for the division.
// \return Reference to the assigned unilower element.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniLowerElement<MT>& UniLowerElement<MT>::operator/=( const T& v )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *pos_ /= v;

   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  ACCESS OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Direct access to the unilower element.
//
// \return Reference to the value of the unilower element.
*/
template< typename MT >  // Type of the adapted matrix
inline typename UniLowerElement<MT>::Pointer UniLowerElement<MT>::operator->() noexcept
{
   return this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Access to the current value of the unilower element.
//
// \return The current value of the unilower element.
*/
template< typename MT >  // Type of the adapted matrix
inline typename UniLowerElement<MT>::Reference UniLowerElement<MT>::value() const
{
   return Reference( pos_->value(), diagonal_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Access to the current index of the unilower element.
//
// \return The current index of the unilower element.
*/
template< typename MT >  // Type of the adapted matrix
inline typename UniLowerElement<MT>::IndexType UniLowerElement<MT>::index() const
{
   return pos_->index();
}
//*************************************************************************************************

} // namespace blaze

#endif
