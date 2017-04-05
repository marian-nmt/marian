//=================================================================================================
/*!
//  \file blaze/math/adaptors/symmetricmatrix/SymmetricElement.h
//  \brief Header file for the SymmetricElement class
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

#ifndef _BLAZE_MATH_ADAPTORS_SYMMETRICMATRIX_SYMMETRICELEMENT_H_
#define _BLAZE_MATH_ADAPTORS_SYMMETRICMATRIX_SYMMETRICELEMENT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/adaptors/symmetricmatrix/SymmetricValue.h>
#include <blaze/math/constraints/Expression.h>
#include <blaze/math/constraints/Hermitian.h>
#include <blaze/math/constraints/Lower.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/Upper.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/sparse/SparseElement.h>
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
/*!\brief Representation of two synchronized elements within the sparse symmetric matrix.
// \ingroup symmetric_matrix
//
// The SymmetricElement class represents two synchronized elements (i.e. two value/index pairs)
// within a sparse symmetric matrix. It guarantees that a modification of element \f$ a_{ij} \f$
// via iterator is also applied to element \f$ a_{ji} \f$. The following example illustrates this
// by means of a \f$ 3 \times 3 \f$ dense Hermitian matrix:

   \code
   typedef blaze::SymmetricMatrix< blaze::CompressedMatrix<int> >  Symmetric;

   // Creating a 3x3 symmetric sparse matrix
   //
   // (  0 0 -2 )
   // (  0 3  5 )
   // ( -2 5  0 )
   //
   Symmetric A( 3UL );
   A(0,2) = -2;
   A(1,1) =  3;
   A(1,2) =  5;

   // Modification of the values at position (2,0) and (0,2)
   //
   // ( 0 0 4 )
   // ( 0 3 5 )
   // ( 4 5 0 )
   //
   Symmetric::Iterator it = A.begin( 2UL );
   *it = 4;
   \endcode
*/
template< typename MT >  // Type of the adapted matrix
class SymmetricElement : private SparseElement
{
 private:
   //**Type definitions****************************************************************************
   typedef ElementType_<MT>  ElementType;   //!< Type of the represented matrix element.
   typedef Iterator_<MT>     IteratorType;  //!< Type of the underlying sparse matrix iterators.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef SymmetricValue<MT>        ValueType;       //!< The value type of the value-index-pair.
   typedef size_t                    IndexType;       //!< The index type of the value-index-pair.
   typedef SymmetricValue<MT>        Reference;       //!< Reference return type.
   typedef const SymmetricValue<MT>  ConstReference;  //!< Reference-to-const return type.
   typedef SymmetricElement*         Pointer;         //!< Pointer return type.
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\name Constructors */
   //@{
   inline SymmetricElement( IteratorType pos, MT* matrix, size_t idx );
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   template< typename T > inline SymmetricElement& operator= ( const T& v );
   template< typename T > inline SymmetricElement& operator+=( const T& v );
   template< typename T > inline SymmetricElement& operator-=( const T& v );
   template< typename T > inline SymmetricElement& operator*=( const T& v );
   template< typename T > inline SymmetricElement& operator/=( const T& v );
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
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline void sync();
   inline bool isSynced() const;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   IteratorType pos_;     //!< Iterator to the current sparse Hermitian matrix element.
   MT*          matrix_;  //!< The sparse matrix containing the iterator.
   size_t       index_;   //!< The row/column index of the iterator.
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
/*!\brief Constructor for the SymmetricElement class.
//
// \param pos The initial position of the iterator.
// \param matrix The sparse matrix containing the iterator.
// \param idx The row/column index of the iterator.
*/
template< typename MT >  // Type of the adapted matrix
inline SymmetricElement<MT>::SymmetricElement( IteratorType pos, MT* matrix, size_t idx )
   : pos_   ( pos    )  // Iterator to the current sparse symmetric matrix element
   , matrix_( matrix )  // The sparse matrix containing the iterator
   , index_ ( idx    )  // The row/column index of the iterator
{
   BLAZE_INTERNAL_ASSERT( isSynced(), "Missing matrix element detected" );
}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Assignment to the symmetric element.
//
// \param v The new value of the symmetric element.
// \return Reference to the assigned Hermitian element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline SymmetricElement<MT>& SymmetricElement<MT>::operator=( const T& v )
{
   *pos_ = v;
   sync();
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment to the Hermitian element.
//
// \param v The right-hand side value for the addition.
// \return Reference to the assigned Hermitian element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline SymmetricElement<MT>& SymmetricElement<MT>::operator+=( const T& v )
{
   *pos_ += v;
   sync();
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment to the Hermitian element.
//
// \param v The right-hand side value for the subtraction.
// \return Reference to the assigned Hermitian element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline SymmetricElement<MT>& SymmetricElement<MT>::operator-=( const T& v )
{
   *pos_ -= v;
   sync();
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment to the Hermitian element.
//
// \param v The right-hand side value for the multiplication.
// \return Reference to the assigned Hermitian element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline SymmetricElement<MT>& SymmetricElement<MT>::operator*=( const T& v )
{
   *pos_ *= v;
   sync();
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment to the Hermitian element.
//
// \param v The right-hand side value for the division.
// \return Reference to the assigned Hermitian element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline SymmetricElement<MT>& SymmetricElement<MT>::operator/=( const T& v )
{
   *pos_ /= v;
   sync();
   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  ACCESS OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Direct access to the Hermitian element.
//
// \return Reference to the value of the Hermitian element.
*/
template< typename MT >  // Type of the adapted matrix
inline typename SymmetricElement<MT>::Pointer SymmetricElement<MT>::operator->() noexcept
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
/*!\brief Access to the current value of the Hermitian element.
//
// \return The current value of the Hermitian element.
*/
template< typename MT >  // Type of the adapted matrix
inline typename SymmetricElement<MT>::Reference SymmetricElement<MT>::value() const
{
   return Reference( pos_, matrix_, index_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Access to the current index of the Hermitian element.
//
// \return The current index of the Hermitian element.
*/
template< typename MT >  // Type of the adapted matrix
inline typename SymmetricElement<MT>::IndexType SymmetricElement<MT>::index() const
{
   return pos_->index();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Synchronization of the current sparse element to the according paired element.
//
// \return void
*/
template< typename MT >  // Type of the adapted matrix
inline void SymmetricElement<MT>::sync()
{
   if( pos_->index() == index_ || isDefault( pos_->value() ) )
      return;

   const size_t row   ( ( IsRowMajorMatrix<MT>::value )?( pos_->index() ):( index_ ) );
   const size_t column( ( IsRowMajorMatrix<MT>::value )?( index_ ):( pos_->index() ) );

   matrix_->set( row, column, pos_->value() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking if the current sparse element is in sync.
//
// \return \a true if the current sparse element is in sync, \a false if not.
*/
template< typename MT >  // Type of the adapted matrix
inline bool SymmetricElement<MT>::isSynced() const
{
   const size_t row   ( ( IsRowMajorMatrix<MT>::value )?( pos_->index() ):( index_ ) );
   const size_t column( ( IsRowMajorMatrix<MT>::value )?( index_ ):( pos_->index() ) );

   const IteratorType pos2( matrix_->find( row, column ) );
   const IteratorType end( matrix_->end( pos_->index() ) );

   return ( isDefault( pos_->value() ) && ( pos2 == end || isDefault( pos2->value() ) ) ) ||
          ( pos2 != end && pos_->value() == pos2->value() );
}
//*************************************************************************************************

} // namespace blaze

#endif
