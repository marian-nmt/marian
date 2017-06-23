//=================================================================================================
/*!
//  \file blaze/math/expressions/DVecScalarMultExpr.h
//  \brief Header file for the dense vector/scalar multiplication expression
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

#ifndef _BLAZE_MATH_EXPRESSIONS_DVECSCALARMULTEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_DVECSCALARMULTEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/constraints/TransposeFlag.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/expressions/SparseVector.h>
#include <blaze/math/expressions/VecScalarMultExpr.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/traits/DivExprTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/MultExprTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/SubvectorExprTrait.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsColumnMajorMatrix.h>
#include <blaze/math/typetraits/IsColumnVector.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsInvertible.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/IsRowVector.h>
#include <blaze/math/typetraits/IsSparseMatrix.h>
#include <blaze/math/typetraits/IsSparseVector.h>
#include <blaze/math/typetraits/IsTemporary.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>
#include <blaze/math/typetraits/UnderlyingElement.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/FloatingPoint.h>
#include <blaze/util/constraints/Numeric.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/InvalidType.h>
#include <blaze/util/logging/FunctionTrace.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsNumeric.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DVECSCALARMULTEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for dense vector-scalar multiplications.
// \ingroup dense_vector_expression
//
// The DVecScalarMultExpr class represents the compile time expression for multiplications between
// a dense vector and a scalar value.
*/
template< typename VT  // Type of the left-hand side dense vector
        , typename ST  // Type of the right-hand side scalar value
        , bool TF >    // Transpose flag
class DVecScalarMultExpr : public DenseVector< DVecScalarMultExpr<VT,ST,TF>, TF >
                         , private VecScalarMultExpr
                         , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef ResultType_<VT>     RT;  //!< Result type of the dense vector expression.
   typedef ReturnType_<VT>     RN;  //!< Return type of the dense vector expression.
   typedef ElementType_<VT>    ET;  //!< Element type of the dense vector expression.
   typedef CompositeType_<VT>  CT;  //!< Composite type of the dense vector expression.
   //**********************************************************************************************

   //**Return type evaluation**********************************************************************
   //! Compilation switch for the selection of the subscript operator return type.
   /*! The \a returnExpr compile time constant expression is a compilation switch for the
       selection of the \a ReturnType. If the vector operand returns a temporary vector
       or matrix, \a returnExpr will be set to \a false and the subscript operator will
       return it's result by value. Otherwise \a returnExpr will be set to \a true and
       the subscript operator may return it's result as an expression. */
   enum : bool { returnExpr = !IsTemporary<RN>::value };

   //! Expression return type for the subscript operator.
   typedef MultExprTrait_<RN,ST>  ExprReturnType;
   //**********************************************************************************************

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the multiplication expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for
       the serial evaluation strategy of the multiplication expression. In case the given dense
       vector expression of type \a VT is a computation expression and requires an intermediate
       evaluation, \a useAssign will be set to 1 and the multiplication expression will be
       evaluated via the \a assign function family. Otherwise \a useAssign will be set to 0
       and the expression will be evaluated via the subscript operator. */
   enum : bool { useAssign = IsComputation<VT>::value && RequiresEvaluation<VT>::value };

   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT2 >
   struct UseAssign {
      enum : bool { value = useAssign };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**Parallel evaluation strategy****************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! The UseSMPAssign struct is a helper struct for the selection of the parallel evaluation
       strategy. In case either the target vector ir the dense vector operand is not SMP assignable
       and the vector operand is a computation expression that requires an intermediate evaluation,
       \a value is set to 1 and the expression specific evaluation strategy is selected. Otherwise
       \a value is set to 0 and the default strategy is chosen. */
   template< typename VT2 >
   struct UseSMPAssign {
      enum : bool { value = ( !VT2::smpAssignable || !VT::smpAssignable ) && useAssign };
   };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef DVecScalarMultExpr<VT,ST,TF>  This;           //!< Type of this DVecScalarMultExpr instance.
   typedef MultTrait_<RT,ST>             ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>    TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<ResultType>      ElementType;    //!< Resulting element type.

   //! Return type for expression template evaluations.
   typedef const IfTrue_< returnExpr, ExprReturnType, ElementType >  ReturnType;

   //! Data type for composite expression templates.
   typedef IfTrue_< useAssign, const ResultType, const DVecScalarMultExpr& >  CompositeType;

   //! Composite type of the left-hand side dense vector expression.
   typedef If_< IsExpression<VT>, const VT, const VT& >  LeftOperand;

   //! Composite type of the right-hand side scalar value.
   typedef ST  RightOperand;
   //**********************************************************************************************

   //**ConstIterator class definition**************************************************************
   /*!\brief Iterator over the elements of the dense vector.
   */
   class ConstIterator
   {
    public:
      //**Type definitions*************************************************************************
      typedef std::random_access_iterator_tag  IteratorCategory;  //!< The iterator category.
      typedef ElementType                      ValueType;         //!< Type of the underlying elements.
      typedef ElementType*                     PointerType;       //!< Pointer return type.
      typedef ElementType&                     ReferenceType;     //!< Reference return type.
      typedef ptrdiff_t                        DifferenceType;    //!< Difference between two iterators.

      // STL iterator requirements
      typedef IteratorCategory  iterator_category;  //!< The iterator category.
      typedef ValueType         value_type;         //!< Type of the underlying elements.
      typedef PointerType       pointer;            //!< Pointer return type.
      typedef ReferenceType     reference;          //!< Reference return type.
      typedef DifferenceType    difference_type;    //!< Difference between two iterators.

      //! ConstIterator type of the dense vector expression.
      typedef ConstIterator_<VT>  IteratorType;
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the ConstIterator class.
      //
      // \param iterator Iterator to the initial element.
      // \param scalar Scalar of the multiplication expression.
      */
      explicit inline ConstIterator( IteratorType iterator, RightOperand scalar )
         : iterator_( iterator )  // Iterator to the current element
         , scalar_  ( scalar   )  // Scalar of the multiplication expression
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline ConstIterator& operator+=( size_t inc ) {
         iterator_ += inc;
         return *this;
      }
      //*******************************************************************************************

      //**Subtraction assignment operator**********************************************************
      /*!\brief Subtraction assignment operator.
      //
      // \param dec The decrement of the iterator.
      // \return The decremented iterator.
      */
      inline ConstIterator& operator-=( size_t dec ) {
         iterator_ -= dec;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline ConstIterator& operator++() {
         ++iterator_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator++( int ) {
         return ConstIterator( iterator_++ );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline ConstIterator& operator--() {
         --iterator_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator--( int ) {
         return ConstIterator( iterator_-- );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReturnType operator*() const {
         return *iterator_ * scalar_;
      }
      //*******************************************************************************************

      //**Load function****************************************************************************
      /*!\brief Access to the SIMD elements of the vector.
      //
      // \return The resulting SIMD element.
      */
      inline auto load() const noexcept {
         return iterator_.load() * set( scalar_ );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const ConstIterator& rhs ) const {
         return iterator_ == rhs.iterator_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const ConstIterator& rhs ) const {
         return iterator_ != rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const ConstIterator& rhs ) const {
         return iterator_ < rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const ConstIterator& rhs ) const {
         return iterator_ > rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const ConstIterator& rhs ) const {
         return iterator_ <= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const ConstIterator& rhs ) const {
         return iterator_ >= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const ConstIterator& rhs ) const {
         return iterator_ - rhs.iterator_;
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between a ConstIterator and an integral value.
      //
      // \param it The iterator to be incremented.
      // \param inc The number of elements the iterator is incremented.
      // \return The incremented iterator.
      */
      friend inline const ConstIterator operator+( const ConstIterator& it, size_t inc ) {
         return ConstIterator( it.iterator_ + inc );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between an integral value and a ConstIterator.
      //
      // \param inc The number of elements the iterator is incremented.
      // \param it The iterator to be incremented.
      // \return The incremented iterator.
      */
      friend inline const ConstIterator operator+( size_t inc, const ConstIterator& it ) {
         return ConstIterator( it.iterator_ + inc );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Subtraction between a ConstIterator and an integral value.
      //
      // \param it The iterator to be decremented.
      // \param dec The number of elements the iterator is decremented.
      // \return The decremented iterator.
      */
      friend inline const ConstIterator operator-( const ConstIterator& it, size_t dec ) {
         return ConstIterator( it.iterator_ - dec );
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType iterator_;  //!< Iterator to the current element.
      RightOperand scalar_;    //!< Scalar of the multiplication expression.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = VT::simdEnabled &&
                               IsNumeric<ET>::value &&
                               ( HasSIMDMult<ET,ST>::value ||
                                 HasSIMDMult<UnderlyingElement_<ET>,ST>::value ) };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = VT::smpAssignable };
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   enum : size_t { SIMDSIZE = SIMDTrait<ElementType>::size };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DVecScalarMultExpr class.
   //
   // \param vector The left-hand side dense vector of the multiplication expression.
   // \param scalar The right-hand side scalar of the multiplication expression.
   */
   explicit inline DVecScalarMultExpr( const VT& vector, ST scalar ) noexcept
      : vector_( vector )  // Left-hand side dense vector of the multiplication expression
      , scalar_( scalar )  // Right-hand side scalar of the multiplication expression
   {}
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the vector elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator[]( size_t index ) const {
      BLAZE_INTERNAL_ASSERT( index < vector_.size(), "Invalid vector access index" );
      return vector_[index] * scalar_;
   }
   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the vector elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid vector access index.
   */
   inline ReturnType at( size_t index ) const {
      if( index >= vector_.size() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid vector access index" );
      }
      return (*this)[index];
   }
   //**********************************************************************************************

   //**Load function*******************************************************************************
   /*!\brief Access to the SIMD elements of the vector.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return Reference to the accessed values.
   */
   BLAZE_ALWAYS_INLINE auto load( size_t index ) const noexcept {
      BLAZE_INTERNAL_ASSERT( index < vector_.size() , "Invalid vector access index" );
      BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL, "Invalid vector access index" );
      return vector_.load( index ) * set( scalar_ );
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of the dense vector.
   //
   // \return Iterator to the first non-zero element of the dense vector.
   */
   inline ConstIterator begin() const {
      return ConstIterator( vector_.begin(), scalar_ );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of the dense vector.
   //
   // \return Iterator just past the last non-zero element of the dense vector.
   */
   inline ConstIterator end() const {
      return ConstIterator( vector_.end(), scalar_ );
   }
   //**********************************************************************************************

   //**Size function*******************************************************************************
   /*!\brief Returns the current size/dimension of the vector.
   //
   // \return The size of the vector.
   */
   inline size_t size() const noexcept {
      return vector_.size();
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side dense vector operand.
   //
   // \return The left-hand side dense vector operand.
   */
   inline LeftOperand leftOperand() const noexcept {
      return vector_;
   }
   //**********************************************************************************************

   //**Right operand access************************************************************************
   /*!\brief Returns the right-hand side scalar operand.
   //
   // \return The right-hand side scalar operand.
   */
   inline RightOperand rightOperand() const noexcept {
      return scalar_;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the expression can alias, \a false otherwise.
   */
   template< typename T >
   inline bool canAlias( const T* alias ) const noexcept {
      return IsComputation<VT>::value && vector_.canAlias( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression is aliased with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case an alias effect is detected, \a false otherwise.
   */
   template< typename T >
   inline bool isAliased( const T* alias ) const noexcept {
      return vector_.isAliased( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const noexcept {
      return vector_.isAligned();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return vector_.canSMPAssign() || ( size() > SMP_DVECSCALARMULT_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  vector_;  //!< Left-hand side dense vector of the multiplication expression.
   RightOperand scalar_;  //!< Right-hand side scalar of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense vectors*****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense vector-scalar multiplication to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense vector-scalar
   // multiplication expression to a dense vector. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the vector
   // operand is a computation expression and requires an intermediate evaluation.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseAssign<VT2> >
      assign( DenseVector<VT2,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      assign( ~lhs, rhs.vector_ );
      assign( ~lhs, (~lhs) * rhs.scalar_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to sparse vectors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense vector-scalar multiplication to a sparse vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense vector-scalar
   // multiplication expression to a sparse vector. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the vector
   // operand is a computation expression and requires an intermediate evaluation.
   */
   template< typename VT2 >  // Type of the target sparse vector
   friend inline EnableIf_< UseAssign<VT2> >
      assign( SparseVector<VT2,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      assign( ~lhs, rhs.vector_ );
      (~lhs) *= rhs.scalar_;
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense vectors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a dense vector-scalar multiplication to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a dense vector-
   // scalar multiplication expression to a dense vector. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case the
   // vector operand is a computation expression and requires an intermediate evaluation.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseAssign<VT2> >
      addAssign( DenseVector<VT2,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( serial( rhs ) );
      addAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to sparse vectors*******************************************************
   // No special implementation for the addition assignment to sparse vectors.
   //**********************************************************************************************

   //**Subtraction assignment to dense vectors*****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a dense vector-scalar multiplication to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a dense vector-
   // scalar multiplication expression to a dense vector. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the vector
   // operand is a computation expression and requires an intermediate evaluation.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseAssign<VT2> >
      subAssign( DenseVector<VT2,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( serial( rhs ) );
      subAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to sparse vectors****************************************************
   // No special implementation for the subtraction assignment to sparse vectors.
   //**********************************************************************************************

   //**Multiplication assignment to dense vectors**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Multiplication assignment of a dense vector-scalar multiplication to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a dense
   // vector-scalar multiplication expression to a dense vector. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // vector operand is a computation expression and requires an intermediate evaluation.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseAssign<VT2> >
      multAssign( DenseVector<VT2,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( serial( rhs ) );
      multAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Multiplication assignment to sparse vectors*************************************************
   // No special implementation for the multiplication assignment to sparse vectors.
   //**********************************************************************************************

   //**Division assignment to dense vectors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Division assignment of a dense vector-scalar multiplication to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression divisor.
   // \return void
   //
   // This function implements the performance optimized division assignment of a dense vector-
   // scalar multiplication expression to a dense vector. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case the
   // vector operand is a computation expression and requires an intermediate evaluation.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseAssign<VT2> >
      divAssign( DenseVector<VT2,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( serial( rhs ) );
      divAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Division assignment to sparse vectors*******************************************************
   // No special implementation for the division assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP assignment to dense vectors*************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a dense vector-scalar multiplication to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense vector-scalar
   // multiplication expression to a dense vector. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT2> >
      smpAssign( DenseVector<VT2,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      smpAssign( ~lhs, rhs.vector_ );
      smpAssign( ~lhs, (~lhs) * rhs.scalar_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to sparse vectors************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a dense vector-scalar multiplication to a sparse vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense vector-scalar
   // multiplication expression to a sparse vector. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename VT2 >  // Type of the target sparse vector
   friend inline EnableIf_< UseSMPAssign<VT2> >
      smpAssign( SparseVector<VT2,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      smpAssign( ~lhs, rhs.vector_ );
      (~lhs) *= rhs.scalar_;
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to dense vectors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a dense vector-scalar multiplication to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a dense
   // vector-scalar multiplication expression to a dense vector. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT2> >
      smpAddAssign( DenseVector<VT2,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( rhs );
      smpAddAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to sparse vectors***************************************************
   // No special implementation for the SMP addition assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP subtraction assignment to dense vectors*************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a dense vector-scalar multiplication to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a dense
   // vector-scalar multiplication expression to a dense vector. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT2> >
      smpSubAssign( DenseVector<VT2,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( rhs );
      smpSubAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to sparse vectors************************************************
   // No special implementation for the SMP subtraction assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP multiplication assignment to dense vectors**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP multiplication assignment of a dense vector-scalar multiplication to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a dense
   // vector-scalar multiplication expression to a dense vector. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT2> >
      smpMultAssign( DenseVector<VT2,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( rhs );
      smpMultAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP multiplication assignment to sparse vectors*********************************************
   // No special implementation for the SMP multiplication assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP division assignment to dense vectors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP division assignment of a dense vector-scalar multiplication to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression divisor.
   // \return void
   //
   // This function implements the performance optimized SMP division assignment of a dense vector-
   // scalar multiplication expression to a dense vector. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT2> >
      smpDivAssign( DenseVector<VT2,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( rhs );
      smpDivAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP division assignment to sparse vectors***************************************************
   // No special implementation for the SMP division assignment to sparse vectors.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( VT );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( VT, TF );
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( ST );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ST, RightOperand );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL UNARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Unary minus operator for the negation of a dense vector (\f$ \vec{a} = -\vec{b} \f$).
// \ingroup dense_vector
//
// \param dv The dense vector to be negated.
// \return The negation of the vector.
//
// This operator represents the negation of a dense vector:

   \code
   blaze::DynamicVector<double> a, b;
   // ... Resizing and initialization
   b = -a;
   \endcode

// The operator returns an expression representing the negation of the given dense vector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline const DVecScalarMultExpr<VT,UnderlyingBuiltin_<VT>,TF>
   operator-( const DenseVector<VT,TF>& dv )
{
   BLAZE_FUNCTION_TRACE;

   typedef UnderlyingBuiltin_<VT>  ElementType;
   return DVecScalarMultExpr<VT,ElementType,TF>( ~dv, ElementType(-1) );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Multiplication operator for the multiplication of a dense vector and a scalar value
//        (\f$ \vec{a}=\vec{b}*s \f$).
// \ingroup dense_vector
//
// \param vec The left-hand side dense vector for the multiplication.
// \param scalar The right-hand side scalar value for the multiplication.
// \return The scaled result vector.
//
// This operator represents the multiplication between a dense vector and a scalar value:

   \code
   blaze::DynamicVector<double> a, b;
   // ... Resizing and initialization
   b = a * 1.25;
   \endcode

// The operator returns an expression representing a dense vector of the higher-order element type
// of the involved data types \a T1::ElementType and \a T2. Both data types \a T1::ElementType and
// \a T2 have to be supported by the MultTrait class template. Note that this operator only works
// for scalar values of built-in data type.
*/
template< typename T1  // Type of the left-hand side dense vector
        , typename T2  // Type of the right-hand side scalar
        , bool TF >    // Transpose flag
inline const EnableIf_< IsNumeric<T2>, MultExprTrait_<T1,T2> >
   operator*( const DenseVector<T1,TF>& vec, T2 scalar )
{
   BLAZE_FUNCTION_TRACE;

   return MultExprTrait_<T1,T2>( ~vec, scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication operator for the multiplication of a scalar value and a dense vector
//        (\f$ \vec{a}=s*\vec{b} \f$).
// \ingroup dense_vector
//
// \param scalar The left-hand side scalar value for the multiplication.
// \param vec The right-hand side vector for the multiplication.
// \return The scaled result vector.
//
// This operator represents the multiplication between a a scalar value and dense vector:

   \code
   blaze::DynamicVector<double> a, b;
   // ... Resizing and initialization
   b = 1.25 * a;
   \endcode

// The operator returns an expression representing a dense vector of the higher-order element
// type of the involved data types \a T1::ElementType and \a T2. Both data types \a T1 and
// \a T2::ElementType have to be supported by the MultTrait class template. Note that this
// operator only works for scalar values of built-in data type.
*/
template< typename T1  // Type of the left-hand side scalar
        , typename T2  // Type of the right-hand side dense vector
        , bool TF >    // Transpose flag
inline const EnableIf_< IsNumeric<T1>, MultExprTrait_<T1,T2> >
   operator*( T1 scalar, const DenseVector<T2,TF>& vec )
{
   BLAZE_FUNCTION_TRACE;

   return MultExprTrait_<T1,T2>( ~vec, scalar );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Normalization of the dense vector (\f$|\vec{a}|=1\f$).
//
// \param vec The given dense vector.
// \return The normalized result vector.
//
// This function represents the normalization of a dense vector:

   \code
   blaze::DynamicVector<double> a;
   // ... Resizing and initialization
   a = normalize( a );
   \endcode

// The function returns an expression representing the normalized dense vector. Note that
// this function only works for floating point vectors. The attempt to use this function for
// an integral vector results in a compile time error.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline const DVecScalarMultExpr<VT,ElementType_<VT>,TF>
   normalize( const DenseVector<VT,TF>& vec )
{
   typedef ElementType_<VT>  ElementType;

   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( ElementType );

   const ElementType len ( length( ~vec ) );
   const ElementType ilen( ( len != ElementType(0) )?( ElementType(1) / len ):( 0 ) );

   return DVecScalarMultExpr<VT,ElementType,TF>( ~vec, ilen );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING UNARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unary minus operator for the negation of a dense vector-scalar multiplication
//        (\f$ \vec{a} = -(\vec{b} * s) \f$).
// \ingroup dense_vector
//
// \param dv The dense vector-scalar multiplication to be negated.
// \return The negation of the dense vector-scalar multiplication.
//
// This operator implements a performance optimized treatment of the negation of a dense vector-
// scalar multiplication expression.
*/
template< typename VT  // Type of the dense vector
        , typename ST  // Type of the scalar
        , bool TF >    // Transpose flag
inline const DVecScalarMultExpr<VT,ST,TF>
   operator-( const DVecScalarMultExpr<VT,ST,TF>& dv )
{
   BLAZE_FUNCTION_TRACE;

   return DVecScalarMultExpr<VT,ST,TF>( dv.leftOperand(), -dv.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense vector-scalar multiplication
//        expression and a scalar value (\f$ \vec{a}=(\vec{b}*s1)*s2 \f$).
// \ingroup dense_vector
//
// \param vec The left-hand side dense vector-scalar multiplication.
// \param scalar The right-hand side scalar value for the multiplication.
// \return The scaled result vector.
//
// This operator implements a performance optimized treatment of the multiplication of a
// dense vector-scalar multiplication expression and a scalar value.
*/
template< typename VT     // Type of the dense vector of the left-hand side expression
        , typename ST1    // Type of the scalar of the left-hand side expression
        , bool TF         // Transpose flag of the dense vector
        , typename ST2 >  // Type of the right-hand side scalar
inline const EnableIf_< IsNumeric<ST2>, MultExprTrait_< DVecScalarMultExpr<VT,ST1,TF>, ST2 > >
   operator*( const DVecScalarMultExpr<VT,ST1,TF>& vec, ST2 scalar )
{
   BLAZE_FUNCTION_TRACE;

   return vec.leftOperand() * ( vec.rightOperand() * scalar );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense vector-scalar multiplication
//        expression and a scalar value (\f$ \vec{a}=s2*(\vec{b}*s1) \f$).
// \ingroup dense_vector
//
// \param scalar The left-hand side scalar value for the multiplication.
// \param vec The right-hand side dense vector-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements a performance optimized treatment of the multiplication of a
// scalar value and a dense vector-scalar multiplication expression.
*/
template< typename ST1  // Type of the left-hand side scalar
        , typename VT   // Type of the dense vector of the right-hand side expression
        , typename ST2  // Type of the scalar of the right-hand side expression
        , bool TF >     // Transpose flag of the dense vector
inline const EnableIf_< IsNumeric<ST1>, MultExprTrait_< ST1, DVecScalarMultExpr<VT,ST2,TF> > >
   operator*( ST1 scalar, const DVecScalarMultExpr<VT,ST2,TF>& vec )
{
   BLAZE_FUNCTION_TRACE;

   return vec.leftOperand() * ( scalar * vec.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division operator for the division of a dense vector-scalar multiplication
//        expression by a scalar value (\f$ \vec{a}=(\vec{b}*s1)/s2 \f$).
// \ingroup dense_vector
//
// \param vec The left-hand side dense vector-scalar multiplication.
// \param scalar The right-hand side scalar value for the division.
// \return The scaled result vector.
//
// This operator implements a performance optimized treatment of the division of a
// dense vector-scalar multiplication expression by a scalar value.
*/
template< typename VT     // Type of the dense vector of the left-hand side expression
        , typename ST1    // Type of the scalar of the left-hand side expression
        , bool TF         // Transpose flag of the dense vector
        , typename ST2 >  // Type of the right-hand side scalar
inline const EnableIf_< And< IsNumeric<ST2>, Or< IsInvertible<ST1>, IsInvertible<ST2> > >
                      , DivExprTrait_< DVecScalarMultExpr<VT,ST1,TF>, ST2 > >
   operator/( const DVecScalarMultExpr<VT,ST1,TF>& vec, ST2 scalar )
{
   BLAZE_FUNCTION_TRACE;

   return vec.leftOperand() * ( vec.rightOperand() / scalar );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense vector-scalar multiplication
//        expression and a dense vector (\f$ \vec{a}=(\vec{b}*s1)*\vec{c} \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side dense vector-scalar multiplication.
// \param rhs The right-hand side dense vector.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// dense vector-scalar multiplication and a dense vector. It restructures the expression
// \f$ \vec{a}=(\vec{b}*s1)*\vec{c} \f$ to the expression \f$ \vec{a}=(\vec{b}*\vec{c})*s1 \f$.
*/
template< typename VT1    // Type of the dense vector of the left-hand side expression
        , typename ST     // Type of the scalar of the left-hand side expression
        , bool TF         // Transpose flag of the dense vectors
        , typename VT2 >  // Type of the right-hand side dense vector
inline const MultExprTrait_< DVecScalarMultExpr<VT1,ST,TF>, VT2 >
   operator*( const DVecScalarMultExpr<VT1,ST,TF>& lhs, const DenseVector<VT2,TF>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * (~rhs) ) * lhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense vector and a dense vector-
//        scalar multiplication expression (\f$ \vec{a}=\vec{b}*(\vec{c}*s1) \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side dense vector.
// \param rhs The right-hand side dense vector-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// dense vector and a dense vector-scalar multiplication. It restructures the expression
// \f$ \vec{a}=\vec{b}*(\vec{c}*s1) \f$ to the expression \f$ \vec{a}=(\vec{b}*\vec{c})*s1 \f$.
*/
template< typename VT1   // Type of the left-hand side dense vector
        , bool TF        // Transpose flag of the dense vectors
        , typename VT2   // Type of the dense vector of the right-hand side expression
        , typename ST >  // Type of the scalar of the right-hand side expression
inline const MultExprTrait_< VT1, DVecScalarMultExpr<VT2,ST,TF> >
   operator*( const DenseVector<VT1,TF>& lhs, const DVecScalarMultExpr<VT2,ST,TF>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( (~lhs) * rhs.leftOperand() ) * rhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of two dense vector-scalar
//        multiplication expressions (\f$ \vec{a}=(\vec{b}*s1)*(\vec{c}*s2) \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side dense vector-scalar multiplication.
// \param rhs The right-hand side dense vector-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of
// two dense vector-scalar multiplication expressions. It restructures the expression
// \f$ \vec{a}=(\vec{b}*s1)*(\vec{c}*s2) \f$ to the expression \f$ \vec{a}=(\vec{b}*\vec{c})*(s1*s2) \f$.
*/
template< typename VT1    // Type of the dense vector of the left-hand side expression
        , typename ST1    // Type of the scalar of the left-hand side expression
        , bool TF         // Transpose flag of the dense vectors
        , typename VT2    // Type of the dense vector of the right-hand side expression
        , typename ST2 >  // Type of the scalar of the right-hand side expression
inline const MultExprTrait_< DVecScalarMultExpr<VT1,ST1,TF>, DVecScalarMultExpr<VT2,ST2,TF> >
   operator*( const DVecScalarMultExpr<VT1,ST1,TF>& lhs, const DVecScalarMultExpr<VT2,ST2,TF>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * rhs.leftOperand() ) * ( lhs.rightOperand() * rhs.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the outer product of a dense vector-scalar multiplication
//        expression and a dense vector (\f$ A=(\vec{b}*s1)*\vec{c}^T \f$).
// \ingroup dense_matrix
//
// \param lhs The left-hand side dense vector-scalar multiplication.
// \param rhs The right-hand side dense vector.
// \return The scaled result matrix.
//
// This operator implements the performance optimized treatment of the outer product of a
// dense vector-scalar multiplication and a dense vector. It restructures the expression
// \f$ A=(\vec{b}*s1)*\vec{c}^T \f$ to the expression \f$ A=(\vec{b}*\vec{c}^T)*s1 \f$.
*/
template< typename VT1    // Type of the dense vector of the left-hand side expression
        , typename ST     // Type of the scalar of the left-hand side expression
        , typename VT2 >  // Type of the right-hand side dense vector
inline const MultExprTrait_< DVecScalarMultExpr<VT1,ST,false>, VT2 >
   operator*( const DVecScalarMultExpr<VT1,ST,false>& lhs, const DenseVector<VT2,true>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * (~rhs) ) * lhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the outer product of a dense vector and a dense vector-
//        scalar multiplication expression (\f$ A=\vec{b}*(\vec{c}^T*s1) \f$).
// \ingroup dense_matrix
//
// \param lhs The left-hand side dense vector.
// \param rhs The right-hand side dense vector-scalar multiplication.
// \return The scaled result matrix.
//
// This operator implements the performance optimized treatment of the outer product of a
// dense vector and a dense vector-scalar multiplication. It restructures the expression
// \f$ A=\vec{b}*(\vec{c}^T*s1) \f$ to the expression \f$ A=(\vec{b}*\vec{c}^T)*s1 \f$.
*/
template< typename VT1   // Type of the left-hand side dense vector
        , typename VT2   // Type of the dense vector of the right-hand side expression
        , typename ST >  // Type of the scalar of the right-hand side expression
inline const MultExprTrait_< VT1, DVecScalarMultExpr<VT2,ST,true> >
   operator*( const DenseVector<VT1,false>& lhs, const DVecScalarMultExpr<VT2,ST,true>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( (~lhs) * rhs.leftOperand() ) * rhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the outer product of two a dense vector-scalar
//        multiplication expressions (\f$ A=(\vec{b}*s1)*(\vec{c}^T*s2) \f$).
// \ingroup dense_matrix
//
// \param lhs The left-hand side dense vector-scalar multiplication.
// \param rhs The right-hand side dense vector-scalar multiplication.
// \return The scaled result matrix.
//
// This operator implements the performance optimized treatment of the outer product
// of two dense vector-scalar multiplications. It restructures the expression
// \f$ A=(\vec{b}*s1)*(\vec{c}^T*s2) \f$ to the expression \f$ A=(\vec{b}*\vec{c}^T)*(s1*s2) \f$.
*/
template< typename VT1    // Type of the dense vector of the left-hand side expression
        , typename ST1    // Type of the scalar of the left-hand side expression
        , typename VT2    // Type of the dense vector of the right-hand side expression
        , typename ST2 >  // Type of the scalar of the right-hand side expression
inline const MultExprTrait_< DVecScalarMultExpr<VT1,ST1,false>, DVecScalarMultExpr<VT2,ST2,true> >
   operator*( const DVecScalarMultExpr<VT1,ST1,false>& lhs, const DVecScalarMultExpr<VT2,ST2,true>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * rhs.leftOperand() ) * ( lhs.rightOperand() * rhs.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense vector-scalar multiplication
//        expression and a sparse vector (\f$ \vec{a}=(\vec{b}*s1)*\vec{c} \f$).
// \ingroup sparse_vector
//
// \param lhs The left-hand side dense vector-scalar multiplication.
// \param rhs The right-hand side sparse vector.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// dense vector-scalar multiplication and a sparse vector. It restructures the expression
// \f$ \vec{a}=(\vec{b}*s1)*\vec{c} \f$ to the expression \f$ \vec{a}=(\vec{b}*\vec{c})*s1 \f$.
*/
template< typename VT1    // Type of the dense vector of the left-hand side expression
        , typename ST     // Type of the scalar of the left-hand side expression
        , bool TF         // Transpose flag of the vectors
        , typename VT2 >  // Type of the right-hand side sparse vector
inline const MultExprTrait_< DVecScalarMultExpr<VT1,ST,TF>, VT2 >
   operator*( const DVecScalarMultExpr<VT1,ST,TF>& lhs, const SparseVector<VT2,TF>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * (~rhs) ) * lhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a sparse vector and a dense vector-
//        scalar multiplication expression (\f$ \vec{a}=\vec{b}*(\vec{c}*s1) \f$).
// \ingroup sparse_vector
//
// \param lhs The left-hand side sparse vector.
// \param rhs The right-hand side dense vector-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// sparse vector and a dense vector-scalar multiplication. It restructures the expression
// \f$ \vec{a}=\vec{b}*(\vec{c}*s1) \f$ to the expression \f$ \vec{a}=(\vec{b}*\vec{c})*s1 \f$.
*/
template< typename VT1   // Type of the left-hand side sparse vector
        , bool TF        // Transpose flag of the vectors
        , typename VT2   // Type of the dense vector of the right-hand side expression
        , typename ST >  // Type of the scalar of the right-hand side expression
inline const MultExprTrait_< VT1, DVecScalarMultExpr<VT2,ST,TF> >
   operator*( const SparseVector<VT1,TF>& lhs, const DVecScalarMultExpr<VT2,ST,TF>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( (~lhs) * rhs.leftOperand() ) * rhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense vector-scalar multiplication
//        expression and a sparse vector-scalar multiplication (\f$ \vec{a}=(\vec{b}*s1)*(\vec{c}*s2) \f$).
// \ingroup sparse_vector
//
// \param lhs The left-hand side dense vector-scalar multiplication.
// \param rhs The right-hand side sparse vector-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication
// of a dense vector-scalar multiplication and a sparse vector-scalar multiplication.
// It restructures the expression \f$ \vec{a}=(\vec{b}*s1)*(\vec{c}*s2) \f$ to the
// expression \f$ \vec{a}=(\vec{b}*\vec{c})*(s1*s2) \f$.
*/
template< typename VT1    // Type of the dense vector of the left-hand side expression
        , typename ST1    // Type of the scalar of the left-hand side expression
        , bool TF         // Transpose flag of the vectors
        , typename VT2    // Type of the sparse vector of the right-hand side expression
        , typename ST2 >  // Type of the scalar o the right-hand side expression
inline const MultExprTrait_< DVecScalarMultExpr<VT1,ST1,TF>, SVecScalarMultExpr<VT2,ST2,TF> >
   operator*( const DVecScalarMultExpr<VT1,ST1,TF>& lhs, const SVecScalarMultExpr<VT2,ST2,TF>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * rhs.leftOperand() ) * ( lhs.rightOperand() * rhs.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a sparse vector-scalar multiplication
//        expression and a dense vector-scalar multiplication (\f$ \vec{a}=(\vec{b}*s1)*(\vec{c}*s2) \f$).
// \ingroup sparse_vector
//
// \param lhs The left-hand side sparse vector-scalar multiplication.
// \param rhs The right-hand side dense vector-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication
// of a sparse vector-scalar multiplication and a dense vector-scalar multiplication.
// It restructures the expression \f$ \vec{a}=(\vec{b}*s1)*(\vec{c}*s2) \f$ to the
// expression \f$ \vec{a}=(\vec{b}*\vec{c})*(s1*s2) \f$.
*/
template< typename VT1    // Type of the sparse vector of the left-hand side expression
        , typename ST1    // Type of the scalar of the left-hand side expression
        , bool TF         // Transpose flag of the vectors
        , typename VT2    // Type of the dense vector of the right-hand side expression
        , typename ST2 >  // Type of the scalar o the right-hand side expression
inline const MultExprTrait_< SVecScalarMultExpr<VT1,ST1,TF>, DVecScalarMultExpr<VT2,ST2,TF> >
   operator*( const SVecScalarMultExpr<VT1,ST1,TF>& lhs, const DVecScalarMultExpr<VT2,ST2,TF>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * rhs.leftOperand() ) * ( lhs.rightOperand() * rhs.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the outer product of a dense vector-scalar multiplication
//        expression and a sparse vector (\f$ A=(\vec{b}*s1)*\vec{c}^T \f$).
// \ingroup sparse_matrix
//
// \param lhs The left-hand side dense vector-scalar multiplication.
// \param rhs The right-hand side sparse vector.
// \return The scaled result matrix.
//
// This operator implements the performance optimized treatment of the outer product of a
// dense vector-scalar multiplication and a sparse vector. It restructures the expression
// \f$ A=(\vec{b}*s1)*\vec{c}^T \f$ to the expression \f$ A=(\vec{b}*\vec{c}^T)*s1 \f$.
*/
template< typename VT1    // Type of the dense vector of the left-hand side expression
        , typename ST     // Type of the scalar of the left-hand side expression
        , typename VT2 >  // Type of the right-hand side sparse vector
inline const MultExprTrait_< DVecScalarMultExpr<VT1,ST,false>, VT2 >
   operator*( const DVecScalarMultExpr<VT1,ST,false>& lhs, const SparseVector<VT2,true>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * (~rhs) ) * lhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the outer product of a sparse vector and a dense vector-
//        scalar multiplication expression (\f$ A=\vec{b}*(\vec{c}^T*s1) \f$).
// \ingroup sparse_matrix
//
// \param lhs The left-hand side sparse vector.
// \param rhs The right-hand side dense vector-scalar multiplication.
// \return The scaled result matrix.
//
// This operator implements the performance optimized treatment of the outer product of a
// sparse vector and a dense vector-scalar multiplication. It restructures the expression
// \f$ A=\vec{b}*(\vec{c}^T*s1) \f$ to the expression \f$ A=(\vec{b}*\vec{c}^T)*s1 \f$.
*/
template< typename VT1   // Type of the left-hand side sparse vector
        , typename VT2   // Type of the dense vector of the right-hand side expression
        , typename ST >  // Type of the scalar of the right-hand side expression
inline const MultExprTrait_< VT1, DVecScalarMultExpr<VT2,ST,true> >
   operator*( const SparseVector<VT1,false>& lhs, const DVecScalarMultExpr<VT2,ST,true>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( (~lhs) * rhs.leftOperand() ) * rhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the outer product of a dense vector-scalar multiplication
//        expression and a sparse vector-scalar multiplication (\f$ A=(\vec{b}*s1)*(\vec{c}^T*s2) \f$).
// \ingroup sparse_matrix
//
// \param lhs The left-hand side dense vector-scalar multiplication.
// \param rhs The right-hand side sparse vector-scalar multiplication.
// \return The scaled result matrix.
//
// This operator implements the performance optimized treatment of the outer product
// of a dense vector-scalar multiplication and a sparse vector-scalar multiplication.
// It restructures the expression \f$ A=(\vec{b}*s1)*(\vec{c}^T*s2) \f$ to the
// expression \f$ A=(\vec{b}*\vec{c}^T)*(s1*s2) \f$.
*/
template< typename VT1    // Type of the dense vector of the left-hand side expression
        , typename ST1    // Type of the scalar of the left-hand side expression
        , typename VT2    // Type of the sparse vector of the right-hand side expression
        , typename ST2 >  // Type of the scalar o the right-hand side expression
inline const MultExprTrait_< DVecScalarMultExpr<VT1,ST1,false>, SVecScalarMultExpr<VT2,ST2,true> >
   operator*( const DVecScalarMultExpr<VT1,ST1,false>& lhs, const SVecScalarMultExpr<VT2,ST2,true>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * rhs.leftOperand() ) * ( lhs.rightOperand() * rhs.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the outer product of a sparse vector-scalar multiplication
//        expression and a dense vector-scalar multiplication (\f$ A=(\vec{b}*s1)*(\vec{c}^T*s2) \f$).
// \ingroup sparse_matrix
//
// \param lhs The left-hand side sparse vector-scalar multiplication.
// \param rhs The right-hand side dense vector-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication
// of a sparse vector-scalar multiplication and a dense vector-scalar multiplication.
// It restructures the expression \f$ A=(\vec{b}*s1)*(\vec{c}^T*s2) \f$ to the
// expression \f$ A=(\vec{b}*\vec{c}^T)*(s1*s2) \f$.
*/
template< typename VT1    // Type of the sparse vector of the left-hand side expression
        , typename ST1    // Type of the scalar of the left-hand side expression
        , typename VT2    // Type of the dense vector of the right-hand side expression
        , typename ST2 >  // Type of the scalar o the right-hand side expression
inline const MultExprTrait_< SVecScalarMultExpr<VT1,ST1,false>, DVecScalarMultExpr<VT2,ST2,true> >
   operator*( const SVecScalarMultExpr<VT1,ST1,false>& lhs, const DVecScalarMultExpr<VT2,ST2,true>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * rhs.leftOperand() ) * ( lhs.rightOperand() * rhs.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense matrix and a dense
//        vector-scalar multiplication expression (\f$ \vec{a}=B*(\vec{c}*s1) \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side dense matrix.
// \param rhs The right-hand side dense vector-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// dense matrix and a dense vector-scalar multiplication. It restructures the expression
// \f$ \vec{a}=B*(\vec{c}*s1) \f$ to the expression \f$ \vec{a}=(B*\vec{c})*s1 \f$.
*/
template< typename MT    // Type of the left-hand side dense matrix
        , bool SO        // Storage order of the left-hand side dense matrix
        , typename VT    // Type of the dense vector of the right-hand side expression
        , typename ST >  // Type of the scalar of the right-hand side expression
inline const MultExprTrait_< MT, DVecScalarMultExpr<VT,ST,false> >
   operator*( const DenseMatrix<MT,SO>& mat, const DVecScalarMultExpr<VT,ST,false>& vec )
{
   BLAZE_FUNCTION_TRACE;

   return ( (~mat) * vec.leftOperand() ) * vec.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a transpose dense vector-scalar
//        multiplication expression and a dense matrix (\f$ \vec{a}^T=(\vec{b}^T*s1)*C \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side transpose dense vector-scalar multiplication.
// \param rhs The right-hand side dense matrix.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// transpose dense vector-scalar multiplication and a dense matrix. It restructures the
// expression \f$ \vec{a}^T=(\vec{b}^T*s1)*C \f$ to the expression \f$ \vec{a}^T=(\vec{b}^T*C)*s1 \f$.
*/
template< typename VT  // Type of the dense vector of the left-hand side expression
        , typename ST  // Type of the scalar of the left-hand side expression
        , typename MT  // Type of the right-hand side dense matrix
        , bool SO >    // Storage order of the right-hand side dense matrix
inline const MultExprTrait_< DVecScalarMultExpr<VT,ST,true>, MT >
   operator*( const DVecScalarMultExpr<VT,ST,true>& vec, const DenseMatrix<MT,SO>& mat )
{
   BLAZE_FUNCTION_TRACE;

   return ( vec.leftOperand() * (~mat) ) * vec.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a sparse matrix and a dense
//        vector-scalar multiplication expression (\f$ \vec{a}=B*(\vec{c}*s1) \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side sparse matrix.
// \param rhs The right-hand side dense vector-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// sparse matrix and a dense vector-scalar multiplication. It restructures the expression
// \f$ \vec{a}=B*(\vec{c}*s1) \f$ to the expression \f$ \vec{a}=(B*\vec{c})*s1 \f$.
*/
template< typename MT    // Type of the left-hand side sparse matrix
        , bool SO        // Storage order of the left-hand side sparse matrix
        , typename VT    // Type of the dense vector of the right-hand side expression
        , typename ST >  // Type of the scalar of the right-hand side expression
inline const MultExprTrait_< MT, DVecScalarMultExpr<VT,ST,false> >
   operator*( const SparseMatrix<MT,SO>& mat, const DVecScalarMultExpr<VT,ST,false>& vec )
{
   BLAZE_FUNCTION_TRACE;

   return ( (~mat) * vec.leftOperand() ) * vec.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a transpose dense vector-scalar
//        multiplication expression and a sparse matrix (\f$ \vec{a}^T=(\vec{b}^T*s1)*C \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side transpose dense vector-scalar multiplication.
// \param rhs The right-hand side sparse matrix.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// transpose dense vector-scalar multiplication and a sparse matrix. It restructures the
// expression \f$ \vec{a}^T=(\vec{b}^T*s1)*C \f$ to the expression \f$ \vec{a}^T=(\vec{b}^T*C)*s1 \f$.
*/
template< typename VT  // Type of the dense vector of the left-hand side expression
        , typename ST  // Type of the scalar of the left-hand side expression
        , typename MT  // Type of the right-hand side sparse matrix
        , bool SO >    // Storage order of the right-hand side sparse matrix
inline const MultExprTrait_< DVecScalarMultExpr<VT,ST,true>, MT >
   operator*( const DVecScalarMultExpr<VT,ST,true>& vec, const SparseMatrix<MT,SO>& mat )
{
   BLAZE_FUNCTION_TRACE;

   return ( vec.leftOperand() * (~mat) ) * vec.rightOperand();
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SIZE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename ST, bool TF >
struct Size< DVecScalarMultExpr<VT,ST,TF> > : public Size<VT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISALIGNED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename ST, bool TF >
struct IsAligned< DVecScalarMultExpr<VT,ST,TF> >
   : public BoolConstant< IsAligned<VT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISPADDED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename ST, bool TF >
struct IsPadded< DVecScalarMultExpr<VT,ST,TF> >
   : public BoolConstant< IsPadded<VT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DVECSCALARMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename ST1, typename ST2 >
struct DVecScalarMultExprTrait< DVecScalarMultExpr<VT,ST1,false>, ST2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT>, IsColumnVector<VT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , DVecScalarMultExprTrait_< VT, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TDVECSCALARMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename ST1, typename ST2 >
struct TDVecScalarMultExprTrait< DVecScalarMultExpr<VT,ST1,true>, ST2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT>, IsRowVector<VT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , TDVecScalarMultExprTrait_< VT, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DVECSCALARDIVEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename ST1, typename ST2 >
struct DVecScalarDivExprTrait< DVecScalarMultExpr<VT,ST1,false>, ST2 >
{
 private:
   //**********************************************************************************************
   typedef DivTrait_<ST1,ST2>  ScalarType;
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT>, IsColumnVector<VT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , If_< IsInvertible<ScalarType>
                        , DVecScalarMultExprTrait_<VT,ScalarType>
                        , DVecScalarDivExprTrait_<VT,ScalarType> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TDVECSCALARDIVEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename ST1, typename ST2 >
struct TDVecScalarDivExprTrait< DVecScalarMultExpr<VT,ST1,true>, ST2 >
{
 private:
   //**********************************************************************************************
   typedef DivTrait_<ST1,ST2>  ScalarType;
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT>, IsRowVector<VT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , If_< IsInvertible<ScalarType>
                        , TDVecScalarMultExprTrait_<VT,ScalarType>
                        , TDVecScalarDivExprTrait_<VT,ScalarType> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DVECDVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST, typename VT2 >
struct DVecDVecMultExprTrait< DVecScalarMultExpr<VT1,ST,false>, VT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsColumnVector<VT1>
                        , IsDenseVector<VT2>, IsColumnVector<VT2>
                        , IsNumeric<ST> >
                   , DVecScalarMultExprTrait_< DVecDVecMultExprTrait_<VT1,VT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2, typename ST >
struct DVecDVecMultExprTrait< VT1, DVecScalarMultExpr<VT2,ST,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsColumnVector<VT1>
                        , IsDenseVector<VT2>, IsColumnVector<VT2>
                        , IsNumeric<ST> >
                   , DVecScalarMultExprTrait_< DVecDVecMultExprTrait_<VT1,VT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST1, typename VT2, typename ST2 >
struct DVecDVecMultExprTrait< DVecScalarMultExpr<VT1,ST1,false>, DVecScalarMultExpr<VT2,ST2,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsColumnVector<VT1>
                        , IsDenseVector<VT2>, IsColumnVector<VT2>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , DVecScalarMultExprTrait_< DVecDVecMultExprTrait_<VT1,VT2>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DVECTDVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST, typename VT2 >
struct DVecTDVecMultExprTrait< DVecScalarMultExpr<VT1,ST,false>, VT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsColumnVector<VT1>
                        , IsDenseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST> >
                   , DMatScalarMultExprTrait_< DVecTDVecMultExprTrait_<VT1,VT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2, typename ST >
struct DVecTDVecMultExprTrait< VT1, DVecScalarMultExpr<VT2,ST,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsColumnVector<VT1>
                        , IsDenseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST> >
                   , DMatScalarMultExprTrait_< DVecTDVecMultExprTrait_<VT1,VT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST1, typename VT2, typename ST2 >
struct DVecTDVecMultExprTrait< DVecScalarMultExpr<VT1,ST1,false>, DVecScalarMultExpr<VT2,ST2,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsColumnVector<VT1>
                        , IsDenseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , DMatScalarMultExprTrait_< DVecTDVecMultExprTrait_<VT1,VT2>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TDVECTDVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST, typename VT2 >
struct TDVecTDVecMultExprTrait< DVecScalarMultExpr<VT1,ST,true>, VT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsRowVector<VT1>
                        , IsDenseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST> >
                   , TDVecScalarMultExprTrait_< TDVecTDVecMultExprTrait_<VT1,VT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2, typename ST >
struct TDVecTDVecMultExprTrait< VT1, DVecScalarMultExpr<VT2,ST,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsRowVector<VT1>
                        , IsDenseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST> >
                   , TDVecScalarMultExprTrait_< TDVecTDVecMultExprTrait_<VT1,VT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST1, typename VT2, typename ST2 >
struct TDVecTDVecMultExprTrait< DVecScalarMultExpr<VT1,ST1,true>, DVecScalarMultExpr<VT2,ST2,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsRowVector<VT1>
                        , IsDenseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , TDVecScalarMultExprTrait_< TDVecTDVecMultExprTrait_<VT1,VT2>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DVECSVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2, typename ST >
struct DVecSVecMultExprTrait< DVecScalarMultExpr<VT1,ST,false>, VT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsColumnVector<VT1>
                        , IsSparseVector<VT2>, IsColumnVector<VT2>
                        , IsNumeric<ST> >
                   , SVecScalarMultExprTrait_< DVecSVecMultExprTrait_<VT1,VT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST1, typename VT2, typename ST2 >
struct DVecSVecMultExprTrait< DVecScalarMultExpr<VT1,ST1,false>, SVecScalarMultExpr<VT2,ST2,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsColumnVector<VT1>
                        , IsSparseVector<VT2>, IsColumnVector<VT2>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , SVecScalarMultExprTrait_< DVecSVecMultExprTrait_<VT1,VT2>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DVECTSVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST, typename VT2 >
struct DVecTSVecMultExprTrait< DVecScalarMultExpr<VT1,ST,false>, VT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsColumnVector<VT1>
                        , IsSparseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST> >
                   , SMatScalarMultExprTrait_< DVecTSVecMultExprTrait_<VT1,VT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST1, typename VT2, typename ST2 >
struct DVecTSVecMultExprTrait< DVecScalarMultExpr<VT1,ST1,false>, SVecScalarMultExpr<VT2,ST2,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsColumnVector<VT1>
                        , IsSparseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , SMatScalarMultExprTrait_< DVecTSVecMultExprTrait_<VT1,VT2>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TDVECTSVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST, typename VT2 >
struct TDVecTSVecMultExprTrait< DVecScalarMultExpr<VT1,ST,true>, VT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsRowVector<VT1>
                        , IsSparseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST> >
                   , TSVecScalarMultExprTrait_< TDVecTSVecMultExprTrait_<VT1,VT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST1, typename VT2, typename ST2 >
struct TDVecTSVecMultExprTrait< DVecScalarMultExpr<VT1,ST1,true>, SVecScalarMultExpr<VT2,ST2,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT1>, IsRowVector<VT1>
                        , IsSparseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , TSVecScalarMultExprTrait_< TDVecTSVecMultExprTrait_<VT1,VT2>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SVECDVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2, typename ST >
struct SVecDVecMultExprTrait< VT1, DVecScalarMultExpr<VT2,ST,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT1>, IsColumnVector<VT1>
                        , IsDenseVector<VT2>, IsColumnVector<VT2>
                        , IsNumeric<ST> >
                   , SVecScalarMultExprTrait_< SVecDVecMultExprTrait_<VT1,VT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST1, typename VT2, typename ST2 >
struct SVecDVecMultExprTrait< SVecScalarMultExpr<VT1,ST1,false>, DVecScalarMultExpr<VT2,ST2,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT1>, IsColumnVector<VT1>
                        , IsDenseVector<VT2>, IsColumnVector<VT2>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , SVecScalarMultExprTrait_< SVecDVecMultExprTrait_<VT1,VT2>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SVECTDVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2, typename ST >
struct SVecTDVecMultExprTrait< VT1, DVecScalarMultExpr<VT2,ST,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT1>, IsColumnVector<VT1>
                        , IsDenseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST> >
                   , TSMatScalarMultExprTrait_< SVecTDVecMultExprTrait_<VT1,VT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST1, typename VT2, typename ST2 >
struct SVecTDVecMultExprTrait< SVecScalarMultExpr<VT1,ST1,false>, DVecScalarMultExpr<VT2,ST2,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT1>, IsColumnVector<VT1>
                        , IsDenseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , TSMatScalarMultExprTrait_< SVecTDVecMultExprTrait_<VT1,VT2>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TSVECTDVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2, typename ST >
struct TSVecTDVecMultExprTrait< VT1, DVecScalarMultExpr<VT2,ST,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT1>, IsRowVector<VT1>
                        , IsDenseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST> >
                   , TSVecScalarMultExprTrait_< TSVecTDVecMultExprTrait_<VT1,VT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename ST1, typename VT2, typename ST2 >
struct TSVecTDVecMultExprTrait< SVecScalarMultExpr<VT1,ST1,true>, DVecScalarMultExpr<VT2,ST2,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT1>, IsRowVector<VT1>
                        , IsDenseVector<VT2>, IsRowVector<VT2>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , TSVecScalarMultExprTrait_< TSVecTDVecMultExprTrait_<VT1,VT2>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DMATDVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename VT, typename ST >
struct DMatDVecMultExprTrait< MT, DVecScalarMultExpr<VT,ST,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseMatrix<MT>, IsRowMajorMatrix<MT>
                        , IsDenseVector<VT>, IsColumnVector<VT>
                        , IsNumeric<ST> >
                   , DVecScalarMultExprTrait_< DMatDVecMultExprTrait_<MT,VT>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TDMATDVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename VT, typename ST >
struct TDMatDVecMultExprTrait< MT, DVecScalarMultExpr<VT,ST,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseMatrix<MT>, IsColumnMajorMatrix<MT>
                        , IsDenseVector<VT>, IsColumnVector<VT>
                        , IsNumeric<ST> >
                   , DVecScalarMultExprTrait_< TDMatDVecMultExprTrait_<MT,VT>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TDVECDMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename MT, typename ST >
struct TDVecDMatMultExprTrait< DVecScalarMultExpr<VT,ST,true>, MT >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT>, IsRowVector<VT>
                        , IsDenseMatrix<MT>, IsRowMajorMatrix<MT>
                        , IsNumeric<ST> >
                   , TDVecScalarMultExprTrait_< TDVecDMatMultExprTrait_<VT,MT>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TDVECTDMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename MT, typename ST >
struct TDVecTDMatMultExprTrait< DVecScalarMultExpr<VT,ST,true>, MT >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT>, IsRowVector<VT>
                        , IsDenseMatrix<MT>, IsColumnMajorMatrix<MT>
                        , IsNumeric<ST> >
                   , TDVecScalarMultExprTrait_< TDVecTDMatMultExprTrait_<VT,MT>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SMATDVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename VT, typename ST >
struct SMatDVecMultExprTrait< MT, DVecScalarMultExpr<VT,ST,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsRowMajorMatrix<MT>
                        , IsDenseVector<VT>, IsColumnVector<VT>
                        , IsNumeric<ST> >
                   , DVecScalarMultExprTrait_< SMatDVecMultExprTrait_<MT,VT>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TSMATDVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename VT, typename ST >
struct TSMatDVecMultExprTrait< MT, DVecScalarMultExpr<VT,ST,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsColumnMajorMatrix<MT>
                        , IsDenseVector<VT>, IsColumnVector<VT>
                        , IsNumeric<ST> >
                   , DVecScalarMultExprTrait_< TSMatDVecMultExprTrait_<MT,VT>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TDVECSMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename MT, typename ST >
struct TDVecSMatMultExprTrait< DVecScalarMultExpr<VT,ST,true>, MT >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT>, IsRowVector<VT>
                        , IsSparseMatrix<MT>, IsRowMajorMatrix<MT>
                        , IsNumeric<ST> >
                   , TDVecScalarMultExprTrait_< TDVecSMatMultExprTrait_<VT,MT>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TDVECTSMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename MT, typename ST >
struct TDVecTSMatMultExprTrait< DVecScalarMultExpr<VT,ST,true>, MT >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT>, IsRowVector<VT>
                        , IsSparseMatrix<MT>, IsColumnMajorMatrix<MT>
                        , IsNumeric<ST> >
                   , TDVecScalarMultExprTrait_< TDVecTSMatMultExprTrait_<VT,MT>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBVECTOREXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename ST, bool TF, bool AF >
struct SubvectorExprTrait< DVecScalarMultExpr<VT,ST,TF>, AF >
{
 public:
   //**********************************************************************************************
   using Type = MultExprTrait_< SubvectorExprTrait_<const VT,AF>, ST >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
