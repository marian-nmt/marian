//=================================================================================================
/*!
//  \file blaze/math/expressions/SVecForEachExpr.h
//  \brief Header file for the sparse vector for-each expression
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

#ifndef _BLAZE_MATH_EXPRESSIONS_SVECFOREACHEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_SVECFOREACHEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <utility>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/SparseVector.h>
#include <blaze/math/constraints/TransposeFlag.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/SparseVector.h>
#include <blaze/math/expressions/VecForEachExpr.h>
#include <blaze/math/Functors.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/sparse/ValueIndexPair.h>
#include <blaze/math/traits/CTransExprTrait.h>
#include <blaze/math/traits/ForEachExprTrait.h>
#include <blaze/math/traits/ForEachTrait.h>
#include <blaze/math/traits/SubvectorExprTrait.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/math/typetraits/UnderlyingNumeric.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/logging/FunctionTrace.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Not.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsSame.h>
#include <blaze/util/typetraits/RemoveReference.h>


namespace blaze {

//=================================================================================================
//
//  CLASS SVECFOREACHEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for the sparse vector forEach() function.
// \ingroup sparse_vector_expression
//
// The SVecForEachExpr class represents the compile time expression for the evaluation of a
// custom operation on each element of a sparse vector via the forEach() function.
*/
template< typename VT  // Type of the sparse vector
        , typename OP  // Type of the custom operation
        , bool TF >    // Transpose flag
class SVecForEachExpr : public SparseVector< SVecForEachExpr<VT,OP,TF>, TF >
                      , private VecForEachExpr
                      , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef ResultType_<VT>  RT;  //!< Result type of the sparse vector expression.
   typedef ReturnType_<VT>  RN;  //!< Return type of the sparse vector expression.
   //**********************************************************************************************

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the for-each expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for the
       serial evaluation strategy of the for-each expression. In case the given sparse vector
       expression of type \a VT requires an intermediate evaluation, \a useAssign will be set
       to 1 and the for-each expression will be evaluated via the \a assign function family.
       Otherwise \a useAssign will be set to 0 and the expression will be evaluated via the
       subscript operator. */
   enum : bool { useAssign = RequiresEvaluation<VT>::value };

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
       strategy. In case either the target vector or the sparse vector operand is not SMP
       assignable and the vector operand requires an intermediate evaluation, \a value is set
       to 1 and the expression specific evaluation strategy is selected. Otherwise \a value is
       set to 0 and the default strategy is chosen. */
   template< typename VT2 >
   struct UseSMPAssign {
      enum : bool { value = ( !VT2::smpAssignable || !VT::smpAssignable ) && useAssign };
   };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef SVecForEachExpr<VT,OP,TF>   This;           //!< Type of this SVecForEachExpr instance.
   typedef ForEachTrait_<VT,OP>        ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>  TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<ResultType>    ElementType;    //!< Resulting element type.

   //! Return type for expression template evaluations.
   typedef decltype( std::declval<OP>()( std::declval<RN>() ) )  ReturnType;

   //! Data type for composite expression templates.
   typedef IfTrue_< useAssign, const ResultType, const SVecForEachExpr& >  CompositeType;

   //! Composite data type of the sparse vector expression.
   typedef If_< IsExpression<VT>, const VT, const VT& >  Operand;

   //! Data type of the custom unary operation.
   typedef OP  Operation;
   //**********************************************************************************************

   //**ConstIterator class definition**************************************************************
   /*!\brief Iterator over the elements of the sparse vector for-each expression.
   */
   class ConstIterator
   {
    public:
      //**Type definitions*************************************************************************
      //! Element type of the sparse vector expression.
      typedef ValueIndexPair<ElementType>  Element;

      //! Iterator type of the sparse vector expression.
      typedef ConstIterator_< RemoveReference_<Operand> >  IteratorType;

      typedef std::forward_iterator_tag  IteratorCategory;  //!< The iterator category.
      typedef Element                    ValueType;         //!< Type of the underlying pointers.
      typedef ValueType*                 PointerType;       //!< Pointer return type.
      typedef ValueType&                 ReferenceType;     //!< Reference return type.
      typedef ptrdiff_t                  DifferenceType;    //!< Difference between two iterators.

      // STL iterator requirements
      typedef IteratorCategory  iterator_category;  //!< The iterator category.
      typedef ValueType         value_type;         //!< Type of the underlying pointers.
      typedef PointerType       pointer;            //!< Pointer return type.
      typedef ReferenceType     reference;          //!< Reference return type.
      typedef DifferenceType    difference_type;    //!< Difference between two iterators.
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the ConstIterator class.
      //
      // \param it Iterator to the initial vector element.
      // \param op The custom unary operation.
      */
      inline ConstIterator( IteratorType it, OP op )
         : it_( it )  // Iterator over the elements of the sparse vector expression
         , op_( op )  // The custom unary operation
      {}
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented expression iterator.
      */
      inline ConstIterator& operator++() {
         ++it_;
         return *this;
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the sparse vector element at the current iterator position.
      //
      // \return The current value of the sparse element.
      */
      inline const Element operator*() const {
         return Element( op_( it_->value() ), it_->index() );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the sparse vector element at the current iterator position.
      //
      // \return Reference to the sparse vector element at the current iterator position.
      */
      inline const ConstIterator* operator->() const {
         return this;
      }
      //*******************************************************************************************

      //**Value function***************************************************************************
      /*!\brief Access to the current value of the sparse element.
      //
      // \return The current value of the sparse element.
      */
      inline ReturnType value() const {
         return op_( it_->value() );
      }
      //*******************************************************************************************

      //**Index function***************************************************************************
      /*!\brief Access to the current index of the sparse element.
      //
      // \return The current index of the sparse element.
      */
      inline size_t index() const {
         return it_->index();
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side expression iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const ConstIterator& rhs ) const {
         return it_ == rhs.it_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side expression iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const ConstIterator& rhs ) const {
         return it_ != rhs.it_;
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two expression iterators.
      //
      // \param rhs The right-hand side expression iterator.
      // \return The number of elements between the two expression iterators.
      */
      inline DifferenceType operator-( const ConstIterator& rhs ) const {
         return it_ - rhs.it_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType it_;  //!< Iterator over the elements of the sparse vector expression.
      OP           op_;  //!< The custom unary operation.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = VT::smpAssignable };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SVecForEachExpr class.
   //
   // \param sv The sparse vector operand of the for-each expression.
   // \param op The custom unary operation.
   */
   explicit inline SVecForEachExpr( const VT& sv, OP op ) noexcept
      : sv_( sv )  // Sparse vector of the for-each expression
      , op_( op )  // The custom unary operation
   {}
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the vector elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator[]( size_t index ) const {
      BLAZE_INTERNAL_ASSERT( index < sv_.size(), "Invalid vector access index" );
      return op_( sv_[index] );
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
      if( index >= sv_.size() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid vector access index" );
      }
      return (*this)[index];
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of the sparse vector.
   //
   // \return Iterator to the first non-zero element of the sparse vector.
   */
   inline ConstIterator begin() const {
      return ConstIterator( sv_.begin(), op_ );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of the sparse vector.
   //
   // \return Iterator just past the last non-zero element of the sparse vector.
   */
   inline ConstIterator end() const {
      return ConstIterator( sv_.end(), op_ );
   }
   //**********************************************************************************************

   //**Size function*******************************************************************************
   /*!\brief Returns the current size/dimension of the vector.
   //
   // \return The size of the vector.
   */
   inline size_t size() const noexcept {
      return sv_.size();
   }
   //**********************************************************************************************

   //**NonZeros function***************************************************************************
   /*!\brief Returns the number of non-zero elements in the sparse vector.
   //
   // \return The number of non-zero elements in the sparse vector.
   */
   inline size_t nonZeros() const {
      return sv_.nonZeros();
   }
   //**********************************************************************************************

   //**Find function*******************************************************************************
   /*!\brief Searches for a specific vector element.
   //
   // \param index The index of the search element.
   // \return Iterator to the element in case the index is found, end() iterator otherwise.
   */
   inline ConstIterator find( size_t index ) const {
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );
      return ConstIterator( sv_.find( index ), op_ );
   }
   //**********************************************************************************************

   //**LowerBound function*************************************************************************
   /*!\brief Returns an iterator to the first index not less then the given index.
   //
   // \param index The index of the search element.
   // \return Iterator to the first index not less then the given index, end() iterator otherwise.
   */
   inline ConstIterator lowerBound( size_t index ) const {
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );
      return ConstIterator( sv_.lowerBound( index ), op_ );
   }
   //**********************************************************************************************

   //**UpperBound function*************************************************************************
   /*!\brief Returns an iterator to the first index greater then the given index.
   //
   // \param index The index of the search element.
   // \return Iterator to the first index greater then the given index, end() iterator otherwise.
   */
   inline ConstIterator upperBound( size_t index ) const {
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );
      return ConstIterator( sv_.upperBound( index ), op_ );
   }
   //**********************************************************************************************

   //**Operand access******************************************************************************
   /*!\brief Returns the sparse vector operand.
   //
   // \return The sparse vector operand.
   */
   inline Operand operand() const noexcept {
      return sv_;
   }
   //**********************************************************************************************

   //**Operation access****************************************************************************
   /*!\brief Returns a copy of the custom operation.
   //
   // \return A copy of the custom operation.
   */
   inline Operation operation() const {
      return op_;
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
      return sv_.canAlias( alias );
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
      return sv_.isAliased( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return sv_.canSMPAssign();
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   Operand   sv_;  //!< Sparse vector of the for-each expression.
   Operation op_;  //!< The custom unary operation.
   //**********************************************************************************************

   //**Assignment to dense vectors*****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse vector for-each expression to a dense vector.
   // \ingroup sparse_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side for-each expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse vector for-each
   // expression to a dense vector. Due to the explicit application of the SFINAE principle, this
   // function can only be selected by the compiler in case the operand requires an intermediate
   // evaluation.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseAssign<VT2> >
      assign( DenseVector<VT2,TF>& lhs, const SVecForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( RT, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const RT tmp( serial( rhs.sv_ ) );
      assign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to sparse vectors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse vector for-each expression to a sparse vector.
   // \ingroup sparse_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side for-each expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse vector for-each
   // expression to a sparse vector. Due to the explicit application of the SFINAE principle, this
   // function can only be selected by the compiler in case the operand requires an intermediate
   // evaluation and the underlying numeric data type of the operand and the target vector are
   // identical.
   */
   template< typename VT2 >  // Type of the target sparse vector
   friend inline EnableIf_< And< UseAssign<VT2>
                               , IsSame< UnderlyingNumeric<VT>, UnderlyingNumeric<VT2> > > >
      assign( SparseVector<VT2,TF>& lhs, const SVecForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      typedef Iterator_<VT2>  Iterator;

      assign( ~lhs, rhs.sv_ );

      const Iterator end( (~lhs).end() );
      for( Iterator element=(~lhs).begin(); element!=end; ++element ) {
         element->value() = rhs.op_( element->value() );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to sparse vectors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse vector for-each expression to a sparse vector.
   // \ingroup sparse_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side for-each expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse vector for-each
   // expression to a sparse vector. Due to the explicit application of the SFINAE principle, this
   // function can only be selected by the compiler in case the operand requires an intermediate
   // evaluation and the underlying numeric data type of the operand and the target vector differ.
   */
   template< typename VT2 >  // Type of the target sparse vector
   friend inline EnableIf_< And< UseAssign<VT2>
                               , Not< IsSame< UnderlyingNumeric<VT>, UnderlyingNumeric<VT2> > > > >
      assign( SparseVector<VT2,TF>& lhs, const SVecForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( RT, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const RT tmp( serial( rhs.sv_ ) );
      (~lhs).reserve( tmp.nonZeros() );
      assign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense vectors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a sparse vector for-each expression to a dense vector.
   // \ingroup sparse_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side for-each expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a sparse vector
   // for-each expression to a dense vector. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the operand requires
   // an intermediate evaluation.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseAssign<VT2> >
      addAssign( DenseVector<VT2,TF>& lhs, const SVecForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( RT, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const RT tmp( serial( rhs.sv_ ) );
      addAssign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to sparse vectors*******************************************************
   // No special implementation for the addition assignment to sparse vectors.
   //**********************************************************************************************

   //**Subtraction assignment to dense vectors*****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a sparse vector for-each expression to a dense vector.
   // \ingroup sparse_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side for-each expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a sparse
   // vector for-each expression to a dense vector. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the
   // operand requires an intermediate evaluation.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseAssign<VT2> >
      subAssign( DenseVector<VT2,TF>& lhs, const SVecForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( RT, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const RT tmp( serial( rhs.sv_ ) );
      subAssign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to sparse vectors****************************************************
   // No special implementation for the subtraction assignment to sparse vectors.
   //**********************************************************************************************

   //**Multiplication assignment to dense vectors**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Multiplication assignment of a sparse vector for-each expression to a dense vector.
   // \ingroup sparse_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side for-each expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a sparse
   // vector for-each expression to a dense vector. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the operand requires
   // an intermediate evaluation.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseAssign<VT2> >
      multAssign( DenseVector<VT2,TF>& lhs, const SVecForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( RT, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const RT tmp( serial( rhs.sv_ ) );
      multAssign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Multiplication assignment to sparse vectors*************************************************
   // No special implementation for the multiplication assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP assignment to dense vectors*************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a sparse vector for-each expression to a dense vector.
   // \ingroup sparse_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side for-each expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a sparse vector
   // for-each expression to a dense vector. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT2> >
      smpAssign( DenseVector<VT2,TF>& lhs, const SVecForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( RT, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const RT tmp( rhs.sv_ );
      smpAssign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to sparse vectors************************************************************
   // No special implementation for the SMP assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP addition assignment to dense vectors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a sparse vector for-each expression to a dense vector.
   // \ingroup sparse_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side for-each expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a sparse
   // vector for-each expression to a dense vector. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT2> >
      smpAddAssign( DenseVector<VT2,TF>& lhs, const SVecForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( RT, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const RT tmp( rhs.sv_ );
      smpAddAssign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to sparse vectors***************************************************
   // No special implementation for the SMP addition assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP subtraction assignment to dense vectors*************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a sparse vector for-each expression to a dense vector.
   // \ingroup sparse_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side for-each expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a sparse
   // vector for-each expression to a dense vector. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT2> >
      smpSubAssign( DenseVector<VT2,TF>& lhs, const SVecForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( RT, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const RT tmp( rhs.sv_ );
      smpSubAssign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to sparse vectors************************************************
   // No special implementation for the SMP subtraction assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP multiplication assignment to dense vectors**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP multiplication assignment of a sparse vector for-each expression to a dense vector.
   // \ingroup sparse_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side for-each expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a
   // sparse vector for-each expression to a dense vector. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT2 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT2> >
      smpMultAssign( DenseVector<VT2,TF>& lhs, const SVecForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( RT, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const RT tmp( rhs.sv_ );
      smpMultAssign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP multiplication assignment to sparse vectors*********************************************
   // No special implementation for the SMP multiplication assignment to sparse vectors.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( VT );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( VT, TF );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Evaluates the given custom operation on each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \param op The custom operation.
// \return The custom operation applied to each single element of \a sv.
//
// The \a forEach() function evaluates the given custom operation on each non-zero element of the
// input vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a forEach() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = forEach( a, []( double a ){ return std::sqrt( a ); } );
   \endcode
*/
template< typename VT    // Type of the sparse vector
        , bool TF        // Transpose flag
        , typename OP >  // Type of the custom operation
inline const SVecForEachExpr<VT,OP,TF> forEach( const SparseVector<VT,TF>& sv, OP op )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,OP,TF>( ~sv, op );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Applies the \a abs() function to each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The resulting sparse vector.
//
// This function applies the \a abs() function to each non-zero element of the input vector
// \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a abs() function:

   \code
   blaze::DynamicVector<double> a, b;
   // ... Resizing and initialization
   b = abs( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Abs,TF> abs( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Abs,TF>( ~sv, Abs() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Applies the \a floor() function to each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The resulting sparse vector.
//
// This function applies the \a floor() function to each non-zero element of the input vector
// \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a floor() function:

   \code
   blaze::DynamicVector<double> a, b;
   // ... Resizing and initialization
   b = floor( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Floor,TF> floor( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Floor,TF>( ~sv, Floor() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Applies the \a ceil() function to each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The resulting sparse vector.
//
// This function applies the \a ceil() function to each non-zero element of the input vector
// \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a ceil() function:

   \code
   blaze::DynamicVector<double> a, b;
   // ... Resizing and initialization
   b = ceil( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Ceil,TF> ceil( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Ceil,TF>( ~sv, Ceil() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a vector containing the complex conjugate of each single element of \a sv.
// \ingroup sparse_vector
//
// \param sv The integral sparse input vector.
// \return The complex conjugate of each single element of \a sv.
//
// The \a conj function calculates the complex conjugate of each element of the sparse input
// vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a conj function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = conj( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Conj,TF> conj( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Conj,TF>( ~sv, Conj() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the conjugate transpose vector of \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The conjugate transpose of \a sv.
//
// The \a ctrans function returns an expression representing the conjugate transpose (also called
// adjoint matrix, Hermitian conjugate matrix or transjugate matrix) of the given input vector
// \a sv.\n
// The following example demonstrates the use of the \a ctrans function:

   \code
   blaze::CompressedVector< complex<double> > a, b;
   // ... Resizing and initialization
   b = ctrans( a );
   \endcode

// Note that the \a ctrans function has the same effect as manually applying the \a conj and
// \a trans function in any order:

   \code
   b = trans( conj( a ) );  // Computing the conjugate transpose vector
   b = conj( trans( a ) );  // Computing the conjugate transpose vector
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const CTransExprTrait_<VT> ctrans( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return trans( conj( ~sv ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a vector containing the real parts of each single element of \a sv.
// \ingroup sparse_vector
//
// \param sv The integral sparse input vector.
// \return The real part of each single element of \a sv.
//
// The \a real function calculates the real part of each element of the sparse input vector
// \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a real function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = real( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Real,TF> real( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Real,TF>( ~sv, Real() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a vector containing the imaginary parts of each single element of \a sv.
// \ingroup sparse_vector
//
// \param sv The integral sparse input vector.
// \return The imaginary part of each single element of \a sv.
//
// The \a imag function calculates the imaginary part of each element of the sparse input vector
// \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a imag function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = imag( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Imag,TF> imag( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Imag,TF>( ~sv, Imag() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the square root of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector; all non-zero elements must be in the range \f$[0..\infty)\f$.
// \return The square root of each single element of \a sv.
//
// The \a sqrt() function computes the square root of each non-zero element of the input vector
// \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a sqrt() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = sqrt( a );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[0..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Sqrt,TF> sqrt( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Sqrt,TF>( ~sv, Sqrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse square root of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector; all non-zero elements must be in the range \f$(0..\infty)\f$.
// \return The inverse square root of each single element of \a sv.
//
// The \a invsqrt() function computes the inverse square root of each non-zero element of the
// input vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a invsqrt() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = invsqrt( a );
   \endcode

// \note All non-zero elements are expected to be in the range \f$(0..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,InvSqrt,TF> invsqrt( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,InvSqrt,TF>( ~sv, InvSqrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the cubic root of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector; all non-zero elements must be in the range \f$[0..\infty)\f$.
// \return The cubic root of each single element of \a sv.
//
// The \a cbrt() function computes the cubic root of each non-zero element of the input vector
// \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a cbrt() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = cbrt( a );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[0..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Cbrt,TF> cbrt( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Cbrt,TF>( ~sv, Cbrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse cubic root of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector; all non-zero elements must be in the range \f$(0..\infty)\f$.
// \return The inverse cubic root of each single element of \a sv.
//
// The \a invcbrt() function computes the inverse cubic root of each non-zero element of the
// input vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a invcbrt() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = invcbrt( a );
   \endcode

// \note All non-zero elements are expected to be in the range \f$(0..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,InvCbrt,TF> invcbrt( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,InvCbrt,TF>( ~sv, InvCbrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the exponential value for each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \param exp The exponent.
// \return The exponential value of each non-zero element of \a sv.
//
// The \a pow() function computes the exponential value for each non-zero element of the input
// vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a pow() function:

   \code
   blaze::DynamicVector<double> A, B;
   // ... Resizing and initialization
   B = pow( A, 4.2 );
   \endcode
*/
template< typename VT    // Type of the sparse vector
        , bool TF        // Transpose flag
        , typename ET >  // Type of the exponent
inline const SVecForEachExpr<VT,Pow<ET>,TF> pow( const SparseVector<VT,TF>& sv, ET exp )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( ET );

   return SVecForEachExpr<VT,Pow<ET>,TF>( ~sv, Pow<ET>( exp ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes \f$ e^x \f$ of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The resulting sparse vector.
//
// The \a exp() function computes \f$ e^x \f$ for each non-zero element of the input vector \a sv.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a exp() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = exp( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Exp,TF> exp( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Exp,TF>( ~sv, Exp() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the natural logarithm of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector; all non-zero elements must be in the range \f$[0..\infty)\f$.
// \return The natural logaritm of each non-zero element of \a sv.
//
// The \a log() function computes the natural logarithm for each non-zero element of the input
// vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a log() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = log( a );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[0..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Log,TF> log( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Log,TF>( ~sv, Log() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the common logarithm of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector; all non-zero elements must be in the range \f$[0..\infty)\f$.
// \return The common logaritm of each non-zero element of \a sv.
//
// The \a log10() function computes the common logarithm for each non-zero element of the input
// vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a log10() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = log10( a );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[0..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Log10,TF> log10( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Log10,TF>( ~sv, Log10() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the sine of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The sine of each non-zero element of \a sv.
//
// The \a sin() function computes the sine for each non-zero element of the input vector \a sv.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a sin() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = sin( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Sin,TF> sin( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Sin,TF>( ~sv, Sin() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse sine of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector; all non-zero elements must be in the range \f$[-1..1]\f$.
// \return The inverse sine of each non-zero element of \a sv.
//
// The \a asin() function computes the inverse sine for each non-zero element of the input vector
// \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a asin() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = asin( a );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[-1..1]\f$. No runtime checks
// are performed to assert this precondition!
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Asin,TF> asin( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Asin,TF>( ~sv, Asin() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the hyperbolic sine of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The hyperbolic sine of each non-zero element of \a sv.
//
// The \a sinh() function computes the hyperbolic sine for each non-zero element of the input
// vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a sinh() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = sinh( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Sinh,TF> sinh( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Sinh,TF>( ~sv, Sinh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse hyperbolic sine of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The inverse hyperbolic sine of each non-zero element of \a sv.
//
// The \a asinh() function computes the inverse hyperbolic sine for each non-zero element of the
// input vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a asinh() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = asinh( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Asinh,TF> asinh( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Asinh,TF>( ~sv, Asinh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the cosine of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The cosine of each non-zero element of \a sv.
//
// The \a cos() function computes the cosine for each non-zero element of the input vector \a sv.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a cos() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = cos( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Cos,TF> cos( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Cos,TF>( ~sv, Cos() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse cosine of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector; all non-zero elements must be in the range \f$[-1..1]\f$.
// \return The inverse cosine of each non-zero element of \a sv.
//
// The \a acos() function computes the inverse cosine for each non-zero element of the input vector
// \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a acos() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = acos( a );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[-1..1]\f$. No runtime checks
// are performed to assert this precondition!
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Acos,TF> acos( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Acos,TF>( ~sv, Acos() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the hyperbolic cosine of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The hyperbolic cosine of each non-zero element of \a sv.
//
// The \a cosh() function computes the hyperbolic cosine for each non-zero element of the input
// vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a cosh() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = cosh( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Cosh,TF> cosh( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Cosh,TF>( ~sv, Cosh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse hyperbolic cosine of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector; all non-zero elements must be in the range \f$[1..\infty)\f$.
// \return The inverse hyperbolic cosine of each non-zero element of \a sv.
//
// The \a acosh() function computes the inverse hyperbolic cosine for each non-zero element of
// the input vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a acosh() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = acosh( a );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[1..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Acosh,TF> acosh( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Acosh,TF>( ~sv, Acosh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the tangent of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The tangent of each non-zero element of \a sv.
//
// The \a tan() function computes the tangent for each non-zero element of the input vector \a sv.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a tan() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = tan( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Tan,TF> tan( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Tan,TF>( ~sv, Tan() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse tangent of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The inverse tangent of each non-zero element of \a sv.
//
// The \a atan() function computes the inverse tangent for each non-zero element of the input
// vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a atan() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = atan( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Atan,TF> atan( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Atan,TF>( ~sv, Atan() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the hyperbolic tangent of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector; all non-zero elements must be in the range \f$[-1..1]\f$.
// \return The hyperbolic tangent of each non-zero element of \a sv.
//
// The \a tanh() function computes the hyperbolic tangent for each non-zero element of the input
// vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a tanh() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = tanh( a );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[-1..1]\f$. No runtime checks
// are performed to assert this precondition!
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Tanh,TF> tanh( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Tanh,TF>( ~sv, Tanh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse hyperbolic tangent of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector; all non-zero elements must be in the range \f$[-1..1]\f$.
// \return The inverse hyperbolic tangent of each non-zero element of \a sv.
//
// The \a atanh() function computes the inverse hyperbolic tangent for each non-zero element of
// the input vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a atanh() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = atanh( a );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[-1..1]\f$. No runtime checks
// are performed to assert this precondition!
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Atanh,TF> atanh( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Atanh,TF>( ~sv, Atanh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the error function of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The error function of each non-zero element of \a sv.
//
// The \a erf() function computes the error function for each non-zero element of the input
// vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a erf() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = erf( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Erf,TF> erf( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Erf,TF>( ~sv, Erf() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the complementary error function of each non-zero element of the sparse vector \a sv.
// \ingroup sparse_vector
//
// \param sv The input vector.
// \return The complementary error function of each non-zero element of \a sv.
//
// The \a erfc() function computes the complementary error function for each non-zero element of
// the input vector \a sv. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a erfc() function:

   \code
   blaze::CompressedVector<double> a, b;
   // ... Resizing and initialization
   b = erfc( a );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Erfc,TF> erfc( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecForEachExpr<VT,Erfc,TF>( ~sv, Erfc() );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Absolute value function for absolute value sparse vector expressions.
// \ingroup sparse_vector
//
// \param sv The absolute value sparse vector expression.
// \return The absolute value of each single element of \a sv.
//
// This function implements a performance optimized treatment of the absolute value operation
// on a sparse vector absolute value expression.
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Abs,TF>& abs( const SVecForEachExpr<VT,Abs,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return sv;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Applies the \a floor() function to a sparse vector \a floor() expressions.
// \ingroup sparse_vector
//
// \param sv The sparse vector \a floor expression.
// \return The resulting sparse vector.
//
// This function implements a performance optimized treatment of the \a floor() operation on
// a sparse vector \a floor() expression.
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Floor,TF>& floor( const SVecForEachExpr<VT,Floor,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return sv;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Applies the \a ceil() function to a sparse vector \a ceil() expressions.
// \ingroup sparse_vector
//
// \param sv The sparse vector \a ceil expression.
// \return The resulting sparse vector.
//
// This function implements a performance optimized treatment of the \a ceil() operation on
// a sparse vector \a ceil() expression.
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Ceil,TF>& ceil( const SVecForEachExpr<VT,Ceil,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return sv;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Complex conjugate function for complex conjugate sparse vector expressions.
// \ingroup sparse_vector
//
// \param sv The complex conjugate sparse vector expression.
// \return The original sparse vector.
//
// This function implements a performance optimized treatment of the complex conjugate operation
// on a sparse vector complex conjugate expression. It returns an expression representing the
// original sparse vector:

   \code
   blaze::CompressedVector< complex<double> > a, b;
   // ... Resizing and initialization
   b = conj( conj( a ) );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline typename SVecForEachExpr<VT,Conj,TF>::Operand conj( const SVecForEachExpr<VT,Conj,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return sv.operand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Complex conjugate function for conjugate transpose sparse vector expressions.
// \ingroup sparse_vector
//
// \param sv The conjugate transpose sparse vector expression.
// \return The transpose sparse vector.
//
// This function implements a performance optimized treatment of the complex conjugate operation
// on a sparse vector conjugate transpose expression. It returns an expression representing the
// transpose of the sparse vector:

   \code
   blaze::CompressedVector< complex<double> > a, b;
   // ... Resizing and initialization
   b = conj( ctrans( a ) );
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecTransExpr<VT,!TF> conj( const SVecTransExpr<SVecForEachExpr<VT,Conj,TF>,!TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return SVecTransExpr<VT,!TF>( sv.operand().operand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Real part function for real part sparse vector expressions.
// \ingroup sparse_vector
//
// \param sv The real part sparse vector expression.
// \return The real part of each single element of \a sv.
//
// This function implements a performance optimized treatment of the real part operation on
// a sparse vector real part expression.
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline const SVecForEachExpr<VT,Real,TF>& real( const SVecForEachExpr<VT,Real,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return sv;
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
template< typename VT, typename OP, bool TF >
struct Size< SVecForEachExpr<VT,OP,TF> > : public Size<VT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  EXPRESSION TRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT >
struct SVecForEachExprTrait< SVecForEachExpr<VT,Abs,false>, Abs >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsColumnVector<VT> >
                   , SVecForEachExpr<VT,Abs,false>
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT >
struct TSVecForEachExprTrait< SVecForEachExpr<VT,Abs,true>, Abs >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsRowVector<VT> >
                   , SVecForEachExpr<VT,Abs,true>
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT >
struct SVecForEachExprTrait< SVecForEachExpr<VT,Conj,false>, Conj >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsColumnVector<VT> >
                   , Operand_< SVecForEachExpr<VT,Conj,false> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT >
struct TSVecForEachExprTrait< SVecForEachExpr<VT,Conj,true>, Conj >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsRowVector<VT> >
                   , Operand_< SVecForEachExpr<VT,Conj,true> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT >
struct SVecForEachExprTrait< SVecTransExpr< SVecForEachExpr<VT,Conj,true>, false >, Conj >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsRowVector<VT> >
                   , SVecTransExpr<VT,false>
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT >
struct TSVecForEachExprTrait< SVecTransExpr< SVecForEachExpr<VT,Conj,false>, true >, Conj >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsColumnVector<VT> >
                   , SVecTransExpr<VT,true>
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT >
struct SVecForEachExprTrait< SVecForEachExpr<VT,Real,false>, Real >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsColumnVector<VT> >
                   , SVecForEachExpr<VT,Real,false>
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT >
struct TSVecForEachExprTrait< SVecForEachExpr<VT,Real,true>, Real >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsRowVector<VT> >
                   , SVecForEachExpr<VT,Real,true>
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename OP, bool TF, bool AF >
struct SubvectorExprTrait< SVecForEachExpr<VT,OP,TF>, AF >
{
 public:
   //**********************************************************************************************
   using Type = ForEachExprTrait_< SubvectorExprTrait_<const VT,AF>, OP >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
