//=================================================================================================
/*!
//  \file blaze/math/expressions/SMatForEachExpr.h
//  \brief Header file for the sparse matrix for-each expression
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

#ifndef _BLAZE_MATH_EXPRESSIONS_SMATFOREACHEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_SMATFOREACHEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <utility>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/constraints/StorageOrder.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/MatForEachExpr.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/Functors.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/sparse/ValueIndexPair.h>
#include <blaze/math/traits/CTransExprTrait.h>
#include <blaze/math/traits/ForEachExprTrait.h>
#include <blaze/math/traits/ForEachTrait.h>
#include <blaze/math/traits/ColumnExprTrait.h>
#include <blaze/math/traits/RowExprTrait.h>
#include <blaze/math/traits/SubmatrixExprTrait.h>
#include <blaze/math/typetraits/Columns.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsHermitian.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/math/typetraits/IsUniUpper.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/math/typetraits/Rows.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
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
//  CLASS SMATFOREACHEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for the sparse matrix forEach() function.
// \ingroup sparse_matrix_expression
//
// The SMatForEachExpr class represents the compile time expression for the evaluation of a
// custom operation on each element of a sparse matrix via the forEach() function.
*/
template< typename MT  // Type of the sparse matrix
        , typename OP  // Type of the custom operation
        , bool SO >    // Storage order
class SMatForEachExpr : public SparseMatrix< SMatForEachExpr<MT,OP,SO>, SO >
                      , private MatForEachExpr
                      , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef ResultType_<MT>    RT;  //!< Result type of the sparse matrix expression.
   typedef OppositeType_<MT>  OT;  //!< Opposite type of the sparse matrix expression.
   typedef ReturnType_<MT>    RN;  //!< Return type of the sparse matrix expression.
   //**********************************************************************************************

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the for-each expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for
       the serial evaluation strategy of the for-each expression. In case the given sparse
       matrix expression of type \a MT requires an intermediate evaluation, \a useAssign will
       be set to 1 and the for-each expression will be evaluated via the \a assign function
       family. Otherwise \a useAssign will be set to 0 and the expression will be evaluated
       via the subscript operator. */
   enum : bool { useAssign = RequiresEvaluation<MT>::value };

   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT2 >
   struct UseAssign {
      enum : bool { value = useAssign };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**Parallel evaluation strategy****************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! The UseSMPAssign struct is a helper struct for the selection of the parallel evaluation
       strategy. In case either the target matrix or the sparse matrix operand is not SMP
       assignable or the matrix operand requires an intermediate evaluation, \a value is set to
       1 and the expression specific evaluation strategy is selected. Otherwise \a value is set
       to 0 and the default strategy is chosen. */
   template< typename MT2 >
   struct UseSMPAssign {
      enum : bool { value = ( !MT2::smpAssignable || !MT::smpAssignable ) && useAssign };
   };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef SMatForEachExpr<MT,OP,SO>   This;           //!< Type of this SMatForEachExpr instance.
   typedef ForEachTrait_<MT,OP>        ResultType;     //!< Result type for expression template evaluations.
   typedef OppositeType_<ResultType>   OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef TransposeType_<ResultType>  TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<ResultType>    ElementType;    //!< Resulting element type.

   //! Return type for expression template evaluations.
   typedef decltype( std::declval<OP>()( std::declval<RN>() ) )  ReturnType;

   //! Data type for composite expression templates.
   typedef IfTrue_< useAssign, const ResultType, const SMatForEachExpr& >  CompositeType;

   //! Composite data type of the sparse matrix expression.
   typedef If_< IsExpression<MT>, const MT, const MT& >  Operand;

   //! Data type of the custom unary operation.
   typedef OP  Operation;
   //**********************************************************************************************

   //**ConstIterator class definition**************************************************************
   /*!\brief Iterator over the elements of the sparse matrix for-each expression.
   */
   class ConstIterator
   {
    public:
      //**Type definitions*************************************************************************
      //! Element type of the sparse matrix expression.
      typedef ValueIndexPair<ElementType>  Element;

      //! Iterator type of the sparse matrix expression.
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
      // \param it Iterator to the initial matrix element.
      // \param op The custom unary operation.
      */
      inline ConstIterator( IteratorType it, OP op )
         : it_( it )  // Iterator over the elements of the sparse matrix expression
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
      /*!\brief Direct access to the sparse matrix element at the current iterator position.
      //
      // \return The current value of the sparse element.
      */
      inline const Element operator*() const {
         return Element( op_( it_->value() ), it_->index() );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the sparse matrix element at the current iterator position.
      //
      // \return Reference to the sparse matrix element at the current iterator position.
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
      IteratorType it_;  //!< Iterator over the elements of the sparse matrix expression.
      OP           op_;  //!< The custom unary operation.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = MT::smpAssignable };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SMatForEachExpr class.
   //
   // \param sm The sparse matrix operand of the for-each expression.
   // \param op The custom unary operation.
   */
   explicit inline SMatForEachExpr( const MT& sm, OP op ) noexcept
      : sm_( sm )  // Sparse matrix of the for-each expression
      , op_( op )  // The custom unary operation
   {}
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 2D-access to the matrix elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator()( size_t i, size_t j ) const {
      BLAZE_INTERNAL_ASSERT( i < sm_.rows()   , "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < sm_.columns(), "Invalid column access index" );
      return op_( sm_(i,j) );
   }
   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the matrix elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid matrix access index.
   */
   inline ReturnType at( size_t i, size_t j ) const {
      if( i >= sm_.rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( j >= sm_.columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
      }
      return (*this)(i,j);
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of row \a i.
   //
   // \param i The row index.
   // \return Iterator to the first non-zero element of row \a i.
   */
   inline ConstIterator begin( size_t i ) const {
      return ConstIterator( sm_.begin(i), op_ );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of row \a i.
   //
   // \param i The row index.
   // \return Iterator just past the last non-zero element of row \a i.
   */
   inline ConstIterator end( size_t i ) const {
      return ConstIterator( sm_.end(i), op_ );
   }
   //**********************************************************************************************

   //**Rows function*******************************************************************************
   /*!\brief Returns the current number of rows of the matrix.
   //
   // \return The number of rows of the matrix.
   */
   inline size_t rows() const noexcept {
      return sm_.rows();
   }
   //**********************************************************************************************

   //**Columns function****************************************************************************
   /*!\brief Returns the current number of columns of the matrix.
   //
   // \return The number of columns of the matrix.
   */
   inline size_t columns() const noexcept {
      return sm_.columns();
   }
   //**********************************************************************************************

   //**NonZeros function***************************************************************************
   /*!\brief Returns the number of non-zero elements in the sparse matrix.
   //
   // \return The number of non-zero elements in the sparse matrix.
   */
   inline size_t nonZeros() const {
      return sm_.nonZeros();
   }
   //**********************************************************************************************

   //**NonZeros function***************************************************************************
   /*!\brief Returns the number of non-zero elements in the specified row.
   //
   // \param i The index of the row.
   // \return The number of non-zero elements of row \a i.
   */
   inline size_t nonZeros( size_t i ) const {
      return sm_.nonZeros(i);
   }
   //**********************************************************************************************

   //**Find function*******************************************************************************
   /*!\brief Searches for a specific matrix element.
   //
   // \param i The row index of the search element.
   // \param j The column index of the search element.
   // \return Iterator to the element in case the index is found, end() iterator otherwise.
   */
   inline ConstIterator find( size_t i, size_t j ) const {
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT );
      return ConstIterator( sm_.find( i, j ), op_ );
   }
   //**********************************************************************************************

   //**LowerBound function*************************************************************************
   /*!\brief Returns an iterator to the first index not less then the given index.
   //
   // \param i The row index of the search element.
   // \param j The column index of the search element.
   // \return Iterator to the first index not less then the given index, end() iterator otherwise.
   */
   inline ConstIterator lowerBound( size_t i, size_t j ) const {
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT );
      return ConstIterator( sm_.lowerBound( i, j ), op_ );
   }
   //**********************************************************************************************

   //**UpperBound function*************************************************************************
   /*!\brief Returns an iterator to the first index greater then the given index.
   //
   // \param i The row index of the search element.
   // \param j The column index of the search element.
   // \return Iterator to the first index greater then the given index, end() iterator otherwise.
   */
   inline ConstIterator upperBound( size_t i, size_t j ) const {
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT );
      return ConstIterator( sm_.upperBound( i, j ), op_ );
   }
   //**********************************************************************************************

   //**Operand access******************************************************************************
   /*!\brief Returns the sparse matrix operand.
   //
   // \return The sparse matrix operand.
   */
   inline Operand operand() const noexcept {
      return sm_;
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
      return sm_.canAlias( alias );
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
      return sm_.isAliased( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return sm_.canSMPAssign();
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   Operand   sm_;  //!< Sparse matrix of the absolute value expression.
   Operation op_;  //!< The custom unary operation.
   //**********************************************************************************************

   //**Assignment to dense matrices****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse matrix for-each expression to a dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side for-each expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse matrix for-each
   // expression to a dense matrix. Due to the explicit application of the SFINAE principle, this
   // function can only be selected by the compiler in case the operand requires an intermediate
   // evaluation.
   */
   template< typename MT2  // Type of the target dense matrix
           , bool SO2 >    // Storage order of the target dense matrix
   friend inline EnableIf_< UseAssign<MT2> >
      assign( DenseMatrix<MT2,SO2>& lhs, const SMatForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( RT, SO );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const RT tmp( serial( rhs.sm_ ) );
      assign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to row-major sparse matrices*****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse matrix for-each expression to a row-major sparse matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side sparse matrix.
   // \param rhs The right-hand side for-each expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse matrix for-each
   // expression to a row-major sparse matrix. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the operand requires
   // an intermediate evaluation and the underlying numeric data type of the operand and the
   // target matrix are identical.
   */
   template< typename MT2 >  // Type of the target sparse matrix
   friend inline EnableIf_< And< UseAssign<MT2>
                               , IsSame< UnderlyingNumeric<MT>, UnderlyingNumeric<MT2> > > >
      assign( SparseMatrix<MT2,false>& lhs, const SMatForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      typedef Iterator_<MT2>  Iterator;

      assign( ~lhs, rhs.sm_ );

      const size_t m( rhs.rows() );

      for( size_t i=0UL; i<m; ++i ) {
         const Iterator end( (~lhs).end(i) );
         for( Iterator element=(~lhs).begin(i); element!=end; ++element ) {
            element->value() = rhs.op_( element->value() );
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to column-major sparse matrices**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse matrix for-each expression to a column-major sparse matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side sparse matrix.
   // \param rhs The right-hand side for-each expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse matrix for-each
   // expression to a column-major sparse matrix. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the operand requires
   // an intermediate evaluation and the underlying numeric data type of the operand and the
   // target matrix are identical.
   */
   template< typename MT2 >  // Type of the target sparse matrix
   friend inline EnableIf_< And< UseAssign<MT2>
                               , IsSame< UnderlyingNumeric<MT>, UnderlyingNumeric<MT2> > > >
      assign( SparseMatrix<MT2,true>& lhs, const SMatForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      typedef Iterator_<MT2>  Iterator;

      assign( ~lhs, rhs.sm_ );

      const size_t n( rhs.columns() );

      for( size_t j=0UL; j<n; ++j ) {
         const Iterator end( (~lhs).end(j) );
         for( Iterator element=(~lhs).begin(j); element!=end; ++element ) {
            element->value() = rhs.op_( element->value() );
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to sparse matrices***************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse matrix for-each expression to a sparse matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side sparse matrix.
   // \param rhs The right-hand side for-each expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse matrix for-each
   // expression to a sparse matrix. Due to the explicit application of the SFINAE principle, this
   // function can only be selected by the compiler in case the operand requires an intermediate
   // evaluation and the underlying numeric data type of the operand and the target vector differ.
   */
   template< typename MT2  // Type of the target sparse matrix
           , bool SO2 >    // Storage order of the target sparse matrix
   friend inline EnableIf_< And< UseAssign<MT2>
                               , Not< IsSame< UnderlyingNumeric<MT>, UnderlyingNumeric<MT2> > > > >
      assign( SparseMatrix<MT2,SO2>& lhs, const SMatForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( RT, SO );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const RT tmp( serial( rhs.sm_ ) );
      (~lhs).reserve( tmp.nonZeros() );
      assign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense matrices*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a sparse matrix for-each expression to a dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side for-each expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a sparse
   // matrix for-each expression to a dense matrix. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the
   // operand requires an intermediate evaluation.
   */
   template< typename MT2  // Type of the target dense matrix
           , bool SO2 >    // Storage order of the target dense matrix
   friend inline EnableIf_< UseAssign<MT2> >
      addAssign( DenseMatrix<MT2,SO2>& lhs, const SMatForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( RT, SO );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const RT tmp( serial( rhs.sm_ ) );
      addAssign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to sparse matrices******************************************************
   // No special implementation for the addition assignment to sparse matrices.
   //**********************************************************************************************

   //**Subtraction assignment to dense matrices****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a sparse matrix for-each expression to a dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side for-each expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a sparse
   // matrix for-each expression to a dense matrix. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the
   // operand requires an intermediate evaluation.
   */
   template< typename MT2  // Type of the target dense matrix
           , bool SO2 >    // Storage order of the target sparse matrix
   friend inline EnableIf_< UseAssign<MT2> >
      subAssign( DenseMatrix<MT2,SO2>& lhs, const SMatForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( RT, SO );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const RT tmp( serial( rhs.sm_ ) );
      subAssign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to sparse matrices***************************************************
   // No special implementation for the subtraction assignment to sparse matrices.
   //**********************************************************************************************

   //**Multiplication assignment to dense matrices*************************************************
   // No special implementation for the multiplication assignment to dense matrices.
   //**********************************************************************************************

   //**Multiplication assignment to sparse matrices************************************************
   // No special implementation for the multiplication assignment to sparse matrices.
   //**********************************************************************************************

   //**SMP assignment to dense matrices************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a sparse matrix for-each expression to a dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side for-each expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a sparse matrix
   // for-each expression to a dense matrix. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename MT2  // Type of the target dense matrix
           , bool SO2 >    // Storage order of the target dense matrix
   friend inline EnableIf_< UseSMPAssign<MT2> >
      smpAssign( DenseMatrix<MT2,SO2>& lhs, const SMatForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( RT, SO );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const RT tmp( rhs.sm_ );
      smpAssign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to sparse matrices***********************************************************
   // No special implementation for the SMP assignment to sparse matrices.
   //**********************************************************************************************

   //**SMP addition assignment to dense matrices***************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a sparse matrix for-each expression to a dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side for-each expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a sparse
   // matrix for-each expression to a dense matrix. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename MT2  // Type of the target dense matrix
           , bool SO2 >    // Storage order of the target dense matrix
   friend inline EnableIf_< UseSMPAssign<MT2> >
      smpAddAssign( DenseMatrix<MT2,SO2>& lhs, const SMatForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( RT, SO );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const RT tmp( rhs.sm_ );
      smpAddAssign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to sparse matrices**************************************************
   // No special implementation for the SMP addition assignment to sparse matrices.
   //**********************************************************************************************

   //**SMP subtraction assignment to dense matrices************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a sparse matrix for-each expression to a dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side for-each expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a sparse
   // matrix for-each expression to a dense matrix. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename MT2  // Type of the target dense matrix
           , bool SO2 >    // Storage order of the target sparse matrix
   friend inline EnableIf_< UseSMPAssign<MT2> >
      smpSubAssign( DenseMatrix<MT2,SO2>& lhs, const SMatForEachExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( RT, SO );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<RT> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const RT tmp( rhs.sm_ );
      smpSubAssign( ~lhs, forEach( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to sparse matrices***********************************************
   // No special implementation for the SMP subtraction assignment to sparse matrices.
   //**********************************************************************************************

   //**SMP multiplication assignment to dense matrices*********************************************
   // No special implementation for the SMP multiplication assignment to dense matrices.
   //**********************************************************************************************

   //**SMP multiplication assignment to sparse matrices********************************************
   // No special implementation for the SMP multiplication assignment to sparse matrices.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( MT, SO );
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
/*!\brief Evaluates the given custom operation on each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \param op The custom operation.
// \return The custom operation applied to each single element of \a sm.
//
// The \a forEach() function evaluates the given custom operation on each non-zero element of the
// input matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a forEach() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = forEach( A, []( double a ){ return std::sqrt( a ); } );
   \endcode
*/
template< typename MT    // Type of the sparse matrix
        , bool SO        // Storage order
        , typename OP >  // Type of the custom operation
inline const SMatForEachExpr<MT,OP,SO> forEach( const SparseMatrix<MT,SO>& sm, OP op )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,OP,SO>( ~sm, op );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Applies the \a abs() function to each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The resulting sparse matrix.
//
// This function applies the abs() function to each non-zero element of the input matrix \a sm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a abs() function:

   \code
   blaze::DynamicMatrix<double> A, B;
   // ... Resizing and initialization
   B = abs( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Abs,SO> abs( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Abs,SO>( ~sm, Abs() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Applies the \a floor() function to each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The resulting sparse matrix.
//
// This function applies the floor() function to each non-zero element of the input matrix \a sm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a floor() function:

   \code
   blaze::DynamicMatrix<double> A, B;
   // ... Resizing and initialization
   B = floor( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Floor,SO> floor( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Floor,SO>( ~sm, Floor() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Applies the \a ceil() function to each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The resulting sparse matrix.
//
// This function applies the ceil() function to each non-zero element of the input matrix \a sm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a ceil() function:

   \code
   blaze::DynamicMatrix<double> A, B;
   // ... Resizing and initialization
   B = ceil( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Ceil,SO> ceil( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Ceil,SO>( ~sm, Ceil() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a matrix containing the complex conjugate of each single element of \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The complex conjugate of each single element of \a sm.
//
// The \a conj function calculates the complex conjugate of each element of the input matrix
// \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a conj function:

   \code
   blaze::CompressedMatrix< complex<double> > A, B;
   // ... Resizing and initialization
   B = conj( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Conj,SO> conj( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Conj,SO>( ~sm, Conj() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the conjugate transpose matrix of \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The conjugate transpose of \a sm.
//
// The \a ctrans function returns an expression representing the conjugate transpose (also called
// adjoint matrix, Hermitian conjugate matrix or transjugate matrix) of the given input matrix
// \a sm.\n
// The following example demonstrates the use of the \a ctrans function:

   \code
   blaze::CompressedMatrix< complex<double> > A, B;
   // ... Resizing and initialization
   B = ctrans( A );
   \endcode

// Note that the \a ctrans function has the same effect as manually applying the \a conj and
// \a trans function in any order:

   \code
   B = trans( conj( A ) );  // Computing the conjugate transpose matrix
   B = conj( trans( A ) );  // Computing the conjugate transpose matrix
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const CTransExprTrait_<MT> ctrans( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return trans( conj( ~sm ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a matrix containing the real parts of each single element of \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The real part of each single element of \a sm.
//
// The \a real function calculates the real part of each element of the input matrix \a sm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a real function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = real( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Real,SO> real( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Real,SO>( ~sm, Real() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a matrix containing the imaginary parts of each single element of \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The imaginary part of each single element of \a sm.
//
// The \a imag function calculates the imaginary part of each element of the input matrix \a sm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a imag function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = imag( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Imag,SO> imag( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Imag,SO>( ~sm, Imag() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the square root of each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix; all non-zero elements must be in the range \f$[0..\infty)\f$.
// \return The square root of each single element of \a sm.
//
// The \a sqrt() function computes the square root of each non-zero element of the input matrix
// \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a sqrt() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = sqrt( A );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[0..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Sqrt,SO> sqrt( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Sqrt,SO>( ~sm, Sqrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse square root of each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix; all non-zero elements must be in the range \f$(0..\infty)\f$.
// \return The inverse square root of each single element of \a sm.
//
// The \a invsqrt() function computes the inverse square root of each non-zero element of the
// input matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a invsqrt() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = invsqrt( A );
   \endcode

// \note All non-zero elements are expected to be in the range \f$(0..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,InvSqrt,SO> invsqrt( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,InvSqrt,SO>( ~sm, InvSqrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the cubic root of each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix; all non-zero elements must be in the range \f$[0..\infty)\f$.
// \return The cubic root of each single element of \a sm.
//
// The \a cbrt() function computes the cubic root of each non-zero element of the input matrix
// \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a cbrt() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = cbrt( A );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[0..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Cbrt,SO> cbrt( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Cbrt,SO>( ~sm, Cbrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse cubic root of each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix; all non-zero elements must be in the range \f$(0..\infty)\f$.
// \return The inverse cubic root of each single element of \a sm.
//
// The \a invcbrt() function computes the inverse cubic root of each non-zero element of the
// input matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a invcbrt() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = invcbrt( A );
   \endcode

// \note All non-zero elements are expected to be in the range \f$(0..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,InvCbrt,SO> invcbrt( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,InvCbrt,SO>( ~sm, InvCbrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the exponential value for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \param exp The exponent.
// \return The exponential value of each non-zero element of \a sm.
//
// The \a pow() function computes the exponential value for each non-zero element of the input
// matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a pow() function:

   \code
   blaze::DynamicMatrix<double> A, B;
   // ... Resizing and initialization
   B = pow( A, 4.2 );
   \endcode
*/
template< typename MT    // Type of the sparse matrix
        , bool SO        // Storage order
        , typename ET >  // Type of the exponent
inline const SMatForEachExpr<MT,Pow<ET>,SO> pow( const SparseMatrix<MT,SO>& sm, ET exp )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( ET );

   return SMatForEachExpr<MT,Pow<ET>,SO>( ~sm, Pow<ET>( exp ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes \f$ e^x \f$ for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The resulting sparse matrix.
//
// The \a exp() function computes \f$ e^x \f$ for each non-zero element of the input matrix \a sm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a exp() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = exp( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Exp,SO> exp( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Exp,SO>( ~sm, Exp() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the natural logarithm for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix; all non-zero elements must be in the range \f$[0..\infty)\f$.
// \return The natural logarithm of each non-zero element of \a sm.
//
// The \a log() function computes the natural logarithm for each non-zero element of the input
// matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a log() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = log( A );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[0..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Log,SO> log( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Log,SO>( ~sm, Log() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the common logarithm for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix; all non-zero elements must be in the range \f$[0..\infty)\f$.
// \return The common logarithm of each non-zero element of \a sm.
//
// The \a log10() function computes the common logarithm for each non-zero element of the input
// matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a log10() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = log10( A );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[0..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Log10,SO> log10( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Log10,SO>( ~sm, Log10() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the sine for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The sine of each non-zero element of \a sm.
//
// The \a sin() function computes the sine for each non-zero element of the input matrix \a sm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a sin() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = sin( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Sin,SO> sin( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Sin,SO>( ~sm, Sin() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse sine for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix; all non-zero elements must be in the range \f$[-1..1]\f$.
// \return The inverse sine of each non-zero element of \a sm.
//
// The \a asin() function computes the inverse sine for each non-zero element of the input matrix
// \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a asin() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = asin( A );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[-1..1]\f$. No runtime checks
// are performed to assert this precondition!
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Asin,SO> asin( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Asin,SO>( ~sm, Asin() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the hyperbolic sine for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The hyperbolic sine of each non-zero element of \a sm.
//
// The \a sinh() function computes the hyperbolic sine for each non-zero element of the input
// matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a sinh() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = sinh( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Sinh,SO> sinh( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Sinh,SO>( ~sm, Sinh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse hyperbolic sine for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The inverse hyperbolic sine of each non-zero element of \a sm.
//
// The \a asinh() function computes the inverse hyperbolic sine for each non-zero element of
// the input matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a asinh() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = asinh( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Asinh,SO> asinh( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Asinh,SO>( ~sm, Asinh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the cosine for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The cosine of each non-zero element of \a sm.
//
// The \a cos() function computes the cosine for each non-zero element of the input matrix
// \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a cos() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = cos( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Cos,SO> cos( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Cos,SO>( ~sm, Cos() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse cosine for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix; all non-zero elements must be in the range \f$[-1..1]\f$.
// \return The inverse cosine of each non-zero element of \a sm.
//
// The \a acos() function computes the inverse cosine for each non-zero element of the input
// matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a acos() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = acos( A );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[-1..1]\f$. No runtime checks
// are performed to assert this precondition!
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Acos,SO> acos( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Acos,SO>( ~sm, Acos() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the hyperbolic cosine for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The hyperbolic cosine of each non-zero element of \a sm.
//
// The \a cosh() function computes the hyperbolic cosine for each non-zero element of the input
// matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a cosh() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = cosh( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Cosh,SO> cosh( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Cosh,SO>( ~sm, Cosh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse hyperbolic cosine for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix; all non-zero elements must be in the range \f$[1..\infty)\f$.
// \return The inverse hyperbolic cosine of each non-zero element of \a sm.
//
// The \a acosh() function computes the inverse hyperbolic cosine for each non-zero element of
// the input matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a acosh() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = acosh( A );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[1..\infty)\f$. No runtime
// checks are performed to assert this precondition!
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Acosh,SO> acosh( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Acosh,SO>( ~sm, Acosh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the tangent for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The tangent of each non-zero element of \a sm.
//
// The \a tan() function computes the tangent for each non-zero element of the input matrix \a sm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a tan() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = tan( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Tan,SO> tan( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Tan,SO>( ~sm, Tan() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse tangent for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The inverse tangent of each non-zero element of \a sm.
//
// The \a atan() function computes the inverse tangent for each non-zero element of the input
// matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a atan() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = atan( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Atan,SO> atan( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Atan,SO>( ~sm, Atan() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the hyperbolic tangent for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix; all non-zero elements must be in the range \f$[-1..1]\f$.
// \return The hyperbolic tangent of each non-zero element of \a sm.
//
// The \a tanh() function computes the hyperbolic tangent for each non-zero element of the input
// matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a tanh() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = tanh( A );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[-1..1]\f$. No runtime checks
// are performed to assert this precondition!
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Tanh,SO> tanh( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Tanh,SO>( ~sm, Tanh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse hyperbolic tangent for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix; all non-zero elements must be in the range \f$[-1..1]\f$.
// \return The inverse hyperbolic tangent of each non-zero element of \a sm.
//
// The \a atanh() function computes the inverse hyperbolic tangent for each non-zero element of
// the input matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a atanh() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = atanh( A );
   \endcode

// \note All non-zero elements are expected to be in the range \f$[-1..1]\f$. No runtime checks
// are performed to assert this precondition!
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Atanh,SO> atanh( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Atanh,SO>( ~sm, Atanh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the error function for each non-zero element of the sparse matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The error function of each non-zero element of \a sm.
//
// The \a erf() function computes the error function for each non-zero element of the input
// matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a erf() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = erf( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Erf,SO> erf( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Erf,SO>( ~sm, Erf() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the complementary error function for each non-zero element of the sparse
//        matrix \a sm.
// \ingroup sparse_matrix
//
// \param sm The input matrix.
// \return The complementary error function of each non-zero element of \a sm.
//
// The \a erfc() function computes the complementary error function for each non-zero element of
// the input matrix \a sm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a erfc() function:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = erfc( A );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Erfc,SO> erfc( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatForEachExpr<MT,Erfc,SO>( ~sm, Erfc() );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Absolute value function for absolute value sparse matrix expressions.
// \ingroup sparse_matrix
//
// \param sm The absolute value sparse matrix expression.
// \return The absolute value of each single element of \a sm.
//
// This function implements a performance optimized treatment of the absolute value operation
// on a sparse matrix absolute value expression.
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Abs,SO>& abs( const SMatForEachExpr<MT,Abs,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return sm;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Applies the \a floor() function to a sparse matrix \a floor() expressions.
// \ingroup sparse_matrix
//
// \param sm The sparse matrix \a floor() expression.
// \return The resulting sparse matrix.
//
// This function implements a performance optimized treatment of the \a floor() operation on
// a sparse matrix \a floor() expression.
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Floor,SO>& floor( const SMatForEachExpr<MT,Floor,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return sm;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Applies the \a ceil() function to a sparse matrix \a ceil() expressions.
// \ingroup sparse_matrix
//
// \param sm The sparse matrix \a ceil() expression.
// \return The resulting sparse matrix.
//
// This function implements a performance optimized treatment of the \a ceil() operation on
// a sparse matrix \a ceil() expression.
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Ceil,SO>& ceil( const SMatForEachExpr<MT,Ceil,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return sm;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Complex conjugate function for complex conjugate sparse matrix expressions.
// \ingroup sparse_matrix
//
// \param sm The complex conjugate sparse matrix expression.
// \return The original sparse matrix.
//
// This function implements a performance optimized treatment of the complex conjugate operation
// on a sparse matrix complex conjugate expression. It returns an expression representing the
// original sparse matrix:

   \code
   blaze::CompressedMatrix< complex<double> > A, B;
   // ... Resizing and initialization
   B = conj( conj( A ) );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool TF >    // Transpose flag
inline typename SMatForEachExpr<MT,Conj,TF>::Operand conj( const SMatForEachExpr<MT,Conj,TF>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return sm.operand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Complex conjugate function for conjugate transpose sparse matrix expressions.
// \ingroup sparse_matrix
//
// \param dm The conjugate transpose sparse matrix expression.
// \return The transpose sparse matrix.
//
// This function implements a performance optimized treatment of the complex conjugate operation
// on a sparse matrix conjugate transpose expression. It returns an expression representing the
// transpose of the sparse matrix:

   \code
   blaze::CompressedMatrix< complex<double> > A, B;
   // ... Resizing and initialization
   B = conj( ctrans( A ) );
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatTransExpr<MT,!SO> conj( const SMatTransExpr<SMatForEachExpr<MT,Conj,SO>,!SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatTransExpr<MT,!SO>( sm.operand().operand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Real part function for real part sparse matrix expressions.
// \ingroup sparse_matrix
//
// \param sm The real part sparse matrix expression.
// \return The real part of each single element of \a sm.
//
// This function implements a performance optimized treatment of the real part operation on
// a sparse matrix real part expression.
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Real,SO>& real( const SMatForEachExpr<MT,Real,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return sm;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Imaginary part function for imaginary part sparse matrix expressions.
// \ingroup sparse_matrix
//
// \param sm The imaginary part sparse matrix expression.
// \return The imaginary part of each single element of \a sm.
//
// This function implements a performance optimized treatment of the imaginary part operation
// on a sparse matrix imaginary part expression.
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatForEachExpr<MT,Imag,SO>& imag( const SMatForEachExpr<MT,Imag,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return sm;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ROWS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename OP, bool SO >
struct Rows< SMatForEachExpr<MT,OP,SO> > : public Rows<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COLUMNS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename OP, bool SO >
struct Columns< SMatForEachExpr<MT,OP,SO> > : public Columns<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSYMMETRIC SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Abs,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Floor,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Ceil,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Conj,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Real,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Imag,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Sqrt,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,InvSqrt,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Cbrt,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,InvCbrt,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, typename ET, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Pow<ET>,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Exp,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Log,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Log10,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Sin,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Asin,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Sinh,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Asinh,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Cos,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Acos,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Cosh,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Acosh,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Tan,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Atan,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Tanh,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Atanh,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Erf,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};

template< typename MT, bool SO >
struct IsSymmetric< SMatForEachExpr<MT,Erfc,SO> >
   : public BoolConstant< IsSymmetric<MT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISHERMITIAN SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Abs,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Floor,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Ceil,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Conj,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Real,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Imag,SO> >
   : public BoolConstant< IsBuiltin< ElementType_<MT> >::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Sqrt,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,InvSqrt,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Cbrt,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,InvCbrt,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, typename ET, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Pow<ET>,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Exp,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Log,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Log10,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Sin,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Asin,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Sinh,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Asinh,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Cos,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Acos,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Cosh,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Acosh,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Tan,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Atan,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Tanh,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Atanh,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Erf,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};

template< typename MT, bool SO >
struct IsHermitian< SMatForEachExpr<MT,Erfc,SO> >
   : public BoolConstant< IsHermitian<MT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISLOWER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename OP, bool SO >
struct IsLower< SMatForEachExpr<MT,OP,SO> >
   : public BoolConstant< IsLower<MT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISUNILOWER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ET, bool SO >
struct IsUniLower< SMatForEachExpr<MT,Pow<ET>,SO> >
   : public BoolConstant< IsUniLower<MT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSTRICTLYLOWER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename OP, bool SO >
struct IsStrictlyLower< SMatForEachExpr<MT,OP,SO> >
   : public BoolConstant< IsStrictlyLower<MT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISUPPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename OP, bool SO >
struct IsUpper< SMatForEachExpr<MT,OP,SO> >
   : public BoolConstant< IsUpper<MT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISUNIUPPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ET, bool SO >
struct IsUniUpper< SMatForEachExpr<MT,Pow<ET>,SO> >
   : public BoolConstant< IsUniUpper<MT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSTRICTLYUPPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename OP, bool SO >
struct IsStrictlyUpper< SMatForEachExpr<MT,OP,SO> >
   : public BoolConstant< IsStrictlyUpper<MT>::value >
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
template< typename MT >
struct SMatForEachExprTrait< SMatForEachExpr<MT,Abs,false>, Abs >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsRowMajorMatrix<MT> >
                   , SMatForEachExpr<MT,Abs,false>
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT >
struct TSMatForEachExprTrait< SMatForEachExpr<MT,Abs,true>, Abs >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsColumnMajorMatrix<MT> >
                   , SMatForEachExpr<MT,Abs,true>
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT >
struct SMatForEachExprTrait< SMatForEachExpr<MT,Conj,false>, Conj >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsRowMajorMatrix<MT> >
                   , Operand_< SMatForEachExpr<MT,Conj,false> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT >
struct TSMatForEachExprTrait< SMatForEachExpr<MT,Conj,true>, Conj >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsColumnMajorMatrix<MT> >
                   , Operand_< SMatForEachExpr<MT,Conj,true> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT >
struct SMatForEachExprTrait< SMatTransExpr< SMatForEachExpr<MT,Conj,true>, false >, Conj >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsColumnMajorMatrix<MT> >
                   , SMatTransExpr<MT,false>
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT >
struct TSMatForEachExprTrait< SMatTransExpr< SMatForEachExpr<MT,Conj,false>, true >, Conj >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsRowMajorMatrix<MT> >
                   , SMatTransExpr<MT,true>
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT >
struct SMatForEachExprTrait< SMatForEachExpr<MT,Real,false>, Real >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsRowMajorMatrix<MT> >
                   , SMatForEachExpr<MT,Real,false>
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT >
struct TSMatForEachExprTrait< SMatForEachExpr<MT,Real,true>, Real >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsColumnMajorMatrix<MT> >
                   , SMatForEachExpr<MT,Real,true>
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename OP, bool SO, bool AF >
struct SubmatrixExprTrait< SMatForEachExpr<MT,OP,SO>, AF >
{
 public:
   //**********************************************************************************************
   using Type = ForEachExprTrait_< SubmatrixExprTrait_<const MT,AF>, OP >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename OP, bool SO >
struct RowExprTrait< SMatForEachExpr<MT,OP,SO> >
{
 public:
   //**********************************************************************************************
   using Type = ForEachExprTrait_< RowExprTrait_<const MT>, OP >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename OP, bool SO >
struct ColumnExprTrait< SMatForEachExpr<MT,OP,SO> >
{
 public:
   //**********************************************************************************************
   using Type = ForEachExprTrait_< ColumnExprTrait_<const MT>, OP >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
