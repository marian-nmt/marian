//=================================================================================================
/*!
//  \file blaze/math/expressions/SMatScalarMultExpr.h
//  \brief Header file for the sparse matrix/scalar multiplication expression
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

#ifndef _BLAZE_MATH_EXPRESSIONS_SMATSCALARMULTEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_SMATSCALARMULTEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/constraints/StorageOrder.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/MatScalarMultExpr.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/sparse/ValueIndexPair.h>
#include <blaze/math/traits/ColumnExprTrait.h>
#include <blaze/math/traits/DivExprTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/MultExprTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/RowExprTrait.h>
#include <blaze/math/traits/SubmatrixExprTrait.h>
#include <blaze/math/typetraits/Columns.h>
#include <blaze/math/typetraits/IsColumnMajorMatrix.h>
#include <blaze/math/typetraits/IsColumnVector.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsHermitian.h>
#include <blaze/math/typetraits/IsInvertible.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/IsRowVector.h>
#include <blaze/math/typetraits/IsSparseMatrix.h>
#include <blaze/math/typetraits/IsSparseVector.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsTemporary.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/math/typetraits/Rows.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>
#include <blaze/util/Assert.h>
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
#include <blaze/util/typetraits/RemoveReference.h>


namespace blaze {

//=================================================================================================
//
//  CLASS SMATSCALARMULTEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for sparse matrix-scalar multiplications.
// \ingroup sparse_matrix_expression
//
// The SMatScalarMult class represents the compile time expression for multiplications between
// a sparse matrix and a scalar value.
*/
template< typename MT  // Type of the left-hand side sparse matrix
        , typename ST  // Type of the right-hand side scalar value
        , bool SO >    // Storage order
class SMatScalarMultExpr : public SparseMatrix< SMatScalarMultExpr<MT,ST,SO>, SO >
                         , private MatScalarMultExpr
                         , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef ResultType_<MT>     RT;  //!< Result type of the sparse matrix expression.
   typedef ReturnType_<MT>     RN;  //!< Return type of the sparse matrix expression.
   typedef CompositeType_<MT>  CT;  //!< Composite type of the sparse matrix expression.
   //**********************************************************************************************

   //**Return type evaluation**********************************************************************
   //! Compilation switch for the selection of the subscript operator return type.
   /*! The \a returnExpr compile time constant expression is a compilation switch for the
       selection of the \a ReturnType. If the matrix operand returns a temporary vector
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
       the serial evaluation strategy of the multiplication expression. In case the given sparse
       matrix expression of type \a MT requires an intermediate evaluation, \a useAssign will be
       set to 1 and the multiplication expression will be evaluated via the \a assign function
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
       assignable and the matrix operand requires an intermediate evaluation, \a value is set
       to 1 and the expression specific evaluation strategy is selected. Otherwise \a value is
       set to 0 and the default strategy is chosen. */
   template< typename MT2 >
   struct UseSMPAssign {
      enum : bool { value = ( !MT2::smpAssignable || !MT::smpAssignable ) && useAssign };
   };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef SMatScalarMultExpr<MT,ST,SO>  This;           //!< Type of this SMatScalarMultExpr instance.
   typedef MultTrait_<RT,ST>             ResultType;     //!< Result type for expression template evaluations.
   typedef OppositeType_<ResultType>     OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef TransposeType_<ResultType>    TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<ResultType>      ElementType;    //!< Resulting element type.

   //! Return type for expression template evaluations.
   typedef const IfTrue_< returnExpr, ExprReturnType, ElementType >  ReturnType;

   //! Data type for composite expression templates.
   typedef IfTrue_< useAssign, const ResultType, const SMatScalarMultExpr& >  CompositeType;

   //! Composite data type of the sparse matrix expression.
   typedef If_< IsExpression<MT>, const MT, const MT& >  LeftOperand;

   //! Composite type of the right-hand side scalar value.
   typedef ST  RightOperand;
   //**********************************************************************************************

   //**ConstIterator class definition**************************************************************
   /*!\brief Iterator over the elements of the sparse matrix/scalar multiplication expression.
   */
   class ConstIterator
   {
    public:
      //**Type definitions*************************************************************************
      //! Element type of the sparse matrix expression.
      typedef ValueIndexPair<ElementType>  Element;

      //! Iterator type of the sparse matrix expression.
      typedef ConstIterator_< RemoveReference_<LeftOperand> >  IteratorType;

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
      */
      inline ConstIterator( IteratorType matrix, RightOperand scalar )
         : matrix_( matrix )  // Iterator over the elements of the left-hand side sparse matrix expression
         , scalar_( scalar )  // Right-hand side scalar of the multiplication expression
      {}
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented expression iterator.
      */
      inline ConstIterator& operator++() {
         ++matrix_;
         return *this;
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the sparse matrix element at the current iterator position.
      //
      // \return The element at the current iterator position.
      */
      inline const Element operator*() const {
         return Element( matrix_->value() * scalar_, matrix_->index() );
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
         return matrix_->value() * scalar_;
      }
      //*******************************************************************************************

      //**Index function***************************************************************************
      /*!\brief Access to the current index of the sparse element.
      //
      // \return The current index of the sparse element.
      */
      inline size_t index() const {
         return matrix_->index();
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side expression iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const ConstIterator& rhs ) const {
         return matrix_ == rhs.matrix_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side expression iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const ConstIterator& rhs ) const {
         return matrix_ != rhs.matrix_;
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two expression iterators.
      //
      // \param rhs The right-hand side expression iterator.
      // \return The number of elements between the two expression iterators.
      */
      inline DifferenceType operator-( const ConstIterator& rhs ) const {
         return matrix_ - rhs.matrix_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType matrix_;  //!< Iterator over the elements of the left-hand side sparse matrix expression.
      RightOperand scalar_;  //!< Right-hand side scalar of the multiplication expression.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = false };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SMatScalarMultExpr class.
   //
   // \param matrix The left-hand side sparse matrix of the multiplication expression.
   // \param scalar The right-hand side scalar of the multiplication expression.
   */
   explicit inline SMatScalarMultExpr( const MT& matrix, ST scalar ) noexcept
      : matrix_( matrix )  // Left-hand side sparse matrix of the multiplication expression
      , scalar_( scalar )  // Right-hand side scalar of the multiplication expression
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
      BLAZE_INTERNAL_ASSERT( i < matrix_.rows()   , "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < matrix_.columns(), "Invalid column access index" );
      return matrix_(i,j) * scalar_;
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
      if( i >= matrix_.rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( j >= matrix_.columns() ) {
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
      return ConstIterator( matrix_.begin(i), scalar_ );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of row \a i.
   //
   // \param i The row index.
   // \return Iterator just past the last non-zero element of row \a i.
   */
   inline ConstIterator end( size_t i ) const {
      return ConstIterator( matrix_.end(i), scalar_ );
   }
   //**********************************************************************************************

   //**Rows function*******************************************************************************
   /*!\brief Returns the current number of rows of the matrix.
   //
   // \return The number of rows of the matrix.
   */
   inline size_t rows() const noexcept {
      return matrix_.rows();
   }
   //**********************************************************************************************

   //**Columns function****************************************************************************
   /*!\brief Returns the current number of columns of the matrix.
   //
   // \return The number of columns of the matrix.
   */
   inline size_t columns() const noexcept {
      return matrix_.columns();
   }
   //**********************************************************************************************

   //**NonZeros function***************************************************************************
   /*!\brief Returns the number of non-zero elements in the sparse matrix.
   //
   // \return The number of non-zero elements in the sparse matrix.
   */
   inline size_t nonZeros() const {
      return matrix_.nonZeros();
   }
   //**********************************************************************************************

   //**NonZeros function***************************************************************************
   /*!\brief Returns the number of non-zero elements in the specified row.
   //
   // \param i The index of the row.
   // \return The number of non-zero elements of row \a i.
   */
   inline size_t nonZeros( size_t i ) const {
      return matrix_.nonZeros(i);
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
      return ConstIterator( matrix_.find( i, j ), scalar_ );
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
      return ConstIterator( matrix_.lowerBound( i, j ), scalar_ );
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
      return ConstIterator( matrix_.upperBound( i, j ), scalar_ );
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side sparse matrix operand.
   //
   // \return The left-hand side sparse matrix operand.
   */
   inline LeftOperand leftOperand() const noexcept {
      return matrix_;
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
      return matrix_.canAlias( alias );
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
      return matrix_.isAliased( alias );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  matrix_;  //!< Left-hand side sparse matrix of the multiplication expression.
   RightOperand scalar_;  //!< Right-hand side scalar of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense matrices****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse matrix-scalar multiplication to a dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse matrix-scalar
   // multiplication expression to a dense matrix. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the operand requires
   // an intermediate evaluation.
   */
   template< typename MT2  // Type of the target dense matrix
           , bool SO2 >    // Storage order of the target dense matrix
   friend inline EnableIf_< UseAssign<MT2> >
      assign( DenseMatrix<MT2,SO2>& lhs, const SMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      assign( ~lhs, rhs.matrix_ );
      (~lhs) *= rhs.scalar_;
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to sparse matrices***************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse matrix-scalar multiplication to a sparse matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side sparse matrix.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse matrix-scalar
   // multiplication expression to a sparse matrix. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the operand requires
   // an intermediate evaluation.
   */
   template< typename MT2  // Type of the target sparse matrix
           , bool SO2 >    // Storage order of the target sparse matrix
   friend inline EnableIf_< UseAssign<MT2> >
      assign( SparseMatrix<MT2,SO2>& lhs, const SMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      assign( ~lhs, rhs.matrix_ );
      (~lhs) *= rhs.scalar_;
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense matrices*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a sparse matrix-scalar multiplication to a dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a sparse matrix-
   // scalar multiplication expression to a dense matrix. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the operand
   // requires an intermediate evaluation.
   */
   template< typename MT2  // Type of the target dense matrix
           , bool SO2 >    // Storage order of the target dense matrix
   friend inline EnableIf_< UseAssign<MT2> >
      addAssign( DenseMatrix<MT2,SO2>& lhs, const SMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const ResultType tmp( serial( rhs ) );
      addAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to sparse matrices******************************************************
   // No special implementation for the addition assignment to sparse matrices.
   //**********************************************************************************************

   //**Subtraction assignment to dense matrices****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a sparse matrix-scalar multiplication to a dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a sparse matrix-
   // scalar multiplication expression to a dense matrix. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the operand
   // requires an intermediate evaluation.
   */
   template< typename MT2  // Type of the target dense matrix
           , bool SO2 >    // Storage order of the target dense matrix
   friend inline EnableIf_< UseAssign<MT2> >
      subAssign( DenseMatrix<MT2,SO2>& lhs, const SMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const ResultType tmp( serial( rhs ) );
      subAssign( ~lhs, tmp );
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
   // No special implementation for the SMP assignment to dense matrices.
   //**********************************************************************************************

   //**SMP assignment to sparse matrices***********************************************************
   // No special implementation for the SMP assignment to sparse matrices.
   //**********************************************************************************************

   //**SMP addition assignment to dense matrices***************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a sparse matrix-scalar multiplication to a dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a sparse
   // matrix-scalar multiplication expression to a dense matrix. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename MT2  // Type of the target dense matrix
           , bool SO2 >    // Storage order of the target dense matrix
   friend inline EnableIf_< UseSMPAssign<MT2> >
      smpAddAssign( DenseMatrix<MT2,SO2>& lhs, const SMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const ResultType tmp( rhs );
      smpAddAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to sparse matrices**************************************************
   // No special implementation for the SMP addition assignment to sparse matrices.
   //**********************************************************************************************

   //**SMP subtraction assignment to dense matrices************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a sparse matrix-scalar multiplication to a dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a sparse
   // matrix-scalar multiplication expression to a dense matrix. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename MT2  // Type of the target dense matrix
           , bool SO2 >    // Storage order of the target dense matrix
   friend inline EnableIf_< UseSMPAssign<MT2> >
      smpSubAssign( DenseMatrix<MT2,SO2>& lhs, const SMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const ResultType tmp( rhs );
      smpSubAssign( ~lhs, tmp );
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
/*!\brief Unary minus operator for the negation of a sparse matrix (\f$ A = -B \f$).
// \ingroup sparse_matrix
//
// \param sm The sparse matrix to be negated.
// \return The negation of the matrix.
//
// This operator represents the negation of a sparse matrix:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = -A;
   \endcode

// The operator returns an expression representing the negation of the given sparse matrix.
*/
template< typename MT  // Data type of the sparse matrix
        , bool SO >    // Storage order
inline const SMatScalarMultExpr<MT,UnderlyingBuiltin_<MT>,SO>
   operator-( const SparseMatrix<MT,SO>& sm )
{
   BLAZE_FUNCTION_TRACE;

   typedef UnderlyingBuiltin_<MT>  ElementType;
   return SMatScalarMultExpr<MT,ElementType,SO>( ~sm, ElementType(-1) );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Multiplication operator for the multiplication of a sparse matrix and a scalar value
//        (\f$ A=B*s \f$).
// \ingroup sparse_matrix
//
// \param mat The left-hand side sparse matrix for the multiplication.
// \param scalar The right-hand side scalar value for the multiplication.
// \return The scaled result matrix.
//
// This operator represents the multiplication between a sparse matrix and a scalar value:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = A * 1.25;
   \endcode

// The operator returns an expression representing a sparse matrix of the higher-order element
// type of the involved data types \a T1::ElementType and \a T2. Note that this operator only
// works for scalar values of built-in data type.
*/
template< typename T1    // Type of the left-hand side sparse matrix
        , bool SO        // Storage order of the left-hand side sparse matrix
        , typename T2 >  // Type of the right-hand side scalar
inline const EnableIf_< IsNumeric<T2>, MultExprTrait_<T1,T2> >
   operator*( const SparseMatrix<T1,SO>& mat, T2 scalar )
{
   BLAZE_FUNCTION_TRACE;

   return MultExprTrait_<T1,T2>( ~mat, scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication operator for the multiplication of a scalar value and a sparse matrix
//        (\f$ A=s*B \f$).
// \ingroup sparse_matrix
//
// \param scalar The left-hand side scalar value for the multiplication.
// \param mat The right-hand side sparse matrix for the multiplication.
// \return The scaled result matrix.
//
// This operator represents the multiplication between a scalar value and a sparse matrix:

   \code
   blaze::CompressedMatrix<double> A, B;
   // ... Resizing and initialization
   B = 1.25 * A;
   \endcode

// The operator returns an expression representing a sparse matrix of the higher-order element
// type of the involved data types \a T1::ElementType and \a T2. Note that this operator only
// works for scalar values of built-in data type.
*/
template< typename T1  // Type of the left-hand side scalar
        , typename T2  // Type of the right-hand side sparse matrix
        , bool SO >    // Storage order of the right-hand side sparse matrix
inline const EnableIf_< IsNumeric<T1>, MultExprTrait_<T1,T2> >
   operator*( T1 scalar, const SparseMatrix<T2,SO>& mat )
{
   BLAZE_FUNCTION_TRACE;

   return MultExprTrait_<T1,T2>( ~mat, scalar );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING UNARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unary minus operator for the negation of a sparse matrix-scalar multiplication
//        (\f$ A = -(B*s) \f$).
// \ingroup sparse_matrix
//
// \param sm The sparse matrix-scalar multiplication to be negated.
// \return The negation of the sparse matrix-scalar multiplication.
//
// This operator implements a performance optimized treatment of the negation of a sparse matrix-
// scalar multiplication expression.
*/
template< typename VT  // Type of the sparse matrix
        , typename ST  // Type of the scalar
        , bool TF >    // Transpose flag
inline const SMatScalarMultExpr<VT,ST,TF>
   operator-( const SMatScalarMultExpr<VT,ST,TF>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return SMatScalarMultExpr<VT,ST,TF>( sm.leftOperand(), -sm.rightOperand() );
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
/*!\brief Multiplication operator for the multiplication of a sparse matrix-scalar multiplication
//        expression and a scalar value (\f$ A=(B*s1)*s2 \f$).
// \ingroup sparse_matrix
//
// \param mat The left-hand side sparse matrix-scalar multiplication.
// \param scalar The right-hand side scalar value for the multiplication.
// \return The scaled result matrix.
//
// This operator implements a performance optimized treatment of the multiplication of a
// sparse matrix-scalar multiplication expression and a scalar value.
*/
template< typename MT     // Type of the sparse matrix
        , typename ST1    // Type of the first scalar
        , bool SO         // Storage order of the sparse matrix
        , typename ST2 >  // Type of the second scalar
inline const EnableIf_< IsNumeric<ST2>
                      , MultExprTrait_< SMatScalarMultExpr<MT,ST1,SO>, ST2 > >
   operator*( const SMatScalarMultExpr<MT,ST1,SO>& mat, ST2 scalar )
{
   BLAZE_FUNCTION_TRACE;

   return mat.leftOperand() * ( mat.rightOperand() * scalar );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a sparse matrix-scalar multiplication
//        expression and a scalar value (\f$ A=s2*(B*s1) \f$).
// \ingroup sparse_matrix
//
// \param scalar The left-hand side scalar value for the multiplication.
// \param mat The right-hand side sparse matrix-scalar multiplication.
// \return The scaled result matrix.
//
// This operator implements a performance optimized treatment of the multiplication of a
// scalar value and a sparse matrix-scalar multiplication expression.
*/
template< typename ST1  // Type of the first scalar
        , typename MT   // Type of the sparse matrix
        , typename ST2  // Type of the second scalar
        , bool SO >     // Storage order of the sparse matrix
inline const EnableIf_< IsNumeric<ST1>
                      , MultExprTrait_< ST1, SMatScalarMultExpr<MT,ST2,SO> > >
   operator*( ST1 scalar, const SMatScalarMultExpr<MT,ST2,SO>& mat )
{
   BLAZE_FUNCTION_TRACE;

   return mat.leftOperand() * ( scalar * mat.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division operator for the division of a sparse matrix-scalar multiplication
//        expression by a scalar value (\f$ A=(B*s1)/s2 \f$).
// \ingroup sparse_matrix
//
// \param mat The left-hand side sparse matrix-scalar multiplication.
// \param scalar The right-hand side scalar value for the division.
// \return The scaled result matrix.
//
// This operator implements a performance optimized treatment of the division of a
// sparse matrix-scalar multiplication expression by a scalar value.
*/
template< typename MT     // Type of the sparse matrix
        , typename ST1    // Type of the first scalar
        , bool SO         // Storage order of the sparse matrix
        , typename ST2 >  // Type of the second scalar
inline const EnableIf_< And< IsNumeric<ST2>, Or< IsInvertible<ST1>, IsInvertible<ST2> > >
                      , DivExprTrait_< SMatScalarMultExpr<MT,ST1,SO>, ST2 > >
   operator/( const SMatScalarMultExpr<MT,ST1,SO>& mat, ST2 scalar )
{
   BLAZE_FUNCTION_TRACE;

   return mat.leftOperand() * ( mat.rightOperand() / scalar );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a sparse matrix-scalar
//        multiplication expression and a dense vector (\f$ \vec{a}=(B*s1)*\vec{c} \f$).
// \ingroup dense_vector
//
// \param mat The left-hand side sparse matrix-scalar multiplication.
// \param vec The right-hand side dense vector.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// sparse matrix-scalar multiplication and a dense vector. It restructures the expression
// \f$ \vec{a}=(B*s1)*\vec{c} \f$ to the expression \f$ \vec{a}=(B*\vec{c})*s1 \f$.
*/
template< typename MT    // Type of the sparse matrix of the left-hand side expression
        , typename ST    // Type of the scalar of the left-hand side expression
        , bool SO        // Storage order of the left-hand side expression
        , typename VT >  // Type of the right-hand side dense vector
inline const MultExprTrait_< SMatScalarMultExpr<MT,ST,SO>, VT >
   operator*( const SMatScalarMultExpr<MT,ST,SO>& mat, const DenseVector<VT,false>& vec )
{
   BLAZE_FUNCTION_TRACE;

   return ( mat.leftOperand() * (~vec) ) * mat.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense vector and a sparse
//        matrix-scalar multiplication expression (\f$ \vec{a}^T=\vec{c}^T*(B*s1) \f$).
// \ingroup dense_vector
//
// \param vec The left-hand side dense vector.
// \param mat The right-hand side sparse matrix-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// dense vector and a sparse matrix-scalar multiplication. It restructures the expression
// \f$ \vec{a}=\vec{c}^T*(B*s1) \f$ to the expression \f$ \vec{a}^T=(\vec{c}^T*B)*s1 \f$.
*/
template< typename VT  // Type of the left-hand side dense vector
        , typename MT  // Type of the sparse matrix of the right-hand side expression
        , typename ST  // Type of the scalar of the right-hand side expression
        , bool SO >    // Storage order of the right-hand side expression
inline const MultExprTrait_< VT, SMatScalarMultExpr<MT,ST,SO> >
   operator*( const DenseVector<VT,true>& vec, const SMatScalarMultExpr<MT,ST,SO>& mat )
{
   BLAZE_FUNCTION_TRACE;

   return ( (~vec) * mat.leftOperand() ) * mat.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a sparse matrix-scalar
//        multiplication expression and a dense vector-scalar multiplication expression
//        (\f$ \vec{a}=(B*s1)*(\vec{c}*s2) \f$).
// \ingroup dense_vector
//
// \param mat The left-hand side sparse matrix-scalar multiplication.
// \param vec The right-hand side dense vector-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication
// of a sparse matrix-scalar multiplication and a dense vector-scalar multiplication. It
// restructures the expression \f$ \vec{a}=(B*s1)*(\vec{c}*s2) \f$ to the expression
// \f$ \vec{a}=(B*\vec{c})*(s1*s2) \f$.
*/
template< typename MT     // Type of the sparse matrix of the left-hand side expression
        , typename ST1    // Type of the scalar of the left-hand side expression
        , bool SO         // Storage order of the left-hand side expression
        , typename VT     // Type of the dense vector of the right-hand side expression
        , typename ST2 >  // Type of the scalar of the right-hand side expression
inline const MultExprTrait_< SMatScalarMultExpr<MT,ST1,SO>, DVecScalarMultExpr<VT,ST2,false> >
   operator*( const SMatScalarMultExpr<MT,ST1,SO>& mat, const DVecScalarMultExpr<VT,ST2,false>& vec )
{
   BLAZE_FUNCTION_TRACE;

   return ( mat.leftOperand() * vec.leftOperand() ) * ( mat.rightOperand() * vec.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense vector-scalar
//        multiplication expression and a sparse matrix-scalar multiplication expression
//        (\f$ \vec{a}^T=\vec{b}^T*(C*s1) \f$).
// \ingroup dense_vector
//
// \param vec The left-hand side dense vector-scalar multiplication.
// \param mat The right-hand side sparse matrix-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication
// of a dense vector-scalar multiplication and a sparse matrix-scalar multiplication. It
// restructures the expression \f$ \vec{a}=(\vec{b}^T*s1)*(C*s2) \f$ to the expression
// \f$ \vec{a}^T=(\vec{b}^T*C)*(s1*s2) \f$.
*/
template< typename VT   // Type of the dense vector of the left-hand side expression
        , typename ST1  // Type of the scalar of the left-hand side expression
        , typename MT   // Type of the sparse matrix of the right-hand side expression
        , typename ST2  // Type of the scalar of the right-hand side expression
        , bool SO >     // Storage order of the right-hand side expression
inline const MultExprTrait_< DVecScalarMultExpr<VT,ST1,true>, SMatScalarMultExpr<MT,ST2,SO> >
   operator*( const DVecScalarMultExpr<VT,ST1,true>& vec, const SMatScalarMultExpr<MT,ST2,SO>& mat )
{
   BLAZE_FUNCTION_TRACE;

   return ( vec.leftOperand() * mat.leftOperand() ) * ( vec.rightOperand() * mat.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a sparse matrix-scalar
//        multiplication expression and a sparse vector (\f$ \vec{a}=(B*s1)*\vec{c} \f$).
// \ingroup sparse_vector
//
// \param mat The left-hand side sparse matrix-scalar multiplication.
// \param vec The right-hand side sparse vector.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// sparse matrix-scalar multiplication and a sparse vector. It restructures the expression
// \f$ \vec{a}=(B*s1)*\vec{c} \f$ to the expression \f$ \vec{a}=(B*\vec{c})*s1 \f$.
*/
template< typename MT    // Type of the sparse matrix of the left-hand side expression
        , typename ST    // Type of the scalar of the left-hand side expression
        , bool SO        // Storage order of the left-hand side expression
        , typename VT >  // Type of the right-hand side sparse vector
inline const MultExprTrait_< SMatScalarMultExpr<MT,ST,SO>, VT >
   operator*( const SMatScalarMultExpr<MT,ST,SO>& mat, const SparseVector<VT,false>& vec )
{
   BLAZE_FUNCTION_TRACE;

   return ( mat.leftOperand() * (~vec) ) * mat.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a sparse vector and a sparse
//        matrix-scalar multiplication expression (\f$ \vec{a}^T=\vec{c}^T*(B*s1) \f$).
// \ingroup sparse_vector
//
// \param vec The left-hand side sparse vector.
// \param mat The right-hand side sparse matrix-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// sparse vector and a sparse matrix-scalar multiplication. It restructures the expression
// \f$ \vec{a}=\vec{c}^T*(B*s1) \f$ to the expression \f$ \vec{a}^T=(\vec{c}^T*B)*s1 \f$.
*/
template< typename VT  // Type of the left-hand side sparse vector
        , typename MT  // Type of the sparse matrix of the right-hand side expression
        , typename ST  // Type of the scalar of the right-hand side expression
        , bool SO >    // Storage order of the right-hand side expression
inline const MultExprTrait_< VT, SMatScalarMultExpr<MT,ST,SO> >
   operator*( const SparseVector<VT,true>& vec, const SMatScalarMultExpr<MT,ST,SO>& mat )
{
   BLAZE_FUNCTION_TRACE;

   return ( (~vec) * mat.leftOperand() ) * mat.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a sparse matrix-scalar
//        multiplication expression and a sparse vector-scalar multiplication expression
//        (\f$ \vec{a}=(B*s1)*(\vec{c}*s2) \f$).
// \ingroup sparse_vector
//
// \param mat The left-hand side sparse matrix-scalar multiplication.
// \param vec The right-hand side sparse vector-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication
// of a sparse matrix-scalar multiplication and a sparse vector-scalar multiplication. It
// restructures the expression \f$ \vec{a}=(B*s1)*(\vec{c}*s2) \f$ to the expression
// \f$ \vec{a}=(B*\vec{c})*(s1*s2) \f$.
*/
template< typename MT     // Type of the sparse matrix of the left-hand side expression
        , typename ST1    // Type of the scalar of the left-hand side expression
        , bool SO         // Storage order of the left-hand side expression
        , typename VT     // Type of the sparse vector of the right-hand side expression
        , typename ST2 >  // Type of the scalar of the right-hand side expression
inline const MultExprTrait_< SMatScalarMultExpr<MT,ST1,SO>, SVecScalarMultExpr<VT,ST2,false> >
   operator*( const SMatScalarMultExpr<MT,ST1,SO>& mat, const SVecScalarMultExpr<VT,ST2,false>& vec )
{
   BLAZE_FUNCTION_TRACE;

   return ( mat.leftOperand() * vec.leftOperand() ) * ( mat.rightOperand() * vec.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a sparse vector-scalar
//        multiplication expression and a sparse matrix-scalar multiplication expression
//        (\f$ \vec{a}^T=\vec{b}^T*(C*s1) \f$).
// \ingroup sparse_vector
//
// \param vec The left-hand side sparse vector-scalar multiplication.
// \param mat The right-hand side sparse matrix-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication
// of a sparse vector-scalar multiplication and a sparse matrix-scalar multiplication. It
// restructures the expression \f$ \vec{a}=(\vec{b}^T*s1)*(C*s2) \f$ to the expression
// \f$ \vec{a}^T=(\vec{b}^T*C)*(s1*s2) \f$.
*/
template< typename VT   // Type of the sparse vector of the left-hand side expression
        , typename ST1  // Type of the scalar of the left-hand side expression
        , typename MT   // Type of the sparse matrix of the right-hand side expression
        , typename ST2  // Type of the scalar of the right-hand side expression
        , bool SO >     // Storage order of the right-hand side expression
inline const MultExprTrait_< SVecScalarMultExpr<VT,ST1,true>, SMatScalarMultExpr<MT,ST2,SO> >
   operator*( const SVecScalarMultExpr<VT,ST1,true>& vec, const SMatScalarMultExpr<MT,ST2,SO>& mat )
{
   BLAZE_FUNCTION_TRACE;

   return ( vec.leftOperand() * mat.leftOperand() ) * ( vec.rightOperand() * mat.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a sparse matrix-scalar multiplication
//        expression and a dense matrix (\f$ A=(B*s1)*C \f$).
// \ingroup dense_matrix
//
// \param lhs The left-hand side sparse matrix-scalar multiplication.
// \param rhs The right-hand side dense matrix.
// \return The scaled result matrix.
//
// This operator implements the performance optimized treatment of the multiplication of a
// sparse matrix-scalar multiplication and a dense matrix. It restructures the expression
// \f$ A=(B*s1)*C \f$ to the expression \f$ A=(B*C)*s1 \f$.
*/
template< typename MT1    // Type of the sparse matrix of the left-hand side expression
        , typename ST     // Type of the scalar of the left-hand side expression
        , bool SO1        // Storage order of the left-hand side expression
        , typename MT2    // Type of the right-hand side dense matrix
        , bool SO2 >      // Storage order of the right-hand side dense matrix
inline const MultExprTrait_< SMatScalarMultExpr<MT1,ST,SO1>, MT2 >
   operator*( const SMatScalarMultExpr<MT1,ST,SO1>& lhs, const DenseMatrix<MT2,SO2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * (~rhs) ) * lhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense matrix and a sparse matrix-
//        scalar multiplication expression (\f$ A=(B*s1)*C \f$).
// \ingroup dense_matrix
//
// \param lhs The left-hand side dense matrix.
// \param rhs The right-hand side sparse matrix-scalar multiplication.
// \return The scaled result matrix.
//
// This operator implements the performance optimized treatment of the multiplication of a
// dense matrix and a sparse matrix-scalar multiplication. It restructures the expression
// \f$ A=B*(C*s1) \f$ to the expression \f$ A=(B*C)*s1 \f$.
*/
template< typename MT1    // Type of the left-hand side dense matrix
        , bool SO1        // Storage order of the left-hand side dense matrix
        , typename MT2    // Type of the sparse matrix of the right-hand side expression
        , typename ST     // Type of the scalar of the right-hand side expression
        , bool SO2 >      // Storage order of the right-hand side expression
inline const MultExprTrait_< MT1, SMatScalarMultExpr<MT2,ST,SO2> >
   operator*( const DenseMatrix<MT1,SO1>& lhs, const SMatScalarMultExpr<MT2,ST,SO2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( (~lhs) * rhs.leftOperand() ) * rhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a sparse matrix-scalar multiplication
//        expression and a sparse matrix (\f$ A=(B*s1)*C \f$).
// \ingroup sparse_matrix
//
// \param lhs The left-hand side sparse matrix-scalar multiplication.
// \param rhs The right-hand side sparse matrix.
// \return The scaled result matrix.
//
// This operator implements the performance optimized treatment of the multiplication of a
// sparse matrix-scalar multiplication and a sparse matrix. It restructures the expression
// \f$ A=(B*s1)*C \f$ to the expression \f$ A=(B*C)*s1 \f$.
*/
template< typename MT1    // Type of the sparse matrix of the left-hand side expression
        , typename ST     // Type of the scalar of the left-hand side expression
        , bool SO1        // Storage order of the left-hand side expression
        , typename MT2    // Type of the right-hand side sparse matrix
        , bool SO2 >      // Storage order of the right-hand side sparse matrix
inline const MultExprTrait_< SMatScalarMultExpr<MT1,ST,SO1>, MT2 >
   operator*( const SMatScalarMultExpr<MT1,ST,SO1>& lhs, const SparseMatrix<MT2,SO2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * (~rhs) ) * lhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a sparse matrix and a sparse matrix-
//        scalar multiplication expression (\f$ A=(B*s1)*C \f$).
// \ingroup sparse_matrix
//
// \param lhs The left-hand side sparse matrix.
// \param rhs The right-hand side sparse matrix-scalar multiplication.
// \return The scaled result matrix.
//
// This operator implements the performance optimized treatment of the multiplication of a
// sparse matrix and a sparse matrix-scalar multiplication. It restructures the expression
// \f$ A=B*(C*s1) \f$ to the expression \f$ A=(B*C)*s1 \f$.
*/
template< typename MT1    // Type of the left-hand side sparse matrix
        , bool SO1        // Storage order of the left-hand side sparse matrix
        , typename MT2    // Type of the sparse matrix of the right-hand side expression
        , typename ST     // Type of the scalar of the right-hand side expression
        , bool SO2 >      // Storage order of the right-hand side expression
inline const MultExprTrait_< MT1, SMatScalarMultExpr<MT2,ST,SO2> >
   operator*( const SparseMatrix<MT1,SO1>& lhs, const SMatScalarMultExpr<MT2,ST,SO2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( (~lhs) * rhs.leftOperand() ) * rhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of two sparse matrix-scalar
//        multiplication expressions (\f$ A=(B*s1)*(C*s2) \f$).
// \ingroup sparse_matrix
//
// \param lhs The left-hand side sparse matrix-scalar multiplication.
// \param rhs The right-hand side sparse matrix-scalar multiplication.
// \return The scaled result matrix.
//
// This operator implements the performance optimized treatment of the multiplication of
// two sparse matrix-scalar multiplication expressions. It restructures the expression
// \f$ A=(B*s1)*(C*s2) \f$ to the expression \f$ A=(B*C)*(s1*s2) \f$.
*/
template< typename MT1  // Type of the sparse matrix of the left-hand side expression
        , typename ST1  // Type of the scalar of the left-hand side expression
        , bool SO1      // Storage order of the left-hand side expression
        , typename MT2  // Type of the right-hand side sparse matrix
        , typename ST2  // Type of the scalar of the right-hand side expression
        , bool SO2 >    // Storage order of the right-hand side expression
inline const MultExprTrait_< SMatScalarMultExpr<MT1,ST1,SO1>, SMatScalarMultExpr<MT2,ST2,SO2> >
   operator*( const SMatScalarMultExpr<MT1,ST1,SO1>& lhs, const SMatScalarMultExpr<MT2,ST2,SO2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * rhs.leftOperand() ) * ( lhs.rightOperand() * rhs.rightOperand() );
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
template< typename MT, typename ST, bool SO >
struct Rows< SMatScalarMultExpr<MT,ST,SO> > : public Columns<MT>
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
template< typename MT, typename ST, bool SO >
struct Columns< SMatScalarMultExpr<MT,ST,SO> > : public Rows<MT>
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
template< typename MT, typename ST, bool SO >
struct IsSymmetric< SMatScalarMultExpr<MT,ST,SO> >
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
template< typename MT, typename ST, bool SO >
struct IsHermitian< SMatScalarMultExpr<MT,ST,SO> >
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
template< typename MT, typename ST, bool SO >
struct IsLower< SMatScalarMultExpr<MT,ST,SO> >
   : public BoolConstant< IsLower<MT>::value >
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
template< typename MT, typename ST, bool SO >
struct IsStrictlyLower< SMatScalarMultExpr<MT,ST,SO> >
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
template< typename MT, typename ST, bool SO >
struct IsUpper< SMatScalarMultExpr<MT,ST,SO> >
   : public BoolConstant< IsUpper<MT>::value >
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
template< typename MT, typename ST, bool SO >
struct IsStrictlyUpper< SMatScalarMultExpr<MT,ST,SO> >
   : public BoolConstant< IsStrictlyUpper<MT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SMATSCALARMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ST1, typename ST2 >
struct SMatScalarMultExprTrait< SMatScalarMultExpr<MT,ST1,false>, ST2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsRowMajorMatrix<MT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , SMatScalarMultExprTrait_< MT, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TSMATSCALARMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ST1, typename ST2 >
struct TSMatScalarMultExprTrait< SMatScalarMultExpr<MT,ST1,true>, ST2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsColumnMajorMatrix<MT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , TSMatScalarMultExprTrait_< MT, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SMATSCALARDIVEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ST1, typename ST2 >
struct SMatScalarDivExprTrait< SMatScalarMultExpr<MT,ST1,false>, ST2 >
{
 private:
   //**********************************************************************************************
   using ScalarType = DivTrait_<ST1,ST2>;
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsRowMajorMatrix<MT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , If_< IsInvertible<ScalarType>
                        , SMatScalarMultExprTrait_<MT,ScalarType>
                        , SMatScalarDivExprTrait_<MT,ScalarType> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TSMATSCALARDIVEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ST1, typename ST2 >
struct TSMatScalarDivExprTrait< SMatScalarMultExpr<MT,ST1,true>, ST2 >
{
 private:
   //**********************************************************************************************
   using ScalarType = DivTrait_<ST1,ST2>;
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsColumnMajorMatrix<MT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , If_< IsInvertible<ScalarType>
                        , TSMatScalarMultExprTrait_<MT,ScalarType>
                        , TSMatScalarDivExprTrait_<MT,ScalarType> >
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
template< typename MT, typename ST, typename VT >
struct SMatDVecMultExprTrait< SMatScalarMultExpr<MT,ST,false>, VT >
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


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ST1, typename VT, typename ST2 >
struct SMatDVecMultExprTrait< SMatScalarMultExpr<MT,ST1,false>, DVecScalarMultExpr<VT,ST2,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsRowMajorMatrix<MT>
                        , IsDenseVector<VT>, IsColumnVector<VT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , DVecScalarMultExprTrait_< SMatDVecMultExprTrait_<MT,VT>, MultTrait_<ST1,ST2> >
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
template< typename MT, typename ST, typename VT >
struct TSMatDVecMultExprTrait< SMatScalarMultExpr<MT,ST,true>, VT >
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


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ST1, typename VT, typename ST2 >
struct TSMatDVecMultExprTrait< SMatScalarMultExpr<MT,ST1,true>, DVecScalarMultExpr<VT,ST2,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsColumnMajorMatrix<MT>
                        , IsDenseVector<VT>, IsColumnVector<VT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , DVecScalarMultExprTrait_< TSMatDVecMultExprTrait_<MT,VT>, MultTrait_<ST1,ST2> >
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
struct TDVecSMatMultExprTrait< VT, SMatScalarMultExpr<MT,ST,false> >
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


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename ST1, typename MT, typename ST2 >
struct TDVecSMatMultExprTrait< DVecScalarMultExpr<VT,ST1,true>, SMatScalarMultExpr<MT,ST2,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT>, IsRowVector<VT>
                        , IsSparseMatrix<MT>, IsRowMajorMatrix<MT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , TDVecScalarMultExprTrait_< TDVecSMatMultExprTrait_<VT,MT>, MultTrait_<ST1,ST2> >
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
struct TDVecTSMatMultExprTrait< VT, SMatScalarMultExpr<MT,ST,true> >
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


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename ST1, typename MT, typename ST2 >
struct TDVecTSMatMultExprTrait< DVecScalarMultExpr<VT,ST1,true>, SMatScalarMultExpr<MT,ST2,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT>, IsRowVector<VT>
                        , IsSparseMatrix<MT>, IsColumnMajorMatrix<MT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , TDVecScalarMultExprTrait_< TDVecTSMatMultExprTrait_<VT,MT>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SMATSVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ST, typename VT >
struct SMatSVecMultExprTrait< SMatScalarMultExpr<MT,ST,false>, VT >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsRowMajorMatrix<MT>
                        , IsSparseVector<VT>, IsColumnVector<VT>
                        , IsNumeric<ST> >
                   , SVecScalarMultExprTrait_< SMatSVecMultExprTrait_<MT,VT>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ST1, typename VT, typename ST2 >
struct SMatSVecMultExprTrait< SMatScalarMultExpr<MT,ST1,false>, SVecScalarMultExpr<VT,ST2,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsRowMajorMatrix<MT>
                        , IsSparseVector<VT>, IsColumnVector<VT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , SVecScalarMultExprTrait_< SMatSVecMultExprTrait_<MT,VT>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TSMATSVECMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ST, typename VT >
struct TSMatSVecMultExprTrait< SMatScalarMultExpr<MT,ST,true>, VT >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsColumnMajorMatrix<MT>
                        , IsSparseVector<VT>, IsColumnVector<VT>
                        , IsNumeric<ST> >
                   , SVecScalarMultExprTrait_< TSMatSVecMultExprTrait_<MT,VT>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ST1, typename VT, typename ST2 >
struct TSMatSVecMultExprTrait< SMatScalarMultExpr<MT,ST1,true>, SVecScalarMultExpr<VT,ST2,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT>, IsColumnMajorMatrix<MT>
                        , IsSparseVector<VT>, IsColumnVector<VT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , SVecScalarMultExprTrait_< TSMatSVecMultExprTrait_<MT,VT>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TSVECSMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename MT, typename ST >
struct TSVecSMatMultExprTrait< VT, SMatScalarMultExpr<MT,ST,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsRowVector<VT>
                        , IsSparseMatrix<MT>, IsRowMajorMatrix<MT>
                        , IsNumeric<ST> >
                   , TSVecScalarMultExprTrait_< TSVecSMatMultExprTrait_<VT,MT>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename ST1, typename MT, typename ST2 >
struct TSVecSMatMultExprTrait< SVecScalarMultExpr<VT,ST1,true>, SMatScalarMultExpr<MT,ST2,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsRowVector<VT>
                        , IsSparseMatrix<MT>, IsRowMajorMatrix<MT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , TSVecScalarMultExprTrait_< TSVecSMatMultExprTrait_<VT,MT>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TSVECTSMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename MT, typename ST >
struct TSVecTSMatMultExprTrait< VT, SMatScalarMultExpr<MT,ST,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsRowVector<VT>
                        , IsSparseMatrix<MT>, IsColumnMajorMatrix<MT>
                        , IsNumeric<ST> >
                   , TSVecScalarMultExprTrait_< TSVecTSMatMultExprTrait_<VT,MT>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename ST1, typename MT, typename ST2 >
struct TSVecTSMatMultExprTrait< SVecScalarMultExpr<VT,ST1,true>, SMatScalarMultExpr<MT,ST2,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsRowVector<VT>
                        , IsSparseMatrix<MT>, IsColumnMajorMatrix<MT>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , TSVecScalarMultExprTrait_< TSVecTSMatMultExprTrait_<VT,MT>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DMATSMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2, typename ST >
struct DMatSMatMultExprTrait< MT1, SMatScalarMultExpr<MT2,ST,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , DMatScalarMultExprTrait_< DMatSMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DMATTSMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2, typename ST >
struct DMatTSMatMultExprTrait< MT1, SMatScalarMultExpr<MT2,ST,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsColumnMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , DMatScalarMultExprTrait_< DMatTSMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TDMATSMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2, typename ST >
struct TDMatSMatMultExprTrait< MT1, SMatScalarMultExpr<MT2,ST,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , TDMatScalarMultExprTrait_< TDMatSMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TDMATTSMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2, typename ST >
struct TDMatTSMatMultExprTrait< MT1, SMatScalarMultExpr<MT2,ST,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsColumnMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , TDMatScalarMultExprTrait_< TDMatTSMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SMATDMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename ST, typename MT2 >
struct SMatDMatMultExprTrait< SMatScalarMultExpr<MT1,ST,false>, MT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsDenseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , DMatScalarMultExprTrait_< SMatDMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SMATTDMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename ST, typename MT2 >
struct SMatTDMatMultExprTrait< SMatScalarMultExpr<MT1,ST,false>, MT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsDenseMatrix<MT2>, IsColumnMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , DMatScalarMultExprTrait_< SMatTDMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TSMATDMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename ST, typename MT2 >
struct TSMatDMatMultExprTrait< SMatScalarMultExpr<MT1,ST,true>, MT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsDenseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , TDMatScalarMultExprTrait_< TSMatDMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TSMATTDMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename ST, typename MT2 >
struct TSMatTDMatMultExprTrait< SMatScalarMultExpr<MT1,ST,true>, MT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsDenseMatrix<MT2>, IsColumnMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , TDMatScalarMultExprTrait_< TSMatTDMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SMATSMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename ST, typename MT2 >
struct SMatSMatMultExprTrait< SMatScalarMultExpr<MT1,ST,false>, MT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , SMatScalarMultExprTrait_< SMatSMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2, typename ST >
struct SMatSMatMultExprTrait< MT1, SMatScalarMultExpr<MT2,ST,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , SMatScalarMultExprTrait_< SMatSMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename ST1, typename MT2, typename ST2 >
struct SMatSMatMultExprTrait< SMatScalarMultExpr<MT1,ST1,false>, SMatScalarMultExpr<MT2,ST2,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , SMatScalarMultExprTrait_< SMatSMatMultExprTrait_<MT1,MT2>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SMATTSMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename ST, typename MT2 >
struct SMatTSMatMultExprTrait< SMatScalarMultExpr<MT1,ST,false>, MT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsColumnMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , SMatScalarMultExprTrait_< SMatTSMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2, typename ST >
struct SMatTSMatMultExprTrait< MT1, SMatScalarMultExpr<MT2,ST,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsColumnMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , SMatScalarMultExprTrait_< SMatTSMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename ST1, typename MT2, typename ST2 >
struct SMatTSMatMultExprTrait< SMatScalarMultExpr<MT1,ST1,false>, SMatScalarMultExpr<MT2,ST2,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsColumnMajorMatrix<MT2>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , SMatScalarMultExprTrait_< SMatTSMatMultExprTrait_<MT1,MT2>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TSMATSMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename ST, typename MT2 >
struct TSMatSMatMultExprTrait< SMatScalarMultExpr<MT1,ST,true>, MT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , TSMatScalarMultExprTrait_< TSMatSMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2, typename ST >
struct TSMatSMatMultExprTrait< MT1, SMatScalarMultExpr<MT2,ST,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , TSMatScalarMultExprTrait_< TSMatSMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename ST1, typename MT2, typename ST2 >
struct TSMatSMatMultExprTrait< SMatScalarMultExpr<MT1,ST1,true>, SMatScalarMultExpr<MT2,ST2,false> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , TSMatScalarMultExprTrait_< TSMatSMatMultExprTrait_<MT1,MT2>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TSMATSMATMULTEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename ST, typename MT2 >
struct TSMatTSMatMultExprTrait< SMatScalarMultExpr<MT1,ST,true>, MT2 >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsColumnMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , TSMatScalarMultExprTrait_< TSMatTSMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2, typename ST >
struct TSMatTSMatMultExprTrait< MT1, SMatScalarMultExpr<MT2,ST,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsColumnMajorMatrix<MT2>
                        , IsNumeric<ST> >
                   , TSMatScalarMultExprTrait_< TSMatTSMatMultExprTrait_<MT1,MT2>, ST >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename ST1, typename MT2, typename ST2 >
struct TSMatTSMatMultExprTrait< SMatScalarMultExpr<MT1,ST1,true>, SMatScalarMultExpr<MT2,ST2,true> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsSparseMatrix<MT2>, IsColumnMajorMatrix<MT2>
                        , IsNumeric<ST1>, IsNumeric<ST2> >
                   , TSMatScalarMultExprTrait_< TSMatTSMatMultExprTrait_<MT1,MT2>, MultTrait_<ST1,ST2> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBMATRIXEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ST, bool SO, bool AF >
struct SubmatrixExprTrait< SMatScalarMultExpr<MT,ST,SO>, AF >
{
 public:
   //**********************************************************************************************
   using Type = MultExprTrait_< SubmatrixExprTrait_<const MT,AF>, ST >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ROWEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ST, bool SO >
struct RowExprTrait< SMatScalarMultExpr<MT,ST,SO> >
{
 public:
   //**********************************************************************************************
   using Type = MultExprTrait_< RowExprTrait_<const MT>, ST >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COLUMNEXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename ST, bool SO >
struct ColumnExprTrait< SMatScalarMultExpr<MT,ST,SO> >
{
 public:
   //**********************************************************************************************
   using Type = MultExprTrait_< ColumnExprTrait_<const MT>, ST >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
