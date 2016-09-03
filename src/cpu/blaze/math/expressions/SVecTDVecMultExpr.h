//=================================================================================================
/*!
//  \file blaze/math/expressions/SVecTDVecMultExpr.h
//  \brief Header file for the sparse vector/dense vector outer product expression
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

#ifndef _BLAZE_MATH_EXPRESSIONS_SVECTDVECMULTEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_SVECTDVECMULTEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/ColumnVector.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/RowVector.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/constraints/SparseVector.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/VecTVecMultExpr.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/expressions/VecTVecMultExpr.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/sparse/ValueIndexPair.h>
#include <blaze/math/traits/ColumnExprTrait.h>
#include <blaze/math/traits/MultExprTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/RowExprTrait.h>
#include <blaze/math/traits/SubmatrixExprTrait.h>
#include <blaze/math/traits/SubvectorExprTrait.h>
#include <blaze/math/typetraits/Columns.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsTemporary.h>
#include <blaze/math/typetraits/Rows.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/system/Optimizations.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/logging/FunctionTrace.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsSame.h>
#include <blaze/util/typetraits/RemoveReference.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  CLASS SVECTDVECMULTEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for sparse vector-dense vector outer products.
// \ingroup sparse_matrix_expression
//
// The SVecTDVecMultExpr class represents the compile time expression for sparse vector-dense
// vector outer products.
*/
template< typename VT1    // Type of the left-hand side sparse vector
        , typename VT2 >  // Type of the right-hand side dense vector
class SVecTDVecMultExpr : public SparseMatrix< SVecTDVecMultExpr<VT1,VT2>, true >
                        , private VecTVecMultExpr
                        , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef ResultType_<VT1>     RT1;  //!< Result type of the left-hand side sparse vector expression.
   typedef ResultType_<VT2>     RT2;  //!< Result type of the right-hand side dense vector expression.
   typedef ReturnType_<VT1>     RN1;  //!< Return type of the left-hand side sparse vector expression.
   typedef ReturnType_<VT2>     RN2;  //!< Return type of the right-hand side dense vector expression.
   typedef CompositeType_<VT1>  CT1;  //!< Composite type of the left-hand side sparse vector expression.
   typedef CompositeType_<VT2>  CT2;  //!< Composite type of the right-hand side dense vector expression.
   typedef ElementType_<VT1>    ET1;  //!< Element type of the left-hand side sparse vector expression.
   typedef ElementType_<VT2>    ET2;  //!< Element type of the right-hand side dense vector expression.
   //**********************************************************************************************

   //**Return type evaluation**********************************************************************
   //! Compilation switch for the selection of the subscript operator return type.
   /*! The \a returnExpr compile time constant expression is a compilation switch for the
       selection of the \a ReturnType. If either vector operand returns a temporary vector
       or matrix, \a returnExpr will be set to \a false and the subscript operator will
       return it's result by value. Otherwise \a returnExpr will be set to \a true and
       the subscript operator may return it's result as an expression. */
   enum : bool { returnExpr = !IsTemporary<RN1>::value && !IsTemporary<RN2>::value };

   //! Expression return type for the subscript operator.
   typedef MultExprTrait_<RN1,RN2>  ExprReturnType;
   //**********************************************************************************************

   //**Evaluation strategy*************************************************************************
   //! Compilation switch for the evaluation strategy of the multiplication expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for
       the evaluation strategy of the multiplication expression. In case either the sparse or
       the dense vector operand is an expression or if any of two involved element types is not
       a numeric data type, \a useAssign will be set to \a true and the multiplication expression
       will be evaluated via the \a assign function family. Otherwise \a useAssign will be set to
       \a false and the expression will be evaluated via the subscript operator. */
   enum : bool { useAssign = ( IsComputation<VT1>::value || !IsNumeric<ET1>::value ||
                               IsComputation<VT2>::value || !IsNumeric<ET2>::value ) };

   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT >
   struct UseAssign {
      enum : bool { value = useAssign };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! In case all three involved data types are suited for a vectorized computation of the
       outer product, the nested \value will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   struct UseVectorizedKernel {
      enum : bool { value = useOptimizedKernels &&
                            T1::simdEnabled && T3::simdEnabled &&
                            IsSame< ElementType_<T1>, ElementType_<T2> >::value &&
                            IsSame< ElementType_<T1>, ElementType_<T3> >::value &&
                            HasSIMDMult< ElementType_<T1>, ElementType_<T1> >::value };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! In case no vectorized computation is possible, the nested \value will be set to 1,
       otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   struct UseDefaultKernel {
      enum : bool { value = !UseVectorizedKernel<T1,T2,T3>::value };
   };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef SVecTDVecMultExpr<VT1,VT2>  This;           //!< Type of this SVecTDVecMultExpr instance.
   typedef MultTrait_<RT1,RT2>         ResultType;     //!< Result type for expression template evaluations.
   typedef OppositeType_<ResultType>   OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef TransposeType_<ResultType>  TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<ResultType>    ElementType;    //!< Resulting element type.

   //! Return type for expression template evaluations.
   typedef const IfTrue_< returnExpr, ExprReturnType, ElementType >  ReturnType;

   //! Data type for composite expression templates.
   typedef IfTrue_< useAssign, const ResultType, const SVecTDVecMultExpr& >  CompositeType;

   //! Composite type of the left-hand side sparse vector expression.
   typedef If_< IsExpression<VT1>, const VT1, const VT1& >  LeftOperand;

   //! Composite type of the right-hand side dense vector expression.
   typedef If_< IsExpression<VT2>, const VT2, const VT2& >  RightOperand;

   //! Type for the assignment of the left-hand side dense vector operand.
   typedef If_< IsComputation<VT1>, const RT1, CT1 >  LT;

   //! Type for the assignment of the right-hand side dense vector operand.
   typedef If_< IsComputation<VT2>, const RT2, CT2 >  RT;
   //**********************************************************************************************

   //**ConstIterator class definition**************************************************************
   /*!\brief Iterator over the elements of the sparse vector-dense vector outer product expression.
   */
   class ConstIterator
   {
    public:
      //**Type definitions*************************************************************************
      //! Element type of the sparse matrix expression.
      typedef ValueIndexPair<ElementType>  Element;

      //! Iterator type of the sparse vector expression.
      typedef ConstIterator_< RemoveReference_<LeftOperand> >  IteratorType;

      //! Element type of the dense vector expression
      typedef ET2  RightElement;

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
      inline ConstIterator( IteratorType it, RightElement v )
         : it_( it )  // Iterator over the elements of the left-hand side sparse vector expression
         , v_ ( v  )  // Element of the right-hand side dense vector expression.
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
         return Element( it_->value() * v_, it_->index() );
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
         return it_->value() * v_;
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
      IteratorType it_;  //!< Iterator over the elements of the left-hand side sparse vector expression
      RightElement v_;   //!< Element of the right-hand side dense vector expression.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = false };
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   enum : size_t { SIMDSIZE = SIMDTrait<ElementType>::size };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SVecTDVecMultExpr class.
   //
   // \param lhs The left-hand side sparse vector operand of the multiplication expression.
   // \param rhs The right-hand side dense vector operand of the multiplication expression.
   */
   explicit inline SVecTDVecMultExpr( const VT1& lhs, const VT2& rhs ) noexcept
      : lhs_( lhs )  // Left-hand side sparse vector of the multiplication expression
      , rhs_( rhs )  // Right-hand side dense vector of the multiplication expression
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
      BLAZE_INTERNAL_ASSERT( i < lhs_.size(), "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < rhs_.size(), "Invalid column access index" );

      return lhs_[i] * rhs_[j];
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
      if( i >= lhs_.size() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( j >= rhs_.size() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
      }
      return (*this)(i,j);
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of column \a i.
   //
   // \param i The row index.
   // \return Iterator to the first non-zero element of column \a i.
   */
   inline ConstIterator begin( size_t i ) const {
      return ConstIterator( lhs_.begin(), rhs_[i] );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of column \a i.
   //
   // \param i The row index.
   // \return Iterator just past the last non-zero element of column \a i.
   */
   inline ConstIterator end( size_t i ) const {
      return ConstIterator( lhs_.end(), rhs_[i] );
   }
   //**********************************************************************************************

   //**Rows function*******************************************************************************
   /*!\brief Returns the current number of rows of the matrix.
   //
   // \return The number of rows of the matrix.
   */
   inline size_t rows() const noexcept {
      return lhs_.size();
   }
   //**********************************************************************************************

   //**Columns function****************************************************************************
   /*!\brief Returns the current number of columns of the matrix.
   //
   // \return The number of columns of the matrix.
   */
   inline size_t columns() const noexcept {
      return rhs_.size();
   }
   //**********************************************************************************************

   //**NonZeros function***************************************************************************
   /*!\brief Returns the number of non-zero elements in the sparse matrix.
   //
   // \return The number of non-zero elements in the sparse matrix.
   */
   inline size_t nonZeros() const {
      return lhs_.nonZeros() * rhs_.size();
   }
   //**********************************************************************************************

   //**NonZeros function***************************************************************************
   /*!\brief Returns the number of non-zero elements in the specified column.
   //
   // \param i The index of the column.
   // \return The number of non-zero elements of column \a i.
   */
   inline size_t nonZeros( size_t i ) const {
      UNUSED_PARAMETER( i );
      return lhs_.nonZeros();
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
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT1 );
      return ConstIterator( lhs_.find( i ), rhs_[j] );
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
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT1 );
      return ConstIterator( lhs_.lowerBound( i ), rhs_[j] );
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
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT1 );
      return ConstIterator( lhs_.upperBound( i ), rhs_[j] );
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side sparse vector operand.
   //
   // \return The left-hand side sparse vector operand.
   */
   inline LeftOperand leftOperand() const noexcept {
      return lhs_;
   }
   //**********************************************************************************************

   //**Right operand access************************************************************************
   /*!\brief Returns the right-hand side dense vector operand.
   //
   // \return The right-hand side dense vector operand.
   */
   inline RightOperand rightOperand() const noexcept {
      return rhs_;
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
      return ( lhs_.canAlias( alias ) || rhs_.canAlias( alias ) );
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
      return ( lhs_.isAliased( alias ) || rhs_.isAliased( alias ) );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  lhs_;  //!< Left-hand side sparse vector of the multiplication expression.
   RightOperand rhs_;  //!< Right-hand side dense vector of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to row-major dense matrices******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse vector-dense vector outer product to a row-major dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side outer product expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse vector-dense
   // vector outer product expression to a row-major dense matrix.
   */
   template< typename MT >  // Type of the target dense matrix
   friend inline void assign( DenseMatrix<MT,false>& lhs, const SVecTDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( x.size() == rhs.lhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == rhs.rhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( x.size() == (~lhs).rows()   , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == (~lhs).columns(), "Invalid vector size" );

      SVecTDVecMultExpr::selectAssignKernel( ~lhs, x, y );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to row-major dense matrices**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a sparse vector-dense vector outer product to a row-major
   //        dense matrix (\f$ A=\vec{x}*\vec{y}^T \f$).
   // \ingroup dense_vector
   //
   // \param A The target left-hand side dense matrix.
   // \param x The left-hand side sparse vector operand.
   // \param y The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the default assignment kernel for the sparse vector-dense vector
   // outer product.
   */
   template< typename MT     // Type of the left-hand side target matrix
           , typename VT3    // Type of the left-hand side vector operand
           , typename VT4 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseDefaultKernel<MT,VT3,VT4> >
      selectAssignKernel( DenseMatrix<MT,false>& A, const VT3& x, const VT4& y )
   {
      typedef ConstIterator_< RemoveReference_<LT> >  ConstIterator;

      const ConstIterator end( x.end() );

      for( ConstIterator element=x.begin(); element!=end; ++element ) {
         if( !isDefault( element->value() ) ) {
            for( size_t j=0UL; j<y.size(); ++j ) {
               (~A)(element->index(),j) = element->value() * y[j];
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized assignment to row-major dense matrices*******************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized assignment of a sparse vector-dense vector outer product to a row-major
   //        dense matrix (\f$ A=\vec{x}*\vec{y}^T \f$).
   // \ingroup dense_vector
   //
   // \param A The target left-hand side dense matrix.
   // \param x The left-hand side sparse vector operand.
   // \param y The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the vectorized assignment kernel for the sparse vector-dense vector
   // outer product.
   */
   template< typename MT     // Type of the left-hand side target matrix
           , typename VT3    // Type of the left-hand side vector operand
           , typename VT4 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseVectorizedKernel<MT,VT3,VT4> >
      selectAssignKernel( DenseMatrix<MT,false>& A, const VT3& x, const VT4& y )
   {
      typedef ConstIterator_< RemoveReference_<LT> >  ConstIterator;

      const size_t N( (~A).columns() );

      const bool remainder( !IsPadded<MT>::value || !IsPadded<VT4>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      const ConstIterator begin( x.begin() );
      const ConstIterator end  ( x.end()   );

      for( ConstIterator element=begin; element!=end; ++element )
      {
         const SIMDTrait_<ElementType> x1( set( element->value() ) );

         size_t j( 0UL );

         for( ; j<jpos; j+=SIMDSIZE ) {
            (~A).store( element->index(), j, x1 * y.load(j) );
         }
         for( ; remainder && j<N; ++j ) {
            (~A)(element->index(),j) = element->value() * y[j];
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to column-major dense matrices***************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse vector-dense vector outer product to a column-major
   //        dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side outer product expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse vector-dense
   // vector outer product expression to a column-major dense matrix. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case either of the two operands is an expression or any of the two involved element
   // types is non-numeric data type.
   */
   template< typename MT >  // Type of the target dense matrix
   friend inline EnableIf_< UseAssign<MT> >
      assign( DenseMatrix<MT,true>& lhs, const SVecTDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      typedef ConstIterator_< RemoveReference_<LT> >  ConstIterator;

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( x.size() == rhs.lhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == rhs.rhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( x.size() == (~lhs).rows()   , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == (~lhs).columns(), "Invalid vector size" );

      const ConstIterator end( x.end() );

      for( size_t i=0UL; i<y.size(); ++i ) {
         if( !isDefault( y[i] ) ) {
            for( ConstIterator element=x.begin(); element!=end; ++element ) {
               (~lhs)(element->index(),i) = element->value() * y[i];
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to row-major sparse matrices*****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse vector-dense vector outer product to a row-major sparse matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side sparse matrix.
   // \param rhs The right-hand side outer product expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse vector-dense
   // vector outer product expression to a row-major sparse matrix.
   */
   template< typename MT >  // Type of the target sparse matrix
   friend inline void assign( SparseMatrix<MT,false>& lhs, const SVecTDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns() , "Invalid number of columns" );

      typedef ConstIterator_< RemoveReference_<LT> >  ConstIterator;

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( x.size() == rhs.lhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == rhs.rhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( x.size() == (~lhs).rows()   , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == (~lhs).columns(), "Invalid vector size" );

      const ConstIterator begin( x.begin() );
      const ConstIterator end  ( x.end()   );

      if( begin == end )
         return;

      (~lhs).reserve( begin->index(), rhs.nonZeros() );

      size_t index( 0UL );

      for( ConstIterator element=begin; element!=end; ++element ) {
         if( !isDefault( element->value() ) ) {
            for( ; index < element->index(); ++index ) {
               (~lhs).finalize( index );
            }
            for( size_t i=0UL; i<y.size(); ++i ) {
               (~lhs).append( element->index(), i, element->value() * y[i] );
            }
            (~lhs).finalize( index++ );
         }
      }

      for( ; index < x.size(); ++index ) {
         (~lhs).finalize( index );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to column-major sparse matrices**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse vector-dense vector outer product to a column-major
   //        sparse matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side sparse matrix.
   // \param rhs The right-hand side outer product expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse vector-dense
   // vector outer product expression to a column-major sparse matrix. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case either of the two operands is an expression or any of the two involved element
   // types is non-numeric data type.
   */
   template< typename MT >  // Type of the target sparse matrix
   friend inline EnableIf_< UseAssign<MT> >
      assign( SparseMatrix<MT,true>& lhs, const SVecTDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()     == rhs.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns()  == rhs.columns() , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (~lhs).capacity() >= rhs.nonZeros(), "Insufficient capacity"     );

      typedef ConstIterator_< RemoveReference_<LT> >  ConstIterator;

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( x.size() == rhs.lhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == rhs.rhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( x.size() == (~lhs).rows()   , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == (~lhs).columns(), "Invalid vector size" );

      const ConstIterator begin( x.begin() );
      const ConstIterator end  ( x.end()   );

      if( begin == end )
         return;

      for( size_t i=0UL; i<y.size(); ++i ) {
         if( !isDefault( y[i] ) ) {
            for( ConstIterator element=begin; element!=end; ++element ) {
               (~lhs).append( element->index(), i, element->value() * y[i] );
            }
         }
         (~lhs).finalize( i );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to row-major dense matrices*********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a sparse vector-dense vector outer product to a row-major
   //        dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side outer product expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a sparse vector-
   // dense vector outer product expression to a row-major dense matrix.
   */
   template< typename MT >  // Type of the target dense matrix
   friend inline void addAssign( DenseMatrix<MT,false>& lhs, const SVecTDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( x.size() == rhs.lhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == rhs.rhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( x.size() == (~lhs).rows()   , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == (~lhs).columns(), "Invalid vector size" );

      SVecTDVecMultExpr::selectAddAssignKernel( ~lhs, x, y );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to row-major dense matrices*************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a sparse vector-dense vector outer product to a
   //        row-major dense matrix (\f$ A+=\vec{x}*\vec{y}^T \f$).
   // \ingroup dense_vector
   //
   // \param A The target left-hand side dense matrix.
   // \param x The left-hand side sparse vector operand.
   // \param y The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the default addition assignment kernel for the sparse vector-dense
   // vector outer product.
   */
   template< typename MT     // Type of the left-hand side target matrix
           , typename VT3    // Type of the left-hand side vector operand
           , typename VT4 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseDefaultKernel<MT,VT3,VT4> >
      selectAddAssignKernel( DenseMatrix<MT,false>& A, const VT3& x, const VT4& y )
   {
      typedef ConstIterator_< RemoveReference_<LT> >  ConstIterator;

      const ConstIterator end( x.end() );

      for( ConstIterator element=x.begin(); element!=end; ++element ) {
         if( !isDefault( element->value() ) ) {
            for( size_t i=0UL; i<y.size(); ++i ) {
               (~A)(element->index(),i) += element->value() * y[i];
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized addition assignment to row-major dense matrices**********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized addition assignment of a sparse vector-dense vector outer product to a
   //        row-major dense matrix (\f$ A+=\vec{x}*\vec{y}^T \f$).
   // \ingroup dense_vector
   //
   // \param A The target left-hand side dense matrix.
   // \param x The left-hand side sparse vector operand.
   // \param y The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the vectorized addition assignment kernel for the sparse vector-
   // dense vector outer product.
   */
   template< typename MT     // Type of the left-hand side target matrix
           , typename VT3    // Type of the left-hand side vector operand
           , typename VT4 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseVectorizedKernel<MT,VT3,VT4> >
      selectAddAssignKernel( DenseMatrix<MT,false>& A, const VT3& x, const VT4& y )
   {
      typedef ConstIterator_< RemoveReference_<LT> >  ConstIterator;

      const size_t N( (~A).columns() );

      const bool remainder( !IsPadded<MT>::value || !IsPadded<VT4>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      const ConstIterator begin( x.begin() );
      const ConstIterator end  ( x.end()   );

      for( ConstIterator element=begin; element!=end; ++element )
      {
         const SIMDTrait_<ElementType> x1( set( element->value() ) );

         size_t j( 0UL );

         for( ; j<jpos; j+=SIMDSIZE ) {
            (~A).store( element->index(), j, (~A).load(element->index(),j) + x1 * y.load(j) );
         }
         for( ; remainder && j<N; ++j ) {
            (~A)(element->index(),j) += element->value() * y[j];
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to column-major dense matrices******************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a sparse vector-dense vector outer product to a column-major
   //        dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side outer product expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a sparse vector-
   // dense vector outer product expression to a column-major dense matrix. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case either of the two operands is an expression or any of the two involved element
   // types is non-numeric data type.
   */
   template< typename MT >  // Type of the target dense matrix
   friend inline EnableIf_< UseAssign<MT> >
      addAssign( DenseMatrix<MT,true>& lhs, const SVecTDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      typedef ConstIterator_< RemoveReference_<LT> >  ConstIterator;

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( x.size() == rhs.lhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == rhs.rhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( x.size() == (~lhs).rows()   , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == (~lhs).columns(), "Invalid vector size" );

      const ConstIterator end( x.end() );

      for( size_t i=0UL; i<y.size(); ++i ) {
         if( !isDefault( y[i] ) ) {
            for( ConstIterator element=x.begin(); element!=end; ++element ) {
               (~lhs)(element->index(),i) += element->value() * y[i];
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to sparse matrices******************************************************
   // No special implementation for the addition assignment to sparse matrices.
   //**********************************************************************************************

   //**Subtraction assignment to row-major dense matrices******************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a sparse vector-dense vector outer product to a row-major
   //        dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side outer product expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a sparse vector-
   // dense vector outer product expression to a row-major dense matrix.
   */
   template< typename MT >  // Type of the target dense matrix
   friend inline void subAssign( DenseMatrix<MT,false>& lhs, const SVecTDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( x.size() == rhs.lhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == rhs.rhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( x.size() == (~lhs).rows()   , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == (~lhs).columns(), "Invalid vector size" );

      SVecTDVecMultExpr::selectSubAssignKernel( ~lhs, x, y );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to row-major dense matrices**********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a sparse vector-dense vector outer product to a
   //        row-major dense matrix (\f$ A-=\vec{x}*\vec{y}^T \f$).
   // \ingroup dense_vector
   //
   // \param A The target left-hand side dense matrix.
   // \param x The left-hand side sparse vector operand.
   // \param y The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the default subtraction assignment kernel for the sparse vector-
   // dense vector outer product.
   */
   template< typename MT     // Type of the left-hand side target matrix
           , typename VT3    // Type of the left-hand side vector operand
           , typename VT4 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseDefaultKernel<MT,VT3,VT4> >
      selectSubAssignKernel( DenseMatrix<MT,false>& A, const VT3& x, const VT4& y )
   {
      typedef ConstIterator_< RemoveReference_<LT> >  ConstIterator;

      const ConstIterator end( x.end() );

      for( ConstIterator element=x.begin(); element!=end; ++element ) {
         if( !isDefault( element->value() ) ) {
            for( size_t i=0UL; i<y.size(); ++i ) {
               (~A)(element->index(),i) -= element->value() * y[i];
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized subtraction assignment to row-major dense matrices*******************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized subtraction assignment of a sparse vector-dense vector outer product to a
   //        row-major dense matrix (\f$ A-=\vec{x}*\vec{y}^T \f$).
   // \ingroup dense_vector
   //
   // \param A The target left-hand side dense matrix.
   // \param x The left-hand side sparse vector operand.
   // \param y The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the vectorized subtraction assignment kernel for the sparse vector-
   // dense vector outer product.
   */
   template< typename MT     // Type of the left-hand side target matrix
           , typename VT3    // Type of the left-hand side vector operand
           , typename VT4 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseVectorizedKernel<MT,VT3,VT4> >
      selectSubAssignKernel( DenseMatrix<MT,false>& A, const VT3& x, const VT4& y )
   {
      typedef ConstIterator_< RemoveReference_<LT> >  ConstIterator;

      const size_t N( (~A).columns() );

      const bool remainder( !IsPadded<MT>::value || !IsPadded<VT4>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      const ConstIterator begin( x.begin() );
      const ConstIterator end  ( x.end()   );

      for( ConstIterator element=begin; element!=end; ++element )
      {
         const SIMDTrait_<ElementType> x1( set( element->value() ) );

         size_t j( 0UL );

         for( ; j<jpos; j+=SIMDSIZE ) {
            (~A).store( element->index(), j, (~A).load(element->index(),j) - x1 * y.load(j) );
         }
         for( ; remainder && j<N; ++j ) {
            (~A)(element->index(),j) -= element->value() * y[j];
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to column-major dense matrices***************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a sparse vector-dense vector outer product to a column-major
   //        dense matrix.
   // \ingroup sparse_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side outer product expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a sparse vector-
   // dense vector outer product expression to a column-major dense matrix. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case either of the two operands is an expression or any of the two involved element
   // types is non-numeric data type.
   */
   template< typename MT >  // Type of the target dense matrix
   friend inline EnableIf_< UseAssign<MT> >
      subAssign( DenseMatrix<MT,true>& lhs, const SVecTDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      typedef ConstIterator_< RemoveReference_<LT> >  ConstIterator;

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( x.size() == rhs.lhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == rhs.rhs_.size() , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( x.size() == (~lhs).rows()   , "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( y.size() == (~lhs).columns(), "Invalid vector size" );

      const ConstIterator end( x.end() );

      for( size_t i=0UL; i<y.size(); ++i ) {
         if( !isDefault( y[i] ) ) {
            for( ConstIterator element=x.begin(); element!=end; ++element ) {
               (~lhs)(element->index(),i) -= element->value() * y[i];
            }
         }
      }
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

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( VT1 );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( VT1 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( VT2 );
   BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE   ( VT2 );
   BLAZE_CONSTRAINT_MUST_FORM_VALID_VECTVECMULTEXPR( VT1, VT2 );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Multiplication operator for the sparse vector-dense vector outer product
//        (\f$ A=\vec{b}*\vec{c}^T \f$).
// \ingroup sparse_matrix
//
// \param lhs The left-hand side sparse vector for the outer product.
// \param rhs The right-hand side transpose dense vector for the outer product.
// \return The resulting sparse matrix.
//
// This operator represents the outer product between a sparse vector and a transpose dense
// vector:

   \code
   using blaze::columnVector;
   using blaze::rowVector;
   using blaze::rowMajor;

   blaze::CompressedVector<double,columnVector> a;
   blaze::DynamicVector<double,rowVector> b;
   blaze::CompressedMatrix<double<rowMajor> A;
   // ... Resizing and initialization
   A = a * b;
   \endcode

// The operator returns an expression representing a sparse matrix of the higher-order element
// type of the two involved element types \a T1::ElementType and \a T2::ElementType. Both
// vector types \a T1 and \a T2 as well as the two element types \a T1::ElementType and
// \a T2::ElementType have to be supported by the MultTrait class template.
*/
template< typename T1    // Type of the left-hand side sparse vector
        , typename T2 >  // Type of the right-hand side dense vector
inline const SVecTDVecMultExpr<T1,T2>
   operator*( const SparseVector<T1,false>& lhs, const DenseVector<T2,true>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return SVecTDVecMultExpr<T1,T2>( ~lhs, ~rhs );
}
//*************************************************************************************************




//=================================================================================================
//
//  ROWS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2 >
struct Rows< SVecTDVecMultExpr<VT1,VT2> > : public Size<VT1>
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
template< typename VT1, typename VT2 >
struct Columns< SVecTDVecMultExpr<VT1,VT2> > : public Size<VT2>
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
template< typename VT1, typename VT2, bool AF >
struct SubmatrixExprTrait< SVecTDVecMultExpr<VT1,VT2>, AF >
{
 public:
   //**********************************************************************************************
   using Type = MultExprTrait_< SubvectorExprTrait_<const VT1,AF>
                              , SubvectorExprTrait_<const VT2,AF> >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2 >
struct RowExprTrait< SVecTDVecMultExpr<VT1,VT2> >
{
 public:
   //**********************************************************************************************
   using Type = MultExprTrait_< ReturnType_<VT1>, VT2 >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2 >
struct ColumnExprTrait< SVecTDVecMultExpr<VT1,VT2> >
{
 public:
   //**********************************************************************************************
   using Type = MultExprTrait_< VT1, ReturnType_<VT2> >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
