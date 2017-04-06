//=================================================================================================
/*!
//  \file blaze/math/expressions/SMatDMatMultExpr.h
//  \brief Header file for the sparse matrix/dense matrix multiplication expression
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

#ifndef _BLAZE_MATH_EXPRESSIONS_SMATDMATMULTEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_SMATDMATMULTEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/ColumnMajorMatrix.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/MatMatMultExpr.h>
#include <blaze/math/constraints/RowMajorMatrix.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/constraints/StorageOrder.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/MatMatMultExpr.h>
#include <blaze/math/Functions.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/traits/ColumnExprTrait.h>
#include <blaze/math/traits/DMatDVecMultExprTrait.h>
#include <blaze/math/traits/DMatSVecMultExprTrait.h>
#include <blaze/math/traits/MultExprTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/RowExprTrait.h>
#include <blaze/math/traits/SMatDVecMultExprTrait.h>
#include <blaze/math/traits/SubmatrixExprTrait.h>
#include <blaze/math/traits/TDVecDMatMultExprTrait.h>
#include <blaze/math/traits/TDVecSMatMultExprTrait.h>
#include <blaze/math/traits/TSVecDMatMultExprTrait.h>
#include <blaze/math/traits/TSVecSMatMultExprTrait.h>
#include <blaze/math/typetraits/AreSIMDCombinable.h>
#include <blaze/math/typetraits/Columns.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsColumnVector.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsDiagonal.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsRowVector.h>
#include <blaze/math/typetraits/IsSparseMatrix.h>
#include <blaze/math/typetraits/IsSparseVector.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsTriangular.h>
#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/math/typetraits/IsUniUpper.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/math/typetraits/Rows.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/InvalidType.h>
#include <blaze/util/logging/FunctionTrace.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/RemoveReference.h>


namespace blaze {

//=================================================================================================
//
//  CLASS SMATDMATMULTEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for sparse matrix-dense matrix multiplications.
// \ingroup dense_matrix_expression
//
// The SMatDMatMultExpr class represents the compile time expression for multiplications between
// a row-major sparse matrix and a row-major dense matrix.
*/
template< typename MT1    // Type of the left-hand side sparse matrix
        , typename MT2 >  // Type of the right-hand side dense matrix
class SMatDMatMultExpr : public DenseMatrix< SMatDMatMultExpr<MT1,MT2>, false >
                       , private MatMatMultExpr
                       , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef ResultType_<MT1>     RT1;  //!< Result type of the left-hand side sparse matrix expression.
   typedef ResultType_<MT2>     RT2;  //!< Result type of the right-hand side dense matrix expression.
   typedef ElementType_<RT1>    ET1;  //!< Element type of the left-hand side sparse matrix expression.
   typedef ElementType_<RT2>    ET2;  //!< Element type of the right-hand side dense matrix expression.
   typedef CompositeType_<MT1>  CT1;  //!< Composite type of the left-hand side sparse matrix expression.
   typedef CompositeType_<MT2>  CT2;  //!< Composite type of the right-hand side dense matrix expression.
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the left-hand side sparse matrix expression.
   enum : bool { evaluateLeft = IsComputation<MT1>::value || RequiresEvaluation<MT1>::value };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the right-hand side dense matrix expression.
   enum : bool { evaluateRight = IsComputation<MT2>::value || RequiresEvaluation<MT2>::value };
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! The IsEvaluationRequired struct is a helper struct for the selection of the parallel
       evaluation strategy. In case either of the two matrix operands requires an intermediate
       evaluation, the nested \value will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   struct IsEvaluationRequired {
      enum : bool { value = ( evaluateLeft || evaluateRight ) };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! In case all three involved data types are suited for a vectorized computation of the
       matrix multiplication, the nested \value will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   struct UseVectorizedKernel {
      enum : bool { value = useOptimizedKernels &&
                            !IsDiagonal<T3>::value &&
                            T1::simdEnabled && T3::simdEnabled &&
                            IsRowMajorMatrix<T1>::value &&
                            AreSIMDCombinable< ElementType_<T1>
                                             , ElementType_<T2>
                                             , ElementType_<T3> >::value &&
                            HasSIMDAdd< ElementType_<T2>, ElementType_<T3> >::value &&
                            HasSIMDMult< ElementType_<T2>, ElementType_<T3> >::value };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! In case a vectorized computation of the matrix multiplication is not possible, but a
       loop-unrolled computation is feasible, the nested \value will be set to 1, otherwise
       it will be 0. */
   template< typename T1, typename T2, typename T3 >
   struct UseOptimizedKernel {
      enum : bool { value = useOptimizedKernels &&
                            !UseVectorizedKernel<T1,T2,T3>::value &&
                            !IsDiagonal<T3>::value &&
                            !IsResizable< ElementType_<T1> >::value &&
                            !IsResizable<ET1>::value };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! In case neither a vectorized nor optimized computation is possible, the nested \value will
       be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   struct UseDefaultKernel {
      enum : bool { value = !UseVectorizedKernel<T1,T2,T3>::value &&
                            !UseOptimizedKernel<T1,T2,T3>::value };
   };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef SMatDMatMultExpr<MT1,MT2>   This;           //!< Type of this SMatDMatMultExpr instance.
   typedef MultTrait_<RT1,RT2>         ResultType;     //!< Result type for expression template evaluations.
   typedef OppositeType_<ResultType>   OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef TransposeType_<ResultType>  TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<ResultType>    ElementType;    //!< Resulting element type.
   typedef SIMDTrait_<ElementType>     SIMDType;       //!< Resulting SIMD element type.
   typedef const ElementType           ReturnType;     //!< Return type for expression template evaluations.
   typedef const ResultType            CompositeType;  //!< Data type for composite expression templates.

   //! Composite type of the left-hand side sparse matrix expression.
   typedef If_< IsExpression<MT1>, const MT1, const MT1& >  LeftOperand;

   //! Composite type of the right-hand side dense matrix expression.
   typedef If_< IsExpression<MT2>, const MT2, const MT2& >  RightOperand;

   //! Type for the assignment of the left-hand side sparse matrix operand.
   typedef IfTrue_< evaluateLeft, const RT1, CT1 >  LT;

   //! Type for the assignment of the right-hand side dense matrix operand.
   typedef IfTrue_< evaluateRight, const RT2, CT2 >  RT;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = !IsDiagonal<MT2>::value &&
                               MT2::simdEnabled &&
                               HasSIMDAdd<ET1,ET2>::value &&
                               HasSIMDMult<ET1,ET2>::value };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = !evaluateLeft  && MT1::smpAssignable &&
                                 !evaluateRight && MT2::smpAssignable };
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   enum : size_t { SIMDSIZE = SIMDTrait<ElementType>::size };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SMatDMatMultExpr class.
   //
   // \param lhs The left-hand side sparse matrix operand of the multiplication expression.
   // \param rhs The right-hand side dense matrix operand of the multiplication expression.
   */
   explicit inline SMatDMatMultExpr( const MT1& lhs, const MT2& rhs ) noexcept
      : lhs_( lhs )  // Left-hand side sparse matrix of the multiplication expression
      , rhs_( rhs )  // Right-hand side dense matrix of the multiplication expression
   {
      BLAZE_INTERNAL_ASSERT( lhs.columns() == rhs.rows(), "Invalid matrix sizes" );
   }
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 2D-access to the matrix elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator()( size_t i, size_t j ) const {
      BLAZE_INTERNAL_ASSERT( i < lhs_.rows()   , "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < rhs_.columns(), "Invalid column access index" );

      if( IsDiagonal<MT1>::value ) {
         return lhs_(i,i) * rhs_(i,j);
      }
      else if( IsDiagonal<MT2>::value ) {
         return lhs_(i,j) * rhs_(j,j);
      }
      else if( IsTriangular<MT1>::value || IsTriangular<MT2>::value ) {
         const size_t begin( ( IsUpper<MT1>::value )
                             ?( ( IsLower<MT2>::value )
                                ?( max( ( IsStrictlyUpper<MT1>::value ? i+1UL : i )
                                      , ( IsStrictlyLower<MT2>::value ? j+1UL : j ) ) )
                                :( IsStrictlyUpper<MT1>::value ? i+1UL : i ) )
                             :( ( IsLower<MT2>::value )
                                ?( IsStrictlyLower<MT2>::value ? j+1UL : j )
                                :( 0UL ) ) );
         const size_t end( ( IsLower<MT1>::value )
                           ?( ( IsUpper<MT2>::value )
                              ?( min( ( IsStrictlyLower<MT1>::value ? i : i+1UL )
                                    , ( IsStrictlyUpper<MT2>::value ? j : j+1UL ) ) )
                              :( IsStrictlyLower<MT1>::value ? i : i+1UL ) )
                           :( ( IsUpper<MT2>::value )
                              ?( IsStrictlyUpper<MT2>::value ? j : j+1UL )
                              :( lhs_.columns() ) ) );

         if( begin >= end ) return ElementType();

         const size_t n( end - begin );

         return subvector( row( lhs_, i ), begin, n ) * subvector( column( rhs_, j ), begin, n );
      }
      else {
         return row( lhs_, i ) * column( rhs_, j );
      }
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
      if( i >= lhs_.rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( j >= rhs_.columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
      }
      return (*this)(i,j);
   }
   //**********************************************************************************************

   //**Rows function*******************************************************************************
   /*!\brief Returns the current number of rows of the matrix.
   //
   // \return The number of rows of the matrix.
   */
   inline size_t rows() const noexcept {
      return lhs_.rows();
   }
   //**********************************************************************************************

   //**Columns function****************************************************************************
   /*!\brief Returns the current number of columns of the matrix.
   //
   // \return The number of columns of the matrix.
   */
   inline size_t columns() const noexcept {
      return rhs_.columns();
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side sparse matrix operand.
   //
   // \return The left-hand side sparse matrix operand.
   */
   inline LeftOperand leftOperand() const noexcept {
      return lhs_;
   }
   //**********************************************************************************************

   //**Right operand access************************************************************************
   /*!\brief Returns the right-hand side dense matrix operand.
   //
   // \return The right-hand side dense matrix operand.
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
      return ( lhs_.isAliased( alias ) || rhs_.isAliased( alias ) );
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

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const noexcept {
      return rhs_.isAligned();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return ( rows() > SMP_SMATDMATMULT_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  lhs_;  //!< Left-hand side sparse matrix of the multiplication expression.
   RightOperand rhs_;  //!< Right-hand side dense matrix of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense matrices****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse matrix-dense matrix multiplication to a dense matrix
   //        (\f$ A=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse matrix-dense
   // matrix multiplication expression to a dense matrix.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline void assign( DenseMatrix<MT,SO>& lhs, const SMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LT A( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse matrix operand
      RT B( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      SMatDMatMultExpr::selectAssignKernel( ~lhs, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense matrices********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a sparse matrix-dense matrix multiplication to dense matrices
   //        (\f$ A=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side sparse matrix operand.
   // \param B The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the default assignment kernel for the sparse matrix-dense matrix
   // multiplication.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseDefaultKernel<MT3,MT4,MT5> >
      selectAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      typedef ConstIterator_<MT4>  ConstIterator;

      const size_t block( Or< IsRowMajorMatrix<MT3>, IsDiagonal<MT5> >::value ? B.columns() : 64UL );

      reset( C );

      for( size_t jj=0UL; jj<B.columns(); jj+=block )
      {
         const size_t jtmp( min( jj+block, B.columns() ) );

         for( size_t i=0UL; i<A.rows(); ++i )
         {
            ConstIterator element( A.begin(i) );
            const ConstIterator end( A.end(i) );

            for( ; element!=end; ++element )
            {
               const size_t i1( element->index() );

               if( IsDiagonal<MT5>::value )
               {
                  C(i,i1) = element->value() * B(i1,i1);
               }
               else
               {
                  const size_t jbegin( ( IsUpper<MT5>::value )
                                       ?( max( jj, ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) ) )
                                       :( jj ) );
                  const size_t jend( ( IsLower<MT5>::value )
                                     ?( min( jtmp, ( IsStrictlyLower<MT5>::value ? i1 : i1+1UL ) ) )
                                     :( jtmp ) );

                  if( IsTriangular<MT5>::value && jbegin >= jend )
                     continue;

                  BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

                  for( size_t j=jbegin; j<jend; ++j ) {
                     if( isDefault( C(i,j) ) )
                        C(i,j) = element->value() * B(i1,j);
                     else
                        C(i,j) += element->value() * B(i1,j);
                  }
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Optimized assignment to dense matrices******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Optimized assignment of a sparse matrix-dense matrix multiplication to dense matrices
   //        (\f$ A=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side sparse matrix operand.
   // \param B The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the optimized assignment kernel for the sparse matrix-dense matrix
   // multiplication.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseOptimizedKernel<MT3,MT4,MT5> >
      selectAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      typedef ConstIterator_<MT4>  ConstIterator;

      const size_t block( IsRowMajorMatrix<MT3>::value ? B.columns() : 64UL );

      reset( C );

      for( size_t jj=0UL; jj<B.columns(); jj+=block )
      {
         const size_t jtmp( min( jj+block, B.columns() ) );

         for( size_t i=0UL; i<A.rows(); ++i )
         {
            const ConstIterator end( A.end(i) );
            ConstIterator element( A.begin(i) );

            const size_t nonzeros( A.nonZeros(i) );
            const size_t kpos( nonzeros & size_t(-4) );
            BLAZE_INTERNAL_ASSERT( ( nonzeros - ( nonzeros % 4UL ) ) == kpos, "Invalid end calculation" );

            for( size_t k=0UL; k<kpos; k+=4UL )
            {
               const size_t i1( element->index() );
               const ET1    v1( element->value() );
               ++element;
               const size_t i2( element->index() );
               const ET1    v2( element->value() );
               ++element;
               const size_t i3( element->index() );
               const ET1    v3( element->value() );
               ++element;
               const size_t i4( element->index() );
               const ET1    v4( element->value() );
               ++element;

               BLAZE_INTERNAL_ASSERT( i1 < i2 && i2 < i3 && i3 < i4, "Invalid sparse matrix index detected" );

               const size_t jbegin( ( IsUpper<MT5>::value )
                                    ?( max( jj, ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) ) )
                                    :( jj ) );
               const size_t jend( ( IsLower<MT5>::value )
                                  ?( min( jtmp, ( IsStrictlyLower<MT5>::value ? i4 : i4+1UL ) ) )
                                  :( jtmp ) );

               if( IsTriangular<MT5>::value && jbegin >= jend )
                  continue;

               BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

               const size_t jnum( jend - jbegin );
               const size_t jpos( jbegin + ( jnum & size_t(-4) ) );
               BLAZE_INTERNAL_ASSERT( ( jbegin + jnum - ( jnum % 4UL ) ) == jpos, "Invalid end calculation" );

               for( size_t j=jbegin; j<jpos; j+=4UL ) {
                  C(i,j    ) += v1 * B(i1,j    ) + v2 * B(i2,j    ) + v3 * B(i3,j    ) + v4 * B(i4,j    );
                  C(i,j+1UL) += v1 * B(i1,j+1UL) + v2 * B(i2,j+1UL) + v3 * B(i3,j+1UL) + v4 * B(i4,j+1UL);
                  C(i,j+2UL) += v1 * B(i1,j+2UL) + v2 * B(i2,j+2UL) + v3 * B(i3,j+2UL) + v4 * B(i4,j+2UL);
                  C(i,j+3UL) += v1 * B(i1,j+3UL) + v2 * B(i2,j+3UL) + v3 * B(i3,j+3UL) + v4 * B(i4,j+3UL);
               }
               for( size_t j=jpos; j<jend; ++j ) {
                  C(i,j) += v1 * B(i1,j) + v2 * B(i2,j) + v3 * B(i3,j) + v4 * B(i4,j);
               }
            }

            for( ; element!=end; ++element )
            {
               const size_t i1( element->index() );
               const ET1    v1( element->value() );

               const size_t jbegin( ( IsUpper<MT5>::value )
                                    ?( max( jj, ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) ) )
                                    :( jj ) );
               const size_t jend( ( IsLower<MT5>::value )
                                  ?( min( jtmp, ( IsStrictlyLower<MT5>::value ? i1 : i1+1UL ) ) )
                                  :( jtmp ) );

               if( IsTriangular<MT5>::value && jbegin >= jend )
                  continue;

               BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

               const size_t jnum( jend - jbegin );
               const size_t jpos( jbegin + ( jnum & size_t(-4) ) );
               BLAZE_INTERNAL_ASSERT( ( jbegin + jnum - ( jnum % 4UL ) ) == jpos, "Invalid end calculation" );

               for( size_t j=jbegin; j<jpos; j+=4UL ) {
                  C(i,j    ) += v1 * B(i1,j    );
                  C(i,j+1UL) += v1 * B(i1,j+1UL);
                  C(i,j+2UL) += v1 * B(i1,j+2UL);
                  C(i,j+3UL) += v1 * B(i1,j+3UL);
               }
               for( size_t j=jpos; j<jend; ++j ) {
                  C(i,j) += v1 * B(i1,j);
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized assignment to dense matrices*****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized assignment of a sparse matrix-dense matrix multiplication to dense matrices
   //        (\f$ A=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side sparse matrix operand.
   // \param B The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the vectorized assignment kernel for the sparse matrix-dense matrix
   // multiplication.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedKernel<MT3,MT4,MT5> >
      selectAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      typedef ConstIterator_<MT4>  ConstIterator;

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      reset( C );

      for( size_t i=0UL; i<A.rows(); ++i )
      {
         const ConstIterator end( A.end(i) );
         ConstIterator element( A.begin(i) );

         const size_t nonzeros( A.nonZeros(i) );
         const size_t kpos( nonzeros & size_t(-4) );
         BLAZE_INTERNAL_ASSERT( ( nonzeros - ( nonzeros % 4UL ) ) == kpos, "Invalid end calculation" );

         for( size_t k=0UL; k<kpos; k+=4UL )
         {
            const size_t i1( element->index() );
            const ET1    v1( element->value() );
            ++element;
            const size_t i2( element->index() );
            const ET1    v2( element->value() );
            ++element;
            const size_t i3( element->index() );
            const ET1    v3( element->value() );
            ++element;
            const size_t i4( element->index() );
            const ET1    v4( element->value() );
            ++element;

            BLAZE_INTERNAL_ASSERT( i1 < i2 && i2 < i3 && i3 < i4, "Invalid sparse matrix index detected" );

            const SIMDType xmm1( set( v1 ) );
            const SIMDType xmm2( set( v2 ) );
            const SIMDType xmm3( set( v3 ) );
            const SIMDType xmm4( set( v4 ) );

            const size_t jbegin( ( IsUpper<MT5>::value )
                                 ?( ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) & size_t(-SIMDSIZE) )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT5>::value )
                               ?( IsStrictlyLower<MT5>::value ? i4 : i4+1UL )
                               :( B.columns() ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
            BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

            size_t j( jbegin );

            for( ; j<jpos; j+=SIMDSIZE ) {
               C.store( i, j, C.load(i,j) + xmm1 * B.load(i1,j) + xmm2 * B.load(i2,j) + xmm3 * B.load(i3,j) + xmm4 * B.load(i4,j) );
            }
            for( ; remainder && j<jend; ++j ) {
               C(i,j) += v1 * B(i1,j) + v2 * B(i2,j) + v3 * B(i3,j) + v4 * B(i4,j);
            }
         }

         for( ; element!=end; ++element )
         {
            const size_t i1( element->index() );
            const ET1    v1( element->value() );

            const SIMDType xmm1( set( v1 ) );

            const size_t jbegin( ( IsUpper<MT5>::value )
                                 ?( ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) & size_t(-SIMDSIZE) )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT5>::value )
                               ?( IsStrictlyLower<MT5>::value ? i1 : i1+1UL )
                               :( B.columns() ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
            BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

            size_t j( jbegin );

            for( ; j<jpos; j+=SIMDSIZE ) {
               C.store( i, j, C.load(i,j) + xmm1 * B.load(i1,j) );
            }
            for( ; remainder && j<jend; ++j ) {
               C(i,j) += v1 * B(i1,j);
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to sparse matrices***************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse matrix-dense matrix multiplication to a sparse matrix
   //        (\f$ A=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side sparse matrix.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse matrix-dense
   // matrix multiplication expression to a sparse matrix.
   */
   template< typename MT  // Type of the target sparse matrix
           , bool SO >    // Storage order of the target sparse matrix
   friend inline void assign( SparseMatrix<MT,SO>& lhs, const SMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      typedef IfTrue_< SO, OppositeType, ResultType >  TmpType;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( OppositeType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( OppositeType );
      BLAZE_CONSTRAINT_MATRICES_MUST_HAVE_SAME_STORAGE_ORDER( MT, TmpType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<TmpType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const TmpType tmp( serial( rhs ) );
      assign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense matrices*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a sparse matrix-dense matrix multiplication to a dense matrix
   //        (\f$ A+=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a sparse matrix-
   // dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline void addAssign( DenseMatrix<MT,SO>& lhs, const SMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LT A( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse matrix operand
      RT B( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      SMatDMatMultExpr::selectAddAssignKernel( ~lhs, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense matrices***********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a sparse matrix-dense matrix multiplication to
   //        dense matrices (\f$ A+=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side sparse matrix operand.
   // \param B The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the default addition assignment kernel for the sparse matrix-dense
   // matrix multiplication.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseDefaultKernel<MT3,MT4,MT5> >
      selectAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      typedef ConstIterator_<MT4>  ConstIterator;

      const size_t block( Or< IsRowMajorMatrix<MT3>, IsDiagonal<MT5> >::value ? B.columns() : 64UL );

      for( size_t jj=0UL; jj<B.columns(); jj+=block )
      {
         const size_t jtmp( min( jj+block, B.columns() ) );

         for( size_t i=0UL; i<A.rows(); ++i )
         {
            const ConstIterator end( A.end(i) );
            ConstIterator element( A.begin(i) );

            for( ; element!=end; ++element )
            {
               const size_t i1( element->index() );

               if( IsDiagonal<MT5>::value )
               {
                  C(i,i1) += element->value() * B(i1,i1);
               }
               else
               {
                  const size_t jbegin( ( IsUpper<MT5>::value )
                                       ?( max( jj, ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) ) )
                                       :( jj ) );
                  const size_t jend( ( IsLower<MT5>::value )
                                     ?( min( jtmp, ( IsStrictlyLower<MT5>::value ? i1 : i1+1UL ) ) )
                                     :( jtmp ) );

                  if( IsTriangular<MT5>::value && jbegin >= jend )
                     continue;

                  BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

                  const size_t jnum( jend - jbegin );
                  const size_t jpos( jbegin + ( jnum & size_t(-4) ) );
                  BLAZE_INTERNAL_ASSERT( ( jbegin + jnum - ( jnum % 4UL ) ) == jpos, "Invalid end calculation" );

                  for( size_t j=jbegin; j<jpos; j+=4UL ) {
                     C(i,j    ) += element->value() * B(i1,j    );
                     C(i,j+1UL) += element->value() * B(i1,j+1UL);
                     C(i,j+2UL) += element->value() * B(i1,j+2UL);
                     C(i,j+3UL) += element->value() * B(i1,j+3UL);
                  }
                  for( size_t j=jpos; j<jend; ++j ) {
                     C(i,j) += element->value() * B(i1,j);
                  }
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Optimized addition assignment to dense matrices*********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Optimized addition assignment of a sparse matrix-dense matrix multiplication to
   //        dense matrices (\f$ A+=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side sparse matrix operand.
   // \param B The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the optimized addition assignment kernel for the sparse matrix-
   // dense matrix multiplication.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseOptimizedKernel<MT3,MT4,MT5> >
      selectAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      typedef ConstIterator_<MT4>  ConstIterator;

      const size_t block( IsRowMajorMatrix<MT3>::value ? B.columns() : 64UL );

      for( size_t jj=0UL; jj<B.columns(); jj+=block )
      {
         const size_t jtmp( min( jj+block, B.columns() ) );

         for( size_t i=0UL; i<A.rows(); ++i )
         {
            const ConstIterator end( A.end(i) );
            ConstIterator element( A.begin(i) );

            const size_t nonzeros( A.nonZeros(i) );
            const size_t kpos( nonzeros & size_t(-4) );
            BLAZE_INTERNAL_ASSERT( ( nonzeros - ( nonzeros % 4UL ) ) == kpos, "Invalid end calculation" );

            for( size_t k=0UL; k<kpos; k+=4UL )
            {
               const size_t i1( element->index() );
               const ET1    v1( element->value() );
               ++element;
               const size_t i2( element->index() );
               const ET1    v2( element->value() );
               ++element;
               const size_t i3( element->index() );
               const ET1    v3( element->value() );
               ++element;
               const size_t i4( element->index() );
               const ET1    v4( element->value() );
               ++element;

               BLAZE_INTERNAL_ASSERT( i1 < i2 && i2 < i3 && i3 < i4, "Invalid sparse matrix index detected" );

               const size_t jbegin( ( IsUpper<MT5>::value )
                                    ?( max( jj, ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) ) )
                                    :( jj ) );
               const size_t jend( ( IsLower<MT5>::value )
                                  ?( min( jtmp, ( IsStrictlyLower<MT5>::value ? i4 : i4+1UL ) ) )
                                  :( jtmp ) );

               if( IsTriangular<MT5>::value && jbegin >= jend )
                  continue;

               BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

               const size_t jnum( jend - jbegin );
               const size_t jpos( jbegin + ( jnum & size_t(-4) ) );
               BLAZE_INTERNAL_ASSERT( ( jbegin + jnum - ( jnum % 4UL ) ) == jpos, "Invalid end calculation" );

               for( size_t j=jbegin; j<jpos; j+=4UL ) {
                  C(i,j    ) += v1 * B(i1,j    ) + v2 * B(i2,j    ) + v3 * B(i3,j    ) + v4 * B(i4,j    );
                  C(i,j+1UL) += v1 * B(i1,j+1UL) + v2 * B(i2,j+1UL) + v3 * B(i3,j+1UL) + v4 * B(i4,j+1UL);
                  C(i,j+2UL) += v1 * B(i1,j+2UL) + v2 * B(i2,j+2UL) + v3 * B(i3,j+2UL) + v4 * B(i4,j+2UL);
                  C(i,j+3UL) += v1 * B(i1,j+3UL) + v2 * B(i2,j+3UL) + v3 * B(i3,j+3UL) + v4 * B(i4,j+3UL);
               }
               for( size_t j=jpos; j<jend; ++j ) {
                  C(i,j) += v1 * B(i1,j) + v2 * B(i2,j) + v3 * B(i3,j) + v4 * B(i4,j);
               }
            }

            for( ; element!=end; ++element )
            {
               const size_t i1( element->index() );
               const ET1    v1( element->value() );

               const size_t jbegin( ( IsUpper<MT5>::value )
                                    ?( max( jj, ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) ) )
                                    :( jj ) );
               const size_t jend( ( IsLower<MT5>::value )
                                  ?( min( jtmp, ( IsStrictlyLower<MT5>::value ? i1 : i1+1UL ) ) )
                                  :( jtmp ) );

               if( IsTriangular<MT5>::value && jbegin >= jend )
                  continue;

               BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

               const size_t jnum( jend - jbegin );
               const size_t jpos( jbegin + ( jnum & size_t(-4) ) );
               BLAZE_INTERNAL_ASSERT( ( jbegin + jnum - ( jnum % 4UL ) ) == jpos, "Invalid end calculation" );

               for( size_t j=jbegin; j<jpos; j+=4UL ) {
                  C(i,j    ) += v1 * B(i1,j    );
                  C(i,j+1UL) += v1 * B(i1,j+1UL);
                  C(i,j+2UL) += v1 * B(i1,j+2UL);
                  C(i,j+3UL) += v1 * B(i1,j+3UL);
               }
               for( size_t j=jpos; j<jend; ++j ) {
                  C(i,j) += v1 * B(i1,j);
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized addition assignment to dense matrices********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized addition assignment of a sparse matrix-dense matrix multiplication to
   //        dense matrices (\f$ A+=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side sparse matrix operand.
   // \param B The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the vectorized addition assignment kernel for the sparse matrix-
   // dense matrix multiplication.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedKernel<MT3,MT4,MT5> >
      selectAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      typedef ConstIterator_<MT4>  ConstIterator;

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      for( size_t i=0UL; i<A.rows(); ++i )
      {
         const ConstIterator end( A.end(i) );
         ConstIterator element( A.begin(i) );

         const size_t nonzeros( A.nonZeros(i) );
         const size_t kpos( nonzeros & size_t(-4) );
         BLAZE_INTERNAL_ASSERT( ( nonzeros - ( nonzeros % 4UL ) ) == kpos, "Invalid end calculation" );

         for( size_t k=0UL; k<kpos; k+=4UL )
         {
            const size_t i1( element->index() );
            const ET1    v1( element->value() );
            ++element;
            const size_t i2( element->index() );
            const ET1    v2( element->value() );
            ++element;
            const size_t i3( element->index() );
            const ET1    v3( element->value() );
            ++element;
            const size_t i4( element->index() );
            const ET1    v4( element->value() );
            ++element;

            BLAZE_INTERNAL_ASSERT( i1 < i2 && i2 < i3 && i3 < i4, "Invalid sparse matrix index detected" );

            const SIMDType xmm1( set( v1 ) );
            const SIMDType xmm2( set( v2 ) );
            const SIMDType xmm3( set( v3 ) );
            const SIMDType xmm4( set( v4 ) );

            const size_t jbegin( ( IsUpper<MT5>::value )
                                 ?( ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) & size_t(-SIMDSIZE) )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT5>::value )
                               ?( IsStrictlyLower<MT5>::value ? i4 : i4+1UL )
                               :( B.columns() ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
            BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

            size_t j( jbegin );

            for( ; j<jpos; j+=SIMDSIZE ) {
               C.store( i, j, C.load(i,j) + xmm1 * B.load(i1,j) + xmm2 * B.load(i2,j) + xmm3 * B.load(i3,j) + xmm4 * B.load(i4,j) );
            }
            for( ; remainder && j<jend; ++j ) {
               C(i,j) += v1 * B(i1,j) + v2 * B(i2,j) + v3 * B(i3,j) + v4 * B(i4,j);
            }
         }

         for( ; element!=end; ++element )
         {
            const size_t i1( element->index() );
            const ET1    v1( element->value() );

            const SIMDType xmm1( set( v1 ) );

            const size_t jbegin( ( IsUpper<MT5>::value )
                                 ?( ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) & size_t(-SIMDSIZE) )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT5>::value )
                               ?( IsStrictlyLower<MT5>::value ? i1 : i1+1UL )
                               :( B.columns() ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
            BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

            size_t j( jbegin );

            for( ; j<jpos; j+=SIMDSIZE ) {
               C.store( i, j, C.load(i,j) + xmm1 * B.load(i1,j) );
            }
            for( ; remainder && j<jend; ++j ) {
               C(i,j) += v1 * B(i1,j);
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to sparse matrices******************************************************
   // No special implementation for the addition assignment to sparse matrices.
   //**********************************************************************************************

   //**Subtraction assignment to dense matrices****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a sparse matrix-dense matrix multiplication to a dense
   //        matrix (\f$ A-=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a sparse matrix-
   // dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline void subAssign( DenseMatrix<MT,SO>& lhs, const SMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LT A( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse matrix operand
      RT B( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      SMatDMatMultExpr::selectSubAssignKernel( ~lhs, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense matrices********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a sparse matrix-dense matrix multiplication
   //        (\f$ A-=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side sparse matrix operand.
   // \param B The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the default subtraction assignment kernel for the sparse matrix-
   // dense matrix multiplication.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseDefaultKernel<MT3,MT4,MT5> >
      selectSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      typedef ConstIterator_<MT4>  ConstIterator;

      const size_t block( Or< IsRowMajorMatrix<MT3>, IsDiagonal<MT5> >::value ? B.columns() : 64UL );

      for( size_t jj=0UL; jj<B.columns(); jj+=block )
      {
         const size_t jtmp( min( jj+block, B.columns() ) );

         for( size_t i=0UL; i<A.rows(); ++i )
         {
            const ConstIterator end( A.end(i) );
            ConstIterator element( A.begin(i) );

            for( ; element!=end; ++element )
            {
               const size_t i1( element->index() );

               if( IsDiagonal<MT5>::value )
               {
                  C(i,i1) -= element->value() * B(i1,i1);
               }
               else
               {
                  const size_t jbegin( ( IsUpper<MT5>::value )
                                       ?( max( jj, ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) ) )
                                       :( jj ) );
                  const size_t jend( ( IsLower<MT5>::value )
                                     ?( min( jtmp, ( IsStrictlyLower<MT5>::value ? i1 : i1+1UL ) ) )
                                     :( jtmp ) );

                  if( IsTriangular<MT5>::value && jbegin >= jend )
                     continue;

                  BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

                  const size_t jnum( jend - jbegin );
                  const size_t jpos( jbegin + ( jnum & size_t(-4) ) );
                  BLAZE_INTERNAL_ASSERT( ( jbegin + jnum - ( jnum % 4UL ) ) == jpos, "Invalid end calculation" );

                  for( size_t j=jbegin; j<jpos; j+=4UL ) {
                     C(i,j    ) -= element->value() * B(i1,j    );
                     C(i,j+1UL) -= element->value() * B(i1,j+1UL);
                     C(i,j+2UL) -= element->value() * B(i1,j+2UL);
                     C(i,j+3UL) -= element->value() * B(i1,j+3UL);
                  }
                  for( size_t j=jpos; j<jend; ++j ) {
                     C(i,j) -= element->value() * B(i1,j);
                  }
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Optimized subtraction assignment to dense matrices******************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Optimized subtraction assignment of a sparse matrix-dense matrix multiplication
   //        (\f$ A-=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side sparse matrix operand.
   // \param B The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the optimized subtraction assignment kernel for the sparse matrix-
   // dense matrix multiplication.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseOptimizedKernel<MT3,MT4,MT5> >
      selectSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      typedef ConstIterator_<MT4>  ConstIterator;

      const size_t block( IsRowMajorMatrix<MT3>::value ? B.columns() : 64UL );

      for( size_t jj=0UL; jj<B.columns(); jj+=block )
      {
         const size_t jtmp( min( jj+block, B.columns() ) );

         for( size_t i=0UL; i<A.rows(); ++i )
         {
            const ConstIterator end( A.end(i) );
            ConstIterator element( A.begin(i) );

            const size_t nonzeros( A.nonZeros(i) );
            const size_t kpos( nonzeros & size_t(-4) );
            BLAZE_INTERNAL_ASSERT( ( nonzeros - ( nonzeros % 4UL ) ) == kpos, "Invalid end calculation" );

            for( size_t k=0UL; k<kpos; k+=4UL )
            {
               const size_t i1( element->index() );
               const ET1    v1( element->value() );
               ++element;
               const size_t i2( element->index() );
               const ET1    v2( element->value() );
               ++element;
               const size_t i3( element->index() );
               const ET1    v3( element->value() );
               ++element;
               const size_t i4( element->index() );
               const ET1    v4( element->value() );
               ++element;

               BLAZE_INTERNAL_ASSERT( i1 < i2 && i2 < i3 && i3 < i4, "Invalid sparse matrix index detected" );

               const size_t jbegin( ( IsUpper<MT5>::value )
                                    ?( max( jj, ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) ) )
                                    :( jj ) );
               const size_t jend( ( IsLower<MT5>::value )
                                  ?( min( jtmp, ( IsStrictlyLower<MT5>::value ? i4 : i4+1UL ) ) )
                                  :( jtmp ) );

               if( IsTriangular<MT5>::value && jbegin >= jend )
                  continue;

               BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

               const size_t jnum( jend - jbegin );
               const size_t jpos( jbegin + ( jnum & size_t(-4) ) );
               BLAZE_INTERNAL_ASSERT( ( jbegin + jnum - ( jnum % 4UL ) ) == jpos, "Invalid end calculation" );

               for( size_t j=jbegin; j<jpos; j+=4UL ) {
                  C(i,j    ) -= v1 * B(i1,j    ) + v2 * B(i2,j    ) + v3 * B(i3,j    ) + v4 * B(i4,j    );
                  C(i,j+1UL) -= v1 * B(i1,j+1UL) + v2 * B(i2,j+1UL) + v3 * B(i3,j+1UL) + v4 * B(i4,j+1UL);
                  C(i,j+2UL) -= v1 * B(i1,j+2UL) + v2 * B(i2,j+2UL) + v3 * B(i3,j+2UL) + v4 * B(i4,j+2UL);
                  C(i,j+3UL) -= v1 * B(i1,j+3UL) + v2 * B(i2,j+3UL) + v3 * B(i3,j+3UL) + v4 * B(i4,j+3UL);
               }
               for( size_t j=jpos; j<jend; ++j ) {
                  C(i,j) -= v1 * B(i1,j) + v2 * B(i2,j) + v3 * B(i3,j) + v4 * B(i4,j);
               }
            }

            for( ; element!=end; ++element )
            {
               const size_t i1( element->index() );
               const ET1    v1( element->value() );

               const size_t jbegin( ( IsUpper<MT5>::value )
                                    ?( max( jj, ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) ) )
                                    :( jj ) );
               const size_t jend( ( IsLower<MT5>::value )
                                  ?( min( jtmp, ( IsStrictlyLower<MT5>::value ? i1 : i1+1UL ) ) )
                                  :( jtmp ) );

               if( IsTriangular<MT5>::value && jbegin >= jend )
                  continue;

               BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

               const size_t jnum( jend - jbegin );
               const size_t jpos( jbegin + ( jnum & size_t(-4) ) );
               BLAZE_INTERNAL_ASSERT( ( jbegin + jnum - ( jnum % 4UL ) ) == jpos, "Invalid end calculation" );

               for( size_t j=jbegin; j<jpos; j+=4UL ) {
                  C(i,j    ) -= v1 * B(i1,j    );
                  C(i,j+1UL) -= v1 * B(i1,j+1UL);
                  C(i,j+2UL) -= v1 * B(i1,j+2UL);
                  C(i,j+3UL) -= v1 * B(i1,j+3UL);
               }
               for( size_t j=jpos; j<jend; ++j ) {
                  C(i,j) -= v1 * B(i1,j);
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized subtraction assignment to dense matrices*****************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized subtraction assignment of a sparse matrix-dense matrix multiplication
   //        (\f$ A-=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side sparse matrix operand.
   // \param B The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the vectorized subtraction assignment kernel for the sparse matrix-
   // dense matrix multiplication.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedKernel<MT3,MT4,MT5> >
      selectSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      typedef ConstIterator_<MT4>  ConstIterator;

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      for( size_t i=0UL; i<A.rows(); ++i )
      {
         const ConstIterator end( A.end(i) );
         ConstIterator element( A.begin(i) );

         const size_t nonzeros( A.nonZeros(i) );
         const size_t kpos( nonzeros & size_t(-4) );
         BLAZE_INTERNAL_ASSERT( ( nonzeros - ( nonzeros % 4UL ) ) == kpos, "Invalid end calculation" );

         for( size_t k=0UL; k<kpos; k+=4UL )
         {
            const size_t i1( element->index() );
            const ET1    v1( element->value() );
            ++element;
            const size_t i2( element->index() );
            const ET1    v2( element->value() );
            ++element;
            const size_t i3( element->index() );
            const ET1    v3( element->value() );
            ++element;
            const size_t i4( element->index() );
            const ET1    v4( element->value() );
            ++element;

            BLAZE_INTERNAL_ASSERT( i1 < i2 && i2 < i3 && i3 < i4, "Invalid sparse matrix index detected" );

            const SIMDType xmm1( set( v1 ) );
            const SIMDType xmm2( set( v2 ) );
            const SIMDType xmm3( set( v3 ) );
            const SIMDType xmm4( set( v4 ) );

            const size_t jbegin( ( IsUpper<MT5>::value )
                                 ?( ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) & size_t(-SIMDSIZE) )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT5>::value )
                               ?( IsStrictlyLower<MT5>::value ? i4 : i4+1UL )
                               :( B.columns() ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
            BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

            size_t j( jbegin );

            for( ; j<jpos; j+=SIMDSIZE ) {
               C.store( i, j, C.load(i,j) - xmm1 * B.load(i1,j) - xmm2 * B.load(i2,j) - xmm3 * B.load(i3,j) - xmm4 * B.load(i4,j) );
            }
            for( ; remainder && j<jend; ++j ) {
               C(i,j) -= v1 * B(i1,j) + v2 * B(i2,j) + v3 * B(i3,j) + v4 * B(i4,j);
            }
         }

         for( ; element!=end; ++element )
         {
            const size_t i1( element->index() );
            const ET1    v1( element->value() );

            const SIMDType xmm1( set( v1 ) );

            const size_t jbegin( ( IsUpper<MT5>::value )
                                 ?( ( IsStrictlyUpper<MT5>::value ? i1+1UL : i1 ) & size_t(-SIMDSIZE) )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT5>::value )
                               ?( IsStrictlyLower<MT5>::value ? i1 : i1+1UL )
                               :( B.columns() ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
            BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

            size_t j( jbegin );

            for( ; j<jpos; j+=SIMDSIZE ) {
               C.store( i, j, C.load(i,j) - xmm1 * B.load(i1,j) );
            }
            for( ; remainder && j<jend; ++j ) {
               C(i,j) -= v1 * B(i1,j);
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

   //**SMP assignment to dense matrices************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a sparse matrix-dense matrix multiplication to a dense matrix
   //        (\f$ A=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a sparse matrix-dense
   // matrix multiplication expression to a dense matrix. Due to the explicit application of the
   // SFINAE principle this function can only be selected by the compiler in case either of the
   // two matrix operands requires an intermediate evaluation and no symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline EnableIf_< IsEvaluationRequired<MT,MT1,MT2> >
      smpAssign( DenseMatrix<MT,SO>& lhs, const SMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LT A( rhs.lhs_ );  // Evaluation of the left-hand side sparse matrix operand
      RT B( rhs.rhs_ );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      smpAssign( ~lhs, A * B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to sparse matrices***********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a sparse matrix-dense matrix multiplication to a sparse matrix
   //        (\f$ A=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side sparse matrix.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a sparse matrix-dense
   // matrix multiplication expression to a sparse matrix. Due to the explicit application of the
   // SFINAE principle this function can only be selected by the compiler in case either of the
   // two matrix operands requires an intermediate evaluation and no symmetry can be exploited.
   */
   template< typename MT  // Type of the target sparse matrix
           , bool SO >    // Storage order of the target sparse matrix
   friend inline EnableIf_< IsEvaluationRequired<MT,MT1,MT2> >
      smpAssign( SparseMatrix<MT,SO>& lhs, const SMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      typedef IfTrue_< SO, OppositeType, ResultType >  TmpType;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( OppositeType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( OppositeType );
      BLAZE_CONSTRAINT_MATRICES_MUST_HAVE_SAME_STORAGE_ORDER( MT, TmpType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<TmpType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const TmpType tmp( rhs );
      smpAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to dense matrices***************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a sparse matrix-dense matrix multiplication to a dense
   //        matrix (\f$ A+=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a sparse
   // matrix-dense matrix multiplication expression to a dense matrix. Due to the explicit
   // application of the SFINAE principle this function can only be selected by the compiler
   // in case either of the two matrix operands requires an intermediate evaluation and no
   // symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline EnableIf_< IsEvaluationRequired<MT,MT1,MT2> >
      smpAddAssign( DenseMatrix<MT,SO>& lhs, const SMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LT A( rhs.lhs_ );  // Evaluation of the left-hand side sparse matrix operand
      RT B( rhs.rhs_ );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      smpAddAssign( ~lhs, A * B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to sparse matrices**************************************************
   // No special implementation for the SMP addition assignment to sparse matrices.
   //**********************************************************************************************

   //**SMP subtraction assignment to dense matrices************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a sparse matrix-dense matrix multiplication to a dense
   //        matrix (\f$ A-=B*C \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a sparse
   // matrix-dense matrix multiplication expression to a dense matrix. Due to the explicit
   // application of the SFINAE principle this function can only be selected by the compiler
   // in case either of the two matrix operands requires an intermediate evaluation and no
   // symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline EnableIf_< IsEvaluationRequired<MT,MT1,MT2> >
      smpSubAssign( DenseMatrix<MT,SO>& lhs, const SMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LT A( rhs.lhs_ );  // Evaluation of the left-hand side sparse matrix operand
      RT B( rhs.rhs_ );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      smpSubAssign( ~lhs, A * B );
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
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_FORM_VALID_MATMATMULTEXPR( MT1, MT2 );
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
/*!\brief Multiplication operator for the multiplication of a row-major sparse matrix and a
//        row-major dense matrix (\f$ A=B*C \f$).
// \ingroup dense_matrix
//
// \param lhs The left-hand side sparse matrix for the multiplication.
// \param rhs The right-hand side dense matrix for the multiplication.
// \return The resulting matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// This operator represents the multiplication of a row-major sparse matrix and a row-major
// dense matrix:

   \code
   using blaze::rowMajor;

   blaze::CompressedMatrix<double,rowMajor> A;
   blaze::DynamicMatrix<double,rowMajor> B, C;
   // ... Resizing and initialization
   C = A * B;
   \endcode

// The operator returns an expression representing a dense matrix of the higher-order element
// type of the two involved matrix element types \a T1::ElementType and \a T2::ElementType.
// Both matrix types \a T1 and \a T2 as well as the two element types \a T1::ElementType and
// \a T2::ElementType have to be supported by the MultTrait class template.\n
// In case the current sizes of the two given matrices don't match, a \a std::invalid_argument
// is thrown.
*/
template< typename T1    // Type of the left-hand side sparse matrix
        , typename T2 >  // Type of the right-hand side dense matrix
inline const SMatDMatMultExpr<T1,T2>
   operator*( const SparseMatrix<T1,false>& lhs, const DenseMatrix<T2,false>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   if( (~lhs).columns() != (~rhs).rows() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   return SMatDMatMultExpr<T1,T2>( ~lhs, ~rhs );
}
//*************************************************************************************************




//=================================================================================================
//
//  ROWS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct Rows< SMatDMatMultExpr<MT1,MT2> > : public Rows<MT1>
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
template< typename MT1, typename MT2 >
struct Columns< SMatDMatMultExpr<MT1,MT2> > : public Columns<MT2>
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
template< typename MT1, typename MT2 >
struct IsAligned< SMatDMatMultExpr<MT1,MT2> >
   : public BoolConstant< IsAligned<MT2>::value >
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
template< typename MT1, typename MT2 >
struct IsLower< SMatDMatMultExpr<MT1,MT2> >
   : public BoolConstant< And< IsLower<MT1>, IsLower<MT2> >::value >
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
template< typename MT1, typename MT2 >
struct IsUniLower< SMatDMatMultExpr<MT1,MT2> >
   : public BoolConstant< And< IsUniLower<MT1>, IsUniLower<MT2> >::value >
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
template< typename MT1, typename MT2 >
struct IsStrictlyLower< SMatDMatMultExpr<MT1,MT2> >
   : public BoolConstant< Or< And< IsStrictlyLower<MT1>, IsLower<MT2> >
                            , And< IsStrictlyLower<MT2>, IsLower<MT1> > >::value >
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
template< typename MT1, typename MT2 >
struct IsUpper< SMatDMatMultExpr<MT1,MT2> >
   : public BoolConstant< And< IsUpper<MT1>, IsUpper<MT2> >::value >
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
template< typename MT1, typename MT2 >
struct IsUniUpper< SMatDMatMultExpr<MT1,MT2> >
   : public BoolConstant< And< IsUniUpper<MT1>, IsUniUpper<MT2> >::value >
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
template< typename MT1, typename MT2 >
struct IsStrictlyUpper< SMatDMatMultExpr<MT1,MT2> >
   : public BoolConstant< Or< And< IsStrictlyUpper<MT1>, IsUpper<MT2> >
                            , And< IsStrictlyUpper<MT2>, IsUpper<MT1> > >::value >
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
template< typename MT1, typename MT2, typename VT >
struct DMatDVecMultExprTrait< SMatDMatMultExpr<MT1,MT2>, VT >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsDenseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsDenseVector<VT>, IsColumnVector<VT> >
                   , SMatDVecMultExprTrait_< MT1, DMatDVecMultExprTrait_<MT2,VT> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2, typename VT >
struct DMatSVecMultExprTrait< SMatDMatMultExpr<MT1,MT2>, VT >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsDenseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsSparseVector<VT>, IsColumnVector<VT> >
                   , SMatDVecMultExprTrait_< MT1, DMatSVecMultExprTrait_<MT2,VT> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename MT1, typename MT2 >
struct TDVecDMatMultExprTrait< VT, SMatDMatMultExpr<MT1,MT2> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT>, IsRowVector<VT>
                        , IsSparseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsDenseMatrix<MT2>, IsRowMajorMatrix<MT2> >
                   , TDVecDMatMultExprTrait_< TDVecSMatMultExprTrait_<VT,MT1>, MT2 >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename MT1, typename MT2 >
struct TSVecDMatMultExprTrait< VT, SMatDMatMultExpr<MT1,MT2> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsRowVector<VT>
                        , IsSparseMatrix<MT1>, IsRowMajorMatrix<MT1>
                        , IsDenseMatrix<MT2>, IsRowMajorMatrix<MT2> >
                   , TSVecDMatMultExprTrait_< TSVecSMatMultExprTrait_<VT,MT1>, MT2 >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2, bool AF >
struct SubmatrixExprTrait< SMatDMatMultExpr<MT1,MT2>, AF >
{
 public:
   //**********************************************************************************************
   using Type = MultExprTrait_< SubmatrixExprTrait_<const MT1,AF>
                              , SubmatrixExprTrait_<const MT2,AF> >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct RowExprTrait< SMatDMatMultExpr<MT1,MT2> >
{
 public:
   //**********************************************************************************************
   using Type = MultExprTrait_< RowExprTrait_<const MT1>, MT2 >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct ColumnExprTrait< SMatDMatMultExpr<MT1,MT2> >
{
 public:
   //**********************************************************************************************
   using Type = MultExprTrait_< MT1, ColumnExprTrait_<const MT2> >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
