//=================================================================================================
/*!
//  \file blaze/math/expressions/TDMatDMatMultExpr.h
//  \brief Header file for the transpose dense matrix/dense matrix multiplication expression
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

#ifndef _BLAZE_MATH_EXPRESSIONS_TDMATDMATMULTEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_TDMATDMATMULTEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/blas/gemm.h>
#include <blaze/math/blas/trmm.h>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/ColumnMajorMatrix.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/MatMatMultExpr.h>
#include <blaze/math/constraints/RowMajorMatrix.h>
#include <blaze/math/constraints/StorageOrder.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/MatMatMultExpr.h>
#include <blaze/math/expressions/MatScalarMultExpr.h>
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
#include <blaze/math/traits/SubmatrixExprTrait.h>
#include <blaze/math/traits/TDMatDVecMultExprTrait.h>
#include <blaze/math/traits/TDMatSVecMultExprTrait.h>
#include <blaze/math/traits/TDVecDMatMultExprTrait.h>
#include <blaze/math/traits/TDVecTDMatMultExprTrait.h>
#include <blaze/math/traits/TSVecTDMatMultExprTrait.h>
#include <blaze/math/typetraits/AreSIMDCombinable.h>
#include <blaze/math/typetraits/Columns.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsBLASCompatible.h>
#include <blaze/math/typetraits/IsColumnMajorMatrix.h>
#include <blaze/math/typetraits/IsColumnVector.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsDiagonal.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/IsRowVector.h>
#include <blaze/math/typetraits/IsSparseVector.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyTriangular.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsTriangular.h>
#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/math/typetraits/IsUniUpper.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/math/typetraits/Rows.h>
#include <blaze/system/BLAS.h>
#include <blaze/system/Blocking.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Complex.h>
#include <blaze/util/constraints/Numeric.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/InvalidType.h>
#include <blaze/util/logging/FunctionTrace.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Not.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsBuiltin.h>
#include <blaze/util/typetraits/IsComplex.h>
#include <blaze/util/typetraits/IsComplexDouble.h>
#include <blaze/util/typetraits/IsComplexFloat.h>
#include <blaze/util/typetraits/IsDouble.h>
#include <blaze/util/typetraits/IsFloat.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsSame.h>


namespace blaze {

//=================================================================================================
//
//  CLASS TDMATDMATMULTEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for transpose dense matrix-dense matrix multiplications.
// \ingroup dense_matrix_expression
//
// The TDMatDMatMultExpr class represents the compile time expression for multiplications between
// a column-major dense matrix and a row-major dense matrix.
*/
template< typename MT1    // Type of the left-hand side dense matrix
        , typename MT2 >  // Type of the right-hand side dense matrix
class TDMatDMatMultExpr : public DenseMatrix< TDMatDMatMultExpr<MT1,MT2>, true >
                        , private MatMatMultExpr
                        , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef ResultType_<MT1>     RT1;  //!< Result type of the left-hand side dense matrix expression.
   typedef ResultType_<MT2>     RT2;  //!< Result type of the right-hand side dense matrix expression.
   typedef ElementType_<RT1>    ET1;  //!< Element type of the left-hand side dense matrix expression.
   typedef ElementType_<RT2>    ET2;  //!< Element type of the right-hand side dense matrix expression.
   typedef CompositeType_<MT1>  CT1;  //!< Composite type of the left-hand side dense matrix expression.
   typedef CompositeType_<MT2>  CT2;  //!< Composite type of the right-hand side dense matrix expression.
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the left-hand side dense matrix expression.
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
   /*! In case the types of all three involved matrices are suited for a BLAS kernel, the nested
       \a value will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   struct UseBlasKernel {
      enum : bool { value = BLAZE_BLAS_MODE &&
                            HasMutableDataAccess<T1>::value &&
                            HasConstDataAccess<T2>::value &&
                            HasConstDataAccess<T3>::value &&
                            !IsDiagonal<T2>::value && !IsDiagonal<T3>::value &&
                            T1::simdEnabled && T2::simdEnabled && T3::simdEnabled &&
                            IsBLASCompatible< ElementType_<T1> >::value &&
                            IsBLASCompatible< ElementType_<T2> >::value &&
                            IsBLASCompatible< ElementType_<T3> >::value &&
                            IsSame< ElementType_<T1>, ElementType_<T2> >::value &&
                            IsSame< ElementType_<T1>, ElementType_<T3> >::value };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! In case all three involved data types are suited for a vectorized computation of the
       matrix multiplication, the nested \value will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   struct UseVectorizedDefaultKernel {
      enum : bool { value = useOptimizedKernels &&
                            !( IsDiagonal<T2>::value && IsDiagonal<T3>::value ) &&
                            !( IsDiagonal<T2>::value && IsColumnMajorMatrix<T1>::value ) &&
                            !( IsDiagonal<T3>::value && IsRowMajorMatrix<T1>::value ) &&
                            T1::simdEnabled && T2::simdEnabled && T3::simdEnabled &&
                            AreSIMDCombinable< ElementType_<T1>
                                             , ElementType_<T2>
                                             , ElementType_<T3> >::value &&
                            HasSIMDAdd< ElementType_<T2>, ElementType_<T3> >::value &&
                            HasSIMDMult< ElementType_<T2>, ElementType_<T3> >::value };
   };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef TDMatDMatMultExpr<MT1,MT2>  This;           //!< Type of this TDMatDMatMultExpr instance.
   typedef MultTrait_<RT1,RT2>         ResultType;     //!< Result type for expression template evaluations.
   typedef OppositeType_<ResultType>   OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef TransposeType_<ResultType>  TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<ResultType>    ElementType;    //!< Resulting element type.
   typedef SIMDTrait_<ElementType>     SIMDType;       //!< Resulting SIMD element type.
   typedef const ElementType           ReturnType;     //!< Return type for expression template evaluations.
   typedef const ResultType            CompositeType;  //!< Data type for composite expression templates.

   //! Composite type of the left-hand side dense matrix expression.
   typedef If_< IsExpression<MT1>, const MT1, const MT1& >  LeftOperand;

   //! Composite type of the right-hand side dense matrix expression.
   typedef If_< IsExpression<MT2>, const MT2, const MT2& >  RightOperand;

   //! Type for the assignment of the left-hand side dense matrix operand.
   typedef IfTrue_< evaluateLeft, const RT1, CT1 >  LT;

   //! Type for the assignment of the right-hand side dense matrix operand.
   typedef IfTrue_< evaluateRight, const RT2, CT2 >  RT;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = !( IsDiagonal<MT1>::value && IsDiagonal<MT2>::value ) &&
                               MT1::simdEnabled && MT2::simdEnabled &&
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
   /*!\brief Constructor for the TDMatDMatMultExpr class.
   //
   // \param lhs The left-hand side operand of the multiplication expression.
   // \param rhs The right-hand side operand of the multiplication expression.
   */
   explicit inline TDMatDMatMultExpr( const MT1& lhs, const MT2& rhs ) noexcept
      : lhs_( lhs )  // Left-hand side dense matrix of the multiplication expression
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
   /*!\brief Returns the left-hand side transpose dense matrix operand.
   //
   // \return The left-hand side transpose dense matrix operand.
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
      return lhs_.isAligned() && rhs_.isAligned();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return ( !BLAZE_BLAS_IS_PARALLEL ||
               ( rows() * columns() < TDMATDMATMULT_THRESHOLD ) ) &&
             ( columns() > SMP_TDMATDMATMULT_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  lhs_;  //!< Left-hand side dense matrix of the multiplication expression.
   RightOperand rhs_;  //!< Right-hand side dense matrix of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense matrices****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a transpose dense matrix-dense matrix multiplication to a dense matrix
   //        (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a transpose dense matrix-
   // dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline void assign( DenseMatrix<MT,SO>& lhs, const TDMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL ) {
         return;
      }
      else if( rhs.lhs_.columns() == 0UL ) {
         reset( ~lhs );
         return;
      }

      LT A( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense matrix operand
      RT B( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      TDMatDMatMultExpr::selectAssignKernel( ~lhs, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to dense matrices (kernel selection)*********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for an assignment of a transpose dense matrix-dense matrix
   //        multiplication to a dense matrix (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline void selectAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      if( ( IsDiagonal<MT4>::value && IsDiagonal<MT5>::value ) ||
          ( C.rows() * C.columns() < TDMATDMATMULT_THRESHOLD ) )
         selectSmallAssignKernel( C, A, B );
      else
         selectBlasAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to row-major dense matrices (general/general)****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a general transpose dense matrix-general dense matrix
   //        multiplication (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default assignment of a general transpose dense matrix-general
   // dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, Not< IsDiagonal<MT5> > > >
      selectDefaultAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t kbegin( ( IsUpper<MT4>::value )
                              ?( IsStrictlyUpper<MT4>::value ? i+1UL : i )
                              :( 0UL ) );
         const size_t kend( ( IsLower<MT4>::value )
                            ?( IsStrictlyLower<MT4>::value ? i : i+1UL )
                            :( K ) );
         BLAZE_INTERNAL_ASSERT( kbegin <= kend, "Invalid loop indices detected" );

         if( IsStrictlyTriangular<MT4>::value && kbegin == kend ) {
            for( size_t j=0UL; j<N; ++j ) {
               reset( (~C)(i,j) );
            }
            continue;
         }

         {
            const size_t jbegin( ( IsUpper<MT5>::value )
                                 ?( IsStrictlyUpper<MT5>::value ? kbegin+1UL : kbegin )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT5>::value )
                               ?( IsStrictlyLower<MT5>::value ? kbegin : kbegin+1UL )
                               :( N ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            if( IsUpper<MT4>::value && IsUpper<MT5>::value ) {
               for( size_t j=0UL; j<jbegin; ++j ) {
                  reset( (~C)(i,j) );
               }
            }
            else if( IsStrictlyUpper<MT5>::value ) {
               reset( (~C)(i,0UL) );
            }
            for( size_t j=jbegin; j<jend; ++j ) {
               (~C)(i,j) = A(i,kbegin) * B(kbegin,j);
            }
            if( IsLower<MT4>::value && IsLower<MT5>::value ) {
               for( size_t j=jend; j<N; ++j ) {
                  reset( (~C)(i,j) );
               }
            }
            else if( IsStrictlyLower<MT5>::value ) {
               reset( (~C)(i,N-1UL) );
            }
         }

         for( size_t k=kbegin+1UL; k<kend; ++k )
         {
            const size_t jbegin( ( IsUpper<MT5>::value )
                                 ?( IsStrictlyUpper<MT5>::value ? k+1UL : k )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT5>::value )
                               ?( IsStrictlyLower<MT5>::value ? k-1UL : k )
                               :( N ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            for( size_t j=jbegin; j<jend; ++j ) {
               (~C)(i,j) += A(i,k) * B(k,j);
            }
            if( IsLower<MT5>::value ) {
               (~C)(i,jend) = A(i,k) * B(k,jend);
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to column-major dense matrices (general/general)*************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a general transpose dense matrix-general dense matrix
   //        multiplication (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default assignment of a general transpose dense matrix-general
   // dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, Not< IsDiagonal<MT5> > > >
      selectDefaultAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      for( size_t j=0UL; j<N; ++j )
      {
         const size_t kbegin( ( IsLower<MT5>::value )
                              ?( IsStrictlyLower<MT5>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t kend( ( IsUpper<MT5>::value )
                            ?( IsStrictlyUpper<MT5>::value ? j : j+1UL )
                            :( K ) );
         BLAZE_INTERNAL_ASSERT( kbegin <= kend, "Invalid loop indices detected" );

         if( IsStrictlyTriangular<MT5>::value && kbegin == kend ) {
            for( size_t i=0UL; i<M; ++i ) {
               reset( (~C)(i,j) );
            }
            continue;
         }

         {
            const size_t ibegin( ( IsLower<MT4>::value )
                                 ?( IsStrictlyLower<MT4>::value ? kbegin+1UL : kbegin )
                                 :( 0UL ) );
            const size_t iend( ( IsUpper<MT4>::value )
                               ?( IsStrictlyUpper<MT4>::value ? kbegin : kbegin+1UL )
                               :( M ) );
            BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

            if( IsLower<MT4>::value && IsLower<MT5>::value ) {
               for( size_t i=0UL; i<ibegin; ++i ) {
                  reset( (~C)(i,j) );
               }
            }
            else if( IsStrictlyLower<MT4>::value ) {
               reset( (~C)(0UL,j) );
            }
            for( size_t i=ibegin; i<iend; ++i ) {
               (~C)(i,j) = A(i,kbegin) * B(kbegin,j);
            }
            if( IsUpper<MT4>::value && IsUpper<MT5>::value ) {
               for( size_t i=iend; i<M; ++i ) {
                  reset( (~C)(i,j) );
               }
            }
            else if( IsStrictlyUpper<MT4>::value ) {
               reset( (~C)(M-1UL,j) );
            }
         }

         for( size_t k=kbegin+1UL; k<kend; ++k )
         {
            const size_t ibegin( ( IsLower<MT4>::value )
                                 ?( IsStrictlyLower<MT4>::value ? k+1UL : k )
                                 :( 0UL ) );
            const size_t iend( ( IsUpper<MT4>::value )
                               ?( IsStrictlyUpper<MT4>::value ? k-1UL : k )
                               :( M ) );
            BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

            for( size_t i=ibegin; i<iend; ++i ) {
               (~C)(i,j) += A(i,k) * B(k,j);
            }
            if( IsUpper<MT4>::value ) {
               (~C)(iend,j) = A(iend,k) * B(k,j);
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to row-major dense matrices (general/diagonal)***************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a general transpose dense matrix-diagonal dense matrix
   //        multiplication (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default assignment of a general transpose dense matrix-diagonal
   // dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, IsDiagonal<MT5> > >
      selectDefaultAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      const size_t block( BLOCK_SIZE );

      for( size_t ii=0UL; ii<M; ii+=block ) {
         const size_t iend( min( M, ii+block ) );
         for( size_t jj=0UL; jj<N; jj+=block ) {
            const size_t jend( min( N, jj+block ) );
            for( size_t i=ii; i<iend; ++i )
            {
               const size_t jbegin( ( IsUpper<MT4>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), jj ) )
                                    :( jj ) );
               const size_t jpos( ( IsLower<MT4>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i : i+1UL ), jend ) )
                                  :( jend ) );

               if( IsUpper<MT4>::value ) {
                  for( size_t j=jj; j<jbegin; ++j ) {
                     reset( (~C)(i,j) );
                  }
               }
               for( size_t j=jbegin; j<jpos; ++j ) {
                  (~C)(i,j) = A(i,j) * B(j,j);
               }
               if( IsLower<MT4>::value ) {
                  for( size_t j=jpos; j<jend; ++j ) {
                     reset( (~C)(i,j) );
                  }
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to column-major dense matrices (general/diagonal)************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a general transpose dense matrix-diagonal dense matrix
   //        multiplication (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default assignment of a general transpose dense matrix-diagonal
   // dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, IsDiagonal<MT5> > >
      selectDefaultAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t j=0UL; j<N; ++j )
      {
         const size_t ibegin( ( IsLower<MT4>::value )
                              ?( IsStrictlyLower<MT4>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT4>::value )
                            ?( IsStrictlyUpper<MT4>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         if( IsLower<MT4>::value ) {
            for( size_t i=0UL; i<ibegin; ++i ) {
               reset( (~C)(i,j) );
            }
         }
         for( size_t i=ibegin; i<iend; ++i ) {
            (~C)(i,j) = A(i,j) * B(j,j);
         }
         if( IsUpper<MT4>::value ) {
            for( size_t i=iend; i<M; ++i ) {
               reset( (~C)(i,j) );
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to row-major dense matrices (diagonal/general)***************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a diagonal transpose dense matrix-general dense matrix
   //        multiplication (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default assignment of a diagonal transpose dense matrix-general
   // dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< IsDiagonal<MT4>, Not< IsDiagonal<MT5> > > >
      selectDefaultAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper<MT5>::value )
                              ?( IsStrictlyUpper<MT5>::value ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT5>::value )
                            ?( IsStrictlyLower<MT5>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         if( IsUpper<MT5>::value ) {
            for( size_t j=0UL; j<jbegin; ++j ) {
               reset( (~C)(i,j) );
            }
         }
         for( size_t j=jbegin; j<jend; ++j ) {
            (~C)(i,j) = A(i,i) * B(i,j);
         }
         if( IsLower<MT5>::value ) {
            for( size_t j=jend; j<N; ++j ) {
               reset( (~C)(i,j) );
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to column-major dense matrices (diagonal/general)************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a diagonal transpose dense matrix-general dense matrix
   //        multiplication (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default assignment of a diagonal transpose dense matrix-general
   // dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< IsDiagonal<MT4>, Not< IsDiagonal<MT5> > > >
      selectDefaultAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      const size_t block( BLOCK_SIZE );

      for( size_t jj=0UL; jj<N; jj+=block ) {
         const size_t jend( min( N, jj+block ) );
         for( size_t ii=0UL; ii<M; ii+=block ) {
            const size_t iend( min( M, ii+block ) );
            for( size_t j=jj; j<jend; ++j )
            {
               const size_t ibegin( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyLower<MT5>::value ? j+1UL : j ), ii ) )
                                    :( ii ) );
               const size_t ipos( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyUpper<MT5>::value ? j : j+1UL ), iend ) )
                                  :( iend ) );

               if( IsLower<MT5>::value ) {
                  for( size_t i=ii; i<ibegin; ++i ) {
                     reset( (~C)(i,j) );
                  }
               }
               for( size_t i=ibegin; i<ipos; ++i ) {
                  (~C)(i,j) = A(i,i) * B(i,j);
               }
               if( IsUpper<MT5>::value ) {
                  for( size_t i=ipos; i<iend; ++i ) {
                     reset( (~C)(i,j) );
                  }
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense matrices (diagonal/diagonal)************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a diagonal transpose dense matrix-diagonal dense matrix
   //        multiplication (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default assignment of a diagonal transpose dense matrix-
   // diagonal dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< IsDiagonal<MT4>, IsDiagonal<MT5> > >
      selectDefaultAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      reset( C );

      for( size_t i=0UL; i<A.rows(); ++i ) {
         C(i,i) = A(i,i) * B(i,i);
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense matrices (small matrices)***************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a small transpose dense matrix-dense matrix multiplication
   //        (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a transpose dense
   // matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectSmallAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectDefaultAssignKernel( ~C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default assignment to row-major dense matrices (small matrices)******************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default assignment of a small transpose dense matrix-dense matrix
   //        multiplication (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default assignment of a transpose dense matrix-
   // dense matrix multiplication expression to a row-major dense matrix. This kernel is
   // optimized for small matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectSmallAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );

      for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL ) {
         for( size_t i=0UL; i<M; ++i )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i : i+1UL ), j+SIMDSIZE*8UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i : i+1UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*8UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j             );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE    );
               xmm3 = xmm3 + a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 = xmm4 + a1 * B.load(k,j+SIMDSIZE*3UL);
               xmm5 = xmm5 + a1 * B.load(k,j+SIMDSIZE*4UL);
               xmm6 = xmm6 + a1 * B.load(k,j+SIMDSIZE*5UL);
               xmm7 = xmm7 + a1 * B.load(k,j+SIMDSIZE*6UL);
               xmm8 = xmm8 + a1 * B.load(k,j+SIMDSIZE*7UL);
            }

            (~C).store( i, j             , xmm1 );
            (~C).store( i, j+SIMDSIZE    , xmm2 );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
            (~C).store( i, j+SIMDSIZE*4UL, xmm5 );
            (~C).store( i, j+SIMDSIZE*5UL, xmm6 );
            (~C).store( i, j+SIMDSIZE*6UL, xmm7 );
            (~C).store( i, j+SIMDSIZE*7UL, xmm8 );
         }
      }

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ), j+SIMDSIZE*4UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a1 * b2;
               xmm3 = xmm3 + a1 * b3;
               xmm4 = xmm4 + a1 * b4;
               xmm5 = xmm5 + a2 * b1;
               xmm6 = xmm6 + a2 * b2;
               xmm7 = xmm7 + a2 * b3;
               xmm8 = xmm8 + a2 * b4;
            }

            (~C).store( i    , j             , xmm1 );
            (~C).store( i    , j+SIMDSIZE    , xmm2 );
            (~C).store( i    , j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i    , j+SIMDSIZE*3UL, xmm4 );
            (~C).store( i+1UL, j             , xmm5 );
            (~C).store( i+1UL, j+SIMDSIZE    , xmm6 );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm7 );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, xmm8 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j             );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE    );
               xmm3 = xmm3 + a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 = xmm4 + a1 * B.load(k,j+SIMDSIZE*3UL);
            }

            (~C).store( i, j             , xmm1 );
            (~C).store( i, j+SIMDSIZE    , xmm2 );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
         }
      }

      for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a1 * b2;
               xmm3 = xmm3 + a2 * b1;
               xmm4 = xmm4 + a2 * b2;
            }

            (~C).store( i    , j         , xmm1 );
            (~C).store( i    , j+SIMDSIZE, xmm2 );
            (~C).store( i+1UL, j         , xmm3 );
            (~C).store( i+1UL, j+SIMDSIZE, xmm4 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j         );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE);
            }

            (~C).store( i, j         , xmm1 );
            (~C).store( i, j+SIMDSIZE, xmm2 );
         }
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL )
                               :( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 = xmm1 + set( A(i    ,k) ) * b1;
               xmm2 = xmm2 + set( A(i+1UL,k) ) * b1;
            }

            (~C).store( i    , j, xmm1 );
            (~C).store( i+1UL, j, xmm2 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );

            SIMDType xmm1;

            for( size_t k=kbegin; k<K; ++k ) {
               xmm1 = xmm1 + set( A(i,k) ) * B.load(k,j);
            }

            (~C).store( i, j, xmm1 );
         }
      }

      for( ; remainder && j<N; ++j )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL )
                               :( K ) );

            ElementType value1 = ElementType();
            ElementType value2 = ElementType();

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 += A(i    ,k) * B(k,j);
               value2 += A(i+1UL,k) * B(k,j);
            }

            (~C)(i    ,j) = value1;
            (~C)(i+1UL,j) = value2;
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );

            ElementType value = ElementType();

            for( size_t k=kbegin; k<K; ++k ) {
               value += A(i,k) * B(k,j);
            }

            (~C)(i,j) = value;
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default assignment to column-major dense matrices (small matrices)***************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default assignment of a small transpose dense matrix-dense matrix
   //        multiplication (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default assignment of a transpose dense matrix-
   // dense matrix multiplication expression to a column-major dense matrix. This kernel is
   // optimized for small matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectSmallAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT4>::value );

      const size_t ipos( remainder ? ( M & size_t(-SIMDSIZE) ) : M );
      BLAZE_INTERNAL_ASSERT( !remainder || ( M - ( M % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

      size_t i( 0UL );

      for( ; (i+SIMDSIZE*7UL) < ipos; i+=SIMDSIZE*8UL ) {
         for( size_t j=0UL; j<N; ++j )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*8UL, K, ( IsStrictlyUpper<MT5>::value ? j : j+1UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j : j+1UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*8UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i             ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE    ,k) * b1;
               xmm3 = xmm3 + A.load(i+SIMDSIZE*2UL,k) * b1;
               xmm4 = xmm4 + A.load(i+SIMDSIZE*3UL,k) * b1;
               xmm5 = xmm5 + A.load(i+SIMDSIZE*4UL,k) * b1;
               xmm6 = xmm6 + A.load(i+SIMDSIZE*5UL,k) * b1;
               xmm7 = xmm7 + A.load(i+SIMDSIZE*6UL,k) * b1;
               xmm8 = xmm8 + A.load(i+SIMDSIZE*7UL,k) * b1;
            }

            (~C).store( i             , j, xmm1 );
            (~C).store( i+SIMDSIZE    , j, xmm2 );
            (~C).store( i+SIMDSIZE*2UL, j, xmm3 );
            (~C).store( i+SIMDSIZE*3UL, j, xmm4 );
            (~C).store( i+SIMDSIZE*4UL, j, xmm5 );
            (~C).store( i+SIMDSIZE*5UL, j, xmm6 );
            (~C).store( i+SIMDSIZE*6UL, j, xmm7 );
            (~C).store( i+SIMDSIZE*7UL, j, xmm8 );
         }
      }

      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*4UL, K, ( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i             ,k) );
               const SIMDType a2( A.load(i+SIMDSIZE    ,k) );
               const SIMDType a3( A.load(i+SIMDSIZE*2UL,k) );
               const SIMDType a4( A.load(i+SIMDSIZE*3UL,k) );
               const SIMDType b1( set( B(k,j    ) ) );
               const SIMDType b2( set( B(k,j+1UL) ) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a2 * b1;
               xmm3 = xmm3 + a3 * b1;
               xmm4 = xmm4 + a4 * b1;
               xmm5 = xmm5 + a1 * b2;
               xmm6 = xmm6 + a2 * b2;
               xmm7 = xmm7 + a3 * b2;
               xmm8 = xmm8 + a4 * b2;
            }

            (~C).store( i             , j    , xmm1 );
            (~C).store( i+SIMDSIZE    , j    , xmm2 );
            (~C).store( i+SIMDSIZE*2UL, j    , xmm3 );
            (~C).store( i+SIMDSIZE*3UL, j    , xmm4 );
            (~C).store( i             , j+1UL, xmm5 );
            (~C).store( i+SIMDSIZE    , j+1UL, xmm6 );
            (~C).store( i+SIMDSIZE*2UL, j+1UL, xmm7 );
            (~C).store( i+SIMDSIZE*3UL, j+1UL, xmm8 );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i             ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE    ,k) * b1;
               xmm3 = xmm3 + A.load(i+SIMDSIZE*2UL,k) * b1;
               xmm4 = xmm4 + A.load(i+SIMDSIZE*3UL,k) * b1;
            }

            (~C).store( i             , j, xmm1 );
            (~C).store( i+SIMDSIZE    , j, xmm2 );
            (~C).store( i+SIMDSIZE*2UL, j, xmm3 );
            (~C).store( i+SIMDSIZE*3UL, j, xmm4 );
         }
      }

      for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*2UL, K, ( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i         ,k) );
               const SIMDType a2( A.load(i+SIMDSIZE,k) );
               const SIMDType b1( set( B(k,j    ) ) );
               const SIMDType b2( set( B(k,j+1UL) ) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a2 * b1;
               xmm3 = xmm3 + a1 * b2;
               xmm4 = xmm4 + a2 * b2;
            }

            (~C).store( i         , j    , xmm1 );
            (~C).store( i+SIMDSIZE, j    , xmm2 );
            (~C).store( i         , j+1UL, xmm3 );
            (~C).store( i+SIMDSIZE, j+1UL, xmm4 );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i         ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE,k) * b1;
            }

            (~C).store( i         , j, xmm1 );
            (~C).store( i+SIMDSIZE, j, xmm2 );
         }
      }

      for( ; i<ipos; i+=SIMDSIZE )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL )
                               :( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i,k) );
               xmm1 = xmm1 + a1 * set( B(k,j    ) );
               xmm2 = xmm2 + a1 * set( B(k,j+1UL) );
            }

            (~C).store( i, j    , xmm1 );
            (~C).store( i, j+1UL, xmm2 );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );

            SIMDType xmm1;

            for( size_t k=kbegin; k<K; ++k ) {
               xmm1 = xmm1 + A.load(i,k) * set( B(k,j) );
            }

            (~C).store( i, j, xmm1 );
         }
      }

      for( ; remainder && i<M; ++i )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL )
                               :( K ) );

            ElementType value1 = ElementType();
            ElementType value2 = ElementType();

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 += A(i,k) * B(k,j    );
               value2 += A(i,k) * B(k,j+1UL);
            }

            (~C)(i,j    ) = value1;
            (~C)(i,j+1UL) = value2;
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );

            ElementType value = ElementType();

            for( size_t k=kbegin; k<K; ++k ) {
               value += A(i,k) * B(k,j);
            }

            (~C)(i,j) = value;
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense matrices (large matrices)***************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a large transpose dense matrix-dense matrix multiplication
   //        (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a transpose dense
   // matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectLargeAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectDefaultAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default assignment to row-major dense matrices (large matrices)******************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default assignment of a large transpose dense matrix-dense matrix
   //        multiplication (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default assignment of a transpose dense matrix-
   // dense matrix multiplication expression to a row-major dense matrix. This kernel is
   // optimized for large matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectLargeAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      for( size_t jj=0UL; jj<N; jj+=DMATDMATMULT_DEFAULT_JBLOCK_SIZE )
      {
         const size_t jend( min( jj+DMATDMATMULT_DEFAULT_JBLOCK_SIZE, N ) );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

         for( size_t ii=0UL; ii<M; ii+=DMATDMATMULT_DEFAULT_IBLOCK_SIZE )
         {
            const size_t iend( min( ii+DMATDMATMULT_DEFAULT_IBLOCK_SIZE, M ) );

            for( size_t i=ii; i<iend; ++i ) {
               for( size_t j=jj; j<jend; ++j ) {
                  reset( (~C)(i,j) );
               }
            }

            for( size_t kk=0UL; kk<K; kk+=DMATDMATMULT_DEFAULT_KBLOCK_SIZE )
            {
               const size_t ktmp( min( kk+DMATDMATMULT_DEFAULT_KBLOCK_SIZE, K ) );

               size_t j( jj );

               for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
               {
                  const size_t j1( j+SIMDSIZE     );
                  const size_t j2( j+SIMDSIZE*2UL );
                  const size_t j3( j+SIMDSIZE*3UL );

                  size_t i( ii );

                  for( ; (i+2UL) <= iend; i+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+2UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i    ,j ) );
                     SIMDType xmm2( (~C).load(i    ,j1) );
                     SIMDType xmm3( (~C).load(i    ,j2) );
                     SIMDType xmm4( (~C).load(i    ,j3) );
                     SIMDType xmm5( (~C).load(i+1UL,j ) );
                     SIMDType xmm6( (~C).load(i+1UL,j1) );
                     SIMDType xmm7( (~C).load(i+1UL,j2) );
                     SIMDType xmm8( (~C).load(i+1UL,j3) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        const SIMDType b3( B.load(k,j2) );
                        const SIMDType b4( B.load(k,j3) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a1 * b3;
                        xmm4 = xmm4 + a1 * b4;
                        xmm5 = xmm5 + a2 * b1;
                        xmm6 = xmm6 + a2 * b2;
                        xmm7 = xmm7 + a2 * b3;
                        xmm8 = xmm8 + a2 * b4;
                     }

                     (~C).store( i    , j , xmm1 );
                     (~C).store( i    , j1, xmm2 );
                     (~C).store( i    , j2, xmm3 );
                     (~C).store( i    , j3, xmm4 );
                     (~C).store( i+1UL, j , xmm5 );
                     (~C).store( i+1UL, j1, xmm6 );
                     (~C).store( i+1UL, j2, xmm7 );
                     (~C).store( i+1UL, j3, xmm8 );
                  }

                  if( i < iend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i,j ) );
                     SIMDType xmm2( (~C).load(i,j1) );
                     SIMDType xmm3( (~C).load(i,j2) );
                     SIMDType xmm4( (~C).load(i,j3) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j );
                        xmm2 = xmm2 + a1 * B.load(k,j1);
                        xmm3 = xmm3 + a1 * B.load(k,j2);
                        xmm4 = xmm4 + a1 * B.load(k,j3);
                     }

                     (~C).store( i, j , xmm1 );
                     (~C).store( i, j1, xmm2 );
                     (~C).store( i, j2, xmm3 );
                     (~C).store( i, j3, xmm4 );
                  }
               }

               for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
               {
                  const size_t j1( j+SIMDSIZE );

                  size_t i( ii );

                  for( ; (i+4UL) <= iend; i+=4UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+4UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i    ,j ) );
                     SIMDType xmm2( (~C).load(i    ,j1) );
                     SIMDType xmm3( (~C).load(i+1UL,j ) );
                     SIMDType xmm4( (~C).load(i+1UL,j1) );
                     SIMDType xmm5( (~C).load(i+2UL,j ) );
                     SIMDType xmm6( (~C).load(i+2UL,j1) );
                     SIMDType xmm7( (~C).load(i+3UL,j ) );
                     SIMDType xmm8( (~C).load(i+3UL,j1) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType a3( set( A(i+2UL,k) ) );
                        const SIMDType a4( set( A(i+3UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a2 * b1;
                        xmm4 = xmm4 + a2 * b2;
                        xmm5 = xmm5 + a3 * b1;
                        xmm6 = xmm6 + a3 * b2;
                        xmm7 = xmm7 + a4 * b1;
                        xmm8 = xmm8 + a4 * b2;
                     }

                     (~C).store( i    , j , xmm1 );
                     (~C).store( i    , j1, xmm2 );
                     (~C).store( i+1UL, j , xmm3 );
                     (~C).store( i+1UL, j1, xmm4 );
                     (~C).store( i+2UL, j , xmm5 );
                     (~C).store( i+2UL, j1, xmm6 );
                     (~C).store( i+3UL, j , xmm7 );
                     (~C).store( i+3UL, j1, xmm8 );
                  }

                  for( ; (i+2UL) <= iend; i+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+2UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i    ,j ) );
                     SIMDType xmm2( (~C).load(i    ,j1) );
                     SIMDType xmm3( (~C).load(i+1UL,j ) );
                     SIMDType xmm4( (~C).load(i+1UL,j1) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a2 * b1;
                        xmm4 = xmm4 + a2 * b2;
                     }

                     (~C).store( i    , j , xmm1 );
                     (~C).store( i    , j1, xmm2 );
                     (~C).store( i+1UL, j , xmm3 );
                     (~C).store( i+1UL, j1, xmm4 );
                  }

                  if( i < iend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i,j ) );
                     SIMDType xmm2( (~C).load(i,j1) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j );
                        xmm2 = xmm2 + a1 * B.load(k,j1);
                     }

                     (~C).store( i, j , xmm1 );
                     (~C).store( i, j1, xmm2 );
                  }
               }

               for( ; j<jpos; j+=SIMDSIZE )
               {
                  for( size_t i=ii; i<iend; ++i )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j);
                     }

                     (~C).store( i, j, xmm1 );
                  }
               }

               for( ; remainder && j<jend; ++j )
               {
                  for( size_t i=ii; i<iend; ++i )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+1UL, ktmp ) ):( ktmp ) ) );

                     ElementType value( (~C)(i,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        value += A(i,k) * B(k,j);
                     }

                     (~C)(i,j) = value;
                  }
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default assignment to column-major dense matrices (large matrices)***************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default assignment of a large transpose dense matrix-dense matrix
   //        multiplication (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default assignment of a transpose dense matrix-
   // dense matrix multiplication expression to a column-major dense matrix. This kernel is
   // optimized for large matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectLargeAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT4>::value );

      for( size_t ii=0UL; ii<M; ii+=TDMATTDMATMULT_DEFAULT_IBLOCK_SIZE )
      {
         const size_t iend( min( ii+TDMATTDMATMULT_DEFAULT_IBLOCK_SIZE, M ) );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

         for( size_t jj=0UL; jj<N; jj+=TDMATTDMATMULT_DEFAULT_JBLOCK_SIZE )
         {
            const size_t jend( min( jj+TDMATTDMATMULT_DEFAULT_JBLOCK_SIZE, N ) );

            for( size_t j=jj; j<jend; ++j ) {
               for( size_t i=ii; i<iend; ++i ) {
                  reset( (~C)(i,j) );
               }
            }

            for( size_t kk=0UL; kk<K; kk+=TDMATTDMATMULT_DEFAULT_KBLOCK_SIZE )
            {
               const size_t ktmp( min( kk+TDMATTDMATMULT_DEFAULT_KBLOCK_SIZE, K ) );

               size_t i( ii );

               for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL )
               {
                  const size_t i1( i+SIMDSIZE     );
                  const size_t i2( i+SIMDSIZE*2UL );
                  const size_t i3( i+SIMDSIZE*3UL );

                  size_t j( jj );

                  for( ; (j+2UL) <= jend; j+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+2UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j    ) );
                     SIMDType xmm2( (~C).load(i1,j    ) );
                     SIMDType xmm3( (~C).load(i2,j    ) );
                     SIMDType xmm4( (~C).load(i3,j    ) );
                     SIMDType xmm5( (~C).load(i ,j+1UL) );
                     SIMDType xmm6( (~C).load(i1,j+1UL) );
                     SIMDType xmm7( (~C).load(i2,j+1UL) );
                     SIMDType xmm8( (~C).load(i3,j+1UL) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType a3( A.load(i2,k) );
                        const SIMDType a4( A.load(i3,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a3 * b1;
                        xmm4 = xmm4 + a4 * b1;
                        xmm5 = xmm5 + a1 * b2;
                        xmm6 = xmm6 + a2 * b2;
                        xmm7 = xmm7 + a3 * b2;
                        xmm8 = xmm8 + a4 * b2;
                     }

                     (~C).store( i , j    , xmm1 );
                     (~C).store( i1, j    , xmm2 );
                     (~C).store( i2, j    , xmm3 );
                     (~C).store( i3, j    , xmm4 );
                     (~C).store( i , j+1UL, xmm5 );
                     (~C).store( i1, j+1UL, xmm6 );
                     (~C).store( i2, j+1UL, xmm7 );
                     (~C).store( i3, j+1UL, xmm8 );
                  }

                  if( j < jend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j) );
                     SIMDType xmm2( (~C).load(i1,j) );
                     SIMDType xmm3( (~C).load(i2,j) );
                     SIMDType xmm4( (~C).load(i3,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i ,k) * b1;
                        xmm2 = xmm2 + A.load(i1,k) * b1;
                        xmm3 = xmm3 + A.load(i2,k) * b1;
                        xmm4 = xmm4 + A.load(i3,k) * b1;
                     }

                     (~C).store( i , j, xmm1 );
                     (~C).store( i1, j, xmm2 );
                     (~C).store( i2, j, xmm3 );
                     (~C).store( i3, j, xmm4 );
                  }
               }

               for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL )
               {
                  const size_t i1( i+SIMDSIZE );

                  size_t j( jj );

                  for( ; (j+4UL) <= jend; j+=4UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+4UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j    ) );
                     SIMDType xmm2( (~C).load(i1,j    ) );
                     SIMDType xmm3( (~C).load(i ,j+1UL) );
                     SIMDType xmm4( (~C).load(i1,j+1UL) );
                     SIMDType xmm5( (~C).load(i ,j+2UL) );
                     SIMDType xmm6( (~C).load(i1,j+2UL) );
                     SIMDType xmm7( (~C).load(i ,j+3UL) );
                     SIMDType xmm8( (~C).load(i1,j+3UL) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        const SIMDType b3( set( B(k,j+2UL) ) );
                        const SIMDType b4( set( B(k,j+3UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a1 * b2;
                        xmm4 = xmm4 + a2 * b2;
                        xmm5 = xmm5 + a1 * b3;
                        xmm6 = xmm6 + a2 * b3;
                        xmm7 = xmm7 + a1 * b4;
                        xmm8 = xmm8 + a2 * b4;
                     }

                     (~C).store( i , j    , xmm1 );
                     (~C).store( i1, j    , xmm2 );
                     (~C).store( i , j+1UL, xmm3 );
                     (~C).store( i1, j+1UL, xmm4 );
                     (~C).store( i , j+2UL, xmm5 );
                     (~C).store( i1, j+2UL, xmm6 );
                     (~C).store( i , j+3UL, xmm7 );
                     (~C).store( i1, j+3UL, xmm8 );
                  }

                  for( ; (j+2UL) <= jend; j+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+2UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j    ) );
                     SIMDType xmm2( (~C).load(i1,j    ) );
                     SIMDType xmm3( (~C).load(i ,j+1UL) );
                     SIMDType xmm4( (~C).load(i1,j+1UL) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a1 * b2;
                        xmm4 = xmm4 + a2 * b2;
                     }

                     (~C).store( i , j    , xmm1 );
                     (~C).store( i1, j    , xmm2 );
                     (~C).store( i , j+1UL, xmm3 );
                     (~C).store( i1, j+1UL, xmm4 );
                  }

                  if( j < jend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j) );
                     SIMDType xmm2( (~C).load(i1,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i ,k) * b1;
                        xmm2 = xmm2 + A.load(i1,k) * b1;
                     }

                     (~C).store( i , j, xmm1 );
                     (~C).store( i1, j, xmm2 );
                  }
               }

               for( ; i<ipos; i+=SIMDSIZE )
               {
                  for( size_t j=jj; j<jend; ++j )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i,k) * b1;
                     }

                     (~C).store( i, j, xmm1 );
                  }
               }

               for( ; remainder && i<iend; ++i )
               {
                  for( size_t j=jj; j<jend; ++j )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+1UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     ElementType value( (~C)(i,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        value += A(i,k) * B(k,j);
                     }

                     (~C)(i,j) = value;
                  }
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based assignment to dense matrices (default)*******************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a transpose dense matrix-dense matrix multiplication
   //        (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a large transpose
   // dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseBlasKernel<MT3,MT4,MT5> >
      selectBlasAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectLargeAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based assignment to dense matrices*****************************************************
#if BLAZE_BLAS_MODE
   /*! \cond BLAZE_INTERNAL */
   /*!\brief BLAS-based assignment of a transpose dense matrix-dense matrix multiplication
   //        (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function performs the transpose dense matrix-dense matrix multiplication based on the
   // according BLAS functionality.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseBlasKernel<MT3,MT4,MT5> >
      selectBlasAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      typedef ElementType_<MT3>  ET;

      if( IsTriangular<MT4>::value ) {
         assign( C, B );
         trmm( C, A, CblasLeft, ( IsLower<MT4>::value )?( CblasLower ):( CblasUpper ), ET(1) );
      }
      else if( IsTriangular<MT5>::value ) {
         assign( C, A );
         trmm( C, B, CblasRight, ( IsLower<MT5>::value )?( CblasLower ):( CblasUpper ), ET(1) );
      }
      else {
         gemm( C, A, B, ET(1), ET(0) );
      }
   }
   /*! \endcond */
#endif
   //**********************************************************************************************

   //**Assignment to sparse matrices***************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a transpose dense matrix-dense matrix multiplication to a sparse matrix
   //        (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side sparse matrix.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a transpose dense matrix-
   // dense matrix multiplication expression to a sparse matrix.
   */
   template< typename MT  // Type of the target sparse matrix
           , bool SO >    // Storage order of the target sparse matrix
   friend inline void assign( SparseMatrix<MT,SO>& lhs, const TDMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      typedef IfTrue_< SO, ResultType, OppositeType >  TmpType;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( OppositeType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( OppositeType );
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
   /*!\brief Addition assignment of a transpose dense matrix-dense matrix multiplication to a
   //        dense matrix (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a transpose dense
   // matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline void addAssign( DenseMatrix<MT,SO>& lhs, const TDMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || rhs.lhs_.columns() == 0UL ) {
         return;
      }

      LT A( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense matrix operand
      RT B( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      TDMatDMatMultExpr::selectAddAssignKernel( ~lhs, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense matrices (kernel selection)************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for an addition assignment of a transpose dense matrix-dense
   //        matrix multiplication to a dense matrix (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline void selectAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      if( ( IsDiagonal<MT4>::value && IsDiagonal<MT5>::value ) ||
          ( C.rows() * C.columns() < TDMATDMATMULT_THRESHOLD ) )
         selectSmallAddAssignKernel( C, A, B );
      else
         selectBlasAddAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to row-major dense matrices (general/general)*******************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a general transpose dense matrix-general dense matrix
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default addition assignment of a general transpose dense
   // matrix-general dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, Not< IsDiagonal<MT5> > > >
      selectDefaultAddAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t kbegin( ( IsUpper<MT4>::value )
                              ?( IsStrictlyUpper<MT4>::value ? i+1UL : i )
                              :( 0UL ) );
         const size_t kend( ( IsLower<MT4>::value )
                            ?( IsStrictlyLower<MT4>::value ? i : i+1UL )
                            :( K ) );
         BLAZE_INTERNAL_ASSERT( kbegin <= kend, "Invalid loop indices detected" );

         for( size_t k=kbegin; k<kend; ++k )
         {
            const size_t jbegin( ( IsUpper<MT5>::value )
                                 ?( IsStrictlyUpper<MT5>::value ? k+1UL : k )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT5>::value )
                               ?( IsStrictlyLower<MT5>::value ? k : k+1UL )
                               :( N ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jnum( jend - jbegin );
            const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

            for( size_t j=jbegin; j<jpos; j+=2UL ) {
               (~C)(i,j    ) += A(i,k) * B(k,j    );
               (~C)(i,j+1UL) += A(i,k) * B(k,j+1UL);
            }
            if( jpos < jend ) {
               (~C)(i,jpos) += A(i,k) * B(k,jpos);
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to column-major dense matrices (general/general)****************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a general transpose dense matrix-general dense matrix
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default addition assignment of a general transpose dense
   // matrix-dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, Not< IsDiagonal<MT5> > > >
      selectDefaultAddAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      for( size_t j=0UL; j<N; ++j )
      {
         const size_t kbegin( ( IsLower<MT5>::value )
                              ?( IsStrictlyLower<MT5>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t kend( ( IsUpper<MT5>::value )
                            ?( IsStrictlyUpper<MT5>::value ? j : j+1UL )
                            :( K ) );
         BLAZE_INTERNAL_ASSERT( kbegin <= kend, "Invalid loop indices detected" );

         for( size_t k=kbegin; k<kend; ++k )
         {
            const size_t ibegin( ( IsLower<MT4>::value )
                                 ?( IsStrictlyLower<MT4>::value ? k+1UL : k )
                                 :( 0UL ) );
            const size_t iend( ( IsUpper<MT4>::value )
                               ?( IsStrictlyUpper<MT4>::value ? k : k+1UL )
                               :( M ) );
            BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

            const size_t inum( iend - ibegin );
            const size_t ipos( ibegin + ( inum & size_t(-2) ) );

            for( size_t i=ibegin; i<ipos; i+=2UL ) {
               (~C)(i    ,j) += A(i    ,k) * B(k,j);
               (~C)(i+1UL,j) += A(i+1UL,k) * B(k,j);
            }
            if( ipos < iend ) {
               (~C)(ipos,j) += A(ipos,k) * B(k,j);
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to row-major dense matrices (general/diagonal)******************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a general transpose dense matrix-diagonal dense matrix
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default addition assignment of a general transpose dense
   // matrix-diagonal dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, IsDiagonal<MT5> > >
      selectDefaultAddAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      const size_t block( BLOCK_SIZE );

      for( size_t ii=0UL; ii<M; ii+=block ) {
         const size_t iend( min( M, ii+block ) );
         for( size_t jj=0UL; jj<N; jj+=block ) {
            const size_t jend( min( N, jj+block ) );
            for( size_t i=ii; i<iend; ++i )
            {
               const size_t jbegin( ( IsUpper<MT4>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), jj ) )
                                    :( jj ) );
               const size_t jpos( ( IsLower<MT4>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i : i+1UL ), jend ) )
                                  :( jend ) );

               for( size_t j=jbegin; j<jpos; ++j ) {
                  (~C)(i,j) += A(i,j) * B(j,j);
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to column-major dense matrices (general/diagonal)***************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a general transpose dense matrix-diagonal dense matrix
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default addition assignment of a general transpose dense
   // matrix-diagonal dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, IsDiagonal<MT5> > >
      selectDefaultAddAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t j=0UL; j<N; ++j )
      {
         const size_t ibegin( ( IsLower<MT4>::value )
                              ?( IsStrictlyLower<MT4>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT4>::value )
                            ?( IsStrictlyUpper<MT4>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t inum( iend - ibegin );
         const size_t ipos( ibegin + ( inum & size_t(-2) ) );

         for( size_t i=ibegin; i<ipos; i+=2UL ) {
            (~C)(i    ,j) += A(i    ,j) * B(j,j);
            (~C)(i+1UL,j) += A(i+1UL,j) * B(j,j);
         }
         if( ipos < iend ) {
            (~C)(ipos,j) += A(ipos,j) * B(j,j);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to row-major dense matrices (diagonal/general)******************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a diagonal transpose dense matrix-general dense matrix
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default addition assignment of a diagonal transpose dense
   // matrix-general dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< IsDiagonal<MT4>, Not< IsDiagonal<MT5> > > >
      selectDefaultAddAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper<MT5>::value )
                              ?( IsStrictlyUpper<MT5>::value ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT5>::value )
                            ?( IsStrictlyLower<MT5>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jnum( jend - jbegin );
         const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

         for( size_t j=jbegin; j<jpos; j+=2UL ) {
            (~C)(i,j    ) += A(i,i) * B(i,j    );
            (~C)(i,j+1UL) += A(i,i) * B(i,j+1UL);
         }
         if( jpos < jend ) {
            (~C)(i,jpos) += A(i,i) * B(i,jpos);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to column-major dense matrices (diagonal/general)***************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a diagonal transpose dense matrix-general dense matrix
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default addition assignment of a diagonal transpose dense
   // matrix-general dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< IsDiagonal<MT4>, Not< IsDiagonal<MT5> > > >
      selectDefaultAddAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      const size_t block( BLOCK_SIZE );

      for( size_t jj=0UL; jj<N; jj+=block ) {
         const size_t jend( min( N, jj+block ) );
         for( size_t ii=0UL; ii<M; ii+=block ) {
            const size_t iend( min( M, ii+block ) );
            for( size_t j=jj; j<jend; ++j )
            {
               const size_t ibegin( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyLower<MT5>::value ? j+1UL : j ), ii ) )
                                    :( ii ) );
               const size_t ipos( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyUpper<MT5>::value ? j : j+1UL ), iend ) )
                                  :( iend ) );

               for( size_t i=ibegin; i<ipos; ++i ) {
                  (~C)(i,j) += A(i,i) * B(i,j);
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense matrices (diagonal/diagonal)***************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a diagonal transpose dense matrix-diagonal dense
   //        matrix multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default addition assignment of a diagonal transpose dense
   // matrix-diagonal dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< IsDiagonal<MT4>, IsDiagonal<MT5> > >
      selectDefaultAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      for( size_t i=0UL; i<A.rows(); ++i ) {
         C(i,i) += A(i,i) * B(i,i);
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense matrices (small matrices)******************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a small transpose dense matrix-dense matrix
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a transpose
   // dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectSmallAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectDefaultAddAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default addition assignment to row-major dense matrices (small matrices)*********
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default addition assignment of a small transpose dense matrix-dense matrix
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default addition assignment of a transpose dense
   // matrix-dense matrix multiplication expression to a row-major dense matrix. This kernel
   // is optimized for small matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectSmallAddAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );

      for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL ) {
         for( size_t i=0UL; i<M; ++i )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i : i+1UL ), j+SIMDSIZE*8UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i : i+1UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*8UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i,j             ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i,j+SIMDSIZE*3UL) );
            SIMDType xmm5( (~C).load(i,j+SIMDSIZE*4UL) );
            SIMDType xmm6( (~C).load(i,j+SIMDSIZE*5UL) );
            SIMDType xmm7( (~C).load(i,j+SIMDSIZE*6UL) );
            SIMDType xmm8( (~C).load(i,j+SIMDSIZE*7UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j             );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE    );
               xmm3 = xmm3 + a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 = xmm4 + a1 * B.load(k,j+SIMDSIZE*3UL);
               xmm5 = xmm5 + a1 * B.load(k,j+SIMDSIZE*4UL);
               xmm6 = xmm6 + a1 * B.load(k,j+SIMDSIZE*5UL);
               xmm7 = xmm7 + a1 * B.load(k,j+SIMDSIZE*6UL);
               xmm8 = xmm8 + a1 * B.load(k,j+SIMDSIZE*7UL);
            }

            (~C).store( i, j             , xmm1 );
            (~C).store( i, j+SIMDSIZE    , xmm2 );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
            (~C).store( i, j+SIMDSIZE*4UL, xmm5 );
            (~C).store( i, j+SIMDSIZE*5UL, xmm6 );
            (~C).store( i, j+SIMDSIZE*6UL, xmm7 );
            (~C).store( i, j+SIMDSIZE*7UL, xmm8 );
         }
      }

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ), j+SIMDSIZE*4UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j             ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i    ,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i    ,j+SIMDSIZE*3UL) );
            SIMDType xmm5( (~C).load(i+1UL,j             ) );
            SIMDType xmm6( (~C).load(i+1UL,j+SIMDSIZE    ) );
            SIMDType xmm7( (~C).load(i+1UL,j+SIMDSIZE*2UL) );
            SIMDType xmm8( (~C).load(i+1UL,j+SIMDSIZE*3UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a1 * b2;
               xmm3 = xmm3 + a1 * b3;
               xmm4 = xmm4 + a1 * b4;
               xmm5 = xmm5 + a2 * b1;
               xmm6 = xmm6 + a2 * b2;
               xmm7 = xmm7 + a2 * b3;
               xmm8 = xmm8 + a2 * b4;
            }

            (~C).store( i    , j             , xmm1 );
            (~C).store( i    , j+SIMDSIZE    , xmm2 );
            (~C).store( i    , j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i    , j+SIMDSIZE*3UL, xmm4 );
            (~C).store( i+1UL, j             , xmm5 );
            (~C).store( i+1UL, j+SIMDSIZE    , xmm6 );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm7 );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, xmm8 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i,j             ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i,j+SIMDSIZE*3UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j             );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE    );
               xmm3 = xmm3 + a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 = xmm4 + a1 * B.load(k,j+SIMDSIZE*3UL);
            }

            (~C).store( i, j             , xmm1 );
            (~C).store( i, j+SIMDSIZE    , xmm2 );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
         }
      }

      for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j         ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE) );
            SIMDType xmm3( (~C).load(i+1UL,j         ) );
            SIMDType xmm4( (~C).load(i+1UL,j+SIMDSIZE) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a1 * b2;
               xmm3 = xmm3 + a2 * b1;
               xmm4 = xmm4 + a2 * b2;
            }

            (~C).store( i    , j         , xmm1 );
            (~C).store( i    , j+SIMDSIZE, xmm2 );
            (~C).store( i+1UL, j         , xmm3 );
            (~C).store( i+1UL, j+SIMDSIZE, xmm4 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i,j         ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j         );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE);
            }

            (~C).store( i, j         , xmm1 );
            (~C).store( i, j+SIMDSIZE, xmm2 );
         }
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL )
                               :( K ) );

            SIMDType xmm1( (~C).load(i    ,j) );
            SIMDType xmm2( (~C).load(i+1UL,j) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 = xmm1 + set( A(i    ,k) ) * b1;
               xmm2 = xmm2 + set( A(i+1UL,k) ) * b1;
            }

            (~C).store( i    , j, xmm1 );
            (~C).store( i+1UL, j, xmm2 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );

            SIMDType xmm1( (~C).load(i,j) );

            for( size_t k=kbegin; k<K; ++k ) {
               xmm1 = xmm1 + set( A(i,k) ) * B.load(k,j);
            }

            (~C).store( i, j, xmm1 );
         }
      }

      for( ; remainder && j<N; ++j )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL )
                               :( K ) );

            ElementType value1( (~C)(i    ,j) );
            ElementType value2( (~C)(i+1UL,j) );;

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 += A(i    ,k) * B(k,j);
               value2 += A(i+1UL,k) * B(k,j);
            }

            (~C)(i    ,j) = value1;
            (~C)(i+1UL,j) = value2;
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );

            ElementType value( (~C)(i,j) );

            for( size_t k=kbegin; k<K; ++k ) {
               value += A(i,k) * B(k,j);
            }

            (~C)(i,j) = value;
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default addition assignment to column-major dense matrices (small matrices)******
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default addition assignment of a small transpose dense matrix-dense matrix
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default addition assignment of a transpose dense
   // matrix-dense matrix multiplication expression to a column-major dense matrix. This kernel
   // is optimized for small matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectSmallAddAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT4>::value );

      const size_t ipos( remainder ? ( M & size_t(-SIMDSIZE) ) : M );
      BLAZE_INTERNAL_ASSERT( !remainder || ( M - ( M % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

      size_t i( 0UL );

      for( ; (i+SIMDSIZE*7UL) < ipos; i+=SIMDSIZE*8UL ) {
         for( size_t j=0UL; j<N; ++j )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*8UL, K, ( IsStrictlyUpper<MT5>::value ? j : j+1UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j : j+1UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*8UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i             ,j) );
            SIMDType xmm2( (~C).load(i+SIMDSIZE    ,j) );
            SIMDType xmm3( (~C).load(i+SIMDSIZE*2UL,j) );
            SIMDType xmm4( (~C).load(i+SIMDSIZE*3UL,j) );
            SIMDType xmm5( (~C).load(i+SIMDSIZE*4UL,j) );
            SIMDType xmm6( (~C).load(i+SIMDSIZE*5UL,j) );
            SIMDType xmm7( (~C).load(i+SIMDSIZE*6UL,j) );
            SIMDType xmm8( (~C).load(i+SIMDSIZE*7UL,j) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i             ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE    ,k) * b1;
               xmm3 = xmm3 + A.load(i+SIMDSIZE*2UL,k) * b1;
               xmm4 = xmm4 + A.load(i+SIMDSIZE*3UL,k) * b1;
               xmm5 = xmm5 + A.load(i+SIMDSIZE*4UL,k) * b1;
               xmm6 = xmm6 + A.load(i+SIMDSIZE*5UL,k) * b1;
               xmm7 = xmm7 + A.load(i+SIMDSIZE*6UL,k) * b1;
               xmm8 = xmm8 + A.load(i+SIMDSIZE*7UL,k) * b1;
            }

            (~C).store( i             , j, xmm1 );
            (~C).store( i+SIMDSIZE    , j, xmm2 );
            (~C).store( i+SIMDSIZE*2UL, j, xmm3 );
            (~C).store( i+SIMDSIZE*3UL, j, xmm4 );
            (~C).store( i+SIMDSIZE*4UL, j, xmm5 );
            (~C).store( i+SIMDSIZE*5UL, j, xmm6 );
            (~C).store( i+SIMDSIZE*6UL, j, xmm7 );
            (~C).store( i+SIMDSIZE*7UL, j, xmm8 );
         }
      }

      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*4UL, K, ( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i             ,j    ) );
            SIMDType xmm2( (~C).load(i+SIMDSIZE    ,j    ) );
            SIMDType xmm3( (~C).load(i+SIMDSIZE*2UL,j    ) );
            SIMDType xmm4( (~C).load(i+SIMDSIZE*3UL,j    ) );
            SIMDType xmm5( (~C).load(i             ,j+1UL) );
            SIMDType xmm6( (~C).load(i+SIMDSIZE    ,j+1UL) );
            SIMDType xmm7( (~C).load(i+SIMDSIZE*2UL,j+1UL) );
            SIMDType xmm8( (~C).load(i+SIMDSIZE*3UL,j+1UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i             ,k) );
               const SIMDType a2( A.load(i+SIMDSIZE    ,k) );
               const SIMDType a3( A.load(i+SIMDSIZE*2UL,k) );
               const SIMDType a4( A.load(i+SIMDSIZE*3UL,k) );
               const SIMDType b1( set( B(k,j    ) ) );
               const SIMDType b2( set( B(k,j+1UL) ) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a2 * b1;
               xmm3 = xmm3 + a3 * b1;
               xmm4 = xmm4 + a4 * b1;
               xmm5 = xmm5 + a1 * b2;
               xmm6 = xmm6 + a2 * b2;
               xmm7 = xmm7 + a3 * b2;
               xmm8 = xmm8 + a4 * b2;
            }

            (~C).store( i             , j    , xmm1 );
            (~C).store( i+SIMDSIZE    , j    , xmm2 );
            (~C).store( i+SIMDSIZE*2UL, j    , xmm3 );
            (~C).store( i+SIMDSIZE*3UL, j    , xmm4 );
            (~C).store( i             , j+1UL, xmm5 );
            (~C).store( i+SIMDSIZE    , j+1UL, xmm6 );
            (~C).store( i+SIMDSIZE*2UL, j+1UL, xmm7 );
            (~C).store( i+SIMDSIZE*3UL, j+1UL, xmm8 );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i             ,j) );
            SIMDType xmm2( (~C).load(i+SIMDSIZE    ,j) );
            SIMDType xmm3( (~C).load(i+SIMDSIZE*2UL,j) );
            SIMDType xmm4( (~C).load(i+SIMDSIZE*3UL,j) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i             ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE    ,k) * b1;
               xmm3 = xmm3 + A.load(i+SIMDSIZE*2UL,k) * b1;
               xmm4 = xmm4 + A.load(i+SIMDSIZE*3UL,k) * b1;
            }

            (~C).store( i             , j, xmm1 );
            (~C).store( i+SIMDSIZE    , j, xmm2 );
            (~C).store( i+SIMDSIZE*2UL, j, xmm3 );
            (~C).store( i+SIMDSIZE*3UL, j, xmm4 );
         }
      }

      for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*2UL, K, ( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i         ,j    ) );
            SIMDType xmm2( (~C).load(i+SIMDSIZE,j    ) );
            SIMDType xmm3( (~C).load(i         ,j+1UL) );
            SIMDType xmm4( (~C).load(i+SIMDSIZE,j+1UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i         ,k) );
               const SIMDType a2( A.load(i+SIMDSIZE,k) );
               const SIMDType b1( set( B(k,j    ) ) );
               const SIMDType b2( set( B(k,j+1UL) ) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a2 * b1;
               xmm3 = xmm3 + a1 * b2;
               xmm4 = xmm4 + a2 * b2;
            }

            (~C).store( i         , j    , xmm1 );
            (~C).store( i+SIMDSIZE, j    , xmm2 );
            (~C).store( i         , j+1UL, xmm3 );
            (~C).store( i+SIMDSIZE, j+1UL, xmm4 );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i         ,j) );
            SIMDType xmm2( (~C).load(i+SIMDSIZE,j) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i         ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE,k) * b1;
            }

            (~C).store( i         , j, xmm1 );
            (~C).store( i+SIMDSIZE, j, xmm2 );
         }
      }

      for( ; i<ipos; i+=SIMDSIZE )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL )
                               :( K ) );

            SIMDType xmm1( (~C).load(i,j    ) );
            SIMDType xmm2( (~C).load(i,j+1UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i,k) );
               xmm1 = xmm1 + a1 * set( B(k,j    ) );
               xmm2 = xmm2 + a1 * set( B(k,j+1UL) );
            }

            (~C).store( i, j    , xmm1 );
            (~C).store( i, j+1UL, xmm2 );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );

            SIMDType xmm1( (~C).load(i,j) );

            for( size_t k=kbegin; k<K; ++k ) {
               xmm1 = xmm1 + A.load(i,k) * set( B(k,j) );
            }

            (~C).store( i, j, xmm1 );
         }
      }

      for( ; remainder && i<M; ++i )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL )
                               :( K ) );

            ElementType value1( (~C)(i,j    ) );
            ElementType value2( (~C)(i,j+1UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 += A(i,k) * B(k,j    );
               value2 += A(i,k) * B(k,j+1UL);
            }

            (~C)(i,j    ) = value1;
            (~C)(i,j+1UL) = value2;
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );

            ElementType value( (~C)(i,j) );

            for( size_t k=kbegin; k<K; ++k ) {
               value += A(i,k) * B(k,j);
            }

            (~C)(i,j) = value;
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense matrices (large matrices)******************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a large transpose dense matrix-dense matrix
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a transpose
   // dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectLargeAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectDefaultAddAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default addition assignment to row-major dense matrices (large matrices)*********
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default addition assignment of a large transpose dense matrix-dense matrix
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default addition assignment of a transpose dense
   // matrix-dense matrix multiplication expression to a row-major dense matrix. This kernel
   // is optimized for large matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectLargeAddAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      for( size_t jj=0UL; jj<N; jj+=DMATDMATMULT_DEFAULT_JBLOCK_SIZE )
      {
         const size_t jend( min( jj+DMATDMATMULT_DEFAULT_JBLOCK_SIZE, N ) );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

         for( size_t ii=0UL; ii<M; ii+=DMATDMATMULT_DEFAULT_IBLOCK_SIZE )
         {
            const size_t iend( min( ii+DMATDMATMULT_DEFAULT_IBLOCK_SIZE, M ) );

            for( size_t kk=0UL; kk<K; kk+=DMATDMATMULT_DEFAULT_KBLOCK_SIZE )
            {
               const size_t ktmp( min( kk+DMATDMATMULT_DEFAULT_KBLOCK_SIZE, K ) );

               size_t j( jj );

               for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
               {
                  const size_t j1( j+SIMDSIZE     );
                  const size_t j2( j+SIMDSIZE*2UL );
                  const size_t j3( j+SIMDSIZE*3UL );

                  size_t i( ii );

                  for( ; (i+2UL) <= iend; i+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+2UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i    ,j ) );
                     SIMDType xmm2( (~C).load(i    ,j1) );
                     SIMDType xmm3( (~C).load(i    ,j2) );
                     SIMDType xmm4( (~C).load(i    ,j3) );
                     SIMDType xmm5( (~C).load(i+1UL,j ) );
                     SIMDType xmm6( (~C).load(i+1UL,j1) );
                     SIMDType xmm7( (~C).load(i+1UL,j2) );
                     SIMDType xmm8( (~C).load(i+1UL,j3) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        const SIMDType b3( B.load(k,j2) );
                        const SIMDType b4( B.load(k,j3) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a1 * b3;
                        xmm4 = xmm4 + a1 * b4;
                        xmm5 = xmm5 + a2 * b1;
                        xmm6 = xmm6 + a2 * b2;
                        xmm7 = xmm7 + a2 * b3;
                        xmm8 = xmm8 + a2 * b4;
                     }

                     (~C).store( i    , j , xmm1 );
                     (~C).store( i    , j1, xmm2 );
                     (~C).store( i    , j2, xmm3 );
                     (~C).store( i    , j3, xmm4 );
                     (~C).store( i+1UL, j , xmm5 );
                     (~C).store( i+1UL, j1, xmm6 );
                     (~C).store( i+1UL, j2, xmm7 );
                     (~C).store( i+1UL, j3, xmm8 );
                  }

                  if( i < iend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i,j ) );
                     SIMDType xmm2( (~C).load(i,j1) );
                     SIMDType xmm3( (~C).load(i,j2) );
                     SIMDType xmm4( (~C).load(i,j3) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j );
                        xmm2 = xmm2 + a1 * B.load(k,j1);
                        xmm3 = xmm3 + a1 * B.load(k,j2);
                        xmm4 = xmm4 + a1 * B.load(k,j3);
                     }

                     (~C).store( i, j , xmm1 );
                     (~C).store( i, j1, xmm2 );
                     (~C).store( i, j2, xmm3 );
                     (~C).store( i, j3, xmm4 );
                  }
               }

               for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
               {
                  const size_t j1( j+SIMDSIZE );

                  size_t i( ii );

                  for( ; (i+4UL) <= iend; i+=4UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+4UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i    ,j ) );
                     SIMDType xmm2( (~C).load(i    ,j1) );
                     SIMDType xmm3( (~C).load(i+1UL,j ) );
                     SIMDType xmm4( (~C).load(i+1UL,j1) );
                     SIMDType xmm5( (~C).load(i+2UL,j ) );
                     SIMDType xmm6( (~C).load(i+2UL,j1) );
                     SIMDType xmm7( (~C).load(i+3UL,j ) );
                     SIMDType xmm8( (~C).load(i+3UL,j1) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType a3( set( A(i+2UL,k) ) );
                        const SIMDType a4( set( A(i+3UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a2 * b1;
                        xmm4 = xmm4 + a2 * b2;
                        xmm5 = xmm5 + a3 * b1;
                        xmm6 = xmm6 + a3 * b2;
                        xmm7 = xmm7 + a4 * b1;
                        xmm8 = xmm8 + a4 * b2;
                     }

                     (~C).store( i    , j , xmm1 );
                     (~C).store( i    , j1, xmm2 );
                     (~C).store( i+1UL, j , xmm3 );
                     (~C).store( i+1UL, j1, xmm4 );
                     (~C).store( i+2UL, j , xmm5 );
                     (~C).store( i+2UL, j1, xmm6 );
                     (~C).store( i+3UL, j , xmm7 );
                     (~C).store( i+3UL, j1, xmm8 );
                  }

                  for( ; (i+2UL) <= iend; i+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+2UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i    ,j ) );
                     SIMDType xmm2( (~C).load(i    ,j1) );
                     SIMDType xmm3( (~C).load(i+1UL,j ) );
                     SIMDType xmm4( (~C).load(i+1UL,j1) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a2 * b1;
                        xmm4 = xmm4 + a2 * b2;
                     }

                     (~C).store( i    , j , xmm1 );
                     (~C).store( i    , j1, xmm2 );
                     (~C).store( i+1UL, j , xmm3 );
                     (~C).store( i+1UL, j1, xmm4 );
                  }

                  if( i < iend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i,j ) );
                     SIMDType xmm2( (~C).load(i,j1) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j );
                        xmm2 = xmm2 + a1 * B.load(k,j1);
                     }

                     (~C).store( i, j , xmm1 );
                     (~C).store( i, j1, xmm2 );
                  }
               }

               for( ; j<jpos; j+=SIMDSIZE )
               {
                  for( size_t i=ii; i<iend; ++i )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j);
                     }

                     (~C).store( i, j, xmm1 );
                  }
               }

               for( ; remainder && j<jend; ++j )
               {
                  for( size_t i=ii; i<iend; ++i )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+1UL, ktmp ) ):( ktmp ) ) );

                     ElementType value( (~C)(i,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        value += A(i,k) * B(k,j);
                     }

                     (~C)(i,j) = value;
                  }
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default addition assignment to column-major dense matrices (large matrices)******
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default addition assignment of a large transpose dense matrix-dense matrix
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default addition assignment of a transpose dense
   // matrix-dense matrix multiplication expression to a column-major dense matrix. This kernel
   // is optimized for large matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectLargeAddAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT4>::value );

      for( size_t ii=0UL; ii<M; ii+=TDMATTDMATMULT_DEFAULT_IBLOCK_SIZE )
      {
         const size_t iend( min( ii+TDMATTDMATMULT_DEFAULT_IBLOCK_SIZE, M ) );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

         for( size_t jj=0UL; jj<N; jj+=TDMATTDMATMULT_DEFAULT_JBLOCK_SIZE )
         {
            const size_t jend( min( jj+TDMATTDMATMULT_DEFAULT_JBLOCK_SIZE, N ) );

            for( size_t kk=0UL; kk<K; kk+=TDMATTDMATMULT_DEFAULT_KBLOCK_SIZE )
            {
               const size_t ktmp( min( kk+TDMATTDMATMULT_DEFAULT_KBLOCK_SIZE, K ) );

               size_t i( ii );

               for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL )
               {
                  const size_t i1( i+SIMDSIZE     );
                  const size_t i2( i+SIMDSIZE*2UL );
                  const size_t i3( i+SIMDSIZE*3UL );

                  size_t j( jj );

                  for( ; (j+2UL) <= jend; j+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+2UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j    ) );
                     SIMDType xmm2( (~C).load(i1,j    ) );
                     SIMDType xmm3( (~C).load(i2,j    ) );
                     SIMDType xmm4( (~C).load(i3,j    ) );
                     SIMDType xmm5( (~C).load(i ,j+1UL) );
                     SIMDType xmm6( (~C).load(i1,j+1UL) );
                     SIMDType xmm7( (~C).load(i2,j+1UL) );
                     SIMDType xmm8( (~C).load(i3,j+1UL) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType a3( A.load(i2,k) );
                        const SIMDType a4( A.load(i3,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a3 * b1;
                        xmm4 = xmm4 + a4 * b1;
                        xmm5 = xmm5 + a1 * b2;
                        xmm6 = xmm6 + a2 * b2;
                        xmm7 = xmm7 + a3 * b2;
                        xmm8 = xmm8 + a4 * b2;
                     }

                     (~C).store( i , j    , xmm1 );
                     (~C).store( i1, j    , xmm2 );
                     (~C).store( i2, j    , xmm3 );
                     (~C).store( i3, j    , xmm4 );
                     (~C).store( i , j+1UL, xmm5 );
                     (~C).store( i1, j+1UL, xmm6 );
                     (~C).store( i2, j+1UL, xmm7 );
                     (~C).store( i3, j+1UL, xmm8 );
                  }

                  if( j < jend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j) );
                     SIMDType xmm2( (~C).load(i1,j) );
                     SIMDType xmm3( (~C).load(i2,j) );
                     SIMDType xmm4( (~C).load(i3,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i ,k) * b1;
                        xmm2 = xmm2 + A.load(i1,k) * b1;
                        xmm3 = xmm3 + A.load(i2,k) * b1;
                        xmm4 = xmm4 + A.load(i3,k) * b1;
                     }

                     (~C).store( i , j, xmm1 );
                     (~C).store( i1, j, xmm2 );
                     (~C).store( i2, j, xmm3 );
                     (~C).store( i3, j, xmm4 );
                  }
               }

               for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL )
               {
                  const size_t i1( i+SIMDSIZE );

                  size_t j( jj );

                  for( ; (j+4UL) <= jend; j+=4UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+4UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j    ) );
                     SIMDType xmm2( (~C).load(i1,j    ) );
                     SIMDType xmm3( (~C).load(i ,j+1UL) );
                     SIMDType xmm4( (~C).load(i1,j+1UL) );
                     SIMDType xmm5( (~C).load(i ,j+2UL) );
                     SIMDType xmm6( (~C).load(i1,j+2UL) );
                     SIMDType xmm7( (~C).load(i ,j+3UL) );
                     SIMDType xmm8( (~C).load(i1,j+3UL) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        const SIMDType b3( set( B(k,j+2UL) ) );
                        const SIMDType b4( set( B(k,j+3UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a1 * b2;
                        xmm4 = xmm4 + a2 * b2;
                        xmm5 = xmm5 + a1 * b3;
                        xmm6 = xmm6 + a2 * b3;
                        xmm7 = xmm7 + a1 * b4;
                        xmm8 = xmm8 + a2 * b4;
                     }

                     (~C).store( i , j    , xmm1 );
                     (~C).store( i1, j    , xmm2 );
                     (~C).store( i , j+1UL, xmm3 );
                     (~C).store( i1, j+1UL, xmm4 );
                     (~C).store( i , j+2UL, xmm5 );
                     (~C).store( i1, j+2UL, xmm6 );
                     (~C).store( i , j+3UL, xmm7 );
                     (~C).store( i1, j+3UL, xmm8 );
                  }

                  for( ; (j+2UL) <= jend; j+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+2UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j    ) );
                     SIMDType xmm2( (~C).load(i1,j    ) );
                     SIMDType xmm3( (~C).load(i ,j+1UL) );
                     SIMDType xmm4( (~C).load(i1,j+1UL) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a1 * b2;
                        xmm4 = xmm4 + a2 * b2;
                     }

                     (~C).store( i , j    , xmm1 );
                     (~C).store( i1, j    , xmm2 );
                     (~C).store( i , j+1UL, xmm3 );
                     (~C).store( i1, j+1UL, xmm4 );
                  }

                  if( j < jend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j) );
                     SIMDType xmm2( (~C).load(i1,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i ,k) * b1;
                        xmm2 = xmm2 + A.load(i1,k) * b1;
                     }

                     (~C).store( i , j, xmm1 );
                     (~C).store( i1, j, xmm2 );
                  }
               }

               for( ; i<ipos; i+=SIMDSIZE )
               {
                  for( size_t j=jj; j<jend; ++j )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i,k) * b1;
                     }

                     (~C).store( i, j, xmm1 );
                  }
               }

               for( ; remainder && i<iend; ++i )
               {
                  for( size_t j=jj; j<jend; ++j )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+1UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     ElementType value( (~C)(i,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        value += A(i,k) * B(k,j);
                     }

                     (~C)(i,j) = value;
                  }
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense matrices (default)**********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a transpose dense matrix-dense matrix multiplication
   //        (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a large
   // transpose dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseBlasKernel<MT3,MT4,MT5> >
      selectBlasAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectLargeAddAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense matrices********************************************
#if BLAZE_BLAS_MODE
   /*! \cond BLAZE_INTERNAL */
   /*!\brief BLAS-based addition assignment of a transpose dense matrix-dense matrix multiplication
   //        (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function performs the transpose dense matrix-dense matrix multiplication based on the
   // according BLAS functionality.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseBlasKernel<MT3,MT4,MT5> >
      selectBlasAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      typedef ElementType_<MT3>  ET;

      if( IsTriangular<MT4>::value ) {
         ResultType_<MT3> tmp( serial( B ) );
         trmm( tmp, A, CblasLeft, ( IsLower<MT4>::value )?( CblasLower ):( CblasUpper ), ET(1) );
         addAssign( C, tmp );
      }
      else if( IsTriangular<MT5>::value ) {
         ResultType_<MT3> tmp( serial( A ) );
         trmm( tmp, B, CblasRight, ( IsLower<MT5>::value )?( CblasLower ):( CblasUpper ), ET(1) );
         addAssign( C, tmp );
      }
      else {
         gemm( C, A, B, ET(1), ET(1) );
      }
   }
   /*! \endcond */
#endif
   //**********************************************************************************************

   //**Addition assignment to sparse matrices******************************************************
   // No special implementation for the addition assignment to sparse matrices.
   //**********************************************************************************************

   //**Subtraction assignment to dense matrices****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a transpose dense matrix-dense matrix multiplication to a
   //        dense matrix (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a transpose
   // dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline void subAssign( DenseMatrix<MT,SO>& lhs, const TDMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || rhs.lhs_.columns() == 0UL ) {
         return;
      }

      LT A( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense matrix operand
      RT B( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      TDMatDMatMultExpr::selectSubAssignKernel( ~lhs, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to dense matrices (kernel selection)*********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for a subtraction assignment of a transpose dense matrix-
   //        dense matrix multiplication to a dense matrix (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline void selectSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      if( ( IsDiagonal<MT4>::value && IsDiagonal<MT5>::value ) ||
          ( C.rows() * C.columns() < TDMATDMATMULT_THRESHOLD ) )
         selectSmallSubAssignKernel( C, A, B );
      else
         selectBlasSubAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to row-major dense matrices (general/general)****************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a general transpose dense matrix-general dense
   //        matrix multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default subtraction assignment of a general transpose dense
   // matrix-general dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, Not< IsDiagonal<MT5> > > >
      selectDefaultSubAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t kbegin( ( IsUpper<MT4>::value )
                              ?( IsStrictlyUpper<MT4>::value ? i+1UL : i )
                              :( 0UL ) );
         const size_t kend( ( IsLower<MT4>::value )
                            ?( IsStrictlyLower<MT4>::value ? i : i+1UL )
                            :( K ) );
         BLAZE_INTERNAL_ASSERT( kbegin <= kend, "Invalid loop indices detected" );

         for( size_t k=kbegin; k<kend; ++k )
         {
            const size_t jbegin( ( IsUpper<MT5>::value )
                                 ?( IsStrictlyUpper<MT5>::value ? k+1UL : k )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT5>::value )
                               ?( IsStrictlyLower<MT5>::value ? k : k+1UL )
                               :( N ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jnum( jend - jbegin );
            const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

            for( size_t j=jbegin; j<jpos; j+=2UL ) {
               (~C)(i,j    ) -= A(i,k) * B(k,j    );
               (~C)(i,j+1UL) -= A(i,k) * B(k,j+1UL);
            }
            if( jpos < jend ) {
               (~C)(i,jpos) -= A(i,k) * B(k,jpos);
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to column-major dense matrices (general/general)*************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a general transpose dense matrix-general dense
   //        matrix multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default subtraction assignment of a general transpose dense
   // matrix-general dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, Not< IsDiagonal<MT5> > > >
      selectDefaultSubAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      for( size_t j=0UL; j<N; ++j )
      {
         const size_t kbegin( ( IsLower<MT5>::value )
                              ?( IsStrictlyLower<MT5>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t kend( ( IsUpper<MT5>::value )
                            ?( IsStrictlyUpper<MT5>::value ? j : j+1UL )
                            :( K ) );
         BLAZE_INTERNAL_ASSERT( kbegin <= kend, "Invalid loop indices detected" );

         for( size_t k=kbegin; k<kend; ++k )
         {
            const size_t ibegin( ( IsLower<MT4>::value )
                                 ?( IsStrictlyLower<MT4>::value ? k+1UL : k )
                                 :( 0UL ) );
            const size_t iend( ( IsUpper<MT4>::value )
                               ?( IsStrictlyUpper<MT4>::value ? k : k+1UL )
                               :( M ) );
            BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

            const size_t inum( iend - ibegin );
            const size_t ipos( ibegin + ( inum & size_t(-2) ) );

            for( size_t i=ibegin; i<ipos; i+=2UL ) {
               (~C)(i    ,j) -= A(i    ,k) * B(k,j);
               (~C)(i+1UL,j) -= A(i+1UL,k) * B(k,j);
            }
            if( ipos < iend ) {
               (~C)(ipos,j) -= A(ipos,k) * B(k,j);
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to row-major dense matrices (general/diagonal)***************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a general transpose dense matrix-diagonal dense
   //        matrix multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default subtraction assignment of a general transpose dense
   // matrix-diagonal dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, IsDiagonal<MT5> > >
      selectDefaultSubAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      const size_t block( BLOCK_SIZE );

      for( size_t ii=0UL; ii<M; ii+=block ) {
         const size_t iend( min( M, ii+block ) );
         for( size_t jj=0UL; jj<N; jj+=block ) {
            const size_t jend( min( N, jj+block ) );
            for( size_t i=ii; i<iend; ++i )
            {
               const size_t jbegin( ( IsUpper<MT4>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), jj ) )
                                    :( jj ) );
               const size_t jpos( ( IsLower<MT4>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i : i+1UL ), jend ) )
                                  :( jend ) );

               for( size_t j=jbegin; j<jpos; ++j ) {
                  (~C)(i,j) -= A(i,j) * B(j,j);
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to column-major dense matrices (general/diagonal)************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a general transpose dense matrix-diagonal dense
   //        matrix multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default subtraction assignment of a general transpose dense
   // matrix-diagonal dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, IsDiagonal<MT5> > >
      selectDefaultSubAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t j=0UL; j<N; ++j )
      {
         const size_t ibegin( ( IsLower<MT4>::value )
                              ?( IsStrictlyLower<MT4>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT4>::value )
                            ?( IsStrictlyUpper<MT4>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t inum( iend - ibegin );
         const size_t ipos( ibegin + ( inum & size_t(-2) ) );

         for( size_t i=ibegin; i<ipos; i+=2UL ) {
            (~C)(i    ,j) -= A(i    ,j) * B(j,j);
            (~C)(i+1UL,j) -= A(i+1UL,j) * B(j,j);
         }
         if( ipos < iend ) {
            (~C)(ipos,j) -= A(ipos,j) * B(j,j);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to row-major dense matrices (diagonal/general)***************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a diagonal transpose dense matrix-general dense
   //        matrix multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default subtraction assignment of a diagonal transpose dense
   // matrix-general dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< IsDiagonal<MT4>, Not< IsDiagonal<MT5> > > >
      selectDefaultSubAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper<MT5>::value )
                              ?( IsStrictlyUpper<MT5>::value ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT5>::value )
                            ?( IsStrictlyLower<MT5>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jnum( jend - jbegin );
         const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

         for( size_t j=jbegin; j<jpos; j+=2UL ) {
            (~C)(i,j    ) -= A(i,i) * B(i,j    );
            (~C)(i,j+1UL) -= A(i,i) * B(i,j+1UL);
         }
         if( jpos < jend ) {
            (~C)(i,jpos) -= A(i,i) * B(i,jpos);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to column-major dense matrices (diagonal/general)************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a diagonal transpose dense matrix-general dense matrix
   //        multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default subtraction assignment of a diagonal transpose dense
   // matrix-general dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< IsDiagonal<MT4>, Not< IsDiagonal<MT5> > > >
      selectDefaultSubAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      const size_t block( BLOCK_SIZE );

      for( size_t jj=0UL; jj<N; jj+=block ) {
         const size_t jend( min( N, jj+block ) );
         for( size_t ii=0UL; ii<M; ii+=block ) {
            const size_t iend( min( M, ii+block ) );
            for( size_t j=jj; j<jend; ++j )
            {
               const size_t ibegin( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyLower<MT5>::value ? j+1UL : j ), ii ) )
                                    :( ii ) );
               const size_t ipos( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyUpper<MT5>::value ? j : j+1UL ), iend ) )
                                  :( iend ) );

               for( size_t i=ibegin; i<ipos; ++i ) {
                  (~C)(i,j) -= A(i,i) * B(i,j);
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense matrices (diagonal/diagonal)************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a diagonal transpose dense matrix-diagonal dense
   //        matrix multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default subtraction assignment of a diagonal transpose dense
   // matrix-diagonal dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< And< IsDiagonal<MT4>, IsDiagonal<MT5> > >
      selectDefaultSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      for( size_t i=0UL; i<A.rows(); ++i ) {
         C(i,i) -= A(i,i) * B(i,i);
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense matrices (small matrices)***************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a small transpose dense matrix-dense matrix
   //        multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a
   // transpose dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectSmallSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectDefaultSubAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to row-major dense matrices (small matrices)******
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default subtraction assignment of a small transpose dense matrix-dense
   //        matrix multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment of a transpose
   // dense matrix-dense matrix multiplication expression to a row-major dense matrix. This
   // kernel is optimized for small matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectSmallSubAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );

      for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL ) {
         for( size_t i=0UL; i<M; ++i )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i : i+1UL ), j+SIMDSIZE*8UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i : i+1UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*8UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i,j             ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i,j+SIMDSIZE*3UL) );
            SIMDType xmm5( (~C).load(i,j+SIMDSIZE*4UL) );
            SIMDType xmm6( (~C).load(i,j+SIMDSIZE*5UL) );
            SIMDType xmm7( (~C).load(i,j+SIMDSIZE*6UL) );
            SIMDType xmm8( (~C).load(i,j+SIMDSIZE*7UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 - a1 * B.load(k,j             );
               xmm2 = xmm2 - a1 * B.load(k,j+SIMDSIZE    );
               xmm3 = xmm3 - a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 = xmm4 - a1 * B.load(k,j+SIMDSIZE*3UL);
               xmm5 = xmm5 - a1 * B.load(k,j+SIMDSIZE*4UL);
               xmm6 = xmm6 - a1 * B.load(k,j+SIMDSIZE*5UL);
               xmm7 = xmm7 - a1 * B.load(k,j+SIMDSIZE*6UL);
               xmm8 = xmm8 - a1 * B.load(k,j+SIMDSIZE*7UL);
            }

            (~C).store( i, j             , xmm1 );
            (~C).store( i, j+SIMDSIZE    , xmm2 );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
            (~C).store( i, j+SIMDSIZE*4UL, xmm5 );
            (~C).store( i, j+SIMDSIZE*5UL, xmm6 );
            (~C).store( i, j+SIMDSIZE*6UL, xmm7 );
            (~C).store( i, j+SIMDSIZE*7UL, xmm8 );
         }
      }

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ), j+SIMDSIZE*4UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j             ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i    ,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i    ,j+SIMDSIZE*3UL) );
            SIMDType xmm5( (~C).load(i+1UL,j             ) );
            SIMDType xmm6( (~C).load(i+1UL,j+SIMDSIZE    ) );
            SIMDType xmm7( (~C).load(i+1UL,j+SIMDSIZE*2UL) );
            SIMDType xmm8( (~C).load(i+1UL,j+SIMDSIZE*3UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               xmm1 = xmm1 - a1 * b1;
               xmm2 = xmm2 - a1 * b2;
               xmm3 = xmm3 - a1 * b3;
               xmm4 = xmm4 - a1 * b4;
               xmm5 = xmm5 - a2 * b1;
               xmm6 = xmm6 - a2 * b2;
               xmm7 = xmm7 - a2 * b3;
               xmm8 = xmm8 - a2 * b4;
            }

            (~C).store( i    , j             , xmm1 );
            (~C).store( i    , j+SIMDSIZE    , xmm2 );
            (~C).store( i    , j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i    , j+SIMDSIZE*3UL, xmm4 );
            (~C).store( i+1UL, j             , xmm5 );
            (~C).store( i+1UL, j+SIMDSIZE    , xmm6 );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm7 );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, xmm8 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i,j             ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i,j+SIMDSIZE*3UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 - a1 * B.load(k,j             );
               xmm2 = xmm2 - a1 * B.load(k,j+SIMDSIZE    );
               xmm3 = xmm3 - a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 = xmm4 - a1 * B.load(k,j+SIMDSIZE*3UL);
            }

            (~C).store( i, j             , xmm1 );
            (~C).store( i, j+SIMDSIZE    , xmm2 );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
         }
      }

      for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j         ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE) );
            SIMDType xmm3( (~C).load(i+1UL,j         ) );
            SIMDType xmm4( (~C).load(i+1UL,j+SIMDSIZE) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 = xmm1 - a1 * b1;
               xmm2 = xmm2 - a1 * b2;
               xmm3 = xmm3 - a2 * b1;
               xmm4 = xmm4 - a2 * b2;
            }

            (~C).store( i    , j         , xmm1 );
            (~C).store( i    , j+SIMDSIZE, xmm2 );
            (~C).store( i+1UL, j         , xmm3 );
            (~C).store( i+1UL, j+SIMDSIZE, xmm4 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i,j         ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 - a1 * B.load(k,j         );
               xmm2 = xmm2 - a1 * B.load(k,j+SIMDSIZE);
            }

            (~C).store( i, j         , xmm1 );
            (~C).store( i, j+SIMDSIZE, xmm2 );
         }
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL )
                               :( K ) );

            SIMDType xmm1( (~C).load(i    ,j) );
            SIMDType xmm2( (~C).load(i+1UL,j) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 = xmm1 - set( A(i    ,k) ) * b1;
               xmm2 = xmm2 - set( A(i+1UL,k) ) * b1;
            }

            (~C).store( i    , j, xmm1 );
            (~C).store( i+1UL, j, xmm2 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );

            SIMDType xmm1( (~C).load(i,j) );

            for( size_t k=kbegin; k<K; ++k ) {
               xmm1 = xmm1 - set( A(i,k) ) * B.load(k,j);
            }

            (~C).store( i, j, xmm1 );
         }
      }

      for( ; remainder && j<N; ++j )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL )
                               :( K ) );

            ElementType value1( (~C)(i    ,j) );
            ElementType value2( (~C)(i+1UL,j) );

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 -= A(i    ,k) * B(k,j);
               value2 -= A(i+1UL,k) * B(k,j);
            }

            (~C)(i    ,j) = value1;
            (~C)(i+1UL,j) = value2;
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );

            ElementType value( (~C)(i,j) );

            for( size_t k=kbegin; k<K; ++k ) {
               value -= A(i,k) * B(k,j);
            }

            (~C)(i,j) = value;
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to column-major dense matrices (small matrices)***
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default subtraction assignment of a small transpose dense matrix-dense
   //        matrix multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment of a transpose
   // dense matrix-dense matrix multiplication expression to a column-major dense matrix.
   // This kernel is optimized for small matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectSmallSubAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT4>::value );

      const size_t ipos( remainder ? ( M & size_t(-SIMDSIZE) ) : M );
      BLAZE_INTERNAL_ASSERT( !remainder || ( M - ( M % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

      size_t i( 0UL );

      for( ; (i+SIMDSIZE*7UL) < ipos; i+=SIMDSIZE*8UL ) {
         for( size_t j=0UL; j<N; ++j )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*8UL, K, ( IsStrictlyUpper<MT5>::value ? j : j+1UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j : j+1UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*8UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i             ,j) );
            SIMDType xmm2( (~C).load(i+SIMDSIZE    ,j) );
            SIMDType xmm3( (~C).load(i+SIMDSIZE*2UL,j) );
            SIMDType xmm4( (~C).load(i+SIMDSIZE*3UL,j) );
            SIMDType xmm5( (~C).load(i+SIMDSIZE*4UL,j) );
            SIMDType xmm6( (~C).load(i+SIMDSIZE*5UL,j) );
            SIMDType xmm7( (~C).load(i+SIMDSIZE*6UL,j) );
            SIMDType xmm8( (~C).load(i+SIMDSIZE*7UL,j) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 - A.load(i             ,k) * b1;
               xmm2 = xmm2 - A.load(i+SIMDSIZE    ,k) * b1;
               xmm3 = xmm3 - A.load(i+SIMDSIZE*2UL,k) * b1;
               xmm4 = xmm4 - A.load(i+SIMDSIZE*3UL,k) * b1;
               xmm5 = xmm5 - A.load(i+SIMDSIZE*4UL,k) * b1;
               xmm6 = xmm6 - A.load(i+SIMDSIZE*5UL,k) * b1;
               xmm7 = xmm7 - A.load(i+SIMDSIZE*6UL,k) * b1;
               xmm8 = xmm8 - A.load(i+SIMDSIZE*7UL,k) * b1;
            }

            (~C).store( i             , j, xmm1 );
            (~C).store( i+SIMDSIZE    , j, xmm2 );
            (~C).store( i+SIMDSIZE*2UL, j, xmm3 );
            (~C).store( i+SIMDSIZE*3UL, j, xmm4 );
            (~C).store( i+SIMDSIZE*4UL, j, xmm5 );
            (~C).store( i+SIMDSIZE*5UL, j, xmm6 );
            (~C).store( i+SIMDSIZE*6UL, j, xmm7 );
            (~C).store( i+SIMDSIZE*7UL, j, xmm8 );
         }
      }

      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*4UL, K, ( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i             ,j    ) );
            SIMDType xmm2( (~C).load(i+SIMDSIZE    ,j    ) );
            SIMDType xmm3( (~C).load(i+SIMDSIZE*2UL,j    ) );
            SIMDType xmm4( (~C).load(i+SIMDSIZE*3UL,j    ) );
            SIMDType xmm5( (~C).load(i             ,j+1UL) );
            SIMDType xmm6( (~C).load(i+SIMDSIZE    ,j+1UL) );
            SIMDType xmm7( (~C).load(i+SIMDSIZE*2UL,j+1UL) );
            SIMDType xmm8( (~C).load(i+SIMDSIZE*3UL,j+1UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i             ,k) );
               const SIMDType a2( A.load(i+SIMDSIZE    ,k) );
               const SIMDType a3( A.load(i+SIMDSIZE*2UL,k) );
               const SIMDType a4( A.load(i+SIMDSIZE*3UL,k) );
               const SIMDType b1( set( B(k,j    ) ) );
               const SIMDType b2( set( B(k,j+1UL) ) );
               xmm1 = xmm1 - a1 * b1;
               xmm2 = xmm2 - a2 * b1;
               xmm3 = xmm3 - a3 * b1;
               xmm4 = xmm4 - a4 * b1;
               xmm5 = xmm5 - a1 * b2;
               xmm6 = xmm6 - a2 * b2;
               xmm7 = xmm7 - a3 * b2;
               xmm8 = xmm8 - a4 * b2;
            }

            (~C).store( i             , j    , xmm1 );
            (~C).store( i+SIMDSIZE    , j    , xmm2 );
            (~C).store( i+SIMDSIZE*2UL, j    , xmm3 );
            (~C).store( i+SIMDSIZE*3UL, j    , xmm4 );
            (~C).store( i             , j+1UL, xmm5 );
            (~C).store( i+SIMDSIZE    , j+1UL, xmm6 );
            (~C).store( i+SIMDSIZE*2UL, j+1UL, xmm7 );
            (~C).store( i+SIMDSIZE*3UL, j+1UL, xmm8 );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i             ,j) );
            SIMDType xmm2( (~C).load(i+SIMDSIZE    ,j) );
            SIMDType xmm3( (~C).load(i+SIMDSIZE*2UL,j) );
            SIMDType xmm4( (~C).load(i+SIMDSIZE*3UL,j) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 - A.load(i             ,k) * b1;
               xmm2 = xmm2 - A.load(i+SIMDSIZE    ,k) * b1;
               xmm3 = xmm3 - A.load(i+SIMDSIZE*2UL,k) * b1;
               xmm4 = xmm4 - A.load(i+SIMDSIZE*3UL,k) * b1;
            }

            (~C).store( i             , j, xmm1 );
            (~C).store( i+SIMDSIZE    , j, xmm2 );
            (~C).store( i+SIMDSIZE*2UL, j, xmm3 );
            (~C).store( i+SIMDSIZE*3UL, j, xmm4 );
         }
      }

      for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*2UL, K, ( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i         ,j    ) );
            SIMDType xmm2( (~C).load(i+SIMDSIZE,j    ) );
            SIMDType xmm3( (~C).load(i         ,j+1UL) );
            SIMDType xmm4( (~C).load(i+SIMDSIZE,j+1UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i         ,k) );
               const SIMDType a2( A.load(i+SIMDSIZE,k) );
               const SIMDType b1( set( B(k,j    ) ) );
               const SIMDType b2( set( B(k,j+1UL) ) );
               xmm1 = xmm1 - a1 * b1;
               xmm2 = xmm2 - a2 * b1;
               xmm3 = xmm3 - a1 * b2;
               xmm4 = xmm4 - a2 * b2;
            }

            (~C).store( i         , j    , xmm1 );
            (~C).store( i+SIMDSIZE, j    , xmm2 );
            (~C).store( i         , j+1UL, xmm3 );
            (~C).store( i+SIMDSIZE, j+1UL, xmm4 );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i         ,j) );
            SIMDType xmm2( (~C).load(i+SIMDSIZE,j) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 - A.load(i         ,k) * b1;
               xmm2 = xmm2 - A.load(i+SIMDSIZE,k) * b1;
            }

            (~C).store( i         , j, xmm1 );
            (~C).store( i+SIMDSIZE, j, xmm2 );
         }
      }

      for( ; i<ipos; i+=SIMDSIZE )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL )
                               :( K ) );

            SIMDType xmm1( (~C).load(i,j    ) );
            SIMDType xmm2( (~C).load(i,j+1UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i,k) );
               xmm1 = xmm1 - a1 * set( B(k,j    ) );
               xmm2 = xmm2 - a1 * set( B(k,j+1UL) );
            }

            (~C).store( i, j    , xmm1 );
            (~C).store( i, j+1UL, xmm2 );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );

            SIMDType xmm1( (~C).load(i,j) );

            for( size_t k=kbegin; k<K; ++k ) {
               xmm1 = xmm1 - A.load(i,k) * set( B(k,j) );
            }

            (~C).store( i, j, xmm1 );
         }
      }

      for( ; remainder && i<M; ++i )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL )
                               :( K ) );

            ElementType value1( (~C)(i,j    ) );
            ElementType value2( (~C)(i,j+1UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 -= A(i,k) * B(k,j    );
               value2 -= A(i,k) * B(k,j+1UL);
            }

            (~C)(i,j    ) = value1;
            (~C)(i,j+1UL) = value2;
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );

            ElementType value( (~C)(i,j) );

            for( size_t k=kbegin; k<K; ++k ) {
               value -= A(i,k) * B(k,j);
            }

            (~C)(i,j) = value;
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense matrices (large matrices)***************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a large transpose dense matrix-dense matrix
   //        multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a
   // transpose dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectLargeSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectDefaultSubAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to row-major dense matrices (large matrices)******
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default subtraction assignment of a large transpose dense matrix-dense
   //        matrix multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment of a transpose
   // dense matrix-dense matrix multiplication expression to a row-major dense matrix. This
   // kernel is optimized for large matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectLargeSubAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      for( size_t jj=0UL; jj<N; jj+=DMATDMATMULT_DEFAULT_JBLOCK_SIZE )
      {
         const size_t jend( min( jj+DMATDMATMULT_DEFAULT_JBLOCK_SIZE, N ) );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

         for( size_t ii=0UL; ii<M; ii+=DMATDMATMULT_DEFAULT_IBLOCK_SIZE )
         {
            const size_t iend( min( ii+DMATDMATMULT_DEFAULT_IBLOCK_SIZE, M ) );

            for( size_t kk=0UL; kk<K; kk+=DMATDMATMULT_DEFAULT_KBLOCK_SIZE )
            {
               const size_t ktmp( min( kk+DMATDMATMULT_DEFAULT_KBLOCK_SIZE, K ) );

               size_t j( jj );

               for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
               {
                  const size_t j1( j+SIMDSIZE     );
                  const size_t j2( j+SIMDSIZE*2UL );
                  const size_t j3( j+SIMDSIZE*3UL );

                  size_t i( ii );

                  for( ; (i+2UL) <= iend; i+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+2UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i    ,j ) );
                     SIMDType xmm2( (~C).load(i    ,j1) );
                     SIMDType xmm3( (~C).load(i    ,j2) );
                     SIMDType xmm4( (~C).load(i    ,j3) );
                     SIMDType xmm5( (~C).load(i+1UL,j ) );
                     SIMDType xmm6( (~C).load(i+1UL,j1) );
                     SIMDType xmm7( (~C).load(i+1UL,j2) );
                     SIMDType xmm8( (~C).load(i+1UL,j3) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        const SIMDType b3( B.load(k,j2) );
                        const SIMDType b4( B.load(k,j3) );
                        xmm1 = xmm1 - a1 * b1;
                        xmm2 = xmm2 - a1 * b2;
                        xmm3 = xmm3 - a1 * b3;
                        xmm4 = xmm4 - a1 * b4;
                        xmm5 = xmm5 - a2 * b1;
                        xmm6 = xmm6 - a2 * b2;
                        xmm7 = xmm7 - a2 * b3;
                        xmm8 = xmm8 - a2 * b4;
                     }

                     (~C).store( i    , j , xmm1 );
                     (~C).store( i    , j1, xmm2 );
                     (~C).store( i    , j2, xmm3 );
                     (~C).store( i    , j3, xmm4 );
                     (~C).store( i+1UL, j , xmm5 );
                     (~C).store( i+1UL, j1, xmm6 );
                     (~C).store( i+1UL, j2, xmm7 );
                     (~C).store( i+1UL, j3, xmm8 );
                  }

                  if( i < iend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i,j ) );
                     SIMDType xmm2( (~C).load(i,j1) );
                     SIMDType xmm3( (~C).load(i,j2) );
                     SIMDType xmm4( (~C).load(i,j3) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 - a1 * B.load(k,j );
                        xmm2 = xmm2 - a1 * B.load(k,j1);
                        xmm3 = xmm3 - a1 * B.load(k,j2);
                        xmm4 = xmm4 - a1 * B.load(k,j3);
                     }

                     (~C).store( i, j , xmm1 );
                     (~C).store( i, j1, xmm2 );
                     (~C).store( i, j2, xmm3 );
                     (~C).store( i, j3, xmm4 );
                  }
               }

               for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
               {
                  const size_t j1( j+SIMDSIZE );

                  size_t i( ii );

                  for( ; (i+4UL) <= iend; i+=4UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+4UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i    ,j ) );
                     SIMDType xmm2( (~C).load(i    ,j1) );
                     SIMDType xmm3( (~C).load(i+1UL,j ) );
                     SIMDType xmm4( (~C).load(i+1UL,j1) );
                     SIMDType xmm5( (~C).load(i+2UL,j ) );
                     SIMDType xmm6( (~C).load(i+2UL,j1) );
                     SIMDType xmm7( (~C).load(i+3UL,j ) );
                     SIMDType xmm8( (~C).load(i+3UL,j1) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType a3( set( A(i+2UL,k) ) );
                        const SIMDType a4( set( A(i+3UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        xmm1 = xmm1 - a1 * b1;
                        xmm2 = xmm2 - a1 * b2;
                        xmm3 = xmm3 - a2 * b1;
                        xmm4 = xmm4 - a2 * b2;
                        xmm5 = xmm5 - a3 * b1;
                        xmm6 = xmm6 - a3 * b2;
                        xmm7 = xmm7 - a4 * b1;
                        xmm8 = xmm8 - a4 * b2;
                     }

                     (~C).store( i    , j , xmm1 );
                     (~C).store( i    , j1, xmm2 );
                     (~C).store( i+1UL, j , xmm3 );
                     (~C).store( i+1UL, j1, xmm4 );
                     (~C).store( i+2UL, j , xmm5 );
                     (~C).store( i+2UL, j1, xmm6 );
                     (~C).store( i+3UL, j , xmm7 );
                     (~C).store( i+3UL, j1, xmm8 );
                  }

                  for( ; (i+2UL) <= iend; i+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+2UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i    ,j ) );
                     SIMDType xmm2( (~C).load(i    ,j1) );
                     SIMDType xmm3( (~C).load(i+1UL,j ) );
                     SIMDType xmm4( (~C).load(i+1UL,j1) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        xmm1 = xmm1 - a1 * b1;
                        xmm2 = xmm2 - a1 * b2;
                        xmm3 = xmm3 - a2 * b1;
                        xmm4 = xmm4 - a2 * b2;
                     }

                     (~C).store( i    , j , xmm1 );
                     (~C).store( i    , j1, xmm2 );
                     (~C).store( i+1UL, j , xmm3 );
                     (~C).store( i+1UL, j1, xmm4 );
                  }

                  if( i < iend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i,j ) );
                     SIMDType xmm2( (~C).load(i,j1) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 - a1 * B.load(k,j );
                        xmm2 = xmm2 - a1 * B.load(k,j1);
                     }

                     (~C).store( i, j , xmm1 );
                     (~C).store( i, j1, xmm2 );
                  }
               }

               for( ; j<jpos; j+=SIMDSIZE )
               {
                  for( size_t i=ii; i<iend; ++i )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 - a1 * B.load(k,j);
                     }

                     (~C).store( i, j, xmm1 );
                  }
               }

               for( ; remainder && j<jend; ++j )
               {
                  for( size_t i=ii; i<iend; ++i )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+1UL, ktmp ) ):( ktmp ) ) );

                     ElementType value( (~C)(i,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        value -= A(i,k) * B(k,j);
                     }

                     (~C)(i,j) = value;
                  }
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to column-major dense matrices (large matrices)***
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default subtraction assignment of a large transpose dense matrix-dense
   //        matrix multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment of a transpose
   // dense matrix-dense matrix multiplication expression to a column-major dense matrix.
   // This kernel is optimized for large matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5> >
      selectLargeSubAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      for( size_t ii=0UL; ii<M; ii+=TDMATTDMATMULT_DEFAULT_IBLOCK_SIZE )
      {
         const size_t iend( min( ii+TDMATTDMATMULT_DEFAULT_IBLOCK_SIZE, M ) );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

         for( size_t jj=0UL; jj<N; jj+=TDMATTDMATMULT_DEFAULT_JBLOCK_SIZE )
         {
            const size_t jend( min( jj+TDMATTDMATMULT_DEFAULT_JBLOCK_SIZE, N ) );

            for( size_t kk=0UL; kk<K; kk+=TDMATTDMATMULT_DEFAULT_KBLOCK_SIZE )
            {
               const size_t ktmp( min( kk+TDMATTDMATMULT_DEFAULT_KBLOCK_SIZE, K ) );

               size_t i( ii );

               for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL )
               {
                  const size_t i1( i+SIMDSIZE     );
                  const size_t i2( i+SIMDSIZE*2UL );
                  const size_t i3( i+SIMDSIZE*3UL );

                  size_t j( jj );

                  for( ; (j+2UL) <= jend; j+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+2UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j    ) );
                     SIMDType xmm2( (~C).load(i1,j    ) );
                     SIMDType xmm3( (~C).load(i2,j    ) );
                     SIMDType xmm4( (~C).load(i3,j    ) );
                     SIMDType xmm5( (~C).load(i ,j+1UL) );
                     SIMDType xmm6( (~C).load(i1,j+1UL) );
                     SIMDType xmm7( (~C).load(i2,j+1UL) );
                     SIMDType xmm8( (~C).load(i3,j+1UL) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType a3( A.load(i2,k) );
                        const SIMDType a4( A.load(i3,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        xmm1 = xmm1 - a1 * b1;
                        xmm2 = xmm2 - a2 * b1;
                        xmm3 = xmm3 - a3 * b1;
                        xmm4 = xmm4 - a4 * b1;
                        xmm5 = xmm5 - a1 * b2;
                        xmm6 = xmm6 - a2 * b2;
                        xmm7 = xmm7 - a3 * b2;
                        xmm8 = xmm8 - a4 * b2;
                     }

                     (~C).store( i , j    , xmm1 );
                     (~C).store( i1, j    , xmm2 );
                     (~C).store( i2, j    , xmm3 );
                     (~C).store( i3, j    , xmm4 );
                     (~C).store( i , j+1UL, xmm5 );
                     (~C).store( i1, j+1UL, xmm6 );
                     (~C).store( i2, j+1UL, xmm7 );
                     (~C).store( i3, j+1UL, xmm8 );
                  }

                  if( j < jend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j) );
                     SIMDType xmm2( (~C).load(i1,j) );
                     SIMDType xmm3( (~C).load(i2,j) );
                     SIMDType xmm4( (~C).load(i3,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 - A.load(i ,k) * b1;
                        xmm2 = xmm2 - A.load(i1,k) * b1;
                        xmm3 = xmm3 - A.load(i2,k) * b1;
                        xmm4 = xmm4 - A.load(i3,k) * b1;
                     }

                     (~C).store( i , j, xmm1 );
                     (~C).store( i1, j, xmm2 );
                     (~C).store( i2, j, xmm3 );
                     (~C).store( i3, j, xmm4 );
                  }
               }

               for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL )
               {
                  const size_t i1( i+SIMDSIZE );

                  size_t j( jj );

                  for( ; (j+4UL) <= jend; j+=4UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+4UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j    ) );
                     SIMDType xmm2( (~C).load(i1,j    ) );
                     SIMDType xmm3( (~C).load(i ,j+1UL) );
                     SIMDType xmm4( (~C).load(i1,j+1UL) );
                     SIMDType xmm5( (~C).load(i ,j+2UL) );
                     SIMDType xmm6( (~C).load(i1,j+2UL) );
                     SIMDType xmm7( (~C).load(i ,j+3UL) );
                     SIMDType xmm8( (~C).load(i1,j+3UL) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        const SIMDType b3( set( B(k,j+2UL) ) );
                        const SIMDType b4( set( B(k,j+3UL) ) );
                        xmm1 = xmm1 - a1 * b1;
                        xmm2 = xmm2 - a2 * b1;
                        xmm3 = xmm3 - a1 * b2;
                        xmm4 = xmm4 - a2 * b2;
                        xmm5 = xmm5 - a1 * b3;
                        xmm6 = xmm6 - a2 * b3;
                        xmm7 = xmm7 - a1 * b4;
                        xmm8 = xmm8 - a2 * b4;
                     }

                     (~C).store( i , j    , xmm1 );
                     (~C).store( i1, j    , xmm2 );
                     (~C).store( i , j+1UL, xmm3 );
                     (~C).store( i1, j+1UL, xmm4 );
                     (~C).store( i , j+2UL, xmm5 );
                     (~C).store( i1, j+2UL, xmm6 );
                     (~C).store( i , j+3UL, xmm7 );
                     (~C).store( i1, j+3UL, xmm8 );
                  }

                  for( ; (j+2UL) <= jend; j+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+2UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j    ) );
                     SIMDType xmm2( (~C).load(i1,j    ) );
                     SIMDType xmm3( (~C).load(i ,j+1UL) );
                     SIMDType xmm4( (~C).load(i1,j+1UL) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        xmm1 = xmm1 - a1 * b1;
                        xmm2 = xmm2 - a2 * b1;
                        xmm3 = xmm3 - a1 * b2;
                        xmm4 = xmm4 - a2 * b2;
                     }

                     (~C).store( i , j    , xmm1 );
                     (~C).store( i1, j    , xmm2 );
                     (~C).store( i , j+1UL, xmm3 );
                     (~C).store( i1, j+1UL, xmm4 );
                  }

                  if( j < jend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i ,j) );
                     SIMDType xmm2( (~C).load(i1,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 - A.load(i ,k) * b1;
                        xmm2 = xmm2 - A.load(i1,k) * b1;
                     }

                     (~C).store( i , j, xmm1 );
                     (~C).store( i1, j, xmm2 );
                  }
               }

               for( ; i<ipos; i+=SIMDSIZE )
               {
                  for( size_t j=jj; j<jend; ++j )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1( (~C).load(i,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 - A.load(i,k) * b1;
                     }

                     (~C).store( i, j, xmm1 );
                  }
               }

               for( ; remainder && i<iend; ++i )
               {
                  for( size_t j=jj; j<jend; ++j )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+1UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     ElementType value( (~C)(i,j) );

                     for( size_t k=kbegin; k<kend; ++k ) {
                        value -= A(i,k) * B(k,j);
                     }

                     (~C)(i,j) = value;
                  }
               }
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense matrices (default)*******************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a transpose dense matrix-dense matrix multiplication
   //        (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a
   // large transpose dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseBlasKernel<MT3,MT4,MT5> >
      selectBlasSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectLargeSubAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based subraction assignment to dense matrices******************************************
#if BLAZE_BLAS_MODE
   /*! \cond BLAZE_INTERNAL */
   /*!\brief BLAS-based subraction assignment of a transpose dense matrix-dense matrix
   //        multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function performs the transpose dense matrix-dense matrix multiplication based on the
   // according BLAS functionality.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseBlasKernel<MT3,MT4,MT5> >
      selectBlasSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      typedef ElementType_<MT3>  ET;

      if( IsTriangular<MT4>::value ) {
         ResultType_<MT3> tmp( serial( B ) );
         trmm( tmp, A, CblasLeft, ( IsLower<MT4>::value )?( CblasLower ):( CblasUpper ), ET(1) );
         subAssign( C, tmp );
      }
      else if( IsTriangular<MT5>::value ) {
         ResultType_<MT3> tmp( serial( A ) );
         trmm( tmp, B, CblasRight, ( IsLower<MT5>::value )?( CblasLower ):( CblasUpper ), ET(1) );
         subAssign( C, tmp );
      }
      else {
         gemm( C, A, B, ET(-1), ET(1) );
      }
   }
   /*! \endcond */
#endif
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
   /*!\brief SMP assignment of a transpose dense matrix-dense matrix multiplication to a dense
   //        matrix (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a transpose dense
   // matrix-dense matrix multiplication expression to a dense matrix. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case either of the two matrix operands requires an intermediate evaluation and no
   // symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline EnableIf_< IsEvaluationRequired<MT,MT1,MT2> >
      smpAssign( DenseMatrix<MT,SO>& lhs, const TDMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL ) {
         return;
      }
      else if( rhs.lhs_.columns() == 0UL ) {
         reset( ~lhs );
         return;
      }

      LT A( rhs.lhs_ );  // Evaluation of the left-hand side dense matrix operand
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
   /*!\brief SMP assignment of a transpose dense matrix-dense matrix multiplication to a sparse
   //        matrix (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side sparse matrix.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a transpose dense
   // matrix-dense matrix multiplication expression to a sparse matrix. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case either of the two matrix operands requires an intermediate evaluation and no
   // symmetry can be exploited.
   */
   template< typename MT  // Type of the target sparse matrix
           , bool SO >    // Storage order of the target sparse matrix
   friend inline EnableIf_< IsEvaluationRequired<MT,MT1,MT2> >
      smpAssign( SparseMatrix<MT,SO>& lhs, const TDMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      typedef IfTrue_< SO, ResultType, OppositeType >  TmpType;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( OppositeType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( OppositeType );
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
   /*!\brief SMP addition assignment of a transpose dense matrix-dense matrix multiplication to a
   //        dense matrix (\f$ C+=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a transpose
   // dense matrix-dense matrix multiplication expression to a dense matrix. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case either of the two matrix operands requires an intermediate evaluation and no
   // symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline EnableIf_< IsEvaluationRequired<MT,MT1,MT2> >
      smpAddAssign( DenseMatrix<MT,SO>& lhs, const TDMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || rhs.lhs_.columns() == 0UL ) {
         return;
      }

      LT A( rhs.lhs_ );  // Evaluation of the left-hand side dense matrix operand
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
   /*!\brief SMP subtraction assignment of a transpose dense matrix-dense matrix multiplication
   //        to a dense matrix (\f$ C-=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a transpose
   // dense matrix-dense matrix multiplication expression to a dense matrix. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case either of the two matrix operands requires an intermediate evaluation and no
   // symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline EnableIf_< IsEvaluationRequired<MT,MT1,MT2> >
      smpSubAssign( DenseMatrix<MT,SO>& lhs, const TDMatDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || rhs.lhs_.columns() == 0UL ) {
         return;
      }

      LT A( rhs.lhs_ );  // Evaluation of the left-hand side dense matrix operand
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
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_FORM_VALID_MATMATMULTEXPR( MT1, MT2 );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  DMATSCALARMULTEXPR SPECIALIZATION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Expression object for scaled transpose dense matrix-dense matrix multiplications.
// \ingroup dense_matrix_expression
//
// This specialization of the DMatScalarMultExpr class represents the compile time expression
// for scaled multiplications between a column-major dense matrix and a row-major dense matrix.
*/
template< typename MT1   // Type of the left-hand side dense matrix
        , typename MT2   // Type of the right-hand side dense matrix
        , typename ST >  // Type of the right-hand side scalar value
class DMatScalarMultExpr< TDMatDMatMultExpr<MT1,MT2>, ST, true >
   : public DenseMatrix< DMatScalarMultExpr< TDMatDMatMultExpr<MT1,MT2>, ST, true >, true >
   , private MatScalarMultExpr
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef TDMatDMatMultExpr<MT1,MT2>  MMM;  //!< Type of the dense matrix multiplication expression.
   typedef ResultType_<MMM>            RES;  //!< Result type of the dense matrix multiplication expression.
   typedef ResultType_<MT1>            RT1;  //!< Result type of the left-hand side dense matrix expression.
   typedef ResultType_<MT2>            RT2;  //!< Result type of the right-hand side dense matrix expression.
   typedef ElementType_<RT1>           ET1;  //!< Element type of the left-hand side dense matrix expression.
   typedef ElementType_<RT2>           ET2;  //!< Element type of the right-hand side dense matrix expression.
   typedef CompositeType_<MT1>         CT1;  //!< Composite type of the left-hand side dense matrix expression.
   typedef CompositeType_<MT2>         CT2;  //!< Composite type of the right-hand side dense matrix expression.
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the left-hand side dense matrix expression.
   enum : bool { evaluateLeft = IsComputation<MT1>::value || RequiresEvaluation<MT1>::value };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the right-hand side dense matrix expression.
   enum : bool { evaluateRight = IsComputation<MT2>::value || RequiresEvaluation<MT2>::value };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! The IsEvaluationRequired struct is a helper struct for the selection of the parallel
       evaluation strategy. In case either of the two matrix operands requires an intermediate
       evaluation, the nested \value will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   struct IsEvaluationRequired {
      enum : bool { value = ( evaluateLeft || evaluateRight ) };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! In case the types of all three involved matrices and the scalar type are suited for a BLAS
       kernel, the nested \a value will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3, typename T4 >
   struct UseBlasKernel {
      enum : bool { value = BLAZE_BLAS_MODE &&
                            HasMutableDataAccess<T1>::value &&
                            HasConstDataAccess<T2>::value &&
                            HasConstDataAccess<T3>::value &&
                            !IsDiagonal<T2>::value && !IsDiagonal<T3>::value &&
                            T1::simdEnabled && T2::simdEnabled && T3::simdEnabled &&
                            IsBLASCompatible< ElementType_<T1> >::value &&
                            IsBLASCompatible< ElementType_<T2> >::value &&
                            IsBLASCompatible< ElementType_<T3> >::value &&
                            IsSame< ElementType_<T1>, ElementType_<T2> >::value &&
                            IsSame< ElementType_<T1>, ElementType_<T3> >::value &&
                            !( IsBuiltin< ElementType_<T1> >::value && IsComplex<T4>::value ) };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! In case all four involved data types are suited for a vectorized computation of the
       matrix multiplication, the nested \value will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3, typename T4 >
   struct UseVectorizedDefaultKernel {
      enum : bool { value = useOptimizedKernels &&
                            !( IsDiagonal<T2>::value && IsDiagonal<T3>::value ) &&
                            !( IsDiagonal<T2>::value && IsColumnMajorMatrix<T1>::value ) &&
                            !( IsDiagonal<T3>::value && IsRowMajorMatrix<T1>::value ) &&
                            T1::simdEnabled && T2::simdEnabled && T3::simdEnabled &&
                            AreSIMDCombinable< ElementType_<T1>
                                             , ElementType_<T2>
                                             , ElementType_<T3>
                                             , T4 >::value &&
                            HasSIMDAdd< ElementType_<T2>, ElementType_<T3> >::value &&
                            HasSIMDMult< ElementType_<T2>, ElementType_<T3> >::value };
   };
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef DMatScalarMultExpr<MMM,ST,true>  This;           //!< Type of this DMatScalarMultExpr instance.
   typedef MultTrait_<RES,ST>               ResultType;     //!< Result type for expression template evaluations.
   typedef OppositeType_<ResultType>        OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef TransposeType_<ResultType>       TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<ResultType>         ElementType;    //!< Resulting element type.
   typedef SIMDTrait_<ElementType>          SIMDType;       //!< Resulting SIMD element type.
   typedef const ElementType                ReturnType;     //!< Return type for expression template evaluations.
   typedef const ResultType                 CompositeType;  //!< Data type for composite expression templates.

   //! Composite type of the left-hand side dense matrix expression.
   typedef const TDMatDMatMultExpr<MT1,MT2>  LeftOperand;

   //! Composite type of the right-hand side scalar value.
   typedef ST  RightOperand;

   //! Type for the assignment of the left-hand side dense matrix operand.
   typedef IfTrue_< evaluateLeft, const RT1, CT1 >  LT;

   //! Type for the assignment of the right-hand side dense matrix operand.
   typedef IfTrue_< evaluateRight, const RT2, CT2 >  RT;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = !( IsDiagonal<MT1>::value && IsDiagonal<MT2>::value ) &&
                               MT1::simdEnabled && MT2::simdEnabled &&
                               AreSIMDCombinable<ET1,ET2,ST>::value &&
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
   /*!\brief Constructor for the DMatScalarMultExpr class.
   //
   // \param matrix The left-hand side dense matrix of the multiplication expression.
   // \param scalar The right-hand side scalar of the multiplication expression.
   */
   explicit inline DMatScalarMultExpr( const MMM& matrix, ST scalar )
      : matrix_( matrix )  // Left-hand side dense matrix of the multiplication expression
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
   inline ResultType operator()( size_t i, size_t j ) const {
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

   //**Rows function*******************************************************************************
   /*!\brief Returns the current number of rows of the matrix.
   //
   // \return The number of rows of the matrix.
   */
   inline size_t rows() const {
      return matrix_.rows();
   }
   //**********************************************************************************************

   //**Columns function****************************************************************************
   /*!\brief Returns the current number of columns of the matrix.
   //
   // \return The number of columns of the matrix.
   */
   inline size_t columns() const {
      return matrix_.columns();
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side dense matrix operand.
   //
   // \return The left-hand side dense matrix operand.
   */
   inline LeftOperand leftOperand() const {
      return matrix_;
   }
   //**********************************************************************************************

   //**Right operand access************************************************************************
   /*!\brief Returns the right-hand side scalar operand.
   //
   // \return The right-hand side scalar operand.
   */
   inline RightOperand rightOperand() const {
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
   inline bool canAlias( const T* alias ) const {
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
   inline bool isAliased( const T* alias ) const {
      return matrix_.isAliased( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const {
      return matrix_.isAligned();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const {
      RightOperand_<MMM> B( matrix_.rightOperand() );
      return ( !BLAZE_BLAS_IS_PARALLEL ||
               ( rows() * columns() < TDMATDMATMULT_THRESHOLD ) ) &&
             ( B.columns() > SMP_TDMATDMATMULT_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  matrix_;  //!< Left-hand side dense matrix of the multiplication expression.
   RightOperand scalar_;  //!< Right-hand side scalar of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense matrices****************************************************************
   /*!\brief Assignment of a scaled transpose dense matrix-dense matrix multiplication to a
   //        dense matrix (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a scaled transpose dense
   // matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline void assign( DenseMatrix<MT,SO>& lhs, const DMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LeftOperand_<MMM>  left ( rhs.matrix_.leftOperand()  );
      RightOperand_<MMM> right( rhs.matrix_.rightOperand() );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL ) {
         return;
      }
      else if( left.columns() == 0UL ) {
         reset( ~lhs );
         return;
      }

      LT A( serial( left  ) );  // Evaluation of the left-hand side dense matrix operand
      RT B( serial( right ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == right.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == right.columns() , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns(), "Invalid number of columns" );

      DMatScalarMultExpr::selectAssignKernel( ~lhs, A, B, rhs.scalar_ );
   }
   //**********************************************************************************************

   //**Assignment to dense matrices (kernel selection)*********************************************
   /*!\brief Selection of the kernel for an assignment of a scaled transpose dense matrix-dense
   //        matrix multiplication to a dense matrix (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      if( ( IsDiagonal<MT4>::value && IsDiagonal<MT5>::value ) ||
          ( C.rows() * C.columns() < TDMATDMATMULT_THRESHOLD ) )
         selectSmallAssignKernel( C, A, B, scalar );
      else
         selectBlasAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Default assignment to row-major dense matrices (general/general)****************************
   /*!\brief Default assignment of a scaled general transpose dense matrix-general dense matrix
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment of a scaled general transpose dense matrix-
   // general dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, Not< IsDiagonal<MT5> > > >
      selectDefaultAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t kbegin( ( IsUpper<MT4>::value )
                              ?( IsStrictlyUpper<MT4>::value ? i+1UL : i )
                              :( 0UL ) );
         const size_t kend( ( IsLower<MT4>::value )
                            ?( IsStrictlyLower<MT4>::value ? i : i+1UL )
                            :( K ) );
         BLAZE_INTERNAL_ASSERT( kbegin <= kend, "Invalid loop indices detected" );

         if( IsStrictlyTriangular<MT4>::value && kbegin == kend ) {
            for( size_t j=0UL; j<N; ++j ) {
               reset( (~C)(i,j) );
            }
            continue;
         }

         {
            const size_t jbegin( ( IsUpper<MT5>::value )
                                 ?( IsStrictlyUpper<MT5>::value ? kbegin+1UL : kbegin )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT5>::value )
                               ?( IsStrictlyLower<MT5>::value ? kbegin : kbegin+1UL )
                               :( N ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            if( IsUpper<MT4>::value && IsUpper<MT5>::value ) {
               for( size_t j=0UL; j<jbegin; ++j ) {
                  reset( (~C)(i,j) );
               }
            }
            else if( IsStrictlyUpper<MT5>::value ) {
               reset( (~C)(i,0UL) );
            }
            for( size_t j=jbegin; j<jend; ++j ) {
               (~C)(i,j) = A(i,kbegin) * B(kbegin,j);
            }
            if( IsLower<MT4>::value && IsLower<MT5>::value ) {
               for( size_t j=jend; j<N; ++j ) {
                  reset( (~C)(i,j) );
               }
            }
            else if( IsStrictlyLower<MT5>::value ) {
               reset( (~C)(i,N-1UL) );
            }
         }

         for( size_t k=kbegin+1UL; k<kend; ++k )
         {
            const size_t jbegin( ( IsUpper<MT5>::value )
                                 ?( IsStrictlyUpper<MT5>::value ? k+1UL : k )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT5>::value )
                               ?( IsStrictlyLower<MT5>::value ? k-1UL : k )
                               :( N ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            for( size_t j=jbegin; j<jend; ++j ) {
               (~C)(i,j) += A(i,k) * B(k,j);
            }
            if( IsLower<MT5>::value ) {
               (~C)(i,jend) = A(i,k) * B(k,jend);
            }
         }

         {
            const size_t jbegin( ( IsUpper<MT4>::value && IsUpper<MT5>::value )
                                 ?( IsStrictlyUpper<MT4>::value || IsStrictlyUpper<MT5>::value ? i+1UL : i )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT4>::value && IsLower<MT5>::value )
                               ?( IsStrictlyLower<MT4>::value || IsStrictlyLower<MT5>::value ? i : i+1UL )
                               :( N ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            for( size_t j=jbegin; j<jend; ++j ) {
               (~C)(i,j) *= scalar;
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to column-major dense matrices (general/general)*************************
   /*!\brief Default assignment of a scaled general transpose dense matrix-general dense matrix
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment of a scaled general transpose dense matrix-
   // general dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, Not< IsDiagonal<MT5> > > >
      selectDefaultAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      for( size_t j=0UL; j<N; ++j )
      {
         const size_t kbegin( ( IsLower<MT5>::value )
                              ?( IsStrictlyLower<MT5>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t kend( ( IsUpper<MT5>::value )
                            ?( IsStrictlyUpper<MT5>::value ? j : j+1UL )
                            :( K ) );
         BLAZE_INTERNAL_ASSERT( kbegin <= kend, "Invalid loop indices detected" );

         if( IsStrictlyTriangular<MT5>::value && kbegin == kend ) {
            for( size_t i=0UL; i<M; ++i ) {
               reset( (~C)(i,j) );
            }
            continue;
         }

         {
            const size_t ibegin( ( IsLower<MT4>::value )
                                 ?( IsStrictlyLower<MT4>::value ? kbegin+1UL : kbegin )
                                 :( 0UL ) );
            const size_t iend( ( IsUpper<MT4>::value )
                               ?( IsStrictlyUpper<MT4>::value ? kbegin : kbegin+1UL )
                               :( M ) );
            BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

            if( IsLower<MT4>::value && IsLower<MT5>::value ) {
               for( size_t i=0UL; i<ibegin; ++i ) {
                  reset( (~C)(i,j) );
               }
            }
            else if( IsStrictlyLower<MT4>::value ) {
               reset( (~C)(0UL,j) );
            }
            for( size_t i=ibegin; i<iend; ++i ) {
               (~C)(i,j) = A(i,kbegin) * B(kbegin,j);
            }
            if( IsUpper<MT4>::value && IsUpper<MT5>::value ) {
               for( size_t i=iend; i<M; ++i ) {
                  reset( (~C)(i,j) );
               }
            }
            else if( IsStrictlyUpper<MT4>::value ) {
               reset( (~C)(M-1UL,j) );
            }
         }

         for( size_t k=kbegin+1UL; k<kend; ++k )
         {
            const size_t ibegin( ( IsLower<MT4>::value )
                                 ?( IsStrictlyLower<MT4>::value ? k+1UL : k )
                                 :( 0UL ) );
            const size_t iend( ( IsUpper<MT4>::value )
                               ?( IsStrictlyUpper<MT4>::value ? k-1UL : k )
                               :( M ) );
            BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

            for( size_t i=ibegin; i<iend; ++i ) {
               (~C)(i,j) += A(i,k) * B(k,j);
            }
            if( IsUpper<MT4>::value ) {
               (~C)(iend,j) = A(iend,k) * B(k,j);
            }
         }

         {
            const size_t ibegin( ( IsLower<MT4>::value && IsLower<MT5>::value )
                                 ?( IsStrictlyLower<MT4>::value || IsStrictlyLower<MT5>::value ? j+1UL : j )
                                 :( 0UL ) );
            const size_t iend( ( IsUpper<MT4>::value && IsUpper<MT5>::value )
                               ?( IsStrictlyUpper<MT4>::value || IsStrictlyUpper<MT5>::value ? j : j+1UL )
                               :( M ) );
            BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

            for( size_t i=ibegin; i<iend; ++i ) {
               (~C)(i,j) *= scalar;
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to row-major dense matrices (general/diagonal)***************************
   /*!\brief Default assignment of a scaled general transpose dense matrix-diagonal dense matrix
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment of a scaled general transpose dense matrix-
   // diagonal dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, IsDiagonal<MT5> > >
      selectDefaultAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      const size_t block( BLOCK_SIZE );

      for( size_t ii=0UL; ii<M; ii+=block ) {
         const size_t iend( min( M, ii+block ) );
         for( size_t jj=0UL; jj<N; jj+=block ) {
            const size_t jend( min( N, jj+block ) );
            for( size_t i=ii; i<iend; ++i )
            {
               const size_t jbegin( ( IsUpper<MT4>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), jj ) )
                                    :( jj ) );
               const size_t jpos( ( IsLower<MT4>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i : i+1UL ), jend ) )
                                  :( jend ) );

               if( IsUpper<MT4>::value ) {
                  for( size_t j=jj; j<jbegin; ++j ) {
                     reset( (~C)(i,j) );
                  }
               }
               for( size_t j=jbegin; j<jpos; ++j ) {
                  (~C)(i,j) = A(i,j) * B(j,j) * scalar;
               }
               if( IsLower<MT4>::value ) {
                  for( size_t j=jpos; j<jend; ++j ) {
                     reset( (~C)(i,j) );
                  }
               }
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to column-major dense matrices (general/diagonal)************************
   /*!\brief Default assignment of a scaled general transpose dense matrix-diagonal dense matrix
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment of a scaled general transpose dense matrix-
   // diagonal dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, IsDiagonal<MT5> > >
      selectDefaultAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t j=0UL; j<N; ++j )
      {
         const size_t ibegin( ( IsLower<MT4>::value )
                              ?( IsStrictlyLower<MT4>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT4>::value )
                            ?( IsStrictlyUpper<MT4>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         if( IsLower<MT4>::value ) {
            for( size_t i=0UL; i<ibegin; ++i ) {
               reset( (~C)(i,j) );
            }
         }
         for( size_t i=ibegin; i<iend; ++i ) {
            (~C)(i,j) = A(i,j) * B(j,j) * scalar;
         }
         if( IsUpper<MT4>::value ) {
            for( size_t i=iend; i<M; ++i ) {
               reset( (~C)(i,j) );
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to row-major dense matrices (diagonal/general)***************************
   /*!\brief Default assignment of a scaled diagonal transpose dense matrix-general dense matrix
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment of a scaled diagonal transpose dense matrix-
   // general dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< IsDiagonal<MT4>, Not< IsDiagonal<MT5> > > >
      selectDefaultAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper<MT5>::value )
                              ?( IsStrictlyUpper<MT5>::value ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT5>::value )
                            ?( IsStrictlyLower<MT5>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         if( IsUpper<MT5>::value ) {
            for( size_t j=0UL; j<jbegin; ++j ) {
               reset( (~C)(i,j) );
            }
         }
         for( size_t j=jbegin; j<jend; ++j ) {
            (~C)(i,j) = A(i,i) * B(i,j) * scalar;
         }
         if( IsLower<MT5>::value ) {
            for( size_t j=jend; j<N; ++j ) {
               reset( (~C)(i,j) );
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to column-major dense matrices (diagonal/general)************************
   /*!\brief Default assignment of a scaled diagonal transpose dense matrix-general dense matrix
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment of a scaled diagonal transpose dense matrix-
   // general dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< IsDiagonal<MT4>, Not< IsDiagonal<MT5> > > >
      selectDefaultAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      const size_t block( BLOCK_SIZE );

      for( size_t jj=0UL; jj<N; jj+=block ) {
         const size_t jend( min( N, jj+block ) );
         for( size_t ii=0UL; ii<M; ii+=block ) {
            const size_t iend( min( M, ii+block ) );
            for( size_t j=jj; j<jend; ++j )
            {
               const size_t ibegin( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyLower<MT5>::value ? j+1UL : j ), ii ) )
                                    :( ii ) );
               const size_t ipos( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyUpper<MT5>::value ? j : j+1UL ), iend ) )
                                  :( iend ) );

               if( IsLower<MT5>::value ) {
                  for( size_t i=ii; i<ibegin; ++i ) {
                     reset( (~C)(i,j) );
                  }
               }
               for( size_t i=ibegin; i<ipos; ++i ) {
                  (~C)(i,j) = A(i,i) * B(i,j) * scalar;
               }
               if( IsUpper<MT5>::value ) {
                  for( size_t i=ipos; i<iend; ++i ) {
                     reset( (~C)(i,j) );
                  }
               }
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to dense matrices (diagonal/diagonal)************************************
   /*!\brief Default assignment of a scaled diagonal transpose dense matrix-diagonal dense matrix
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment of a scaled diagonal transpose dense matrix-
   // diagonal dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< IsDiagonal<MT4>, IsDiagonal<MT5> > >
      selectDefaultAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      reset( C );

      for( size_t i=0UL; i<A.rows(); ++i ) {
         C(i,i) = A(i,i) * B(i,i) * scalar;
      }
   }
   //**********************************************************************************************

   //**Default assignment to dense matrices (small matrices)***************************************
   /*!\brief Default assignment of a small scaled transpose dense matrix-dense matrix multiplication
   //        (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a scaled transpose
   // dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectSmallAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectDefaultAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default assignment to row-major dense matrices (small matrices)******************
   /*!\brief Vectorized default assignment of a small scaled transpose dense matrix-dense matrix
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default assignment of a scaled transpose dense
   // matrix-dense matrix multiplication expression to a row-major dense matrix. This kernel
   // is optimized for small matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectSmallAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      const SIMDType factor( set( scalar ) );

      size_t j( 0UL );

      for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL ) {
         for( size_t i=0UL; i<M; ++i )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i : i+1UL ), j+SIMDSIZE*8UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i : i+1UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*8UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j             );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE    );
               xmm3 = xmm3 + a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 = xmm4 + a1 * B.load(k,j+SIMDSIZE*3UL);
               xmm5 = xmm5 + a1 * B.load(k,j+SIMDSIZE*4UL);
               xmm6 = xmm6 + a1 * B.load(k,j+SIMDSIZE*5UL);
               xmm7 = xmm7 + a1 * B.load(k,j+SIMDSIZE*6UL);
               xmm8 = xmm8 + a1 * B.load(k,j+SIMDSIZE*7UL);
            }

            (~C).store( i, j             , xmm1 * factor );
            (~C).store( i, j+SIMDSIZE    , xmm2 * factor );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 * factor );
            (~C).store( i, j+SIMDSIZE*3UL, xmm4 * factor );
            (~C).store( i, j+SIMDSIZE*4UL, xmm5 * factor );
            (~C).store( i, j+SIMDSIZE*5UL, xmm6 * factor );
            (~C).store( i, j+SIMDSIZE*6UL, xmm7 * factor );
            (~C).store( i, j+SIMDSIZE*7UL, xmm8 * factor );
         }
      }

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ), j+SIMDSIZE*4UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a1 * b2;
               xmm3 = xmm3 + a1 * b3;
               xmm4 = xmm4 + a1 * b4;
               xmm5 = xmm5 + a2 * b1;
               xmm6 = xmm6 + a2 * b2;
               xmm7 = xmm7 + a2 * b3;
               xmm8 = xmm8 + a2 * b4;
            }

            (~C).store( i    , j             , xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE    , xmm2 * factor );
            (~C).store( i    , j+SIMDSIZE*2UL, xmm3 * factor );
            (~C).store( i    , j+SIMDSIZE*3UL, xmm4 * factor );
            (~C).store( i+1UL, j             , xmm5 * factor );
            (~C).store( i+1UL, j+SIMDSIZE    , xmm6 * factor );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm7 * factor );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, xmm8 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j             );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE    );
               xmm3 = xmm3 + a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 = xmm4 + a1 * B.load(k,j+SIMDSIZE*3UL);
            }

            (~C).store( i, j             , xmm1 * factor );
            (~C).store( i, j+SIMDSIZE    , xmm2 * factor );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 * factor );
            (~C).store( i, j+SIMDSIZE*3UL, xmm4 * factor );
         }
      }

      for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a1 * b2;
               xmm3 = xmm3 + a2 * b1;
               xmm4 = xmm4 + a2 * b2;
            }

            (~C).store( i    , j         , xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE, xmm2 * factor );
            (~C).store( i+1UL, j         , xmm3 * factor );
            (~C).store( i+1UL, j+SIMDSIZE, xmm4 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j         );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE);
            }

            (~C).store( i, j         , xmm1 * factor );
            (~C).store( i, j+SIMDSIZE, xmm2 * factor );
         }
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL )
                               :( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 = xmm1 + set( A(i    ,k) ) * b1;
               xmm2 = xmm2 + set( A(i+1UL,k) ) * b1;
            }

            (~C).store( i    , j, xmm1 * factor );
            (~C).store( i+1UL, j, xmm2 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );

            SIMDType xmm1;

            for( size_t k=kbegin; k<K; ++k ) {
               xmm1 = xmm1 + set( A(i,k) ) * B.load(k,j);
            }

            (~C).store( i, j, xmm1 * factor );
         }
      }

      for( ; remainder && j<N; ++j )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL )
                               :( K ) );

            ElementType value1 = ElementType();
            ElementType value2 = ElementType();

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 += A(i    ,k) * B(k,j);
               value2 += A(i+1UL,k) * B(k,j);
            }

            (~C)(i    ,j) = value1 * scalar;
            (~C)(i+1UL,j) = value2 * scalar;
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );

            ElementType value = ElementType();

            for( size_t k=kbegin; k<K; ++k ) {
               value += A(i,k) * B(k,j);
            }

            (~C)(i,j) = value * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Vectorized default assignment to column-major dense matrices (small matrices)***************
   /*!\brief Vectorized default assignment of a small scaled transpose dense matrix-dense matrix
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default assignment of a scaled transpose dense
   // matrix-dense matrix multiplication expression to a column-major dense matrix. This kernel
   // is optimized for small matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectSmallAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT4>::value );

      const size_t ipos( remainder ? ( M & size_t(-SIMDSIZE) ) : M );
      BLAZE_INTERNAL_ASSERT( !remainder || ( M - ( M % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

      const SIMDType factor( set( scalar ) );

      size_t i( 0UL );

      for( ; (i+SIMDSIZE*7UL) < ipos; i+=SIMDSIZE*8UL ) {
         for( size_t j=0UL; j<N; ++j )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*8UL, K, ( IsStrictlyUpper<MT5>::value ? j : j+1UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j : j+1UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*8UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i             ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE    ,k) * b1;
               xmm3 = xmm3 + A.load(i+SIMDSIZE*2UL,k) * b1;
               xmm4 = xmm4 + A.load(i+SIMDSIZE*3UL,k) * b1;
               xmm5 = xmm5 + A.load(i+SIMDSIZE*4UL,k) * b1;
               xmm6 = xmm6 + A.load(i+SIMDSIZE*5UL,k) * b1;
               xmm7 = xmm7 + A.load(i+SIMDSIZE*6UL,k) * b1;
               xmm8 = xmm8 + A.load(i+SIMDSIZE*7UL,k) * b1;
            }

            (~C).store( i             , j, xmm1 * factor );
            (~C).store( i+SIMDSIZE    , j, xmm2 * factor );
            (~C).store( i+SIMDSIZE*2UL, j, xmm3 * factor );
            (~C).store( i+SIMDSIZE*3UL, j, xmm4 * factor );
            (~C).store( i+SIMDSIZE*4UL, j, xmm5 * factor );
            (~C).store( i+SIMDSIZE*5UL, j, xmm6 * factor );
            (~C).store( i+SIMDSIZE*6UL, j, xmm7 * factor );
            (~C).store( i+SIMDSIZE*7UL, j, xmm8 * factor );
         }
      }

      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*4UL, K, ( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i             ,k) );
               const SIMDType a2( A.load(i+SIMDSIZE    ,k) );
               const SIMDType a3( A.load(i+SIMDSIZE*2UL,k) );
               const SIMDType a4( A.load(i+SIMDSIZE*3UL,k) );
               const SIMDType b1( set( B(k,j    ) ) );
               const SIMDType b2( set( B(k,j+1UL) ) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a2 * b1;
               xmm3 = xmm3 + a3 * b1;
               xmm4 = xmm4 + a4 * b1;
               xmm5 = xmm5 + a1 * b2;
               xmm6 = xmm6 + a2 * b2;
               xmm7 = xmm7 + a3 * b2;
               xmm8 = xmm8 + a4 * b2;
            }

            (~C).store( i             , j    , xmm1 * factor );
            (~C).store( i+SIMDSIZE    , j    , xmm2 * factor );
            (~C).store( i+SIMDSIZE*2UL, j    , xmm3 * factor );
            (~C).store( i+SIMDSIZE*3UL, j    , xmm4 * factor );
            (~C).store( i             , j+1UL, xmm5 * factor );
            (~C).store( i+SIMDSIZE    , j+1UL, xmm6 * factor );
            (~C).store( i+SIMDSIZE*2UL, j+1UL, xmm7 * factor );
            (~C).store( i+SIMDSIZE*3UL, j+1UL, xmm8 * factor );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i             ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE    ,k) * b1;
               xmm3 = xmm3 + A.load(i+SIMDSIZE*2UL,k) * b1;
               xmm4 = xmm4 + A.load(i+SIMDSIZE*3UL,k) * b1;
            }

            (~C).store( i             , j, xmm1 * factor );
            (~C).store( i+SIMDSIZE    , j, xmm2 * factor );
            (~C).store( i+SIMDSIZE*2UL, j, xmm3 * factor );
            (~C).store( i+SIMDSIZE*3UL, j, xmm4 * factor );
         }
      }

      for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*2UL, K, ( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i         ,k) );
               const SIMDType a2( A.load(i+SIMDSIZE,k) );
               const SIMDType b1( set( B(k,j    ) ) );
               const SIMDType b2( set( B(k,j+1UL) ) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a2 * b1;
               xmm3 = xmm3 + a1 * b2;
               xmm4 = xmm4 + a2 * b2;
            }

            (~C).store( i         , j    , xmm1 * factor );
            (~C).store( i+SIMDSIZE, j    , xmm2 * factor );
            (~C).store( i         , j+1UL, xmm3 * factor );
            (~C).store( i+SIMDSIZE, j+1UL, xmm4 * factor );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i         ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE,k) * b1;
            }

            (~C).store( i         , j, xmm1 * factor );
            (~C).store( i+SIMDSIZE, j, xmm2 * factor );
         }
      }

      for( ; i<ipos; i+=SIMDSIZE )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL )
                               :( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i,k) );
               xmm1 = xmm1 + a1 * set( B(k,j    ) );
               xmm2 = xmm2 + a1 * set( B(k,j+1UL) );
            }

            (~C).store( i, j    , xmm1 * factor );
            (~C).store( i, j+1UL, xmm2 * factor );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );

            SIMDType xmm1;

            for( size_t k=kbegin; k<K; ++k ) {
               xmm1 = xmm1 + A.load(i,k) * set( B(k,j) );
            }

            (~C).store( i, j, xmm1 * factor );
         }
      }

      for( ; remainder && i<M; ++i )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL )
                               :( K ) );

            ElementType value1 = ElementType();
            ElementType value2 = ElementType();

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 += A(i,k) * B(k,j    );
               value2 += A(i,k) * B(k,j+1UL);
            }

            (~C)(i,j    ) = value1 * scalar;
            (~C)(i,j+1UL) = value2 * scalar;
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );

            ElementType value = ElementType();

            for( size_t k=kbegin; k<K; ++k ) {
               value += A(i,k) * B(k,j);
            }

            (~C)(i,j) = value * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to dense matrices (large matrices)***************************************
   /*!\brief Default assignment of a large scaled transpose dense matrix-dense matrix multiplication
   //        (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a scaled transpose
   // dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectLargeAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectDefaultAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default assignment to row-major dense matrices (large matrices)******************
   /*!\brief Vectorized default assignment of a large scaled transpose dense matrix-dense matrix
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default assignment of a scaled transpose dense
   // matrix-dense matrix multiplication expression to a row-major dense matrix. This kernel
   // is optimized for large matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectLargeAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      const SIMDType factor( set( scalar ) );

      for( size_t jj=0UL; jj<N; jj+=DMATDMATMULT_DEFAULT_JBLOCK_SIZE )
      {
         const size_t jend( min( jj+DMATDMATMULT_DEFAULT_JBLOCK_SIZE, N ) );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

         for( size_t ii=0UL; ii<M; ii+=DMATDMATMULT_DEFAULT_IBLOCK_SIZE )
         {
            const size_t iend( min( ii+DMATDMATMULT_DEFAULT_IBLOCK_SIZE, M ) );

            for( size_t i=ii; i<iend; ++i ) {
               for( size_t j=jj; j<jend; ++j ) {
                  reset( (~C)(i,j) );
               }
            }

            for( size_t kk=0UL; kk<K; kk+=DMATDMATMULT_DEFAULT_KBLOCK_SIZE )
            {
               const size_t ktmp( min( kk+DMATDMATMULT_DEFAULT_KBLOCK_SIZE, K ) );

               size_t j( jj );

               for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
               {
                  const size_t j1( j+SIMDSIZE     );
                  const size_t j2( j+SIMDSIZE*2UL );
                  const size_t j3( j+SIMDSIZE*3UL );

                  size_t i( ii );

                  for( ; (i+2UL) <= iend; i+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+2UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        const SIMDType b3( B.load(k,j2) );
                        const SIMDType b4( B.load(k,j3) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a1 * b3;
                        xmm4 = xmm4 + a1 * b4;
                        xmm5 = xmm5 + a2 * b1;
                        xmm6 = xmm6 + a2 * b2;
                        xmm7 = xmm7 + a2 * b3;
                        xmm8 = xmm8 + a2 * b4;
                     }

                     (~C).store( i    , j , (~C).load(i    ,j ) + xmm1 * factor );
                     (~C).store( i    , j1, (~C).load(i    ,j1) + xmm2 * factor );
                     (~C).store( i    , j2, (~C).load(i    ,j2) + xmm3 * factor );
                     (~C).store( i    , j3, (~C).load(i    ,j3) + xmm4 * factor );
                     (~C).store( i+1UL, j , (~C).load(i+1UL,j ) + xmm5 * factor );
                     (~C).store( i+1UL, j1, (~C).load(i+1UL,j1) + xmm6 * factor );
                     (~C).store( i+1UL, j2, (~C).load(i+1UL,j2) + xmm7 * factor );
                     (~C).store( i+1UL, j3, (~C).load(i+1UL,j3) + xmm8 * factor );
                  }

                  if( i < iend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j );
                        xmm2 = xmm2 + a1 * B.load(k,j1);
                        xmm3 = xmm3 + a1 * B.load(k,j2);
                        xmm4 = xmm4 + a1 * B.load(k,j3);
                     }

                     (~C).store( i, j , (~C).load(i,j ) + xmm1 * factor );
                     (~C).store( i, j1, (~C).load(i,j1) + xmm2 * factor );
                     (~C).store( i, j2, (~C).load(i,j2) + xmm3 * factor );
                     (~C).store( i, j3, (~C).load(i,j3) + xmm4 * factor );
                  }
               }

               for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
               {
                  const size_t j1( j+SIMDSIZE );

                  size_t i( ii );

                  for( ; (i+4UL) <= iend; i+=4UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+4UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType a3( set( A(i+2UL,k) ) );
                        const SIMDType a4( set( A(i+3UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a2 * b1;
                        xmm4 = xmm4 + a2 * b2;
                        xmm5 = xmm5 + a3 * b1;
                        xmm6 = xmm6 + a3 * b2;
                        xmm7 = xmm7 + a4 * b1;
                        xmm8 = xmm8 + a4 * b2;
                     }

                     (~C).store( i    , j , (~C).load(i    ,j ) + xmm1 * factor );
                     (~C).store( i    , j1, (~C).load(i    ,j1) + xmm2 * factor );
                     (~C).store( i+1UL, j , (~C).load(i+1UL,j ) + xmm3 * factor );
                     (~C).store( i+1UL, j1, (~C).load(i+1UL,j1) + xmm4 * factor );
                     (~C).store( i+2UL, j , (~C).load(i+2UL,j ) + xmm5 * factor );
                     (~C).store( i+2UL, j1, (~C).load(i+2UL,j1) + xmm6 * factor );
                     (~C).store( i+3UL, j , (~C).load(i+3UL,j ) + xmm7 * factor );
                     (~C).store( i+3UL, j1, (~C).load(i+3UL,j1) + xmm8 * factor );
                  }

                  for( ; (i+2UL) <= iend; i+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+2UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a2 * b1;
                        xmm4 = xmm4 + a2 * b2;
                     }

                     (~C).store( i    , j , (~C).load(i    ,j ) + xmm1 * factor );
                     (~C).store( i    , j1, (~C).load(i    ,j1) + xmm2 * factor );
                     (~C).store( i+1UL, j , (~C).load(i+1UL,j ) + xmm3 * factor );
                     (~C).store( i+1UL, j1, (~C).load(i+1UL,j1) + xmm4 * factor );
                  }

                  if( i < iend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j );
                        xmm2 = xmm2 + a1 * B.load(k,j1);
                     }

                     (~C).store( i, j , (~C).load(i,j ) + xmm1 * factor );
                     (~C).store( i, j1, (~C).load(i,j1) + xmm2 * factor );
                  }
               }

               for( ; j<jpos; j+=SIMDSIZE )
               {
                  for( size_t i=ii; i<iend; ++i )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j);
                     }

                     (~C).store( i, j, (~C).load(i,j) + xmm1 * factor );
                  }
               }

               for( ; remainder && j<jend; ++j )
               {
                  for( size_t i=ii; i<iend; ++i )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+1UL, ktmp ) ):( ktmp ) ) );

                     ElementType value = ElementType();

                     for( size_t k=kbegin; k<kend; ++k ) {
                        value += A(i,k) * B(k,j);
                     }

                     (~C)(i,j) += value * scalar;
                  }
               }
            }
         }
      }
   }
   //**********************************************************************************************

   //**Vectorized default assignment to column-major dense matrices (large matrices)***************
   /*!\brief Vectorized default assignment of a large scaled transpose dense matrix-dense matrix
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default assignment of a scaled transpose dense
   // matrix-dense matrix multiplication expression to a column-major dense matrix. This kernel
   // is optimized for large matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectLargeAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT4>::value );

      const SIMDType factor( set( scalar ) );

      for( size_t ii=0UL; ii<M; ii+=TDMATTDMATMULT_DEFAULT_IBLOCK_SIZE )
      {
         const size_t iend( min( ii+TDMATTDMATMULT_DEFAULT_IBLOCK_SIZE, M ) );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

         for( size_t jj=0UL; jj<N; jj+=TDMATTDMATMULT_DEFAULT_JBLOCK_SIZE )
         {
            const size_t jend( min( jj+TDMATTDMATMULT_DEFAULT_JBLOCK_SIZE, N ) );

            for( size_t j=jj; j<jend; ++j ) {
               for( size_t i=ii; i<iend; ++i ) {
                  reset( (~C)(i,j) );
               }
            }

            for( size_t kk=0UL; kk<K; kk+=TDMATTDMATMULT_DEFAULT_KBLOCK_SIZE )
            {
               const size_t ktmp( min( kk+TDMATTDMATMULT_DEFAULT_KBLOCK_SIZE, K ) );

               size_t i( ii );

               for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL )
               {
                  const size_t i1( i+SIMDSIZE     );
                  const size_t i2( i+SIMDSIZE*2UL );
                  const size_t i3( i+SIMDSIZE*3UL );

                  size_t j( jj );

                  for( ; (j+2UL) <= jend; j+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+2UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType a3( A.load(i2,k) );
                        const SIMDType a4( A.load(i3,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a3 * b1;
                        xmm4 = xmm4 + a4 * b1;
                        xmm5 = xmm5 + a1 * b2;
                        xmm6 = xmm6 + a2 * b2;
                        xmm7 = xmm7 + a3 * b2;
                        xmm8 = xmm8 + a4 * b2;
                     }

                     (~C).store( i , j    , (~C).load(i ,j    ) + xmm1 * factor );
                     (~C).store( i1, j    , (~C).load(i1,j    ) + xmm2 * factor );
                     (~C).store( i2, j    , (~C).load(i2,j    ) + xmm3 * factor );
                     (~C).store( i3, j    , (~C).load(i3,j    ) + xmm4 * factor );
                     (~C).store( i , j+1UL, (~C).load(i ,j+1UL) + xmm5 * factor );
                     (~C).store( i1, j+1UL, (~C).load(i1,j+1UL) + xmm6 * factor );
                     (~C).store( i2, j+1UL, (~C).load(i2,j+1UL) + xmm7 * factor );
                     (~C).store( i3, j+1UL, (~C).load(i3,j+1UL) + xmm8 * factor );
                  }

                  if( j < jend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i ,k) * b1;
                        xmm2 = xmm2 + A.load(i1,k) * b1;
                        xmm3 = xmm3 + A.load(i2,k) * b1;
                        xmm4 = xmm4 + A.load(i3,k) * b1;
                     }

                     (~C).store( i , j, (~C).load(i ,j) + xmm1 * factor );
                     (~C).store( i1, j, (~C).load(i1,j) + xmm2 * factor );
                     (~C).store( i2, j, (~C).load(i2,j) + xmm3 * factor );
                     (~C).store( i3, j, (~C).load(i3,j) + xmm4 * factor );
                  }
               }

               for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL )
               {
                  const size_t i1( i+SIMDSIZE );

                  size_t j( jj );

                  for( ; (j+4UL) <= jend; j+=4UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+4UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        const SIMDType b3( set( B(k,j+2UL) ) );
                        const SIMDType b4( set( B(k,j+3UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a1 * b2;
                        xmm4 = xmm4 + a2 * b2;
                        xmm5 = xmm5 + a1 * b3;
                        xmm6 = xmm6 + a2 * b3;
                        xmm7 = xmm7 + a1 * b4;
                        xmm8 = xmm8 + a2 * b4;
                     }

                     (~C).store( i , j    , (~C).load(i ,j    ) + xmm1 * factor );
                     (~C).store( i1, j    , (~C).load(i1,j    ) + xmm2 * factor );
                     (~C).store( i , j+1UL, (~C).load(i ,j+1UL) + xmm3 * factor );
                     (~C).store( i1, j+1UL, (~C).load(i1,j+1UL) + xmm4 * factor );
                     (~C).store( i , j+2UL, (~C).load(i ,j+2UL) + xmm5 * factor );
                     (~C).store( i1, j+2UL, (~C).load(i1,j+2UL) + xmm6 * factor );
                     (~C).store( i , j+3UL, (~C).load(i ,j+3UL) + xmm7 * factor );
                     (~C).store( i1, j+3UL, (~C).load(i1,j+3UL) + xmm8 * factor );
                  }

                  for( ; (j+2UL) <= jend; j+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+2UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a1 * b2;
                        xmm4 = xmm4 + a2 * b2;
                     }

                     (~C).store( i , j    , (~C).load(i ,j    ) + xmm1 * factor );
                     (~C).store( i1, j    , (~C).load(i1,j    ) + xmm2 * factor );
                     (~C).store( i , j+1UL, (~C).load(i ,j+1UL) + xmm3 * factor );
                     (~C).store( i1, j+1UL, (~C).load(i1,j+1UL) + xmm4 * factor );
                  }

                  if( j < jend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i ,k) * b1;
                        xmm2 = xmm2 + A.load(i1,k) * b1;
                     }

                     (~C).store( i , j, (~C).load(i ,j) + xmm1 * factor );
                     (~C).store( i1, j, (~C).load(i1,j) + xmm2 * factor );
                  }
               }

               for( ; i<ipos; i+=SIMDSIZE )
               {
                  for( size_t j=jj; j<jend; ++j )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i,k) * b1;
                     }

                     (~C).store( i, j, (~C).load(i,j) + xmm1 * factor );
                  }
               }

               for( ; remainder && i<iend; ++i )
               {
                  for( size_t j=jj; j<jend; ++j )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+1UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     ElementType value = ElementType();

                     for( size_t k=kbegin; k<kend; ++k ) {
                        value += A(i,k) * B(k,j);
                     }

                     (~C)(i,j) += value * scalar;
                  }
               }
            }
         }
      }
   }
   //**********************************************************************************************

   //**BLAS-based assignment to dense matrices (default)*******************************************
   /*!\brief Default assignment of a scaled transpose dense matrix-dense matrix multiplication
   //        (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a large scaled
   // transpose dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseBlasKernel<MT3,MT4,MT5,ST2> >
      selectBlasAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectLargeAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**BLAS-based assignment to dense matrices*****************************************************
#if BLAZE_BLAS_MODE
   /*!\brief BLAS-based assignment of a scaled transpose dense matrix-dense matrix multiplication
   //        (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function performs the scaled transpose dense matrix-dense matrix multiplication based
   // on the according BLAS functionality.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseBlasKernel<MT3,MT4,MT5,ST2> >
      selectBlasAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      typedef ElementType_<MT3>  ET;

      if( IsTriangular<MT4>::value ) {
         assign( C, B );
         trmm( C, A, CblasLeft, ( IsLower<MT4>::value )?( CblasLower ):( CblasUpper ), ET(scalar) );
      }
      else if( IsTriangular<MT5>::value ) {
         assign( C, A );
         trmm( C, B, CblasRight, ( IsLower<MT5>::value )?( CblasLower ):( CblasUpper ), ET(scalar) );
      }
      else {
         gemm( C, A, B, ET(scalar), ET(0) );
      }
   }
#endif
   //**********************************************************************************************

   //**Assignment to sparse matrices***************************************************************
   /*!\brief Assignment of a scaled transpose dense matrix-dense matrix multiplication to a sparse
   //        matrix.
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side sparse matrix.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a scaled transpose dense
   // matrix-dense matrix multiplication expression to a sparse matrix.
   */
   template< typename MT  // Type of the target sparse matrix
           , bool SO >    // Storage order of the target sparse matrix
   friend inline void assign( SparseMatrix<MT,SO>& lhs, const DMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      typedef IfTrue_< SO, ResultType, OppositeType >  TmpType;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( OppositeType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( OppositeType );
      BLAZE_CONSTRAINT_MATRICES_MUST_HAVE_SAME_STORAGE_ORDER( MT, TmpType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<TmpType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const TmpType tmp( serial( rhs ) );
      assign( ~lhs, tmp );
   }
   //**********************************************************************************************

   //**Addition assignment to dense matrices*******************************************************
   /*!\brief Addition assignment of a scaled transpose dense matrix-dense matrix multiplication
   //        to a dense matrix (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a scaled transpose
   // dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline void addAssign( DenseMatrix<MT,SO>& lhs, const DMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LeftOperand_<MMM>  left ( rhs.matrix_.leftOperand()  );
      RightOperand_<MMM> right( rhs.matrix_.rightOperand() );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || left.columns() == 0UL ) {
         return;
      }

      LT A( serial( left  ) );  // Evaluation of the left-hand side dense matrix operand
      RT B( serial( right ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == right.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == right.columns() , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns(), "Invalid number of columns" );

      DMatScalarMultExpr::selectAddAssignKernel( ~lhs, A, B, rhs.scalar_ );
   }
   //**********************************************************************************************

   //**Addition assignment to dense matrices (kernel selection)************************************
   /*!\brief Selection of the kernel for an addition assignment of a scaled transpose dense
   //        matrix-dense matrix multiplication to a dense matrix (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      if( ( IsDiagonal<MT4>::value && IsDiagonal<MT5>::value ) ||
          ( C.rows() * C.columns() < TDMATDMATMULT_THRESHOLD ) )
         selectSmallAddAssignKernel( C, A, B, scalar );
      else
         selectBlasAddAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Default addition assignment to dense matrices (general/general)*****************************
   /*!\brief Default addition assignment of a scaled general transpose dense matrix-general dense
   //        matrix multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default addition assignment of a scaled general transpose dense
   // matrix-general dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, Not< IsDiagonal<MT5> > > >
      selectDefaultAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const ResultType tmp( serial( A * B * scalar ) );
      addAssign( C, tmp );
   }
   //**********************************************************************************************

   //**Default addition assignment to row-major dense matrices (general/diagonal)******************
   /*!\brief Default addition assignment of a scaled general transpose dense matrix-diagonal dense
   //        matrix multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default addition assignment of a scaled general transpose dense
   // matrix-diagonal dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, IsDiagonal<MT5> > >
      selectDefaultAddAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      const size_t block( BLOCK_SIZE );

      for( size_t ii=0UL; ii<M; ii+=block ) {
         const size_t iend( min( M, ii+block ) );
         for( size_t jj=0UL; jj<N; jj+=block ) {
            const size_t jend( min( N, jj+block ) );
            for( size_t i=ii; i<iend; ++i )
            {
               const size_t jbegin( ( IsUpper<MT4>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), jj ) )
                                    :( jj ) );
               const size_t jpos( ( IsLower<MT4>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i : i+1UL ), jend ) )
                                  :( jend ) );

               for( size_t j=jbegin; j<jpos; ++j ) {
                  (~C)(i,j) += A(i,j) * B(j,j) * scalar;
               }
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default addition assignment to column-major dense matrices (general/diagonal)***************
   /*!\brief Default addition assignment of a scaled general transpose dense matrix-diagonal dense
   //        matrix multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default addition assignment of a scaled general transpose dense
   // matrix-diagonal dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, IsDiagonal<MT5> > >
      selectDefaultAddAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t j=0UL; j<N; ++j )
      {
         const size_t ibegin( ( IsLower<MT4>::value )
                              ?( IsStrictlyLower<MT4>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT4>::value )
                            ?( IsStrictlyUpper<MT4>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t inum( iend - ibegin );
         const size_t ipos( ibegin + ( inum & size_t(-2) ) );

         for( size_t i=ibegin; i<ipos; i+=2UL ) {
            (~C)(i    ,j) += A(i    ,j) * B(j,j) * scalar;
            (~C)(i+1UL,j) += A(i+1UL,j) * B(j,j) * scalar;
         }
         if( ipos < iend ) {
            (~C)(ipos,j) += A(ipos,j) * B(j,j) * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default addition assignment to row-major dense matrices (diagonal/general)******************
   /*!\brief Default addition assignment of a scaled diagonal transpose dense matrix-general dense
   //        matrix multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default addition assignment of a scaled diagonal transpose
   // dense matrix-general dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< IsDiagonal<MT4>, Not< IsDiagonal<MT5> > > >
      selectDefaultAddAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper<MT5>::value )
                              ?( IsStrictlyUpper<MT5>::value ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT5>::value )
                            ?( IsStrictlyLower<MT5>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jnum( jend - jbegin );
         const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

         for( size_t j=jbegin; j<jpos; j+=2UL ) {
            (~C)(i,j    ) += A(i,i) * B(i,j    ) * scalar;
            (~C)(i,j+1UL) += A(i,i) * B(i,j+1UL) * scalar;
         }
         if( jpos < jend ) {
            (~C)(i,jpos) += A(i,i) * B(i,jpos) * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default addition assignment to column-major dense matrices (diagonal/general)***************
   /*!\brief Default addition assignment of a scaled diagonal transpose dense matrix-general dense
   //        matrix multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default addition assignment of a scaled diagonal transpose
   // dense matrix-general dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< IsDiagonal<MT4>, Not< IsDiagonal<MT5> > > >
      selectDefaultAddAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      const size_t block( BLOCK_SIZE );

      for( size_t jj=0UL; jj<N; jj+=block ) {
         const size_t jend( min( N, jj+block ) );
         for( size_t ii=0UL; ii<M; ii+=block ) {
            const size_t iend( min( M, ii+block ) );
            for( size_t j=jj; j<jend; ++j )
            {
               const size_t ibegin( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyLower<MT5>::value ? j+1UL : j ), ii ) )
                                    :( ii ) );
               const size_t ipos( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyUpper<MT5>::value ? j : j+1UL ), iend ) )
                                  :( iend ) );

               for( size_t i=ibegin; i<ipos; ++i ) {
                  (~C)(i,j) += A(i,i) * B(i,j) * scalar;
               }
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default addition assignment to dense matrices (diagonal/diagonal)***************************
   /*!\brief Default addition assignment of a scaled diagonal transpose dense matrix-diagonal
   //        dense matrix multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default addition assignment of a scaled diagonal transpose
   // dense matrix-diagonal dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< IsDiagonal<MT4>, IsDiagonal<MT5> > >
      selectDefaultAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      for( size_t i=0UL; i<A.rows(); ++i ) {
         C(i,i) += A(i,i) * B(i,i) * scalar;
      }
   }
   //**********************************************************************************************

   //**Default addition assignment to dense matrices (small matrices)******************************
   /*!\brief Default addition assignment of a small scaled transpose dense matrix-dense matrix
   //        multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a scaled
   // transpose dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectSmallAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectDefaultAddAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default addition assignment to row-major dense matrices (small matrices)*********
   /*!\brief Vectorized default addition assignment of a small scaled transpose dense matrix-dense
   //        matrix multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default addition assignment of a scaled transpose
   // dense matrix-dense matrix multiplication expression to a row-major dense matrix. This
   // kernel is optimized for small matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectSmallAddAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      const SIMDType factor( set( scalar ) );

      size_t j( 0UL );

      for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL ) {
         for( size_t i=0UL; i<M; ++i )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i : i+1UL ), j+SIMDSIZE*8UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i : i+1UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*8UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j             );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE    );
               xmm3 = xmm3 + a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 = xmm4 + a1 * B.load(k,j+SIMDSIZE*3UL);
               xmm5 = xmm5 + a1 * B.load(k,j+SIMDSIZE*4UL);
               xmm6 = xmm6 + a1 * B.load(k,j+SIMDSIZE*5UL);
               xmm7 = xmm7 + a1 * B.load(k,j+SIMDSIZE*6UL);
               xmm8 = xmm8 + a1 * B.load(k,j+SIMDSIZE*7UL);
            }

            (~C).store( i, j             , (~C).load(i,j             ) + xmm1 * factor );
            (~C).store( i, j+SIMDSIZE    , (~C).load(i,j+SIMDSIZE    ) + xmm2 * factor );
            (~C).store( i, j+SIMDSIZE*2UL, (~C).load(i,j+SIMDSIZE*2UL) + xmm3 * factor );
            (~C).store( i, j+SIMDSIZE*3UL, (~C).load(i,j+SIMDSIZE*3UL) + xmm4 * factor );
            (~C).store( i, j+SIMDSIZE*4UL, (~C).load(i,j+SIMDSIZE*4UL) + xmm5 * factor );
            (~C).store( i, j+SIMDSIZE*5UL, (~C).load(i,j+SIMDSIZE*5UL) + xmm6 * factor );
            (~C).store( i, j+SIMDSIZE*6UL, (~C).load(i,j+SIMDSIZE*6UL) + xmm7 * factor );
            (~C).store( i, j+SIMDSIZE*7UL, (~C).load(i,j+SIMDSIZE*7UL) + xmm8 * factor );
         }
      }

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ), j+SIMDSIZE*4UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a1 * b2;
               xmm3 = xmm3 + a1 * b3;
               xmm4 = xmm4 + a1 * b4;
               xmm5 = xmm5 + a2 * b1;
               xmm6 = xmm6 + a2 * b2;
               xmm7 = xmm7 + a2 * b3;
               xmm8 = xmm8 + a2 * b4;
            }

            (~C).store( i    , j             , (~C).load(i    ,j             ) + xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE    , (~C).load(i    ,j+SIMDSIZE    ) + xmm2 * factor );
            (~C).store( i    , j+SIMDSIZE*2UL, (~C).load(i    ,j+SIMDSIZE*2UL) + xmm3 * factor );
            (~C).store( i    , j+SIMDSIZE*3UL, (~C).load(i    ,j+SIMDSIZE*3UL) + xmm4 * factor );
            (~C).store( i+1UL, j             , (~C).load(i+1UL,j             ) + xmm5 * factor );
            (~C).store( i+1UL, j+SIMDSIZE    , (~C).load(i+1UL,j+SIMDSIZE    ) + xmm6 * factor );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, (~C).load(i+1UL,j+SIMDSIZE*2UL) + xmm7 * factor );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, (~C).load(i+1UL,j+SIMDSIZE*3UL) + xmm8 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j             );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE    );
               xmm3 = xmm3 + a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 = xmm4 + a1 * B.load(k,j+SIMDSIZE*3UL);
            }

            (~C).store( i, j             , (~C).load(i,j             ) + xmm1 * factor );
            (~C).store( i, j+SIMDSIZE    , (~C).load(i,j+SIMDSIZE    ) + xmm2 * factor );
            (~C).store( i, j+SIMDSIZE*2UL, (~C).load(i,j+SIMDSIZE*2UL) + xmm3 * factor );
            (~C).store( i, j+SIMDSIZE*3UL, (~C).load(i,j+SIMDSIZE*3UL) + xmm4 * factor );
         }
      }

      for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a1 * b2;
               xmm3 = xmm3 + a2 * b1;
               xmm4 = xmm4 + a2 * b2;
            }

            (~C).store( i    , j         , (~C).load(i    ,j         ) + xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE, (~C).load(i    ,j+SIMDSIZE) + xmm2 * factor );
            (~C).store( i+1UL, j         , (~C).load(i+1UL,j         ) + xmm3 * factor );
            (~C).store( i+1UL, j+SIMDSIZE, (~C).load(i+1UL,j+SIMDSIZE) + xmm4 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j         );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE);
            }

            (~C).store( i, j         , (~C).load(i,j         ) + xmm1 * factor );
            (~C).store( i, j+SIMDSIZE, (~C).load(i,j+SIMDSIZE) + xmm2 * factor );
         }
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL )
                               :( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 = xmm1 + set( A(i    ,k) ) * b1;
               xmm2 = xmm2 + set( A(i+1UL,k) ) * b1;
            }

            (~C).store( i    , j, (~C).load(i    ,j) + xmm1 * factor );
            (~C).store( i+1UL, j, (~C).load(i+1UL,j) + xmm2 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );

            SIMDType xmm1;

            for( size_t k=kbegin; k<K; ++k ) {
               xmm1 = xmm1 + set( A(i,k) ) * B.load(k,j);
            }

            (~C).store( i, j, (~C).load(i,j) + xmm1 * factor );
         }
      }

      for( ; remainder && j<N; ++j )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL )
                               :( K ) );

            ElementType value1 = ElementType();
            ElementType value2 = ElementType();

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 += A(i    ,k) * B(k,j);
               value2 += A(i+1UL,k) * B(k,j);
            }

            (~C)(i    ,j) += value1 * scalar;
            (~C)(i+1UL,j) += value2 * scalar;
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );

            ElementType value = ElementType();

            for( size_t k=kbegin; k<K; ++k ) {
               value += A(i,k) * B(k,j);
            }

            (~C)(i,j) += value * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Vectorized default addition assignment to column-major dense matrices (small matrices)******
   /*!\brief Vectorized default addition assignment of a small scaled transpose dense matrix-dense
   //        matrix multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default addition assignment of a scaled transpose
   // dense matrix-dense matrix multiplication expression to a column-major dense matrix. This
   // kernel is optimized for small matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectSmallAddAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT4>::value );

      const size_t ipos( remainder ? ( M & size_t(-SIMDSIZE) ) : M );
      BLAZE_INTERNAL_ASSERT( !remainder || ( M - ( M % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

      const SIMDType factor( set( scalar ) );

      size_t i( 0UL );

      for( ; (i+SIMDSIZE*7UL) < ipos; i+=SIMDSIZE*8UL ) {
         for( size_t j=0UL; j<N; ++j )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*8UL, K, ( IsStrictlyUpper<MT5>::value ? j : j+1UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j : j+1UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*8UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i             ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE    ,k) * b1;
               xmm3 = xmm3 + A.load(i+SIMDSIZE*2UL,k) * b1;
               xmm4 = xmm4 + A.load(i+SIMDSIZE*3UL,k) * b1;
               xmm5 = xmm5 + A.load(i+SIMDSIZE*4UL,k) * b1;
               xmm6 = xmm6 + A.load(i+SIMDSIZE*5UL,k) * b1;
               xmm7 = xmm7 + A.load(i+SIMDSIZE*6UL,k) * b1;
               xmm8 = xmm8 + A.load(i+SIMDSIZE*7UL,k) * b1;
            }

            (~C).store( i             , j, (~C).load(i             ,j) + xmm1 * factor );
            (~C).store( i+SIMDSIZE    , j, (~C).load(i+SIMDSIZE    ,j) + xmm2 * factor );
            (~C).store( i+SIMDSIZE*2UL, j, (~C).load(i+SIMDSIZE*2UL,j) + xmm3 * factor );
            (~C).store( i+SIMDSIZE*3UL, j, (~C).load(i+SIMDSIZE*3UL,j) + xmm4 * factor );
            (~C).store( i+SIMDSIZE*4UL, j, (~C).load(i+SIMDSIZE*4UL,j) + xmm5 * factor );
            (~C).store( i+SIMDSIZE*5UL, j, (~C).load(i+SIMDSIZE*5UL,j) + xmm6 * factor );
            (~C).store( i+SIMDSIZE*6UL, j, (~C).load(i+SIMDSIZE*6UL,j) + xmm7 * factor );
            (~C).store( i+SIMDSIZE*7UL, j, (~C).load(i+SIMDSIZE*7UL,j) + xmm8 * factor );
         }
      }

      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*4UL, K, ( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i             ,k) );
               const SIMDType a2( A.load(i+SIMDSIZE    ,k) );
               const SIMDType a3( A.load(i+SIMDSIZE*2UL,k) );
               const SIMDType a4( A.load(i+SIMDSIZE*3UL,k) );
               const SIMDType b1( set( B(k,j    ) ) );
               const SIMDType b2( set( B(k,j+1UL) ) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a2 * b1;
               xmm3 = xmm3 + a3 * b1;
               xmm4 = xmm4 + a4 * b1;
               xmm5 = xmm5 + a1 * b2;
               xmm6 = xmm6 + a2 * b2;
               xmm7 = xmm7 + a3 * b2;
               xmm8 = xmm8 + a4 * b2;
            }

            (~C).store( i             , j    , (~C).load(i             ,j    ) + xmm1 * factor );
            (~C).store( i+SIMDSIZE    , j    , (~C).load(i+SIMDSIZE    ,j    ) + xmm2 * factor );
            (~C).store( i+SIMDSIZE*2UL, j    , (~C).load(i+SIMDSIZE*2UL,j    ) + xmm3 * factor );
            (~C).store( i+SIMDSIZE*3UL, j    , (~C).load(i+SIMDSIZE*3UL,j    ) + xmm4 * factor );
            (~C).store( i             , j+1UL, (~C).load(i             ,j+1UL) + xmm5 * factor );
            (~C).store( i+SIMDSIZE    , j+1UL, (~C).load(i+SIMDSIZE    ,j+1UL) + xmm6 * factor );
            (~C).store( i+SIMDSIZE*2UL, j+1UL, (~C).load(i+SIMDSIZE*2UL,j+1UL) + xmm7 * factor );
            (~C).store( i+SIMDSIZE*3UL, j+1UL, (~C).load(i+SIMDSIZE*3UL,j+1UL) + xmm8 * factor );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i             ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE    ,k) * b1;
               xmm3 = xmm3 + A.load(i+SIMDSIZE*2UL,k) * b1;
               xmm4 = xmm4 + A.load(i+SIMDSIZE*3UL,k) * b1;
            }

            (~C).store( i             , j, (~C).load(i             ,j) + xmm1 * factor );
            (~C).store( i+SIMDSIZE    , j, (~C).load(i+SIMDSIZE    ,j) + xmm2 * factor );
            (~C).store( i+SIMDSIZE*2UL, j, (~C).load(i+SIMDSIZE*2UL,j) + xmm3 * factor );
            (~C).store( i+SIMDSIZE*3UL, j, (~C).load(i+SIMDSIZE*3UL,j) + xmm4 * factor );
         }
      }

      for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*2UL, K, ( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i         ,k) );
               const SIMDType a2( A.load(i+SIMDSIZE,k) );
               const SIMDType b1( set( B(k,j    ) ) );
               const SIMDType b2( set( B(k,j+1UL) ) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a2 * b1;
               xmm3 = xmm3 + a1 * b2;
               xmm4 = xmm4 + a2 * b2;
            }

            (~C).store( i         , j    , (~C).load(i         ,j    ) + xmm1 * factor );
            (~C).store( i+SIMDSIZE, j    , (~C).load(i+SIMDSIZE,j    ) + xmm2 * factor );
            (~C).store( i         , j+1UL, (~C).load(i         ,j+1UL) + xmm3 * factor );
            (~C).store( i+SIMDSIZE, j+1UL, (~C).load(i+SIMDSIZE,j+1UL) + xmm4 * factor );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i         ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE,k) * b1;
            }

            (~C).store( i         , j, (~C).load(i         ,j) + xmm1 * factor );
            (~C).store( i+SIMDSIZE, j, (~C).load(i+SIMDSIZE,j) + xmm2 * factor );
         }
      }

      for( ; i<ipos; i+=SIMDSIZE )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL )
                               :( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i,k) );
               xmm1 = xmm1 + a1 * set( B(k,j    ) );
               xmm2 = xmm2 + a1 * set( B(k,j+1UL) );
            }

            (~C).store( i, j    , (~C).load(i,j    ) + xmm1 * factor );
            (~C).store( i, j+1UL, (~C).load(i,j+1UL) + xmm2 * factor );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );

            SIMDType xmm1;

            for( size_t k=kbegin; k<K; ++k ) {
               xmm1 = xmm1 + A.load(i,k) * set( B(k,j) );
            }

            (~C).store( i, j, (~C).load(i,j) + xmm1 * factor );
         }
      }

      for( ; remainder && i<M; ++i )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL )
                               :( K ) );

            ElementType value1 = ElementType();
            ElementType value2 = ElementType();

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 += A(i,k) * B(k,j    );
               value2 += A(i,k) * B(k,j+1UL);
            }

            (~C)(i,j    ) += value1 * scalar;
            (~C)(i,j+1UL) += value2 * scalar;
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );

            ElementType value = ElementType();

            for( size_t k=kbegin; k<K; ++k ) {
               value += A(i,k) * B(k,j);
            }

            (~C)(i,j) += value * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default addition assignment to dense matrices (large matrices)******************************
   /*!\brief Default addition assignment of a large scaled transpose dense matrix-dense matrix
   //        multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a scaled
   // transpose dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectLargeAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectDefaultAddAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default addition assignment to row-major dense matrices (large matrices)*********
   /*!\brief Vectorized default addition assignment of a large scaled transpose dense matrix-dense
   //        matrix multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default addition assignment of a scaled transpose
   // dense matrix-dense matrix multiplication expression to a row-major dense matrix. This
   // kernel is optimized for large matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectLargeAddAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      const SIMDType factor( set( scalar ) );

      for( size_t jj=0UL; jj<N; jj+=DMATDMATMULT_DEFAULT_JBLOCK_SIZE )
      {
         const size_t jend( min( jj+DMATDMATMULT_DEFAULT_JBLOCK_SIZE, N ) );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

         for( size_t ii=0UL; ii<M; ii+=DMATDMATMULT_DEFAULT_IBLOCK_SIZE )
         {
            const size_t iend( min( ii+DMATDMATMULT_DEFAULT_IBLOCK_SIZE, M ) );

            for( size_t kk=0UL; kk<K; kk+=DMATDMATMULT_DEFAULT_KBLOCK_SIZE )
            {
               const size_t ktmp( min( kk+DMATDMATMULT_DEFAULT_KBLOCK_SIZE, K ) );

               size_t j( jj );

               for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
               {
                  const size_t j1( j+SIMDSIZE     );
                  const size_t j2( j+SIMDSIZE*2UL );
                  const size_t j3( j+SIMDSIZE*3UL );

                  size_t i( ii );

                  for( ; (i+2UL) <= iend; i+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+2UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        const SIMDType b3( B.load(k,j2) );
                        const SIMDType b4( B.load(k,j3) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a1 * b3;
                        xmm4 = xmm4 + a1 * b4;
                        xmm5 = xmm5 + a2 * b1;
                        xmm6 = xmm6 + a2 * b2;
                        xmm7 = xmm7 + a2 * b3;
                        xmm8 = xmm8 + a2 * b4;
                     }

                     (~C).store( i    , j , (~C).load(i    ,j ) + xmm1 * factor );
                     (~C).store( i    , j1, (~C).load(i    ,j1) + xmm2 * factor );
                     (~C).store( i    , j2, (~C).load(i    ,j2) + xmm3 * factor );
                     (~C).store( i    , j3, (~C).load(i    ,j3) + xmm4 * factor );
                     (~C).store( i+1UL, j , (~C).load(i+1UL,j ) + xmm5 * factor );
                     (~C).store( i+1UL, j1, (~C).load(i+1UL,j1) + xmm6 * factor );
                     (~C).store( i+1UL, j2, (~C).load(i+1UL,j2) + xmm7 * factor );
                     (~C).store( i+1UL, j3, (~C).load(i+1UL,j3) + xmm8 * factor );
                  }

                  if( i < iend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j );
                        xmm2 = xmm2 + a1 * B.load(k,j1);
                        xmm3 = xmm3 + a1 * B.load(k,j2);
                        xmm4 = xmm4 + a1 * B.load(k,j3);
                     }

                     (~C).store( i, j , (~C).load(i,j ) + xmm1 * factor );
                     (~C).store( i, j1, (~C).load(i,j1) + xmm2 * factor );
                     (~C).store( i, j2, (~C).load(i,j2) + xmm3 * factor );
                     (~C).store( i, j3, (~C).load(i,j3) + xmm4 * factor );
                  }
               }

               for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
               {
                  const size_t j1( j+SIMDSIZE );

                  size_t i( ii );

                  for( ; (i+4UL) <= iend; i+=4UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+4UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType a3( set( A(i+2UL,k) ) );
                        const SIMDType a4( set( A(i+3UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a2 * b1;
                        xmm4 = xmm4 + a2 * b2;
                        xmm5 = xmm5 + a3 * b1;
                        xmm6 = xmm6 + a3 * b2;
                        xmm7 = xmm7 + a4 * b1;
                        xmm8 = xmm8 + a4 * b2;
                     }

                     (~C).store( i    , j , (~C).load(i    ,j ) + xmm1 * factor );
                     (~C).store( i    , j1, (~C).load(i    ,j1) + xmm2 * factor );
                     (~C).store( i+1UL, j , (~C).load(i+1UL,j ) + xmm3 * factor );
                     (~C).store( i+1UL, j1, (~C).load(i+1UL,j1) + xmm4 * factor );
                     (~C).store( i+2UL, j , (~C).load(i+2UL,j ) + xmm5 * factor );
                     (~C).store( i+2UL, j1, (~C).load(i+2UL,j1) + xmm6 * factor );
                     (~C).store( i+3UL, j , (~C).load(i+3UL,j ) + xmm7 * factor );
                     (~C).store( i+3UL, j1, (~C).load(i+3UL,j1) + xmm8 * factor );
                  }

                  for( ; (i+2UL) <= iend; i+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+2UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a2 * b1;
                        xmm4 = xmm4 + a2 * b2;
                     }

                     (~C).store( i    , j , (~C).load(i    ,j ) + xmm1 * factor );
                     (~C).store( i    , j1, (~C).load(i    ,j1) + xmm2 * factor );
                     (~C).store( i+1UL, j , (~C).load(i+1UL,j ) + xmm3 * factor );
                     (~C).store( i+1UL, j1, (~C).load(i+1UL,j1) + xmm4 * factor );
                  }

                  if( i < iend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j );
                        xmm2 = xmm2 + a1 * B.load(k,j1);
                     }

                     (~C).store( i, j , (~C).load(i,j ) + xmm1 * factor );
                     (~C).store( i, j1, (~C).load(i,j1) + xmm2 * factor );
                  }
               }

               for( ; j<jpos; j+=SIMDSIZE )
               {
                  for( size_t i=ii; i<iend; ++i )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j);
                     }

                     (~C).store( i, j, (~C).load(i,j) + xmm1 * factor );
                  }
               }

               for( ; remainder && j<jend; ++j )
               {
                  for( size_t i=ii; i<iend; ++i )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+1UL, ktmp ) ):( ktmp ) ) );

                     ElementType value = ElementType();

                     for( size_t k=kbegin; k<kend; ++k ) {
                        value += A(i,k) * B(k,j);
                     }

                     (~C)(i,j) += value * scalar;
                  }
               }
            }
         }
      }
   }
   //**********************************************************************************************

   //**Vectorized default addition assignment to column-major dense matrices (large matrices)******
   /*!\brief Vectorized default addition assignment of a large scaled transpose dense matrix-dense
   //        matrix multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default addition assignment of a scaled transpose
   // dense matrix-dense matrix multiplication expression to a column-major dense matrix. This
   // kernel is optimized for large matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectLargeAddAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT4>::value );

      const SIMDType factor( set( scalar ) );

      for( size_t ii=0UL; ii<M; ii+=TDMATTDMATMULT_DEFAULT_IBLOCK_SIZE )
      {
         const size_t iend( min( ii+TDMATTDMATMULT_DEFAULT_IBLOCK_SIZE, M ) );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

         for( size_t jj=0UL; jj<N; jj+=TDMATTDMATMULT_DEFAULT_JBLOCK_SIZE )
         {
            const size_t jend( min( jj+TDMATTDMATMULT_DEFAULT_JBLOCK_SIZE, N ) );

            for( size_t kk=0UL; kk<K; kk+=TDMATTDMATMULT_DEFAULT_KBLOCK_SIZE )
            {
               const size_t ktmp( min( kk+TDMATTDMATMULT_DEFAULT_KBLOCK_SIZE, K ) );

               size_t i( ii );

               for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL )
               {
                  const size_t i1( i+SIMDSIZE     );
                  const size_t i2( i+SIMDSIZE*2UL );
                  const size_t i3( i+SIMDSIZE*3UL );

                  size_t j( jj );

                  for( ; (j+2UL) <= jend; j+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+2UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType a3( A.load(i2,k) );
                        const SIMDType a4( A.load(i3,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a3 * b1;
                        xmm4 = xmm4 + a4 * b1;
                        xmm5 = xmm5 + a1 * b2;
                        xmm6 = xmm6 + a2 * b2;
                        xmm7 = xmm7 + a3 * b2;
                        xmm8 = xmm8 + a4 * b2;
                     }

                     (~C).store( i , j    , (~C).load(i ,j    ) + xmm1 * factor );
                     (~C).store( i1, j    , (~C).load(i1,j    ) + xmm2 * factor );
                     (~C).store( i2, j    , (~C).load(i2,j    ) + xmm3 * factor );
                     (~C).store( i3, j    , (~C).load(i3,j    ) + xmm4 * factor );
                     (~C).store( i , j+1UL, (~C).load(i ,j+1UL) + xmm5 * factor );
                     (~C).store( i1, j+1UL, (~C).load(i1,j+1UL) + xmm6 * factor );
                     (~C).store( i2, j+1UL, (~C).load(i2,j+1UL) + xmm7 * factor );
                     (~C).store( i3, j+1UL, (~C).load(i3,j+1UL) + xmm8 * factor );
                  }

                  if( j < jend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i ,k) * b1;
                        xmm2 = xmm2 + A.load(i1,k) * b1;
                        xmm3 = xmm3 + A.load(i2,k) * b1;
                        xmm4 = xmm4 + A.load(i3,k) * b1;
                     }

                     (~C).store( i , j, (~C).load(i ,j) + xmm1 * factor );
                     (~C).store( i1, j, (~C).load(i1,j) + xmm2 * factor );
                     (~C).store( i2, j, (~C).load(i2,j) + xmm3 * factor );
                     (~C).store( i3, j, (~C).load(i3,j) + xmm4 * factor );
                  }
               }

               for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL )
               {
                  const size_t i1( i+SIMDSIZE );

                  size_t j( jj );

                  for( ; (j+4UL) <= jend; j+=4UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+4UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        const SIMDType b3( set( B(k,j+2UL) ) );
                        const SIMDType b4( set( B(k,j+3UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a1 * b2;
                        xmm4 = xmm4 + a2 * b2;
                        xmm5 = xmm5 + a1 * b3;
                        xmm6 = xmm6 + a2 * b3;
                        xmm7 = xmm7 + a1 * b4;
                        xmm8 = xmm8 + a2 * b4;
                     }

                     (~C).store( i , j    , (~C).load(i ,j    ) + xmm1 * factor );
                     (~C).store( i1, j    , (~C).load(i1,j    ) + xmm2 * factor );
                     (~C).store( i , j+1UL, (~C).load(i ,j+1UL) + xmm3 * factor );
                     (~C).store( i1, j+1UL, (~C).load(i1,j+1UL) + xmm4 * factor );
                     (~C).store( i , j+2UL, (~C).load(i ,j+2UL) + xmm5 * factor );
                     (~C).store( i1, j+2UL, (~C).load(i1,j+2UL) + xmm6 * factor );
                     (~C).store( i , j+3UL, (~C).load(i ,j+3UL) + xmm7 * factor );
                     (~C).store( i1, j+3UL, (~C).load(i1,j+3UL) + xmm8 * factor );
                  }

                  for( ; (j+2UL) <= jend; j+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+2UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a1 * b2;
                        xmm4 = xmm4 + a2 * b2;
                     }

                     (~C).store( i , j    , (~C).load(i ,j    ) + xmm1 * factor );
                     (~C).store( i1, j    , (~C).load(i1,j    ) + xmm2 * factor );
                     (~C).store( i , j+1UL, (~C).load(i ,j+1UL) + xmm3 * factor );
                     (~C).store( i1, j+1UL, (~C).load(i1,j+1UL) + xmm4 * factor );
                  }

                  if( j < jend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i ,k) * b1;
                        xmm2 = xmm2 + A.load(i1,k) * b1;
                     }

                     (~C).store( i , j, (~C).load(i ,j) + xmm1 * factor );
                     (~C).store( i1, j, (~C).load(i1,j) + xmm2 * factor );
                  }
               }

               for( ; i<ipos; i+=SIMDSIZE )
               {
                  for( size_t j=jj; j<jend; ++j )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i,k) * b1;
                     }

                     (~C).store( i, j, (~C).load(i,j) + xmm1 * factor );
                  }
               }

               for( ; remainder && i<iend; ++i )
               {
                  for( size_t j=jj; j<jend; ++j )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+1UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     ElementType value = ElementType();

                     for( size_t k=kbegin; k<kend; ++k ) {
                        value += A(i,k) * B(k,j);
                     }

                     (~C)(i,j) += value * scalar;
                  }
               }
            }
         }
      }
   }
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense matrices (default)**********************************
   /*!\brief Default addition assignment of a scaled transpose dense matrix-dense matrix
   //        multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a large
   // scaled transpose dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseBlasKernel<MT3,MT4,MT5,ST2> >
      selectBlasAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectLargeAddAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense matrices********************************************
#if BLAZE_BLAS_MODE
   /*!\brief BLAS-based addition assignment of a scaled transpose dense matrix-dense matrix
   //        multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function performs the scaled transpose dense matrix-dense matrix multiplication based
   // on the according BLAS functionality.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseBlasKernel<MT3,MT4,MT5,ST2> >
      selectBlasAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      typedef ElementType_<MT3>  ET;

      if( IsTriangular<MT4>::value ) {
         ResultType_<MT3> tmp( serial( B ) );
         trmm( tmp, A, CblasLeft, ( IsLower<MT4>::value )?( CblasLower ):( CblasUpper ), ET(scalar) );
         addAssign( C, tmp );
      }
      else if( IsTriangular<MT5>::value ) {
         ResultType_<MT3> tmp( serial( A ) );
         trmm( tmp, B, CblasRight, ( IsLower<MT5>::value )?( CblasLower ):( CblasUpper ), ET(scalar) );
         addAssign( C, tmp );
      }
      else {
         gemm( C, A, B, ET(scalar), ET(1) );
      }
   }
#endif
   //**********************************************************************************************

   //**Addition assignment to sparse matrices******************************************************
   // No special implementation for the addition assignment to sparse matrices.
   //**********************************************************************************************

   //**Subtraction assignment to dense matrices****************************************************
   /*!\brief Subtraction assignment of a scaled transpose dense matrix-dense matrix multiplication
   //        to a dense matrix (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a scaled transpose
   // dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline void subAssign( DenseMatrix<MT,SO>& lhs, const DMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LeftOperand_<MMM>  left ( rhs.matrix_.leftOperand()  );
      RightOperand_<MMM> right( rhs.matrix_.rightOperand() );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || left.columns() == 0UL ) {
         return;
      }

      LT A( serial( left  ) );  // Evaluation of the left-hand side dense matrix operand
      RT B( serial( right ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == right.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == right.columns() , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns(), "Invalid number of columns" );

      DMatScalarMultExpr::selectSubAssignKernel( ~lhs, A, B, rhs.scalar_ );
   }
   //**********************************************************************************************

   //**Subtraction assignment to dense matrices (kernel selection)*********************************
   /*!\brief Selection of the kernel for a subtraction assignment of a scaled transpose dense
   //        matrix-dense matrix multiplication to a dense matrix (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      if( ( IsDiagonal<MT4>::value && IsDiagonal<MT5>::value ) ||
          ( C.rows() * C.columns() < TDMATDMATMULT_THRESHOLD ) )
         selectSmallSubAssignKernel( C, A, B, scalar );
      else
         selectBlasSubAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense matrices********************************************
   /*!\brief Default subtraction assignment of a scaled transpose dense matrix-dense matrix
   //        multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default subtraction assignment of a scaled transpose dense
   // matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, Not< IsDiagonal<MT5> > > >
      selectDefaultSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const ResultType tmp( serial( A * B * scalar ) );
      subAssign( C, tmp );
   }
   //**********************************************************************************************

   //**Default subtraction assignment to row-major dense matrices (general/diagonal)***************
   /*!\brief Default subtraction assignment of a scaled general transpose dense matrix-diagonal
   //        dense matrix multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default subtraction assignment of a scaled general transpose
   // dense matrix-diagonal dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, IsDiagonal<MT5> > >
      selectDefaultSubAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      const size_t block( BLOCK_SIZE );

      for( size_t ii=0UL; ii<M; ii+=block ) {
         const size_t iend( min( M, ii+block ) );
         for( size_t jj=0UL; jj<N; jj+=block ) {
            const size_t jend( min( N, jj+block ) );
            for( size_t i=ii; i<iend; ++i )
            {
               const size_t jbegin( ( IsUpper<MT4>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), jj ) )
                                    :( jj ) );
               const size_t jpos( ( IsLower<MT4>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i : i+1UL ), jend ) )
                                  :( jend ) );

               for( size_t j=jbegin; j<jpos; ++j ) {
                  (~C)(i,j) -= A(i,j) * B(j,j) * scalar;
               }
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default subtraction assignment to column-major dense matrices (general/diagonal)************
   /*!\brief Default subtraction assignment of a scaled general transpose dense matrix-diagonal
   //        dense matrix multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default subtraction assignment of a scaled general transpose
   // dense matrix-diagonal dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< Not< IsDiagonal<MT4> >, IsDiagonal<MT5> > >
      selectDefaultSubAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t j=0UL; j<N; ++j )
      {
         const size_t ibegin( ( IsLower<MT4>::value )
                              ?( IsStrictlyLower<MT4>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT4>::value )
                            ?( IsStrictlyUpper<MT4>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t inum( iend - ibegin );
         const size_t ipos( ibegin + ( inum & size_t(-2) ) );

         for( size_t i=ibegin; i<ipos; i+=2UL ) {
            (~C)(i    ,j) -= A(i    ,j) * B(j,j) * scalar;
            (~C)(i+1UL,j) -= A(i+1UL,j) * B(j,j) * scalar;
         }
         if( ipos < iend ) {
            (~C)(ipos,j) -= A(ipos,j) * B(j,j) * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default subtraction assignment to row-major dense matrices (diagonal/general)***************
   /*!\brief Default subtraction assignment of a scaled diagonal transpose dense matrix-general
   //        dense matrix multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default subtraction assignment of a scaled diagonal transpose
   // dense matrix-general dense matrix multiplication expression to a row-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< IsDiagonal<MT4>, Not< IsDiagonal<MT5> > > >
      selectDefaultSubAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper<MT5>::value )
                              ?( IsStrictlyUpper<MT5>::value ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT5>::value )
                            ?( IsStrictlyLower<MT5>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jnum( jend - jbegin );
         const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

         for( size_t j=jbegin; j<jpos; j+=2UL ) {
            (~C)(i,j    ) -= A(i,i) * B(i,j    ) * scalar;
            (~C)(i,j+1UL) -= A(i,i) * B(i,j+1UL) * scalar;
         }
         if( jpos < jend ) {
            (~C)(i,jpos) -= A(i,i) * B(i,jpos) * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default subtraction assignment to column-major dense matrices (diagonal/general)************
   /*!\brief Default subtraction assignment of a scaled diagonal transpose dense matrix-general
   //        dense matrix multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default subtraction assignment of a scaled diagonal transpose
   // dense matrix-general dense matrix multiplication expression to a column-major dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< IsDiagonal<MT4>, Not< IsDiagonal<MT5> > > >
      selectDefaultSubAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      const size_t block( BLOCK_SIZE );

      for( size_t jj=0UL; jj<N; jj+=block ) {
         const size_t jend( min( N, jj+block ) );
         for( size_t ii=0UL; ii<M; ii+=block ) {
            const size_t iend( min( M, ii+block ) );
            for( size_t j=jj; j<jend; ++j )
            {
               const size_t ibegin( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyLower<MT5>::value ? j+1UL : j ), ii ) )
                                    :( ii ) );
               const size_t ipos( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyUpper<MT5>::value ? j : j+1UL ), iend ) )
                                  :( iend ) );

               for( size_t i=ibegin; i<ipos; ++i ) {
                  (~C)(i,j) -= A(i,i) * B(i,j) * scalar;
               }
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense matrices (diagonal/diagonal)************************
   /*!\brief Default subtraction assignment of a scaled diagonal transpose dense matrix-diagonal
   //        dense matrix multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default subtraction assignment of a scaled diagonal transpose
   // dense matrix-diagonal dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< And< IsDiagonal<MT4>, IsDiagonal<MT5> > >
      selectDefaultSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      for( size_t i=0UL; i<A.rows(); ++i ) {
         C(i,i) -= A(i,i) * B(i,i) * scalar;
      }
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense matrices (small matrices)***************************
   /*!\brief Default subtraction assignment of a small scaled transpose dense matrix-dense matrix
   //        multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a scaled
   // transpose dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectSmallSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectDefaultSubAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to row-major dense matrices (small matrices)******
   /*!\brief Vectorized default subtraction assignment of a small scaled transpose dense matrix-
   //        dense matrix multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment of a scaled transpose
   // dense matrix-dense matrix multiplication expression to a row-major dense matrix. This kernel
   // is optimized for small matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectSmallSubAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      const SIMDType factor( set( scalar ) );

      size_t j( 0UL );

      for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL ) {
         for( size_t i=0UL; i<M; ++i )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i : i+1UL ), j+SIMDSIZE*8UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i : i+1UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*8UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j             );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE    );
               xmm3 = xmm3 + a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 = xmm4 + a1 * B.load(k,j+SIMDSIZE*3UL);
               xmm5 = xmm5 + a1 * B.load(k,j+SIMDSIZE*4UL);
               xmm6 = xmm6 + a1 * B.load(k,j+SIMDSIZE*5UL);
               xmm7 = xmm7 + a1 * B.load(k,j+SIMDSIZE*6UL);
               xmm8 = xmm8 + a1 * B.load(k,j+SIMDSIZE*7UL);
            }

            (~C).store( i, j             , (~C).load(i,j             ) - xmm1 * factor );
            (~C).store( i, j+SIMDSIZE    , (~C).load(i,j+SIMDSIZE    ) - xmm2 * factor );
            (~C).store( i, j+SIMDSIZE*2UL, (~C).load(i,j+SIMDSIZE*2UL) - xmm3 * factor );
            (~C).store( i, j+SIMDSIZE*3UL, (~C).load(i,j+SIMDSIZE*3UL) - xmm4 * factor );
            (~C).store( i, j+SIMDSIZE*4UL, (~C).load(i,j+SIMDSIZE*4UL) - xmm5 * factor );
            (~C).store( i, j+SIMDSIZE*5UL, (~C).load(i,j+SIMDSIZE*5UL) - xmm6 * factor );
            (~C).store( i, j+SIMDSIZE*6UL, (~C).load(i,j+SIMDSIZE*6UL) - xmm7 * factor );
            (~C).store( i, j+SIMDSIZE*7UL, (~C).load(i,j+SIMDSIZE*7UL) - xmm8 * factor );
         }
      }

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ), j+SIMDSIZE*4UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a1 * b2;
               xmm3 = xmm3 + a1 * b3;
               xmm4 = xmm4 + a1 * b4;
               xmm5 = xmm5 + a2 * b1;
               xmm6 = xmm6 + a2 * b2;
               xmm7 = xmm7 + a2 * b3;
               xmm8 = xmm8 + a2 * b4;
            }

            (~C).store( i    , j             , (~C).load(i    ,j             ) - xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE    , (~C).load(i    ,j+SIMDSIZE    ) - xmm2 * factor );
            (~C).store( i    , j+SIMDSIZE*2UL, (~C).load(i    ,j+SIMDSIZE*2UL) - xmm3 * factor );
            (~C).store( i    , j+SIMDSIZE*3UL, (~C).load(i    ,j+SIMDSIZE*3UL) - xmm4 * factor );
            (~C).store( i+1UL, j             , (~C).load(i+1UL,j             ) - xmm5 * factor );
            (~C).store( i+1UL, j+SIMDSIZE    , (~C).load(i+1UL,j+SIMDSIZE    ) - xmm6 * factor );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, (~C).load(i+1UL,j+SIMDSIZE*2UL) - xmm7 * factor );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, (~C).load(i+1UL,j+SIMDSIZE*3UL) - xmm8 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j             );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE    );
               xmm3 = xmm3 + a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 = xmm4 + a1 * B.load(k,j+SIMDSIZE*3UL);
            }

            (~C).store( i, j             , (~C).load(i,j             ) - xmm1 * factor );
            (~C).store( i, j+SIMDSIZE    , (~C).load(i,j+SIMDSIZE    ) - xmm2 * factor );
            (~C).store( i, j+SIMDSIZE*2UL, (~C).load(i,j+SIMDSIZE*2UL) - xmm3 * factor );
            (~C).store( i, j+SIMDSIZE*3UL, (~C).load(i,j+SIMDSIZE*3UL) - xmm4 * factor );
         }
      }

      for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( ( IsUpper<MT5>::value )
                                  ?( min( ( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL ) )
                               :( IsUpper<MT5>::value ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a1 * b2;
               xmm3 = xmm3 + a2 * b1;
               xmm4 = xmm4 + a2 * b2;
            }

            (~C).store( i    , j         , (~C).load(i    ,j         ) - xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE, (~C).load(i    ,j+SIMDSIZE) - xmm2 * factor );
            (~C).store( i+1UL, j         , (~C).load(i+1UL,j         ) - xmm3 * factor );
            (~C).store( i+1UL, j+SIMDSIZE, (~C).load(i+1UL,j+SIMDSIZE) - xmm4 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 = xmm1 + a1 * B.load(k,j         );
               xmm2 = xmm2 + a1 * B.load(k,j+SIMDSIZE);
            }

            (~C).store( i, j         , (~C).load(i,j         ) - xmm1 * factor );
            (~C).store( i, j+SIMDSIZE, (~C).load(i,j+SIMDSIZE) - xmm2 * factor );
         }
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL )
                               :( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 = xmm1 + set( A(i    ,k) ) * b1;
               xmm2 = xmm2 + set( A(i+1UL,k) ) * b1;
            }

            (~C).store( i    , j, (~C).load(i    ,j) - xmm1 * factor );
            (~C).store( i+1UL, j, (~C).load(i+1UL,j) - xmm2 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );

            SIMDType xmm1;

            for( size_t k=kbegin; k<K; ++k ) {
               xmm1 = xmm1 + set( A(i,k) ) * B.load(k,j);
            }

            (~C).store( i, j, (~C).load(i,j) - xmm1 * factor );
         }
      }

      for( ; remainder && j<N; ++j )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )
                               ?( IsStrictlyLower<MT4>::value ? i+1UL : i+2UL )
                               :( K ) );

            ElementType value1 = ElementType();
            ElementType value2 = ElementType();

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 += A(i    ,k) * B(k,j);
               value2 += A(i+1UL,k) * B(k,j);
            }

            (~C)(i    ,j) -= value1 * scalar;
            (~C)(i+1UL,j) -= value2 * scalar;
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper<MT4>::value )
                                 ?( ( IsLower<MT5>::value )
                                    ?( max( ( IsStrictlyUpper<MT4>::value ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper<MT4>::value ? i+1UL : i ) )
                                 :( IsLower<MT5>::value ? j : 0UL ) );

            ElementType value = ElementType();

            for( size_t k=kbegin; k<K; ++k ) {
               value += A(i,k) * B(k,j);
            }

            (~C)(i,j) -= value * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to column-major dense matrices (small matrices)***
   /*!\brief Vectorized default subtraction assignment of a small scaled transpose dense matrix-
   //        dense matrix multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment of a scaled transpose
   // dense matrix-dense matrix multiplication expression to a column-major dense matrix. This
   // kernel is optimized for small matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectSmallSubAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT4>::value );

      const size_t ipos( remainder ? ( M & size_t(-SIMDSIZE) ) : M );
      BLAZE_INTERNAL_ASSERT( !remainder || ( M - ( M % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

      const SIMDType factor( set( scalar ) );

      size_t i( 0UL );

      for( ; (i+SIMDSIZE*7UL) < ipos; i+=SIMDSIZE*8UL ) {
         for( size_t j=0UL; j<N; ++j )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*8UL, K, ( IsStrictlyUpper<MT5>::value ? j : j+1UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j : j+1UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*8UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i             ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE    ,k) * b1;
               xmm3 = xmm3 + A.load(i+SIMDSIZE*2UL,k) * b1;
               xmm4 = xmm4 + A.load(i+SIMDSIZE*3UL,k) * b1;
               xmm5 = xmm5 + A.load(i+SIMDSIZE*4UL,k) * b1;
               xmm6 = xmm6 + A.load(i+SIMDSIZE*5UL,k) * b1;
               xmm7 = xmm7 + A.load(i+SIMDSIZE*6UL,k) * b1;
               xmm8 = xmm8 + A.load(i+SIMDSIZE*7UL,k) * b1;
            }

            (~C).store( i             , j, (~C).load(i             ,j) - xmm1 * factor );
            (~C).store( i+SIMDSIZE    , j, (~C).load(i+SIMDSIZE    ,j) - xmm2 * factor );
            (~C).store( i+SIMDSIZE*2UL, j, (~C).load(i+SIMDSIZE*2UL,j) - xmm3 * factor );
            (~C).store( i+SIMDSIZE*3UL, j, (~C).load(i+SIMDSIZE*3UL,j) - xmm4 * factor );
            (~C).store( i+SIMDSIZE*4UL, j, (~C).load(i+SIMDSIZE*4UL,j) - xmm5 * factor );
            (~C).store( i+SIMDSIZE*5UL, j, (~C).load(i+SIMDSIZE*5UL,j) - xmm6 * factor );
            (~C).store( i+SIMDSIZE*6UL, j, (~C).load(i+SIMDSIZE*6UL,j) - xmm7 * factor );
            (~C).store( i+SIMDSIZE*7UL, j, (~C).load(i+SIMDSIZE*7UL,j) - xmm8 * factor );
         }
      }

      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*4UL, K, ( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i             ,k) );
               const SIMDType a2( A.load(i+SIMDSIZE    ,k) );
               const SIMDType a3( A.load(i+SIMDSIZE*2UL,k) );
               const SIMDType a4( A.load(i+SIMDSIZE*3UL,k) );
               const SIMDType b1( set( B(k,j    ) ) );
               const SIMDType b2( set( B(k,j+1UL) ) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a2 * b1;
               xmm3 = xmm3 + a3 * b1;
               xmm4 = xmm4 + a4 * b1;
               xmm5 = xmm5 + a1 * b2;
               xmm6 = xmm6 + a2 * b2;
               xmm7 = xmm7 + a3 * b2;
               xmm8 = xmm8 + a4 * b2;
            }

            (~C).store( i             , j    , (~C).load(i             ,j    ) - xmm1 * factor );
            (~C).store( i+SIMDSIZE    , j    , (~C).load(i+SIMDSIZE    ,j    ) - xmm2 * factor );
            (~C).store( i+SIMDSIZE*2UL, j    , (~C).load(i+SIMDSIZE*2UL,j    ) - xmm3 * factor );
            (~C).store( i+SIMDSIZE*3UL, j    , (~C).load(i+SIMDSIZE*3UL,j    ) - xmm4 * factor );
            (~C).store( i             , j+1UL, (~C).load(i             ,j+1UL) - xmm5 * factor );
            (~C).store( i+SIMDSIZE    , j+1UL, (~C).load(i+SIMDSIZE    ,j+1UL) - xmm6 * factor );
            (~C).store( i+SIMDSIZE*2UL, j+1UL, (~C).load(i+SIMDSIZE*2UL,j+1UL) - xmm7 * factor );
            (~C).store( i+SIMDSIZE*3UL, j+1UL, (~C).load(i+SIMDSIZE*3UL,j+1UL) - xmm8 * factor );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i             ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE    ,k) * b1;
               xmm3 = xmm3 + A.load(i+SIMDSIZE*2UL,k) * b1;
               xmm4 = xmm4 + A.load(i+SIMDSIZE*3UL,k) * b1;
            }

            (~C).store( i             , j, (~C).load(i             ,j) - xmm1 * factor );
            (~C).store( i+SIMDSIZE    , j, (~C).load(i+SIMDSIZE    ,j) - xmm2 * factor );
            (~C).store( i+SIMDSIZE*2UL, j, (~C).load(i+SIMDSIZE*2UL,j) - xmm3 * factor );
            (~C).store( i+SIMDSIZE*3UL, j, (~C).load(i+SIMDSIZE*3UL,j) - xmm4 * factor );
         }
      }

      for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( ( IsLower<MT4>::value )
                                  ?( min( i+SIMDSIZE*2UL, K, ( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) ) )
                                  :( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL ) )
                               :( IsLower<MT4>::value ? min( i+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i         ,k) );
               const SIMDType a2( A.load(i+SIMDSIZE,k) );
               const SIMDType b1( set( B(k,j    ) ) );
               const SIMDType b2( set( B(k,j+1UL) ) );
               xmm1 = xmm1 + a1 * b1;
               xmm2 = xmm2 + a2 * b1;
               xmm3 = xmm3 + a1 * b2;
               xmm4 = xmm4 + a2 * b2;
            }

            (~C).store( i         , j    , (~C).load(i         ,j    ) - xmm1 * factor );
            (~C).store( i+SIMDSIZE, j    , (~C).load(i+SIMDSIZE,j    ) - xmm2 * factor );
            (~C).store( i         , j+1UL, (~C).load(i         ,j+1UL) - xmm3 * factor );
            (~C).store( i+SIMDSIZE, j+1UL, (~C).load(i+SIMDSIZE,j+1UL) - xmm4 * factor );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType b1( set( B(k,j) ) );
               xmm1 = xmm1 + A.load(i         ,k) * b1;
               xmm2 = xmm2 + A.load(i+SIMDSIZE,k) * b1;
            }

            (~C).store( i         , j, (~C).load(i         ,j) - xmm1 * factor );
            (~C).store( i+SIMDSIZE, j, (~C).load(i+SIMDSIZE,j) - xmm2 * factor );
         }
      }

      for( ; i<ipos; i+=SIMDSIZE )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL )
                               :( K ) );

            SIMDType xmm1, xmm2;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( A.load(i,k) );
               xmm1 = xmm1 + a1 * set( B(k,j    ) );
               xmm2 = xmm2 + a1 * set( B(k,j+1UL) );
            }

            (~C).store( i, j    , (~C).load(i,j    ) - xmm1 * factor );
            (~C).store( i, j+1UL, (~C).load(i,j+1UL) - xmm2 * factor );
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );

            SIMDType xmm1;

            for( size_t k=kbegin; k<K; ++k ) {
               xmm1 = xmm1 + A.load(i,k) * set( B(k,j) );
            }

            (~C).store( i, j, (~C).load(i,j) - xmm1 * factor );
         }
      }

      for( ; remainder && i<M; ++i )
      {
         size_t j( 0UL );

         for( ; (j+2UL) <= N; j+=2UL )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );
            const size_t kend( ( IsUpper<MT5>::value )
                               ?( IsStrictlyUpper<MT5>::value ? j+1UL : j+2UL )
                               :( K ) );

            ElementType value1 = ElementType();
            ElementType value2 = ElementType();

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 += A(i,k) * B(k,j    );
               value2 += A(i,k) * B(k,j+1UL);
            }

            (~C)(i,j    ) -= value1 * scalar;
            (~C)(i,j+1UL) -= value2 * scalar;
         }

         if( j < N )
         {
            const size_t kbegin( ( IsLower<MT5>::value )
                                 ?( ( IsUpper<MT4>::value )
                                    ?( max( i, ( IsStrictlyLower<MT5>::value ? j+1UL : j ) ) )
                                    :( IsStrictlyLower<MT5>::value ? j+1UL : j ) )
                                 :( IsUpper<MT4>::value ? i : 0UL ) );

            ElementType value = ElementType();

            for( size_t k=kbegin; k<K; ++k ) {
               value += A(i,k) * B(k,j);
            }

            (~C)(i,j) -= value * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense matrices (large matrices)***************************
   /*!\brief Default subtraction assignment of a large scaled transpose dense matrix-dense matrix
   //        multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a scaled
   // transpose dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectLargeSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectDefaultSubAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to row-major dense matrices (large matrices)******
   /*!\brief Vectorized default subtraction assignment of a large scaled transpose dense matrix-
   //        dense matrix multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment of a scaled transpose
   // dense matrix-dense matrix multiplication expression to a row-major dense matrix. This kernel
   // is optimized for large matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectLargeSubAssignKernel( DenseMatrix<MT3,false>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT5>::value );

      const SIMDType factor( set( scalar ) );

      for( size_t jj=0UL; jj<N; jj+=DMATDMATMULT_DEFAULT_JBLOCK_SIZE )
      {
         const size_t jend( min( jj+DMATDMATMULT_DEFAULT_JBLOCK_SIZE, N ) );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

         for( size_t ii=0UL; ii<M; ii+=DMATDMATMULT_DEFAULT_IBLOCK_SIZE )
         {
            const size_t iend( min( ii+DMATDMATMULT_DEFAULT_IBLOCK_SIZE, M ) );

            for( size_t kk=0UL; kk<K; kk+=DMATDMATMULT_DEFAULT_KBLOCK_SIZE )
            {
               const size_t ktmp( min( kk+DMATDMATMULT_DEFAULT_KBLOCK_SIZE, K ) );

               size_t j( jj );

               for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
               {
                  const size_t j1( j+SIMDSIZE     );
                  const size_t j2( j+SIMDSIZE*2UL );
                  const size_t j3( j+SIMDSIZE*3UL );

                  size_t i( ii );

                  for( ; (i+2UL) <= iend; i+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+2UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        const SIMDType b3( B.load(k,j2) );
                        const SIMDType b4( B.load(k,j3) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a1 * b3;
                        xmm4 = xmm4 + a1 * b4;
                        xmm5 = xmm5 + a2 * b1;
                        xmm6 = xmm6 + a2 * b2;
                        xmm7 = xmm7 + a2 * b3;
                        xmm8 = xmm8 + a2 * b4;
                     }

                     (~C).store( i    , j , (~C).load(i    ,j ) - xmm1 * factor );
                     (~C).store( i    , j1, (~C).load(i    ,j1) - xmm2 * factor );
                     (~C).store( i    , j2, (~C).load(i    ,j2) - xmm3 * factor );
                     (~C).store( i    , j3, (~C).load(i    ,j3) - xmm4 * factor );
                     (~C).store( i+1UL, j , (~C).load(i+1UL,j ) - xmm5 * factor );
                     (~C).store( i+1UL, j1, (~C).load(i+1UL,j1) - xmm6 * factor );
                     (~C).store( i+1UL, j2, (~C).load(i+1UL,j2) - xmm7 * factor );
                     (~C).store( i+1UL, j3, (~C).load(i+1UL,j3) - xmm8 * factor );
                  }

                  if( i < iend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*4UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j );
                        xmm2 = xmm2 + a1 * B.load(k,j1);
                        xmm3 = xmm3 + a1 * B.load(k,j2);
                        xmm4 = xmm4 + a1 * B.load(k,j3);
                     }

                     (~C).store( i, j , (~C).load(i,j ) - xmm1 * factor );
                     (~C).store( i, j1, (~C).load(i,j1) - xmm2 * factor );
                     (~C).store( i, j2, (~C).load(i,j2) - xmm3 * factor );
                     (~C).store( i, j3, (~C).load(i,j3) - xmm4 * factor );
                  }
               }

               for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
               {
                  const size_t j1( j+SIMDSIZE );

                  size_t i( ii );

                  for( ; (i+4UL) <= iend; i+=4UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+4UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType a3( set( A(i+2UL,k) ) );
                        const SIMDType a4( set( A(i+3UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a2 * b1;
                        xmm4 = xmm4 + a2 * b2;
                        xmm5 = xmm5 + a3 * b1;
                        xmm6 = xmm6 + a3 * b2;
                        xmm7 = xmm7 + a4 * b1;
                        xmm8 = xmm8 + a4 * b2;
                     }

                     (~C).store( i    , j , (~C).load(i    ,j ) - xmm1 * factor );
                     (~C).store( i    , j1, (~C).load(i    ,j1) - xmm2 * factor );
                     (~C).store( i+1UL, j , (~C).load(i+1UL,j ) - xmm3 * factor );
                     (~C).store( i+1UL, j1, (~C).load(i+1UL,j1) - xmm4 * factor );
                     (~C).store( i+2UL, j , (~C).load(i+2UL,j ) - xmm5 * factor );
                     (~C).store( i+2UL, j1, (~C).load(i+2UL,j1) - xmm6 * factor );
                     (~C).store( i+3UL, j , (~C).load(i+3UL,j ) - xmm7 * factor );
                     (~C).store( i+3UL, j1, (~C).load(i+3UL,j1) - xmm8 * factor );
                  }

                  for( ; (i+2UL) <= iend; i+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+2UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i    ,k) ) );
                        const SIMDType a2( set( A(i+1UL,k) ) );
                        const SIMDType b1( B.load(k,j ) );
                        const SIMDType b2( B.load(k,j1) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a1 * b2;
                        xmm3 = xmm3 + a2 * b1;
                        xmm4 = xmm4 + a2 * b2;
                     }

                     (~C).store( i    , j , (~C).load(i    ,j ) - xmm1 * factor );
                     (~C).store( i    , j1, (~C).load(i    ,j1) - xmm2 * factor );
                     (~C).store( i+1UL, j , (~C).load(i+1UL,j ) - xmm3 * factor );
                     (~C).store( i+1UL, j1, (~C).load(i+1UL,j1) - xmm4 * factor );
                  }

                  if( i < iend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE*2UL, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1, xmm2;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j );
                        xmm2 = xmm2 + a1 * B.load(k,j1);
                     }

                     (~C).store( i, j , (~C).load(i,j ) - xmm1 * factor );
                     (~C).store( i, j1, (~C).load(i,j1) - xmm2 * factor );
                  }
               }

               for( ; j<jpos; j+=SIMDSIZE )
               {
                  for( size_t i=ii; i<iend; ++i )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+SIMDSIZE, ktmp ) ):( ktmp ) ) );

                     SIMDType xmm1;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( set( A(i,k) ) );
                        xmm1 = xmm1 + a1 * B.load(k,j);
                     }

                     (~C).store( i, j, (~C).load(i,j) - xmm1 * factor );
                  }
               }

               for( ; remainder && j<jend; ++j )
               {
                  for( size_t i=ii; i<iend; ++i )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( i+1UL ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( min( j+1UL, ktmp ) ):( ktmp ) ) );

                     ElementType value = ElementType();

                     for( size_t k=kbegin; k<kend; ++k ) {
                        value += A(i,k) * B(k,j);
                     }

                     (~C)(i,j) -= value * scalar;
                  }
               }
            }
         }
      }
   }
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to column-major dense matrices (large matrices)***
   /*!\brief Vectorized default subtraction assignment of a large scaled transpose dense matrix-
   //        dense matrix multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment of a scaled transpose
   // dense matrix-dense matrix multiplication expression to a column-major dense matrix. This
   // kernel is optimized for large matrices.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<MT3,MT4,MT5,ST2> >
      selectLargeSubAssignKernel( DenseMatrix<MT3,true>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      const bool remainder( !IsPadded<MT3>::value || !IsPadded<MT4>::value );

      const SIMDType factor( set( scalar ) );

      for( size_t ii=0UL; ii<M; ii+=TDMATTDMATMULT_DEFAULT_IBLOCK_SIZE )
      {
         const size_t iend( min( ii+TDMATTDMATMULT_DEFAULT_IBLOCK_SIZE, M ) );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

         for( size_t jj=0UL; jj<N; jj+=TDMATTDMATMULT_DEFAULT_JBLOCK_SIZE )
         {
            const size_t jend( min( jj+TDMATTDMATMULT_DEFAULT_JBLOCK_SIZE, N ) );

            for( size_t kk=0UL; kk<K; kk+=TDMATTDMATMULT_DEFAULT_KBLOCK_SIZE )
            {
               const size_t ktmp( min( kk+TDMATTDMATMULT_DEFAULT_KBLOCK_SIZE, K ) );

               size_t i( ii );

               for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL )
               {
                  const size_t i1( i+SIMDSIZE     );
                  const size_t i2( i+SIMDSIZE*2UL );
                  const size_t i3( i+SIMDSIZE*3UL );

                  size_t j( jj );

                  for( ; (j+2UL) <= jend; j+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+2UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType a3( A.load(i2,k) );
                        const SIMDType a4( A.load(i3,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a3 * b1;
                        xmm4 = xmm4 + a4 * b1;
                        xmm5 = xmm5 + a1 * b2;
                        xmm6 = xmm6 + a2 * b2;
                        xmm7 = xmm7 + a3 * b2;
                        xmm8 = xmm8 + a4 * b2;
                     }

                     (~C).store( i , j    , (~C).load(i ,j    ) - xmm1 * factor );
                     (~C).store( i1, j    , (~C).load(i1,j    ) - xmm2 * factor );
                     (~C).store( i2, j    , (~C).load(i2,j    ) - xmm3 * factor );
                     (~C).store( i3, j    , (~C).load(i3,j    ) - xmm4 * factor );
                     (~C).store( i , j+1UL, (~C).load(i ,j+1UL) - xmm5 * factor );
                     (~C).store( i1, j+1UL, (~C).load(i1,j+1UL) - xmm6 * factor );
                     (~C).store( i2, j+1UL, (~C).load(i2,j+1UL) - xmm7 * factor );
                     (~C).store( i3, j+1UL, (~C).load(i3,j+1UL) - xmm8 * factor );
                  }

                  if( j < jend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*4UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i ,k) * b1;
                        xmm2 = xmm2 + A.load(i1,k) * b1;
                        xmm3 = xmm3 + A.load(i2,k) * b1;
                        xmm4 = xmm4 + A.load(i3,k) * b1;
                     }

                     (~C).store( i , j, (~C).load(i ,j) - xmm1 * factor );
                     (~C).store( i1, j, (~C).load(i1,j) - xmm2 * factor );
                     (~C).store( i2, j, (~C).load(i2,j) - xmm3 * factor );
                     (~C).store( i3, j, (~C).load(i3,j) - xmm4 * factor );
                  }
               }

               for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL )
               {
                  const size_t i1( i+SIMDSIZE );

                  size_t j( jj );

                  for( ; (j+4UL) <= jend; j+=4UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+4UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        const SIMDType b3( set( B(k,j+2UL) ) );
                        const SIMDType b4( set( B(k,j+3UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a1 * b2;
                        xmm4 = xmm4 + a2 * b2;
                        xmm5 = xmm5 + a1 * b3;
                        xmm6 = xmm6 + a2 * b3;
                        xmm7 = xmm7 + a1 * b4;
                        xmm8 = xmm8 + a2 * b4;
                     }

                     (~C).store( i , j    , (~C).load(i ,j    ) - xmm1 * factor );
                     (~C).store( i1, j    , (~C).load(i1,j    ) - xmm2 * factor );
                     (~C).store( i , j+1UL, (~C).load(i ,j+1UL) - xmm3 * factor );
                     (~C).store( i1, j+1UL, (~C).load(i1,j+1UL) - xmm4 * factor );
                     (~C).store( i , j+2UL, (~C).load(i ,j+2UL) - xmm5 * factor );
                     (~C).store( i1, j+2UL, (~C).load(i1,j+2UL) - xmm6 * factor );
                     (~C).store( i , j+3UL, (~C).load(i ,j+3UL) - xmm7 * factor );
                     (~C).store( i1, j+3UL, (~C).load(i1,j+3UL) - xmm8 * factor );
                  }

                  for( ; (j+2UL) <= jend; j+=2UL )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+2UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2, xmm3, xmm4;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType a1( A.load(i ,k) );
                        const SIMDType a2( A.load(i1,k) );
                        const SIMDType b1( set( B(k,j    ) ) );
                        const SIMDType b2( set( B(k,j+1UL) ) );
                        xmm1 = xmm1 + a1 * b1;
                        xmm2 = xmm2 + a2 * b1;
                        xmm3 = xmm3 + a1 * b2;
                        xmm4 = xmm4 + a2 * b2;
                     }

                     (~C).store( i , j    , (~C).load(i ,j    ) - xmm1 * factor );
                     (~C).store( i1, j    , (~C).load(i1,j    ) - xmm2 * factor );
                     (~C).store( i , j+1UL, (~C).load(i ,j+1UL) - xmm3 * factor );
                     (~C).store( i1, j+1UL, (~C).load(i1,j+1UL) - xmm4 * factor );
                  }

                  if( j < jend )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE*2UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1, xmm2;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i ,k) * b1;
                        xmm2 = xmm2 + A.load(i1,k) * b1;
                     }

                     (~C).store( i , j, (~C).load(i ,j) - xmm1 * factor );
                     (~C).store( i1, j, (~C).load(i1,j) - xmm2 * factor );
                  }
               }

               for( ; i<ipos; i+=SIMDSIZE )
               {
                  for( size_t j=jj; j<jend; ++j )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+SIMDSIZE, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     SIMDType xmm1;

                     for( size_t k=kbegin; k<kend; ++k ) {
                        const SIMDType b1( set( B(k,j) ) );
                        xmm1 = xmm1 + A.load(i,k) * b1;
                     }

                     (~C).store( i, j, (~C).load(i,j) - xmm1 * factor );
                  }
               }

               for( ; remainder && i<iend; ++i )
               {
                  for( size_t j=jj; j<jend; ++j )
                  {
                     const size_t kbegin( max( ( IsUpper<MT4>::value )?( max( i, kk ) ):( kk ),
                                               ( IsLower<MT5>::value )?( max( j, kk ) ):( kk ) ) );
                     const size_t kend  ( min( ( IsLower<MT4>::value )?( min( i+1UL, ktmp ) ):( ktmp ),
                                               ( IsUpper<MT5>::value )?( j+1UL ):( ktmp ) ) );

                     ElementType value = ElementType();

                     for( size_t k=kbegin; k<kend; ++k ) {
                        value += A(i,k) * B(k,j);
                     }

                     (~C)(i,j) -= value * scalar;
                  }
               }
            }
         }
      }
   }
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense matrices (default)*******************************
   /*!\brief Default subtraction assignment of a scaled transpose dense matrix-dense matrix
   //        multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a large
   // scaled transpose dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseBlasKernel<MT3,MT4,MT5,ST2> >
      selectBlasSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectLargeSubAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**BLAS-based subraction assignment to dense matrices******************************************
#if BLAZE_BLAS_MODE
   /*!\brief BLAS-based subraction assignment of a scaled transpose dense matrix-dense matrix
   //        multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function performs the scaled transpose dense matrix-dense matrix multiplication based
   // on the according BLAS functionality.
   */
   template< typename MT3    // Type of the left-hand side target matrix
           , typename MT4    // Type of the left-hand side matrix operand
           , typename MT5    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseBlasKernel<MT3,MT4,MT5,ST2> >
      selectBlasSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      typedef ElementType_<MT3>  ET;

      if( IsTriangular<MT4>::value ) {
         ResultType_<MT3> tmp( serial( B ) );
         trmm( tmp, A, CblasLeft, ( IsLower<MT4>::value )?( CblasLower ):( CblasUpper ), ET(scalar) );
         subAssign( C, tmp );
      }
      else if( IsTriangular<MT5>::value ) {
         ResultType_<MT3> tmp( serial( A ) );
         trmm( tmp, B, CblasRight, ( IsLower<MT5>::value )?( CblasLower ):( CblasUpper ), ET(scalar) );
         subAssign( C, tmp );
      }
      else {
         gemm( C, A, B, ET(-scalar), ET(1) );
      }
   }
#endif
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
   /*!\brief SMP assignment of a scaled transpose dense matrix-dense matrix multiplication to a
   //        dense matrix (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a scaled transpose
   // dense matrix-dense matrix multiplication expression to a dense matrix. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case either of the two matrix operands requires an intermediate evaluation and no
   // symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline EnableIf_< IsEvaluationRequired<MT,MT1,MT2> >
      smpAssign( DenseMatrix<MT,SO>& lhs, const DMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LeftOperand_<MMM>  left ( rhs.matrix_.leftOperand()  );
      RightOperand_<MMM> right( rhs.matrix_.rightOperand() );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL ) {
         return;
      }
      else if( left.columns() == 0UL ) {
         reset( ~lhs );
         return;
      }

      LT A( left  );  // Evaluation of the left-hand side dense matrix operand
      RT B( right );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == right.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == right.columns() , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns(), "Invalid number of columns" );

      smpAssign( ~lhs, A * B * rhs.scalar_ );
   }
   //**********************************************************************************************

   //**SMP assignment to sparse matrices***********************************************************
   /*!\brief SMP assignment of a scaled transpose dense matrix-dense matrix multiplication to a
   //        sparse matrix.
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side sparse matrix.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a scaled transpose dense
   // matrix-dense matrix multiplication expression to a sparse matrix. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case either of the two matrix operands requires an intermediate evaluation and no
   // symmetry can be exploited.
   */
   template< typename MT  // Type of the target sparse matrix
           , bool SO >    // Storage order of the target sparse matrix
   friend inline EnableIf_< IsEvaluationRequired<MT,MT1,MT2> >
      smpAssign( SparseMatrix<MT,SO>& lhs, const DMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      typedef IfTrue_< SO, ResultType, OppositeType >  TmpType;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( OppositeType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( OppositeType );
      BLAZE_CONSTRAINT_MATRICES_MUST_HAVE_SAME_STORAGE_ORDER( MT, TmpType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<TmpType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const TmpType tmp( rhs );
      smpAssign( ~lhs, tmp );
   }
   //**********************************************************************************************

   //**SMP addition assignment to dense matrices***************************************************
   /*!\brief SMP addition assignment of a scaled transpose dense matrix-dense matrix multiplication
   //        to a dense matrix (\f$ C+=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a scaled
   // transpose dense matrix-dense matrix multiplication expression to a dense matrix. Due to
   // the explicit application of the SFINAE principle, this function can only be selected by
   // the compiler in case either of the two matrix operands requires an intermediate evaluation
   // and no symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline EnableIf_< IsEvaluationRequired<MT,MT1,MT2> >
      smpAddAssign( DenseMatrix<MT,SO>& lhs, const DMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LeftOperand_<MMM>  left ( rhs.matrix_.leftOperand()  );
      RightOperand_<MMM> right( rhs.matrix_.rightOperand() );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || left.columns() == 0UL ) {
         return;
      }

      LT A( left  );  // Evaluation of the left-hand side dense matrix operand
      RT B( right );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == right.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == right.columns() , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns(), "Invalid number of columns" );

      smpAddAssign( ~lhs, A * B * rhs.scalar_ );
   }
   //**********************************************************************************************

   //**SMP addition assignment to sparse matrices**************************************************
   // No special implementation for the SMP addition assignment to sparse matrices.
   //**********************************************************************************************

   //**SMP subtraction assignment to dense matrices************************************************
   /*!\brief SMP subtraction assignment of a scaled transpose dense matrix-dense matrix
   //        multiplication to a dense matrix (\f$ C-=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side dense matrix.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a scaled
   // transpose dense matrix-dense matrix multiplication expression to a dense matrix. Due to
   // the explicit application of the SFINAE principle, this function can only be selected by
   // the compiler in case either of the two matrix operands requires an intermediate evaluation
   // and no symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense matrix
           , bool SO >    // Storage order of the target dense matrix
   friend inline EnableIf_< IsEvaluationRequired<MT,MT1,MT2> >
      smpSubAssign( DenseMatrix<MT,SO>& lhs, const DMatScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LeftOperand_<MMM>  left ( rhs.matrix_.leftOperand()  );
      RightOperand_<MMM> right( rhs.matrix_.rightOperand() );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || left.columns() == 0UL ) {
         return;
      }

      LT A( left  );  // Evaluation of the left-hand side dense matrix operand
      RT B( right );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == right.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == right.columns() , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns(), "Invalid number of columns" );

      smpSubAssign( ~lhs, A * B * rhs.scalar_ );
   }
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
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MMM );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( MMM );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( ST );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ST, RightOperand );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Multiplication operator for the multiplication of a column-major dense matrix and a
//        row-major dense matrix (\f$ A=B*C \f$).
// \ingroup dense_matrix
//
// \param lhs The left-hand side matrix for the multiplication.
// \param rhs The right-hand side matrix for the multiplication.
// \return The resulting matrix.
//
// This operator represents the multiplication of a column-major dense matrix and a row-major
// dense matrix:

   \code
   using blaze::rowMajor;
   using blaze::columnMajor;

   blaze::DynamicMatrix<double,columnMajor> A, C;
   blaze::DynamicMatrix<double,rowMajor> B;
   // ... Resizing and initialization
   C = A * B;
   \endcode

// The operator returns an expression representing a dense matrix of the higher-order element
// type of the two involved matrix element types \a T1::ElementType and \a T2::ElementType.
// Both matrix types \a T1 and \a T2 as well as the two element types \a T1::ElementType and
// \a T2::ElementType have to be supported by the MultTrait class template.\n
// In case the current number of columns of \a lhs and the current number of rows of \a rhs
// don't match, a \a std::invalid_argument is thrown.
*/
template< typename T1    // Type of the left-hand side dense matrix
        , typename T2 >  // Type of the right-hand side dense matrix
inline const TDMatDMatMultExpr<T1,T2>
   operator*( const DenseMatrix<T1,true>& lhs, const DenseMatrix<T2,false>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   if( (~lhs).columns() != (~rhs).rows() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   return TDMatDMatMultExpr<T1,T2>( ~lhs, ~rhs );
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
struct Rows< TDMatDMatMultExpr<MT1,MT2> > : public Rows<MT1>
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
struct Columns< TDMatDMatMultExpr<MT1,MT2> > : public Columns<MT2>
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
struct IsAligned< TDMatDMatMultExpr<MT1,MT2> >
   : public BoolConstant< And< IsAligned<MT1>, IsAligned<MT2> >::value >
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
struct IsLower< TDMatDMatMultExpr<MT1,MT2> >
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
struct IsUniLower< TDMatDMatMultExpr<MT1,MT2> >
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
struct IsStrictlyLower< TDMatDMatMultExpr<MT1,MT2> >
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
struct IsUpper< TDMatDMatMultExpr<MT1,MT2> >
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
struct IsUniUpper< TDMatDMatMultExpr<MT1,MT2> >
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
struct IsStrictlyUpper< TDMatDMatMultExpr<MT1,MT2> >
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
struct TDMatDVecMultExprTrait< TDMatDMatMultExpr<MT1,MT2>, VT >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsDenseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsDenseVector<VT>, IsColumnVector<VT> >
                   , TDMatDVecMultExprTrait_< MT1, DMatDVecMultExprTrait_<MT2,VT> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2, typename VT >
struct TDMatSVecMultExprTrait< TDMatDMatMultExpr<MT1,MT2>, VT >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsDenseMatrix<MT2>, IsRowMajorMatrix<MT2>
                        , IsSparseVector<VT>, IsColumnVector<VT> >
                   , TDMatDVecMultExprTrait_< MT1, DMatSVecMultExprTrait_<MT2,VT> >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename MT1, typename MT2 >
struct TDVecTDMatMultExprTrait< VT, TDMatDMatMultExpr<MT1,MT2> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsDenseVector<VT>, IsRowVector<VT>
                        , IsDenseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsDenseMatrix<MT2>, IsRowMajorMatrix<MT2> >
                   , TDVecDMatMultExprTrait_< TDVecTDMatMultExprTrait_<VT,MT1>, MT2 >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename MT1, typename MT2 >
struct TSVecTDMatMultExprTrait< VT, TDMatDMatMultExpr<MT1,MT2> >
{
 public:
   //**********************************************************************************************
   using Type = If_< And< IsSparseVector<VT>, IsRowVector<VT>
                        , IsDenseMatrix<MT1>, IsColumnMajorMatrix<MT1>
                        , IsDenseMatrix<MT2>, IsRowMajorMatrix<MT2> >
                   , TDVecDMatMultExprTrait_< TSVecTDMatMultExprTrait_<VT,MT1>, MT2 >
                   , INVALID_TYPE >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2, bool AF >
struct SubmatrixExprTrait< TDMatDMatMultExpr<MT1,MT2>, AF >
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
struct RowExprTrait< TDMatDMatMultExpr<MT1,MT2> >
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
struct ColumnExprTrait< TDMatDMatMultExpr<MT1,MT2> >
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
