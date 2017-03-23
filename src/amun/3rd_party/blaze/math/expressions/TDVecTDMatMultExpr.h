//=================================================================================================
/*!
//  \file blaze/math/expressions/TDVecTDMatMultExpr.h
//  \brief Header file for the transpose dense vector/transpose dense matrix multiplication expression
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

#ifndef _BLAZE_MATH_EXPRESSIONS_TDVECTDMATMULTEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_TDVECTDMATMULTEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/blas/gemv.h>
#include <blaze/math/blas/trmv.h>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/ColumnMajorMatrix.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/constraints/RowVector.h>
#include <blaze/math/constraints/TVecMatMultExpr.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/TVecMatMultExpr.h>
#include <blaze/math/expressions/VecScalarMultExpr.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/traits/MultExprTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/SubmatrixExprTrait.h>
#include <blaze/math/traits/SubvectorExprTrait.h>
#include <blaze/math/typetraits/AreSIMDCombinable.h>
#include <blaze/math/typetraits/Columns.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsBLASCompatible.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsDiagonal.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsMatMatMultExpr.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsTriangular.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/system/BLAS.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Complex.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/logging/FunctionTrace.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/mpl/If.h>
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
//  CLASS TDVECTDMATMULTEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for transpose dense vector-transpose dense matrix multiplications.
// \ingroup dense_vector_expression
//
// The TDVecTDMatMultExpr class represents the compile time expression for multiplications
// between transpose dense vectors and column-major dense matrices.
*/
template< typename VT    // Type of the left-hand side dense vector
        , typename MT >  // Type of the right-hand side dense matrix
class TDVecTDMatMultExpr : public DenseVector< TDVecTDMatMultExpr<VT,MT>, true >
                         , private TVecMatMultExpr
                         , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef ResultType_<VT>     VRT;  //!< Result type of the left-hand side dense vector expression.
   typedef ResultType_<MT>     MRT;  //!< Result type of the right-hand side dense matrix expression.
   typedef ElementType_<VRT>   VET;  //!< Element type of the left-hand side dense vector epxression.
   typedef ElementType_<MRT>   MET;  //!< Element type of the right-hand side dense matrix expression.
   typedef CompositeType_<VT>  VCT;  //!< Composite type of the left-hand side dense vector expression.
   typedef CompositeType_<MT>  MCT;  //!< Composite type of the right-hand side dense matrix expression.
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the left-hand side dense vector expression.
   enum : bool { evaluateVector = IsComputation<VT>::value || RequiresEvaluation<VT>::value };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the right-hand side dense matrix expression.
   enum : bool { evaluateMatrix = ( IsComputation<MT>::value && IsSame<MET,VET>::value &&
                                    IsBLASCompatible<MET>::value ) || RequiresEvaluation<MT>::value };
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! The UseSMPAssign struct is a helper struct for the selection of the parallel evaluation
       strategy. In case either the vector or the matrix operand requires an intermediate
       evaluation, the nested \a value will be set to 1, otherwise it will be 0. */
   template< typename T1 >
   struct UseSMPAssign {
      enum : bool { value = ( evaluateVector || evaluateMatrix ) };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! In case the two involved vector types and the matrix type are suited for a BLAS kernel,
       the nested \a value will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   struct UseBlasKernel {
      enum : bool { value = BLAZE_BLAS_MODE &&
                            HasMutableDataAccess<T1>::value &&
                            HasConstDataAccess<T2>::value &&
                            HasConstDataAccess<T3>::value &&
                            !IsDiagonal<T3>::value &&
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
   /*! In case the two involved vector types and the matrix type are suited for a vectorized
       computation of the vector/matrix multiplication, the nested \a value will be set to 1,
       otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   struct UseVectorizedDefaultKernel {
      enum : bool { value = useOptimizedKernels &&
                            !IsDiagonal<T3>::value &&
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
   typedef TDVecTDMatMultExpr<VT,MT>   This;           //!< Type of this TDVecTDMatMultExpr instance.
   typedef MultTrait_<VRT,MRT>         ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>  TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<ResultType>    ElementType;    //!< Resulting element type.
   typedef SIMDTrait_<ElementType>     SIMDType;       //!< Resulting SIMD element type.
   typedef const ElementType           ReturnType;     //!< Return type for expression template evaluations.
   typedef const ResultType            CompositeType;  //!< Data type for composite expression templates.

   //! Composite type of the left-hand side dense vector expression.
   typedef If_< IsExpression<VT>, const VT, const VT& >  LeftOperand;

   //! Composite type of the right-hand side dense matrix expression.
   typedef If_< IsExpression<MT>, const MT, const MT& >  RightOperand;

   //! Type for the assignment of the left-hand side dense matrix operand.
   typedef IfTrue_< evaluateVector, const VRT, VCT >  LT;

   //! Type for the assignment of the right-hand side dense vector operand.
   typedef IfTrue_< evaluateMatrix, const MRT, MCT >  RT;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = !IsDiagonal<MT>::value &&
                               VT::simdEnabled && MT::simdEnabled &&
                               HasSIMDAdd<VET,MET>::value &&
                               HasSIMDMult<VET,MET>::value };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = !evaluateVector && VT::smpAssignable &&
                                 !evaluateMatrix && MT::smpAssignable };
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   enum : size_t { SIMDSIZE = SIMDTrait<ElementType>::size };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the TDVecTDMatMultExpr class.
   //
   // \param vec The left-hand side vector operand of the multiplication expression.
   // \param mat The right-hand side matrix operand of the multiplication expression.
   */
   explicit inline TDVecTDMatMultExpr( const VT& vec, const MT& mat ) noexcept
      : vec_( vec )  // Left-hand side dense vector of the multiplication expression
      , mat_( mat )  // Right-hand side dense matrix of the multiplication expression
   {
      BLAZE_INTERNAL_ASSERT( vec_.size() == mat_.rows(), "Invalid vector and matrix sizes" );
   }
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the vector elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator[]( size_t index ) const {
      BLAZE_INTERNAL_ASSERT( index < mat_.columns(), "Invalid vector access index" );

      if( IsDiagonal<MT>::value )
      {
         return vec_[index] * mat_(index,index);
      }
      else if( IsLower<MT>::value && ( index > 8UL ) )
      {
         const size_t begin( IsStrictlyLower<MT>::value ? index+1UL : index );
         const size_t n    ( mat_.rows() - begin );
         return subvector( vec_, begin, n ) * subvector( column( mat_, index ), begin, n );
      }
      else if( IsUpper<MT>::value && ( index + 8UL < mat_.rows() ) )
      {
         const size_t n( IsStrictlyUpper<MT>::value ? index : index+1UL );
         return subvector( vec_, 0UL, n ) * subvector( column( mat_, index ), 0UL, n );
      }
      else
      {
         return vec_ * column( mat_, index );
      }
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
      if( index >= mat_.columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid vector access index" );
      }
      return (*this)[index];
   }
   //**********************************************************************************************

   //**Size function*******************************************************************************
   /*!\brief Returns the current size/dimension of the vector.
   //
   // \return The size of the vector.
   */
   inline size_t size() const noexcept {
      return mat_.columns();
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side dense vector operand.
   //
   // \return The left-hand side dense vector operand.
   */
   inline LeftOperand leftOperand() const noexcept {
      return vec_;
   }
   //**********************************************************************************************

   //**Right operand access************************************************************************
   /*!\brief Returns the right-hand side transpose dense matrix operand.
   //
   // \return The right-hand side transpose dense matrix operand.
   */
   inline RightOperand rightOperand() const noexcept {
      return mat_;
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
      return ( vec_.isAliased( alias ) || mat_.isAliased( alias ) );
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
      return ( vec_.isAliased( alias ) || mat_.isAliased( alias ) );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const noexcept {
      return vec_.isAligned() && mat_.isAligned();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return ( !BLAZE_BLAS_IS_PARALLEL ||
               ( IsComputation<MT>::value && !evaluateMatrix ) ||
               ( mat_.rows() * mat_.columns() < TDVECTDMATMULT_THRESHOLD ) ) &&
             ( size() > SMP_TDVECTDMATMULT_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  vec_;  //!< Left-hand side dense vector of the multiplication expression.
   RightOperand mat_;  //!< Right-hand side dense matrix of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense vectors*****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a transpose dense vector-transpose dense matrix multiplication to a
   //        transpose dense vector (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a transpose dense vector-
   // transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void assign( DenseVector<VT1,true>& lhs, const TDVecTDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      if( rhs.mat_.rows() == 0UL ) {
         reset( ~lhs );
         return;
      }
      else if( rhs.mat_.columns() == 0UL ) {
         return;
      }

      LT x( serial( rhs.vec_ ) );  // Evaluation of the left-hand side dense vector operand
      RT A( serial( rhs.mat_ ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( x.size()    == rhs.vec_.size()   , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.mat_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.mat_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.columns() == (~lhs).size()     , "Invalid vector size"       );

      TDVecTDMatMultExpr::selectAssignKernel( ~lhs, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to dense vectors (kernel selection)**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for an assignment of a transpose dense vector-transpose
   //        dense matrix multiplication to a dense vector (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline void selectAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      if( ( IsDiagonal<MT1>::value ) ||
          ( IsComputation<MT>::value && !evaluateMatrix ) ||
          ( A.rows() * A.columns() < TDVECTDMATMULT_THRESHOLD ) )
         selectSmallAssignKernel( y, x, A );
      else
         selectBlasAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense vectors*********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a transpose dense vector-transpose dense matrix multiplication
   //        (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function implements the default assignment kernel for the transpose dense vector-
   // transpose dense matrix multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline void selectDefaultAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      y.assign( x * A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense vectors (small matrices)****************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a small transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a transpose dense
   // vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectSmallAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      selectDefaultAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default assignment to dense vectors (small matrices)*****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default assignment of a small transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function implements the vectorized default assignment kernel for the transpose dense
   // vector-transpose dense matrix multiplication. This kernel is optimized for small matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectSmallAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT2>::value || !IsPadded<MT1>::value );

      size_t j( 0UL );

      for( ; (j+8UL) <= N; j+=8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+7UL : j+8UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+3UL);
            xmm5 = xmm5 + x1 * A.load(i,j+4UL);
            xmm6 = xmm6 + x1 * A.load(i,j+5UL);
            xmm7 = xmm7 + x1 * A.load(i,j+6UL);
            xmm8 = xmm8 + x1 * A.load(i,j+7UL);
         }

         y[j    ] = sum( xmm1 );
         y[j+1UL] = sum( xmm2 );
         y[j+2UL] = sum( xmm3 );
         y[j+3UL] = sum( xmm4 );
         y[j+4UL] = sum( xmm5 );
         y[j+5UL] = sum( xmm6 );
         y[j+6UL] = sum( xmm7 );
         y[j+7UL] = sum( xmm8 );

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
            y[j+2UL] += x[i] * A(i,j+2UL);
            y[j+3UL] += x[i] * A(i,j+3UL);
            y[j+4UL] += x[i] * A(i,j+4UL);
            y[j+5UL] += x[i] * A(i,j+5UL);
            y[j+6UL] += x[i] * A(i,j+6UL);
            y[j+7UL] += x[i] * A(i,j+7UL);
         }
      }

      for( ; (j+4UL) <= N; j+=4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+3UL : j+4UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+3UL);
         }

         y[j    ] = sum( xmm1 );
         y[j+1UL] = sum( xmm2 );
         y[j+2UL] = sum( xmm3 );
         y[j+3UL] = sum( xmm4 );

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
            y[j+2UL] += x[i] * A(i,j+2UL);
            y[j+3UL] += x[i] * A(i,j+3UL);
         }
      }

      for( ; (j+3UL) <= N; j+=3UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+2UL : j+3UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
         }

         y[j    ] = sum( xmm1 );
         y[j+1UL] = sum( xmm2 );
         y[j+2UL] = sum( xmm3 );

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
            y[j+2UL] += x[i] * A(i,j+2UL);
         }
      }

      for( ; (j+2UL) <= N; j+=2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+1UL : j+2UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
         }

         y[j    ] = sum( xmm1 );
         y[j+1UL] = sum( xmm2 );

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
         }
      }

      if( j < N )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            xmm1 = xmm1 + x.load(i) * A.load(i,j);
         }

         y[j] = sum( xmm1 );

         for( ; remainder && i<iend; ++i ) {
            y[j] += x[i] * A(i,j);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense vectors (large matrices)****************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a large transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a transpose dense
   // vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectLargeAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      selectDefaultAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default assignment to dense vectors (large matrices)*****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default assignment of a large transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function implements the vectorized default assignment kernel for the transpose dense
   // vector-transpose dense matrix multiplication. This kernel is optimized for large matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectLargeAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT2>::value || !IsPadded<MT1>::value );

      reset( y );

      size_t j( 0UL );

      for( ; (j+8UL) <= N; j+=8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+7UL : j+8UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) + x3 * A.load(i2,j+2UL) + x4 * A.load(i3,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) + x3 * A.load(i2,j+3UL) + x4 * A.load(i3,j+3UL) );
            y[j+4UL] += sum( x1 * A.load(i,j+4UL) + x2 * A.load(i1,j+4UL) + x3 * A.load(i2,j+4UL) + x4 * A.load(i3,j+4UL) );
            y[j+5UL] += sum( x1 * A.load(i,j+5UL) + x2 * A.load(i1,j+5UL) + x3 * A.load(i2,j+5UL) + x4 * A.load(i3,j+5UL) );
            y[j+6UL] += sum( x1 * A.load(i,j+6UL) + x2 * A.load(i1,j+6UL) + x3 * A.load(i2,j+6UL) + x4 * A.load(i3,j+6UL) );
            y[j+7UL] += sum( x1 * A.load(i,j+7UL) + x2 * A.load(i1,j+7UL) + x3 * A.load(i2,j+7UL) + x4 * A.load(i3,j+7UL) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) );
            y[j+4UL] += sum( x1 * A.load(i,j+4UL) + x2 * A.load(i1,j+4UL) );
            y[j+5UL] += sum( x1 * A.load(i,j+5UL) + x2 * A.load(i1,j+5UL) );
            y[j+6UL] += sum( x1 * A.load(i,j+6UL) + x2 * A.load(i1,j+6UL) );
            y[j+7UL] += sum( x1 * A.load(i,j+7UL) + x2 * A.load(i1,j+7UL) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] += sum( x1 * A.load(i,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) );
            y[j+4UL] += sum( x1 * A.load(i,j+4UL) );
            y[j+5UL] += sum( x1 * A.load(i,j+5UL) );
            y[j+6UL] += sum( x1 * A.load(i,j+6UL) );
            y[j+7UL] += sum( x1 * A.load(i,j+7UL) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
            y[j+2UL] += x[i] * A(i,j+2UL);
            y[j+3UL] += x[i] * A(i,j+3UL);
            y[j+4UL] += x[i] * A(i,j+4UL);
            y[j+5UL] += x[i] * A(i,j+5UL);
            y[j+6UL] += x[i] * A(i,j+6UL);
            y[j+7UL] += x[i] * A(i,j+7UL);
         }
      }

      for( ; (j+4UL) <= N; j+=4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+3UL : j+4UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) + x3 * A.load(i2,j+2UL) + x4 * A.load(i3,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) + x3 * A.load(i2,j+3UL) + x4 * A.load(i3,j+3UL) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] += sum( x1 * A.load(i,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
            y[j+2UL] += x[i] * A(i,j+2UL);
            y[j+3UL] += x[i] * A(i,j+3UL);
         }
      }

      for( ; (j+2UL) <= N; j+=2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+1UL : j+2UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] += sum( x1 * A.load(i,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
         }
      }

      if( j < N )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j] += sum( x1 * A.load(i,j) + x2 * A.load(i1,j) + x3 * A.load(i2,j) + x4 * A.load(i3,j) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j] += sum( x1 * A.load(i,j) + x2 * A.load(i1,j) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j] += sum( x1 * A.load(i,j) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j] += x[i] * A(i,j);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based assignment to dense vectors (default)********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a transpose dense vector-transpose dense matrix multiplication
   //        (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a large transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseBlasKernel<VT1,VT2,MT1> >
      selectBlasAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      selectLargeAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based assignment to dense vectors******************************************************
#if BLAZE_BLAS_MODE
   /*! \cond BLAZE_INTERNAL */
   /*!\brief BLAS-based assignment of a transpose dense vector-transpose dense matrix multiplication
   //        (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function performs the transpose dense vector-transpose dense matrix multiplication
   // based on the according BLAS functionality.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseBlasKernel<VT1,VT2,MT1> >
      selectBlasAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      typedef ElementType_<VT1>  ET;

      if( IsTriangular<MT1>::value ) {
         assign( y, x );
         trmv( y, A, ( IsLower<MT1>::value )?( CblasLower ):( CblasUpper ) );
      }
      else {
         gemv( y, x, A, ET(1), ET(0) );
      }
   }
   /*! \endcond */
#endif
   //**********************************************************************************************

   //**Assignment to sparse vectors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a transpose dense vector-transpose dense matrix multiplication to a
   //        transpose sparse vector (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a transpose dense vector-
   // transpose dense matrix multiplication expression to a sparse vector.
   */
   template< typename VT1 >  // Type of the target sparse vector
   friend inline void assign( SparseVector<VT1,true>& lhs, const TDVecTDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( serial( rhs ) );
      assign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense vectors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a transpose dense vector-transpose dense matrix multiplication
   //        to a transpose dense vector (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a transpose dense
   // vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void addAssign( DenseVector<VT1,true>& lhs, const TDVecTDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      if( rhs.mat_.rows() == 0UL || rhs.mat_.columns() == 0UL ) {
         return;
      }

      LT x( serial( rhs.vec_ ) );  // Evaluation of the left-hand side dense vector operand
      RT A( serial( rhs.mat_ ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( x.size()    == rhs.vec_.size()   , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.mat_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.mat_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.columns() == (~lhs).size()     , "Invalid vector size"       );

      TDVecTDMatMultExpr::selectAddAssignKernel( ~lhs, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense vectors (kernel selection)*************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for an addition assignment of a transpose dense vector-
   //        transpose dense matrix multiplication to a dense vector (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline void selectAddAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      if( ( IsDiagonal<MT1>::value ) ||
          ( IsComputation<MT>::value && !evaluateMatrix ) ||
          ( A.rows() * A.columns() < TDVECTDMATMULT_THRESHOLD ) )
         selectSmallAddAssignKernel( y, x, A );
      else
         selectBlasAddAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense vectors************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function implements the default addition assignment kernel for the transpose dense
   // vector-transpose dense matrix multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline void selectDefaultAddAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      y.addAssign( x * A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense vectors (small matrices)*******************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a small transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectSmallAddAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      selectDefaultAddAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default addition assignment to dense vectors (small matrices)********************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default addition assignment of a small transpose dense vector-transpose
   //        dense matrix multiplication (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function implements the vectorized default addition assignment kernel for the
   // transpose dense vector-transpose dense matrix multiplication. This kernel is optimized
   // for small matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectSmallAddAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT2>::value || !IsPadded<MT1>::value );

      size_t j( 0UL );

      for( ; (j+8UL) <= N; j+=8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+7UL : j+8UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+3UL);
            xmm5 = xmm5 + x1 * A.load(i,j+4UL);
            xmm6 = xmm6 + x1 * A.load(i,j+5UL);
            xmm7 = xmm7 + x1 * A.load(i,j+6UL);
            xmm8 = xmm8 + x1 * A.load(i,j+7UL);
         }

         y[j    ] += sum( xmm1 );
         y[j+1UL] += sum( xmm2 );
         y[j+2UL] += sum( xmm3 );
         y[j+3UL] += sum( xmm4 );
         y[j+4UL] += sum( xmm5 );
         y[j+5UL] += sum( xmm6 );
         y[j+6UL] += sum( xmm7 );
         y[j+7UL] += sum( xmm8 );

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
            y[j+2UL] += x[i] * A(i,j+2UL);
            y[j+3UL] += x[i] * A(i,j+3UL);
            y[j+4UL] += x[i] * A(i,j+4UL);
            y[j+5UL] += x[i] * A(i,j+5UL);
            y[j+6UL] += x[i] * A(i,j+6UL);
            y[j+7UL] += x[i] * A(i,j+7UL);
         }
      }

      for( ; (j+4UL) <= N; j+=4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+3UL : j+4UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+3UL);
         }

         y[j    ] += sum( xmm1 );
         y[j+1UL] += sum( xmm2 );
         y[j+2UL] += sum( xmm3 );
         y[j+3UL] += sum( xmm4 );

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
            y[j+2UL] += x[i] * A(i,j+2UL);
            y[j+3UL] += x[i] * A(i,j+3UL);
         }
      }

      for( ; (j+3UL) <= N; j+=3UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+2UL : j+3UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
         }

         y[j    ] += sum( xmm1 );
         y[j+1UL] += sum( xmm2 );
         y[j+2UL] += sum( xmm3 );

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
            y[j+2UL] += x[i] * A(i,j+2UL);
         }
      }

      for( ; (j+2UL) <= N; j+=2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+1UL : j+2UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
         }

         y[j    ] += sum( xmm1 );
         y[j+1UL] += sum( xmm2 );

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
         }
      }

      if( j < N )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            xmm1 = xmm1 + A.load(i,j) * x.load(i);
         }

         y[j] += sum( xmm1 );

         for( ; remainder && i<iend; ++i ) {
            y[j] += x[i] * A(i,j);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense vectors (large matrices)*******************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a large transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectLargeAddAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      selectDefaultAddAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default addition assignment to dense vectors (large matrices)********************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default addition assignment of a large transpose dense vector-transpose
   //        dense matrix multiplication (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function implements the vectorized default addition assignment kernel for the
   // transpose dense vector-transpose dense matrix multiplication. This kernel is optimized
   // for large matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectLargeAddAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT2>::value || !IsPadded<MT1>::value );

      size_t j( 0UL );

      for( ; (j+8UL) <= N; j+=8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+7UL : j+8UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) + x3 * A.load(i2,j+2UL) + x4 * A.load(i3,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) + x3 * A.load(i2,j+3UL) + x4 * A.load(i3,j+3UL) );
            y[j+4UL] += sum( x1 * A.load(i,j+4UL) + x2 * A.load(i1,j+4UL) + x3 * A.load(i2,j+4UL) + x4 * A.load(i3,j+4UL) );
            y[j+5UL] += sum( x1 * A.load(i,j+5UL) + x2 * A.load(i1,j+5UL) + x3 * A.load(i2,j+5UL) + x4 * A.load(i3,j+5UL) );
            y[j+6UL] += sum( x1 * A.load(i,j+6UL) + x2 * A.load(i1,j+6UL) + x3 * A.load(i2,j+6UL) + x4 * A.load(i3,j+6UL) );
            y[j+7UL] += sum( x1 * A.load(i,j+7UL) + x2 * A.load(i1,j+7UL) + x3 * A.load(i2,j+7UL) + x4 * A.load(i3,j+7UL) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) );
            y[j+4UL] += sum( x1 * A.load(i,j+4UL) + x2 * A.load(i1,j+4UL) );
            y[j+5UL] += sum( x1 * A.load(i,j+5UL) + x2 * A.load(i1,j+5UL) );
            y[j+6UL] += sum( x1 * A.load(i,j+6UL) + x2 * A.load(i1,j+6UL) );
            y[j+7UL] += sum( x1 * A.load(i,j+7UL) + x2 * A.load(i1,j+7UL) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] += sum( x1 * A.load(i,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) );
            y[j+4UL] += sum( x1 * A.load(i,j+4UL) );
            y[j+5UL] += sum( x1 * A.load(i,j+5UL) );
            y[j+6UL] += sum( x1 * A.load(i,j+6UL) );
            y[j+7UL] += sum( x1 * A.load(i,j+7UL) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
            y[j+2UL] += x[i] * A(i,j+2UL);
            y[j+3UL] += x[i] * A(i,j+3UL);
            y[j+4UL] += x[i] * A(i,j+4UL);
            y[j+5UL] += x[i] * A(i,j+5UL);
            y[j+6UL] += x[i] * A(i,j+6UL);
            y[j+7UL] += x[i] * A(i,j+7UL);
         }
      }

      for( ; (j+4UL) <= N; j+=4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+3UL : j+4UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) + x3 * A.load(i2,j+2UL) + x4 * A.load(i3,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) + x3 * A.load(i2,j+3UL) + x4 * A.load(i3,j+3UL) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] += sum( x1 * A.load(i,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
            y[j+2UL] += x[i] * A(i,j+2UL);
            y[j+3UL] += x[i] * A(i,j+3UL);
         }
      }

      for( ; (j+2UL) <= N; j+=2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+1UL : j+2UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] += sum( x1 * A.load(i,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
         }
      }

      if( j < N )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j] += sum( x1 * A.load(i,j) + x2 * A.load(i1,j) + x3 * A.load(i2,j) + x4 * A.load(i3,j) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j] += sum( x1 * A.load(i,j) + x2 * A.load(i1,j) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j] += sum( x1 * A.load(i,j) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j] += x[i] * A(i,j);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense vectors (default)***********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a large
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseBlasKernel<VT1,VT2,MT1> >
      selectBlasAddAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      selectLargeAddAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense vectors*********************************************
#if BLAZE_BLAS_MODE
   /*! \cond BLAZE_INTERNAL */
   /*!\brief BLAS-based addition assignment of a vector-matrix multiplication
   //        (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function performs the transpose dense vector-transpose dense matrix multiplication
   // based on the according BLAS functionality.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseBlasKernel<VT1,VT2,MT1> >
      selectBlasAddAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      typedef ElementType_<VT1>  ET;

      if( IsTriangular<MT1>::value ) {
         ResultType_<VT1> tmp( serial( x ) );
         trmv( tmp, A, ( IsLower<MT1>::value )?( CblasLower ):( CblasUpper ) );
         addAssign( y, tmp );
      }
      else {
         gemv( y, x, A, ET(1), ET(1) );
      }
   }
   /*! \endcond */
#endif
   //**********************************************************************************************

   //**Addition assignment to sparse vectors*******************************************************
   // No special implementation for the addition assignment to sparse vectors.
   //**********************************************************************************************

   //**Subtraction assignment to dense vectors*****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void subAssign( DenseVector<VT1,true>& lhs, const TDVecTDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      if( rhs.mat_.rows() == 0UL || rhs.mat_.columns() == 0UL ) {
         return;
      }

      LT x( serial( rhs.vec_ ) );  // Evaluation of the left-hand side dense vector operand
      RT A( serial( rhs.mat_ ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( x.size()    == rhs.vec_.size()   , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.mat_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.mat_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.columns() == (~lhs).size()     , "Invalid vector size"       );

      TDVecTDMatMultExpr::selectSubAssignKernel( ~lhs, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to dense vectors (kernel selection)**********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for a subtraction assignment of a transpose dense vector-
   //        transpose dense matrix multiplication to a dense vector (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline void selectSubAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      if( ( IsDiagonal<MT1>::value ) ||
          ( IsComputation<MT>::value && !evaluateMatrix ) ||
          ( A.rows() * A.columns() < TDVECTDMATMULT_THRESHOLD ) )
         selectSmallSubAssignKernel( y, x, A );
      else
         selectBlasSubAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors*********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function implements the default subtraction assignment kernel for the transpose dense
   // vector-transpose dense matrix multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline void selectDefaultSubAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      y.subAssign( x * A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors (small matrices)****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a small transpose dense vector-transpose dense
   //        matrix multiplication (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectSmallSubAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      selectDefaultSubAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to dense vectors (small matrices)*****************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default subtraction assignment of a small transpose dense vector-transpose
   //        dense matrix multiplication (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment kernel for the
   // transpose dense vector-transpose dense matrix multiplication. This kernel is optimized
   // for small matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectSmallSubAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT2>::value || !IsPadded<MT1>::value );

      size_t j( 0UL );

      for( ; (j+8UL) <= N; j+=8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+7UL : j+8UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+3UL);
            xmm5 = xmm5 + x1 * A.load(i,j+4UL);
            xmm6 = xmm6 + x1 * A.load(i,j+5UL);
            xmm7 = xmm7 + x1 * A.load(i,j+6UL);
            xmm8 = xmm8 + x1 * A.load(i,j+7UL);
         }

         y[j    ] -= sum( xmm1 );
         y[j+1UL] -= sum( xmm2 );
         y[j+2UL] -= sum( xmm3 );
         y[j+3UL] -= sum( xmm4 );
         y[j+4UL] -= sum( xmm5 );
         y[j+5UL] -= sum( xmm6 );
         y[j+6UL] -= sum( xmm7 );
         y[j+7UL] -= sum( xmm8 );

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    );
            y[j+1UL] -= x[i] * A(i,j+1UL);
            y[j+2UL] -= x[i] * A(i,j+2UL);
            y[j+3UL] -= x[i] * A(i,j+3UL);
            y[j+4UL] -= x[i] * A(i,j+4UL);
            y[j+5UL] -= x[i] * A(i,j+5UL);
            y[j+6UL] -= x[i] * A(i,j+6UL);
            y[j+7UL] -= x[i] * A(i,j+7UL);
         }
      }

      for( ; (j+4UL) <= N; j+=4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+3UL : j+4UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+3UL);
         }

         y[j    ] -= sum( xmm1 );
         y[j+1UL] -= sum( xmm2 );
         y[j+2UL] -= sum( xmm3 );
         y[j+3UL] -= sum( xmm4 );

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    );
            y[j+1UL] -= x[i] * A(i,j+1UL);
            y[j+2UL] -= x[i] * A(i,j+2UL);
            y[j+3UL] -= x[i] * A(i,j+3UL);
         }
      }

      for( ; (j+3UL) <= N; j+=3UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+2UL : j+3UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
         }

         y[j    ] -= sum( xmm1 );
         y[j+1UL] -= sum( xmm2 );
         y[j+2UL] -= sum( xmm3 );

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    );
            y[j+1UL] -= x[i] * A(i,j+1UL);
            y[j+2UL] -= x[i] * A(i,j+2UL);
         }
      }

      for( ; (j+2UL) <= N; j+=2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+1UL : j+2UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
         }

         y[j    ] -= sum( xmm1 );
         y[j+1UL] -= sum( xmm2 );

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    );
            y[j+1UL] -= x[i] * A(i,j+1UL);
         }
      }

      if( j < N )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            xmm1 = xmm1 + A.load(i,j) * x.load(i);
         }

         y[j] -= sum( xmm1 );

         for( ; remainder && i<iend; ++i ) {
            y[j] -= x[i] * A(i,j);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors (large matrices)****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a large transpose dense vector-transpose dense
   //        matrix multiplication (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectLargeSubAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      selectDefaultSubAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to dense vectors (large matrices)*****************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default subtraction assignment of a large transpose dense vector-transpose
   //        dense matrix multiplication (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment kernel for the
   // transpose dense vector-transpose dense matrix multiplication. This kernel is optimized
   // for large matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectLargeSubAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT2>::value || !IsPadded<MT1>::value );

      size_t j( 0UL );

      for( ; (j+8UL) <= N; j+=8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+7UL : j+8UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] -= sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) );
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) );
            y[j+2UL] -= sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) + x3 * A.load(i2,j+2UL) + x4 * A.load(i3,j+2UL) );
            y[j+3UL] -= sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) + x3 * A.load(i2,j+3UL) + x4 * A.load(i3,j+3UL) );
            y[j+4UL] -= sum( x1 * A.load(i,j+4UL) + x2 * A.load(i1,j+4UL) + x3 * A.load(i2,j+4UL) + x4 * A.load(i3,j+4UL) );
            y[j+5UL] -= sum( x1 * A.load(i,j+5UL) + x2 * A.load(i1,j+5UL) + x3 * A.load(i2,j+5UL) + x4 * A.load(i3,j+5UL) );
            y[j+6UL] -= sum( x1 * A.load(i,j+6UL) + x2 * A.load(i1,j+6UL) + x3 * A.load(i2,j+6UL) + x4 * A.load(i3,j+6UL) );
            y[j+7UL] -= sum( x1 * A.load(i,j+7UL) + x2 * A.load(i1,j+7UL) + x3 * A.load(i2,j+7UL) + x4 * A.load(i3,j+7UL) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] -= sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) );
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) );
            y[j+2UL] -= sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) );
            y[j+3UL] -= sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) );
            y[j+4UL] -= sum( x1 * A.load(i,j+4UL) + x2 * A.load(i1,j+4UL) );
            y[j+5UL] -= sum( x1 * A.load(i,j+5UL) + x2 * A.load(i1,j+5UL) );
            y[j+6UL] -= sum( x1 * A.load(i,j+6UL) + x2 * A.load(i1,j+6UL) );
            y[j+7UL] -= sum( x1 * A.load(i,j+7UL) + x2 * A.load(i1,j+7UL) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] -= sum( x1 * A.load(i,j    ) );
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) );
            y[j+2UL] -= sum( x1 * A.load(i,j+2UL) );
            y[j+3UL] -= sum( x1 * A.load(i,j+3UL) );
            y[j+4UL] -= sum( x1 * A.load(i,j+4UL) );
            y[j+5UL] -= sum( x1 * A.load(i,j+5UL) );
            y[j+6UL] -= sum( x1 * A.load(i,j+6UL) );
            y[j+7UL] -= sum( x1 * A.load(i,j+7UL) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    );
            y[j+1UL] -= x[i] * A(i,j+1UL);
            y[j+2UL] -= x[i] * A(i,j+2UL);
            y[j+3UL] -= x[i] * A(i,j+3UL);
            y[j+4UL] -= x[i] * A(i,j+4UL);
            y[j+5UL] -= x[i] * A(i,j+5UL);
            y[j+6UL] -= x[i] * A(i,j+6UL);
            y[j+7UL] -= x[i] * A(i,j+7UL);
         }
      }

      for( ; (j+4UL) <= N; j+=4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+3UL : j+4UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] -= sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) );
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) );
            y[j+2UL] -= sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) + x3 * A.load(i2,j+2UL) + x4 * A.load(i3,j+2UL) );
            y[j+3UL] -= sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) + x3 * A.load(i2,j+3UL) + x4 * A.load(i3,j+3UL) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] -= sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) );
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) );
            y[j+2UL] -= sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) );
            y[j+3UL] -= sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] -= sum( x1 * A.load(i,j    ) );
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) );
            y[j+2UL] -= sum( x1 * A.load(i,j+2UL) );
            y[j+3UL] -= sum( x1 * A.load(i,j+3UL) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    );
            y[j+1UL] -= x[i] * A(i,j+1UL);
            y[j+2UL] -= x[i] * A(i,j+2UL);
            y[j+3UL] -= x[i] * A(i,j+3UL);
         }
      }

      for( ; (j+2UL) <= N; j+=2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+1UL : j+2UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] -= sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) );
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] -= sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) );
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] -= sum( x1 * A.load(i,j    ) );
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    );
            y[j+1UL] -= x[i] * A(i,j+1UL);
         }
      }

      if( j < N )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j] -= sum( x1 * A.load(i,j) + x2 * A.load(i1,j) + x3 * A.load(i2,j) + x4 * A.load(i3,j) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j] -= sum( x1 * A.load(i,j) + x2 * A.load(i1,j) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j] -= sum( x1 * A.load(i,j) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j] -= x[i] * A(i,j);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense vectors (default)********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a large
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline DisableIf_< UseBlasKernel<VT1,VT2,MT1> >
      selectBlasSubAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      selectLargeSubAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense vectors******************************************
#if BLAZE_BLAS_MODE
   /*! \cond BLAZE_INTERNAL */
   /*!\brief BLAS-based subtraction assignment of a vector-matrix multiplication
   //        (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side transpose dense vector operand.
   // \param A The right-hand side column-major dense matrix operand.
   // \return void
   //
   // This function performs the transpose dense vector-transpose dense matrix multiplication
   // based on the according BLAS functionality.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseBlasKernel<VT1,VT2,MT1> >
      selectBlasSubAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      typedef ElementType_<VT1>  ET;

      if( IsTriangular<MT1>::value ) {
         ResultType_<VT1> tmp( serial( x ) );
         trmv( tmp, A, ( IsLower<MT1>::value )?( CblasLower ):( CblasUpper ) );
         subAssign( y, tmp );
      }
      else {
         gemv( y, x, A, ET(-1), ET(1) );
      }
   }
   /*! \endcond */
#endif
   //**********************************************************************************************

   //**Subtraction assignment to sparse vectors****************************************************
   // No special implementation for the subtraction assignment to sparse vectors.
   //**********************************************************************************************

   //**Multiplication assignment to dense vectors**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Multiplication assignment of a transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T*=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void multAssign( DenseVector<VT1,true>& lhs, const TDVecTDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( ResultType );
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
   /*!\brief Division assignment of a transpose dense vector-transpose dense matrix multiplication
   //        to a transpose dense vector (\f$ \vec{y}^T/=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression divisor.
   // \return void
   //
   // This function implements the performance optimized division assignment of a transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void divAssign( DenseVector<VT1,true>& lhs, const TDVecTDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( ResultType );
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
   /*!\brief SMP assignment of a transpose dense vector-transpose dense matrix multiplication to
   //        a transpose dense vector (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a transpose dense
   // vector-transpose dense matrix multiplication expression to a dense vector. Due to the
   // explicit application of the SFINAE principle, this function can only be selected by the
   // compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAssign( DenseVector<VT1,true>& lhs, const TDVecTDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      if( rhs.mat_.rows() == 0UL ) {
         reset( ~lhs );
         return;
      }
      else if( rhs.mat_.columns() == 0UL ) {
         return;
      }

      LT x( rhs.vec_ );  // Evaluation of the left-hand side dense vector operand
      RT A( rhs.mat_ );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( x.size()    == rhs.vec_.size()   , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.mat_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.mat_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.columns() == (~lhs).size()     , "Invalid vector size"       );

      smpAssign( ~lhs, x * A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to sparse vectors************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a transpose dense vector-transpose dense matrix multiplication to
   //        a transpose sparse vector (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a transpose dense
   // vector-transpose dense matrix multiplication expression to a sparse vector. Due to the
   // explicit application of the SFINAE principle, this function can only be selected by the
   // compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target sparse vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAssign( SparseVector<VT1,true>& lhs, const TDVecTDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( rhs );
      smpAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to dense vectors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector. Due to
   // the explicit application of the SFINAE principle, this function can only be selected by
   // the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAddAssign( DenseVector<VT1,true>& lhs, const TDVecTDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      if( rhs.mat_.rows() == 0UL || rhs.mat_.columns() == 0UL ) {
         return;
      }

      LT x( rhs.vec_ );  // Evaluation of the left-hand side dense vector operand
      RT A( rhs.mat_ );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( x.size()    == rhs.vec_.size()   , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.mat_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.mat_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.columns() == (~lhs).size()     , "Invalid vector size"       );

      smpAddAssign( ~lhs, x * A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to sparse vectors***************************************************
   // No special implementation for the SMP addition assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP subtraction assignment to dense vectors*************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector. Due to the
   // explicit application of the SFINAE principle, this function can only be selected by the
   // compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpSubAssign( DenseVector<VT1,true>& lhs, const TDVecTDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      if( rhs.mat_.rows() == 0UL || rhs.mat_.columns() == 0UL ) {
         return;
      }

      LT x( rhs.vec_ );  // Evaluation of the left-hand side dense vector operand
      RT A( rhs.mat_ );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( x.size()    == rhs.vec_.size()   , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.mat_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.mat_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.columns() == (~lhs).size()     , "Invalid vector size"       );

      smpSubAssign( ~lhs, x * A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to sparse vectors************************************************
   // No special implementation for the SMP subtraction assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP multiplication assignment to dense vectors**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP multiplication assignment of a transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T*=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   // Due to the explicit application of the SFINAE principle, this function can only be selected
   // by the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpMultAssign( DenseVector<VT1,true>& lhs, const TDVecTDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( ResultType );
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
   /*!\brief SMP division assignment of a transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T/=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression divisor.
   // \return void
   //
   // This function implements the performance optimized SMP division assignment of a transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector. Due to the
   // explicit application of the SFINAE principle, this function can only be selected by the
   // compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpDivAssign( DenseVector<VT1,true>& lhs, const TDVecTDMatMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( ResultType );
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
   BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( VT );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_FORM_VALID_TVECMATMULTEXPR( VT, MT );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  DVECSCALARMULTEXPR SPECIALIZATION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Expression object for scaled transpose dense vector-transpose dense matrix multiplications.
// \ingroup dense_vector_expression
//
// This specialization of the DVecScalarMultExpr class represents the compile time expression
// for scaled multiplications between a non-transpose dense vector and a column-major dense matrix.
*/
template< typename VT    // Type of the left-hand side dense vector
        , typename MT    // Type of the right-hand side dense matrix
        , typename ST >  // Type of the side scalar value
class DVecScalarMultExpr< TDVecTDMatMultExpr<VT,MT>, ST, true >
   : public DenseVector< DVecScalarMultExpr< TDVecTDMatMultExpr<VT,MT>, ST, true >, true >
   , private VecScalarMultExpr
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef TDVecTDMatMultExpr<VT,MT>  VMM;  //!< Type of the dense vector-dense matrix multiplication expression.
   typedef ResultType_<VMM>           RES;  //!< Result type of the dense vector-dense matrix multiplication expression.
   typedef ResultType_<VT>            VRT;  //!< Result type of the left-hand side dense vector expression.
   typedef ResultType_<MT>            MRT;  //!< Result type of the right-hand side dense matrix expression.
   typedef ElementType_<VRT>          VET;  //!< Element type of the left-hand side dense vector epxression.
   typedef ElementType_<MRT>          MET;  //!< Element type of the right-hand side dense matrix expression.
   typedef CompositeType_<VT>         VCT;  //!< Composite type of the left-hand side dense vector expression.
   typedef CompositeType_<MT>         MCT;  //!< Composite type of the right-hand side dense matrix expression.
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the left-hand side dense vector expression.
   enum : bool { evaluateVector = IsComputation<VT>::value || RequiresEvaluation<VT>::value };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the right-hand side dense matrix expression.
   enum : bool { evaluateMatrix = ( IsComputation<MT>::value && IsSame<MET,VET>::value &&
                                    IsBLASCompatible<MET>::value ) || RequiresEvaluation<MT>::value };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! In case the target vector is SMP assignable and either the vector or the matrix operand
       require an intermediate evaluation, the nested \a value will be set to 1, otherwise it
       will be 0. */
   template< typename T1 >
   struct UseSMPAssign {
      enum : bool { value = T1::smpAssignable && ( evaluateVector || evaluateMatrix ) };
   };
   //**********************************************************************************************

      //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! In case the two involved vector types, the matrix type, and the scalar type are suited
       for a BLAS kernel, the nested \a value will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3, typename T4 >
   struct UseBlasKernel {
      enum : bool { value = BLAZE_BLAS_MODE &&
                            HasMutableDataAccess<T1>::value &&
                            HasConstDataAccess<T2>::value &&
                            HasConstDataAccess<T3>::value &&
                            !IsDiagonal<T3>::value &&
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
   /*! In case the two involved vector types, the matrix type, and the scalar type are suited
       for a vectorized computation of the scaled vector/matrix multiplication, the nested
       \a value will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3, typename T4 >
   struct UseVectorizedDefaultKernel {
      enum : bool { value = useOptimizedKernels &&
                            !IsDiagonal<T3>::value &&
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
   typedef DVecScalarMultExpr<VMM,ST,true>  This;           //!< Type of this DVecScalarMultExpr instance.
   typedef MultTrait_<RES,ST>               ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>       TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<ResultType>         ElementType;    //!< Resulting element type.
   typedef SIMDTrait_<ElementType>          SIMDType;       //!< Resulting SIMD element type.
   typedef const ElementType                ReturnType;     //!< Return type for expression template evaluations.
   typedef const ResultType                 CompositeType;  //!< Data type for composite expression templates.

   //! Composite type of the left-hand side dense vector expression.
   typedef const TDVecTDMatMultExpr<VT,MT>  LeftOperand;

   //! Composite type of the right-hand side scalar value.
   typedef ST  RightOperand;

   //! Type for the assignment of the dense vector operand of the left-hand side expression.
   typedef IfTrue_< evaluateVector, const VRT, VCT >  LT;

   //! Type for the assignment of the dense matrix operand of the left-hand side expression.
   typedef IfTrue_< evaluateMatrix, const MRT, MCT >  RT;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = !IsDiagonal<MT>::value &&
                               VT::simdEnabled && MT::simdEnabled &&
                               AreSIMDCombinable<VET,MET,ST>::value &&
                               HasSIMDAdd<VET,MET>::value &&
                               HasSIMDMult<VET,MET>::value };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = !evaluateVector && VT::smpAssignable &&
                                 !evaluateMatrix && MT::smpAssignable };
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
   explicit inline DVecScalarMultExpr( const VMM& vector, ST scalar )
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

   //**Size function*******************************************************************************
   /*!\brief Returns the current size/dimension of the vector.
   //
   // \return The size of the vector.
   */
   inline size_t size() const {
      return vector_.size();
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side dense vector operand.
   //
   // \return The left-hand side dense vector operand.
   */
   inline LeftOperand leftOperand() const {
      return vector_;
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
      return vector_.canAlias( alias );
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
      return vector_.isAliased( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const {
      return vector_.isAligned();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const {
      RightOperand_<VMM> A( vector_.rightOperand() );
      return ( !BLAZE_BLAS_IS_PARALLEL ||
               ( IsComputation<MT>::value && !evaluateMatrix ) ||
               ( A.rows() * A.columns() < TDVECTDMATMULT_THRESHOLD ) ) &&
             ( size() > SMP_TDVECTDMATMULT_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  vector_;  //!< Left-hand side dense vector of the multiplication expression.
   RightOperand scalar_;  //!< Right-hand side scalar of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense vectors*****************************************************************
   /*!\brief Assignment of a scaled transpose dense vector-transpose dense matrix multiplication
   //        to a transpose dense vector (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a scaled transpose dense
   // vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1  // Type of the target dense vector
           , bool TF >     // Transpose flag of the target dense vector
   friend inline void assign( DenseVector<VT1,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      LeftOperand_<VMM>  left ( rhs.vector_.leftOperand()  );
      RightOperand_<VMM> right( rhs.vector_.rightOperand() );

      if( right.rows() == 0UL ) {
         reset( ~lhs );
         return;
      }
      else if( right.columns() == 0UL ) {
         return;
      }

      LT x( serial( left  ) );  // Evaluation of the left-hand side dense vector operand
      RT A( serial( right ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( x.size()    == left.size()    , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == right.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == right.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.columns() == (~lhs).size()  , "Invalid vector size"       );

      DVecScalarMultExpr::selectAssignKernel( ~lhs, x, A, rhs.scalar_ );
   }
   //**********************************************************************************************

   //**Assignment to dense vectors (kernel selection)**********************************************
   /*!\brief Selection of the kernel for an assignment of a scaled transpose dense vector-transpose
   //        dense matrix multiplication to a dense vector (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      if( ( IsDiagonal<MT1>::value ) ||
          ( IsComputation<MT>::value && !evaluateMatrix ) ||
          ( A.rows() * A.columns() < TDVECTDMATMULT_THRESHOLD ) )
         selectSmallAssignKernel( y, x, A, scalar );
      else
         selectBlasAssignKernel( y, x, A, scalar );
   }
   //**********************************************************************************************

   //**Default assignment to dense vectors*********************************************************
   /*!\brief Default assignment of a scaled transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment kernel for the scaled transpose dense vector-
   // transpose dense matrix multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectDefaultAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      y.assign( x * A * scalar );
   }
   //**********************************************************************************************

   //**Default assignment to dense vectors (small matrices)****************************************
   /*!\brief Default assignment of a small scaled transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a scaled transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1,ST2> >
      selectSmallAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      selectDefaultAssignKernel( y, x, A, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default assignment to dense vectors (small matrices)*****************************
   /*!\brief Vectorized default assignment of a small scaled transpose dense vector-transpose
   //        dense matrix multiplication (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default assignment kernel for the scaled transpose
   // dense vector-transpose dense matrix multiplication. This kernel is optimized for small
   // matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1,ST2> >
      selectSmallAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT2>::value || !IsPadded<MT1>::value );

      size_t j( 0UL );

      for( ; (j+8UL) <= N; j+=8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+7UL : j+8UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+3UL);
            xmm5 = xmm5 + x1 * A.load(i,j+4UL);
            xmm6 = xmm6 + x1 * A.load(i,j+5UL);
            xmm7 = xmm7 + x1 * A.load(i,j+6UL);
            xmm8 = xmm8 + x1 * A.load(i,j+7UL);
         }

         y[j    ] = sum( xmm1 ) * scalar;
         y[j+1UL] = sum( xmm2 ) * scalar;
         y[j+2UL] = sum( xmm3 ) * scalar;
         y[j+3UL] = sum( xmm4 ) * scalar;
         y[j+4UL] = sum( xmm5 ) * scalar;
         y[j+5UL] = sum( xmm6 ) * scalar;
         y[j+6UL] = sum( xmm7 ) * scalar;
         y[j+7UL] = sum( xmm8 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    ) * scalar;
            y[j+1UL] += x[i] * A(i,j+1UL) * scalar;
            y[j+2UL] += x[i] * A(i,j+2UL) * scalar;
            y[j+3UL] += x[i] * A(i,j+3UL) * scalar;
            y[j+4UL] += x[i] * A(i,j+4UL) * scalar;
            y[j+5UL] += x[i] * A(i,j+5UL) * scalar;
            y[j+6UL] += x[i] * A(i,j+6UL) * scalar;
            y[j+7UL] += x[i] * A(i,j+7UL) * scalar;
         }
      }

      for( ; (j+4UL) <= N; j+=4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+3UL : j+4UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+3UL);
         }

         y[j    ] = sum( xmm1 ) * scalar;
         y[j+1UL] = sum( xmm2 ) * scalar;
         y[j+2UL] = sum( xmm3 ) * scalar;
         y[j+3UL] = sum( xmm4 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    ) * scalar;
            y[j+1UL] += x[i] * A(i,j+1UL) * scalar;
            y[j+2UL] += x[i] * A(i,j+2UL) * scalar;
            y[j+3UL] += x[i] * A(i,j+3UL) * scalar;
         }
      }

      for( ; (j+3UL) <= N; j+=3UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+2UL : j+3UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
         }

         y[j    ] = sum( xmm1 ) * scalar;
         y[j+1UL] = sum( xmm2 ) * scalar;
         y[j+2UL] = sum( xmm3 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    ) * scalar;
            y[j+1UL] += x[i] * A(i,j+1UL) * scalar;
            y[j+2UL] += x[i] * A(i,j+2UL) * scalar;
         }
      }

      for( ; (j+2UL) <= N; j+=2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+1UL : j+2UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
         }

         y[j    ] = sum( xmm1 ) * scalar;
         y[j+1UL] = sum( xmm2 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    ) * scalar;
            y[j+1UL] += x[i] * A(i,j+1UL) * scalar;
         }
      }

      if( j < N )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            xmm1 = xmm1 + A.load(i,j) * x.load(i);
         }

         y[j] = sum( xmm1 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j] += x[i] * A(i,j) * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to dense vectors (large matrices)****************************************
   /*!\brief Default assignment of a large scaled transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a scaled transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1,ST2> >
      selectLargeAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      selectDefaultAssignKernel( y, x, A, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default assignment to dense vectors (large matrices)*****************************
   /*!\brief Vectorized default assignment of a large scaled transpose dense vector-transpose
   //        dense matrix multiplication (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default assignment kernel for the scaled transpose
   // dense vector-transpose dense matrix multiplication. This kernel is optimized for large
   // matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1,ST2> >
      selectLargeAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT2>::value || !IsPadded<MT1>::value );

      reset( y );

      size_t j( 0UL );

      for( ; (j+8UL) <= N; j+=8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+7UL : j+8UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) + x3 * A.load(i2,j+2UL) + x4 * A.load(i3,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) + x3 * A.load(i2,j+3UL) + x4 * A.load(i3,j+3UL) );
            y[j+4UL] += sum( x1 * A.load(i,j+4UL) + x2 * A.load(i1,j+4UL) + x3 * A.load(i2,j+4UL) + x4 * A.load(i3,j+4UL) );
            y[j+5UL] += sum( x1 * A.load(i,j+5UL) + x2 * A.load(i1,j+5UL) + x3 * A.load(i2,j+5UL) + x4 * A.load(i3,j+5UL) );
            y[j+6UL] += sum( x1 * A.load(i,j+6UL) + x2 * A.load(i1,j+6UL) + x3 * A.load(i2,j+6UL) + x4 * A.load(i3,j+6UL) );
            y[j+7UL] += sum( x1 * A.load(i,j+7UL) + x2 * A.load(i1,j+7UL) + x3 * A.load(i2,j+7UL) + x4 * A.load(i3,j+7UL) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) );
            y[j+4UL] += sum( x1 * A.load(i,j+4UL) + x2 * A.load(i1,j+4UL) );
            y[j+5UL] += sum( x1 * A.load(i,j+5UL) + x2 * A.load(i1,j+5UL) );
            y[j+6UL] += sum( x1 * A.load(i,j+6UL) + x2 * A.load(i1,j+6UL) );
            y[j+7UL] += sum( x1 * A.load(i,j+7UL) + x2 * A.load(i1,j+7UL) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] += sum( x1 * A.load(i,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) );
            y[j+4UL] += sum( x1 * A.load(i,j+4UL) );
            y[j+5UL] += sum( x1 * A.load(i,j+5UL) );
            y[j+6UL] += sum( x1 * A.load(i,j+6UL) );
            y[j+7UL] += sum( x1 * A.load(i,j+7UL) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
            y[j+2UL] += x[i] * A(i,j+2UL);
            y[j+3UL] += x[i] * A(i,j+3UL);
            y[j+4UL] += x[i] * A(i,j+4UL);
            y[j+5UL] += x[i] * A(i,j+5UL);
            y[j+6UL] += x[i] * A(i,j+6UL);
            y[j+7UL] += x[i] * A(i,j+7UL);
         }

         y[j    ] *= scalar;
         y[j+1UL] *= scalar;
         y[j+2UL] *= scalar;
         y[j+3UL] *= scalar;
         y[j+4UL] *= scalar;
         y[j+5UL] *= scalar;
         y[j+6UL] *= scalar;
         y[j+7UL] *= scalar;
      }

      for( ; (j+4UL) <= N; j+=4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+3UL : j+4UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) + x3 * A.load(i2,j+2UL) + x4 * A.load(i3,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) + x3 * A.load(i2,j+3UL) + x4 * A.load(i3,j+3UL) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] += sum( x1 * A.load(i,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) );
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) );
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
            y[j+2UL] += x[i] * A(i,j+2UL);
            y[j+3UL] += x[i] * A(i,j+3UL);
         }

         y[j    ] *= scalar;
         y[j+1UL] *= scalar;
         y[j+2UL] *= scalar;
         y[j+3UL] *= scalar;
      }

      for( ; (j+2UL) <= N; j+=2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+1UL : j+2UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] += sum( x1 * A.load(i,j    ) );
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    );
            y[j+1UL] += x[i] * A(i,j+1UL);
         }

         y[j    ] *= scalar;
         y[j+1UL] *= scalar;
      }

      if( j < N )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j] += sum( x1 * A.load(i,j) + x2 * A.load(i1,j) + x3 * A.load(i2,j) + x4 * A.load(i3,j) );
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j] += sum( x1 * A.load(i,j) + x2 * A.load(i1,j) );
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j] += sum( x1 * A.load(i,j) );
         }

         for( ; remainder && i<iend; ++i ) {
            y[j] += x[i] * A(i,j);
         }

         y[j] *= scalar;
      }
   }
   //**********************************************************************************************

   //**BLAS-based assignment to dense vectors (default)********************************************
   /*!\brief Default assignment of a scaled transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a large scaled
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseBlasKernel<VT1,VT2,MT1,ST2> >
      selectBlasAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      selectLargeAssignKernel( y, x, A, scalar );
   }
   //**********************************************************************************************

   //**BLAS-based assignment to dense vectors******************************************************
#if BLAZE_BLAS_MODE
   /*!\brief BLAS-based assignment of a scaled transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function performs the scaled transpose dense vector-transpose dense matrix
   // multiplication based on the according BLAS functionality.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseBlasKernel<VT1,VT2,MT1,ST2> >
      selectBlasAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      typedef ElementType_<VT1>  ET;

      if( IsTriangular<MT1>::value ) {
         assign( y, scalar * x );
         trmv( y, A, ( IsLower<MT1>::value )?( CblasLower ):( CblasUpper ) );
      }
      else {
         gemv( y, x, A, ET(scalar), ET(0) );
      }
   }
#endif
   //**********************************************************************************************

   //**Assignment to sparse vectors****************************************************************
   /*!\brief Assignment of a scaled transpose dense vector-transpose dense matrix multiplication
   //        to a transpose sparse vector (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a scaled transpose dense
   // vector-transpose dense matrix multiplication expression to a sparse vector.
   */
   template< typename VT1  // Type of the target sparse vector
           , bool TF >     // Transpose flag of the target sparse vector
   friend inline void assign( SparseVector<VT1,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( serial( rhs ) );
      assign( ~lhs, tmp );
   }
   //**********************************************************************************************

   //**Addition assignment to dense vectors********************************************************
   /*!\brief Addition assignment of a scaled transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a scaled transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1  // Type of the target dense vector
           , bool TF >     // Transpose flag of the target dense vector
   friend inline void addAssign( DenseVector<VT1,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      LeftOperand_<VMM>  left ( rhs.vector_.leftOperand()  );
      RightOperand_<VMM> right( rhs.vector_.rightOperand() );

      if( right.rows() == 0UL || right.columns() == 0UL ) {
         return;
      }

      LT x( serial( left  ) );  // Evaluation of the left-hand side dense vector operand
      RT A( serial( right ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( x.size()    == left.size()    , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == right.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == right.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.columns() == (~lhs).size()  , "Invalid vector size"       );

      DVecScalarMultExpr::selectAddAssignKernel( ~lhs, x, A, rhs.scalar_ );
   }
   //**********************************************************************************************

   //**Addition assignment to dense vectors (kernel selection)*************************************
   /*!\brief Selection of the kernel for an addition assignment of a scaled transpose dense vector-
   //        transpose dense matrix multiplication to a dense vector (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectAddAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      if( ( IsDiagonal<MT1>::value ) ||
          ( IsComputation<MT>::value && !evaluateMatrix ) ||
          ( A.rows() * A.columns() < TDVECDMATMULT_THRESHOLD ) )
         selectSmallAddAssignKernel( y, x, A, scalar );
      else
         selectBlasAddAssignKernel( y, x, A, scalar );
   }
   //**********************************************************************************************

   //**Default addition assignment to dense vectors************************************************
   /*!\brief Default addition assignment of a scaled transpose dense vector-transpose dense
   //        matrix multiplication (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default addition assignment kernel for the scaled transpose
   // dense vector-transpose dense matrix multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectDefaultAddAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      y.addAssign( x * A * scalar );
   }
   //**********************************************************************************************

   //**Default addition assignment to dense vectors (small matrices)*******************************
   /*!\brief Default addition assignment of a small scaled transpose dense vector-transpose dense
   //        matrix multiplication (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a scaled
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1,ST2> >
      selectSmallAddAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      selectDefaultAddAssignKernel( y, x, A, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default addition assignment to dense vectors (small matrices)********************
   /*!\brief Vectorized default addition assignment of a small scaled transpose dense vector-
   //        transpose dense matrix multiplication (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default addition assignment kernel for the scaled
   // transpose dense vector-transpose dense matrix multiplication. This kernel is optimized for
   // small matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1,ST2> >
      selectSmallAddAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT2>::value || !IsPadded<MT1>::value );

      size_t j( 0UL );

      for( ; (j+8UL) <= N; j+=8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+7UL : j+8UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+3UL);
            xmm5 = xmm5 + x1 * A.load(i,j+4UL);
            xmm6 = xmm6 + x1 * A.load(i,j+5UL);
            xmm7 = xmm7 + x1 * A.load(i,j+6UL);
            xmm8 = xmm8 + x1 * A.load(i,j+7UL);
         }

         y[j    ] += sum( xmm1 ) * scalar;
         y[j+1UL] += sum( xmm2 ) * scalar;
         y[j+2UL] += sum( xmm3 ) * scalar;
         y[j+3UL] += sum( xmm4 ) * scalar;
         y[j+4UL] += sum( xmm5 ) * scalar;
         y[j+5UL] += sum( xmm6 ) * scalar;
         y[j+6UL] += sum( xmm7 ) * scalar;
         y[j+7UL] += sum( xmm8 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    ) * scalar;
            y[j+1UL] += x[i] * A(i,j+1UL) * scalar;
            y[j+2UL] += x[i] * A(i,j+2UL) * scalar;
            y[j+3UL] += x[i] * A(i,j+3UL) * scalar;
            y[j+4UL] += x[i] * A(i,j+4UL) * scalar;
            y[j+5UL] += x[i] * A(i,j+5UL) * scalar;
            y[j+6UL] += x[i] * A(i,j+6UL) * scalar;
            y[j+7UL] += x[i] * A(i,j+7UL) * scalar;
         }
      }

      for( ; (j+4UL) <= N; j+=4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+3UL : j+4UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+3UL);
         }

         y[j    ] += sum( xmm1 ) * scalar;
         y[j+1UL] += sum( xmm2 ) * scalar;
         y[j+2UL] += sum( xmm3 ) * scalar;
         y[j+3UL] += sum( xmm4 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    ) * scalar;
            y[j+1UL] += x[i] * A(i,j+1UL) * scalar;
            y[j+2UL] += x[i] * A(i,j+2UL) * scalar;
            y[j+3UL] += x[i] * A(i,j+3UL) * scalar;
         }
      }

      for( ; (j+3UL) <= N; j+=3UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+2UL : j+3UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
         }

         y[j    ] += sum( xmm1 ) * scalar;
         y[j+1UL] += sum( xmm2 ) * scalar;
         y[j+2UL] += sum( xmm3 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    ) * scalar;
            y[j+1UL] += x[i] * A(i,j+1UL) * scalar;
            y[j+2UL] += x[i] * A(i,j+2UL) * scalar;
         }
      }

      for( ; (j+2UL) <= N; j+=2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+1UL : j+2UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
         }

         y[j    ] += sum( xmm1 ) * scalar;
         y[j+1UL] += sum( xmm2 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    ) * scalar;
            y[j+1UL] += x[i] * A(i,j+1UL) * scalar;
         }
      }

      if( j < N )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            xmm1 = xmm1 + A.load(i,j) * x.load(i);
         }

         y[j] += sum( xmm1 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j] += x[i] * A(i,j) * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default addition assignment to dense vectors (large matrices)*******************************
   /*!\brief Default addition assignment of a large scaled transpose dense vector-transpose dense
   //        matrix multiplication (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a scaled
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1,ST2> >
      selectLargeAddAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      selectDefaultAddAssignKernel( y, x, A, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default addition assignment to dense vectors (large matrices)********************
   /*!\brief Vectorized default addition assignment of a large scaled transpose dense vector-
   //        transpose dense matrix multiplication (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default addition assignment kernel for the scaled
   // transpose dense vector-transpose dense matrix multiplication. This kernel is optimized for
   // large matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1,ST2> >
      selectLargeAddAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT2>::value || !IsPadded<MT1>::value );

      size_t j( 0UL );

      for( ; (j+8UL) <= N; j+=8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+7UL : j+8UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) ) * scalar;
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) ) * scalar;
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) + x3 * A.load(i2,j+2UL) + x4 * A.load(i3,j+2UL) ) * scalar;
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) + x3 * A.load(i2,j+3UL) + x4 * A.load(i3,j+3UL) ) * scalar;
            y[j+4UL] += sum( x1 * A.load(i,j+4UL) + x2 * A.load(i1,j+4UL) + x3 * A.load(i2,j+4UL) + x4 * A.load(i3,j+4UL) ) * scalar;
            y[j+5UL] += sum( x1 * A.load(i,j+5UL) + x2 * A.load(i1,j+5UL) + x3 * A.load(i2,j+5UL) + x4 * A.load(i3,j+5UL) ) * scalar;
            y[j+6UL] += sum( x1 * A.load(i,j+6UL) + x2 * A.load(i1,j+6UL) + x3 * A.load(i2,j+6UL) + x4 * A.load(i3,j+6UL) ) * scalar;
            y[j+7UL] += sum( x1 * A.load(i,j+7UL) + x2 * A.load(i1,j+7UL) + x3 * A.load(i2,j+7UL) + x4 * A.load(i3,j+7UL) ) * scalar;
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) ) * scalar;
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) ) * scalar;
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) ) * scalar;
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) ) * scalar;
            y[j+4UL] += sum( x1 * A.load(i,j+4UL) + x2 * A.load(i1,j+4UL) ) * scalar;
            y[j+5UL] += sum( x1 * A.load(i,j+5UL) + x2 * A.load(i1,j+5UL) ) * scalar;
            y[j+6UL] += sum( x1 * A.load(i,j+6UL) + x2 * A.load(i1,j+6UL) ) * scalar;
            y[j+7UL] += sum( x1 * A.load(i,j+7UL) + x2 * A.load(i1,j+7UL) ) * scalar;
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] += sum( x1 * A.load(i,j    ) ) * scalar;
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) ) * scalar;
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) ) * scalar;
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) ) * scalar;
            y[j+4UL] += sum( x1 * A.load(i,j+4UL) ) * scalar;
            y[j+5UL] += sum( x1 * A.load(i,j+5UL) ) * scalar;
            y[j+6UL] += sum( x1 * A.load(i,j+6UL) ) * scalar;
            y[j+7UL] += sum( x1 * A.load(i,j+7UL) ) * scalar;
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    ) * scalar;
            y[j+1UL] += x[i] * A(i,j+1UL) * scalar;
            y[j+2UL] += x[i] * A(i,j+2UL) * scalar;
            y[j+3UL] += x[i] * A(i,j+3UL) * scalar;
            y[j+4UL] += x[i] * A(i,j+4UL) * scalar;
            y[j+5UL] += x[i] * A(i,j+5UL) * scalar;
            y[j+6UL] += x[i] * A(i,j+6UL) * scalar;
            y[j+7UL] += x[i] * A(i,j+7UL) * scalar;
         }
      }

      for( ; (j+4UL) <= N; j+=4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+3UL : j+4UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) ) * scalar;
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) ) * scalar;
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) + x3 * A.load(i2,j+2UL) + x4 * A.load(i3,j+2UL) ) * scalar;
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) + x3 * A.load(i2,j+3UL) + x4 * A.load(i3,j+3UL) ) * scalar;
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) ) * scalar;
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) ) * scalar;
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) ) * scalar;
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) ) * scalar;
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] += sum( x1 * A.load(i,j    ) ) * scalar;
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) ) * scalar;
            y[j+2UL] += sum( x1 * A.load(i,j+2UL) ) * scalar;
            y[j+3UL] += sum( x1 * A.load(i,j+3UL) ) * scalar;
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    ) * scalar;
            y[j+1UL] += x[i] * A(i,j+1UL) * scalar;
            y[j+2UL] += x[i] * A(i,j+2UL) * scalar;
            y[j+3UL] += x[i] * A(i,j+3UL) * scalar;
         }
      }

      for( ; (j+2UL) <= N; j+=2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+1UL : j+2UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) ) * scalar;
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) ) * scalar;
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] += sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) ) * scalar;
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) ) * scalar;
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] += sum( x1 * A.load(i,j    ) ) * scalar;
            y[j+1UL] += sum( x1 * A.load(i,j+1UL) ) * scalar;
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] += x[i] * A(i,j    ) * scalar;
            y[j+1UL] += x[i] * A(i,j+1UL) * scalar;
         }
      }

      if( j < N )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j] += sum( x1 * A.load(i,j) + x2 * A.load(i1,j) + x3 * A.load(i2,j) + x4 * A.load(i3,j) ) * scalar;
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j] += sum( x1 * A.load(i,j) + x2 * A.load(i1,j) ) * scalar;
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j] += sum( x1 * A.load(i,j) ) * scalar;
         }

         for( ; remainder && i<iend; ++i ) {
            y[j] += x[i] * A(i,j) * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense vectors (default)***********************************
   /*!\brief Default addition assignment of a scaled transpose dense vector-transpose dense matrix
   //        multiplication (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a large
   // scaled transpose dense vector-transpose dense matrix multiplication expression to a dense
   // vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseBlasKernel<VT1,VT2,MT1,ST2> >
      selectBlasAddAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      selectLargeAddAssignKernel( y, x, A, scalar );
   }
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense vectors*********************************************
#if BLAZE_BLAS_MODE
   /*!\brief BLAS-based addition assignment of a scaled transpose dense vector-transpose dense
   //        matrix multiplication (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor
   // \return void
   //
   // This function performs the scaled transpose dense vector-transpose dense matrix
   // multiplication based on the according BLAS functionality.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseBlasKernel<VT1,VT2,MT1,ST2> >
      selectBlasAddAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      typedef ElementType_<VT1>  ET;

      if( IsTriangular<MT1>::value ) {
         ResultType_<VT1> tmp( serial( scalar * x ) );
         trmv( tmp, A, ( IsLower<MT1>::value )?( CblasLower ):( CblasUpper ) );
         addAssign( y, tmp );
      }
      else {
         gemv( y, x, A, ET(scalar), ET(1) );
      }
   }
#endif
   //**********************************************************************************************

   //**Addition assignment to sparse vectors*******************************************************
   // No special implementation for the addition assignment to sparse vectors.
   //**********************************************************************************************

   //**Subtraction assignment to dense vectors*****************************************************
   /*!\brief Subtraction assignment of a scaled transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a scaled
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1  // Type of the target dense vector
           , bool TF >     // Transpose flag of the target dense vector
   friend inline void subAssign( DenseVector<VT1,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      LeftOperand_<VMM>  left ( rhs.vector_.leftOperand()  );
      RightOperand_<VMM> right( rhs.vector_.rightOperand() );

      if( right.rows() == 0UL || right.columns() == 0UL ) {
         return;
      }

      LT x( serial( left  ) );  // Evaluation of the left-hand side dense vector operand
      RT A( serial( right ) );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( x.size()    == left.size()    , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == right.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == right.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.columns() == (~lhs).size()  , "Invalid vector size"       );

      DVecScalarMultExpr::selectSubAssignKernel( ~lhs, x, A, rhs.scalar_ );
   }
   //**********************************************************************************************

   //**Subtraction assignment to dense vectors (kernel selection)**********************************
   /*!\brief Selection of the kernel for a subtraction assignment of a scaled transpose dense vector-
   //        transpose dense matrix multiplication to a dense vector (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectSubAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      if( ( IsDiagonal<MT1>::value ) ||
          ( IsComputation<MT>::value && !evaluateMatrix ) ||
          ( A.rows() * A.columns() < TDVECDMATMULT_THRESHOLD ) )
         selectSmallSubAssignKernel( y, x, A, scalar );
      else
         selectBlasSubAssignKernel( y, x, A, scalar );
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors*********************************************
   /*!\brief Default subtraction assignment of a scaled transpose dense vector-transpose dense
   //        matrix multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default subtraction assignment kernel for the scaled transpose
   // dense vector-transpose dense matrix multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectDefaultSubAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      y.subAssign( x * A * scalar );
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors (small matrices)****************************
   /*!\brief Default subtraction assignment of a small scaled transpose dense vector-transpose
   //        dense matrix multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a scaled
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1,ST2> >
      selectSmallSubAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      selectDefaultSubAssignKernel( y, x, A, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to dense vectors (small matrices)*****************
   /*!\brief Vectorized default subtraction assignment of a small scaled transpose dense vector-
   //        transpose matrix multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment kernel for the scaled
   // transpose dense vector-transpose dense matrix multiplication. This kernel is optimized for
   // small matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1,ST2> >
      selectSmallSubAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT2>::value || !IsPadded<MT1>::value );

      size_t j( 0UL );

      for( ; (j+8UL) <= N; j+=8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+7UL : j+8UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+3UL);
            xmm5 = xmm5 + x1 * A.load(i,j+4UL);
            xmm6 = xmm6 + x1 * A.load(i,j+5UL);
            xmm7 = xmm7 + x1 * A.load(i,j+6UL);
            xmm8 = xmm8 + x1 * A.load(i,j+7UL);
         }

         y[j    ] -= sum( xmm1 ) * scalar;
         y[j+1UL] -= sum( xmm2 ) * scalar;
         y[j+2UL] -= sum( xmm3 ) * scalar;
         y[j+3UL] -= sum( xmm4 ) * scalar;
         y[j+4UL] -= sum( xmm5 ) * scalar;
         y[j+5UL] -= sum( xmm6 ) * scalar;
         y[j+6UL] -= sum( xmm7 ) * scalar;
         y[j+7UL] -= sum( xmm8 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    ) * scalar;
            y[j+1UL] -= x[i] * A(i,j+1UL) * scalar;
            y[j+2UL] -= x[i] * A(i,j+2UL) * scalar;
            y[j+3UL] -= x[i] * A(i,j+3UL) * scalar;
            y[j+4UL] -= x[i] * A(i,j+4UL) * scalar;
            y[j+5UL] -= x[i] * A(i,j+5UL) * scalar;
            y[j+6UL] -= x[i] * A(i,j+6UL) * scalar;
            y[j+7UL] -= x[i] * A(i,j+7UL) * scalar;
         }
      }

      for( ; (j+4UL) <= N; j+=4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+3UL : j+4UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+3UL);
         }

         y[j    ] -= sum( xmm1 ) * scalar;
         y[j+1UL] -= sum( xmm2 ) * scalar;
         y[j+2UL] -= sum( xmm3 ) * scalar;
         y[j+3UL] -= sum( xmm4 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    ) * scalar;
            y[j+1UL] -= x[i] * A(i,j+1UL) * scalar;
            y[j+2UL] -= x[i] * A(i,j+2UL) * scalar;
            y[j+3UL] -= x[i] * A(i,j+3UL) * scalar;
         }
      }

      for( ; (j+3UL) <= N; j+=3UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+2UL : j+3UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
            xmm3 = xmm3 + x1 * A.load(i,j+2UL);
         }

         y[j    ] -= sum( xmm1 ) * scalar;
         y[j+1UL] -= sum( xmm2 ) * scalar;
         y[j+2UL] -= sum( xmm3 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    ) * scalar;
            y[j+1UL] -= x[i] * A(i,j+1UL) * scalar;
            y[j+2UL] -= x[i] * A(i,j+2UL) * scalar;
         }
      }

      for( ; (j+2UL) <= N; j+=2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+1UL : j+2UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1, xmm2;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            xmm1 = xmm1 + x1 * A.load(i,j    );
            xmm2 = xmm2 + x1 * A.load(i,j+1UL);
         }

         y[j    ] -= sum( xmm1 ) * scalar;
         y[j+1UL] -= sum( xmm2 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    ) * scalar;
            y[j+1UL] -= x[i] * A(i,j+1UL) * scalar;
         }
      }

      if( j < N )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         SIMDType xmm1;
         size_t i( ibegin );

         for( ; i<ipos; i+=SIMDSIZE ) {
            xmm1 = xmm1 + A.load(i,j) * x.load(i);
         }

         y[j] -= sum( xmm1 ) * scalar;

         for( ; remainder && i<iend; ++i ) {
            y[j] -= x[i] * A(i,j) * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors (large matrices)****************************
   /*!\brief Default subtraction assignment of a large scaled transpose dense vector-transpose
   //        dense matrix multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a scaled
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1,ST2> >
      selectLargeSubAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      selectDefaultSubAssignKernel( y, x, A, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to dense vectors (large matrices)*****************
   /*!\brief Vectorized default subtraction assignment of a large scaled transpose dense vector-
   //        transpose matrix multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment kernel for the scaled
   // transpose dense vector-transpose dense matrix multiplication. This kernel is optimized for
   // large matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1,ST2> >
      selectLargeSubAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT2>::value || !IsPadded<MT1>::value );

      size_t j( 0UL );

      for( ; (j+8UL) <= N; j+=8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+7UL : j+8UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] -= sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) ) * scalar;
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) ) * scalar;
            y[j+2UL] -= sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) + x3 * A.load(i2,j+2UL) + x4 * A.load(i3,j+2UL) ) * scalar;
            y[j+3UL] -= sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) + x3 * A.load(i2,j+3UL) + x4 * A.load(i3,j+3UL) ) * scalar;
            y[j+4UL] -= sum( x1 * A.load(i,j+4UL) + x2 * A.load(i1,j+4UL) + x3 * A.load(i2,j+4UL) + x4 * A.load(i3,j+4UL) ) * scalar;
            y[j+5UL] -= sum( x1 * A.load(i,j+5UL) + x2 * A.load(i1,j+5UL) + x3 * A.load(i2,j+5UL) + x4 * A.load(i3,j+5UL) ) * scalar;
            y[j+6UL] -= sum( x1 * A.load(i,j+6UL) + x2 * A.load(i1,j+6UL) + x3 * A.load(i2,j+6UL) + x4 * A.load(i3,j+6UL) ) * scalar;
            y[j+7UL] -= sum( x1 * A.load(i,j+7UL) + x2 * A.load(i1,j+7UL) + x3 * A.load(i2,j+7UL) + x4 * A.load(i3,j+7UL) ) * scalar;
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] -= sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) ) * scalar;
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) ) * scalar;
            y[j+2UL] -= sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) ) * scalar;
            y[j+3UL] -= sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) ) * scalar;
            y[j+4UL] -= sum( x1 * A.load(i,j+4UL) + x2 * A.load(i1,j+4UL) ) * scalar;
            y[j+5UL] -= sum( x1 * A.load(i,j+5UL) + x2 * A.load(i1,j+5UL) ) * scalar;
            y[j+6UL] -= sum( x1 * A.load(i,j+6UL) + x2 * A.load(i1,j+6UL) ) * scalar;
            y[j+7UL] -= sum( x1 * A.load(i,j+7UL) + x2 * A.load(i1,j+7UL) ) * scalar;
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] -= sum( x1 * A.load(i,j    ) ) * scalar;
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) ) * scalar;
            y[j+2UL] -= sum( x1 * A.load(i,j+2UL) ) * scalar;
            y[j+3UL] -= sum( x1 * A.load(i,j+3UL) ) * scalar;
            y[j+4UL] -= sum( x1 * A.load(i,j+4UL) ) * scalar;
            y[j+5UL] -= sum( x1 * A.load(i,j+5UL) ) * scalar;
            y[j+6UL] -= sum( x1 * A.load(i,j+6UL) ) * scalar;
            y[j+7UL] -= sum( x1 * A.load(i,j+7UL) ) * scalar;
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    ) * scalar;
            y[j+1UL] -= x[i] * A(i,j+1UL) * scalar;
            y[j+2UL] -= x[i] * A(i,j+2UL) * scalar;
            y[j+3UL] -= x[i] * A(i,j+3UL) * scalar;
            y[j+4UL] -= x[i] * A(i,j+4UL) * scalar;
            y[j+5UL] -= x[i] * A(i,j+5UL) * scalar;
            y[j+6UL] -= x[i] * A(i,j+6UL) * scalar;
            y[j+7UL] -= x[i] * A(i,j+7UL) * scalar;
         }
      }

      for( ; (j+4UL) <= N; j+=4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+3UL : j+4UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] -= sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) ) * scalar;
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) ) * scalar;
            y[j+2UL] -= sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) + x3 * A.load(i2,j+2UL) + x4 * A.load(i3,j+2UL) ) * scalar;
            y[j+3UL] -= sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) + x3 * A.load(i2,j+3UL) + x4 * A.load(i3,j+3UL) ) * scalar;
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] -= sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) ) * scalar;
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) ) * scalar;
            y[j+2UL] -= sum( x1 * A.load(i,j+2UL) + x2 * A.load(i1,j+2UL) ) * scalar;
            y[j+3UL] -= sum( x1 * A.load(i,j+3UL) + x2 * A.load(i1,j+3UL) ) * scalar;
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] -= sum( x1 * A.load(i,j    ) ) * scalar;
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) ) * scalar;
            y[j+2UL] -= sum( x1 * A.load(i,j+2UL) ) * scalar;
            y[j+3UL] -= sum( x1 * A.load(i,j+3UL) ) * scalar;
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    ) * scalar;
            y[j+1UL] -= x[i] * A(i,j+1UL) * scalar;
            y[j+2UL] -= x[i] * A(i,j+2UL) * scalar;
            y[j+3UL] -= x[i] * A(i,j+3UL) * scalar;
         }
      }

      for( ; (j+2UL) <= N; j+=2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j+1UL : j+2UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j    ] -= sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) + x3 * A.load(i2,j    ) + x4 * A.load(i3,j    ) ) * scalar;
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) + x3 * A.load(i2,j+1UL) + x4 * A.load(i3,j+1UL) ) * scalar;
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j    ] -= sum( x1 * A.load(i,j    ) + x2 * A.load(i1,j    ) ) * scalar;
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) + x2 * A.load(i1,j+1UL) ) * scalar;
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j    ] -= sum( x1 * A.load(i,j    ) ) * scalar;
            y[j+1UL] -= sum( x1 * A.load(i,j+1UL) ) * scalar;
         }

         for( ; remainder && i<iend; ++i ) {
            y[j    ] -= x[i] * A(i,j    ) * scalar;
            y[j+1UL] -= x[i] * A(i,j+1UL) * scalar;
         }
      }

      if( j < N )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( ( IsStrictlyLower<MT1>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( IsStrictlyUpper<MT1>::value ? j : j+1UL )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         const size_t ipos( remainder ? ( iend & size_t(-SIMDSIZE) ) : iend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

         size_t i( ibegin );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            const size_t i1( i+SIMDSIZE     );
            const size_t i2( i+SIMDSIZE*2UL );
            const size_t i3( i+SIMDSIZE*3UL );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            const SIMDType x3( x.load(i2) );
            const SIMDType x4( x.load(i3) );
            y[j] -= sum( x1 * A.load(i,j) + x2 * A.load(i1,j) + x3 * A.load(i2,j) + x4 * A.load(i3,j) ) * scalar;
         }

         for( ; (i+SIMDSIZE) < ipos; i+=SIMDSIZE*2UL ) {
            const size_t i1( i+SIMDSIZE );
            const SIMDType x1( x.load(i ) );
            const SIMDType x2( x.load(i1) );
            y[j] -= sum( x1 * A.load(i,j) + x2 * A.load(i1,j) ) * scalar;
         }

         for( ; i<ipos; i+=SIMDSIZE ) {
            const SIMDType x1( x.load(i) );
            y[j] -= sum( x1 * A.load(i,j) ) * scalar;
         }

         for( ; remainder && i<iend; ++i ) {
            y[j] -= x[i] * A(i,j) * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense vectors (default)********************************
   /*!\brief Default subtraction assignment of a scaled transpose dense vector-transpose dense
   //        matrix multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a large
   // scaled transpose dense vector-transpose dense matrix multiplication expression to a dense
   // vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseBlasKernel<VT1,VT2,MT1,ST2> >
      selectBlasSubAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      selectLargeSubAssignKernel( y, x, A, scalar );
   }
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense vectors******************************************
#if BLAZE_BLAS_MODE
   /*!\brief BLAS-based subtraction assignment of a scaled transpose dense vector-transpose dense
   //        matrix multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function performs the scaled transpose dense vector-transpose dense matrix
   // multiplication based on the according BLAS functionality.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseBlasKernel<VT1,VT2,MT1,ST2> >
      selectBlasSubAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      typedef ElementType_<VT1>  ET;

      if( IsTriangular<MT1>::value ) {
         ResultType_<VT1> tmp( serial( scalar * x ) );
         trmv( tmp, A, ( IsLower<MT1>::value )?( CblasLower ):( CblasUpper ) );
         subAssign( y, tmp );
      }
      else {
         gemv( y, x, A, ET(-scalar), ET(1) );
      }
   }
#endif
   //**********************************************************************************************

   //**Subtraction assignment to sparse vectors****************************************************
   // No special implementation for the subtraction assignment to sparse vectors.
   //**********************************************************************************************

   //**Multiplication assignment to dense vectors**************************************************
   /*!\brief Multiplication assignment of a scaled transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T*=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a scaled
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1  // Type of the target dense vector
           , bool TF >     // Transpose flag of the target dense vector
   friend inline void multAssign( DenseVector<VT1,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( serial( rhs ) );
      multAssign( ~lhs, tmp );
   }
   //**********************************************************************************************

   //**Multiplication assignment to sparse vectors*************************************************
   // No special implementation for the multiplication assignment to sparse vectors.
   //**********************************************************************************************

   //**Division assignment to dense vectors********************************************************
   /*!\brief Division assignment of a scaled transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T/=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression divisor.
   // \return void
   //
   // This function implements the performance optimized division assignment of a scaled transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1  // Type of the target dense vector
           , bool TF >     // Transpose flag of the target dense vector
   friend inline void divAssign( DenseVector<VT1,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( serial( rhs ) );
      divAssign( ~lhs, tmp );
   }
   //**********************************************************************************************

   //**Division assignment to sparse vectors*******************************************************
   // No special implementation for the division assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP assignment to dense vectors*************************************************************
   /*!\brief SMP assignment of a scaled transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a scaled transpose
   // dense vector-transpose dense matrix multiplication expression to a dense vector. Due to
   // the explicit application of the SFINAE principle, this function can only be selected by
   // the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1  // Type of the target dense vector
           , bool TF >     // Transpose flag of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAssign( DenseVector<VT1,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      LeftOperand_<VMM>  left ( rhs.vector_.leftOperand()  );
      RightOperand_<VMM> right( rhs.vector_.rightOperand() );

      if( right.rows() == 0UL ) {
         reset( ~lhs );
         return;
      }
      else if( right.columns() == 0UL ) {
         return;
      }

      LT x( left  );  // Evaluation of the left-hand side dense vector operand
      RT A( right );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( x.size()    == left.size()    , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == right.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == right.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.columns() == (~lhs).size()  , "Invalid vector size"       );

      smpAssign( ~lhs, x * A * rhs.scalar_ );
   }
   //**********************************************************************************************

   //**SMP assignment to sparse vectors************************************************************
   /*!\brief SMP assignment of a scaled transpose dense vector-transpose dense matrix
   //        multiplication to a transpose sparse vector (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a scaled transpose
   // dense vector-transpose dense matrix multiplication expression to a sparse vector. Due to
   // the explicit application of the SFINAE principle, this function can only be selected by
   // the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1  // Type of the target sparse vector
           , bool TF >     // Transpose flag of the target sparse vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAssign( SparseVector<VT1,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( rhs );
      smpAssign( ~lhs, tmp );
   }
   //**********************************************************************************************

   //**SMP addition assignment to dense vectors****************************************************
   /*!\brief SMP addition assignment of a scaled transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a scaled
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   // Due to the explicit application of the SFINAE principle, this function can only be selected
   // by the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1  // Type of the target dense vector
           , bool TF >     // Transpose flag of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAddAssign( DenseVector<VT1,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      LeftOperand_<VMM>  left ( rhs.vector_.leftOperand()  );
      RightOperand_<VMM> right( rhs.vector_.rightOperand() );

      if( right.rows() == 0UL || right.columns() == 0UL ) {
         return;
      }

      LT x( left  );  // Evaluation of the left-hand side dense vector operand
      RT A( right );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( x.size()    == left.size()    , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == right.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == right.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.columns() == (~lhs).size()  , "Invalid vector size"       );

      smpAddAssign( ~lhs, x * A * rhs.scalar_ );
   }
   //**********************************************************************************************

   //**SMP addition assignment to sparse vectors***************************************************
   // No special implementation for the SMP addition assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP subtraction assignment to dense vectors*************************************************
   /*!\brief SMP subtraction assignment of a scaled transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a scaled
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   // Due to the explicit application of the SFINAE principle, this function can only be selected
   // by the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1  // Type of the target dense vector
           , bool TF >     // Transpose flag of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpSubAssign( DenseVector<VT1,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      LeftOperand_<VMM>  left ( rhs.vector_.leftOperand()  );
      RightOperand_<VMM> right( rhs.vector_.rightOperand() );

      if( right.rows() == 0UL || right.columns() == 0UL ) {
         return;
      }

      LT x( left  );  // Evaluation of the left-hand side dense vector operand
      RT A( right );  // Evaluation of the right-hand side dense matrix operand

      BLAZE_INTERNAL_ASSERT( x.size()    == left.size()    , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == right.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == right.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.columns() == (~lhs).size()  , "Invalid vector size"       );

      smpSubAssign( ~lhs, x * A * rhs.scalar_ );
   }
   //**********************************************************************************************

   //**SMP subtraction assignment to sparse vectors************************************************
   // No special implementation for the SMP subtraction assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP multiplication assignment to dense vectors**********************************************
   /*!\brief SMP multiplication assignment of a scaled transpose dense vector-transpose dense
   //        matrix multiplication to a transpose dense vector (\f$ \vec{y}^T*=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a scaled
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   // Due to the explicit application of the SFINAE principle, this function can only be selected
   // by the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1  // Type of the target dense vector
           , bool TF >     // Transpose flag of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpMultAssign( DenseVector<VT1,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( rhs );
      smpMultAssign( ~lhs, tmp );
   }
   //**********************************************************************************************

   //**SMP multiplication assignment to sparse vectors*********************************************
   // No special implementation for the SMP multiplication assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP division assignment to dense vectors****************************************************
   /*!\brief SMP division assignment of a scaled transpose dense vector-transpose dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T/=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression divisor.
   // \return void
   //
   // This function implements the performance optimized SMP division assignment of a scaled
   // transpose dense vector-transpose dense matrix multiplication expression to a dense vector.
   // Due to the explicit application of the SFINAE principle, this function can only be selected
   // by the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1  // Type of the target dense vector
           , bool TF >     // Transpose flag of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpDivAssign( DenseVector<VT1,TF>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( rhs );
      smpDivAssign( ~lhs, tmp );
   }
   //**********************************************************************************************

   //**SMP division assignment to sparse vectors***************************************************
   // No special implementation for the SMP division assignment to sparse vectors.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( VMM );
   BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( VMM );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( VT );
   BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE  ( VT );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( MT );
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
/*!\brief Multiplication operator for the multiplication of a transpose dense vector and a
//        column-major dense matrix (\f$ \vec{y}^T=\vec{x}^T*A \f$).
// \ingroup dense_matrix
//
// \param vec The left-hand side transpose dense vector for the multiplication.
// \param mat The right-hand side column-major dense matrix for the multiplication.
// \return The resulting transpose vector.
// \exception std::invalid_argument Vector and matrix sizes do not match.
//
// This operator represents the multiplication between a transpose dense vector and a column-major
// dense matrix:

   \code
   using blaze::rowVector;
   using blaze::columnMajor;

   blaze::DynamicVector<double,rowVector> x, y;
   blaze::DynamicMatrix<double,columnMajor> A;
   // ... Resizing and initialization
   y = x * A;
   \endcode

// The operator returns an expression representing a transpose dense vector of the higher-order
// element type of the two involved element types \a T1::ElementType and \a T2::ElementType.
// Both the dense matrix type \a T1 and the dense vector type \a T2 as well as the two element
// types \a T1::ElementType and \a T2::ElementType have to be supported by the MultTrait class
// template.\n
// In case the current size of the vector \a vec doesn't match the current number of rows of
// the matrix \a mat, a \a std::invalid_argument is thrown.
*/
template< typename T1    // Type of the left-hand side dense vector
        , typename T2 >  // Type of the right-hand side dense matrix
inline const DisableIf_< IsMatMatMultExpr<T2>, TDVecTDMatMultExpr<T1,T2> >
   operator*( const DenseVector<T1,true>& vec, const DenseMatrix<T2,true>& mat )
{
   BLAZE_FUNCTION_TRACE;

   if( (~vec).size() != (~mat).rows() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector and matrix sizes do not match" );
   }

   return TDVecTDMatMultExpr<T1,T2>( ~vec, ~mat );
}
//*************************************************************************************************




//=================================================================================================
//
//  SIZE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, typename MT >
struct Size< TDVecTDMatMultExpr<VT,MT> > : public Columns<MT>
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
template< typename VT, typename MT >
struct IsAligned< TDVecTDMatMultExpr<VT,MT> >
   : public BoolConstant< And< IsAligned<VT>, IsAligned<MT> >::value >
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
template< typename VT, typename MT, bool AF >
struct SubvectorExprTrait< TDVecTDMatMultExpr<VT,MT>, AF >
{
 public:
   //**********************************************************************************************
   using Type = MultExprTrait_< SubvectorExprTrait_<const VT,AF>
                              , SubmatrixExprTrait_<const MT,AF> >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
