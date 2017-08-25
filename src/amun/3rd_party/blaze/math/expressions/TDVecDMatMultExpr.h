//=================================================================================================
/*!
//  \file blaze/math/expressions/TDVecDMatMultExpr.h
//  \brief Header file for the transpose dense vector/dense matrix multiplication expression
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

#ifndef _BLAZE_MATH_EXPRESSIONS_TDVECDMATMULTEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_TDVECDMATMULTEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/blas/gemv.h>
#include <blaze/math/blas/trmv.h>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/constraints/RowMajorMatrix.h>
#include <blaze/math/constraints/RowVector.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/TVecMatMultExpr.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/TVecMatMultExpr.h>
#include <blaze/math/expressions/VecScalarMultExpr.h>
#include <blaze/math/Functions.h>
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
//  CLASS TDVECDMATMULTEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for transpose dense vector-dense matrix multiplications.
// \ingroup dense_vector_expression
//
// The TDVecDMatMultExpr class represents the compile time expression for multiplications
// between transpose dense vectors and dense matrices.
*/
template< typename VT    // Type of the left-hand side dense vector
        , typename MT >  // Type of the right-hand side dense matrix
class TDVecDMatMultExpr : public DenseVector< TDVecDMatMultExpr<VT,MT>, true >
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
   typedef TDVecDMatMultExpr<VT,MT>    This;           //!< Type of this TDVecDMatMultExpr instance.
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
   /*!\brief Constructor for the TDVecDMatMultExpr class.
   //
   // \param vec The left-hand side vector operand of the multiplication expression.
   // \param mat The right-hand side matrix operand of the multiplication expression.
   */
   explicit inline TDVecDMatMultExpr( const VT& vec, const MT& mat ) noexcept
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
   /*!\brief Returns the right-hand side dense matrix operand.
   //
   // \return The right-hand side dense matrix operand.
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
               ( mat_.rows() * mat_.columns() < TDVECDMATMULT_THRESHOLD ) ) &&
             ( size() > SMP_TDVECDMATMULT_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  vec_;  //!< Left-hand side dense vector of the multiplication expression.
   RightOperand mat_;  //!< Right-hand side dense matrix of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense vectors*****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a transpose dense vector-dense matrix multiplication to a transpose
   //        dense vector (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a transpose dense vector-
   // dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void assign( DenseVector<VT1,true>& lhs, const TDVecDMatMultExpr& rhs )
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

      TDVecDMatMultExpr::selectAssignKernel( ~lhs, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to dense vectors (kernel selection)**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for an assignment of a transpose dense vector-dense matrix
   //        multiplication to a dense vector (\f$ \vec{y}^T=\vec{x}^T*A \f$).
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
          ( A.rows() * A.columns() < TDVECDMATMULT_THRESHOLD ) )
         selectSmallAssignKernel( y, x, A );
      else
         selectBlasAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense vectors*********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the default assignment kernel for the transpose dense vector-
   // dense matrix multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline void selectDefaultAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      if( IsStrictlyUpper<MT1>::value ) {
         reset( y[0] );
      }

      if( !IsLower<MT1>::value )
      {
         const size_t jbegin( IsStrictlyUpper<MT1>::value ? 1UL : 0UL );
         for( size_t j=jbegin; j<N; ++j ) {
            y[j] = x[0UL] * A(0UL,j);
         }
      }

      for( size_t i=( IsLower<MT1>::value && !IsStrictlyLower<MT1>::value ? 0UL : 1UL ); i<M; ++i )
      {
         if( IsDiagonal<MT1>::value )
         {
            y[i] = x[i] * A(i,i);
         }
         else
         {
            const size_t jbegin( ( IsUpper<MT1>::value )
                                 ?( IsStrictlyUpper<MT1>::value ? i+1UL : i )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT1>::value )
                               ?( IsStrictlyLower<MT1>::value ? i-1UL : i )
                               :( N ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jnum( jend - jbegin );
            const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

            for( size_t j=jbegin; j<jpos; j+=2UL ) {
               y[j    ] += x[i] * A(i,j    );
               y[j+1UL] += x[i] * A(i,j+1UL);
            }
            if( jpos < jend ) {
               y[jpos] += x[i] * A(i,jpos);
            }
            if( IsLower<MT1>::value ) {
               y[jend] = x[i] * A(i,jend);
            }
         }
      }

      if( IsStrictlyLower<MT1>::value ) {
         reset( y[N-1UL] );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense vectors (small matrices)****************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a small transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a transpose dense
   // vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief Vectorized default assignment of a small transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the vectorized default assignment kernel for the transpose dense
   // vector-dense matrix multiplication. This kernel is optimized for small matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectSmallAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT1>::value || !IsPadded<MT1>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );

      for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*8UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
            xmm5 = xmm5 + x1 * A.load(i,j+SIMDSIZE*4UL);
            xmm6 = xmm6 + x1 * A.load(i,j+SIMDSIZE*5UL);
            xmm7 = xmm7 + x1 * A.load(i,j+SIMDSIZE*6UL);
            xmm8 = xmm8 + x1 * A.load(i,j+SIMDSIZE*7UL);
         }

         y.store( j             , xmm1 );
         y.store( j+SIMDSIZE    , xmm2 );
         y.store( j+SIMDSIZE*2UL, xmm3 );
         y.store( j+SIMDSIZE*3UL, xmm4 );
         y.store( j+SIMDSIZE*4UL, xmm5 );
         y.store( j+SIMDSIZE*5UL, xmm6 );
         y.store( j+SIMDSIZE*6UL, xmm7 );
         y.store( j+SIMDSIZE*7UL, xmm8 );
      }

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*4UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2, xmm3, xmm4;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
         }

         y.store( j             , xmm1 );
         y.store( j+SIMDSIZE    , xmm2 );
         y.store( j+SIMDSIZE*2UL, xmm3 );
         y.store( j+SIMDSIZE*3UL, xmm4 );
      }

      for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*3UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2, xmm3;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
         }

         y.store( j             , xmm1 );
         y.store( j+SIMDSIZE    , xmm2 );
         y.store( j+SIMDSIZE*2UL, xmm3 );
      }

      for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*2UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j         );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE);
         }

         y.store( j         , xmm1 );
         y.store( j+SIMDSIZE, xmm2 );
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1;

         for( size_t i=ibegin; i<iend; ++i ) {
            xmm1 = xmm1 + set( x[i] ) * A.load(i,j);
         }

         y.store( j, xmm1 );
      }

      for( ; remainder && j<N; ++j )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+1UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         ElementType value = ElementType();

         for( size_t i=ibegin; i<iend; ++i ) {
            value += x[i] * A(i,j);
         }

         y[j] = value;
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense vectors (large matrices)****************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a large transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a transpose dense
   // vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief Vectorized default assignment of a large transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the vectorized default assignment kernel for the transpose dense
   // vector-dense matrix multiplication. This kernel is optimized for large matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectLargeAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT1>::value || !IsPadded<MT1>::value );

      const size_t jblock( 32768UL / sizeof( ElementType ) );
      const size_t iblock( ( N < jblock )?( 8UL ):( 4UL ) );

      BLAZE_INTERNAL_ASSERT( ( jblock % SIMDSIZE ) == 0UL, "Invalid block size detected" );

      reset( y );

      for( size_t jj=0U; jj<N; jj+=jblock ) {
         for( size_t ii=0UL; ii<M; ii+=iblock )
         {
            const size_t iend( min( ii+iblock, M ) );
            const size_t jtmp( min( jj+jblock, N ) );
            const size_t jend( ( IsLower<MT1>::value )
                               ?( min( jtmp, ( IsStrictlyLower<MT1>::value ? iend-1UL : iend ) ) )
                               :( jtmp ) );

            const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
            BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

            size_t j( ( IsUpper<MT1>::value )
                      ?( max( jj, ( IsStrictlyUpper<MT1>::value ? ii+1UL : ii ) & size_t(-SIMDSIZE) ) )
                      :( jj ) );

            for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL )
            {
               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
                  xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
                  xmm5 = xmm5 + x1 * A.load(i,j+SIMDSIZE*4UL);
                  xmm6 = xmm6 + x1 * A.load(i,j+SIMDSIZE*5UL);
                  xmm7 = xmm7 + x1 * A.load(i,j+SIMDSIZE*6UL);
                  xmm8 = xmm8 + x1 * A.load(i,j+SIMDSIZE*7UL);
               }

               y.store( j             , y.load(j             ) + xmm1 );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2 );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3 );
               y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) + xmm4 );
               y.store( j+SIMDSIZE*4UL, y.load(j+SIMDSIZE*4UL) + xmm5 );
               y.store( j+SIMDSIZE*5UL, y.load(j+SIMDSIZE*5UL) + xmm6 );
               y.store( j+SIMDSIZE*6UL, y.load(j+SIMDSIZE*6UL) + xmm7 );
               y.store( j+SIMDSIZE*7UL, y.load(j+SIMDSIZE*7UL) + xmm8 );
            }

            for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
            {
               SIMDType xmm1, xmm2, xmm3, xmm4;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
                  xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
               }

               y.store( j             , y.load(j             ) + xmm1 );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2 );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3 );
               y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) + xmm4 );
            }

            for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
            {
               SIMDType xmm1, xmm2, xmm3;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
               }

               y.store( j             , y.load(j             ) + xmm1 );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2 );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3 );
            }

            for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
            {
               SIMDType xmm1, xmm2;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j         );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE);
               }

               y.store( j         , y.load(j         ) + xmm1 );
               y.store( j+SIMDSIZE, y.load(j+SIMDSIZE) + xmm2 );
            }

            for( ; j<jpos; j+=SIMDSIZE )
            {
               SIMDType xmm1;

               for( size_t i=ii; i<iend; ++i ) {
                  xmm1 = xmm1 + set( x[i] ) * A.load(i,j);
               }

               y.store( j, y.load(j) + xmm1 );
            }

            for( ; remainder && j<jend; ++j )
            {
               ElementType value = ElementType();

               for( size_t i=ii; i<iend; ++i ) {
                  value += x[i] * A(i,j);
               }

               y[j] += value;
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based assignment to dense vectors (default)********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a large transpose
   // dense vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief BLAS-based assignment of a transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function performs the transpose dense vector-dense matrix multiplication based on the
   // according BLAS functionality.
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
   /*!\brief Assignment of a transpose dense vector-dense matrix multiplication to a transpose
   //        sparse vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a transpose dense vector-
   // dense matrix multiplication expression to a sparse vector.
   */
   template< typename VT1 >  // Type of the target sparse vector
   friend inline void assign( SparseVector<VT1,true>& lhs, const TDVecDMatMultExpr& rhs )
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
   /*!\brief Addition assignment of a transpose dense vector-dense matrix multiplication to a
   //        transpose dense vector (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a transpose dense
   // vector-dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void addAssign( DenseVector<VT1,true>& lhs, const TDVecDMatMultExpr& rhs )
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

      TDVecDMatMultExpr::selectAddAssignKernel( ~lhs, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense vectors (kernel selection)*************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for an addition assignment of a transpose dense vector-dense
   //        matrix multiplication to a dense vector (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
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
          ( A.rows() * A.columns() < TDVECDMATMULT_THRESHOLD ) )
         selectSmallAddAssignKernel( y, x, A );
      else
         selectBlasAddAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense vectors************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the default addition assignment kernel for the transpose dense
   // vector-dense matrix multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline void selectDefaultAddAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         if( IsDiagonal<MT1>::value )
         {
            y[i] += x[i] * A(i,i);
         }
         else
         {
            const size_t jbegin( ( IsUpper<MT1>::value )
                                 ?( IsStrictlyUpper<MT1>::value ? i+1UL : i )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT1>::value )
                               ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                               :( N ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jnum( jend - jbegin );
            const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

            for( size_t j=jbegin; j<jpos; j+=2UL ) {
               y[j    ] += x[i] * A(i,j    );
               y[j+1UL] += x[i] * A(i,j+1UL);
            }
            if( jpos < jend ) {
               y[jpos] += x[i] * A(i,jpos);
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense vectors (small matrices)*******************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a small transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a transpose
   // dense vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief Vectorized default addition assignment of a small transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the vectorized default addition assignment kernel for the transpose
   // dense vector-dense matrix multiplication. This kernel is optimized for small matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectSmallAddAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT1>::value || !IsPadded<MT1>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );

      for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*8UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1( y.load(j             ) );
         SIMDType xmm2( y.load(j+SIMDSIZE    ) );
         SIMDType xmm3( y.load(j+SIMDSIZE*2UL) );
         SIMDType xmm4( y.load(j+SIMDSIZE*3UL) );
         SIMDType xmm5( y.load(j+SIMDSIZE*4UL) );
         SIMDType xmm6( y.load(j+SIMDSIZE*5UL) );
         SIMDType xmm7( y.load(j+SIMDSIZE*6UL) );
         SIMDType xmm8( y.load(j+SIMDSIZE*7UL) );

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
            xmm5 = xmm5 + x1 * A.load(i,j+SIMDSIZE*4UL);
            xmm6 = xmm6 + x1 * A.load(i,j+SIMDSIZE*5UL);
            xmm7 = xmm7 + x1 * A.load(i,j+SIMDSIZE*6UL);
            xmm8 = xmm8 + x1 * A.load(i,j+SIMDSIZE*7UL);
         }

         y.store( j             , xmm1 );
         y.store( j+SIMDSIZE    , xmm2 );
         y.store( j+SIMDSIZE*2UL, xmm3 );
         y.store( j+SIMDSIZE*3UL, xmm4 );
         y.store( j+SIMDSIZE*4UL, xmm5 );
         y.store( j+SIMDSIZE*5UL, xmm6 );
         y.store( j+SIMDSIZE*6UL, xmm7 );
         y.store( j+SIMDSIZE*7UL, xmm8 );
      }

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*4UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1( y.load(j             ) );
         SIMDType xmm2( y.load(j+SIMDSIZE    ) );
         SIMDType xmm3( y.load(j+SIMDSIZE*2UL) );
         SIMDType xmm4( y.load(j+SIMDSIZE*3UL) );

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
         }

         y.store( j             , xmm1 );
         y.store( j+SIMDSIZE    , xmm2 );
         y.store( j+SIMDSIZE*2UL, xmm3 );
         y.store( j+SIMDSIZE*3UL, xmm4 );
      }

      for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*3UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1( y.load(j             ) );
         SIMDType xmm2( y.load(j+SIMDSIZE    ) );
         SIMDType xmm3( y.load(j+SIMDSIZE*2UL) );

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
         }

         y.store( j             , xmm1 );
         y.store( j+SIMDSIZE    , xmm2 );
         y.store( j+SIMDSIZE*2UL, xmm3 );
      }

      for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*2UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1( y.load(j         ) );
         SIMDType xmm2( y.load(j+SIMDSIZE) );

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j         );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE);
         }

         y.store( j         , xmm1 );
         y.store( j+SIMDSIZE, xmm2 );
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1( y.load(j) );

         for( size_t i=ibegin; i<iend; ++i ) {
            xmm1 = xmm1 + set( x[i] ) * A.load(i,j);
         }

         y.store( j, xmm1 );
      }

      for( ; remainder && j<N; ++j )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+1UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         ElementType value = ElementType();

         for( size_t i=ibegin; i<iend; ++i ) {
            value += x[i] * A(i,j);
         }

         y[j] += value;
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense vectors (large matrices)*******************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a large transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a transpose
   // dense vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief Vectorized default addition assignment of a large transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the vectorized default addition assignment kernel for the transpose
   // dense vector-dense matrix multiplication. This kernel is optimized for large matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectLargeAddAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT1>::value || !IsPadded<MT1>::value );

      const size_t jblock( 32768UL / sizeof( ElementType ) );
      const size_t iblock( ( N < jblock )?( 8UL ):( 4UL ) );

      BLAZE_INTERNAL_ASSERT( ( jblock % SIMDSIZE ) == 0UL, "Invalid block size detected" );

      for( size_t jj=0U; jj<N; jj+=jblock ) {
         for( size_t ii=0UL; ii<M; ii+=iblock )
         {
            const size_t iend( min( ii+iblock, M ) );
            const size_t jtmp( min( jj+jblock, N ) );
            const size_t jend( ( IsLower<MT1>::value )
                               ?( min( jtmp, ( IsStrictlyLower<MT1>::value ? iend-1UL : iend ) ) )
                               :( jtmp ) );

            const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
            BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

            size_t j( ( IsUpper<MT1>::value )
                      ?( max( jj, ( IsStrictlyUpper<MT1>::value ? ii+1UL : ii ) & size_t(-SIMDSIZE) ) )
                      :( jj ) );

            for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL )
            {
               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
                  xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
                  xmm5 = xmm5 + x1 * A.load(i,j+SIMDSIZE*4UL);
                  xmm6 = xmm6 + x1 * A.load(i,j+SIMDSIZE*5UL);
                  xmm7 = xmm7 + x1 * A.load(i,j+SIMDSIZE*6UL);
                  xmm8 = xmm8 + x1 * A.load(i,j+SIMDSIZE*7UL);
               }

               y.store( j             , y.load(j             ) + xmm1 );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2 );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3 );
               y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) + xmm4 );
               y.store( j+SIMDSIZE*4UL, y.load(j+SIMDSIZE*4UL) + xmm5 );
               y.store( j+SIMDSIZE*5UL, y.load(j+SIMDSIZE*5UL) + xmm6 );
               y.store( j+SIMDSIZE*6UL, y.load(j+SIMDSIZE*6UL) + xmm7 );
               y.store( j+SIMDSIZE*7UL, y.load(j+SIMDSIZE*7UL) + xmm8 );
            }

            for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
            {
               SIMDType xmm1, xmm2, xmm3, xmm4;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
                  xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
               }

               y.store( j             , y.load(j             ) + xmm1 );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2 );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3 );
               y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) + xmm4 );
            }

            for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
            {
               SIMDType xmm1, xmm2, xmm3;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
               }

               y.store( j             , y.load(j             ) + xmm1 );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2 );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3 );
            }

            for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
            {
               SIMDType xmm1, xmm2;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j         );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE);
               }

               y.store( j         , y.load(j         ) + xmm1 );
               y.store( j+SIMDSIZE, y.load(j+SIMDSIZE) + xmm2 );
            }

            for( ; j<jpos; j+=SIMDSIZE )
            {
               SIMDType xmm1;

               for( size_t i=ii; i<iend; ++i ) {
                  xmm1 = xmm1 + set( x[i] ) * A.load(i,j);
               }

               y.store( j, y.load(j) + xmm1 );
            }

            for( ; remainder && j<jend; ++j )
            {
               ElementType value = ElementType();

               for( size_t i=ii; i<iend; ++i ) {
                  value += x[i] * A(i,j);
               }

               y[j] += value;
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense vectors (default)***********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a large
   // transpose dense vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief BLAS-based addition assignment of a transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function performs the transpose dense vector-dense matrix multiplication based on the
   // according BLAS functionality.
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
   /*!\brief Subtraction assignment of a transpose dense vector-dense matrix multiplication to a
   //        transpose dense vector (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a transpose
   // dense vector-dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void subAssign( DenseVector<VT1,true>& lhs, const TDVecDMatMultExpr& rhs )
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

      TDVecDMatMultExpr::selectSubAssignKernel( ~lhs, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to dense vectors (kernel selection)**********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for a subtraction assignment of a transpose dense vector-
   //        dense matrix multiplication to a dense vector (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
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
          ( A.rows() * A.columns() < TDVECDMATMULT_THRESHOLD ) )
         selectSmallSubAssignKernel( y, x, A );
      else
         selectBlasSubAssignKernel( y, x, A );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors*********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the default subtraction assignment kernel for the transpose dense
   // vector-dense matrix multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline void selectDefaultSubAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         if( IsDiagonal<MT1>::value )
         {
            y[i] -= x[i] * A(i,i);
         }
         else
         {
            const size_t jbegin( ( IsUpper<MT1>::value )
                                 ?( IsStrictlyUpper<MT1>::value ? i+1UL : i )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT1>::value )
                               ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                               :( N ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jnum( jend - jbegin );
            const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

            for( size_t j=jbegin; j<jpos; j+=2UL ) {
               y[j    ] -= x[i] * A(i,j    );
               y[j+1UL] -= x[i] * A(i,j+1UL);
            }
            if( jpos < jend ) {
               y[jpos] -= x[i] * A(i,jpos);
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors (small matrices)****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a small transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a
   // transpose dense vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief Vectorized default subtraction assignment of a small transpose dense vector-dense
   //        matrix multiplication (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment kernel for the
   // transpose dense vector-dense matrix multiplication. This kernel is optimized for small
   // matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectSmallSubAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT1>::value || !IsPadded<MT1>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );

      for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*8UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1( y.load(j             ) );
         SIMDType xmm2( y.load(j+SIMDSIZE    ) );
         SIMDType xmm3( y.load(j+SIMDSIZE*2UL) );
         SIMDType xmm4( y.load(j+SIMDSIZE*3UL) );
         SIMDType xmm5( y.load(j+SIMDSIZE*4UL) );
         SIMDType xmm6( y.load(j+SIMDSIZE*5UL) );
         SIMDType xmm7( y.load(j+SIMDSIZE*6UL) );
         SIMDType xmm8( y.load(j+SIMDSIZE*7UL) );

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 - x1 * A.load(i,j             );
            xmm2 = xmm2 - x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 - x1 * A.load(i,j+SIMDSIZE*2UL);
            xmm4 = xmm4 - x1 * A.load(i,j+SIMDSIZE*3UL);
            xmm5 = xmm5 - x1 * A.load(i,j+SIMDSIZE*4UL);
            xmm6 = xmm6 - x1 * A.load(i,j+SIMDSIZE*5UL);
            xmm7 = xmm7 - x1 * A.load(i,j+SIMDSIZE*6UL);
            xmm8 = xmm8 - x1 * A.load(i,j+SIMDSIZE*7UL);
         }

         y.store( j             , xmm1 );
         y.store( j+SIMDSIZE    , xmm2 );
         y.store( j+SIMDSIZE*2UL, xmm3 );
         y.store( j+SIMDSIZE*3UL, xmm4 );
         y.store( j+SIMDSIZE*4UL, xmm5 );
         y.store( j+SIMDSIZE*5UL, xmm6 );
         y.store( j+SIMDSIZE*6UL, xmm7 );
         y.store( j+SIMDSIZE*7UL, xmm8 );
      }

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*4UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1( y.load(j             ) );
         SIMDType xmm2( y.load(j+SIMDSIZE    ) );
         SIMDType xmm3( y.load(j+SIMDSIZE*2UL) );
         SIMDType xmm4( y.load(j+SIMDSIZE*3UL) );

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 - x1 * A.load(i,j             );
            xmm2 = xmm2 - x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 - x1 * A.load(i,j+SIMDSIZE*2UL);
            xmm4 = xmm4 - x1 * A.load(i,j+SIMDSIZE*3UL);
         }

         y.store( j             , xmm1 );
         y.store( j+SIMDSIZE    , xmm2 );
         y.store( j+SIMDSIZE*2UL, xmm3 );
         y.store( j+SIMDSIZE*3UL, xmm4 );
      }

      for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*3UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1( y.load(j             ) );
         SIMDType xmm2( y.load(j+SIMDSIZE    ) );
         SIMDType xmm3( y.load(j+SIMDSIZE*2UL) );

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 - x1 * A.load(i,j             );
            xmm2 = xmm2 - x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 - x1 * A.load(i,j+SIMDSIZE*2UL);
         }

         y.store( j             , xmm1 );
         y.store( j+SIMDSIZE    , xmm2 );
         y.store( j+SIMDSIZE*2UL, xmm3 );
      }

      for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*2UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1( y.load(j         ) );
         SIMDType xmm2( y.load(j+SIMDSIZE) );

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 - x1 * A.load(i,j         );
            xmm2 = xmm2 - x1 * A.load(i,j+SIMDSIZE);
         }

         y.store( j         , xmm1 );
         y.store( j+SIMDSIZE, xmm2 );
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1( y.load(j) );

         for( size_t i=ibegin; i<iend; ++i ) {
            xmm1 = xmm1 - set( x[i] ) * A.load(i,j);
         }

         y.store( j, xmm1 );
      }

      for( ; remainder && j<N; ++j )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+1UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         ElementType value = ElementType();

         for( size_t i=ibegin; i<iend; ++i ) {
            value += x[i] * A(i,j);
         }

         y[j] -= value;
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors (large matrices)****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a large transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a
   // transpose dense vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief Vectorized default subtraction assignment of a large transpose dense vector-dense
   //        matrix multiplication (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment kernel for the
   // transpose dense vector-dense matrix multiplication. This kernel is optimized for large
   // matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1 >  // Type of the right-hand side matrix operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,VT2,MT1> >
      selectLargeSubAssignKernel( VT1& y, const VT2& x, const MT1& A )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<VT1>::value || !IsPadded<MT1>::value );

      const size_t jblock( 32768UL / sizeof( ElementType ) );
      const size_t iblock( ( N < jblock )?( 8UL ):( 4UL ) );

      BLAZE_INTERNAL_ASSERT( ( jblock % SIMDSIZE ) == 0UL, "Invalid block size detected" );

      for( size_t jj=0U; jj<N; jj+=jblock ) {
         for( size_t ii=0UL; ii<M; ii+=iblock )
         {
            const size_t iend( min( ii+iblock, M ) );
            const size_t jtmp( min( jj+jblock, N ) );
            const size_t jend( ( IsLower<MT1>::value )
                               ?( min( jtmp, ( IsStrictlyLower<MT1>::value ? iend-1UL : iend ) ) )
                               :( jtmp ) );

            const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
            BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

            size_t j( ( IsUpper<MT1>::value )
                      ?( max( jj, ( IsStrictlyUpper<MT1>::value ? ii+1UL : ii ) & size_t(-SIMDSIZE) ) )
                      :( jj ) );

            for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL )
            {
               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
                  xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
                  xmm5 = xmm5 + x1 * A.load(i,j+SIMDSIZE*4UL);
                  xmm6 = xmm6 + x1 * A.load(i,j+SIMDSIZE*5UL);
                  xmm7 = xmm7 + x1 * A.load(i,j+SIMDSIZE*6UL);
                  xmm8 = xmm8 + x1 * A.load(i,j+SIMDSIZE*7UL);
               }

               y.store( j             , y.load(j             ) - xmm1 );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) - xmm2 );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) - xmm3 );
               y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) - xmm4 );
               y.store( j+SIMDSIZE*4UL, y.load(j+SIMDSIZE*4UL) - xmm5 );
               y.store( j+SIMDSIZE*5UL, y.load(j+SIMDSIZE*5UL) - xmm6 );
               y.store( j+SIMDSIZE*6UL, y.load(j+SIMDSIZE*6UL) - xmm7 );
               y.store( j+SIMDSIZE*7UL, y.load(j+SIMDSIZE*7UL) - xmm8 );
            }

            for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
            {
               SIMDType xmm1, xmm2, xmm3, xmm4;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
                  xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
               }

               y.store( j             , y.load(j             ) - xmm1 );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) - xmm2 );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) - xmm3 );
               y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) - xmm4 );
            }

            for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
            {
               SIMDType xmm1, xmm2, xmm3;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
               }

               y.store( j             , y.load(j             ) - xmm1 );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) - xmm2 );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) - xmm3 );
            }

            for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
            {
               SIMDType xmm1, xmm2;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j         );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE);
               }

               y.store( j         , y.load(j         ) - xmm1 );
               y.store( j+SIMDSIZE, y.load(j+SIMDSIZE) - xmm2 );
            }

            for( ; j<jpos; j+=SIMDSIZE )
            {
               SIMDType xmm1;

               for( size_t i=ii; i<iend; ++i ) {
                  xmm1 = xmm1 + set( x[i] ) * A.load(i,j);
               }

               y.store( j, y.load(j) - xmm1 );
            }

            for( ; remainder && j<jend; ++j )
            {
               ElementType value = ElementType();

               for( size_t i=ii; i<iend; ++i ) {
                  value += x[i] * A(i,j);
               }

               y[j] -= value;
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense vectors (default)********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a large
   // transpose dense vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief BLAS-based subtraction assignment of a transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function performs the transpose dense vector-dense matrix multiplication based on the
   // according BLAS functionality.
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
   /*!\brief Multiplication assignment of a transpose dense vector-dense matrix multiplication to
   //        a transpose dense vector (\f$ \vec{y}^T*=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a transpose
   // dense vector-dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void multAssign( DenseVector<VT1,true>& lhs, const TDVecDMatMultExpr& rhs )
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
   /*!\brief Division assignment of a transpose dense vector-dense matrix multiplication to a
   //        transpose dense vector (\f$ \vec{y}^T/=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression divisor.
   // \return void
   //
   // This function implements the performance optimized division assignment of a transpose dense
   // vector-dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void divAssign( DenseVector<VT1,true>& lhs, const TDVecDMatMultExpr& rhs )
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
   /*!\brief SMP assignment of a transpose dense vector-dense matrix multiplication to a transpose
   //        dense vector (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a transpose dense
   // vector-dense matrix multiplication expression to a dense vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAssign( DenseVector<VT1,true>& lhs, const TDVecDMatMultExpr& rhs )
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
   /*!\brief SMP assignment of a transpose dense vector-dense matrix multiplication to a transpose
   //        sparse vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a transpose dense
   // vector-dense matrix multiplication expression to a sparse vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target sparse vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAssign( SparseVector<VT1,true>& lhs, const TDVecDMatMultExpr& rhs )
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
   /*!\brief SMP addition assignment of a transpose dense vector-dense matrix multiplication to a
   //        transpose dense vector (\f$ \vec{y}^T+=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a transpose
   // dense vector-dense matrix multiplication expression to a dense vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAddAssign( DenseVector<VT1,true>& lhs, const TDVecDMatMultExpr& rhs )
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
   /*!\brief SMP subtraction assignment of a transpose dense vector-dense matrix multiplication
   //        to a transpose dense vector (\f$ \vec{y}^T-=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a transpose
   // dense vector-dense matrix multiplication expression to a dense vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler in
   // case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpSubAssign( DenseVector<VT1,true>& lhs, const TDVecDMatMultExpr& rhs )
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
   /*!\brief SMP multiplication assignment of a transpose dense vector-dense matrix multiplication
   //        to a transpose dense vector (\f$ \vec{y}^T*=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a
   // transpose dense vector-dense matrix multiplication expression to a dense vector. Due to
   // the explicit application of the SFINAE principle, this function can only be selected by
   // the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpMultAssign( DenseVector<VT1,true>& lhs, const TDVecDMatMultExpr& rhs )
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
   /*!\brief SMP division assignment of a transpose dense vector-dense matrix multiplication
   //        to a transpose dense vector (\f$ \vec{y}^T/=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression divisor.
   // \return void
   //
   // This function implements the performance optimized SMP division assignment of a transpose
   // dense vector-dense matrix multiplication expression to a dense vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler in
   // case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpDivAssign( DenseVector<VT1,true>& lhs, const TDVecDMatMultExpr& rhs )
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
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT );
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
/*!\brief Expression object for scaled transpose dense vector-dense matrix multiplications.
// \ingroup dense_vector_expression
//
// This specialization of the DVecScalarMultExpr class represents the compile time expression
// for scaled multiplications between a non-transpose dense vector and a row-major dense matrix.
*/
template< typename VT    // Type of the left-hand side dense vector
        , typename MT    // Type of the right-hand side dense matrix
        , typename ST >  // Type of the side scalar value
class DVecScalarMultExpr< TDVecDMatMultExpr<VT,MT>, ST, true >
   : public DenseVector< DVecScalarMultExpr< TDVecDMatMultExpr<VT,MT>, ST, true >, true >
   , private VecScalarMultExpr
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef TDVecDMatMultExpr<VT,MT>  VMM;  //!< Type of the dense vector-dense matrix multiplication expression.
   typedef ResultType_<VMM>          RES;  //!< Result type of the dense vector-dense matrix multiplication expression.
   typedef ResultType_<VT>           VRT;  //!< Result type of the left-hand side dense vector expression.
   typedef ResultType_<MT>           MRT;  //!< Result type of the right-hand side dense matrix expression.
   typedef ElementType_<VRT>         VET;  //!< Element type of the left-hand side dense vector epxression.
   typedef ElementType_<MRT>         MET;  //!< Element type of the right-hand side dense matrix expression.
   typedef CompositeType_<VT>        VCT;  //!< Composite type of the left-hand side dense vector expression.
   typedef CompositeType_<MT>        MCT;  //!< Composite type of the right-hand side dense matrix expression.
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
   /*! The UseSMPAssign struct is a helper struct for the selection of the parallel evaluation
       strategy. In case either the vector or the matrix operand requires an intermediate
       evaluation, the nested \a value will be set to 1, otherwise it will be 0. */
   template< typename T1 >
   struct UseSMPAssign {
      enum : bool { value = ( evaluateVector || evaluateMatrix ) };
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
   typedef const TDVecDMatMultExpr<VT,MT>  LeftOperand;

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
               ( A.rows() * A.columns() < TDVECDMATMULT_THRESHOLD ) ) &&
             ( size() > SMP_TDVECDMATMULT_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  vector_;  //!< Left-hand side dense vector of the multiplication expression.
   RightOperand scalar_;  //!< Right-hand side scalar of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense vectors*****************************************************************
   /*!\brief Assignment of a scaled transpose dense vector-dense matrix multiplication to a
   //        transpose dense vector (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a scaled transpose dense
   // vector-dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void assign( DenseVector<VT1,true>& lhs, const DVecScalarMultExpr& rhs )
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
   /*!\brief Selection of the kernel for an assignment of a scaled transpose dense vector-dense
   //        matrix multiplication to a dense vector (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
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
          ( A.rows() * A.columns() < TDVECDMATMULT_THRESHOLD ) )
         selectSmallAssignKernel( y, x, A, scalar );
      else
         selectBlasAssignKernel( y, x, A, scalar );
   }
   //**********************************************************************************************

   //**Default assignment to dense vectors*********************************************************
   /*!\brief Default assignment of a scaled transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment kernel for the scaled transpose dense vector-
   // dense matrix multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename VT2    // Type of the left-hand side vector operand
           , typename MT1    // Type of the right-hand side matrix operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectDefaultAssignKernel( VT1& y, const VT2& x, const MT1& A, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      if( IsStrictlyUpper<MT1>::value ) {
         reset( y[0] );
      }

      if( !IsLower<MT1>::value )
      {
         for( size_t j=( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ); j<N; ++j ) {
            y[j] = x[0UL] * A(0UL,j);
         }
      }

      for( size_t i=( IsLower<MT1>::value && !IsStrictlyLower<MT1>::value ? 0UL : 1UL ); i<M; ++i )
      {
         if( IsDiagonal<MT1>::value )
         {
            y[i] = x[i] * A(i,i) * scalar;
         }
         else
         {
            const size_t jbegin( ( IsUpper<MT1>::value )
                                 ?( IsStrictlyUpper<MT1>::value ? i+1UL : i )
                                 :( 0UL ) );
            const size_t jend( ( IsLower<MT1>::value )
                               ?( IsStrictlyLower<MT1>::value ? i-1UL : i )
                               :( N ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jnum( jend - jbegin );
            const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

            for( size_t j=jbegin; j<jpos; j+=2UL ) {
               y[j    ] += x[i] * A(i,j    );
               y[j+1UL] += x[i] * A(i,j+1UL);
            }
            if( jpos < jend ) {
               y[jpos] += x[i] * A(i,jpos);
            }
            if( IsLower<MT1>::value ) {
               y[jend] = x[i] * A(i,jend);
            }
         }
      }

      if( IsStrictlyLower<MT1>::value ) {
         reset( y[N-1UL] );
      }

      if( !IsDiagonal<MT1>::value )
      {
         const size_t iend( IsStrictlyLower<MT1>::value ? N-1UL : N );
         for( size_t j=( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ); j<iend; ++j ) {
            y[j] *= scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to dense vectors (small matrices)****************************************
   /*!\brief Default assignment of a small scaled transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a scaled transpose
   // dense vector-dense matrix multiplication expression to a dense vector.
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

   //**Default assignment to dense vectors (small matrices)****************************************
   /*!\brief Default assignment of a small scaled transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment kernel for the scaled transpose dense vector-
   // dense matrix multiplication. This kernel is optimized for small matrices.
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

      const bool remainder( !IsPadded<VT1>::value || !IsPadded<MT1>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      const SIMDType factor( set( scalar ) );

      size_t j( 0UL );

      for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*8UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
            xmm5 = xmm5 + x1 * A.load(i,j+SIMDSIZE*4UL);
            xmm6 = xmm6 + x1 * A.load(i,j+SIMDSIZE*5UL);
            xmm7 = xmm7 + x1 * A.load(i,j+SIMDSIZE*6UL);
            xmm8 = xmm8 + x1 * A.load(i,j+SIMDSIZE*7UL);
         }

         y.store( j             , xmm1*factor );
         y.store( j+SIMDSIZE    , xmm2*factor );
         y.store( j+SIMDSIZE*2UL, xmm3*factor );
         y.store( j+SIMDSIZE*3UL, xmm4*factor );
         y.store( j+SIMDSIZE*4UL, xmm5*factor );
         y.store( j+SIMDSIZE*5UL, xmm6*factor );
         y.store( j+SIMDSIZE*6UL, xmm7*factor );
         y.store( j+SIMDSIZE*7UL, xmm8*factor );
      }

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*4UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2, xmm3, xmm4;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
         }

         y.store( j             , xmm1*factor );
         y.store( j+SIMDSIZE    , xmm2*factor );
         y.store( j+SIMDSIZE*2UL, xmm3*factor );
         y.store( j+SIMDSIZE*3UL, xmm4*factor );
      }

      for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*3UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2, xmm3;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
         }

         y.store( j             , xmm1*factor );
         y.store( j+SIMDSIZE    , xmm2*factor );
         y.store( j+SIMDSIZE*2UL, xmm3*factor );
      }

      for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*2UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j         );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE);
         }

         y.store( j         , xmm1*factor );
         y.store( j+SIMDSIZE, xmm2*factor );
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1;

         for( size_t i=ibegin; i<iend; ++i ) {
            xmm1 = xmm1 + set( x[i] ) * A.load(i,j);
         }

         y.store( j, xmm1*factor );
      }

      for( ; remainder && j<N; ++j )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+1UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         ElementType value = ElementType();

         for( size_t i=ibegin; i<iend; ++i ) {
            value += x[i] * A(i,j);
         }

         y[j] = value * scalar;
      }
   }
   //**********************************************************************************************

   //**Default assignment to dense vectors (large matrices)****************************************
   /*!\brief Default assignment of a large scaled transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a scaled transpose
   // dense vector-dense matrix multiplication expression to a dense vector.
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

   //**Default assignment to dense vectors (large matrices)****************************************
   /*!\brief Default assignment of a large scaled transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment kernel for the scaled transpose dense vector-
   // dense matrix multiplication. This kernel is optimized for large matrices.
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

      const bool remainder( !IsPadded<VT1>::value || !IsPadded<MT1>::value );

      const size_t jblock( 32768UL / sizeof( ElementType ) );
      const size_t iblock( ( N < jblock )?( 8UL ):( 4UL ) );

      const SIMDType factor( set( scalar ) );

      BLAZE_INTERNAL_ASSERT( ( jblock % SIMDSIZE ) == 0UL, "Invalid block size detected" );

      reset( y );

      for( size_t jj=0U; jj<N; jj+=jblock ) {
         for( size_t ii=0UL; ii<M; ii+=iblock )
         {
            const size_t iend( min( ii+iblock, M ) );
            const size_t jtmp( min( jj+jblock, N ) );
            const size_t jend( ( IsLower<MT1>::value )
                               ?( min( jtmp, ( IsStrictlyLower<MT1>::value ? iend-1UL : iend ) ) )
                               :( jtmp ) );

            const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
            BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

            size_t j( ( IsUpper<MT1>::value )
                      ?( max( jj, ( IsStrictlyUpper<MT1>::value ? ii+1UL : ii ) & size_t(-SIMDSIZE) ) )
                      :( jj ) );

            for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL )
            {
               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
                  xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
                  xmm5 = xmm5 + x1 * A.load(i,j+SIMDSIZE*4UL);
                  xmm6 = xmm6 + x1 * A.load(i,j+SIMDSIZE*5UL);
                  xmm7 = xmm7 + x1 * A.load(i,j+SIMDSIZE*6UL);
                  xmm8 = xmm8 + x1 * A.load(i,j+SIMDSIZE*7UL);
               }

               y.store( j             , y.load(j             ) + xmm1*factor );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2*factor );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3*factor );
               y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) + xmm4*factor );
               y.store( j+SIMDSIZE*4UL, y.load(j+SIMDSIZE*4UL) + xmm5*factor );
               y.store( j+SIMDSIZE*5UL, y.load(j+SIMDSIZE*5UL) + xmm6*factor );
               y.store( j+SIMDSIZE*6UL, y.load(j+SIMDSIZE*6UL) + xmm7*factor );
               y.store( j+SIMDSIZE*7UL, y.load(j+SIMDSIZE*7UL) + xmm8*factor );
            }

            for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
            {
               SIMDType xmm1, xmm2, xmm3, xmm4;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
                  xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
               }

               y.store( j             , y.load(j             ) + xmm1*factor );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2*factor );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3*factor );
               y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) + xmm4*factor );
            }

            for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
            {
               SIMDType xmm1, xmm2, xmm3;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
               }

               y.store( j             , y.load(j             ) + xmm1*factor );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2*factor );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3*factor );
            }

            for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
            {
               SIMDType xmm1, xmm2;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j         );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE);
               }

               y.store( j         , y.load(j         ) + xmm1*factor );
               y.store( j+SIMDSIZE, y.load(j+SIMDSIZE) + xmm2*factor );
            }

            for( ; j<jpos; j+=SIMDSIZE )
            {
               SIMDType xmm1;

               for( size_t i=ii; i<iend; ++i ) {
                  xmm1 = xmm1 + set( x[i] ) * A.load(i,j);
               }

               y.store( j, y.load(j) + xmm1*factor );
            }

            for( ; remainder && j<jend; ++j )
            {
               ElementType value = ElementType();

               for( size_t i=ii; i<iend; ++i ) {
                  value += x[i] * A(i,j);
               }

               y[j] += value * scalar;
            }
         }
      }
   }
   //**********************************************************************************************

   //**BLAS-based assignment to dense vectors (default)********************************************
   /*!\brief Default assignment of a scaled transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T=\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a large scaled
   // transpose dense vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief BLAS-based assignment of a scaled transpose dense vector-dense matrix multiplication
   //        (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function performs the scaled transpose dense vector-dense matrix multiplication based
   // on the according BLAS functionality.
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
   /*!\brief Assignment of a scaled transpose dense vector-dense matrix multiplication to a
   //        transpose sparse vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a scaled transpose dense
   // vector-dense matrix multiplication expression to a sparse vector.
   */
   template< typename VT1 >  // Type of the target sparse vector
   friend inline void assign( SparseVector<VT1,true>& lhs, const DVecScalarMultExpr& rhs )
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
   /*!\brief Addition assignment of a scaled transpose dense vector-dense matrix multiplication
   //        to a transpose dense vector (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a scaled transpose
   // dense vector-dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void addAssign( DenseVector<VT1,true>& lhs, const DVecScalarMultExpr& rhs )
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
   //        dense matrix multiplication to a dense vector (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
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
   /*!\brief Default addition assignment of a scaled transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default addition assignment kernel for the scaled transpose
   // dense vector-dense matrix multiplication.
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
   /*!\brief Default addition assignment of a small scaled transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a scaled
   // transpose dense vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief Vectorized default addition assignment of a small scaled transpose dense vector-dense
   //        matrix multiplication (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default addition assignment kernel for the scaled
   // transpose dense vector-dense matrix multiplication. This kernel is optimized for small
   // matrices.
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

      const bool remainder( !IsPadded<VT1>::value || !IsPadded<MT1>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      const SIMDType factor( set( scalar ) );

      size_t j( 0UL );

      for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*8UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
            xmm5 = xmm5 + x1 * A.load(i,j+SIMDSIZE*4UL);
            xmm6 = xmm6 + x1 * A.load(i,j+SIMDSIZE*5UL);
            xmm7 = xmm7 + x1 * A.load(i,j+SIMDSIZE*6UL);
            xmm8 = xmm8 + x1 * A.load(i,j+SIMDSIZE*7UL);
         }

         y.store( j             , y.load(j             ) + xmm1*factor );
         y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2*factor );
         y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3*factor );
         y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) + xmm4*factor );
         y.store( j+SIMDSIZE*4UL, y.load(j+SIMDSIZE*4UL) + xmm5*factor );
         y.store( j+SIMDSIZE*5UL, y.load(j+SIMDSIZE*5UL) + xmm6*factor );
         y.store( j+SIMDSIZE*6UL, y.load(j+SIMDSIZE*6UL) + xmm7*factor );
         y.store( j+SIMDSIZE*7UL, y.load(j+SIMDSIZE*7UL) + xmm8*factor );
      }

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*4UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2, xmm3, xmm4;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
         }

         y.store( j             , y.load(j             ) + xmm1*factor );
         y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2*factor );
         y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3*factor );
         y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) + xmm4*factor );
      }

      for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*3UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2, xmm3;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
         }

         y.store( j             , y.load(j             ) + xmm1*factor );
         y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2*factor );
         y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3*factor );
      }

      for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*2UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j         );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE);
         }

         y.store( j         , y.load(j         ) + xmm1*factor );
         y.store( j+SIMDSIZE, y.load(j+SIMDSIZE) + xmm2*factor );
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1;

         for( size_t i=ibegin; i<iend; ++i ) {
            xmm1 = xmm1 + set( x[i] ) * A.load(i,j);
         }

         y.store( j, y.load(j) + xmm1*factor );
      }

      for( ; remainder && j<N; ++j )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+1UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         ElementType value = ElementType();

         for( size_t i=ibegin; i<iend; ++i ) {
            value += x[i] * A(i,j);
         }

         y[j] += value * scalar;
      }
   }
   //**********************************************************************************************

   //**Default addition assignment to dense vectors (large matrices)*******************************
   /*!\brief Default addition assignment of a large scaled transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a scaled
   // transpose dense vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief Vectorized default addition assignment of a large scaled transpose dense vector-dense
   //        matrix multiplication (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default addition assignment kernel for the scaled
   // transpose dense vector-dense matrix multiplication. This kernel is optimized for large
   // matrices.
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

      const bool remainder( !IsPadded<VT1>::value || !IsPadded<MT1>::value );

      const size_t jblock( 32768UL / sizeof( ElementType ) );
      const size_t iblock( ( N < jblock )?( 8UL ):( 4UL ) );

      const SIMDType factor( set( scalar ) );

      BLAZE_INTERNAL_ASSERT( ( jblock % SIMDSIZE ) == 0UL, "Invalid block size detected" );

      for( size_t jj=0U; jj<N; jj+=jblock ) {
         for( size_t ii=0UL; ii<M; ii+=iblock )
         {
            const size_t iend( min( ii+iblock, M ) );
            const size_t jtmp( min( jj+jblock, N ) );
            const size_t jend( ( IsLower<MT1>::value )
                               ?( min( jtmp, ( IsStrictlyLower<MT1>::value ? iend-1UL : iend ) ) )
                               :( jtmp ) );

            const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
            BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

            size_t j( ( IsUpper<MT1>::value )
                      ?( max( jj, ( IsStrictlyUpper<MT1>::value ? ii+1UL : ii ) & size_t(-SIMDSIZE) ) )
                      :( jj ) );

            for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL )
            {
               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
                  xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
                  xmm5 = xmm5 + x1 * A.load(i,j+SIMDSIZE*4UL);
                  xmm6 = xmm6 + x1 * A.load(i,j+SIMDSIZE*5UL);
                  xmm7 = xmm7 + x1 * A.load(i,j+SIMDSIZE*6UL);
                  xmm8 = xmm8 + x1 * A.load(i,j+SIMDSIZE*7UL);
               }

               y.store( j             , y.load(j             ) + xmm1*factor );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2*factor );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3*factor );
               y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) + xmm4*factor );
               y.store( j+SIMDSIZE*4UL, y.load(j+SIMDSIZE*4UL) + xmm5*factor );
               y.store( j+SIMDSIZE*5UL, y.load(j+SIMDSIZE*5UL) + xmm6*factor );
               y.store( j+SIMDSIZE*6UL, y.load(j+SIMDSIZE*6UL) + xmm7*factor );
               y.store( j+SIMDSIZE*7UL, y.load(j+SIMDSIZE*7UL) + xmm8*factor );
            }

            for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
            {
               SIMDType xmm1, xmm2, xmm3, xmm4;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
                  xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
               }

               y.store( j             , y.load(j             ) + xmm1*factor );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2*factor );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3*factor );
               y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) + xmm4*factor );
            }

            for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
            {
               SIMDType xmm1, xmm2, xmm3;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
               }

               y.store( j             , y.load(j             ) + xmm1*factor );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) + xmm2*factor );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) + xmm3*factor );
            }

            for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
            {
               SIMDType xmm1, xmm2;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j         );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE);
               }

               y.store( j         , y.load(j         ) + xmm1*factor );
               y.store( j+SIMDSIZE, y.load(j+SIMDSIZE) + xmm2*factor );
            }

            for( ; j<jpos; j+=SIMDSIZE )
            {
               SIMDType xmm1;

               for( size_t i=ii; i<iend; ++i ) {
                  xmm1 = xmm1 + set( x[i] ) * A.load(i,j);
               }

               y.store( j, y.load(j) + xmm1*factor );
            }

            for( ; remainder && j<jend; ++j )
            {
               ElementType value = ElementType();

               for( size_t i=ii; i<iend; ++i ) {
                  value += x[i] * A(i,j);
               }

               y[j] += value * scalar;
            }
         }
      }
   }
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense vectors (default)***********************************
   /*!\brief Default addition assignment of a scaled transpose dense vector-dense matrix
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
   // scaled transpose dense vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief BLAS-based addition assignment of a scaled transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor
   // \return void
   //
   // This function performs the scaled transpose dense vector-dense matrix multiplication based
   // on the according BLAS functionality.
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
   /*!\brief Subtraction assignment of a scaled transpose dense vector-dense matrix multiplication
   //        to a transpose dense vector (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a scaled
   // transpose dense vector-dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void subAssign( DenseVector<VT1,true>& lhs, const DVecScalarMultExpr& rhs )
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
   //        dense matrix multiplication to a dense vector (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
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
   /*!\brief Default subtraction assignment of a scaled transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default subtraction assignment kernel for the scaled transpose
   // dense vector-dense matrix multiplication.
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
   /*!\brief Default subtraction assignment of a small scaled transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a
   // scaled transpose dense vector-dense matrix multiplication expression to a dense vector.
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
   //        dense matrix multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment kernel for the
   // scaled transpose dense vector-dense matrix multiplication. This kernel is optimized for
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

      const bool remainder( !IsPadded<VT1>::value || !IsPadded<MT1>::value );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      const SIMDType factor( set( scalar ) );

      size_t j( 0UL );

      for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*8UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
            xmm5 = xmm5 + x1 * A.load(i,j+SIMDSIZE*4UL);
            xmm6 = xmm6 + x1 * A.load(i,j+SIMDSIZE*5UL);
            xmm7 = xmm7 + x1 * A.load(i,j+SIMDSIZE*6UL);
            xmm8 = xmm8 + x1 * A.load(i,j+SIMDSIZE*7UL);
         }

         y.store( j             , y.load(j             ) - xmm1*factor );
         y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) - xmm2*factor );
         y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) - xmm3*factor );
         y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) - xmm4*factor );
         y.store( j+SIMDSIZE*4UL, y.load(j+SIMDSIZE*4UL) - xmm5*factor );
         y.store( j+SIMDSIZE*5UL, y.load(j+SIMDSIZE*5UL) - xmm6*factor );
         y.store( j+SIMDSIZE*6UL, y.load(j+SIMDSIZE*6UL) - xmm7*factor );
         y.store( j+SIMDSIZE*7UL, y.load(j+SIMDSIZE*7UL) - xmm8*factor );
      }

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*4UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2, xmm3, xmm4;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
            xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
         }

         y.store( j             , y.load(j             ) - xmm1*factor );
         y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) - xmm2*factor );
         y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) - xmm3*factor );
         y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) - xmm4*factor );
      }

      for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*3UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2, xmm3;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j             );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
            xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
         }

         y.store( j             , y.load(j             ) - xmm1*factor );
         y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) - xmm2*factor );
         y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) - xmm3*factor );
      }

      for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE*2UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1, xmm2;

         for( size_t i=ibegin; i<iend; ++i ) {
            const SIMDType x1( set( x[i] ) );
            xmm1 = xmm1 + x1 * A.load(i,j         );
            xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE);
         }

         y.store( j         , y.load(j         ) - xmm1*factor );
         y.store( j+SIMDSIZE, y.load(j+SIMDSIZE) - xmm2*factor );
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+SIMDSIZE, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         SIMDType xmm1;

         for( size_t i=ibegin; i<iend; ++i ) {
            xmm1 = xmm1 + set( x[i] ) * A.load(i,j);
         }

         y.store( j, y.load(j) - xmm1*factor );
      }

      for( ; remainder && j<N; ++j )
      {
         const size_t ibegin( ( IsLower<MT1>::value )
                              ?( IsStrictlyLower<MT1>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend( ( IsUpper<MT1>::value )
                            ?( min( j+1UL, M ) - ( IsStrictlyUpper<MT1>::value ? 1UL : 0UL ) )
                            :( M ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         ElementType value = ElementType();

         for( size_t i=ibegin; i<iend; ++i ) {
            value += x[i] * A(i,j);
         }

         y[j] -= value * scalar;
      }
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors (large matrices)****************************
   /*!\brief Default subtraction assignment of a large scaled transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a
   // scaled transpose dense vector-dense matrix multiplication expression to a dense vector.
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
   //        dense matrix multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment kernel for the
   // scaled transpose dense vector-dense matrix multiplication. This kernel is optimized for
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

      const bool remainder( !IsPadded<VT1>::value || !IsPadded<MT1>::value );

      const size_t jblock( 32768UL / sizeof( ElementType ) );
      const size_t iblock( ( N < jblock )?( 8UL ):( 4UL ) );

      const SIMDType factor( set( scalar ) );

      BLAZE_INTERNAL_ASSERT( ( jblock % SIMDSIZE ) == 0UL, "Invalid block size detected" );

      for( size_t jj=0U; jj<N; jj+=jblock ) {
         for( size_t ii=0UL; ii<M; ii+=iblock )
         {
            const size_t iend( min( ii+iblock, M ) );
            const size_t jtmp( min( jj+jblock, N ) );
            const size_t jend( ( IsLower<MT1>::value )
                               ?( min( jtmp, ( IsStrictlyLower<MT1>::value ? iend-1UL : iend ) ) )
                               :( jtmp ) );

            const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
            BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

            size_t j( ( IsUpper<MT1>::value )
                      ?( max( jj, ( IsStrictlyUpper<MT1>::value ? ii+1UL : ii ) & size_t(-SIMDSIZE) ) )
                      :( jj ) );

            for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL )
            {
               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
                  xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
                  xmm5 = xmm5 + x1 * A.load(i,j+SIMDSIZE*4UL);
                  xmm6 = xmm6 + x1 * A.load(i,j+SIMDSIZE*5UL);
                  xmm7 = xmm7 + x1 * A.load(i,j+SIMDSIZE*6UL);
                  xmm8 = xmm8 + x1 * A.load(i,j+SIMDSIZE*7UL);
               }

               y.store( j             , y.load(j             ) - xmm1*factor );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) - xmm2*factor );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) - xmm3*factor );
               y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) - xmm4*factor );
               y.store( j+SIMDSIZE*4UL, y.load(j+SIMDSIZE*4UL) - xmm5*factor );
               y.store( j+SIMDSIZE*5UL, y.load(j+SIMDSIZE*5UL) - xmm6*factor );
               y.store( j+SIMDSIZE*6UL, y.load(j+SIMDSIZE*6UL) - xmm7*factor );
               y.store( j+SIMDSIZE*7UL, y.load(j+SIMDSIZE*7UL) - xmm8*factor );
            }

            for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
            {
               SIMDType xmm1, xmm2, xmm3, xmm4;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
                  xmm4 = xmm4 + x1 * A.load(i,j+SIMDSIZE*3UL);
               }

               y.store( j             , y.load(j             ) - xmm1*factor );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) - xmm2*factor );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) - xmm3*factor );
               y.store( j+SIMDSIZE*3UL, y.load(j+SIMDSIZE*3UL) - xmm4*factor );
            }

            for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
            {
               SIMDType xmm1, xmm2, xmm3;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j             );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE    );
                  xmm3 = xmm3 + x1 * A.load(i,j+SIMDSIZE*2UL);
               }

               y.store( j             , y.load(j             ) - xmm1*factor );
               y.store( j+SIMDSIZE    , y.load(j+SIMDSIZE    ) - xmm2*factor );
               y.store( j+SIMDSIZE*2UL, y.load(j+SIMDSIZE*2UL) - xmm3*factor );
            }

            for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
            {
               SIMDType xmm1, xmm2;

               for( size_t i=ii; i<iend; ++i ) {
                  const SIMDType x1( set( x[i] ) );
                  xmm1 = xmm1 + x1 * A.load(i,j         );
                  xmm2 = xmm2 + x1 * A.load(i,j+SIMDSIZE);
               }

               y.store( j         , y.load(j         ) - xmm1*factor );
               y.store( j+SIMDSIZE, y.load(j+SIMDSIZE) - xmm2*factor );
            }

            for( ; j<jpos; j+=SIMDSIZE )
            {
               SIMDType xmm1;

               for( size_t i=ii; i<iend; ++i ) {
                  xmm1 = xmm1 + set( x[i] ) * A.load(i,j);
               }

               y.store( j, y.load(j) - xmm1*factor );
            }

            for( ; remainder && j<jend; ++j )
            {
               ElementType value = ElementType();

               for( size_t i=ii; i<iend; ++i ) {
                  value += x[i] * A(i,j);
               }

               y[j] -= value * scalar;
            }
         }
      }
   }
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense vectors (default)********************************
   /*!\brief Default subtraction assignment of a scaled transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a large
   // scaled transpose dense vector-dense matrix multiplication expression to a dense vector.
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
   /*!\brief BLAS-based subtraction assignment of a scaled transpose dense vector-dense matrix
   //        multiplication (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param x The left-hand side dense vector operand.
   // \param A The right-hand side dense matrix operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function performs the scaled transpose dense vector-dense matrix multiplication based
   // on the according BLAS functionality.
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
   /*!\brief Multiplication assignment of a scaled transpose dense vector-dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}*=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a scaled
   // transpose dense vector-dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void multAssign( DenseVector<VT1,true>& lhs, const DVecScalarMultExpr& rhs )
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
   /*!\brief Division assignment of a scaled transpose dense vector-dense matrix multiplication to
   //        a transpose dense vector (\f$ \vec{y}/=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression divisor.
   // \return void
   //
   // This function implements the performance optimized division assignment of a scaled transpose
   // dense vector-dense matrix multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void divAssign( DenseVector<VT1,true>& lhs, const DVecScalarMultExpr& rhs )
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
   /*!\brief SMP assignment of a scaled transpose dense vector-dense matrix multiplication to a
   //        transpose dense vector (\f$ \vec{y}^T=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a scaled transpose
   // dense vector-dense matrix multiplication expression to a dense vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler in
   // case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAssign( DenseVector<VT1,true>& lhs, const DVecScalarMultExpr& rhs )
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
   /*!\brief SMP assignment of a scaled transpose dense vector-dense matrix multiplication to a
   //        transpose sparse vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a scaled transpose
   // dense vector-dense matrix multiplication expression to a sparse vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler in
   // case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target sparse vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAssign( SparseVector<VT1,true>& lhs, const DVecScalarMultExpr& rhs )
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
   /*!\brief SMP addition assignment of a scaled transpose dense vector-dense matrix multiplication
   //        to a transpose dense vector (\f$ \vec{y}^T+=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a scaled
   // transpose dense vector-dense matrix multiplication expression to a dense vector. Due to
   // the explicit application of the SFINAE principle, this function can only be selected by
   // the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAddAssign( DenseVector<VT1,true>& lhs, const DVecScalarMultExpr& rhs )
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
   /*!\brief SMP subtraction assignment of a scaled transpose dense vector-dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}^T-=s*\vec{x}^T*A \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a scaled
   // transpose dense vector-dense matrix multiplication expression to a dense vector. Due to
   // the explicit application of the SFINAE principle, this function can only be selected by
   // the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpSubAssign( DenseVector<VT1,true>& lhs, const DVecScalarMultExpr& rhs )
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
   /*!\brief SMP multiplication assignment of a scaled transpose dense vector-dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}*=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a
   // scaled transpose dense vector-dense matrix multiplication expression to a dense vector.
   // Due to the explicit application of the SFINAE principle, this function can only be
   // selected by the compiler in case the expression specific parallel evaluation strategy
   // is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpMultAssign( DenseVector<VT1,true>& lhs, const DVecScalarMultExpr& rhs )
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
   /*!\brief SMP dvision assignment of a scaled transpose dense vector-dense matrix
   //        multiplication to a transpose dense vector (\f$ \vec{y}/=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression divisor.
   // \return void
   //
   // This function implements the performance optimized SMP division assignment of a scaled
   // transpose dense vector-dense matrix multiplication expression to a dense vector. Due to
   // the explicit application of the SFINAE principle, this function can only be selected by
   // the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpDivAssign( DenseVector<VT1,true>& lhs, const DVecScalarMultExpr& rhs )
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
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT );
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
//        row-major dense matrix (\f$ \vec{y}^T=\vec{x}^T*A \f$).
// \ingroup dense_vector
//
// \param vec The left-hand side transpose dense vector for the multiplication.
// \param mat The right-hand side row-major dense matrix for the multiplication.
// \return The resulting transpose vector.
// \exception std::invalid_argument Vector and matrix sizes do not match.
//
// This operator represents the multiplication between a transpose dense vector and a row-major
// dense matrix:

   \code
   using blaze::rowVector;
   using blaze::rowMajor;

   blaze::DynamicVector<double,rowVector> x, y;
   blaze::DynamicMatrix<double,rowMajor> A;
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
inline const DisableIf_< IsMatMatMultExpr<T2>, TDVecDMatMultExpr<T1,T2> >
   operator*( const DenseVector<T1,true>& vec, const DenseMatrix<T2,false>& mat )
{
   BLAZE_FUNCTION_TRACE;

   if( (~vec).size() != (~mat).rows() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector and matrix sizes do not match" );
   }

   return TDVecDMatMultExpr<T1,T2>( ~vec, ~mat );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Multiplication operator for the multiplication of a transpose dense vector and a
//        dense matrix-matrix multiplication expression (\f$ \vec{y}^T=\vec{x}^T*(A*B) \f$).
// \ingroup dense_vector
//
// \param vec The left-hand side dense vector for the multiplication.
// \param mat The right-hand side dense matrix-matrix multiplication.
// \return The resulting vector.
//
// This operator implements a performance optimized treatment of the multiplication of a dense
// vector and a dense matrix-matrix multiplication expression. It restructures the expression
// \f$ \vec{y}^T=\vec{x}^T*(A*B) \f$ to the expression \f$ \vec{y}^T=(\vec{x}^T*A)*B \f$.
*/
template< typename T1  // Type of the left-hand side dense vector
        , typename T2  // Type of the right-hand side dense matrix
        , bool SO >    // Storage order of the right-hand side dense matrix
inline const EnableIf_< IsMatMatMultExpr<T2>, MultExprTrait_<T1,T2> >
   operator*( const DenseVector<T1,true>& vec, const DenseMatrix<T2,SO>& mat )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( T1 );

   return ( vec * (~mat).leftOperand() ) * (~mat).rightOperand();
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
struct Size< TDVecDMatMultExpr<VT,MT> > : public Columns<MT>
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
struct IsAligned< TDVecDMatMultExpr<VT,MT> >
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
struct SubvectorExprTrait< TDVecDMatMultExpr<VT,MT>, AF >
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
