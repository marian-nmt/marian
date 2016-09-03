//=================================================================================================
/*!
//  \file blaze/math/expressions/DMatDVecMultExpr.h
//  \brief Header file for the dense matrix/dense vector multiplication expression
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

#ifndef _BLAZE_MATH_EXPRESSIONS_DMATDVECMULTEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_DMATDVECMULTEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/blas/gemv.h>
#include <blaze/math/blas/trmv.h>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/ColumnVector.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/constraints/MatVecMultExpr.h>
#include <blaze/math/constraints/RowMajorMatrix.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/MatVecMultExpr.h>
#include <blaze/math/expressions/VecScalarMultExpr.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/traits/MultExprTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/SubmatrixExprTrait.h>
#include <blaze/math/traits/SubvectorExprTrait.h>
#include <blaze/math/typetraits/AreSIMDCombinable.h>
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
#include <blaze/math/typetraits/Rows.h>
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
//  CLASS DMATDVECMULTEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for dense matrix-dense vector multiplications.
// \ingroup dense_vector_expression
//
// The DMatDVecMultExpr class represents the compile time expression for multiplications
// between row-major dense matrices and dense vectors.
*/
template< typename MT    // Type of the left-hand side dense matrix
        , typename VT >  // Type of the right-hand side dense vector
class DMatDVecMultExpr : public DenseVector< DMatDVecMultExpr<MT,VT>, false >
                       , private MatVecMultExpr
                       , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef ResultType_<MT>     MRT;  //!< Result type of the left-hand side dense matrix expression.
   typedef ResultType_<VT>     VRT;  //!< Result type of the right-hand side dense vector expression.
   typedef ElementType_<MRT>   MET;  //!< Element type of the left-hand side dense matrix expression.
   typedef ElementType_<VRT>   VET;  //!< Element type of the right-hand side dense vector expression.
   typedef CompositeType_<MT>  MCT;  //!< Composite type of the left-hand side dense matrix expression.
   typedef CompositeType_<VT>  VCT;  //!< Composite type of the right-hand side dense vector expression.
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the left-hand side dense matrix expression.
   enum : bool { evaluateMatrix = ( IsComputation<MT>::value && IsSame<MET,VET>::value &&
                                    IsBLASCompatible<MET>::value ) || RequiresEvaluation<MT>::value };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the right-hand side dense vector expression.
   enum : bool { evaluateVector = IsComputation<VT>::value || RequiresEvaluation<VT>::value };
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! The UseSMPAssign struct is a helper struct for the selection of the parallel evaluation
       strategy. In case either the matrix or the vector operand requires an intermediate
       evaluation, the nested \a value will be set to 1, otherwise it will be 0. */
   template< typename T1 >
   struct UseSMPAssign {
      enum : bool { value = ( evaluateMatrix || evaluateVector ) };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! In case the matrix type and the two involved vector types are suited for a BLAS kernel,
       the nested \a value will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   struct UseBlasKernel {
      enum : bool { value = BLAZE_BLAS_MODE &&
                            HasMutableDataAccess<T1>::value &&
                            HasConstDataAccess<T2>::value &&
                            HasConstDataAccess<T3>::value &&
                            !IsDiagonal<T2>::value &&
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
   /*! In case the matrix type and the two involved vector types are suited for a vectorized
       computation of the matrix/vector multiplication, the nested \a value will be set to 1,
       otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   struct UseVectorizedDefaultKernel {
      enum : bool { value = useOptimizedKernels &&
                            !IsDiagonal<T2>::value &&
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
   typedef DMatDVecMultExpr<MT,VT>     This;           //!< Type of this DMatDVecMultExpr instance.
   typedef MultTrait_<MRT,VRT>         ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>  TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<ResultType>    ElementType;    //!< Resulting element type.
   typedef SIMDTrait_<ElementType>     SIMDType;       //!< Resulting SIMD element type.
   typedef const ElementType           ReturnType;     //!< Return type for expression template evaluations.
   typedef const ResultType            CompositeType;  //!< Data type for composite expression templates.

   //! Composite type of the left-hand side dense matrix expression.
   typedef If_< IsExpression<MT>, const MT, const MT& >  LeftOperand;

   //! Composite type of the right-hand side dense vector expression.
   typedef If_< IsExpression<VT>, const VT, const VT& >  RightOperand;

   //! Type for the assignment of the left-hand side dense matrix operand.
   typedef IfTrue_< evaluateMatrix, const MRT, MCT >  LT;

   //! Type for the assignment of the right-hand side dense vector operand.
   typedef IfTrue_< evaluateVector, const VRT, VCT >  RT;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = !IsDiagonal<MT>::value &&
                               MT::simdEnabled && VT::simdEnabled &&
                               HasSIMDAdd<MET,VET>::value &&
                               HasSIMDMult<MET,VET>::value };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = !evaluateMatrix && MT::smpAssignable &&
                                 !evaluateVector && VT::smpAssignable };
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   enum : size_t { SIMDSIZE = SIMDTrait<ElementType>::size };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DMatDVecMultExpr class.
   //
   // \param mat The left-hand side matrix operand of the multiplication expression.
   // \param vec The right-hand side vector operand of the multiplication expression.
   */
   explicit inline DMatDVecMultExpr( const MT& mat, const VT& vec ) noexcept
      : mat_( mat )  // Left-hand side dense matrix of the multiplication expression
      , vec_( vec )  // Right-hand side dense vector of the multiplication expression
   {
      BLAZE_INTERNAL_ASSERT( mat_.columns() == vec_.size(), "Invalid matrix and vector sizes" );
   }
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the vector elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator[]( size_t index ) const {
      BLAZE_INTERNAL_ASSERT( index < mat_.rows(), "Invalid vector access index" );

      if( IsDiagonal<MT>::value )
      {
         return mat_(index,index) * vec_[index];
      }
      else if( IsLower<MT>::value && ( index + 8UL < mat_.rows() ) )
      {
         const size_t n( IsStrictlyLower<MT>::value ? index : index+1UL );
         return subvector( row( mat_, index ), 0UL, n ) * subvector( vec_, 0UL, n );
      }
      else if( IsUpper<MT>::value && ( index > 8UL ) )
      {
         const size_t begin( IsStrictlyUpper<MT>::value ? index+1UL : index );
         const size_t n    ( mat_.columns() - begin );
         return subvector( row( mat_, index ), begin, n ) * subvector( vec_, begin, n );
      }
      else
      {
         return row( mat_, index ) * vec_;
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
      if( index >= mat_.rows() ) {
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
      return mat_.rows();
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side dense matrix operand.
   //
   // \return The left-hand side dense matrix operand.
   */
   inline LeftOperand leftOperand() const  noexcept{
      return mat_;
   }
   //**********************************************************************************************

   //**Right operand access************************************************************************
   /*!\brief Returns the right-hand side dense vector operand.
   //
   // \return The right-hand side dense vector operand.
   */
   inline RightOperand rightOperand() const noexcept {
      return vec_;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case an aliasing effect is possible, \a false if not.
   */
   template< typename T >
   inline bool canAlias( const T* alias ) const noexcept {
      return ( mat_.isAliased( alias ) || vec_.isAliased( alias ) );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression is aliased with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the given alias is contained in this expression, \a false if not.
   */
   template< typename T >
   inline bool isAliased( const T* alias ) const noexcept {
      return ( mat_.isAliased( alias ) || vec_.isAliased( alias ) );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const noexcept {
      return mat_.isAligned() && vec_.isAligned();
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
               ( mat_.rows() * mat_.columns() < DMATDVECMULT_THRESHOLD ) ) &&
             ( size() > SMP_DMATDVECMULT_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  mat_;  //!< Left-hand side dense matrix of the multiplication expression.
   RightOperand vec_;  //!< Right-hand side dense vector of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense vectors*****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense matrix-dense vector multiplication to a dense vector
   //        (\f$ \vec{y}=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense matrix-dense
   // vector multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void assign( DenseVector<VT1,false>& lhs, const DMatDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      if( rhs.mat_.rows() == 0UL ) {
         return;
      }
      else if( rhs.mat_.columns() == 0UL ) {
         reset( ~lhs );
         return;
      }

      LT A( serial( rhs.mat_ ) );  // Evaluation of the left-hand side dense matrix operand
      RT x( serial( rhs.vec_ ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.mat_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.mat_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( x.size()    == rhs.vec_.size()   , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).size()     , "Invalid vector size"       );

      DMatDVecMultExpr::selectAssignKernel( ~lhs, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to dense vectors (kernel selection)**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for an assignment of a dense matrix-dense vector multiplication
   //        to a dense vector (\f$ \vec{y}=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline void selectAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      if( ( IsDiagonal<MT1>::value ) ||
          ( IsComputation<MT>::value && !evaluateMatrix ) ||
          ( A.rows() * A.columns() < DMATDVECMULT_THRESHOLD ) )
         selectSmallAssignKernel( y, A, x );
      else
         selectBlasAssignKernel( y, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense vectors*********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a dense matrix-dense vector multiplication
   //        (\f$ \vec{y}=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the default assignment kernel for the dense matrix-dense vector
   // multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline void selectDefaultAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      y.assign( A * x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense vectors (small matrices)****************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a small dense matrix-dense vector multiplication
   //        (\f$ \vec{y}=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a dense matrix-
   // dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2> >
      selectSmallAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      selectDefaultAssignKernel( y, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default assignment to dense vectors (small matrices)*****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default assignment of a small dense matrix-dense vector multiplication
   //        (\f$ \vec{y}=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the vectorized default assignment kernel for the dense matrix-
   // dense vector multiplication. This kernel is optimized for small matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2> >
      selectSmallAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<MT1>::value || !IsPadded<VT2>::value );

      size_t i( 0UL );

      for( ; (i+8UL) <= M; i+=8UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+7UL : i+8UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
            xmm4 = xmm4 + A.load(i+3UL,j) * x1;
            xmm5 = xmm5 + A.load(i+4UL,j) * x1;
            xmm6 = xmm6 + A.load(i+5UL,j) * x1;
            xmm7 = xmm7 + A.load(i+6UL,j) * x1;
            xmm8 = xmm8 + A.load(i+7UL,j) * x1;
         }

         y[i    ] = sum( xmm1 );
         y[i+1UL] = sum( xmm2 );
         y[i+2UL] = sum( xmm3 );
         y[i+3UL] = sum( xmm4 );
         y[i+4UL] = sum( xmm5 );
         y[i+5UL] = sum( xmm6 );
         y[i+6UL] = sum( xmm7 );
         y[i+7UL] = sum( xmm8 );

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
            y[i+2UL] += A(i+2UL,j) * x[j];
            y[i+3UL] += A(i+3UL,j) * x[j];
            y[i+4UL] += A(i+4UL,j) * x[j];
            y[i+5UL] += A(i+5UL,j) * x[j];
            y[i+6UL] += A(i+6UL,j) * x[j];
            y[i+7UL] += A(i+7UL,j) * x[j];
         }
      }

      for( ; (i+4UL) <= M; i+=4UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+3UL : i+4UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
            xmm4 = xmm4 + A.load(i+3UL,j) * x1;
         }

         y[i    ] = sum( xmm1 );
         y[i+1UL] = sum( xmm2 );
         y[i+2UL] = sum( xmm3 );
         y[i+3UL] = sum( xmm4 );

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
            y[i+2UL] += A(i+2UL,j) * x[j];
            y[i+3UL] += A(i+3UL,j) * x[j];
         }
      }

      for( ; (i+3UL) <= M; i+=3UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+2UL : i+3UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
         }

         y[i    ] = sum( xmm1 );
         y[i+1UL] = sum( xmm2 );
         y[i+2UL] = sum( xmm3 );

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
            y[i+2UL] += A(i+2UL,j) * x[j];
         }
      }

      for( ; (i+2UL) <= M; i+=2UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+1UL : i+2UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
         }

         y[i    ] = sum( xmm1 );
         y[i+1UL] = sum( xmm2 );

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
         }
      }

      if( i < M )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            xmm1 = xmm1 + A.load(i,j) * x.load(j);
         }

         y[i] = sum( xmm1 );

         for( ; remainder && j<jend; ++j ) {
            y[i] += A(i,j) * x[j];
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense vectors (large matrices)****************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a large dense matrix-dense vector multiplication
   //        (\f$ \vec{y}=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a dense matrix-
   // dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2> >
      selectLargeAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      selectDefaultAssignKernel( y, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default assignment to dense vectors (large matrices)*****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default assignment of a large dense matrix-dense vector multiplication
   //        (\f$ \vec{y}=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the vectorized default assignment kernel for the dense matrix-
   // dense vector multiplication. This kernel is optimized for large matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2> >
      selectLargeAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<MT1>::value || !IsPadded<VT2>::value );

      reset( y );

      size_t i( 0UL );

      for( ; (i+8UL) <= M; i+=8UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+7UL : i+8UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 + A.load(i+2UL,j2) * x3 + A.load(i+2UL,j3) * x4 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 + A.load(i+3UL,j2) * x3 + A.load(i+3UL,j3) * x4 );
            y[i+4UL] += sum( A.load(i+4UL,j) * x1 + A.load(i+4UL,j1) * x2 + A.load(i+4UL,j2) * x3 + A.load(i+4UL,j3) * x4 );
            y[i+5UL] += sum( A.load(i+5UL,j) * x1 + A.load(i+5UL,j1) * x2 + A.load(i+5UL,j2) * x3 + A.load(i+5UL,j3) * x4 );
            y[i+6UL] += sum( A.load(i+6UL,j) * x1 + A.load(i+6UL,j1) * x2 + A.load(i+6UL,j2) * x3 + A.load(i+6UL,j3) * x4 );
            y[i+7UL] += sum( A.load(i+7UL,j) * x1 + A.load(i+7UL,j1) * x2 + A.load(i+7UL,j2) * x3 + A.load(i+7UL,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 );
            y[i+4UL] += sum( A.load(i+4UL,j) * x1 + A.load(i+4UL,j1) * x2 );
            y[i+5UL] += sum( A.load(i+5UL,j) * x1 + A.load(i+5UL,j1) * x2 );
            y[i+6UL] += sum( A.load(i+6UL,j) * x1 + A.load(i+6UL,j1) * x2 );
            y[i+7UL] += sum( A.load(i+7UL,j) * x1 + A.load(i+7UL,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] += sum( A.load(i    ,j) * x1 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 );
            y[i+4UL] += sum( A.load(i+4UL,j) * x1 );
            y[i+5UL] += sum( A.load(i+5UL,j) * x1 );
            y[i+6UL] += sum( A.load(i+6UL,j) * x1 );
            y[i+7UL] += sum( A.load(i+7UL,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
            y[i+2UL] += A(i+2UL,j) * x[j];
            y[i+3UL] += A(i+3UL,j) * x[j];
            y[i+4UL] += A(i+4UL,j) * x[j];
            y[i+5UL] += A(i+5UL,j) * x[j];
            y[i+6UL] += A(i+6UL,j) * x[j];
            y[i+7UL] += A(i+7UL,j) * x[j];
         }
      }

      for( ; (i+4UL) <= M; i+=4UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+3UL : i+4UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 + A.load(i+2UL,j2) * x3 + A.load(i+2UL,j3) * x4 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 + A.load(i+3UL,j2) * x3 + A.load(i+3UL,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] += sum( A.load(i    ,j) * x1 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
            y[i+2UL] += A(i+2UL,j) * x[j];
            y[i+3UL] += A(i+3UL,j) * x[j];
         }
      }

      for( ; (i+2UL) <= M; i+=2UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+1UL : i+2UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] += sum( A.load(i    ,j) * x1 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
         }
      }

      if( i < M )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i] += sum( A.load(i,j) * x1 + A.load(i,j1) * x2 + A.load(i,j2) * x3 + A.load(i,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i] += sum( A.load(i,j) * x1 + A.load(i,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i] += sum( A.load(i,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i] += A(i,j) * x[j];
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based assignment to dense vectors (default)********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a dense matrix-dense vector multiplication
   //        (\f$ \vec{y}=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a large dense
   // matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline DisableIf_< UseBlasKernel<VT1,MT1,VT2> >
      selectBlasAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      selectLargeAssignKernel( y, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based assignment to dense vectors******************************************************
#if BLAZE_BLAS_MODE
   /*! \cond BLAZE_INTERNAL */
   /*!\brief BLAS-based assignment of a dense matrix-dense vector multiplication
   //        (\f$ \vec{y}=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function performs the dense matrix-dense vector multiplication based on the according
   // BLAS functionality.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseBlasKernel<VT1,MT1,VT2> >
      selectBlasAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      typedef ElementType_<VT1>  ET;

      if( IsTriangular<MT1>::value ) {
         assign( y, x );
         trmv( y, A, ( IsLower<MT1>::value )?( CblasLower ):( CblasUpper ) );
      }
      else {
         gemv( y, A, x, ET(1), ET(0) );
      }
   }
   /*! \endcond */
#endif
   //**********************************************************************************************

   //**Assignment to sparse vectors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense matrix-dense vector multiplication to a sparse vector
   //        (\f$ \vec{y}=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense matrix-dense
   // vector multiplication expression to a sparse vector.
   */
   template< typename VT1 >  // Type of the target sparse vector
   friend inline void assign( SparseVector<VT1,false>& lhs, const DMatDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( serial( rhs ) );
      assign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense vectors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a dense matrix-dense vector multiplication to a dense vector
   //        (\f$ \vec{y}+=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a dense matrix-
   // dense vector multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void addAssign( DenseVector<VT1,false>& lhs, const DMatDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      if( rhs.mat_.rows() == 0UL || rhs.mat_.columns() == 0UL ) {
         return;
      }

      LT A( serial( rhs.mat_ ) );  // Evaluation of the left-hand side dense matrix operand
      RT x( serial( rhs.vec_ ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.mat_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.mat_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( x.size()    == rhs.vec_.size()   , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).size()     , "Invalid vector size"       );

      DMatDVecMultExpr::selectAddAssignKernel( ~lhs, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense vectors (kernel selection)*************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for an addition assignment of a dense matrix-dense vector
   //        multiplication to a dense vector (\f$ \vec{y}+=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline void selectAddAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      if( ( IsDiagonal<MT1>::value ) ||
          ( IsComputation<MT>::value && !evaluateMatrix ) ||
          ( A.rows() * A.columns() < DMATDVECMULT_THRESHOLD ) )
         selectSmallAddAssignKernel( y, A, x );
      else
         selectBlasAddAssignKernel( y, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense vectors************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a dense matrix-dense vector multiplication
   //        (\f$ \vec{y}+=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the default addition assignment kernel for the dense matrix-dense
   // vector multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline void selectDefaultAddAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      y.addAssign( A * x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense vectors (small matrices)*******************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a small dense matrix-dense vector multiplication
   //        (\f$ \vec{y}+=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a
   // dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2> >
      selectSmallAddAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      selectDefaultAddAssignKernel( y, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default addition assignment to dense vectors (small matrices)********************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default addition assignment of a small dense matrix-dense vector
   //        multiplication (\f$ \vec{y}+=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the vectorized default addition assignment kernel for the dense
   // matrix-dense vector multiplication. This kernel is optimized for small matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2> >
      selectSmallAddAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<MT1>::value || !IsPadded<VT2>::value );

      size_t i( 0UL );

      for( ; (i+8UL) <= M; i+=8UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+7UL : i+8UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
            xmm4 = xmm4 + A.load(i+3UL,j) * x1;
            xmm5 = xmm5 + A.load(i+4UL,j) * x1;
            xmm6 = xmm6 + A.load(i+5UL,j) * x1;
            xmm7 = xmm7 + A.load(i+6UL,j) * x1;
            xmm8 = xmm8 + A.load(i+7UL,j) * x1;
         }

         y[i    ] += sum( xmm1 );
         y[i+1UL] += sum( xmm2 );
         y[i+2UL] += sum( xmm3 );
         y[i+3UL] += sum( xmm4 );
         y[i+4UL] += sum( xmm5 );
         y[i+5UL] += sum( xmm6 );
         y[i+6UL] += sum( xmm7 );
         y[i+7UL] += sum( xmm8 );

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
            y[i+2UL] += A(i+2UL,j) * x[j];
            y[i+3UL] += A(i+3UL,j) * x[j];
            y[i+4UL] += A(i+4UL,j) * x[j];
            y[i+5UL] += A(i+5UL,j) * x[j];
            y[i+6UL] += A(i+6UL,j) * x[j];
            y[i+7UL] += A(i+7UL,j) * x[j];
         }
      }

      for( ; (i+4UL) <= M; i+=4UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+3UL : i+4UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
            xmm4 = xmm4 + A.load(i+3UL,j) * x1;
         }

         y[i    ] += sum( xmm1 );
         y[i+1UL] += sum( xmm2 );
         y[i+2UL] += sum( xmm3 );
         y[i+3UL] += sum( xmm4 );

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
            y[i+2UL] += A(i+2UL,j) * x[j];
            y[i+3UL] += A(i+3UL,j) * x[j];
         }
      }

      for( ; (i+3UL) <= M; i+=3UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+2UL : i+3UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
         }

         y[i    ] += sum( xmm1 );
         y[i+1UL] += sum( xmm2 );
         y[i+2UL] += sum( xmm3 );

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
            y[i+2UL] += A(i+2UL,j) * x[j];
         }
      }

      for( ; (i+2UL) <= M; i+=2UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+1UL : i+2UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
         }

         y[i    ] += sum( xmm1 );
         y[i+1UL] += sum( xmm2 );

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
         }
      }

      if( i < M )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            xmm1 = xmm1 + A.load(i,j) * x.load(j);
         }

         y[i] += sum( xmm1 );

         for( ; remainder && j<jend; ++j ) {
            y[i] += A(i,j) * x[j];
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense vectors (large matrices)*******************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a large dense matrix-dense vector multiplication
   //        (\f$ \vec{y}+=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a
   // dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2> >
      selectLargeAddAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      selectDefaultAddAssignKernel( y, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default addition assignment to dense vectors (large matrices)********************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default addition assignment of a large dense matrix-dense vector
   //        multiplication (\f$ \vec{y}+=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the vectorized default addition assignment kernel for the dense
   // matrix-dense vector multiplication. This kernel is optimized for large matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2> >
      selectLargeAddAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<MT1>::value || !IsPadded<VT2>::value );

      size_t i( 0UL );

      for( ; (i+8UL) <= M; i+=8UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+7UL : i+8UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 + A.load(i+2UL,j2) * x3 + A.load(i+2UL,j3) * x4 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 + A.load(i+3UL,j2) * x3 + A.load(i+3UL,j3) * x4 );
            y[i+4UL] += sum( A.load(i+4UL,j) * x1 + A.load(i+4UL,j1) * x2 + A.load(i+4UL,j2) * x3 + A.load(i+4UL,j3) * x4 );
            y[i+5UL] += sum( A.load(i+5UL,j) * x1 + A.load(i+5UL,j1) * x2 + A.load(i+5UL,j2) * x3 + A.load(i+5UL,j3) * x4 );
            y[i+6UL] += sum( A.load(i+6UL,j) * x1 + A.load(i+6UL,j1) * x2 + A.load(i+6UL,j2) * x3 + A.load(i+6UL,j3) * x4 );
            y[i+7UL] += sum( A.load(i+7UL,j) * x1 + A.load(i+7UL,j1) * x2 + A.load(i+7UL,j2) * x3 + A.load(i+7UL,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 );
            y[i+4UL] += sum( A.load(i+4UL,j) * x1 + A.load(i+4UL,j1) * x2 );
            y[i+5UL] += sum( A.load(i+5UL,j) * x1 + A.load(i+5UL,j1) * x2 );
            y[i+6UL] += sum( A.load(i+6UL,j) * x1 + A.load(i+6UL,j1) * x2 );
            y[i+7UL] += sum( A.load(i+7UL,j) * x1 + A.load(i+7UL,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] += sum( A.load(i    ,j) * x1 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 );
            y[i+4UL] += sum( A.load(i+4UL,j) * x1 );
            y[i+5UL] += sum( A.load(i+5UL,j) * x1 );
            y[i+6UL] += sum( A.load(i+6UL,j) * x1 );
            y[i+7UL] += sum( A.load(i+7UL,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
            y[i+2UL] += A(i+2UL,j) * x[j];
            y[i+3UL] += A(i+3UL,j) * x[j];
            y[i+4UL] += A(i+4UL,j) * x[j];
            y[i+5UL] += A(i+5UL,j) * x[j];
            y[i+6UL] += A(i+6UL,j) * x[j];
            y[i+7UL] += A(i+7UL,j) * x[j];
         }
      }

      for( ; (i+4UL) <= M; i+=4UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+3UL : i+4UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 + A.load(i+2UL,j2) * x3 + A.load(i+2UL,j3) * x4 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 + A.load(i+3UL,j2) * x3 + A.load(i+3UL,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] += sum( A.load(i    ,j) * x1 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
            y[i+2UL] += A(i+2UL,j) * x[j];
            y[i+3UL] += A(i+3UL,j) * x[j];
         }
      }

      for( ; (i+2UL) <= M; i+=2UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+1UL : i+2UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] += sum( A.load(i    ,j) * x1 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
         }
      }

      if( i < M )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i] += sum( A.load(i,j) * x1 + A.load(i,j1) * x2 + A.load(i,j2) * x3 + A.load(i,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i] += sum( A.load(i,j) * x1 + A.load(i,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i] += sum( A.load(i,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i] += A(i,j) * x[j];
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense vectors (default)***********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a dense matrix-dense vector multiplication
   //        (\f$ \vec{y}+=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a large
   // dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline DisableIf_< UseBlasKernel<VT1,MT1,VT2> >
      selectBlasAddAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      selectLargeAddAssignKernel( y, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense vectors*********************************************
#if BLAZE_BLAS_MODE
   /*! \cond BLAZE_INTERNAL */
   /*!\brief BLAS-based addition assignment of a matrix-vector multiplication
   //        (\f$ \vec{y}+=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function performs the dense matrix-dense vector multiplication based on the according
   // BLAS functionality.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseBlasKernel<VT1,MT1,VT2> >
      selectBlasAddAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      typedef ElementType_<VT1>  ET;

      if( IsTriangular<MT1>::value ) {
         ResultType_<VT1> tmp( serial( x ) );
         trmv( tmp, A, ( IsLower<MT1>::value )?( CblasLower ):( CblasUpper ) );
         addAssign( y, tmp );
      }
      else {
         gemv( y, A, x, ET(1), ET(1) );
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
   /*!\brief Subtraction assignment of a dense matrix-dense vector multiplication to a dense vector
   //        (\f$ \vec{y}-=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a dense matrix-
   // dense vector multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void subAssign( DenseVector<VT1,false>& lhs, const DMatDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      if( rhs.mat_.rows() == 0UL || rhs.mat_.columns() == 0UL ) {
         return;
      }

      LT A( serial( rhs.mat_ ) );  // Evaluation of the left-hand side dense matrix operand
      RT x( serial( rhs.vec_ ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.mat_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.mat_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( x.size()    == rhs.vec_.size()   , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).size()     , "Invalid vector size"       );

      DMatDVecMultExpr::selectSubAssignKernel( ~lhs, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to dense vectors (kernel selection)**********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for a subtraction assignment of a dense matrix-dense vector
   //        multiplication to a dense vector (\f$ \vec{y}-=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline void selectSubAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      if( ( IsDiagonal<MT1>::value ) ||
          ( IsComputation<MT>::value && !evaluateMatrix ) ||
          ( A.rows() * A.columns() < DMATDVECMULT_THRESHOLD ) )
         selectSmallSubAssignKernel( y, A, x );
      else
         selectBlasSubAssignKernel( y, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors*********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a dense matrix-dense vector multiplication
   //        (\f$ \vec{y}-=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the default subtraction assignment kernel for the dense matrix-
   // dense vector multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline void selectDefaultSubAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      y.subAssign( A * x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors (small matrices)****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a small dense matrix-dense vector multiplication
   //        (\f$ \vec{y}-=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a
   // dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2> >
      selectSmallSubAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      selectDefaultSubAssignKernel( y, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to dense vectors (small matrices)*****************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default subtraction assignment of a small dense matrix-dense vector
   //        multiplication (\f$ \vec{y}-=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment kernel for the dense
   // matrix-dense vector multiplication. This kernel is optimized for small matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2> >
      selectSmallSubAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<MT1>::value || !IsPadded<VT2>::value );

      size_t i( 0UL );

      for( ; (i+8UL) <= M; i+=8UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+7UL : i+8UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
            xmm4 = xmm4 + A.load(i+3UL,j) * x1;
            xmm5 = xmm5 + A.load(i+4UL,j) * x1;
            xmm6 = xmm6 + A.load(i+5UL,j) * x1;
            xmm7 = xmm7 + A.load(i+6UL,j) * x1;
            xmm8 = xmm8 + A.load(i+7UL,j) * x1;
         }

         y[i    ] -= sum( xmm1 );
         y[i+1UL] -= sum( xmm2 );
         y[i+2UL] -= sum( xmm3 );
         y[i+3UL] -= sum( xmm4 );
         y[i+4UL] -= sum( xmm5 );
         y[i+5UL] -= sum( xmm6 );
         y[i+6UL] -= sum( xmm7 );
         y[i+7UL] -= sum( xmm8 );

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j];
            y[i+1UL] -= A(i+1UL,j) * x[j];
            y[i+2UL] -= A(i+2UL,j) * x[j];
            y[i+3UL] -= A(i+3UL,j) * x[j];
            y[i+4UL] -= A(i+4UL,j) * x[j];
            y[i+5UL] -= A(i+5UL,j) * x[j];
            y[i+6UL] -= A(i+6UL,j) * x[j];
            y[i+7UL] -= A(i+7UL,j) * x[j];
         }
      }

      for( ; (i+4UL) <= M; i+=4UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+3UL : i+4UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
            xmm4 = xmm4 + A.load(i+3UL,j) * x1;
         }

         y[i    ] -= sum( xmm1 );
         y[i+1UL] -= sum( xmm2 );
         y[i+2UL] -= sum( xmm3 );
         y[i+3UL] -= sum( xmm4 );

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j];
            y[i+1UL] -= A(i+1UL,j) * x[j];
            y[i+2UL] -= A(i+2UL,j) * x[j];
            y[i+3UL] -= A(i+3UL,j) * x[j];
         }
      }

      for( ; (i+3UL) <= M; i+=3UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+2UL : i+3UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
         }

         y[i    ] -= sum( xmm1 );
         y[i+1UL] -= sum( xmm2 );
         y[i+2UL] -= sum( xmm3 );

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j];
            y[i+1UL] -= A(i+1UL,j) * x[j];
            y[i+2UL] -= A(i+2UL,j) * x[j];
         }
      }

      for( ; (i+2UL) <= M; i+=2UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+1UL : i+2UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
         }

         y[i    ] -= sum( xmm1 );
         y[i+1UL] -= sum( xmm2 );

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j];
            y[i+1UL] -= A(i+1UL,j) * x[j];
         }
      }

      if( i < M )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            xmm1 = xmm1 + A.load(i,j) * x.load(j);
         }

         y[i] -= sum( xmm1 );

         for( ; remainder && j<jend; ++j ) {
            y[i] -= A(i,j) * x[j];
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors (large matrices)****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a large dense matrix-dense vector multiplication
   //        (\f$ \vec{y}-=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a
   // dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2> >
      selectLargeSubAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      selectDefaultSubAssignKernel( y, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to dense vectors (large matrices)*****************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default subtraction assignment of a large dense matrix-dense vector
   //        multiplication (\f$ \vec{y}-=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment kernel for the dense
   // matrix-dense vector multiplication. This kernel is optimized for large matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2> >
      selectLargeSubAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<MT1>::value || !IsPadded<VT2>::value );

      size_t i( 0UL );

      for( ; (i+8UL) <= M; i+=8UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+7UL : i+8UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] -= sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 );
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 );
            y[i+2UL] -= sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 + A.load(i+2UL,j2) * x3 + A.load(i+2UL,j3) * x4 );
            y[i+3UL] -= sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 + A.load(i+3UL,j2) * x3 + A.load(i+3UL,j3) * x4 );
            y[i+4UL] -= sum( A.load(i+4UL,j) * x1 + A.load(i+4UL,j1) * x2 + A.load(i+4UL,j2) * x3 + A.load(i+4UL,j3) * x4 );
            y[i+5UL] -= sum( A.load(i+5UL,j) * x1 + A.load(i+5UL,j1) * x2 + A.load(i+5UL,j2) * x3 + A.load(i+5UL,j3) * x4 );
            y[i+6UL] -= sum( A.load(i+6UL,j) * x1 + A.load(i+6UL,j1) * x2 + A.load(i+6UL,j2) * x3 + A.load(i+6UL,j3) * x4 );
            y[i+7UL] -= sum( A.load(i+7UL,j) * x1 + A.load(i+7UL,j1) * x2 + A.load(i+7UL,j2) * x3 + A.load(i+7UL,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] -= sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 );
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 );
            y[i+2UL] -= sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 );
            y[i+3UL] -= sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 );
            y[i+4UL] -= sum( A.load(i+4UL,j) * x1 + A.load(i+4UL,j1) * x2 );
            y[i+5UL] -= sum( A.load(i+5UL,j) * x1 + A.load(i+5UL,j1) * x2 );
            y[i+6UL] -= sum( A.load(i+6UL,j) * x1 + A.load(i+6UL,j1) * x2 );
            y[i+7UL] -= sum( A.load(i+7UL,j) * x1 + A.load(i+7UL,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] -= sum( A.load(i    ,j) * x1 );
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 );
            y[i+2UL] -= sum( A.load(i+2UL,j) * x1 );
            y[i+3UL] -= sum( A.load(i+3UL,j) * x1 );
            y[i+4UL] -= sum( A.load(i+4UL,j) * x1 );
            y[i+5UL] -= sum( A.load(i+5UL,j) * x1 );
            y[i+6UL] -= sum( A.load(i+6UL,j) * x1 );
            y[i+7UL] -= sum( A.load(i+7UL,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j];
            y[i+1UL] -= A(i+1UL,j) * x[j];
            y[i+2UL] -= A(i+2UL,j) * x[j];
            y[i+3UL] -= A(i+3UL,j) * x[j];
            y[i+4UL] -= A(i+4UL,j) * x[j];
            y[i+5UL] -= A(i+5UL,j) * x[j];
            y[i+6UL] -= A(i+6UL,j) * x[j];
            y[i+7UL] -= A(i+7UL,j) * x[j];
         }
      }

      for( ; (i+4UL) <= M; i+=4UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+3UL : i+4UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] -= sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 );
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 );
            y[i+2UL] -= sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 + A.load(i+2UL,j2) * x3 + A.load(i+2UL,j3) * x4 );
            y[i+3UL] -= sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 + A.load(i+3UL,j2) * x3 + A.load(i+3UL,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] -= sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 );
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 );
            y[i+2UL] -= sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 );
            y[i+3UL] -= sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] -= sum( A.load(i    ,j) * x1 );
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 );
            y[i+2UL] -= sum( A.load(i+2UL,j) * x1 );
            y[i+3UL] -= sum( A.load(i+3UL,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j];
            y[i+1UL] -= A(i+1UL,j) * x[j];
            y[i+2UL] -= A(i+2UL,j) * x[j];
            y[i+3UL] -= A(i+3UL,j) * x[j];
         }
      }

      for( ; (i+2UL) <= M; i+=2UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+1UL : i+2UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] -= sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 );
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] -= sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 );
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] -= sum( A.load(i    ,j) * x1 );
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j];
            y[i+1UL] -= A(i+1UL,j) * x[j];
         }
      }

      if( i < M )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i] -= sum( A.load(i,j) * x1 + A.load(i,j1) * x2 + A.load(i,j2) * x3 + A.load(i,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i] -= sum( A.load(i,j) * x1 + A.load(i,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i] -= sum( A.load(i,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i] -= A(i,j) * x[j];
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense vectors (default)********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a dense matrix-dense vector multiplication
   //        (\f$ \vec{y}-=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a large
   // dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline DisableIf_< UseBlasKernel<VT1,MT1,VT2> >
      selectBlasSubAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      selectLargeSubAssignKernel( y, A, x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense vectors******************************************
#if BLAZE_BLAS_MODE
   /*! \cond BLAZE_INTERNAL */
   /*!\brief BLAS-based subtraction assignment of a matrix-vector multiplication
   //        (\f$ \vec{y}-=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \return void
   //
   // This function performs the dense matrix-dense vector multiplication based on the according
   // BLAS functionality.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2 >  // Type of the right-hand side vector operand
   static inline EnableIf_< UseBlasKernel<VT1,MT1,VT2> >
      selectBlasSubAssignKernel( VT1& y, const MT1& A, const VT2& x )
   {
      typedef ElementType_<VT1>  ET;

      if( IsTriangular<MT1>::value ) {
         ResultType_<VT1> tmp( serial( x ) );
         trmv( tmp, A, ( IsLower<MT1>::value )?( CblasLower ):( CblasUpper ) );
         subAssign( y, tmp );
      }
      else {
         gemv( y, A, x, ET(-1), ET(1) );
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
   /*!\brief Multiplication assignment of a dense matrix-dense vector multiplication to a dense vector
   //        (\f$ \vec{y}*=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a dense
   // matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void multAssign( DenseVector<VT1,false>& lhs, const DMatDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( ResultType );
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
   /*!\brief Division assignment of a dense matrix-dense vector multiplication to a dense vector
   //        (\f$ \vec{y}/=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression divisor.
   // \return void
   //
   // This function implements the performance optimized division assignment of a dense matrix-
   // dense vector multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void divAssign( DenseVector<VT1,false>& lhs, const DMatDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( ResultType );
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
   /*!\brief SMP assignment of a dense matrix-dense vector multiplication to a dense vector
   //        (\f$ \vec{y}=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense matrix-dense
   // vector multiplication expression to a dense vector. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAssign( DenseVector<VT1,false>& lhs, const DMatDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      if( rhs.mat_.rows() == 0UL ) {
         return;
      }
      else if( rhs.mat_.columns() == 0UL ) {
         reset( ~lhs );
         return;
      }

      LT A( rhs.mat_ );  // Evaluation of the left-hand side dense matrix operand
      RT x( rhs.vec_ );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.mat_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.mat_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( x.size()    == rhs.vec_.size()   , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).size()     , "Invalid vector size"       );

      smpAssign( ~lhs, A * x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to sparse vectors************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a dense matrix-dense vector multiplication to a sparse vector
   //        (\f$ \vec{y}=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense matrix-dense
   // vector multiplication expression to a sparse vector. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target sparse vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAssign( SparseVector<VT1,false>& lhs, const DMatDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( rhs );
      smpAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to dense vectors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a dense matrix-dense vector multiplication to a dense vector
   //        (\f$ \vec{y}+=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a dense matrix-
   // dense vector multiplication expression to a dense vector. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAddAssign( DenseVector<VT1,false>& lhs, const DMatDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      if( rhs.mat_.rows() == 0UL || rhs.mat_.columns() == 0UL ) {
         return;
      }

      LT A( rhs.mat_ );  // Evaluation of the left-hand side dense matrix operand
      RT x( rhs.vec_ );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.mat_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.mat_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( x.size()    == rhs.vec_.size()   , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).size()     , "Invalid vector size"       );

      smpAddAssign( ~lhs, A * x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to sparse vectors***************************************************
   // No special implementation for the SMP addition assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP subtraction assignment to dense vectors*************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a dense matrix-dense vector multiplication to a dense
   //        vector (\f$ \vec{y}-=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a dense
   // matrix-dense vector multiplication expression to a dense vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpSubAssign( DenseVector<VT1,false>& lhs, const DMatDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      if( rhs.mat_.rows() == 0UL || rhs.mat_.columns() == 0UL ) {
         return;
      }

      LT A( rhs.mat_ );  // Evaluation of the left-hand side dense matrix operand
      RT x( rhs.vec_ );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.mat_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.mat_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( x.size()    == rhs.vec_.size()   , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).size()     , "Invalid vector size"       );

      smpSubAssign( ~lhs, A * x );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to sparse vectors************************************************
   // No special implementation for the SMP subtraction assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP multiplication assignment to dense vectors**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP multiplication assignment of a dense matrix-dense vector multiplication to a
   //        dense vector (\f$ \vec{y}*=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a
   // dense matrix-dense vector multiplication expression to a dense vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpMultAssign( DenseVector<VT1,false>& lhs, const DMatDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( ResultType );
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
   /*!\brief SMP division assignment of a dense matrix-dense vector multiplication to a dense
   //        vector (\f$ \vec{y}/=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side multiplication expression divisor.
   // \return void
   //
   // This function implements the performance optimized SMP division assignment of a dense
   // matrix-dense vector multiplication expression to a dense vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpDivAssign( DenseVector<VT1,false>& lhs, const DMatDVecMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( ResultType );
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
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( VT );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( VT );
   BLAZE_CONSTRAINT_MUST_FORM_VALID_MATVECMULTEXPR( MT, VT );
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
/*!\brief Expression object for scaled dense matrix-dense vector multiplications.
// \ingroup dense_vector_expression
//
// This specialization of the DVecScalarMultExpr class represents the compile time expression
// for scaled multiplications between a row-major dense matrix and a non-transpose dense vector.
*/
template< typename MT    // Type of the left-hand side dense matrix
        , typename VT    // Type of the right-hand side dense vector
        , typename ST >  // Type of the scalar value
class DVecScalarMultExpr< DMatDVecMultExpr<MT,VT>, ST, false >
   : public DenseVector< DVecScalarMultExpr< DMatDVecMultExpr<MT,VT>, ST, false >, false >
   , private VecScalarMultExpr
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef DMatDVecMultExpr<MT,VT>  MVM;  //!< Type of the dense matrix-dense vector multiplication expression.
   typedef ResultType_<MVM>         RES;  //!< Result type of the dense matrix-dense vector multiplication expression.
   typedef ResultType_<MT>          MRT;  //!< Result type of the left-hand side dense matrix expression.
   typedef ResultType_<VT>          VRT;  //!< Result type of the right-hand side dense vector expression.
   typedef ElementType_<MRT>        MET;  //!< Element type of the left-hand side dense matrix expression.
   typedef ElementType_<VRT>        VET;  //!< Element type of the right-hand side dense vector expression.
   typedef CompositeType_<MT>       MCT;  //!< Composite type of the left-hand side dense matrix expression.
   typedef CompositeType_<VT>       VCT;  //!< Composite type of the right-hand side dense vector expression.
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the right-hand side dense matrix expression.
   enum : bool { evaluateMatrix = ( IsComputation<MT>::value && IsSame<MET,VET>::value &&
                                    IsBLASCompatible<MET>::value ) || RequiresEvaluation<MT>::value };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the right-hand side dense vector expression.
   enum : bool { evaluateVector = IsComputation<VT>::value || RequiresEvaluation<MT>::value };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! The UseSMPAssign struct is a helper struct for the selection of the parallel evaluation
       strategy. In case either the matrix or the vector operand requires an intermediate
       evaluation, the nested \a value will be set to 1, otherwise it will be 0. */
   template< typename T1 >
   struct UseSMPAssign {
      enum : bool { value = ( evaluateMatrix || evaluateVector ) };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   /*! In case the matrix type, the two involved vector types, and the scalar type are suited
       for a BLAS kernel, the nested \a value will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3, typename T4 >
   struct UseBlasKernel {
      enum : bool { value = BLAZE_BLAS_MODE &&
                            HasMutableDataAccess<T1>::value &&
                            HasConstDataAccess<T2>::value &&
                            HasConstDataAccess<T3>::value &&
                            !IsDiagonal<T2>::value &&
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
                            !IsDiagonal<T2>::value &&
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
   typedef DVecScalarMultExpr<MVM,ST,false>  This;           //!< Type of this DVecScalarMultExpr instance.
   typedef MultTrait_<RES,ST>                ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>        TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<ResultType>          ElementType;    //!< Resulting element type.
   typedef SIMDTrait_<ElementType>           SIMDType;       //!< Resulting SIMD element type.
   typedef const ElementType                 ReturnType;     //!< Return type for expression template evaluations.
   typedef const ResultType                  CompositeType;  //!< Data type for composite expression templates.

   //! Composite type of the left-hand side dense vector expression.
   typedef const DMatDVecMultExpr<MT,VT>  LeftOperand;

   //! Composite type of the right-hand side scalar value.
   typedef ST  RightOperand;

   //! Type for the assignment of the dense matrix operand of the left-hand side expression.
   typedef IfTrue_< evaluateMatrix, const MRT, MCT >  LT;

   //! Type for the assignment of the dense vector operand of the left-hand side expression.
   typedef IfTrue_< evaluateVector, const VRT, VCT >  RT;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = !IsDiagonal<MT>::value &&
                               MT::simdEnabled && VT::simdEnabled &&
                               AreSIMDCombinable<MET,VET,ST>::value &&
                               HasSIMDAdd<MET,VET>::value &&
                               HasSIMDMult<MET,VET>::value };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = !evaluateMatrix && MT::smpAssignable &&
                                 !evaluateVector && VT::smpAssignable };
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
   explicit inline DVecScalarMultExpr( const MVM& vector, ST scalar )
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
      LeftOperand_<MVM> A( vector_.leftOperand() );
      return ( !BLAZE_BLAS_IS_PARALLEL ||
               ( IsComputation<MT>::value && !evaluateMatrix ) ||
               ( A.rows() * A.columns() < DMATDVECMULT_THRESHOLD ) ) &&
             ( size() > SMP_DMATDVECMULT_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  vector_;  //!< Left-hand side dense vector of the multiplication expression.
   RightOperand scalar_;  //!< Right-hand side scalar of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense vectors*****************************************************************
   /*!\brief Assignment of a scaled dense matrix-dense vector multiplication to a dense vector
   //        (\f$ \vec{y}=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a scaled dense matrix-
   // dense vector multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void assign( DenseVector<VT1,false>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      LeftOperand_<MVM>  left ( rhs.vector_.leftOperand()  );
      RightOperand_<MVM> right( rhs.vector_.rightOperand() );

      if( left.rows() == 0UL ) {
         return;
      }
      else if( left.columns() == 0UL ) {
         reset( ~lhs );
         return;
      }

      LT A( serial( left  ) );  // Evaluation of the left-hand side dense matrix operand
      RT x( serial( right ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( x.size()    == right.size()  , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).size() , "Invalid vector size"       );

      DVecScalarMultExpr::selectAssignKernel( ~lhs, A, x, rhs.scalar_ );
   }
   //**********************************************************************************************

   //**Assignment to dense vectors (kernel selection)**********************************************
   /*!\brief Selection of the kernel for an assignment of a scaled dense matrix-dense vector
   //        multiplication to a dense vector (\f$ \vec{y}=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      if( ( IsDiagonal<MT1>::value ) ||
          ( IsComputation<MT>::value && !evaluateMatrix ) ||
          ( A.rows() * A.columns() < DMATDVECMULT_THRESHOLD ) )
         selectSmallAssignKernel( y, A, x, scalar );
      else
         selectBlasAssignKernel( y, A, x, scalar );
   }
   //**********************************************************************************************

   //**Default assignment to dense vectors*********************************************************
   /*!\brief Default assignment of a scaled dense matrix-dense vector multiplication
   //        (\f$ \vec{y}=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment kernel for the scaled dense matrix-dense
   // vector multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2,ST2> >
      selectDefaultAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      y.assign( A * x * scalar );
   }
   //**********************************************************************************************

   //**Default assignment to dense vectors (small matrices)****************************************
   /*!\brief Default assignment of a small scaled dense matrix-dense vector multiplication
   //        (\f$ \vec{y}=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a scaled dense
   // matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2,ST2> >
      selectSmallAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      selectDefaultAssignKernel( y, A, x, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default assignment to dense vectors (small matrices)*****************************
   /*!\brief Vectorized default assignment of a small scaled dense matrix-dense vector
   //        multiplication (\f$ \vec{y}=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default assignment kernel for the scaled dense
   // matrix-dense vector multiplication. This kernel is optimized for small matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2,ST2> >
      selectSmallAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<MT1>::value || !IsPadded<VT2>::value );

      size_t i( 0UL );

      for( ; (i+8UL) <= M; i+=8UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+7UL : i+8UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
            xmm4 = xmm4 + A.load(i+3UL,j) * x1;
            xmm5 = xmm5 + A.load(i+4UL,j) * x1;
            xmm6 = xmm6 + A.load(i+5UL,j) * x1;
            xmm7 = xmm7 + A.load(i+6UL,j) * x1;
            xmm8 = xmm8 + A.load(i+7UL,j) * x1;
         }

         y[i    ] = sum( xmm1 ) * scalar;
         y[i+1UL] = sum( xmm2 ) * scalar;
         y[i+2UL] = sum( xmm3 ) * scalar;
         y[i+3UL] = sum( xmm4 ) * scalar;
         y[i+4UL] = sum( xmm5 ) * scalar;
         y[i+5UL] = sum( xmm6 ) * scalar;
         y[i+6UL] = sum( xmm7 ) * scalar;
         y[i+7UL] = sum( xmm8 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j] * scalar;
            y[i+1UL] += A(i+1UL,j) * x[j] * scalar;
            y[i+2UL] += A(i+2UL,j) * x[j] * scalar;
            y[i+3UL] += A(i+3UL,j) * x[j] * scalar;
            y[i+4UL] += A(i+4UL,j) * x[j] * scalar;
            y[i+5UL] += A(i+5UL,j) * x[j] * scalar;
            y[i+6UL] += A(i+6UL,j) * x[j] * scalar;
            y[i+7UL] += A(i+7UL,j) * x[j] * scalar;
         }
      }

      for( ; (i+4UL) <= M; i+=4UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+3UL : i+4UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
            xmm4 = xmm4 + A.load(i+3UL,j) * x1;
         }

         y[i    ] = sum( xmm1 ) * scalar;
         y[i+1UL] = sum( xmm2 ) * scalar;
         y[i+2UL] = sum( xmm3 ) * scalar;
         y[i+3UL] = sum( xmm4 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j] * scalar;
            y[i+1UL] += A(i+1UL,j) * x[j] * scalar;
            y[i+2UL] += A(i+2UL,j) * x[j] * scalar;
            y[i+3UL] += A(i+3UL,j) * x[j] * scalar;
         }
      }

      for( ; (i+3UL) <= M; i+=3UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+2UL : i+3UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
         }

         y[i    ] = sum( xmm1 ) * scalar;
         y[i+1UL] = sum( xmm2 ) * scalar;
         y[i+2UL] = sum( xmm3 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j] * scalar;
            y[i+1UL] += A(i+1UL,j) * x[j] * scalar;
            y[i+2UL] += A(i+2UL,j) * x[j] * scalar;
         }
      }

      for( ; (i+2UL) <= M; i+=2UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+1UL : i+2UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
         }

         y[i    ] = sum( xmm1 ) * scalar;
         y[i+1UL] = sum( xmm2 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j] * scalar;
            y[i+1UL] += A(i+1UL,j) * x[j] * scalar;
         }
      }

      if( i < M )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            xmm1 = xmm1 + A.load(i,j) * x.load(j);
         }

         y[i] = sum( xmm1 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i] += A(i,j) * x[j] * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to dense vectors (large matrices)****************************************
   /*!\brief Default assignment of a large scaled dense matrix-dense vector multiplication
   //        (\f$ \vec{y}=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a scaled dense
   // matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2,ST2> >
      selectLargeAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      selectDefaultAssignKernel( y, A, x, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default assignment to dense vectors (large matrices)*****************************
   /*!\brief Vectorized default assignment of a large scaled dense matrix-dense vector
   //        multiplication (\f$ \vec{y}=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default assignment kernel for the scaled dense
   // matrix-dense vector multiplication. This kernel is optimized for large matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2,ST2> >
      selectLargeAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<MT1>::value || !IsPadded<VT2>::value );

      reset( y );

      size_t i( 0UL );

      for( ; (i+8UL) <= M; i+=8UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+7UL : i+8UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 + A.load(i+2UL,j2) * x3 + A.load(i+2UL,j3) * x4 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 + A.load(i+3UL,j2) * x3 + A.load(i+3UL,j3) * x4 );
            y[i+4UL] += sum( A.load(i+4UL,j) * x1 + A.load(i+4UL,j1) * x2 + A.load(i+4UL,j2) * x3 + A.load(i+4UL,j3) * x4 );
            y[i+5UL] += sum( A.load(i+5UL,j) * x1 + A.load(i+5UL,j1) * x2 + A.load(i+5UL,j2) * x3 + A.load(i+5UL,j3) * x4 );
            y[i+6UL] += sum( A.load(i+6UL,j) * x1 + A.load(i+6UL,j1) * x2 + A.load(i+6UL,j2) * x3 + A.load(i+6UL,j3) * x4 );
            y[i+7UL] += sum( A.load(i+7UL,j) * x1 + A.load(i+7UL,j1) * x2 + A.load(i+7UL,j2) * x3 + A.load(i+7UL,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 );
            y[i+4UL] += sum( A.load(i+4UL,j) * x1 + A.load(i+4UL,j1) * x2 );
            y[i+5UL] += sum( A.load(i+5UL,j) * x1 + A.load(i+5UL,j1) * x2 );
            y[i+6UL] += sum( A.load(i+6UL,j) * x1 + A.load(i+6UL,j1) * x2 );
            y[i+7UL] += sum( A.load(i+7UL,j) * x1 + A.load(i+7UL,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] += sum( A.load(i    ,j) * x1 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 );
            y[i+4UL] += sum( A.load(i+4UL,j) * x1 );
            y[i+5UL] += sum( A.load(i+5UL,j) * x1 );
            y[i+6UL] += sum( A.load(i+6UL,j) * x1 );
            y[i+7UL] += sum( A.load(i+7UL,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
            y[i+2UL] += A(i+2UL,j) * x[j];
            y[i+3UL] += A(i+3UL,j) * x[j];
            y[i+4UL] += A(i+4UL,j) * x[j];
            y[i+5UL] += A(i+5UL,j) * x[j];
            y[i+6UL] += A(i+6UL,j) * x[j];
            y[i+7UL] += A(i+7UL,j) * x[j];
         }

         y[i    ] *= scalar;
         y[i+1UL] *= scalar;
         y[i+2UL] *= scalar;
         y[i+3UL] *= scalar;
         y[i+4UL] *= scalar;
         y[i+5UL] *= scalar;
         y[i+6UL] *= scalar;
         y[i+7UL] *= scalar;
      }

      for( ; (i+4UL) <= M; i+=4UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+3UL : i+4UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 + A.load(i+2UL,j2) * x3 + A.load(i+2UL,j3) * x4 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 + A.load(i+3UL,j2) * x3 + A.load(i+3UL,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] += sum( A.load(i    ,j) * x1 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 );
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 );
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
            y[i+2UL] += A(i+2UL,j) * x[j];
            y[i+3UL] += A(i+3UL,j) * x[j];
         }

         y[i    ] *= scalar;
         y[i+1UL] *= scalar;
         y[i+2UL] *= scalar;
         y[i+3UL] *= scalar;
      }

      for( ; (i+2UL) <= M; i+=2UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+1UL : i+2UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] += sum( A.load(i    ,j) * x1 );
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j];
            y[i+1UL] += A(i+1UL,j) * x[j];
         }

         y[i    ] *= scalar;
         y[i+1UL] *= scalar;
      }

      if( i < M )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i] += sum( A.load(i,j) * x1 + A.load(i,j1) * x2 + A.load(i,j2) * x3 + A.load(i,j3) * x4 );
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i] += sum( A.load(i,j) * x1 + A.load(i,j1) * x2 );
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i] += sum( A.load(i,j) * x1 );
         }

         for( ; remainder && j<jend; ++j ) {
            y[i] += A(i,j) * x[j];
         }

         y[i] *= scalar;
      }
   }
   //**********************************************************************************************

   //**BLAS-based assignment to dense vectors (default)********************************************
   /*!\brief Default assignment of a scaled dense matrix-dense vector multiplication
   //        (\f$ \vec{y}=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a large scaled
   // dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseBlasKernel<VT1,MT1,VT2,ST2> >
      selectBlasAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      selectLargeAssignKernel( y, A, x, scalar );
   }
   //**********************************************************************************************

   //**BLAS-based assignment to dense vectors******************************************************
#if BLAZE_BLAS_MODE
   /*!\brief BLAS-based assignment of a scaled dense matrix-dense vector multiplication
   //        (\f$ \vec{y}=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function performs the scaled dense matrix-dense vector multiplication based on the
   // according BLAS functionality.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseBlasKernel<VT1,MT1,VT2,ST2> >
      selectBlasAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      typedef ElementType_<VT1>  ET;

      if( IsTriangular<MT1>::value ) {
         assign( y, scalar * x );
         trmv( y, A, ( IsLower<MT1>::value )?( CblasLower ):( CblasUpper ) );
      }
      else {
         gemv( y, A, x, ET(scalar), ET(0) );
      }
   }
#endif
   //**********************************************************************************************

   //**Assignment to sparse vectors****************************************************************
   /*!\brief Assignment of a scaled dense matrix-dense vector multiplication to a sparse vector
   //        (\f$ \vec{y}=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a scaled dense matrix-
   // dense vector multiplication expression to a sparse vector.
   */
   template< typename VT1 >  // Type of the target sparse vector
   friend inline void assign( SparseVector<VT1,false>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( serial( rhs ) );
      assign( ~lhs, tmp );
   }
   //**********************************************************************************************

   //**Addition assignment to dense vectors********************************************************
   /*!\brief Addition assignment of a scaled dense matrix-dense vector multiplication to a dense
   //        vector (\f$ \vec{y}+=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a scaled dense
   // matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void addAssign( DenseVector<VT1,false>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      LeftOperand_<MVM>  left ( rhs.vector_.leftOperand()  );
      RightOperand_<MVM> right( rhs.vector_.rightOperand() );

      if( left.rows() == 0UL || left.columns() == 0UL ) {
         return;
      }

      LT A( serial( left  ) );  // Evaluation of the left-hand side dense matrix operand
      RT x( serial( right ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( x.size()    == right.size()  , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).size() , "Invalid vector size"       );

      DVecScalarMultExpr::selectAddAssignKernel( ~lhs, A, x, rhs.scalar_ );
   }
   //**********************************************************************************************

   //**Addition assignment to dense vectors (kernel selection)*************************************
   /*!\brief Selection of the kernel for an addition assignment of a scaled dense matrix-dense
   //        vector multiplication to a dense vector (\f$ \vec{y}+=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectAddAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      if( ( IsDiagonal<MT1>::value ) ||
          ( IsComputation<MT>::value && !evaluateMatrix ) ||
          ( A.rows() * A.columns() < DMATDVECMULT_THRESHOLD ) )
         selectSmallAddAssignKernel( y, A, x, scalar );
      else
         selectBlasAddAssignKernel( y, A, x, scalar );
   }
   //**********************************************************************************************

   //**Default addition assignment to dense vectors************************************************
   /*!\brief Default addition assignment of a scaled dense matrix-dense vector multiplication
   //        (\f$ \vec{y}+=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default addition assignment kernel for the scaled dense matrix-
   // dense vector multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectDefaultAddAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      y.addAssign( A * x * scalar );
   }
   //**********************************************************************************************

   //**Default addition assignment to dense vectors (small matrices)*******************************
   /*!\brief Default addition assignment of a small scaled dense matrix-dense vector multiplication
   //        (\f$ \vec{y}+=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a scaled
   // dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2,ST2> >
      selectSmallAddAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      selectDefaultAddAssignKernel( y, A, x, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default addition assignment to dense vectors (small matrices)********************
   /*!\brief Vectorized default addition assignment of a small scaled dense matrix-dense vector
   //        multiplication (\f$ \vec{y}+=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default addition assignment kernel for the scaled
   // dense matrix-dense vector multiplication. This kernel is optimized for small matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2,ST2> >
      selectSmallAddAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<MT1>::value || !IsPadded<VT2>::value );

      size_t i( 0UL );

      for( ; (i+8UL) <= M; i+=8UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+7UL : i+8UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
            xmm4 = xmm4 + A.load(i+3UL,j) * x1;
            xmm5 = xmm5 + A.load(i+4UL,j) * x1;
            xmm6 = xmm6 + A.load(i+5UL,j) * x1;
            xmm7 = xmm7 + A.load(i+6UL,j) * x1;
            xmm8 = xmm8 + A.load(i+7UL,j) * x1;
         }

         y[i    ] += sum( xmm1 ) * scalar;
         y[i+1UL] += sum( xmm2 ) * scalar;
         y[i+2UL] += sum( xmm3 ) * scalar;
         y[i+3UL] += sum( xmm4 ) * scalar;
         y[i+4UL] += sum( xmm5 ) * scalar;
         y[i+5UL] += sum( xmm6 ) * scalar;
         y[i+6UL] += sum( xmm7 ) * scalar;
         y[i+7UL] += sum( xmm8 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j] * scalar;
            y[i+1UL] += A(i+1UL,j) * x[j] * scalar;
            y[i+2UL] += A(i+2UL,j) * x[j] * scalar;
            y[i+3UL] += A(i+3UL,j) * x[j] * scalar;
            y[i+4UL] += A(i+4UL,j) * x[j] * scalar;
            y[i+5UL] += A(i+5UL,j) * x[j] * scalar;
            y[i+6UL] += A(i+6UL,j) * x[j] * scalar;
            y[i+7UL] += A(i+7UL,j) * x[j] * scalar;
         }
      }

      for( ; (i+4UL) <= M; i+=4UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+3UL : i+4UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
            xmm4 = xmm4 + A.load(i+3UL,j) * x1;
         }

         y[i    ] += sum( xmm1 ) * scalar;
         y[i+1UL] += sum( xmm2 ) * scalar;
         y[i+2UL] += sum( xmm3 ) * scalar;
         y[i+3UL] += sum( xmm4 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j] * scalar;
            y[i+1UL] += A(i+1UL,j) * x[j] * scalar;
            y[i+2UL] += A(i+2UL,j) * x[j] * scalar;
            y[i+3UL] += A(i+3UL,j) * x[j] * scalar;
         }
      }

      for( ; (i+3UL) <= M; i+=3UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+2UL : i+3UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
         }

         y[i    ] += sum( xmm1 ) * scalar;
         y[i+1UL] += sum( xmm2 ) * scalar;
         y[i+2UL] += sum( xmm3 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j] * scalar;
            y[i+1UL] += A(i+1UL,j) * x[j] * scalar;
            y[i+2UL] += A(i+2UL,j) * x[j] * scalar;
         }
      }

      for( ; (i+2UL) <= M; i+=2UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+1UL : i+2UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
         }

         y[i    ] += sum( xmm1 ) * scalar;
         y[i+1UL] += sum( xmm2 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j] * scalar;
            y[i+1UL] += A(i+1UL,j) * x[j] * scalar;
         }
      }

      if( i < M )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            xmm1 = xmm1 + A.load(i,j) * x.load(j);
         }

         y[i] += sum( xmm1 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i] += A(i,j) * x[j] * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default addition assignment to dense vectors (large matrices)*******************************
   /*!\brief Default addition assignment of a large scaled dense matrix-dense vector multiplication
   //        (\f$ \vec{y}+=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a scaled
   // dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2,ST2> >
      selectLargeAddAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      selectDefaultAddAssignKernel( y, A, x, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default addition assignment to dense vectors (large matrices)********************
   /*!\brief Vectorized default addition assignment of a large scaled dense matrix-dense vector
   //        multiplication (\f$ \vec{y}+=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default addition assignment kernel for the scaled
   // dense matrix-dense vector multiplication. This kernel is optimized for large matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2,ST2> >
      selectLargeAddAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<MT1>::value || !IsPadded<VT2>::value );

      size_t i( 0UL );

      for( ; (i+8UL) <= M; i+=8UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+7UL : i+8UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 ) * scalar;
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 ) * scalar;
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 + A.load(i+2UL,j2) * x3 + A.load(i+2UL,j3) * x4 ) * scalar;
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 + A.load(i+3UL,j2) * x3 + A.load(i+3UL,j3) * x4 ) * scalar;
            y[i+4UL] += sum( A.load(i+4UL,j) * x1 + A.load(i+4UL,j1) * x2 + A.load(i+4UL,j2) * x3 + A.load(i+4UL,j3) * x4 ) * scalar;
            y[i+5UL] += sum( A.load(i+5UL,j) * x1 + A.load(i+5UL,j1) * x2 + A.load(i+5UL,j2) * x3 + A.load(i+5UL,j3) * x4 ) * scalar;
            y[i+6UL] += sum( A.load(i+6UL,j) * x1 + A.load(i+6UL,j1) * x2 + A.load(i+6UL,j2) * x3 + A.load(i+6UL,j3) * x4 ) * scalar;
            y[i+7UL] += sum( A.load(i+7UL,j) * x1 + A.load(i+7UL,j1) * x2 + A.load(i+7UL,j2) * x3 + A.load(i+7UL,j3) * x4 ) * scalar;
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 ) * scalar;
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 ) * scalar;
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 ) * scalar;
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 ) * scalar;
            y[i+4UL] += sum( A.load(i+4UL,j) * x1 + A.load(i+4UL,j1) * x2 ) * scalar;
            y[i+5UL] += sum( A.load(i+5UL,j) * x1 + A.load(i+5UL,j1) * x2 ) * scalar;
            y[i+6UL] += sum( A.load(i+6UL,j) * x1 + A.load(i+6UL,j1) * x2 ) * scalar;
            y[i+7UL] += sum( A.load(i+7UL,j) * x1 + A.load(i+7UL,j1) * x2 ) * scalar;
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] += sum( A.load(i    ,j) * x1 ) * scalar;
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 ) * scalar;
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 ) * scalar;
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 ) * scalar;
            y[i+4UL] += sum( A.load(i+4UL,j) * x1 ) * scalar;
            y[i+5UL] += sum( A.load(i+5UL,j) * x1 ) * scalar;
            y[i+6UL] += sum( A.load(i+6UL,j) * x1 ) * scalar;
            y[i+7UL] += sum( A.load(i+7UL,j) * x1 ) * scalar;
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j] * scalar;
            y[i+1UL] += A(i+1UL,j) * x[j] * scalar;
            y[i+2UL] += A(i+2UL,j) * x[j] * scalar;
            y[i+3UL] += A(i+3UL,j) * x[j] * scalar;
            y[i+4UL] += A(i+4UL,j) * x[j] * scalar;
            y[i+5UL] += A(i+5UL,j) * x[j] * scalar;
            y[i+6UL] += A(i+6UL,j) * x[j] * scalar;
            y[i+7UL] += A(i+7UL,j) * x[j] * scalar;
         }
      }

      for( ; (i+4UL) <= M; i+=4UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+3UL : i+4UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 ) * scalar;
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 ) * scalar;
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 + A.load(i+2UL,j2) * x3 + A.load(i+2UL,j3) * x4 ) * scalar;
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 + A.load(i+3UL,j2) * x3 + A.load(i+3UL,j3) * x4 ) * scalar;
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 ) * scalar;
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 ) * scalar;
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 ) * scalar;
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 ) * scalar;
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] += sum( A.load(i    ,j) * x1 ) * scalar;
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 ) * scalar;
            y[i+2UL] += sum( A.load(i+2UL,j) * x1 ) * scalar;
            y[i+3UL] += sum( A.load(i+3UL,j) * x1 ) * scalar;
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j] * scalar;
            y[i+1UL] += A(i+1UL,j) * x[j] * scalar;
            y[i+2UL] += A(i+2UL,j) * x[j] * scalar;
            y[i+3UL] += A(i+3UL,j) * x[j] * scalar;
         }
      }

      for( ; (i+2UL) <= M; i+=2UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+1UL : i+2UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 ) * scalar;
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 ) * scalar;
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] += sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 ) * scalar;
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 ) * scalar;
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] += sum( A.load(i    ,j) * x1 ) * scalar;
            y[i+1UL] += sum( A.load(i+1UL,j) * x1 ) * scalar;
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] += A(i    ,j) * x[j] * scalar;
            y[i+1UL] += A(i+1UL,j) * x[j] * scalar;
         }
      }

      if( i < M )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i] += sum( A.load(i,j) * x1 + A.load(i,j1) * x2 + A.load(i,j2) * x3 + A.load(i,j3) * x4 ) * scalar;
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i] += sum( A.load(i,j) * x1 + A.load(i,j1) * x2 ) * scalar;
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i] += sum( A.load(i,j) * x1 ) * scalar;
         }

         for( ; remainder && j<jend; ++j ) {
            y[i] += A(i,j) * x[j] * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense vectors (default)***********************************
   /*!\brief Default addition assignment of a scaled dense matrix-dense vector multiplication
   //        (\f$ \vec{y}+=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a large
   // scaled dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseBlasKernel<VT1,MT1,VT2,ST2> >
      selectBlasAddAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      selectLargeAddAssignKernel( y, A, x, scalar );
   }
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense vectors*********************************************
#if BLAZE_BLAS_MODE
   /*!\brief BLAS-based addition assignment of a scaled dense matrix-dense vector multiplication
   //        (\f$ \vec{y}+=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function performs the scaled dense matrix-dense vector multiplication based on the
   // according BLAS functionality.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseBlasKernel<VT1,MT1,VT2,ST2> >
      selectBlasAddAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      typedef ElementType_<VT1>  ET;

      if( IsTriangular<MT1>::value ) {
         ResultType_<VT1> tmp( serial( scalar * x ) );
         trmv( tmp, A, ( IsLower<MT1>::value )?( CblasLower ):( CblasUpper ) );
         addAssign( y, tmp );
      }
      else {
         gemv( y, A, x, ET(scalar), ET(1) );
      }
   }
#endif
   //**********************************************************************************************

   //**Addition assignment to sparse vectors*******************************************************
   // No special implementation for the addition assignment to sparse vectors.
   //**********************************************************************************************

   //**Subtraction assignment to dense vectors*****************************************************
   /*!\brief Subtraction assignment of a scaled dense matrix-dense vector multiplication to a
   //        dense vector (\f$ \vec{y}-=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a scaled
   // dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void subAssign( DenseVector<VT1,false>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      LeftOperand_<MVM>  left ( rhs.vector_.leftOperand()  );
      RightOperand_<MVM> right( rhs.vector_.rightOperand() );

      if( left.rows() == 0UL || left.columns() == 0UL ) {
         return;
      }

      LT A( serial( left  ) );  // Evaluation of the left-hand side dense matrix operand
      RT x( serial( right ) );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( x.size()    == right.size()  , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).size() , "Invalid vector size"       );

      DVecScalarMultExpr::selectSubAssignKernel( ~lhs, A, x, rhs.scalar_ );
   }
   //**********************************************************************************************

   //**Subtraction assignment to dense vectors (kernel selection)**********************************
   /*!\brief Selection of the kernel for a subtraction assignment of a scaled dense matrix-dense
   //        vector multiplication to a dense vector (\f$ \vec{y}-=A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectSubAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      if( ( IsDiagonal<MT1>::value ) ||
          ( IsComputation<MT>::value && !evaluateMatrix ) ||
          ( A.rows() * A.columns() < DMATDVECMULT_THRESHOLD ) )
         selectSmallSubAssignKernel( y, A, x, scalar );
      else
         selectBlasSubAssignKernel( y, A, x, scalar );
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors*********************************************
   /*!\brief Default subtraction assignment of a scaled dense matrix-dense vector multiplication
   //        (\f$ \vec{y}-=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default subtraction assignment kernel for the scaled dense
   // matrix-dense vector multiplication.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectDefaultSubAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      y.subAssign( A * x * scalar );
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors (small matrices)****************************
   /*!\brief Default subtraction assignment of a small scaled dense matrix-dense vector
   //        multiplication (\f$ \vec{y}-=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a
   // scaled dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2,ST2> >
      selectSmallSubAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      selectDefaultSubAssignKernel( y, A, x, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to dense vectors (small matrices)*****************
   /*!\brief Vectorized default subtraction assignment of a small scaled dense matrix-dense vector
   //        multiplication (\f$ \vec{y}-=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment kernel for the scaled
   // dense matrix-dense vector multiplication. This kernel is optimized for small matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2,ST2> >
      selectSmallSubAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<MT1>::value || !IsPadded<VT2>::value );

      size_t i( 0UL );

      for( ; (i+8UL) <= M; i+=8UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+7UL : i+8UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
            xmm4 = xmm4 + A.load(i+3UL,j) * x1;
            xmm5 = xmm5 + A.load(i+4UL,j) * x1;
            xmm6 = xmm6 + A.load(i+5UL,j) * x1;
            xmm7 = xmm7 + A.load(i+6UL,j) * x1;
            xmm8 = xmm8 + A.load(i+7UL,j) * x1;
         }

         y[i    ] -= sum( xmm1 ) * scalar;
         y[i+1UL] -= sum( xmm2 ) * scalar;
         y[i+2UL] -= sum( xmm3 ) * scalar;
         y[i+3UL] -= sum( xmm4 ) * scalar;
         y[i+4UL] -= sum( xmm5 ) * scalar;
         y[i+5UL] -= sum( xmm6 ) * scalar;
         y[i+6UL] -= sum( xmm7 ) * scalar;
         y[i+7UL] -= sum( xmm8 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j] * scalar;
            y[i+1UL] -= A(i+1UL,j) * x[j] * scalar;
            y[i+2UL] -= A(i+2UL,j) * x[j] * scalar;
            y[i+3UL] -= A(i+3UL,j) * x[j] * scalar;
            y[i+4UL] -= A(i+4UL,j) * x[j] * scalar;
            y[i+5UL] -= A(i+5UL,j) * x[j] * scalar;
            y[i+6UL] -= A(i+6UL,j) * x[j] * scalar;
            y[i+7UL] -= A(i+7UL,j) * x[j] * scalar;
         }
      }

      for( ; (i+4UL) <= M; i+=4UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+3UL : i+4UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3, xmm4;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
            xmm4 = xmm4 + A.load(i+3UL,j) * x1;
         }

         y[i    ] -= sum( xmm1 ) * scalar;
         y[i+1UL] -= sum( xmm2 ) * scalar;
         y[i+2UL] -= sum( xmm3 ) * scalar;
         y[i+3UL] -= sum( xmm4 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j] * scalar;
            y[i+1UL] -= A(i+1UL,j) * x[j] * scalar;
            y[i+2UL] -= A(i+2UL,j) * x[j] * scalar;
            y[i+3UL] -= A(i+3UL,j) * x[j] * scalar;
         }
      }

      for( ; (i+3UL) <= M; i+=3UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+2UL : i+3UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2, xmm3;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
            xmm3 = xmm3 + A.load(i+2UL,j) * x1;
         }

         y[i    ] -= sum( xmm1 ) * scalar;
         y[i+1UL] -= sum( xmm2 ) * scalar;
         y[i+2UL] -= sum( xmm3 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j] * scalar;
            y[i+1UL] -= A(i+1UL,j) * x[j] * scalar;
            y[i+2UL] -= A(i+2UL,j) * x[j] * scalar;
         }
      }

      for( ; (i+2UL) <= M; i+=2UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+1UL : i+2UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1, xmm2;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            xmm1 = xmm1 + A.load(i    ,j) * x1;
            xmm2 = xmm2 + A.load(i+1UL,j) * x1;
         }

         y[i    ] -= sum( xmm1 ) * scalar;
         y[i+1UL] -= sum( xmm2 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j] * scalar;
            y[i+1UL] -= A(i+1UL,j) * x[j] * scalar;
         }
      }

      if( i < M )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         SIMDType xmm1;
         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            xmm1 = xmm1 + A.load(i,j) * x.load(j);
         }

         y[i] -= sum( xmm1 ) * scalar;

         for( ; remainder && j<jend; ++j ) {
            y[i] -= A(i,j) * x[j] * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense vectors (large matrices)****************************
   /*!\brief Default subtraction assignment of a large scaled dense matrix-dense vector
   //        multiplication (\f$ \vec{y}-=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a
   // scaled dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2,ST2> >
      selectLargeSubAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      selectDefaultSubAssignKernel( y, A, x, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to dense vectors (large matrices)*****************
   /*!\brief Vectorized default subtraction assignment of a large scaled dense matrix-dense vector
   //        multiplication (\f$ \vec{y}-=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment kernel for the scaled
   // dense matrix-dense vector multiplication. This kernel is optimized for large matrices.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseVectorizedDefaultKernel<VT1,MT1,VT2,ST2> >
      selectLargeSubAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( A.columns() );

      const bool remainder( !IsPadded<MT1>::value || !IsPadded<VT2>::value );

      size_t i( 0UL );

      for( ; (i+8UL) <= M; i+=8UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+7UL : i+8UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] -= sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 ) * scalar;
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 ) * scalar;
            y[i+2UL] -= sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 + A.load(i+2UL,j2) * x3 + A.load(i+2UL,j3) * x4 ) * scalar;
            y[i+3UL] -= sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 + A.load(i+3UL,j2) * x3 + A.load(i+3UL,j3) * x4 ) * scalar;
            y[i+4UL] -= sum( A.load(i+4UL,j) * x1 + A.load(i+4UL,j1) * x2 + A.load(i+4UL,j2) * x3 + A.load(i+4UL,j3) * x4 ) * scalar;
            y[i+5UL] -= sum( A.load(i+5UL,j) * x1 + A.load(i+5UL,j1) * x2 + A.load(i+5UL,j2) * x3 + A.load(i+5UL,j3) * x4 ) * scalar;
            y[i+6UL] -= sum( A.load(i+6UL,j) * x1 + A.load(i+6UL,j1) * x2 + A.load(i+6UL,j2) * x3 + A.load(i+6UL,j3) * x4 ) * scalar;
            y[i+7UL] -= sum( A.load(i+7UL,j) * x1 + A.load(i+7UL,j1) * x2 + A.load(i+7UL,j2) * x3 + A.load(i+7UL,j3) * x4 ) * scalar;
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] -= sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 ) * scalar;
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 ) * scalar;
            y[i+2UL] -= sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 ) * scalar;
            y[i+3UL] -= sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 ) * scalar;
            y[i+4UL] -= sum( A.load(i+4UL,j) * x1 + A.load(i+4UL,j1) * x2 ) * scalar;
            y[i+5UL] -= sum( A.load(i+5UL,j) * x1 + A.load(i+5UL,j1) * x2 ) * scalar;
            y[i+6UL] -= sum( A.load(i+6UL,j) * x1 + A.load(i+6UL,j1) * x2 ) * scalar;
            y[i+7UL] -= sum( A.load(i+7UL,j) * x1 + A.load(i+7UL,j1) * x2 ) * scalar;
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] -= sum( A.load(i    ,j) * x1 ) * scalar;
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 ) * scalar;
            y[i+2UL] -= sum( A.load(i+2UL,j) * x1 ) * scalar;
            y[i+3UL] -= sum( A.load(i+3UL,j) * x1 ) * scalar;
            y[i+4UL] -= sum( A.load(i+4UL,j) * x1 ) * scalar;
            y[i+5UL] -= sum( A.load(i+5UL,j) * x1 ) * scalar;
            y[i+6UL] -= sum( A.load(i+6UL,j) * x1 ) * scalar;
            y[i+7UL] -= sum( A.load(i+7UL,j) * x1 ) * scalar;
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j] * scalar;
            y[i+1UL] -= A(i+1UL,j) * x[j] * scalar;
            y[i+2UL] -= A(i+2UL,j) * x[j] * scalar;
            y[i+3UL] -= A(i+3UL,j) * x[j] * scalar;
            y[i+4UL] -= A(i+4UL,j) * x[j] * scalar;
            y[i+5UL] -= A(i+5UL,j) * x[j] * scalar;
            y[i+6UL] -= A(i+6UL,j) * x[j] * scalar;
            y[i+7UL] -= A(i+7UL,j) * x[j] * scalar;
         }
      }

      for( ; (i+4UL) <= M; i+=4UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+3UL : i+4UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] -= sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 ) * scalar;
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 ) * scalar;
            y[i+2UL] -= sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 + A.load(i+2UL,j2) * x3 + A.load(i+2UL,j3) * x4 ) * scalar;
            y[i+3UL] -= sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 + A.load(i+3UL,j2) * x3 + A.load(i+3UL,j3) * x4 ) * scalar;
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] -= sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 ) * scalar;
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 ) * scalar;
            y[i+2UL] -= sum( A.load(i+2UL,j) * x1 + A.load(i+2UL,j1) * x2 ) * scalar;
            y[i+3UL] -= sum( A.load(i+3UL,j) * x1 + A.load(i+3UL,j1) * x2 ) * scalar;
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] -= sum( A.load(i    ,j) * x1 ) * scalar;
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 ) * scalar;
            y[i+2UL] -= sum( A.load(i+2UL,j) * x1 ) * scalar;
            y[i+3UL] -= sum( A.load(i+3UL,j) * x1 ) * scalar;
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j] * scalar;
            y[i+1UL] -= A(i+1UL,j) * x[j] * scalar;
            y[i+2UL] -= A(i+2UL,j) * x[j] * scalar;
            y[i+3UL] -= A(i+3UL,j) * x[j] * scalar;
         }
      }

      for( ; (i+2UL) <= M; i+=2UL )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i+1UL : i+2UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i    ] -= sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 + A.load(i    ,j2) * x3 + A.load(i    ,j3) * x4 ) * scalar;
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 + A.load(i+1UL,j2) * x3 + A.load(i+1UL,j3) * x4 ) * scalar;
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i    ] -= sum( A.load(i    ,j) * x1 + A.load(i    ,j1) * x2 ) * scalar;
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 + A.load(i+1UL,j1) * x2 ) * scalar;
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i    ] -= sum( A.load(i    ,j) * x1 ) * scalar;
            y[i+1UL] -= sum( A.load(i+1UL,j) * x1 ) * scalar;
         }

         for( ; remainder && j<jend; ++j ) {
            y[i    ] -= A(i    ,j) * x[j] * scalar;
            y[i+1UL] -= A(i+1UL,j) * x[j] * scalar;
         }
      }

      if( i < M )
      {
         const size_t jbegin( ( IsUpper<MT1>::value )
                              ?( ( IsStrictlyUpper<MT1>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                              :( 0UL ) );
         const size_t jend( ( IsLower<MT1>::value )
                            ?( IsStrictlyLower<MT1>::value ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( remainder ? ( jend & size_t(-SIMDSIZE) ) : jend );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            const size_t j1( j+SIMDSIZE     );
            const size_t j2( j+SIMDSIZE*2UL );
            const size_t j3( j+SIMDSIZE*3UL );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            const SIMDType x3( x.load(j2) );
            const SIMDType x4( x.load(j3) );
            y[i] -= sum( A.load(i,j) * x1 + A.load(i,j1) * x2 + A.load(i,j2) * x3 + A.load(i,j3) * x4 ) * scalar;
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL ) {
            const size_t j1( j+SIMDSIZE );
            const SIMDType x1( x.load(j ) );
            const SIMDType x2( x.load(j1) );
            y[i] -= sum( A.load(i,j) * x1 + A.load(i,j1) * x2 ) * scalar;
         }

         for( ; j<jpos; j+=SIMDSIZE ) {
            const SIMDType x1( x.load(j) );
            y[i] -= sum( A.load(i,j) * x1 ) * scalar;
         }

         for( ; remainder && j<jend; ++j ) {
            y[i] -= A(i,j) * x[j] * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense vectors (default)********************************
   /*!\brief Default subtraction assignment of a scaled dense matrix-dense vector multiplication
   //        (\f$ \vec{y}-=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a large
   // scaled dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_< UseBlasKernel<VT1,MT1,VT2,ST2> >
      selectBlasSubAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      selectLargeSubAssignKernel( y, A, x, scalar );
   }
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense vectors******************************************
#if BLAZE_BLAS_MODE
   /*!\brief BLAS-based subtraction assignment of a scaled dense matrix-dense vector multiplication
   //        (\f$ \vec{y}-=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param y The target left-hand side dense vector.
   // \param A The left-hand side dense matrix operand.
   // \param x The right-hand side dense vector operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function performs the scaled dense matrix-dense vector multiplication based on the
   // according BLAS functionality.
   */
   template< typename VT1    // Type of the left-hand side target vector
           , typename MT1    // Type of the left-hand side matrix operand
           , typename VT2    // Type of the right-hand side vector operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_< UseBlasKernel<VT1,MT1,VT2,ST2> >
      selectBlasSubAssignKernel( VT1& y, const MT1& A, const VT2& x, ST2 scalar )
   {
      typedef ElementType_<VT1>  ET;

      if( IsTriangular<MT1>::value ) {
         ResultType_<VT1> tmp( serial( scalar * x ) );
         trmv( tmp, A, ( IsLower<MT1>::value )?( CblasLower ):( CblasUpper ) );
         subAssign( y, tmp );
      }
      else {
         gemv( y, A, x, ET(-scalar), ET(1) );
      }
   }
#endif
   //**********************************************************************************************

   //**Subtraction assignment to sparse vectors****************************************************
   // No special implementation for the subtraction assignment to sparse vectors.
   //**********************************************************************************************

   //**Multiplication assignment to dense vectors**************************************************
   /*!\brief Multiplication assignment of a scaled dense matrix-dense vector multiplication to a
   //        dense vector (\f$ \vec{y}*=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a scaled
   // dense matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void multAssign( DenseVector<VT1,false>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( ResultType );
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
   /*!\brief Division assignment of a scaled dense matrix-dense vector multiplication to a dense
   //        vector (\f$ \vec{y}/=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression divisor.
   // \return void
   //
   // This function implements the performance optimized division assignment of a scaled dense
   // matrix-dense vector multiplication expression to a dense vector.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline void divAssign( DenseVector<VT1,false>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( ResultType );
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
   /*!\brief SMP assignment of a scaled dense matrix-dense vector multiplication to a dense vector
   //        (\f$ \vec{y}=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a scaled dense matrix-
   // dense vector multiplication expression to a dense vector. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAssign( DenseVector<VT1,false>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      LeftOperand_<MVM>  left ( rhs.vector_.leftOperand()  );
      RightOperand_<MVM> right( rhs.vector_.rightOperand() );

      if( left.rows() == 0UL ) {
         return;
      }
      else if( left.columns() == 0UL ) {
         reset( ~lhs );
         return;
      }

      LT A( left  );  // Evaluation of the left-hand side dense matrix operand
      RT x( right );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( x.size()    == right.size()  , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).size() , "Invalid vector size"       );

      smpAssign( ~lhs, A * x * rhs.scalar_ );
   }
   //**********************************************************************************************

   //**SMP assignment to sparse vectors************************************************************
   /*!\brief SMP assignment of a scaled dense matrix-dense vector multiplication to a sparse vector
   //        (\f$ \vec{y}=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a scaled dense matrix-
   // dense vector multiplication expression to a sparse vector. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target sparse vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAssign( SparseVector<VT1,false>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      const ResultType tmp( rhs );
      smpAssign( ~lhs, tmp );
   }
   //**********************************************************************************************

   //**SMP addition assignment to dense vectors****************************************************
   /*!\brief SMP addition assignment of a scaled dense matrix-dense vector multiplication to a
   //        dense vector (\f$ \vec{y}+=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a scaled
   // dense matrix-dense vector multiplication expression to a dense vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler in
   // case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpAddAssign( DenseVector<VT1,false>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      LeftOperand_<MVM>  left ( rhs.vector_.leftOperand()  );
      RightOperand_<MVM> right( rhs.vector_.rightOperand() );

      if( left.rows() == 0UL || left.columns() == 0UL ) {
         return;
      }

      LT A( left  );  // Evaluation of the left-hand side dense matrix operand
      RT x( right );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( x.size()    == right.size()  , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).size() , "Invalid vector size"       );

      smpAddAssign( ~lhs, A * x * rhs.scalar_ );
   }
   //**********************************************************************************************

   //**SMP addition assignment to sparse vectors***************************************************
   // No special implementation for the SMP addition assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP subtraction assignment to dense vectors*************************************************
   /*!\brief SMP subtraction assignment of a scaled dense matrix-dense vector multiplication to a
   //        dense vector (\f$ \vec{y}-=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a scaled
   // dense matrix-dense vector multiplication expression to a dense vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler in
   // case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpSubAssign( DenseVector<VT1,false>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid vector sizes" );

      LeftOperand_<MVM>  left ( rhs.vector_.leftOperand()  );
      RightOperand_<MVM> right( rhs.vector_.rightOperand() );

      if( left.rows() == 0UL || left.columns() == 0UL ) {
         return;
      }

      LT A( left  );  // Evaluation of the left-hand side dense matrix operand
      RT x( right );  // Evaluation of the right-hand side dense vector operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( x.size()    == right.size()  , "Invalid vector size"       );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).size() , "Invalid vector size"       );

      smpSubAssign( ~lhs, A * x * rhs.scalar_ );
   }
   //**********************************************************************************************

   //**SMP subtraction assignment to sparse vectors************************************************
   // No special implementation for the SMP subtraction assignment to sparse vectors.
   //**********************************************************************************************

   //**SMP multiplication assignment to dense vectors**********************************************
   /*!\brief SMP multiplication assignment of a scaled dense matrix-dense vector multiplication
   //        to a dense vector (\f$ \vec{y}*=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a scaled
   // dense matrix-dense vector multiplication expression to a dense vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler in
   // case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpMultAssign( DenseVector<VT1,false>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( ResultType );
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
   /*!\brief SMP division assignment of a scaled dense matrix-dense vector multiplication to a
   //        dense vector (\f$ \vec{y}/=s*A*\vec{x} \f$).
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side scaled multiplication expression division.
   // \return void
   //
   // This function implements the performance optimized SMP division assignment of a scaled
   // dense matrix-dense vector multiplication expression to a dense vector. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler in
   // case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense vector
   friend inline EnableIf_< UseSMPAssign<VT1> >
      smpDivAssign( DenseVector<VT1,false>& lhs, const DVecScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( ResultType );
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
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( MVM );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( MVM );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( MT );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( VT );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE( VT );
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
/*!\brief Multiplication operator for the multiplication of a row-major dense matrix and a dense
//        vector (\f$ \vec{y}=A*\vec{x} \f$).
// \ingroup dense_vector
//
// \param mat The left-hand side row-major dense matrix for the multiplication.
// \param vec The right-hand side dense vector for the multiplication.
// \return The resulting vector.
// \exception std::invalid_argument Matrix and vector sizes do not match.
//
// This operator represents the multiplication between a row-major dense matrix and a dense vector:

   \code
   using blaze::rowMajor;
   using blaze::columnVector;

   blaze::DynamicMatrix<double,rowMajor> A;
   blaze::DynamicVector<double,columnVector> x, y;
   // ... Resizing and initialization
   y = A * x;
   \endcode

// The operator returns an expression representing a dense vector of the higher-order element
// type of the two involved element types \a T1::ElementType and \a T2::ElementType. Both the
// dense matrix type \a T1 and the dense vector type \a T2 as well as the two element types
// \a T1::ElementType and \a T2::ElementType have to be supported by the MultTrait class
// template.\n
// In case the current size of the vector \a vec doesn't match the current number of columns
// of the matrix \a mat, a \a std::invalid_argument is thrown.
*/
template< typename T1    // Type of the left-hand side dense matrix
        , typename T2 >  // Type of the right-hand side dense vector
inline const DisableIf_< IsMatMatMultExpr<T1>, DMatDVecMultExpr<T1,T2> >
   operator*( const DenseMatrix<T1,false>& mat, const DenseVector<T2,false>& vec )
{
   BLAZE_FUNCTION_TRACE;

   if( (~mat).columns() != (~vec).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix and vector sizes do not match" );
   }

   return DMatDVecMultExpr<T1,T2>( ~mat, ~vec );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Multiplication operator for the multiplication of a dense matrix-matrix
//        multiplication expression and a dense vector (\f$ \vec{y}=(A*B)*\vec{x} \f$).
// \ingroup dense_vector
//
// \param mat The left-hand side dense matrix-matrix multiplication.
// \param vec The right-hand side dense vector for the multiplication.
// \return The resulting vector.
//
// This operator implements a performance optimized treatment of the multiplication of a dense
// matrix-matrix multiplication expression and a dense vector. It restructures the expression
// \f$ \vec{x}=(A*B)*\vec{x} \f$ to the expression \f$ \vec{y}=A*(B*\vec{x}) \f$.
*/
template< typename T1    // Type of the left-hand side dense matrix
        , bool SO        // Storage order of the left-hand side dense matrix
        , typename T2 >  // Type of the right-hand side dense vector
inline const EnableIf_< IsMatMatMultExpr<T1>, MultExprTrait_<T1,T2> >
   operator*( const DenseMatrix<T1,SO>& mat, const DenseVector<T2,false>& vec )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( T1 );

   return (~mat).leftOperand() * ( (~mat).rightOperand() * vec );
}
//*************************************************************************************************




//=================================================================================================
//
//  SIZE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, typename VT >
struct Size< DMatDVecMultExpr<MT,VT> > : public Rows<MT>
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
template< typename MT, typename VT >
struct IsAligned< DMatDVecMultExpr<MT,VT> >
   : public BoolConstant< And< IsAligned<MT>, IsAligned<VT> >::value >
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
template< typename MT, typename VT, bool AF >
struct SubvectorExprTrait< DMatDVecMultExpr<MT,VT>, AF >
{
 public:
   //**********************************************************************************************
   using Type = MultExprTrait_< SubmatrixExprTrait_<const MT,AF>
                              , SubvectorExprTrait_<const VT,AF> >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
