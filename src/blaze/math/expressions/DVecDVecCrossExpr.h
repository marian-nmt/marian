//=================================================================================================
/*!
//  \file blaze/math/expressions/DVecDVecCrossExpr.h
//  \brief Header file for the dense vector/dense vector cross product expression
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

#ifndef _BLAZE_MATH_EXPRESSIONS_DVECDVECCROSSEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_DVECDVECCROSSEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/ColumnVector.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/dense/Forward.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/CrossExpr.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/traits/CrossTrait.h>
#include <blaze/math/traits/MultExprTrait.h>
#include <blaze/math/traits/SubExprTrait.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsTemporary.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/logging/FunctionTrace.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/SizeT.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DVECDVECCROSSEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for dense vector-dense vector cross products.
// \ingroup dense_vector_expression
//
// The DVecDVecCrossExpr class represents the compile time expression for cross products
// between dense vectors.
*/
template< typename VT1  // Type of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side dense vector
        , bool TF >     // Transpose flag
class DVecDVecCrossExpr : public DenseVector< DVecDVecCrossExpr<VT1,VT2,TF>, TF >
                        , private CrossExpr
                        , private Computation
{
 private:
   //**Type definitions****************************************************************************
   typedef ResultType_<VT1>     RT1;  //!< Result type of the left-hand side dense vector expression.
   typedef ResultType_<VT2>     RT2;  //!< Result type of the right-hand side dense vector expression.
   typedef ReturnType_<VT1>     RN1;  //!< Return type of the left-hand side dense vector expression.
   typedef ReturnType_<VT2>     RN2;  //!< Return type of the right-hand side dense vector expression.
   typedef CompositeType_<VT1>  CT1;  //!< Composite type of the left-hand side dense vector expression.
   typedef CompositeType_<VT2>  CT2;  //!< Composite type of the right-hand side dense vector expression.
   typedef ElementType_<VT1>    ET1;  //!< Element type of the left-hand side dense vector expression.
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
   typedef SubExprTrait_< MultExprTrait_<RN1,RN2>, MultExprTrait_<RN1,RN2> >  ExprReturnType;
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef DVecDVecCrossExpr<VT1,VT2,TF>  This;           //!< Type of this DVecDVecCrossExpr instance.
   typedef CrossTrait_<RT1,RT2>           ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>     TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<ResultType>       ElementType;    //!< Resulting element type.

   //! Return type for expression template evaluations.
   typedef const IfTrue_< returnExpr, ExprReturnType, ElementType >  ReturnType;

   //! Data type for composite expression templates.
   typedef const ResultType  CompositeType;

   //! Composite type of the left-hand side dense vector expression.
   typedef If_< IsExpression<VT1>, const VT1, const VT1& >  LeftOperand;

   //! Composite type of the right-hand side dense vector expression.
   typedef If_< IsExpression<VT2>, const VT2, const VT2& >  RightOperand;

   //! Composite type of the left-hand side dense vector expression.
   typedef If_< IsComputation<VT1>, const StaticVector<ET1,3UL,TF>, CT1 >  LT;

   //! Composite type of the right-hand side dense vector expression.
   typedef If_< IsComputation<VT2>, const StaticVector<ET2,3UL,TF>, CT2 >  RT;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = false };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = false };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DVecDVecCrossExpr class.
   //
   // \param lhs The left-hand side operand of the cross product expression.
   // \param rhs The right-hand side operand of the cross product expression.
   */
   explicit inline DVecDVecCrossExpr( const VT1& lhs, const VT2& rhs ) noexcept
      : lhs_( lhs )  // Left-hand side dense vector of the cross product expression
      , rhs_( rhs )  // Right-hand side dense vector of the cross product expression
   {
      BLAZE_INTERNAL_ASSERT( lhs.size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( rhs.size() == 3UL, "Invalid vector size" );
   }
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the vector elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..2]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator[]( size_t index ) const {
      BLAZE_INTERNAL_ASSERT( index < 3UL, "Invalid vector access index" );

      if( index == 0UL )
         return lhs_[1UL] * rhs_[2UL] - lhs_[2UL] * rhs_[1UL];
      else if( index == 1UL )
         return lhs_[2UL] * rhs_[0UL] - lhs_[0UL] * rhs_[2UL];
      else
         return lhs_[0UL] * rhs_[1UL] - lhs_[1UL] * rhs_[0UL];
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
      if( index >= 3UL ) {
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
   inline constexpr size_t size() const noexcept {
      return 3UL;
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side dense vector operand.
   //
   // \return The left-hand side dense vector operand.
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
   LeftOperand  lhs_;  //!< Left-hand side dense vector of the cross product expression.
   RightOperand rhs_;  //!< Right-hand side dense vector of the cross product expression.
   //**********************************************************************************************

   //**Assignment to dense vectors*****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense vector-dense vector cross product to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side cross product expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense vector-dense
   // vector cross product expression to a dense vector.
   */
   template< typename VT >  // Type of the target dense vector
   friend inline void assign( DenseVector<VT,TF>& lhs, const DVecDVecCrossExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( (~rhs).size() == 3UL, "Invalid vector size" );

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense vector operand

      (~lhs)[0] = x[1UL]*y[2UL] - x[2UL]*y[1UL];
      (~lhs)[1] = x[2UL]*y[0UL] - x[0UL]*y[2UL];
      (~lhs)[2] = x[0UL]*y[1UL] - x[1UL]*y[0UL];
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to sparse vectors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense vector-dense vector cross product to a sparse vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side cross product expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense vector-dense
   // vector cross product expression to a sparse vector.
   */
   template< typename VT >  // Type of the target sparse vector
   friend inline void assign( SparseVector<VT,TF>& lhs, const DVecDVecCrossExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType, TF );
      BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<ResultType> );

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( (~rhs).size() == 3UL, "Invalid vector size" );

      const ResultType tmp( serial( rhs ) );
      assign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense vectors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a dense vector-dense vector cross product to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side cross product expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a dense vector-
   // dense vector cross product expression to a dense vector.
   */
   template< typename VT >  // Type of the target dense vector
   friend inline void addAssign( DenseVector<VT,TF>& lhs, const DVecDVecCrossExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( (~rhs).size() == 3UL, "Invalid vector size" );

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense vector operand

      (~lhs)[0] += x[1UL]*y[2UL] - x[2UL]*y[1UL];
      (~lhs)[1] += x[2UL]*y[0UL] - x[0UL]*y[2UL];
      (~lhs)[2] += x[0UL]*y[1UL] - x[1UL]*y[0UL];
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to sparse vectors*******************************************************
   // No special implementation for the addition assignment to sparse vectors.
   //**********************************************************************************************

   //**Subtraction assignment to dense vectors*****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a dense vector-dense vector cross product to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side cross product expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a dense vector-
   // dense vector cross product expression to a dense vector.
   */
   template< typename VT >  // Type of the target dense vector
   friend inline void subAssign( DenseVector<VT,TF>& lhs, const DVecDVecCrossExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( (~rhs).size() == 3UL, "Invalid vector size" );

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense vector operand

      (~lhs)[0] -= x[1UL]*y[2UL] - x[2UL]*y[1UL];
      (~lhs)[1] -= x[2UL]*y[0UL] - x[0UL]*y[2UL];
      (~lhs)[2] -= x[0UL]*y[1UL] - x[1UL]*y[0UL];
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to sparse vectors****************************************************
   // No special implementation for the subtraction assignment to sparse vectors.
   //**********************************************************************************************

   //**Multiplication assignment to dense vectors**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Multiplication assignment of a dense vector-dense vector cross product to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side cross product expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a dense
   // vector-dense vector cross product expression to a dense vector.
   */
   template< typename VT >  // Type of the target dense vector
   friend inline void multAssign( DenseVector<VT,TF>& lhs, const DVecDVecCrossExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( (~rhs).size() == 3UL, "Invalid vector size" );

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense vector operand

      (~lhs)[0] *= x[1UL]*y[2UL] - x[2UL]*y[1UL];
      (~lhs)[1] *= x[2UL]*y[0UL] - x[0UL]*y[2UL];
      (~lhs)[2] *= x[0UL]*y[1UL] - x[1UL]*y[0UL];
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Multiplication assignment to sparse vectors*************************************************
   // No special implementation for the multiplication assignment to sparse vectors.
   //**********************************************************************************************

   //**Division assignment to dense vectors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Division assignment of a dense vector-dense vector cross product to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side cross product expression divisor.
   // \return void
   //
   // This function implements the performance optimized division assignment of a dense vector-
   // dense vector cross product expression to a dense vector.
   */
   template< typename VT >  // Type of the target dense vector
   friend inline void divAssign( DenseVector<VT,TF>& lhs, const DVecDVecCrossExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( (~rhs).size() == 3UL, "Invalid vector size" );

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense vector operand

      (~lhs)[0] /= x[1UL]*y[2UL] - x[2UL]*y[1UL];
      (~lhs)[1] /= x[2UL]*y[0UL] - x[0UL]*y[2UL];
      (~lhs)[2] /= x[0UL]*y[1UL] - x[1UL]*y[0UL];
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Division assignment to sparse vectors*******************************************************
   // No special implementation for the division assignment to sparse vectors.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( VT1 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( VT2 );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( VT1, TF );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( VT2, TF );
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
/*!\brief Operator for the cross product of two dense vectors (\f$ \vec{a}=\vec{b} \times \vec{c} \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side dense vector for the cross product.
// \param rhs The right-hand side dense vector for the cross product.
// \return The cross product of the two vectors.
// \exception std::invalid_argument Invalid vector size for cross product.
//
// This operator represents the cross product of two dense vectors:

   \code
   blaze::DynamicVector<double> a( 3UL ), b( 3UL );
   blaze::StaticVector<double,3UL> c;
   // ... Resizing and initialization
   c = a % b;
   \endcode

// The operator returns an expression representing a dense vector of the higher-order element
// type of the two involved vector element types \a T1::ElementType and \a T2::ElementType.
// Both vector types \a T1 and \a T2 as well as the two element types \a T1::ElementType and
// \a T2::ElementType have to be supported by the CrossTrait class template.\n
// In case the current sizes of the two given vectors don't match, a \a std::invalid_argument
// is thrown.
*/
template< typename T1  // Type of the left-hand side dense vector
        , typename T2  // Type of the right-hand side dense vector
        , bool TF >    // Transpose flag
inline const DVecDVecCrossExpr<T1,T2,TF>
   operator%( const DenseVector<T1,TF>& lhs, const DenseVector<T2,TF>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   if( (~lhs).size() != 3UL || (~rhs).size() != 3UL ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid vector size for cross product" );
   }

   return DVecDVecCrossExpr<T1,T2,TF>( ~lhs, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Cross product of two dense vectors (\f$ \vec{a}=\vec{b} \times \vec{c} \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side dense vector for the cross product.
// \param rhs The right-hand side dense vector for the cross product.
// \return The cross product of the two vectors.
// \exception std::invalid_argument Invalid vector size for cross product.
//
// This function computes the cross product of two dense vectors:

   \code
   blaze::DynamicVector<double> a( 3UL ), b( 3UL );
   blaze::StaticVector<double,3UL> c;
   // ... Resizing and initialization
   c = a % b;
   \endcode

// The operator returns an expression representing a dense vector of the higher-order element
// type of the two involved vector element types \a T1::ElementType and \a T2::ElementType.
// Both vector types \a T1 and \a T2 as well as the two element types \a T1::ElementType and
// \a T2::ElementType have to be supported by the CrossTrait class template.\n
// In case the current sizes of the two given vectors don't match, a \a std::invalid_argument
// is thrown.
*/
template< typename T1  // Type of the left-hand side dense vector
        , typename T2  // Type of the right-hand side dense vector
        , bool TF >    // Transpose flag
inline const DVecDVecCrossExpr<T1,T2,TF>
   cross( const DenseVector<T1,TF>& lhs, const DenseVector<T2,TF>& rhs )
{
   return lhs % rhs;
}
//*************************************************************************************************




//=================================================================================================
//
//  SIZE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2, bool TF >
struct Size< DVecDVecCrossExpr<VT1,VT2,TF> > : public SizeT<3UL>
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
