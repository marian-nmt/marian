//=================================================================================================
/*!
//  \file blaze/math/views/Subvector.h
//  \brief Header file for the implementation of the Subvector view
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

#ifndef _BLAZE_MATH_VIEWS_SUBVECTOR_H_
#define _BLAZE_MATH_VIEWS_SUBVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Vector.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/CrossTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/traits/SubvectorExprTrait.h>
#include <blaze/math/traits/SubvectorTrait.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsCrossExpr.h>
#include <blaze/math/typetraits/IsTransExpr.h>
#include <blaze/math/typetraits/IsVecEvalExpr.h>
#include <blaze/math/typetraits/IsVecForEachExpr.h>
#include <blaze/math/typetraits/IsVecScalarDivExpr.h>
#include <blaze/math/typetraits/IsVecScalarMultExpr.h>
#include <blaze/math/typetraits/IsVecSerialExpr.h>
#include <blaze/math/typetraits/IsVecTransExpr.h>
#include <blaze/math/typetraits/IsVecVecAddExpr.h>
#include <blaze/math/typetraits/IsVecVecDivExpr.h>
#include <blaze/math/typetraits/IsVecVecMultExpr.h>
#include <blaze/math/typetraits/IsVecVecSubExpr.h>
#include <blaze/math/views/subvector/BaseTemplate.h>
#include <blaze/math/views/subvector/Dense.h>
#include <blaze/math/views/subvector/Sparse.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/logging/FunctionTrace.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/RemoveReference.h>


namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Creating a view on a specific subvector of the given vector.
// \ingroup views
//
// \param vector The vector containing the subvector.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specific subvector of the vector.
// \exception std::invalid_argument Invalid subvector specification.
//
// This function returns an expression representing the specified subvector of the given vector.
// The following example demonstrates the creation of a dense and sparse subvector:

   \code
   using blaze::columnVector;
   using blaze::rowVector;

   typedef blaze::DynamicVector<double,columnVector>  DenseVector;
   typedef blaze::CompressedVector<int,rowVector>     SparseVector;

   DenseVector  d;
   SparseVector s;
   // ... Resizing and initialization

   // Creating a dense subvector of size 8, starting from index 4
   blaze::Subvector<DenseVector> dsv = subvector( d, 4UL, 8UL );

   // Creating a sparse subvector of size 7, starting from index 5
   blaze::Subvector<SparseVector> ssv = subvector( s, 5UL, 7UL );
   \endcode

// In case the subvector is not properly specified (i.e. if the specified first index is larger
// than the total size of the given vector or the subvector is specified beyond the size of the
// vector) a \a std::invalid_argument exception is thrown.
//
// Please note that this function creates an unaligned dense or sparse subvector. For instance,
// the creation of the dense subvector is equivalent to the following three function calls:

   \code
   blaze::Subvector<DenseVector>           dsv = subvector<unaligned>( v, 4UL, 8UL );
   blaze::Subvector<DenseVector,unaligned> dsv = subvector           ( v, 4UL, 8UL );
   blaze::Subvector<DenseVector,unaligned> dsv = subvector<unaligned>( v, 4UL, 8UL );
   \endcode

// In contrast to unaligned subvectors, which provide full flexibility, aligned subvectors pose
// additional alignment restrictions. However, especially in case of dense subvectors this may
// result in considerable performance improvements. In order to create an aligned subvector the
// following function call has to be used:

   \code
   blaze::Subvector<DenseVector,aligned> = subvector<aligned>( v, 4UL, 8UL );
   \endcode

// Note however that in this case the given \a index and \a size are subject to additional checks
// to guarantee proper alignment.
*/
template< typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline SubvectorExprTrait_<VT,unaligned>
   subvector( Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return subvector<unaligned>( ~vector, index, size );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subvector of the given vector.
// \ingroup views
//
// \param vector The vector containing the subvector.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specific subvector of the vector.
// \exception std::invalid_argument Invalid subvector specification.
//
// This function returns an expression representing the specified subvector of the given vector.
// The following example demonstrates the creation of a dense and sparse subvector:

   \code
   using blaze::columnVector;
   using blaze::rowVector;

   typedef blaze::DynamicVector<double,columnVector>  DenseVector;
   typedef blaze::CompressedVector<int,rowVector>     SparseVector;

   DenseVector  d;
   SparseVector s;
   // ... Resizing and initialization

   // Creating a dense subvector of size 8, starting from index 4
   blaze::Subvector<DenseVector> dsv = subvector( d, 4UL, 8UL );

   // Creating a sparse subvector of size 7, starting from index 5
   blaze::Subvector<SparseVector> ssv = subvector( s, 5UL, 7UL );
   \endcode

// In case the subvector is not properly specified (i.e. if the specified first index is larger
// than the total size of the given vector or the subvector is specified beyond the size of the
// vector) a \a std::invalid_argument exception is thrown.
//
// Please note that this function creates an unaligned dense or sparse subvector. For instance,
// the creation of the dense subvector is equivalent to the following three function calls:

   \code
   blaze::Subvector<DenseVector>           dsv = subvector<unaligned>( v, 4UL, 8UL );
   blaze::Subvector<DenseVector,unaligned> dsv = subvector           ( v, 4UL, 8UL );
   blaze::Subvector<DenseVector,unaligned> dsv = subvector<unaligned>( v, 4UL, 8UL );
   \endcode

// In contrast to unaligned subvectors, which provide full flexibility, aligned subvectors pose
// additional alignment restrictions. However, especially in case of dense subvectors this may
// result in considerable performance improvements. In order to create an aligned subvector the
// following function call has to be used:

   \code
   blaze::Subvector<DenseVector,aligned> = subvector<aligned>( v, 4UL, 8UL );
   \endcode

// Note however that in this case the given \a index and \a size are subject to additional checks
// to guarantee proper alignment.
*/
template< typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline SubvectorExprTrait_<const VT,unaligned>
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return subvector<unaligned>( ~vector, index, size );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subvector of the given vector.
// \ingroup views
//
// \param vector The vector containing the subvector.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specific subvector of the vector.
// \exception std::invalid_argument Invalid subvector specification.
//
// This function returns an expression representing an aligned or unaligned subvector of the
// given dense or sparse vector, based on the specified alignment flag \a AF. The following
// example demonstrates the creation of both an aligned and unaligned subvector:

   \code
   using blaze::columnVector;
   using blaze::rowVector;

   typedef blaze::DynamicVector<double,columnVector>  DenseVector;
   typedef blaze::CompressedVector<int,rowVector>     SparseVector;

   DenseVector  d;
   SparseVector s;
   // ... Resizing and initialization

   // Creating an aligned dense subvector of size 8 starting from index 4
   blaze::Subvector<DenseVector,aligned> dsv = subvector<aligned>( d, 4UL, 8UL );

   // Creating an unaligned subvector of size 7 starting from index 3
   blaze::Subvector<SparseVector,unaligned> ssv = subvector<unaligned>( s, 3UL, 7UL );
   \endcode

// In case the subvector is not properly specified (i.e. if the specified first index is larger
// than the total size of the given vector or the subvector is specified beyond the size of the
// vector) a \a std::invalid_argument exception is thrown.
//
// In contrast to unaligned subvectors, which provide full flexibility, aligned subvectors pose
// additional alignment restrictions and the given \a index is subject to additional checks to
// guarantee proper alignment. However, especially in case of dense subvectors this may result
// in considerable performance improvements.
//
// The alignment restrictions refer to system dependent address restrictions for the used element
// type and the available vectorization mode (SSE, AVX, ...). In order to be properly aligned the
// first element of the subvector must be aligned. The following source code gives some examples
// for a double precision dynamic vector, assuming that AVX is available, which packs 4 \c double
// values into a SIMD vector:

   \code
   using blaze::columnVector;

   typedef blaze::DynamicVector<double,columnVector>  VectorType;
   typedef blaze::Subvector<VectorType,aligned>       SubvectorType;

   VectorType d( 17UL );
   // ... Resizing and initialization

   // OK: Starts at the beginning, i.e. the first element is aligned
   SubvectorType dsv1 = subvector<aligned>( d, 0UL, 13UL );

   // OK: Start index is a multiple of 4, i.e. the first element is aligned
   SubvectorType dsv2 = subvector<aligned>( d, 4UL, 7UL );

   // OK: The start index is a multiple of 4 and the subvector includes the last element
   SubvectorType dsv3 = subvector<aligned>( d, 8UL, 9UL );

   // Error: Start index is not a multiple of 4, i.e. the first element is not aligned
   SubvectorType dsv4 = subvector<aligned>( d, 5UL, 8UL );
   \endcode

// In case any alignment restrictions are violated, a \a std::invalid_argument exception is thrown.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline DisableIf_< Or< IsComputation<VT>, IsTransExpr<VT> >, SubvectorExprTrait_<VT,AF> >
   subvector( Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return SubvectorExprTrait_<VT,AF>( ~vector, index, size );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific subvector of the given vector.
// \ingroup views
//
// \param vector The vector containing the subvector.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specific subvector of the vector.
// \exception std::invalid_argument Invalid subvector specification.
//
// This function returns an expression representing an aligned or unaligned subvector of the
// given dense or sparse vector, based on the specified alignment flag \a AF. The following
// example demonstrates the creation of both an aligned and unaligned subvector:

   \code
   using blaze::columnVector;
   using blaze::rowVector;

   typedef blaze::DynamicVector<double,columnVector>  DenseVector;
   typedef blaze::CompressedVector<int,rowVector>     SparseVector;

   DenseVector  d;
   SparseVector s;
   // ... Resizing and initialization

   // Creating an aligned dense subvector of size 8 starting from index 4
   blaze::Subvector<DenseVector,aligned> dsv = subvector<aligned>( d, 4UL, 8UL );

   // Creating an unaligned subvector of size 7 starting from index 3
   blaze::Subvector<SparseVector,unaligned> ssv = subvector<unaligned>( s, 3UL, 7UL );
   \endcode

// In case the subvector is not properly specified (i.e. if the specified first index is larger
// than the total size of the given vector or the subvector is specified beyond the size of the
// vector) a \a std::invalid_argument exception is thrown.
//
// In contrast to unaligned subvectors, which provide full flexibility, aligned subvectors pose
// additional alignment restrictions and the given \a index is subject to additional checks to
// guarantee proper alignment. However, especially in case of dense subvectors this may result
// in considerable performance improvements.
//
// The alignment restrictions refer to system dependent address restrictions for the used element
// type and the available vectorization mode (SSE, AVX, ...). In order to be properly aligned the
// first element of the subvector must be aligned. The following source code gives some examples
// for a double precision dynamic vector, assuming that AVX is available, which packs 4 \c double
// values into a SIMD vector:

   \code
   using blaze::columnVector;

   typedef blaze::DynamicVector<double,columnVector>  VectorType;
   typedef blaze::Subvector<VectorType,aligned>       SubvectorType;

   VectorType d( 17UL );
   // ... Resizing and initialization

   // OK: Starts at the beginning, i.e. the first element is aligned
   SubvectorType dsv1 = subvector<aligned>( d, 0UL, 13UL );

   // OK: Start index is a multiple of 4, i.e. the first element is aligned
   SubvectorType dsv2 = subvector<aligned>( d, 4UL, 7UL );

   // OK: The start index is a multiple of 4 and the subvector includes the last element
   SubvectorType dsv3 = subvector<aligned>( d, 8UL, 9UL );

   // Error: Start index is not a multiple of 4, i.e. the first element is not aligned
   SubvectorType dsv4 = subvector<aligned>( d, 5UL, 8UL );
   \endcode

// In case any alignment restrictions are violated, a \a std::invalid_argument exception is thrown.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline DisableIf_< Or< IsComputation<VT>, IsTransExpr<VT> >, SubvectorExprTrait_<const VT,AF> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return SubvectorExprTrait_<const VT,AF>( ~vector, index, size );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given vector/vector addition.
// \ingroup views
//
// \param vector The constant vector/vector addition.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the addition.
//
// This function returns an expression representing the specified subvector of the given
// vector/vector addition.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline EnableIf_< IsVecVecAddExpr<VT>, SubvectorExprTrait_<VT,AF> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return subvector<AF>( (~vector).leftOperand() , index, size ) +
          subvector<AF>( (~vector).rightOperand(), index, size );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given vector/vector subtraction.
// \ingroup views
//
// \param vector The constant vector/vector subtraction.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the subtraction.
//
// This function returns an expression representing the specified subvector of the given
// vector/vector subtraction.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline EnableIf_< IsVecVecSubExpr<VT>, SubvectorExprTrait_<VT,AF> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return subvector<AF>( (~vector).leftOperand() , index, size ) -
          subvector<AF>( (~vector).rightOperand(), index, size );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given vector/vector multiplication.
// \ingroup views
//
// \param vector The constant vector/vector multiplication.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the multiplication.
//
// This function returns an expression representing the specified subvector of the given
// vector/vector multiplication.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline EnableIf_< IsVecVecMultExpr<VT>, SubvectorExprTrait_<VT,AF> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return subvector<AF>( (~vector).leftOperand() , index, size ) *
          subvector<AF>( (~vector).rightOperand(), index, size );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given vector/vector division.
// \ingroup views
//
// \param vector The constant vector/vector division.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the division.
//
// This function returns an expression representing the specified subvector of the given
// vector/vector division.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline EnableIf_< IsVecVecDivExpr<VT>, SubvectorExprTrait_<VT,AF> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return subvector<AF>( (~vector).leftOperand() , index, size ) /
          subvector<AF>( (~vector).rightOperand(), index, size );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given vector/vector cross product.
// \ingroup views
//
// \param vector The constant vector/vector cross product.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the cross product.
//
// This function returns an expression representing the specified subvector of the given
// vector/vector cross product.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline EnableIf_< IsCrossExpr<VT>, SubvectorExprTrait_<VT,unaligned> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return SubvectorExprTrait_<VT,unaligned>( ~vector, index, size );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given vector/scalar multiplication.
// \ingroup views
//
// \param vector The constant vector/scalar multiplication.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the multiplication.
//
// This function returns an expression representing the specified subvector of the given
// vector/scalar multiplication.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline EnableIf_< IsVecScalarMultExpr<VT>, SubvectorExprTrait_<VT,AF> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return subvector<AF>( (~vector).leftOperand(), index, size ) * (~vector).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given vector/scalar division.
// \ingroup views
//
// \param vector The constant vector/scalar division.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the division.
//
// This function returns an expression representing the specified subvector of the given
// vector/scalar division.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline EnableIf_< IsVecScalarDivExpr<VT>, SubvectorExprTrait_<VT,AF> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return subvector<AF>( (~vector).leftOperand(), index, size ) / (~vector).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given vector custom operation.
// \ingroup views
//
// \param vector The constant vector custom operation.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the custom operation.
//
// This function returns an expression representing the specified subvector of the given vector
// custom operation.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline EnableIf_< IsVecForEachExpr<VT>, SubvectorExprTrait_<VT,AF> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return forEach( subvector<AF>( (~vector).operand(), index, size ), (~vector).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given vector evaluation operation.
// \ingroup views
//
// \param vector The constant vector evaluation operation.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the evaluation operation.
//
// This function returns an expression representing the specified subvector of the given vector
// evaluation operation.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline EnableIf_< IsVecEvalExpr<VT>, SubvectorExprTrait_<VT,AF> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return eval( subvector<AF>( (~vector).operand(), index, size ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given vector serialization operation.
// \ingroup views
//
// \param vector The constant vector serialization operation.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the serialization operation.
//
// This function returns an expression representing the specified subvector of the given vector
// serialization operation.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline EnableIf_< IsVecSerialExpr<VT>, SubvectorExprTrait_<VT,AF> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return serial( subvector<AF>( (~vector).operand(), index, size ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given vector transpose operation.
// \ingroup views
//
// \param vector The constant vector transpose operation.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the transpose operation.
//
// This function returns an expression representing the specified subvector of the given vector
// transpose operation.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline EnableIf_< IsVecTransExpr<VT>, SubvectorExprTrait_<VT,AF> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   return trans( subvector<AF>( (~vector).operand(), index, size ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of another subvector.
// \ingroup views
//
// \param sv The constant subvector.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the other subvector.
//
// This function returns an expression representing the specified subvector of the given subvector.
*/
template< bool AF1     // Required alignment flag
        , typename VT  // Type of the dense vector
        , bool AF2     // Present alignment flag
        , bool TF      // Transpose flag
        , bool DF >    // Density flag
inline const Subvector<VT,AF1,TF,DF>
   subvector( const Subvector<VT,AF2,TF,DF>& sv, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   if( index + size > sv.size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid subvector specification" );
   }

   return Subvector<VT,AF1,TF,DF>( sv.vector_, sv.offset_ + index, size );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBVECTOR OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Subvector operators */
//@{
template< typename VT, bool AF, bool TF, bool DF >
inline void reset( Subvector<VT,AF,TF,DF>& sv );

template< typename VT, bool AF, bool TF, bool DF >
inline void clear( Subvector<VT,AF,TF,DF>& sv );

template< typename VT, bool AF, bool TF, bool DF >
inline bool isDefault( const Subvector<VT,AF,TF,DF>& sv );

template< typename VT, bool AF, bool TF, bool DF >
inline bool isIntact( const Subvector<VT,AF,TF,DF>& sv ) noexcept;

template< typename VT, bool AF, bool TF, bool DF >
inline bool isSame( const Subvector<VT,AF,TF,DF>& a, const Vector<VT,TF>& b ) noexcept;

template< typename VT, bool AF, bool TF, bool DF >
inline bool isSame( const Vector<VT,TF>& a, const Subvector<VT,AF,TF,DF>& b ) noexcept;

template< typename VT, bool AF, bool TF, bool DF >
inline bool isSame( const Subvector<VT,AF,TF,DF>& a, const Subvector<VT,AF,TF,DF>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given subvector.
// \ingroup subvector
//
// \param sv The subvector to be resetted.
// \return void
*/
template< typename VT  // Type of the vector
        , bool AF      // Alignment flag
        , bool TF      // Transpose flag
        , bool DF >    // Density flag
inline void reset( Subvector<VT,AF,TF,DF>& sv )
{
   sv.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given subvector.
// \ingroup subvector
//
// \param sv The subvector to be cleared.
// \return void
*/
template< typename VT  // Type of the vector
        , bool AF      // Alignment flag
        , bool TF      // Transpose flag
        , bool DF >    // Density flag
inline void clear( Subvector<VT,AF,TF,DF>& sv )
{
   sv.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given subvector is in default state.
// \ingroup subvector
//
// \param sv The subvector to be tested for its default state.
// \return \a true in case the given subvector is component-wise zero, \a false otherwise.
//
// This function checks whether the subvector is in default state. For instance, in case the
// subvector is instantiated for a vector of built-in integral or floating point data type,
// the function returns \a true in case all subvector elements are 0 and \a false in case any
// subvector element is not 0. The following example demonstrates the use of the \a isDefault
// function:

   \code
   blaze::DynamicVector<int,rowVector> v;
   // ... Resizing and initialization
   if( isDefault( subvector( v, 10UL, 20UL ) ) ) { ... }
   \endcode
*/
template< typename VT  // Type of the vector
        , bool AF      // Alignment flag
        , bool TF      // Transpose flag
        , bool DF >    // Density flag
inline bool isDefault( const Subvector<VT,AF,TF,DF>& sv )
{
   for( size_t i=0UL; i<sv.size(); ++i )
      if( !isDefault( sv[i] ) ) return false;
   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given sparse subvector is in default state.
// \ingroup subvector
//
// \param sv The sparse subvector to be tested for its default state.
// \return \a true in case the given subvector is component-wise zero, \a false otherwise.
//
// This function checks whether the sparse subvector is in default state. For instance, in case
// the subvector is instantiated for a vector of built-in integral or floating point data type,
// the function returns \a true in case all subvector elements are 0 and \a false in case any
// element is not 0. The following example demonstrates the use of the \a isDefault function:

   \code
   blaze::CompressedVector<double,rowVector> v;
   // ... Resizing and initialization
   if( isDefault( subvector( v, 10UL, 20UL ) ) ) { ... }
   \endcode
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline bool isDefault( const Subvector<VT,AF,TF,false>& sv )
{
   typedef ConstIterator_< Subvector<VT,AF,TF,false> >  ConstIterator;

   const ConstIterator end( sv.end() );
   for( ConstIterator element=sv.begin(); element!=end; ++element )
      if( !isDefault( element->value() ) ) return false;
   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given subvector vector are intact.
// \ingroup subvector
//
// \param sv The subvector to be tested.
// \return \a true in case the given subvector's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the subvector are intact, i.e. if its state
// is valid. In case the invariants are intact, the function returns \a true, else it will
// return \a false. The following example demonstrates the use of the \a isIntact() function:

   \code
   blaze::DynamicVector<int,rowVector> v;
   // ... Resizing and initialization
   if( isIntact( subvector( v, 10UL, 20UL ) ) ) { ... }
   \endcode
*/
template< typename VT  // Type of the vector
        , bool AF      // Alignment flag
        , bool TF      // Transpose flag
        , bool DF >    // Density flag
inline bool isIntact( const Subvector<VT,AF,TF,DF>& sv ) noexcept
{
   return ( sv.offset_ + sv.size_ <= sv.vector_.size() &&
            isIntact( sv.vector_ ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given vector and subvector represent the same observable state.
// \ingroup subvector
//
// \param a The subvector to be tested for its state.
// \param b The vector to be tested for its state.
// \return \a true in case the subvector and vector share a state, \a false otherwise.
//
// This overload of the isSame function tests if the given subvector refers to the entire
// range of the given vector and by that represents the same observable state. In this case,
// the function returns \a true, otherwise it returns \a false.
*/
template< typename VT  // Type of the vector
        , bool AF      // Alignment flag
        , bool TF      // Transpose flag
        , bool DF >    // Density flag
inline bool isSame( const Subvector<VT,AF,TF,DF>& a, const Vector<VT,TF>& b ) noexcept
{
   return ( isSame( a.vector_, ~b ) && ( a.size() == (~b).size() ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given vector and subvector represent the same observable state.
// \ingroup subvector
//
// \param a The vector to be tested for its state.
// \param b The subvector to be tested for its state.
// \return \a true in case the vector and subvector share a state, \a false otherwise.
//
// This overload of the isSame function tests if the given subvector refers to the entire
// range of the given vector and by that represents the same observable state. In this case,
// the function returns \a true, otherwise it returns \a false.
*/
template< typename VT  // Type of the vector
        , bool AF      // Alignment flag
        , bool TF      // Transpose flag
        , bool DF >    // Density flag
inline bool isSame( const Vector<VT,TF>& a, const Subvector<VT,AF,TF,DF>& b ) noexcept
{
   return ( isSame( ~a, b.vector_ ) && ( (~a).size() == b.size() ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the two given subvectors represent the same observable state.
// \ingroup subvector
//
// \param a The first subvector to be tested for its state.
// \param b The second subvector to be tested for its state.
// \return \a true in case the two subvectors share a state, \a false otherwise.
//
// This overload of the isSame function tests if the two given subvectors refer to exactly the
// same range of the same vector. In case both subvectors represent the same observable state,
// the function returns \a true, otherwise it returns \a false.
*/
template< typename VT  // Type of the vector
        , bool AF      // Alignment flag
        , bool TF      // Transpose flag
        , bool DF >    // Density flag
inline bool isSame( const Subvector<VT,AF,TF,DF>& a, const Subvector<VT,AF,TF,DF>& b ) noexcept
{
   return ( isSame( a.vector_, b.vector_ ) && ( a.offset_ == b.offset_ ) && ( a.size_ == b.size_ ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a vector to a subvector.
// \ingroup subvector
//
// \param lhs The target left-hand side subvector.
// \param rhs The right-hand side vector to be assigned.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1    // Type of the vector
        , bool AF         // Alignment flag
        , bool TF         // Transpose flag
        , bool DF         // Density flag
        , typename VT2 >  // Type of the right-hand side vector
inline bool tryAssign( const Subvector<VT1,AF,TF,DF>& lhs, const Vector<VT2,TF>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.size() - index, "Invalid vector size" );

   return tryAssign( lhs.vector_, ~rhs, lhs.offset_ + index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to a subvector.
// \ingroup subvector
//
// \param lhs The target left-hand side subvector.
// \param rhs The right-hand side vector to be added.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1    // Type of the vector
        , bool AF         // Alignment flag
        , bool TF         // Transpose flag
        , bool DF         // Density flag
        , typename VT2 >  // Type of the right-hand side vector
inline bool tryAddAssign( const Subvector<VT1,AF,TF,DF>& lhs, const Vector<VT2,TF>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.size() - index, "Invalid vector size" );

   return tryAddAssign( lhs.vector_, ~rhs, lhs.offset_ + index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to a subvector.
// \ingroup subvector
//
// \param lhs The target left-hand side subvector.
// \param rhs The right-hand side vector to be subtracted.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1    // Type of the vector
        , bool AF         // Alignment flag
        , bool TF         // Transpose flag
        , bool DF         // Density flag
        , typename VT2 >  // Type of the right-hand side vector
inline bool trySubAssign( const Subvector<VT1,AF,TF,DF>& lhs, const Vector<VT2,TF>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.size() - index, "Invalid vector size" );

   return trySubAssign( lhs.vector_, ~rhs, lhs.offset_ + index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to a subvector.
// \ingroup subvector
//
// \param lhs The target left-hand side subvector.
// \param rhs The right-hand side vector to be multiplied.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1    // Type of the vector
        , bool AF         // Alignment flag
        , bool TF         // Transpose flag
        , bool DF         // Density flag
        , typename VT2 >  // Type of the right-hand side vector
inline bool tryMultAssign( const Subvector<VT1,AF,TF,DF>& lhs, const Vector<VT2,TF>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.size() - index, "Invalid vector size" );

   return tryMultAssign( lhs.vector_, ~rhs, lhs.offset_ + index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to a subvector.
// \ingroup subvector
//
// \param lhs The target left-hand side subvector.
// \param rhs The right-hand side vector divisor.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1    // Type of the vector
        , bool AF         // Alignment flag
        , bool TF         // Transpose flag
        , bool DF         // Density flag
        , typename VT2 >  // Type of the right-hand side vector
inline bool tryDivAssign( const Subvector<VT1,AF,TF,DF>& lhs, const Vector<VT2,TF>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.size() - index, "Invalid vector size" );

   return tryDivAssign( lhs.vector_, ~rhs, lhs.offset_ + index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given subvector.
// \ingroup subvector
//
// \param sv The subvector to be derestricted.
// \return Subvector without access restrictions.
//
// This function removes all restrictions on the data access to the given subvector. It returns a
// subvector that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename VT  // Type of the vector
        , bool AF      // Alignment flag
        , bool TF      // Transpose flag
        , bool DF >    // Density flag
inline DerestrictTrait_< Subvector<VT,AF,TF,DF> > derestrict( Subvector<VT,AF,TF,DF>& sv )
{
   typedef DerestrictTrait_< Subvector<VT,AF,TF,DF> >  ReturnType;
   return ReturnType( derestrict( sv.vector_ ), sv.offset_, sv.size_ );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISRESTRICTED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF, bool TF, bool DF >
struct IsRestricted< Subvector<VT,AF,TF,DF> >
   : public BoolConstant< IsRestricted<VT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DERESTRICTTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF, bool TF, bool DF >
struct DerestrictTrait< Subvector<VT,AF,TF,DF> >
{
   using Type = Subvector< RemoveReference_< DerestrictTrait_<VT> >, AF, TF, DF >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HASCONSTDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF, bool TF >
struct HasConstDataAccess< Subvector<VT,AF,TF,true> >
   : public BoolConstant< HasConstDataAccess<VT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HASMUTABLEDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF, bool TF >
struct HasMutableDataAccess< Subvector<VT,AF,TF,true> >
   : public BoolConstant< HasMutableDataAccess<VT>::value >
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
template< typename VT, bool TF >
struct IsAligned< Subvector<VT,aligned,TF,true> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ADDTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF, bool TF, bool DF, typename T >
struct AddTrait< Subvector<VT,AF,TF,DF>, T >
{
   using Type = AddTrait_< SubvectorTrait_<VT>, T >;
};

template< typename T, typename VT, bool AF, bool TF, bool DF >
struct AddTrait< T, Subvector<VT,AF,TF,DF> >
{
   using Type = AddTrait_< T, SubvectorTrait_<VT> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF, bool TF, bool DF, typename T >
struct SubTrait< Subvector<VT,AF,TF,DF>, T >
{
   using Type = SubTrait_< SubvectorTrait_<VT>, T >;
};

template< typename T, typename VT, bool AF, bool TF, bool DF >
struct SubTrait< T, Subvector<VT,AF,TF,DF> >
{
   using Type = SubTrait_< T, SubvectorTrait_<VT> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MULTTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF, bool TF, bool DF, typename T >
struct MultTrait< Subvector<VT,AF,TF,DF>, T >
{
   using Type = MultTrait_< SubvectorTrait_<VT>, T >;
};

template< typename T, typename VT, bool AF, bool TF, bool DF >
struct MultTrait< T, Subvector<VT,AF,TF,DF> >
{
   using Type = MultTrait_< T, SubvectorTrait_<VT> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CROSSTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF, bool TF, bool DF, typename T >
struct CrossTrait< Subvector<VT,AF,TF,DF>, T >
{
   using Type = CrossTrait_< SubvectorTrait_<VT>, T >;
};

template< typename T, typename VT, bool AF, bool TF, bool DF >
struct CrossTrait< T, Subvector<VT,AF,TF,DF> >
{
   using Type = CrossTrait_< T, SubvectorTrait_<VT> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DIVTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF, bool TF, bool DF, typename T >
struct DivTrait< Subvector<VT,AF,TF,DF>, T >
{
   using Type = DivTrait_< SubvectorTrait_<VT>, T >;
};

template< typename T, typename VT, bool AF, bool TF, bool DF >
struct DivTrait< T, Subvector<VT,AF,TF,DF> >
{
   using Type = DivTrait_< T, SubvectorTrait_<VT> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBVECTORTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF, bool TF, bool DF >
struct SubvectorTrait< Subvector<VT,AF,TF,DF> >
{
   using Type = SubvectorTrait_< ResultType_< Subvector<VT,AF,TF,DF> > >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBVECTOREXPRTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF1, bool TF, bool DF, bool AF2 >
struct SubvectorExprTrait< Subvector<VT,AF1,TF,DF>, AF2 >
{
   using Type = Subvector<VT,AF2,TF,DF>;
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF1, bool TF, bool DF, bool AF2 >
struct SubvectorExprTrait< const Subvector<VT,AF1,TF,DF>, AF2 >
{
   using Type = Subvector<VT,AF2,TF,DF>;
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF1, bool TF, bool DF, bool AF2 >
struct SubvectorExprTrait< volatile Subvector<VT,AF1,TF,DF>, AF2 >
{
   using Type = Subvector<VT,AF2,TF,DF>;
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT, bool AF1, bool TF, bool DF, bool AF2 >
struct SubvectorExprTrait< const volatile Subvector<VT,AF1,TF,DF>, AF2 >
{
   using Type = Subvector<VT,AF2,TF,DF>;
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2, bool TF, bool AF >
struct SubvectorExprTrait< DVecDVecCrossExpr<VT1,VT2,TF>, AF >
{
 public:
   //**********************************************************************************************
   using Type = Subvector< DVecDVecCrossExpr<VT1,VT2,TF>, unaligned, TF, true >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2, bool TF, bool AF >
struct SubvectorExprTrait< DVecSVecCrossExpr<VT1,VT2,TF>, AF >
{
 public:
   //**********************************************************************************************
   using Type = Subvector< DVecSVecCrossExpr<VT1,VT2,TF>, unaligned, TF, true >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2, bool TF, bool AF >
struct SubvectorExprTrait< SVecDVecCrossExpr<VT1,VT2,TF>, AF >
{
 public:
   //**********************************************************************************************
   using Type = Subvector< SVecDVecCrossExpr<VT1,VT2,TF>, unaligned, TF, true >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT1, typename VT2, bool TF, bool AF >
struct SubvectorExprTrait< SVecSVecCrossExpr<VT1,VT2,TF>, AF >
{
 public:
   //**********************************************************************************************
   using Type = Subvector< SVecSVecCrossExpr<VT1,VT2,TF>, unaligned, TF, true >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
