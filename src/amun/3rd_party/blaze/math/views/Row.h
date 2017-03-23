//=================================================================================================
/*!
//  \file blaze/math/views/Row.h
//  \brief Header file for the implementation of the Row view
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

#ifndef _BLAZE_MATH_VIEWS_ROW_H_
#define _BLAZE_MATH_VIEWS_ROW_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Matrix.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/CrossTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/RowExprTrait.h>
#include <blaze/math/traits/RowTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/traits/SubvectorTrait.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsMatEvalExpr.h>
#include <blaze/math/typetraits/IsMatForEachExpr.h>
#include <blaze/math/typetraits/IsMatMatAddExpr.h>
#include <blaze/math/typetraits/IsMatMatMultExpr.h>
#include <blaze/math/typetraits/IsMatMatSubExpr.h>
#include <blaze/math/typetraits/IsMatScalarDivExpr.h>
#include <blaze/math/typetraits/IsMatScalarMultExpr.h>
#include <blaze/math/typetraits/IsMatSerialExpr.h>
#include <blaze/math/typetraits/IsMatTransExpr.h>
#include <blaze/math/typetraits/IsOpposedView.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsTransExpr.h>
#include <blaze/math/typetraits/IsVecTVecMultExpr.h>
#include <blaze/math/views/row/BaseTemplate.h>
#include <blaze/math/views/row/Dense.h>
#include <blaze/math/views/row/Sparse.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/logging/FunctionTrace.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Creating a view on a specific row of the given matrix.
// \ingroup views
//
// \param matrix The matrix containing the row.
// \param index The index of the row.
// \return View on the specified row of the matrix.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given matrix.

   \code
   using blaze::rowMajor;

   typedef blaze::DynamicMatrix<double,rowMajor>     DenseMatrix;
   typedef blaze::CompressedMatrix<double,rowMajor>  SparseMatrix;

   DenseMatrix D;
   SparseMatrix S;
   // ... Resizing and initialization

   // Creating a view on the 3rd row of the dense matrix D
   blaze::Row<DenseMatrix> = row( D, 3UL );

   // Creating a view on the 4th row of the sparse matrix S
   blaze::Row<SparseMatrix> = row( S, 4UL );
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline DisableIf_< Or< IsComputation<MT>, IsTransExpr<MT> >, RowExprTrait_<MT> >
   row( Matrix<MT,SO>& matrix, size_t index )
{
   BLAZE_FUNCTION_TRACE;

   return RowExprTrait_<MT>( ~matrix, index );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific row of the given constant matrix.
// \ingroup views
//
// \param matrix The constant matrix containing the row.
// \param index The index of the row.
// \return View on the specified row of the matrix.
// \exception std::invalid_argument Invalid row access index.
//
// This function returns an expression representing the specified row of the given matrix.

   \code
   using blaze::rowMajor;

   typedef blaze::DynamicMatrix<double,rowMajor>     DenseMatrix;
   typedef blaze::CompressedMatrix<double,rowMajor>  SparseMatrix;

   DenseMatrix D;
   SparseMatrix S;
   // ... Resizing and initialization

   // Creating a view on the 3rd row of the dense matrix D
   blaze::Row<DenseMatrix> = row( D, 3UL );

   // Creating a view on the 4th row of the sparse matrix S
   blaze::Row<SparseMatrix> = row( S, 4UL );
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline DisableIf_< Or< IsComputation<MT>, IsTransExpr<MT> >, RowExprTrait_<const MT> >
   row( const Matrix<MT,SO>& matrix, size_t index )
{
   BLAZE_FUNCTION_TRACE;

   return RowExprTrait_<const MT>( ~matrix, index );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given matrix/matrix addition.
// \ingroup views
//
// \param matrix The constant matrix/matrix addition.
// \param index The index of the row.
// \return View on the specified row of the addition.
//
// This function returns an expression representing the specified row of the given matrix/matrix
// addition.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatMatAddExpr<MT>, RowExprTrait_<MT> >
   row( const Matrix<MT,SO>& matrix, size_t index )
{
   BLAZE_FUNCTION_TRACE;

   return row( (~matrix).leftOperand(), index ) + row( (~matrix).rightOperand(), index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given matrix/matrix subtraction.
// \ingroup views
//
// \param matrix The constant matrix/matrix subtraction.
// \param index The index of the row.
// \return View on the specified row of the subtraction.
//
// This function returns an expression representing the specified row of the given matrix/matrix
// subtraction.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatMatSubExpr<MT>, RowExprTrait_<MT> >
   row( const Matrix<MT,SO>& matrix, size_t index )
{
   BLAZE_FUNCTION_TRACE;

   return row( (~matrix).leftOperand(), index ) - row( (~matrix).rightOperand(), index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given matrix/matrix multiplication.
// \ingroup views
//
// \param matrix The constant matrix/matrix multiplication.
// \param index The index of the row.
// \return View on the specified row of the multiplication.
//
// This function returns an expression representing the specified row of the given matrix/matrix
// multiplication.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatMatMultExpr<MT>, RowExprTrait_<MT> >
   row( const Matrix<MT,SO>& matrix, size_t index )
{
   BLAZE_FUNCTION_TRACE;

   return row( (~matrix).leftOperand(), index ) * (~matrix).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given outer product.
// \ingroup views
//
// \param matrix The constant outer product.
// \param index The index of the row.
// \return View on the specified row of the outer product.
//
// This function returns an expression representing the specified row of the given outer product.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsVecTVecMultExpr<MT>, RowExprTrait_<MT> >
   row( const Matrix<MT,SO>& matrix, size_t index )
{
   BLAZE_FUNCTION_TRACE;

   return (~matrix).leftOperand()[index] * (~matrix).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given matrix/scalar multiplication.
// \ingroup views
//
// \param matrix The constant matrix/scalar multiplication.
// \param index The index of the row.
// \return View on the specified row of the multiplication.
//
// This function returns an expression representing the specified row of the given matrix/scalar
// multiplication.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatScalarMultExpr<MT>, RowExprTrait_<MT> >
   row( const Matrix<MT,SO>& matrix, size_t index )
{
   BLAZE_FUNCTION_TRACE;

   return row( (~matrix).leftOperand(), index ) * (~matrix).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given matrix/scalar division.
// \ingroup views
//
// \param matrix The constant matrix/scalar division.
// \param index The index of the row.
// \return View on the specified row of the division.
//
// This function returns an expression representing the specified row of the given matrix/scalar
// division.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatScalarDivExpr<MT>, RowExprTrait_<MT> >
   row( const Matrix<MT,SO>& matrix, size_t index )
{
   BLAZE_FUNCTION_TRACE;

   return row( (~matrix).leftOperand(), index ) / (~matrix).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given matrix custom operation.
// \ingroup views
//
// \param matrix The constant matrix custom operation.
// \param index The index of the row.
// \return View on the specified row of the custom operation.
//
// This function returns an expression representing the specified row of the given matrix
// custom operation.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatForEachExpr<MT>, RowExprTrait_<MT> >
   row( const Matrix<MT,SO>& matrix, size_t index )
{
   BLAZE_FUNCTION_TRACE;

   return forEach( row( (~matrix).operand(), index ), (~matrix).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given matrix evaluation operation.
// \ingroup views
//
// \param matrix The constant matrix evaluation operation.
// \param index The index of the row.
// \return View on the specified row of the evaluation operation.
//
// This function returns an expression representing the specified row of the given matrix
// evaluation operation.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatEvalExpr<MT>, RowExprTrait_<MT> >
   row( const Matrix<MT,SO>& matrix, size_t index )
{
   BLAZE_FUNCTION_TRACE;

   return eval( row( (~matrix).operand(), index ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given matrix serialization operation.
// \ingroup views
//
// \param matrix The constant matrix serialization operation.
// \param index The index of the row.
// \return View on the specified row of the serialization operation.
//
// This function returns an expression representing the specified row of the given matrix
// serialization operation.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatSerialExpr<MT>, RowExprTrait_<MT> >
   row( const Matrix<MT,SO>& matrix, size_t index )
{
   BLAZE_FUNCTION_TRACE;

   return serial( row( (~matrix).operand(), index ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific row of the given matrix transpose operation.
// \ingroup views
//
// \param matrix The constant matrix transpose operation.
// \param index The index of the row.
// \return View on the specified row of the transpose operation.
//
// This function returns an expression representing the specified row of the given matrix
// transpose operation.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatTransExpr<MT>, RowExprTrait_<MT> >
   row( const Matrix<MT,SO>& matrix, size_t index )
{
   BLAZE_FUNCTION_TRACE;

   return trans( column( (~matrix).operand(), index ) );
}
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  ROW OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Row operators */
//@{
template< typename MT, bool SO, bool DF, bool SF >
inline void reset( Row<MT,SO,DF,SF>& row );

template< typename MT, bool SO, bool DF, bool SF >
inline void clear( Row<MT,SO,DF,SF>& row );

template< typename MT, bool SO, bool DF, bool SF >
inline bool isDefault( const Row<MT,SO,DF,SF>& row );

template< typename MT, bool SO, bool DF, bool SF >
inline bool isIntact( const Row<MT,SO,DF,SF>& row ) noexcept;

template< typename MT, bool SO, bool DF, bool SF >
inline bool isSame( const Row<MT,SO,DF,SF>& a, const Row<MT,SO,DF,SF>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given row.
// \ingroup row
//
// \param row The row to be resetted.
// \return void
*/
template< typename MT  // Type of the matrix
        , bool SO      // Storage order
        , bool DF      // Density flag
        , bool SF >    // Symmetry flag
inline void reset( Row<MT,SO,DF,SF>& row )
{
   row.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given row.
// \ingroup row
//
// \param row The row to be cleared.
// \return void
//
// Clearing a row is equivalent to resetting it via the reset() function.
*/
template< typename MT  // Type of the matrix
        , bool SO      // Storage order
        , bool DF      // Density flag
        , bool SF >    // Symmetry flag
inline void clear( Row<MT,SO,DF,SF>& row )
{
   row.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given row is in default state.
// \ingroup row
//
// \param row The row to be tested for its default state.
// \return \a true in case the given row is component-wise zero, \a false otherwise.
//
// This function checks whether the row is in default state. For instance, in case the row
// is instantiated for a built-in integral or floating point data type, the function returns
// \a true in case all row elements are 0 and \a false in case any row element is not 0. The
// following example demonstrates the use of the \a isDefault function:

   \code
   blaze::DynamicMatrix<int,rowMajor> A;
   // ... Resizing and initialization
   if( isDefault( row( A, 0UL ) ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool SO      // Storage order
        , bool DF      // Density flag
        , bool SF >    // Symmetry flag
inline bool isDefault( const Row<MT,SO,DF,SF>& row )
{
   for( size_t i=0UL; i<row.size(); ++i )
      if( !isDefault( row[i] ) ) return false;
   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given sparse row is in default state.
// \ingroup sparse_row
//
// \param row The sparse row to be tested for its default state.
// \return \a true in case the given row is component-wise zero, \a false otherwise.
//
// This function checks whether the sparse row is in default state. For instance, in case the
// row is instantiated for a built-in integral or floating point data type, the function returns
// \a true in case all row elements are 0 and \a false in case any vector element is not 0. The
// following example demonstrates the use of the \a isDefault function:

   \code
   blaze::CompressedMatrix<double,rowMajor> A;
   // ... Resizing and initialization
   if( isDefault( row( A, 0UL ) ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool SO      // Storage order
        , bool SF >    // Symmetry flag
inline bool isDefault( const Row<MT,SO,false,SF>& row )
{
   typedef ConstIterator_< Row<MT,SO,false,SF> >  ConstIterator;

   const ConstIterator end( row.end() );
   for( ConstIterator element=row.begin(); element!=end; ++element )
      if( !isDefault( element->value() ) ) return false;
   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given row are intact.
// \ingroup row
//
// \param row The row to be tested.
// \return \a true in case the given row's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the row are intact, i.e. if its state is valid.
// In case the invariants are intact, the function returns \a true, else it will return \a false.
// The following example demonstrates the use of the \a isIntact() function:

   \code
   blaze::DynamicMatrix<int,rowMajor> A;
   // ... Resizing and initialization
   if( isIntact( row( A, 0UL ) ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool SO      // Storage order
        , bool DF      // Density flag
        , bool SF >    // Symmetry flag
inline bool isIntact( const Row<MT,SO,DF,SF>& row ) noexcept
{
   return ( row.row_ <= row.matrix_.rows() &&
            isIntact( row.matrix_ ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the two given rows represent the same observable state.
// \ingroup row
//
// \param a The first row to be tested for its state.
// \param b The second row to be tested for its state.
// \return \a true in case the two rows share a state, \a false otherwise.
//
// This overload of the isSame function tests if the two given rows refer to exactly the same
// range of the same matrix. In case both rows represent the same observable state, the function
// returns \a true, otherwise it returns \a false.
*/
template< typename MT  // Type of the matrix
        , bool SO      // Storage order
        , bool DF      // Density flag
        , bool SF >    // Symmetry flag
inline bool isSame( const Row<MT,SO,DF,SF>& a, const Row<MT,SO,DF,SF>& b ) noexcept
{
   return ( isSame( a.matrix_, b.matrix_ ) && ( a.row_ == b.row_ ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a vector to a row.
// \ingroup row
//
// \param lhs The target left-hand side row.
// \param rhs The right-hand side vector to be assigned.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the matrix
        , bool SO        // Storage order
        , bool DF        // Density flag
        , bool SF        // Symmetry flag
        , typename VT >  // Type of the right-hand side vector
inline bool tryAssign( const Row<MT,SO,DF,SF>& lhs, const Vector<VT,true>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.size() - index, "Invalid vector size" );

   return tryAssign( lhs.matrix_, ~rhs, lhs.row_, index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to a row.
// \ingroup row
//
// \param lhs The target left-hand side row.
// \param rhs The right-hand side vector to be added.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the matrix
        , bool SO        // Storage order
        , bool DF        // Density flag
        , bool SF        // Symmetry flag
        , typename VT >  // Type of the right-hand side vector
inline bool tryAddAssign( const Row<MT,SO,DF,SF>& lhs, const Vector<VT,true>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.size() - index, "Invalid vector size" );

   return tryAddAssign( lhs.matrix_, ~rhs, lhs.row_, index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to a row.
// \ingroup row
//
// \param lhs The target left-hand side row.
// \param rhs The right-hand side vector to be subtracted.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the matrix
        , bool SO        // Storage order
        , bool DF        // Density flag
        , bool SF        // Symmetry flag
        , typename VT >  // Type of the right-hand side vector
inline bool trySubAssign( const Row<MT,SO,DF,SF>& lhs, const Vector<VT,true>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.size() - index, "Invalid vector size" );

   return trySubAssign( lhs.matrix_, ~rhs, lhs.row_, index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to a row.
// \ingroup row
//
// \param lhs The target left-hand side row.
// \param rhs The right-hand side vector to be multiplied.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the matrix
        , bool SO        // Storage order
        , bool DF        // Density flag
        , bool SF        // Symmetry flag
        , typename VT >  // Type of the right-hand side vector
inline bool tryMultAssign( const Row<MT,SO,DF,SF>& lhs, const Vector<VT,true>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.size() - index, "Invalid vector size" );

   return tryMultAssign( lhs.matrix_, ~rhs, lhs.row_, index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to a row.
// \ingroup row
//
// \param lhs The target left-hand side row.
// \param rhs The right-hand side vector divisor.
// \param index The index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the matrix
        , bool SO        // Storage order
        , bool DF        // Density flag
        , bool SF        // Symmetry flag
        , typename VT >  // Type of the right-hand side vector
inline bool tryDivAssign( const Row<MT,SO,DF,SF>& lhs, const Vector<VT,true>& rhs, size_t index )
{
   BLAZE_INTERNAL_ASSERT( index <= lhs.size(), "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.size() - index, "Invalid vector size" );

   return tryDivAssign( lhs.matrix_, ~rhs, lhs.row_, index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given row.
// \ingroup row
//
// \param row The row to be derestricted.
// \return Row without access restrictions.
//
// This function removes all restrictions on the data access to the given row. It returns a row
// object that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the matrix
        , bool SO      // Storage order
        , bool DF      // Density flag
        , bool SF >    // Symmetry flag
inline DerestrictTrait_< Row<MT,SO,DF,SF> > derestrict( Row<MT,SO,DF,SF>& row )
{
   typedef DerestrictTrait_< Row<MT,SO,DF,SF> >  ReturnType;
   return ReturnType( derestrict( row.matrix_ ), row.row_ );
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
template< typename MT, bool SO, bool DF, bool SF >
struct IsRestricted< Row<MT,SO,DF,SF> >
   : public BoolConstant< IsRestricted<MT>::value >
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
template< typename MT, bool SO, bool DF, bool SF >
struct DerestrictTrait< Row<MT,SO,DF,SF> >
{
   using Type = Row< RemoveReference_< DerestrictTrait_<MT> > >;
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
template< typename MT, bool SO, bool SF >
struct HasConstDataAccess< Row<MT,SO,true,SF> >
   : public BoolConstant< HasConstDataAccess<MT>::value >
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
template< typename MT, bool SO, bool SF >
struct HasMutableDataAccess< Row<MT,SO,true,SF> >
   : public BoolConstant< HasMutableDataAccess<MT>::value >
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
template< typename MT, bool SO, bool SF >
struct IsAligned< Row<MT,SO,true,SF> >
   : public BoolConstant< And< IsAligned<MT>, Or< IsRowMajorMatrix<MT>, IsSymmetric<MT> > >::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISPADDED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool SF >
struct IsPadded< Row<MT,SO,true,SF> >
   : public BoolConstant< And< IsPadded<MT>, Or< IsRowMajorMatrix<MT>, IsSymmetric<MT> > >::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISOPPOSEDVIEW SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool DF >
struct IsOpposedView< Row<MT,false,DF,false> >
   : public TrueType
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
template< typename MT, bool SO, bool DF, bool SF, typename T >
struct AddTrait< Row<MT,SO,DF,SF>, T >
{
   using Type = AddTrait_< RowTrait_<MT>, T >;
};

template< typename T, typename MT, bool SO, bool DF, bool SF >
struct AddTrait< T, Row<MT,SO,DF,SF> >
{
   using Type = AddTrait_< T, RowTrait_<MT> >;
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
template< typename MT, bool SO, bool DF, bool SF, typename T >
struct SubTrait< Row<MT,SO,DF,SF>, T >
{
   using Type = SubTrait_< RowTrait_<MT>, T >;
};

template< typename T, typename MT, bool SO, bool DF, bool SF >
struct SubTrait< T, Row<MT,SO,DF,SF> >
{
   using Type = SubTrait_< T, RowTrait_<MT> >;
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
template< typename MT, bool SO, bool DF, bool SF, typename T >
struct MultTrait< Row<MT,SO,DF,SF>, T >
{
   using Type = MultTrait_< RowTrait_<MT>, T >;
};

template< typename T, typename MT, bool SO, bool DF, bool SF >
struct MultTrait< T, Row<MT,SO,DF,SF> >
{
   using Type = MultTrait_< T, RowTrait_<MT> >;
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
template< typename MT, bool SO, bool DF, bool SF, typename T >
struct CrossTrait< Row<MT,SO,DF,SF>, T >
{
   using Type = CrossTrait_< RowTrait_<MT>, T >;
};

template< typename T, typename MT, bool SO, bool DF, bool SF >
struct CrossTrait< T, Row<MT,SO,DF,SF> >
{
   using Type = CrossTrait_< T, RowTrait_<MT> >;
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
template< typename MT, bool SO, bool DF, bool SF, typename T >
struct DivTrait< Row<MT,SO,DF,SF>, T >
{
   using Type = DivTrait_< RowTrait_<MT>, T >;
};

template< typename T, typename MT, bool SO, bool DF, bool SF >
struct DivTrait< T, Row<MT,SO,DF,SF> >
{
   using Type = DivTrait_< T, RowTrait_<MT> >;
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
template< typename MT, bool SO, bool DF, bool SF >
struct SubvectorTrait< Row<MT,SO,DF,SF> >
{
   using Type = SubvectorTrait_< ResultType_< Row<MT,SO,DF,SF> > >;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
