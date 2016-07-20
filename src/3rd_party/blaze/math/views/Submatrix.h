//=================================================================================================
/*!
//  \file blaze/math/views/Submatrix.h
//  \brief Header file for the implementation of the Submatrix view
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

#ifndef _BLAZE_MATH_VIEWS_SUBMATRIX_H_
#define _BLAZE_MATH_VIEWS_SUBMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/constraints/MutableDataAccess.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/expressions/Matrix.h>
#include <blaze/math/Functions.h>
#include <blaze/math/InversionFlag.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/ColumnTrait.h>
#include <blaze/math/traits/DerestrictTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/RowTrait.h>
#include <blaze/math/traits/SubmatrixExprTrait.h>
#include <blaze/math/traits/SubmatrixTrait.h>
#include <blaze/math/traits/SubvectorExprTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsMatEvalExpr.h>
#include <blaze/math/typetraits/IsMatForEachExpr.h>
#include <blaze/math/typetraits/IsMatMatAddExpr.h>
#include <blaze/math/typetraits/IsMatMatMultExpr.h>
#include <blaze/math/typetraits/IsMatMatSubExpr.h>
#include <blaze/math/typetraits/IsMatScalarDivExpr.h>
#include <blaze/math/typetraits/IsMatScalarMultExpr.h>
#include <blaze/math/typetraits/IsMatSerialExpr.h>
#include <blaze/math/typetraits/IsMatTransExpr.h>
#include <blaze/math/typetraits/IsMatVecMultExpr.h>
#include <blaze/math/typetraits/IsRestricted.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsTransExpr.h>
#include <blaze/math/typetraits/IsTVecMatMultExpr.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/IsVecTVecMultExpr.h>
#include <blaze/math/views/submatrix/BaseTemplate.h>
#include <blaze/math/views/submatrix/Dense.h>
#include <blaze/math/views/submatrix/Sparse.h>
#include <blaze/math/views/Subvector.h>
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
/*!\brief Creating a view on a specific submatrix of the given matrix.
// \ingroup views
//
// \param matrix The matrix containing the submatrix.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specific submatrix of the matrix.
// \exception std::invalid_argument Invalid submatrix specification.
//
// This function returns an expression representing the specified submatrix of the given matrix.
// The following example demonstrates the creation of a dense and sparse submatrix:

   \code
   using blaze::rowMajor;
   using blaze::columnMajor;

   typedef blaze::DynamicMatrix<double,rowMajor>     DenseMatrix;
   typedef blaze::CompressedMatrix<int,columnMajor>  SparseMatrix;

   DenseMatrix  D;
   SparseMatrix S;
   // ... Resizing and initialization

   // Creating a dense submatrix of size 8x4, starting in row 0 and column 16
   blaze::Submatrix<DenseMatrix> dsm = submatrix( D, 0UL, 16UL, 8UL, 4UL );

   // Creating a sparse submatrix of size 7x3, starting in row 2 and column 4
   blaze::Submatrix<SparseMatrix> ssm = submatrix( S, 2UL, 4UL, 7UL, 3UL );
   \endcode

// In case the submatrix is not properly specified (i.e. if the specified row or column is larger
// than the total number of rows or columns of the given matrix or the submatrix is specified
// beyond the number of rows or columns of the matrix) a \a std::invalid_argument exception is
// thrown.
//
// Please note that this function creates an unaligned dense or sparse submatrix. For instance,
// the creation of the dense submatrix is equivalent to the following three function calls:

   \code
   blaze::Submatrix<DenseMatrix>           dsm = submatrix<unaligned>( D, 0UL, 16UL, 8UL, 4UL );
   blaze::Submatrix<DenseMatrix,unaligned> dsm = submatrix           ( D, 0UL, 16UL, 8UL, 4UL );
   blaze::Submatrix<DenseMatrix,unaligned> dsm = submatrix<unaligned>( D, 0UL, 16UL, 8UL, 4UL );
   \endcode

// In contrast to unaligned submatrices, which provide full flexibility, aligned submatrices pose
// additional alignment restrictions. However, especially in case of dense submatrices this may
// result in considerable performance improvements. In order to create an aligned submatrix the
// following function call has to be used:

   \code
   blaze::Submatrix<DenseMatrix,aligned> dsm = submatrix<aligned>( D, 0UL, 16UL, 8UL, 4UL );
   \endcode

// Note however that in this case the given arguments \a row, \a columns, \a m, and \a n are
// subject to additional checks to guarantee proper alignment.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
inline SubmatrixExprTrait_<MT,unaligned>
   submatrix( Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   return submatrix<unaligned>( ~matrix, row, column, m, n );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific submatrix of the given matrix.
// \ingroup views
//
// \param matrix The matrix containing the submatrix.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specific submatrix of the matrix.
// \exception std::invalid_argument Invalid submatrix specification.
//
// This function returns an expression representing the specified submatrix of the given matrix.
// The following example demonstrates the creation of a dense and sparse submatrix:

   \code
   using blaze::rowMajor;
   using blaze::columnMajor;

   typedef blaze::DynamicMatrix<double,rowMajor>     DenseMatrix;
   typedef blaze::CompressedMatrix<int,columnMajor>  SparseMatrix;

   DenseMatrix  D;
   SparseMatrix S;
   // ... Resizing and initialization

   // Creating a dense submatrix of size 8x4, starting in row 0 and column 16
   blaze::Submatrix<DenseMatrix> dsm = submatrix( D, 0UL, 16UL, 8UL, 4UL );

   // Creating a sparse submatrix of size 7x3, starting in row 2 and column 4
   blaze::Submatrix<SparseMatrix> ssm = submatrix( S, 2UL, 4UL, 7UL, 3UL );
   \endcode

// In case the submatrix is not properly specified (i.e. if the specified row or column is larger
// than the total number of rows or columns of the given matrix or the submatrix is specified
// beyond the number of rows or columns of the matrix) a \a std::invalid_argument exception is
// thrown.
//
// Please note that this function creates an unaligned dense or sparse submatrix. For instance,
// the creation of the dense submatrix is equivalent to the following three function calls:

   \code
   blaze::Submatrix<DenseMatrix>           dsm = submatrix<unaligned>( D, 0UL, 16UL, 8UL, 4UL );
   blaze::Submatrix<DenseMatrix,unaligned> dsm = submatrix           ( D, 0UL, 16UL, 8UL, 4UL );
   blaze::Submatrix<DenseMatrix,unaligned> dsm = submatrix<unaligned>( D, 0UL, 16UL, 8UL, 4UL );
   \endcode

// In contrast to unaligned submatrices, which provide full flexibility, aligned submatrices pose
// additional alignment restrictions. However, especially in case of dense submatrices this may
// result in considerable performance improvements. In order to create an aligned submatrix the
// following function call has to be used:

   \code
   blaze::Submatrix<DenseMatrix,aligned> dsm = submatrix<aligned>( D, 0UL, 16UL, 8UL, 4UL );
   \endcode

// Note however that in this case the given arguments \a row, \a columns, \a m, and \a n are
// subject to additional checks to guarantee proper alignment.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
inline SubmatrixExprTrait_<const MT,unaligned>
   submatrix( const Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   return submatrix<unaligned>( ~matrix, row, column, m, n );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific submatrix of the given matrix.
// \ingroup views
//
// \param matrix The matrix containing the submatrix.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specific submatrix of the matrix.
// \exception std::invalid_argument Invalid submatrix specification.
//
// This function returns an expression representing an aligned or unaligned submatrix of the
// given dense or sparse matrix, based on the specified alignment flag \a AF. The following
// example demonstrates the creation of both an aligned and unaligned submatrix:

   \code
   using blaze::rowMajor;
   using blaze::columnMajor;

   typedef blaze::DynamicMatrix<double,rowMajor>     DenseMatrix;
   typedef blaze::CompressedMatrix<int,columnMajor>  SparseMatrix;

   DenseMatrix  D;
   SparseMatrix S;
   // ... Resizing and initialization

   // Creating an aligned dense submatrix of size 8x4, starting in row 0 and column 16
   blaze::Submatrix<DenseMatrix,aligned> dsm = submatrix<aligned>( D, 0UL, 16UL, 8UL, 4UL );

   // Creating an unaligned sparse submatrix of size 7x3, starting in row 2 and column 4
   blaze::Submatrix<SparseMatrix,unaligned> ssm = submatrix<unaligned>( S, 2UL, 4UL, 7UL, 3UL );
   \endcode

// In case the submatrix is not properly specified (i.e. if the specified row or column is larger
// than the total number of rows or columns of the given matrix or the submatrix is specified
// beyond the number of rows or columns of the matrix) a \a std::invalid_argument exception is
// thrown.
//
// In contrast to unaligned submatrices, which provide full flexibility, aligned submatrices pose
// additional alignment restrictions and the given \a row, and \a column arguments are subject to
// additional checks to guarantee proper alignment. However, especially in case of dense submatrices
// this may result in considerable performance improvements.
//
// The alignment restrictions refer to system dependent address restrictions for the used element
// type and the available vectorization mode (SSE, AVX, ...). In order to be properly aligned the
// first element of each row/column of the submatrix must be aligned. The following source code
// gives some examples for a double precision row-major dynamic matrix, assuming that padding is
// enabled and that AVX is available, which packs 4 \c double values into a SIMD vector:

   \code
   using blaze::rowMajor;

   typedef blaze::DynamicMatrix<double,rowMajor>  MatrixType;
   typedef blaze::Submatrix<MatrixType,aligned>   SubmatrixType;

   MatrixType D( 13UL, 17UL );
   // ... Resizing and initialization

   // OK: Starts at position (0,0), i.e. the first element of each row is aligned (due to padding)
   SubmatrixType dsm1 = submatrix<aligned>( D, 0UL, 0UL, 7UL, 11UL );

   // OK: First column is a multiple of 4, i.e. the first element of each row is aligned (due to padding)
   SubmatrixType dsm2 = submatrix<aligned>( D, 3UL, 12UL, 8UL, 16UL );

   // OK: First column is a multiple of 4 and the submatrix includes the last row and column
   SubmatrixType dsm3 = submatrix<aligned>( D, 4UL, 0UL, 9UL, 17UL );

   // Error: First column is not a multiple of 4, i.e. the first element is not aligned
   SubmatrixType dsm4 = submatrix<aligned>( D, 2UL, 3UL, 12UL, 12UL );
   \endcode

// In case any alignment restrictions are violated, a \a std::invalid_argument exception is thrown.
*/
template< bool AF      // Alignment flag
        , typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
inline DisableIf_< Or< IsComputation<MT>, IsTransExpr<MT> >, SubmatrixExprTrait_<MT,AF> >
   submatrix( Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   return SubmatrixExprTrait_<MT,AF>( ~matrix, row, column, m, n );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a view on a specific submatrix of the given matrix.
// \ingroup views
//
// \param matrix The matrix containing the submatrix.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specific submatrix of the matrix.
// \exception std::invalid_argument Invalid submatrix specification.
//
// This function returns an expression representing an aligned or unaligned submatrix of the
// given dense or sparse matrix, based on the specified alignment flag \a AF. The following
// example demonstrates the creation of both an aligned and unaligned submatrix:

   \code
   using blaze::rowMajor;
   using blaze::columnMajor;

   typedef blaze::DynamicMatrix<double,rowMajor>     DenseMatrix;
   typedef blaze::CompressedMatrix<int,columnMajor>  SparseMatrix;

   DenseMatrix  D;
   SparseMatrix S;
   // ... Resizing and initialization

   // Creating an aligned dense submatrix of size 8x4, starting in row 0 and column 16
   blaze::Submatrix<DenseMatrix,aligned> dsm = submatrix<aligned>( D, 0UL, 16UL, 8UL, 4UL );

   // Creating an unaligned sparse submatrix of size 7x3, starting in row 2 and column 4
   blaze::Submatrix<SparseMatrix,unaligned> ssm = submatrix<unaligned>( S, 2UL, 4UL, 7UL, 3UL );
   \endcode

// In case the submatrix is not properly specified (i.e. if the specified row or column is larger
// than the total number of rows or columns of the given matrix or the submatrix is specified
// beyond the number of rows or columns of the matrix) a \a std::invalid_argument exception is
// thrown.
//
// In contrast to unaligned submatrices, which provide full flexibility, aligned submatrices pose
// additional alignment restrictions and the given \a row, and \a column arguments are subject to
// additional checks to guarantee proper alignment. However, especially in case of dense submatrices
// this may result in considerable performance improvements.
//
// The alignment restrictions refer to system dependent address restrictions for the used element
// type and the available vectorization mode (SSE, AVX, ...). In order to be properly aligned the
// first element of each row/column of the submatrix must be aligned. The following source code
// gives some examples for a double precision row-major dynamic matrix, assuming that padding is
// enabled and that AVX is available, which packs 4 \c double values into a SIMD vector:

   \code
   using blaze::rowMajor;

   typedef blaze::DynamicMatrix<double,rowMajor>  MatrixType;
   typedef blaze::Submatrix<MatrixType,aligned>   SubmatrixType;

   MatrixType D( 13UL, 17UL );
   // ... Resizing and initialization

   // OK: Starts at position (0,0), i.e. the first element of each row is aligned (due to padding)
   SubmatrixType dsm1 = submatrix<aligned>( D, 0UL, 0UL, 7UL, 11UL );

   // OK: First column is a multiple of 4, i.e. the first element of each row is aligned (due to padding)
   SubmatrixType dsm2 = submatrix<aligned>( D, 3UL, 12UL, 8UL, 16UL );

   // OK: First column is a multiple of 4 and the submatrix includes the last row and column
   SubmatrixType dsm3 = submatrix<aligned>( D, 4UL, 0UL, 9UL, 17UL );

   // Error: First column is not a multiple of 4, i.e. the first element is not aligned
   SubmatrixType dsm4 = submatrix<aligned>( D, 2UL, 3UL, 12UL, 12UL );
   \endcode

// In case any alignment restrictions are violated, a \a std::invalid_argument exception is thrown.
*/
template< bool AF      // Alignment flag
        , typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
inline DisableIf_< Or< IsComputation<MT>, IsTransExpr<MT> >, SubmatrixExprTrait_<const MT,AF> >
   submatrix( const Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   return SubmatrixExprTrait_<const MT,AF>( ~matrix, row, column, m, n );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given matrix/vector multiplication.
// \ingroup views
//
// \param vector The constant matrix/vector multiplication.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the multiplication.
//
// This function returns an expression representing the specified subvector of the given
// matrix/vector multiplication.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline EnableIf_< IsMatVecMultExpr<VT>, SubvectorExprTrait_<VT,AF> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   typedef RemoveReference_< LeftOperand_<VT> >  MT;

   LeftOperand_<VT>  left ( (~vector).leftOperand()  );
   RightOperand_<VT> right( (~vector).rightOperand() );

   const size_t column( ( IsUpper<MT>::value )
                        ?( ( !AF && IsStrictlyUpper<MT>::value )?( index + 1UL ):( index ) )
                        :( 0UL ) );
   const size_t n( ( IsLower<MT>::value )
                   ?( ( IsUpper<MT>::value )?( size )
                                            :( ( IsStrictlyLower<MT>::value && size > 0UL )
                                               ?( index + size - 1UL )
                                               :( index + size ) ) )
                   :( ( IsUpper<MT>::value )?( left.columns() - column )
                                            :( left.columns() ) ) );

   return submatrix<AF>( left, index, column, size, n ) * subvector<AF>( right, column, n );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given vector/matrix multiplication.
// \ingroup views
//
// \param vector The constant vector/matrix multiplication.
// \param index The index of the first element of the subvector.
// \param size The size of the subvector.
// \return View on the specified subvector of the multiplication.
//
// This function returns an expression representing the specified subvector of the given
// vector/matrix multiplication.
*/
template< bool AF      // Alignment flag
        , typename VT  // Type of the vector
        , bool TF >    // Transpose flag
inline EnableIf_< IsTVecMatMultExpr<VT>, SubvectorExprTrait_<VT,AF> >
   subvector( const Vector<VT,TF>& vector, size_t index, size_t size )
{
   BLAZE_FUNCTION_TRACE;

   typedef RemoveReference_< RightOperand_<VT> >  MT;

   LeftOperand_<VT>  left ( (~vector).leftOperand()  );
   RightOperand_<VT> right( (~vector).rightOperand() );

   const size_t row( ( IsLower<MT>::value )
                     ?( ( !AF && IsStrictlyLower<MT>::value )?( index + 1UL ):( index ) )
                     :( 0UL ) );
   const size_t m( ( IsUpper<MT>::value )
                   ?( ( IsLower<MT>::value )?( size )
                                            :( ( IsStrictlyUpper<MT>::value && size > 0UL )
                                               ?( index + size - 1UL )
                                               :( index + size ) ) )
                   :( ( IsLower<MT>::value )?( right.rows() - row )
                                            :( right.rows() ) ) );

   return subvector<AF>( left, row, m ) * submatrix<AF>( right, row, index, m, size );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific submatrix of the given matrix/matrix addition.
// \ingroup views
//
// \param matrix The constant matrix/matrix addition.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specified submatrix of the addition.
//
// This function returns an expression representing the specified submatrix of the given
// matrix/matrix addition.
*/
template< bool AF      // Alignment flag
        , typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatMatAddExpr<MT>, SubmatrixExprTrait_<MT,AF> >
   submatrix( const Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   return submatrix<AF>( (~matrix).leftOperand() , row, column, m, n ) +
          submatrix<AF>( (~matrix).rightOperand(), row, column, m, n );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific submatrix of the given matrix/matrix subtraction.
// \ingroup views
//
// \param matrix The constant matrix/matrix subtraction.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specified submatrix of the subtraction.
//
// This function returns an expression representing the specified submatrix of the given
// matrix/matrix subtraction.
*/
template< bool AF      // Alignment flag
        , typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatMatSubExpr<MT>, SubmatrixExprTrait_<MT,AF> >
   submatrix( const Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   return submatrix<AF>( (~matrix).leftOperand() , row, column, m, n ) -
          submatrix<AF>( (~matrix).rightOperand(), row, column, m, n );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific submatrix of the given matrix/matrix multiplication.
// \ingroup views
//
// \param matrix The constant matrix/matrix multiplication.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specified submatrix of the multiplication.
//
// This function returns an expression representing the specified submatrix of the given
// matrix/matrix multiplication.
*/
template< bool AF      // Alignment flag
        , typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatMatMultExpr<MT>, SubmatrixExprTrait_<MT,AF> >
   submatrix( const Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   typedef RemoveReference_< LeftOperand_<MT> >   MT1;
   typedef RemoveReference_< RightOperand_<MT> >  MT2;

   LeftOperand_<MT>  left ( (~matrix).leftOperand()  );
   RightOperand_<MT> right( (~matrix).rightOperand() );

   const size_t begin( max( ( IsUpper<MT1>::value )
                            ?( ( !AF && IsStrictlyUpper<MT1>::value )?( row + 1UL ):( row ) )
                            :( 0UL )
                          , ( IsLower<MT2>::value )
                            ?( ( !AF && IsStrictlyLower<MT2>::value )?( column + 1UL ):( column ) )
                            :( 0UL ) ) );
   const size_t end( min( ( IsLower<MT1>::value )
                          ?( ( IsStrictlyLower<MT1>::value && m > 0UL )?( row + m - 1UL ):( row + m ) )
                          :( left.columns() )
                        , ( IsUpper<MT2>::value )
                          ?( ( IsStrictlyUpper<MT2>::value && n > 0UL )?( column + n - 1UL ):( column + n ) )
                          :( left.columns() ) ) );

   const size_t diff( ( begin < end )?( end - begin ):( 0UL ) );

   return submatrix<AF>( left, row, begin, m, diff ) *
          submatrix<AF>( right, begin, column, diff, n );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific submatrix of the given outer product.
// \ingroup views
//
// \param matrix The constant outer product.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specified submatrix of the outer product.
//
// This function returns an expression representing the specified submatrix of the given
// outer product.
*/
template< bool AF      // Alignment flag
        , typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsVecTVecMultExpr<MT>, SubmatrixExprTrait_<MT,AF> >
   submatrix( const Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   return subvector<AF>( (~matrix).leftOperand(), row, m ) *
          subvector<AF>( (~matrix).rightOperand(), column, n );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific submatrix of the given matrix/scalar multiplication.
// \ingroup views
//
// \param matrix The constant matrix/scalar multiplication.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specified submatrix of the multiplication.
//
// This function returns an expression representing the specified submatrix of the given
// matrix/scalar multiplication.
*/
template< bool AF      // Alignment flag
        , typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatScalarMultExpr<MT>, SubmatrixExprTrait_<MT,AF> >
   submatrix( const Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   return submatrix<AF>( (~matrix).leftOperand(), row, column, m, n ) * (~matrix).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific submatrix of the given matrix/scalar division.
// \ingroup views
//
// \param matrix The constant matrix/scalar division.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specified submatrix of the division.
//
// This function returns an expression representing the specified submatrix of the given
// matrix/scalar division.
*/
template< bool AF      // Alignment flag
        , typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatScalarDivExpr<MT>, SubmatrixExprTrait_<MT,AF> >
   submatrix( const Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   return submatrix<AF>( (~matrix).leftOperand(), row, column, m, n ) / (~matrix).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific submatrix of the given matrix custom operation.
// \ingroup views
//
// \param matrix The constant matrix custom operation.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specified submatrix of the custom operation.
//
// This function returns an expression representing the specified submatrix of the given matrix
// custom operation.
*/
template< bool AF      // Alignment flag
        , typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatForEachExpr<MT>, SubmatrixExprTrait_<MT,AF> >
   submatrix( const Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   return forEach( submatrix<AF>( (~matrix).operand(), row, column, m, n ), (~matrix).operation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific submatrix of the given matrix evaluation operation.
// \ingroup views
//
// \param matrix The constant matrix evaluation operation.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specified submatrix of the evaluation operation.
//
// This function returns an expression representing the specified submatrix of the given matrix
// evaluation operation.
*/
template< bool AF      // Alignment flag
        , typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatEvalExpr<MT>, SubmatrixExprTrait_<MT,AF> >
   submatrix( const Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   return eval( submatrix<AF>( (~matrix).operand(), row, column, m, n ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific submatrix of the given matrix serialization operation.
// \ingroup views
//
// \param matrix The constant matrix serialization operation.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specified submatrix of the serialization operation.
//
// This function returns an expression representing the specified submatrix of the given matrix
// serialization operation.
*/
template< bool AF      // Alignment flag
        , typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatSerialExpr<MT>, SubmatrixExprTrait_<MT,AF> >
   submatrix( const Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   return serial( submatrix<AF>( (~matrix).operand(), row, column, m, n ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific submatrix of the given matrix transpose operation.
// \ingroup views
//
// \param matrix The constant matrix transpose operation.
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specified submatrix of the transpose operation.
//
// This function returns an expression representing the specified submatrix of the given matrix
// transpose operation.
*/
template< bool AF      // Alignment flag
        , typename MT  // Type of the matrix
        , bool SO >    // Storage order
inline EnableIf_< IsMatTransExpr<MT>, SubmatrixExprTrait_<MT,AF> >
   submatrix( const Matrix<MT,SO>& matrix, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   return trans( submatrix<AF>( (~matrix).operand(), column, row, n, m ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific submatrix of another submatrix.
// \ingroup views
//
// \param sm The constant submatrix
// \param row The index of the first row of the submatrix.
// \param column The index of the first column of the submatrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \return View on the specified submatrix of the other submatrix.
//
// This function returns an expression representing the specified submatrix of the given submatrix.
*/
template< bool AF1     // Required alignment flag
        , typename MT  // Type of the sparse submatrix
        , bool AF2     // Present alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline const Submatrix<MT,AF1,SO,DF>
   submatrix( const Submatrix<MT,AF2,SO,DF>& sm, size_t row, size_t column, size_t m, size_t n )
{
   BLAZE_FUNCTION_TRACE;

   if( ( row + m > sm.rows() ) || ( column + n > sm.columns() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid submatrix specification" );
   }

   return Submatrix<MT,AF1,SO,DF>( sm.matrix_, sm.row_ + row, sm.column_ + column, m, n );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBMATRIX OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Submatrix operators */
//@{
template< typename MT, bool AF, bool SO, bool DF >
inline void reset( Submatrix<MT,AF,SO,DF>& sm );

template< typename MT, bool AF, bool SO, bool DF >
inline void reset( Submatrix<MT,AF,SO,DF>& sm, size_t i );

template< typename MT, bool AF, bool SO, bool DF >
inline void clear( Submatrix<MT,AF,SO,DF>& sm );

template< typename MT, bool AF, bool SO, bool DF >
inline bool isDefault( const Submatrix<MT,AF,SO,DF>& sm );

template< typename MT, bool AF, bool SO, bool DF >
inline bool isIntact( const Submatrix<MT,AF,SO,DF>& sm ) noexcept;

template< typename MT, bool AF, bool SO, bool DF >
inline bool isSymmetric( const Submatrix<MT,AF,SO,DF>& sm );

template< typename MT, bool AF, bool SO, bool DF >
inline bool isHermitian( const Submatrix<MT,AF,SO,DF>& sm );

template< typename MT, bool AF, bool SO, bool DF >
inline bool isLower( const Submatrix<MT,AF,SO,DF>& sm );

template< typename MT, bool AF, bool SO, bool DF >
inline bool isUniLower( const Submatrix<MT,AF,SO,DF>& sm );

template< typename MT, bool AF, bool SO, bool DF >
inline bool isStrictlyLower( const Submatrix<MT,AF,SO,DF>& sm );

template< typename MT, bool AF, bool SO, bool DF >
inline bool isUpper( const Submatrix<MT,AF,SO,DF>& sm );

template< typename MT, bool AF, bool SO, bool DF >
inline bool isUniUpper( const Submatrix<MT,AF,SO,DF>& sm );

template< typename MT, bool AF, bool SO, bool DF >
inline bool isStrictlyUpper( const Submatrix<MT,AF,SO,DF>& sm );

template< typename MT, bool AF, bool SO, bool DF >
inline bool isSame( const Submatrix<MT,AF,SO,DF>& a, const Matrix<MT,SO>& b ) noexcept;

template< typename MT, bool AF, bool SO, bool DF >
inline bool isSame( const Matrix<MT,SO>& a, const Submatrix<MT,AF,SO,DF>& b ) noexcept;

template< typename MT, bool AF, bool SO, bool DF >
inline bool isSame( const Submatrix<MT,AF,SO,DF>& a, const Submatrix<MT,AF,SO,DF>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given submatrix.
// \ingroup submatrix
//
// \param sm The submatrix to be resetted.
// \return void
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline void reset( Submatrix<MT,AF,SO,DF>& sm )
{
   sm.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset the specified row/column of the given submatrix.
// \ingroup submatrix
//
// \param sm The submatrix to be resetted.
// \param i The index of the row/column to be resetted.
// \return void
//
// This function resets the values in the specified row/column of the given submatrix to their
// default value. In case the given submatrix is a \a rowMajor matrix the function resets the
// values in row \a i, if it is a \a columnMajor matrix the function resets the values in column
// \a i. Note that the capacity of the row/column remains unchanged.
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline void reset( Submatrix<MT,AF,SO,DF>& sm, size_t i )
{
   sm.reset( i );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given matrix.
// \ingroup submatrix
//
// \param sm The matrix to be cleared.
// \return void
//
// Clearing a submatrix is equivalent to resetting it via the reset() function.
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline void clear( Submatrix<MT,AF,SO,DF>& sm )
{
   sm.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given submatrix is in default state.
// \ingroup submatrix
//
// \param sm The submatrix to be tested for its default state.
// \return \a true in case the given submatrix is component-wise zero, \a false otherwise.
//
// This function checks whether the submatrix is in default state. For instance, in
// case the submatrix is instantiated for a built-in integral or floating point data type, the
// function returns \a true in case all submatrix elements are 0 and \a false in case any submatrix
// element is not 0. The following example demonstrates the use of the \a isDefault function:

   \code
   blaze::DynamicMatrix<double,rowMajor> A;
   // ... Resizing and initialization
   if( isDefault( submatrix( A, 12UL, 13UL, 22UL, 33UL ) ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline bool isDefault( const Submatrix<MT,AF,SO,DF>& sm )
{
   using blaze::isDefault;

   if( SO == rowMajor ) {
      for( size_t i=0UL; i<(~sm).rows(); ++i )
         for( size_t j=0UL; j<(~sm).columns(); ++j )
            if( !isDefault( (~sm)(i,j) ) )
               return false;
   }
   else {
      for( size_t j=0UL; j<(~sm).columns(); ++j )
         for( size_t i=0UL; i<(~sm).rows(); ++i )
            if( !isDefault( (~sm)(i,j) ) )
               return false;
   }

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given sparse submatrix is in default state.
// \ingroup submatrix
//
// \param sm The sparse submatrix to be tested for its default state.
// \return \a true in case the given submatrix is component-wise zero, \a false otherwise.
//
// This function checks whether the submatrix is in default state. For instance, in
// case the submatrix is instantiated for a built-in integral or floating point data type, the
// function returns \a true in case all submatrix elements are 0 and \a false in case any submatrix
// element is not 0. The following example demonstrates the use of the \a isDefault function:

   \code
   blaze::CompressedMatrix<double,rowMajor> A;
   // ... Resizing and initialization
   if( isDefault( submatrix( A, 12UL, 13UL, 22UL, 33UL ) ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the sparse matrix
        , bool AF      // Alignment flag
        , bool SO >    // Storage order
inline bool isDefault( const Submatrix<MT,AF,SO,false>& sm )
{
   using blaze::isDefault;

   typedef ConstIterator_< Submatrix<MT,AF,SO,false> >  ConstIterator;

   const size_t iend( ( SO == rowMajor)?( sm.rows() ):( sm.columns() ) );

   for( size_t i=0UL; i<iend; ++i ) {
      for( ConstIterator element=sm.begin(i); element!=sm.end(i); ++element )
         if( !isDefault( element->value() ) ) return false;
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given submatrix are intact.
// \ingroup submatrix
//
// \param sm The submatrix to be tested.
// \return \a true in case the given submatrix's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the submatrix are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   blaze::DynamicMatrix<double,rowMajor> A;
   // ... Resizing and initialization
   if( isIntact( submatrix( A, 12UL, 13UL, 22UL, 33UL ) ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline bool isIntact( const Submatrix<MT,AF,SO,DF>& sm ) noexcept
{
   return ( sm.row_ + sm.m_ <= sm.matrix_.rows() &&
            sm.column_ + sm.n_ <= sm.matrix_.columns() &&
            isIntact( sm.matrix_ ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given submatrix is symmetric.
// \ingroup submatrix
//
// \param sm The submatrix to be checked.
// \return \a true if the submatrix is symmetric, \a false if not.
//
// This function checks if the given submatrix is symmetric. The submatrix is considered to
// be symmetric if it is a square matrix whose transpose is equal to itself (\f$ A = A^T \f$). The
// following code example demonstrates the use of the function:

   \code
   typedef blaze::DynamicMatrix<int,blaze::rowMajor>  Matrix;

   Matrix A( 32UL, 16UL );
   // ... Initialization

   blaze::Submatrix<Matrix> sm( A, 8UL, 8UL, 16UL, 16UL );

   if( isSymmetric( sm ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline bool isSymmetric( const Submatrix<MT,AF,SO,DF>& sm )
{
   using BaseType = BaseType_< Submatrix<MT,AF,SO,DF> >;

   if( IsSymmetric<MT>::value && sm.row() == sm.column() && sm.rows() == sm.columns() )
      return true;
   else return isSymmetric( static_cast<const BaseType&>( sm ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given submatrix is Hermitian.
// \ingroup submatrix
//
// \param sm The submatrix to be checked.
// \return \a true if the submatrix is Hermitian, \a false if not.
//
// This function checks if the given submatrix is Hermitian. The submatrix is considered to
// be Hermitian if it is a square matrix whose transpose is equal to its conjugate transpose
// (\f$ A = \overline{A^T} \f$). The following code example demonstrates the use of the function:

   \code
   typedef blaze::DynamicMatrix<int,blaze::rowMajor>  Matrix;

   Matrix A( 32UL, 16UL );
   // ... Initialization

   blaze::Submatrix<Matrix> sm( A, 8UL, 8UL, 16UL, 16UL );

   if( isHermitian( sm ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline bool isHermitian( const Submatrix<MT,AF,SO,DF>& sm )
{
   using BaseType = BaseType_< Submatrix<MT,AF,SO,DF> >;

   if( IsHermitian<MT>::value && sm.row() == sm.column() && sm.rows() == sm.columns() )
      return true;
   else return isHermitian( static_cast<const BaseType&>( sm ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given submatrix is a lower triangular matrix.
// \ingroup submatrix
//
// \param sm The submatrix to be checked.
// \return \a true if the submatrix is a lower triangular matrix, \a false if not.
//
// This function checks if the given submatrix is a lower triangular matrix. The matrix is
// considered to be lower triangular if it is a square matrix of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        l_{0,0} & 0       & 0       & \cdots & 0       \\
                        l_{1,0} & l_{1,1} & 0       & \cdots & 0       \\
                        l_{2,0} & l_{2,1} & l_{2,2} & \cdots & 0       \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots  \\
                        l_{N,0} & l_{N,1} & l_{N,2} & \cdots & l_{N,N} \\
                        \end{array}\right).\f]

// \f$ 0 \times 0 \f$ or \f$ 1 \times 1 \f$ matrices are considered as trivially lower triangular.
// The following code example demonstrates the use of the function:

   \code
   typedef blaze::DynamicMatrix<int,blaze::rowMajor>  Matrix;

   Matrix A( 32UL, 16UL );
   // ... Initialization

   blaze::Submatrix<Matrix> sm( A, 8UL, 8UL, 16UL, 16UL );

   if( isLower( sm ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline bool isLower( const Submatrix<MT,AF,SO,DF>& sm )
{
   using BaseType = BaseType_< Submatrix<MT,AF,SO,DF> >;

   if( IsLower<MT>::value && sm.row() == sm.column() && sm.rows() == sm.columns() )
      return true;
   else return isLower( static_cast<const BaseType&>( sm ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given submatrix is a lower unitriangular matrix.
// \ingroup submatrix
//
// \param sm The submatrix to be checked.
// \return \a true if the submatrix is a lower unitriangular matrix, \a false if not.
//
// This function checks if the given submatrix is a lower unitriangular matrix. The matrix is
// considered to be lower triangular if it is a square matrix of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        1       & 0       & 0       & \cdots & 0      \\
                        l_{1,0} & 1       & 0       & \cdots & 0      \\
                        l_{2,0} & l_{2,1} & 1       & \cdots & 0      \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots \\
                        l_{N,0} & l_{N,1} & l_{N,2} & \cdots & 1      \\
                        \end{array}\right).\f]

// The following code example demonstrates the use of the function:

   \code
   typedef blaze::DynamicMatrix<int,blaze::rowMajor>  Matrix;

   Matrix A( 32UL, 16UL );
   // ... Initialization

   blaze::Submatrix<Matrix> sm( A, 8UL, 8UL, 16UL, 16UL );

   if( isUniLower( sm ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline bool isUniLower( const Submatrix<MT,AF,SO,DF>& sm )
{
   using BaseType = BaseType_< Submatrix<MT,AF,SO,DF> >;

   if( IsUniLower<MT>::value && sm.row() == sm.column() && sm.rows() == sm.columns() )
      return true;
   else return isUniLower( static_cast<const BaseType&>( sm ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given submatrix is a strictly lower triangular matrix.
// \ingroup submatrix
//
// \param sm The submatrix to be checked.
// \return \a true if the submatrix is a strictly lower triangular matrix, \a false if not.
//
// This function checks if the given submatrix is a strictly lower triangular matrix. The
// matrix is considered to be lower triangular if it is a square matrix of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        0       & 0       & 0       & \cdots & 0      \\
                        l_{1,0} & 0       & 0       & \cdots & 0      \\
                        l_{2,0} & l_{2,1} & 0       & \cdots & 0      \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots \\
                        l_{N,0} & l_{N,1} & l_{N,2} & \cdots & 0      \\
                        \end{array}\right).\f]

// The following code example demonstrates the use of the function:

   \code
   typedef blaze::DynamicMatrix<int,blaze::rowMajor>  Matrix;

   Matrix A( 32UL, 16UL );
   // ... Initialization

   blaze::Submatrix<Matrix> sm( A, 8UL, 8UL, 16UL, 16UL );

   if( isStrictlyLower( sm ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline bool isStrictlyLower( const Submatrix<MT,AF,SO,DF>& sm )
{
   using BaseType = BaseType_< Submatrix<MT,AF,SO,DF> >;

   if( IsStrictlyLower<MT>::value && sm.row() == sm.column() && sm.rows() == sm.columns() )
      return true;
   else return isStrictlyLower( static_cast<const BaseType&>( sm ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given submatrix is an upper triangular matrix.
// \ingroup submatrix
//
// \param sm The submatrix to be checked.
// \return \a true if the submatrix is an upper triangular matrix, \a false if not.
//
// This function checks if the given sparse submatrix is an upper triangular matrix. The matrix
// is considered to be upper triangular if it is a square matrix of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        u_{0,0} & u_{0,1} & u_{0,2} & \cdots & u_{0,N} \\
                        0       & u_{1,1} & u_{1,2} & \cdots & u_{1,N} \\
                        0       & 0       & u_{2,2} & \cdots & u_{2,N} \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots  \\
                        0       & 0       & 0       & \cdots & u_{N,N} \\
                        \end{array}\right).\f]

// \f$ 0 \times 0 \f$ or \f$ 1 \times 1 \f$ matrices are considered as trivially upper triangular.
// The following code example demonstrates the use of the function:

   \code
   typedef blaze::DynamicMatrix<int,blaze::rowMajor>  Matrix;

   Matrix A( 32UL, 16UL );
   // ... Initialization

   blaze::Submatrix<Matrix> sm( A, 8UL, 8UL, 16UL, 16UL );

   if( isUpper( sm ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline bool isUpper( const Submatrix<MT,AF,SO,DF>& sm )
{
   using BaseType = BaseType_< Submatrix<MT,AF,SO,DF> >;

   if( IsUpper<MT>::value && sm.row() == sm.column() && sm.rows() == sm.columns() )
      return true;
   else return isUpper( static_cast<const BaseType&>( sm ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given submatrix is an upper unitriangular matrix.
// \ingroup submatrix
//
// \param sm The submatrix to be checked.
// \return \a true if the submatrix is an upper unitriangular matrix, \a false if not.
//
// This function checks if the given sparse submatrix is an upper triangular matrix. The matrix
// is considered to be upper triangular if it is a square matrix of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        1      & u_{0,1} & u_{0,2} & \cdots & u_{0,N} \\
                        0      & 1       & u_{1,2} & \cdots & u_{1,N} \\
                        0      & 0       & 1       & \cdots & u_{2,N} \\
                        \vdots & \vdots  & \vdots  & \ddots & \vdots  \\
                        0      & 0       & 0       & \cdots & 1       \\
                        \end{array}\right).\f]

// The following code example demonstrates the use of the function:

   \code
   typedef blaze::DynamicMatrix<int,blaze::rowMajor>  Matrix;

   Matrix A( 32UL, 16UL );
   // ... Initialization

   blaze::Submatrix<Matrix> sm( A, 8UL, 8UL, 16UL, 16UL );

   if( isUniUpper( sm ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline bool isUniUpper( const Submatrix<MT,AF,SO,DF>& sm )
{
   using BaseType = BaseType_< Submatrix<MT,AF,SO,DF> >;

   if( IsUniUpper<MT>::value && sm.row() == sm.column() && sm.rows() == sm.columns() )
      return true;
   else return isUniUpper( static_cast<const BaseType&>( sm ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given submatrix is a strictly upper triangular matrix.
// \ingroup submatrix
//
// \param sm The submatrix to be checked.
// \return \a true if the submatrix is a strictly upper triangular matrix, \a false if not.
//
// This function checks if the given sparse submatrix is a strictly upper triangular matrix. The
// matrix is considered to be upper triangular if it is a square matrix of the form

                        \f[\left(\begin{array}{*{5}{c}}
                        0      & u_{0,1} & u_{0,2} & \cdots & u_{0,N} \\
                        0      & 0       & u_{1,2} & \cdots & u_{1,N} \\
                        0      & 0       & 0       & \cdots & u_{2,N} \\
                        \vdots & \vdots  & \vdots  & \ddots & \vdots  \\
                        0      & 0       & 0       & \cdots & 0       \\
                        \end{array}\right).\f]

// The following code example demonstrates the use of the function:

   \code
   typedef blaze::DynamicMatrix<int,blaze::rowMajor>  Matrix;

   Matrix A( 32UL, 16UL );
   // ... Initialization

   blaze::Submatrix<Matrix> sm( A, 8UL, 8UL, 16UL, 16UL );

   if( isStrictlyUpper( sm ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline bool isStrictlyUpper( const Submatrix<MT,AF,SO,DF>& sm )
{
   using BaseType = BaseType_< Submatrix<MT,AF,SO,DF> >;

   if( IsStrictlyUpper<MT>::value && sm.row() == sm.column() && sm.rows() == sm.columns() )
      return true;
   else return isStrictlyUpper( static_cast<const BaseType&>( sm ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given matrix and submatrix represent the same observable state.
// \ingroup submatrix
//
// \param a The submatrix to be tested for its state.
// \param b The matrix to be tested for its state.
// \return \a true in case the submatrix and matrix share a state, \a false otherwise.
//
// This overload of the isSame function tests if the given submatrix refers to the full given
// matrix and by that represents the same observable state. In this case, the function returns
// \a true, otherwise it returns \a false.
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline bool isSame( const Submatrix<MT,AF,SO,DF>& a, const Matrix<MT,SO>& b ) noexcept
{
   return ( isSame( a.matrix_, ~b ) && ( a.rows() == (~b).rows() ) && ( a.columns() == (~b).columns() ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given matrix and submatrix represent the same observable state.
// \ingroup submatrix
//
// \param a The matrix to be tested for its state.
// \param b The submatrix to be tested for its state.
// \return \a true in case the matrix and submatrix share a state, \a false otherwise.
//
// This overload of the isSame function tests if the given submatrix refers to the full given
// matrix and by that represents the same observable state. In this case, the function returns
// \a true, otherwise it returns \a false.
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline bool isSame( const Matrix<MT,SO>& a, const Submatrix<MT,AF,SO,DF>& b ) noexcept
{
   return ( isSame( ~a, b.matrix_ ) && ( (~a).rows() == b.rows() ) && ( (~a).columns() == b.columns() ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the two given submatrices represent the same observable state.
// \ingroup submatrix
//
// \param a The first submatrix to be tested for its state.
// \param b The second submatrix to be tested for its state.
// \return \a true in case the two submatrices share a state, \a false otherwise.
//
// This overload of the isSame function tests if the two given submatrices refer to exactly the
// same part of the same matrix. In case both submatrices represent the same observable state,
// the function returns \a true, otherwise it returns \a false.
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline bool isSame( const Submatrix<MT,AF,SO,DF>& a, const Submatrix<MT,AF,SO,DF>& b ) noexcept
{
   return ( isSame( a.matrix_, b.matrix_ ) &&
            ( a.row_ == b.row_ ) && ( a.column_ == b.column_ ) &&
            ( a.m_ == b.m_ ) && ( a.n_ == b.n_ ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given submatrix.
// \ingroup submatrix
//
// \param sm The dense submatrix to be inverted.
// \return void
// \exception std::invalid_argument Inversion of singular matrix failed.
//
// This function inverts the given dense submatrix by means of the specified matrix decomposition
// algorithm \a IF:

   \code
   invert<byLU>( A );    // Inversion of a general matrix
   invert<byLDLT>( A );  // Inversion of a symmetric indefinite matrix
   invert<byLDLH>( A );  // Inversion of a Hermitian indefinite matrix
   invert<byLLH>( A );   // Inversion of a Hermitian positive definite matrix
   \endcode

// The matrix inversion fails if ...
//
//  - ... the given submatrix is not a square matrix;
//  - ... the given submatrix is singular and not invertible.
//
// In all failure cases either a compilation error is created if the failure can be predicted at
// compile time or a \a std::invalid_argument exception is thrown.
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a linker error will be created.
//
// \note This function does only provide the basic exception safety guarantee, i.e. in case of an
// exception \a sm may already have been modified.
*/
template< InversionFlag IF  // Inversion algorithm
        , typename MT       // Type of the dense matrix
        , bool AF           // Alignment flag
        , bool SO >         // Storage order
inline DisableIf_< HasMutableDataAccess<MT> > invert( Submatrix<MT,AF,SO,true>& sm )
{
   typedef ResultType_< Submatrix<MT,AF,SO,true> >  RT;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( RT );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( RT );

   RT tmp( sm );
   invert<IF>( tmp );
   sm = tmp;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a vector to a submatrix.
// \ingroup submatrix
//
// \param lhs The target left-hand side submatrix.
// \param rhs The right-hand side vector to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF      // Density flag
        , typename VT  // Type of the right-hand side vector
        , bool TF >    // Transpose flag of the right-hand side vector
inline bool tryAssign( const Submatrix<MT,AF,SO,DF>& lhs, const Vector<VT,TF>& rhs,
                       size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( TF || ( (~rhs).size() <= lhs.rows() - row ), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( !TF || ( (~rhs).size() <= lhs.columns() - column ), "Invalid number of columns" );

   return tryAssign( lhs.matrix_, ~rhs, lhs.row_ + row, lhs.column_ + column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a matrix to a submatrix.
// \ingroup submatrix
//
// \param lhs The target left-hand side submatrix.
// \param rhs The right-hand side matrix to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the matrix
        , bool AF       // Alignment flag
        , bool SO1      // Storage order
        , bool DF       // Density flag
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline bool tryAssign( const Submatrix<MT1,AF,SO1,DF>& lhs, const Matrix<MT2,SO2>& rhs,
                       size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= lhs.rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= lhs.columns() - column, "Invalid number of columns" );

   return tryAssign( lhs.matrix_, ~rhs, lhs.row_ + row, lhs.column_ + column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to a submatrix.
// \ingroup submatrix
//
// \param lhs The target left-hand side submatrix.
// \param rhs The right-hand side vector to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF      // Density flag
        , typename VT  // Type of the right-hand side vector
        , bool TF >    // Transpose flag of the right-hand side vector
inline bool tryAddAssign( const Submatrix<MT,AF,SO,DF>& lhs, const Vector<VT,TF>& rhs,
                          size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( TF || ( (~rhs).size() <= lhs.rows() - row ), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( !TF || ( (~rhs).size() <= lhs.columns() - column ), "Invalid number of columns" );

   return tryAddAssign( lhs.matrix_, ~rhs, lhs.row_ + row, lhs.column_ + column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a matrix to a submatrix.
// \ingroup submatrix
//
// \param lhs The target left-hand side submatrix.
// \param rhs The right-hand side matrix to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the matrix
        , bool AF       // Alignment flag
        , bool SO1      // Storage order
        , bool DF       // Density flag
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline bool tryAddAssign( const Submatrix<MT1,AF,SO1,DF>& lhs, const Matrix<MT2,SO2>& rhs,
                          size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= lhs.rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= lhs.columns() - column, "Invalid number of columns" );

   return tryAddAssign( lhs.matrix_, ~rhs, lhs.row_ + row, lhs.column_ + column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to a submatrix.
// \ingroup submatrix
//
// \param lhs The target left-hand side submatrix.
// \param rhs The right-hand side vector to be subtracted.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF      // Density flag
        , typename VT  // Type of the right-hand side vector
        , bool TF >    // Transpose flag of the right-hand side vector
inline bool trySubAssign( const Submatrix<MT,AF,SO,DF>& lhs, const Vector<VT,TF>& rhs,
                          size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( TF || ( (~rhs).size() <= lhs.rows() - row ), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( !TF || ( (~rhs).size() <= lhs.columns() - column ), "Invalid number of columns" );

   return trySubAssign( lhs.matrix_, ~rhs, lhs.row_ + row, lhs.column_ + column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a matrix to a submatrix.
// \ingroup submatrix
//
// \param lhs The target left-hand side submatrix.
// \param rhs The right-hand side matrix to be subtracted.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the matrix
        , bool AF       // Alignment flag
        , bool SO1      // Storage order
        , bool DF       // Density flag
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline bool trySubAssign( const Submatrix<MT1,AF,SO1,DF>& lhs, const Matrix<MT2,SO2>& rhs,
                          size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= lhs.rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= lhs.columns() - column, "Invalid number of columns" );

   return trySubAssign( lhs.matrix_, ~rhs, lhs.row_ + row, lhs.column_ + column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to a submatrix.
// \ingroup submatrix
//
// \param lhs The target left-hand side submatrix.
// \param rhs The right-hand side vector to be multiplied.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF      // Density flag
        , typename VT  // Type of the right-hand side vector
        , bool TF >    // Transpose flag of the right-hand side vector
inline bool tryMultAssign( const Submatrix<MT,AF,SO,DF>& lhs, const Vector<VT,TF>& rhs,
                           size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( TF || ( (~rhs).size() <= lhs.rows() - row ), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( !TF || ( (~rhs).size() <= lhs.columns() - column ), "Invalid number of columns" );

   return tryMultAssign( lhs.matrix_, ~rhs, lhs.row_ + row, lhs.column_ + column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given submatrix.
// \ingroup submatrix
//
// \param dm The submatrix to be derestricted.
// \return Submatrix without access restrictions.
//
// This function removes all restrictions on the data access to the given submatrix. It returns a
// submatrix that does provide the same interface but does not have any restrictions on the data
// access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the matrix
        , bool AF      // Alignment flag
        , bool SO      // Storage order
        , bool DF >    // Density flag
inline DerestrictTrait_< Submatrix<MT,AF,SO,DF> > derestrict( Submatrix<MT,AF,SO,DF>& dm )
{
   typedef DerestrictTrait_< Submatrix<MT,AF,SO,DF> >  ReturnType;
   return ReturnType( derestrict( dm.matrix_ ), dm.row_, dm.column_, dm.m_, dm.n_ );
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
template< typename MT, bool AF, bool SO, bool DF >
struct IsRestricted< Submatrix<MT,AF,SO,DF> >
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
template< typename MT, bool AF, bool SO, bool DF >
struct DerestrictTrait< Submatrix<MT,AF,SO,DF> >
{
   using Type = Submatrix< RemoveReference_< DerestrictTrait_<MT> >, AF >;
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
template< typename MT, bool AF, bool SO >
struct HasConstDataAccess< Submatrix<MT,AF,SO,true> >
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
template< typename MT, bool AF, bool SO >
struct HasMutableDataAccess< Submatrix<MT,AF,SO,true> >
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
template< typename MT, bool SO >
struct IsAligned< Submatrix<MT,aligned,SO,true> > : public TrueType
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
template< typename MT, bool AF, bool SO, bool DF, typename T >
struct AddTrait< Submatrix<MT,AF,SO,DF>, T >
{
   using Type = AddTrait_< SubmatrixTrait_<MT>, T >;
};

template< typename T, typename MT, bool AF, bool SO, bool DF >
struct AddTrait< T, Submatrix<MT,AF,SO,DF> >
{
   using Type = AddTrait_< T, SubmatrixTrait_<MT> >;
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
template< typename MT, bool AF, bool SO, bool DF, typename T >
struct SubTrait< Submatrix<MT,AF,SO,DF>, T >
{
   using Type = SubTrait_< SubmatrixTrait_<MT>, T >;
};

template< typename T, typename MT, bool AF, bool SO, bool DF >
struct SubTrait< T, Submatrix<MT,AF,SO,DF> >
{
   using Type = SubTrait_< T, SubmatrixTrait_<MT> >;
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
template< typename MT, bool AF, bool SO, bool DF, typename T >
struct MultTrait< Submatrix<MT,AF,SO,DF>, T >
{
   using Type = MultTrait_< SubmatrixTrait_<MT>, T >;
};

template< typename T, typename MT, bool AF, bool SO, bool DF >
struct MultTrait< T, Submatrix<MT,AF,SO,DF> >
{
   using Type = MultTrait_< T, SubmatrixTrait_<MT> >;
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
template< typename MT, bool AF, bool SO, bool DF, typename T >
struct DivTrait< Submatrix<MT,AF,SO,DF>, T >
{
   using Type = DivTrait_< SubmatrixTrait_<MT>, T >;
};

template< typename T, typename MT, bool AF, bool DF, bool SO >
struct DivTrait< T, Submatrix<MT,AF,SO,DF> >
{
   using Type = DivTrait_< T, SubmatrixTrait_<MT> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBMATRIXTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool AF, bool SO, bool DF >
struct SubmatrixTrait< Submatrix<MT,AF,SO,DF> >
{
   using Type = SubmatrixTrait_< ResultType_< Submatrix<MT,AF,SO,DF> > >;
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
template< typename MT, bool AF1, bool SO, bool DF, bool AF2 >
struct SubmatrixExprTrait< Submatrix<MT,AF1,SO,DF>, AF2 >
{
   using Type = Submatrix<MT,AF2,SO,DF>;
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool AF1, bool SO, bool DF, bool AF2 >
struct SubmatrixExprTrait< const Submatrix<MT,AF1,SO,DF>, AF2 >
{
   using Type = Submatrix<MT,AF2,SO,DF>;
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool AF1, bool SO, bool DF, bool AF2 >
struct SubmatrixExprTrait< volatile Submatrix<MT,AF1,SO,DF>, AF2 >
{
   using Type = Submatrix<MT,AF2,SO,DF>;
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool AF1, bool SO, bool DF, bool AF2 >
struct SubmatrixExprTrait< const volatile Submatrix<MT,AF1,SO,DF>, AF2 >
{
   using Type = Submatrix<MT,AF2,SO,DF>;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ROWTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool AF, bool SO, bool DF >
struct RowTrait< Submatrix<MT,AF,SO,DF> >
{
   using Type = RowTrait_< ResultType_< Submatrix<MT,AF,SO,DF> > >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COLUMNTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool AF, bool SO, bool DF >
struct ColumnTrait< Submatrix<MT,AF,SO,DF> >
{
   using Type = ColumnTrait_< ResultType_< Submatrix<MT,AF,SO,DF> > >;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
