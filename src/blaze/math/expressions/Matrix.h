//=================================================================================================
/*!
//  \file blaze/math/expressions/Matrix.h
//  \brief Header file for the Matrix base class
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

#ifndef _BLAZE_MATH_EXPRESSIONS_MATRIX_H_
#define _BLAZE_MATH_EXPRESSIONS_MATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsSquare.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Assert.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/logging/FunctionTrace.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/mpl/Not.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsSame.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup matrix Matrices
// \ingroup math
*/
/*!\brief Base class for matrices.
// \ingroup matrix
//
// The Matrix class is a base class for all dense and sparse matrix classes within the Blaze
// library. It provides an abstraction from the actual type of the matrix, but enables a
// conversion back to this type via the 'Curiously Recurring Template Pattern' (CRTP).
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
struct Matrix
{
   //**Type definitions****************************************************************************
   typedef MT  MatrixType;  //!< Type of the matrix.
   //**********************************************************************************************

   //**Non-const conversion operator***************************************************************
   /*!\brief Conversion operator for non-constant matrices.
   //
   // \return Reference of the actual type of the matrix.
   */
   BLAZE_ALWAYS_INLINE MatrixType& operator~() noexcept {
      return *static_cast<MatrixType*>( this );
   }
   //**********************************************************************************************

   //**Const conversion operator*******************************************************************
   /*!\brief Conversion operator for constant matrices.
   //
   // \return Constant reference of the actual type of the matrix.
   */
   BLAZE_ALWAYS_INLINE const MatrixType& operator~() const noexcept {
      return *static_cast<const MatrixType*>( this );
   }
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Matrix global functions */
//@{
template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE typename MT::Iterator begin( Matrix<MT,SO>& matrix, size_t i );

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE typename MT::ConstIterator begin( const Matrix<MT,SO>& matrix, size_t i );

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE typename MT::ConstIterator cbegin( const Matrix<MT,SO>& matrix, size_t i );

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE typename MT::Iterator end( Matrix<MT,SO>& matrix, size_t i );

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE typename MT::ConstIterator end( const Matrix<MT,SO>& matrix, size_t i );

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE typename MT::ConstIterator cend( const Matrix<MT,SO>& matrix, size_t i );

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE size_t rows( const Matrix<MT,SO>& matrix ) noexcept;

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE size_t columns( const Matrix<MT,SO>& matrix ) noexcept;

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE size_t capacity( const Matrix<MT,SO>& matrix ) noexcept;

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE size_t capacity( const Matrix<MT,SO>& matrix, size_t i ) noexcept;

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE size_t nonZeros( const Matrix<MT,SO>& matrix );

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE size_t nonZeros( const Matrix<MT,SO>& matrix, size_t i );

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE void resize( Matrix<MT,SO>& matrix, size_t rows, size_t columns, bool preserve=true );

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE void transpose( Matrix<MT,SO>& matrix );

template< typename MT, bool SO >
BLAZE_ALWAYS_INLINE bool isSquare( const Matrix<MT,SO>& matrix ) noexcept;

template< typename MT1, bool SO1, typename MT2, bool SO2 >
BLAZE_ALWAYS_INLINE bool isSame( const Matrix<MT1,SO1>& a, const Matrix<MT2,SO2>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
// \ingroup matrix
//
// \param matrix The given dense or sparse matrix.
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the given matrix is a row-major matrix the function returns an iterator to the first element
// of row \a i, in case it is a column-major matrix the function returns an iterator to the first
// element of column \a i.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE typename MT::Iterator begin( Matrix<MT,SO>& matrix, size_t i )
{
   return (~matrix).begin(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
// \ingroup matrix
//
// \param matrix The given dense or sparse matrix.
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the given matrix is a row-major matrix the function returns an iterator to the first element
// of row \a i, in case it is a column-major matrix the function returns an iterator to the first
// element of column \a i.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE typename MT::ConstIterator begin( const Matrix<MT,SO>& matrix, size_t i )
{
   return (~matrix).begin(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
// \ingroup matrix
//
// \param matrix The given dense or sparse matrix.
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the given matrix is a row-major matrix the function returns an iterator to the first element
// of row \a i, in case it is a column-major matrix the function returns an iterator to the first
// element of column \a i.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE typename MT::ConstIterator cbegin( const Matrix<MT,SO>& matrix, size_t i )
{
   return (~matrix).cbegin(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
// \ingroup matrix
//
// \param matrix The given dense or sparse matrix.
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the given matrix is a row-major matrix the function returns an iterator just past
// the last element of row \a i, in case it is a column-major matrix the function returns an
// iterator just past the last element of column \a i.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE typename MT::Iterator end( Matrix<MT,SO>& matrix, size_t i )
{
   return (~matrix).end(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
// \ingroup matrix
//
// \param matrix The given dense or sparse matrix.
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the given matrix is a row-major matrix the function returns an iterator just past
// the last element of row \a i, in case it is a column-major matrix the function returns an
// iterator just past the last element of column \a i.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE typename MT::ConstIterator end( const Matrix<MT,SO>& matrix, size_t i )
{
   return (~matrix).end(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
// \ingroup matrix
//
// \param matrix The given dense or sparse matrix.
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the given matrix is a row-major matrix the function returns an iterator just past
// the last element of row \a i, in case it is a column-major matrix the function returns an
// iterator just past the last element of column \a i.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE typename MT::ConstIterator cend( const Matrix<MT,SO>& matrix, size_t i )
{
   return (~matrix).cend(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of rows of the matrix.
// \ingroup matrix
//
// \param matrix The given matrix.
// \return The number of rows of the matrix.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE size_t rows( const Matrix<MT,SO>& matrix ) noexcept
{
   return (~matrix).rows();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of columns of the matrix.
// \ingroup matrix
//
// \param matrix The given matrix.
// \return The number of columns of the matrix.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE size_t columns( const Matrix<MT,SO>& matrix ) noexcept
{
   return (~matrix).columns();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the matrix.
// \ingroup matrix
//
// \param matrix The given matrix.
// \return The capacity of the matrix.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE size_t capacity( const Matrix<MT,SO>& matrix ) noexcept
{
   return (~matrix).capacity();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current capacity of the specified row/column.
// \ingroup matrix
//
// \param matrix The given matrix.
// \param i The index of the row/column.
// \return The current capacity of row/column \a i.
//
// This function returns the current capacity of the specified row/column. In case the
// storage order is set to \a rowMajor the function returns the capacity of row \a i,
// in case the storage flag is set to \a columnMajor the function returns the capacity
// of column \a i.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE size_t capacity( const Matrix<MT,SO>& matrix, size_t i ) noexcept
{
   return (~matrix).capacity( i );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of non-zero elements in the matrix
// \ingroup matrix
//
// \param matrix The given matrix.
// \return The number of non-zero elements in the dense matrix.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE size_t nonZeros( const Matrix<MT,SO>& matrix )
{
   return (~matrix).nonZeros();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the specified row/column.
// \ingroup matrix
//
// \param matrix The given matrix.
// \param i The index of the row/column.
// \return The number of non-zero elements of row/column \a i.
//
// This function returns the current number of non-zero elements in the specified row/column.
// In case the storage order is set to \a rowMajor the function returns the number of non-zero
// elements in row \a i, in case the storage flag is set to \a columnMajor the function returns
// the number of non-zero elements in column \a i.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE size_t nonZeros( const Matrix<MT,SO>& matrix, size_t i )
{
   return (~matrix).nonZeros( i );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for non-resizable matrices.
// \ingroup matrix
//
// \param matrix The given matrix to be resized.
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Matrix cannot be resized.
//
// This function tries to change the number of rows and columns of a non-resizable matrix. Since
// the matrix cannot be resized, in case the specified number of rows and columns is not identical
// to the current number of rows and columns of the matrix, a \a std::invalid_argument exception
// is thrown.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE DisableIf_< IsResizable<MT> >
   resize_backend( Matrix<MT,SO>& matrix, size_t m, size_t n, bool preserve )
{
   UNUSED_PARAMETER( preserve );

   if( (~matrix).rows() != m || (~matrix).columns() != n ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix cannot be resized" );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for resizable, non-square matrices.
// \ingroup matrix
//
// \param matrix The given matrix to be resized.
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function changes the number of rows and columns of the given resizable, non-square matrix.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE EnableIf_< And< IsResizable<MT>, Not< IsSquare<MT> > > >
   resize_backend( Matrix<MT,SO>& matrix, size_t m, size_t n, bool preserve )
{
   (~matrix).resize( m, n, preserve );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for resizable, square matrices.
// \ingroup matrix
//
// \param matrix The given matrix to be resized.
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Invalid resize arguments for square matrix.
//
// This function changes the number of rows and columns of the given resizable, square matrix.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE EnableIf_< And< IsResizable<MT>, IsSquare<MT> > >
   resize_backend( Matrix<MT,SO>& matrix, size_t m, size_t n, bool preserve )
{
   if( m != n ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid resize arguments for square matrix" );
   }

   (~matrix).resize( m, preserve );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the matrix.
// \ingroup matrix
//
// \param matrix The given matrix to be resized.
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Invalid resize arguments for square matrix.
// \exception std::invalid_argument Matrix cannot be resized.
//
// This function provides a unified interface to resize dense and sparse matrices. In contrast
// to the \c resize() member function, which is only available on resizable matrix types, this
// function can be used on both resizable and non-resizable matrices. In case the given matrix
// of type \a MT is resizable (i.e. provides a \c resize function) the type-specific \c resize()
// member function is called. Depending on the type \a MT, this may result in the allocation of
// new dynamic memory and the invalidation of existing views (submatrices, rows, columns, ...).
// Note that in case the matrix is a compile time square matrix (as for instance the
// blaze::SymmetricMatrix adaptor, ...) the specified number of rows must be identical to the
// number of columns. Otherwise a \a std::invalid_argument exception is thrown. If the matrix
// type \a MT is non-resizable (i.e. does not provide a \c resize() function) and if the specified
// number of rows and columns is not identical to the current number of rows and columns of the
// matrix, a \a std::invalid_argument exception is thrown.

   \code
   blaze::DynamicMatrix<int> A( 3UL, 3UL );
   resize( A, 5UL, 2UL );  // OK: regular resize operation

   blaze::SymmetricMatrix< DynamicMatrix<int> > B( 3UL );
   resize( B, 4UL, 4UL );  // OK: Number of rows and columns is identical
   resize( B, 3UL, 5UL );  // Error: Invalid arguments for square matrix!

   blaze::StaticMatrix<int,3UL,3UL> C;
   resize( C, 3UL, 3UL );  // OK: No resize necessary
   resize( C, 5UL, 2UL );  // Error: Matrix cannot be resized!
   \endcode
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE void resize( Matrix<MT,SO>& matrix, size_t m, size_t n, bool preserve )
{
   resize_backend( matrix, m, n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the given matrix.
// \ingroup matrix
//
// \param matrix The given matrix to be transposed.
// \return void
// \exception std::logic_error Matrix cannot be transposed.
//
// This function transposes the given matrix in-place. The function fails if ...
//
//  - ... the given matrix has a fixed size and is non-square;
//  - ... the given matrix is a triangular matrix;
//  - ... the given submatrix affects the restricted parts of a triangular matrix;
//  - ... the given submatrix would cause non-deterministic results in a symmetric/Hermitian matrix.
//
// In all failure cases a \a std::logic_error exception is thrown.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE void transpose( Matrix<MT,SO>& matrix )
{
   (~matrix).transpose();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the given matrix.
// \ingroup matrix
//
// \param matrix The given matrix to be transposed.
// \return void
// \exception std::logic_error Matrix cannot be transposed.
//
// This function transposes the given matrix in-place. The function fails if ...
//
//  - ... the given matrix has a fixed size and is non-square;
//  - ... the given matrix is a triangular matrix;
//  - ... the given submatrix affects the restricted parts of a triangular matrix;
//  - ... the given submatrix would cause non-deterministic results in a symmetric/Hermitian matrix.
//
// In all failure cases a \a std::logic_error exception is thrown.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order of the matrix
BLAZE_ALWAYS_INLINE void ctranspose( Matrix<MT,SO>& matrix )
{
   (~matrix).ctranspose();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checks if the given matrix is a square matrix.
// \ingroup matrix
//
// \param matrix The matrix to be checked.
// \return \a true if the matrix is a square matrix, \a false if not.
//
// This function checks if the number of rows and columns of the given matrix are equal. If
// they are, the function returns \a true, otherwise it returns \a false.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
BLAZE_ALWAYS_INLINE bool isSquare( const Matrix<MT,SO>& matrix ) noexcept
{
   return ( IsSquare<MT>::value || (~matrix).rows() == (~matrix).columns() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the two given matrices represent the same observable state.
// \ingroup matrix
//
// \param a The first matrix to be tested for its state.
// \param b The second matrix to be tested for its state.
// \return \a true in case the two matrices share a state, \a false otherwise.
//
// The isSame function provides an abstract interface for testing if the two given matrices
// represent the same observable state. This happens for instance in case \c a and \c b refer
// to the same matrix or in case \c a and \c b are aliases for the same matrix. In case both
// matrices represent the same observable state, the function returns \a true, other it returns
// \a false.

   \code
   typedef blaze::DynamicMatrix<int>     MatrixType;
   typedef blaze::Submatrix<MatrixType>  SubmatrixType;

   MatrixType mat1( 4UL, 5UL );  // Setup of a 4x5 dynamic matrix
   MatrixType mat2( 4UL, 5UL );  // Setup of a second 4x5 dynamic matrix

   SubmatrixType sub1 = submatrix( mat1, 0UL, 0UL, 4UL, 5UL );  // Submatrix fully covering mat1
   SubmatrixType sub2 = submatrix( mat1, 1UL, 1UL, 2UL, 3UL );  // Submatrix partially covering mat1
   SubmatrixType sub3 = submatrix( mat1, 1UL, 1UL, 2UL, 3UL );  // Submatrix partially covering mat1

   isSame( mat1, mat1 );  // returns true since both objects refer to the same matrix
   isSame( mat1, mat2 );  // returns false since mat1 and mat2 are two different matrices
   isSame( mat1, sub1 );  // returns true since sub1 represents the same observable state as mat1
   isSame( mat1, sub3 );  // returns false since sub3 only covers part of mat1
   isSame( sub2, sub3 );  // returns true since sub1 and sub2 refer to exactly the same part of mat1
   isSame( sub1, sub3 );  // returns false since sub1 and sub3 refer to different parts of mat1
   \endcode
*/
template< typename MT1  // Type of the left-hand side matrix
        , bool SO1      // Storage order of the left-hand side matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool isSame( const Matrix<MT1,SO1>& a, const Matrix<MT2,SO2>& b ) noexcept
{
   return ( IsSame<MT1,MT2>::value &&
            reinterpret_cast<const void*>( &a ) == reinterpret_cast<const void*>( &b ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the assignment of two matrices with the same storage order.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side matrix to be assigned.
// \return void
*/
template< typename MT1  // Type of the left-hand side matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO >     // Storage order of both matrices
BLAZE_ALWAYS_INLINE void assign_backend( Matrix<MT1,SO>& lhs, const Matrix<MT2,SO>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   (~lhs).assign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the assignment of two matrices with different storage order.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side matrix to be assigned.
// \return void
*/
template< typename MT1    // Type of the left-hand side matrix
        , bool SO         // Storage order of the left-hand side matrix
        , typename MT2 >  // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE DisableIf_< IsSymmetric<MT2> >
   assign_backend( Matrix<MT1,SO>& lhs, const Matrix<MT2,!SO>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT1 );

   (~lhs).assign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the assignment of a symmetric matrix to a matrix with
//        different storage order.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side matrix to be assigned.
// \return void
*/
template< typename MT1    // Type of the left-hand side matrix
        , bool SO         // Storage order of the left-hand side matrix
        , typename MT2 >  // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE EnableIf_< IsSymmetric<MT2> >
   assign_backend( Matrix<MT1,SO>& lhs, const Matrix<MT2,!SO>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT1 );

   BLAZE_INTERNAL_ASSERT( isSquare( ~rhs ), "Non-square symmetric matrix detected" );

   (~lhs).assign( trans( ~rhs ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a matrix to a matrix.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side matrix to be assigned.
// \return void
//
// This function implements the default assignment of a matrix to a matrix.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side matrix
        , bool SO1      // Storage order of the left-hand side matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
BLAZE_ALWAYS_INLINE void assign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   assign_backend( ~lhs, ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the addition assignment of two matrices with the same
//        storage order.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side matrix to be added.
// \return void
*/
template< typename MT1  // Type of the left-hand side matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO >     // Storage order of both matrices
BLAZE_ALWAYS_INLINE void addAssign_backend( Matrix<MT1,SO>& lhs, const Matrix<MT2,SO>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   (~lhs).addAssign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the addition assignment of two matrices with different
//        storage order.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side matrix to be added.
// \return void
*/
template< typename MT1    // Type of the left-hand side matrix
        , bool SO         // Storage order of the left-hand side matrix
        , typename MT2 >  // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE DisableIf_< IsSymmetric<MT2> >
   addAssign_backend( Matrix<MT1,SO>& lhs, const Matrix<MT2,!SO>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT1 );

   (~lhs).addAssign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the addition assignment of a symmetric matrix to a matrix
//        with different storage order.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side matrix to be added.
// \return void
*/
template< typename MT1    // Type of the left-hand side matrix
        , bool SO         // Storage order of the left-hand side matrix
        , typename MT2 >  // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE EnableIf_< IsSymmetric<MT2> >
   addAssign_backend( Matrix<MT1,SO>& lhs, const Matrix<MT2,!SO>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT1 );

   BLAZE_INTERNAL_ASSERT( isSquare( ~rhs ), "Non-square symmetric matrix detected" );

   (~lhs).addAssign( trans( ~rhs ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a matrix to a matrix.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side matrix to be added.
// \return void
//
// This function implements the default addition assignment of a matrix to a matrix.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side matrix
        , bool SO1      // Storage order of the left-hand side matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
BLAZE_ALWAYS_INLINE void addAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   addAssign_backend( ~lhs, ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the subtraction assignment of two matrices with the same
//        storage order.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side matrix to be subtracted.
// \return void
*/
template< typename MT1  // Type of the left-hand side matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO >     // Storage order of both matrices
BLAZE_ALWAYS_INLINE void subAssign_backend( Matrix<MT1,SO>& lhs, const Matrix<MT2,SO>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   (~lhs).subAssign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the subtraction assignment of two matrices with different
//        storage order.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side matrix to be subtracted.
// \return void
*/
template< typename MT1    // Type of the left-hand side matrix
        , bool SO         // Storage order of the left-hand side matrix
        , typename MT2 >  // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE DisableIf_< IsSymmetric<MT2> >
   subAssign_backend( Matrix<MT1,SO>& lhs, const Matrix<MT2,!SO>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT1 );

   (~lhs).subAssign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the subtraction assignment of a symmetric matrix to a matrix
//        with different storage order.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side matrix to be subtracted.
// \return void
*/
template< typename MT1    // Type of the left-hand side matrix
        , bool SO         // Storage order of the left-hand side matrix
        , typename MT2 >  // Type of the right-hand side matrix
BLAZE_ALWAYS_INLINE EnableIf_< IsSymmetric<MT2> >
   subAssign_backend( Matrix<MT1,SO>& lhs, const Matrix<MT2,!SO>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT1 );

   BLAZE_INTERNAL_ASSERT( isSquare( ~rhs ), "Non-square symmetric matrix detected" );

   (~lhs).subAssign( trans( ~rhs ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a matrix to matrix.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side matrix to be subtracted.
// \return void
//
// This function implements the default subtraction assignment of a matrix to a matrix.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side matrix
        , bool SO1      // Storage order of the left-hand side matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
BLAZE_ALWAYS_INLINE void subAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   subAssign_backend( ~lhs, ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the multiplication assignment of a matrix to a matrix.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side matrix to be multiplied.
// \return void
//
// This function implements the default multiplication assignment of a matrix to a matrix.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side matrix
        , bool SO1      // Storage order of the left-hand side matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
BLAZE_ALWAYS_INLINE void multAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).rows(), "Invalid matrix sizes" );

   (~lhs).multAssign( ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a vector to a matrix.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
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
template< typename MT  // Type of the left-hand side matrix
        , bool SO      // Storage order of the left-hand side matrix
        , typename VT  // Type of the right-hand side vector
        , bool TF >    // Transpose flag of the right-hand side vector
BLAZE_ALWAYS_INLINE bool tryAssign( const Matrix<MT,SO>& lhs, const Vector<VT,TF>& rhs,
                                    size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( TF || ( (~rhs).size() <= (~lhs).rows() - row ), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( !TF || ( (~rhs).size() <= (~lhs).columns() - column ), "Invalid number of columns" );

   UNUSED_PARAMETER( lhs, rhs, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a matrix to a matrix.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
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
template< typename MT1  // Type of the left-hand side matrix
        , bool SO1      // Storage order of the left-hand side matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool tryAssign( const Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs,
                                    size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= (~lhs).rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= (~lhs).columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs, rhs, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a vector to a matrix.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
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
template< typename MT  // Type of the left-hand side matrix
        , bool SO      // Storage order of the left-hand side matrix
        , typename VT  // Type of the right-hand side vector
        , bool TF >    // Transpose flag of the right-hand side vector
BLAZE_ALWAYS_INLINE bool tryAddAssign( const Matrix<MT,SO>& lhs, const Vector<VT,TF>& rhs,
                                       size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( TF || ( (~rhs).size() <= (~lhs).rows() - row ), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( !TF || ( (~rhs).size() <= (~lhs).columns() - column ), "Invalid number of columns" );

   UNUSED_PARAMETER( lhs, rhs, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a matrix to a matrix.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
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
template< typename MT1  // Type of the left-hand side matrix
        , bool SO1      // Storage order of the left-hand side matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool tryAddAssign( const Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs,
                                       size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= (~lhs).rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= (~lhs).columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs, rhs, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to a matrix.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
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
template< typename MT  // Type of the left-hand side matrix
        , bool SO      // Storage order of the left-hand side matrix
        , typename VT  // Type of the right-hand side vector
        , bool TF >    // Transpose flag of the right-hand side vector
BLAZE_ALWAYS_INLINE bool trySubAssign( const Matrix<MT,SO>& lhs, const Vector<VT,TF>& rhs,
                                       size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( TF || ( (~rhs).size() <= (~lhs).rows() - row ), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( !TF || ( (~rhs).size() <= (~lhs).columns() - column ), "Invalid number of columns" );

   UNUSED_PARAMETER( lhs, rhs, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a matrix to a matrix.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
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
template< typename MT1  // Type of the left-hand side matrix
        , bool SO1      // Storage order of the left-hand side matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
BLAZE_ALWAYS_INLINE bool trySubAssign( const Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs,
                                       size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= (~lhs).rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= (~lhs).columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs, rhs, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to a matrix.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
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
template< typename MT  // Type of the left-hand side matrix
        , bool SO      // Storage order of the left-hand side matrix
        , typename VT  // Type of the right-hand side vector
        , bool TF >    // Transpose flag of the right-hand side vector
BLAZE_ALWAYS_INLINE bool tryMultAssign( const Matrix<MT,SO>& lhs, const Vector<VT,TF>& rhs,
                                        size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( TF || ( (~rhs).size() <= (~lhs).rows() - row ), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( !TF || ( (~rhs).size() <= (~lhs).columns() - column ), "Invalid number of columns" );

   UNUSED_PARAMETER( lhs, rhs, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to a matrix.
// \ingroup matrix
//
// \param lhs The target left-hand side matrix.
// \param rhs The right-hand side vector divisor.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the left-hand side matrix
        , bool SO      // Storage order of the left-hand side matrix
        , typename VT  // Type of the right-hand side vector
        , bool TF >    // Transpose flag of the right-hand side vector
BLAZE_ALWAYS_INLINE bool tryDivAssign( const Matrix<MT,SO>& lhs, const Vector<VT,TF>& rhs,
                                       size_t row, size_t column )
{
   BLAZE_INTERNAL_ASSERT( row <= (~lhs).rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= (~lhs).columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( TF || ( (~rhs).size() <= (~lhs).rows() - row ), "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( !TF || ( (~rhs).size() <= (~lhs).columns() - column ), "Invalid number of columns" );

   UNUSED_PARAMETER( lhs, rhs, row, column );

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removal of all restrictions on the data access to the given matrix.
// \ingroup matrix
//
// \param matrix The matrix to be derestricted.
// \return Reference to the matrix without access restrictions.
//
// This function removes all restrictions on the data access to the given matrix. It returns a
// reference to the matrix that does provide the same interface but does not have any restrictions
// on the data access.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the matrix
        , bool SO >    // Storage order
BLAZE_ALWAYS_INLINE MT& derestrict( Matrix<MT,SO>& matrix )
{
   return ~matrix;
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
