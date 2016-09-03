//=================================================================================================
/*!
//  \file blaze/math/expressions/SMatTransposer.h
//  \brief Header file for the sparse matrix transposer
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

#ifndef _BLAZE_MATH_EXPRESSIONS_SMATTRANSPOSER_H_
#define _BLAZE_MATH_EXPRESSIONS_SMATTRANSPOSER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <vector>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/ColumnMajorMatrix.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/RowMajorMatrix.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/traits/SubmatrixTrait.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsNumeric.h>


namespace blaze {

//=================================================================================================
//
//  CLASS SMATTRANSPOSER
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for the transposition of a sparse matrix.
// \ingroup sparse_matrix_expression
//
// The SMatTransposer class is a wrapper object for the temporary transposition of a sparse matrix.
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
class SMatTransposer : public SparseMatrix< SMatTransposer<MT,SO>, SO >
{
 public:
   //**Type definitions****************************************************************************
   typedef SMatTransposer<MT,SO>  This;            //!< Type of this SMatTransposer instance.
   typedef TransposeType_<MT>     ResultType;      //!< Result type for expression template evaluations.
   typedef MT                     OppositeType;    //!< Result type with opposite storage order for expression template evaluations.
   typedef ResultType_<MT>        TransposeType;   //!< Transpose type for expression template evaluations.
   typedef ElementType_<MT>       ElementType;     //!< Resulting element type.
   typedef ReturnType_<MT>        ReturnType;      //!< Return type for expression template evaluations.
   typedef const This&            CompositeType;   //!< Data type for composite expression templates.
   typedef Reference_<MT>         Reference;       //!< Reference to a non-constant matrix value.
   typedef ConstReference_<MT>    ConstReference;  //!< Reference to a constant matrix value.
   typedef Iterator_<MT>          Iterator;        //!< Iterator over non-constant elements.
   typedef ConstIterator_<MT>     ConstIterator;   //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SMP assignments.
   /*! The \a smpAssignable compilation flag indicates whether the matrix can be used in SMP
       (shared memory parallel) assignments (both on the left-hand and right-hand side of the
       assignment). */
   enum : bool { smpAssignable = MT::smpAssignable };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SMatTransposer class.
   //
   // \param sm The sparse matrix operand.
   */
   explicit inline SMatTransposer( MT& sm ) noexcept
      : sm_( sm )  // The sparse matrix operand
   {}
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 2D-access to the matrix elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return Reference to the accessed value.
   */
   inline ConstReference operator()( size_t i, size_t j ) const {
      BLAZE_INTERNAL_ASSERT( i < sm_.columns(), "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < sm_.row()    , "Invalid column access index" );
      return sm_(j,i);
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
   inline ConstReference at( size_t i, size_t j ) const {
      if( i >= sm_.columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( j >= sm_.rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
      }
      return (*this)(i,j);
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator to the first non-zero element of row/column \a i.
   //
   // This function returns a row/column iterator to the first non-zero element of row/column \a i.
   // In case the storage order is set to \a rowMajor the function returns an iterator to the first
   // non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
   // returns an iterator to the first non-zero element of column \a i.
   */
   inline Iterator begin( size_t i ) {
      return sm_.begin(i);
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator to the first non-zero element of row/column \a i.
   //
   // This function returns a row/column iterator to the first non-zero element of row/column \a i.
   // In case the storage order is set to \a rowMajor the function returns an iterator to the first
   // non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
   // returns an iterator to the first non-zero element of column \a i.
   */
   inline ConstIterator begin( size_t i ) const {
      return sm_.cbegin(i);
   }
   //**********************************************************************************************

   //**Cbegin function*****************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator to the first non-zero element of row/column \a i.
   //
   // This function returns a row/column iterator to the first non-zero element of row/column \a i.
   // In case the storage order is set to \a rowMajor the function returns an iterator to the first
   // non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
   // returns an iterator to the first non-zero element of column \a i.
   */
   inline ConstIterator cbegin( size_t i ) const {
      return sm_.cbegin(i);
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator just past the last non-zero element of row/column \a i.
   //
   // This function returns an row/column iterator just past the last non-zero element of row/column
   // \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
   // past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
   // the function returns an iterator just past the last non-zero element of column \a i.
   */
   inline Iterator end( size_t i ) {
      return sm_.end(i);
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator just past the last non-zero element of row/column \a i.
   //
   // This function returns an row/column iterator just past the last non-zero element of row/column
   // \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
   // past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
   // the function returns an iterator just past the last non-zero element of column \a i.
   */
   inline ConstIterator end( size_t i ) const {
      return sm_.cend(i);
   }
   //**********************************************************************************************

   //**Cend function*******************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator just past the last non-zero element of row/column \a i.
   //
   // This function returns an row/column iterator just past the last non-zero element of row/column
   // \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
   // past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
   // the function returns an iterator just past the last non-zero element of column \a i.
   */
   inline ConstIterator cend( size_t i ) const {
      return sm_.cend(i);
   }
   //**********************************************************************************************

   //**Multiplication assignment operator**********************************************************
   /*!\brief Multiplication assignment operator for the multiplication between a matrix and
   //        a scalar value (\f$ A*=s \f$).
   //
   // \param rhs The right-hand side scalar value for the multiplication.
   // \return Reference to this SMatTransposer.
   */
   template< typename Other >  // Data type of the right-hand side scalar
   inline EnableIf_< IsNumeric<Other>, SMatTransposer >& operator*=( Other rhs )
   {
      (~sm_) *= rhs;
      return *this;
   }
   //**********************************************************************************************

   //**Division assignment operator****************************************************************
   /*!\brief Division assignment operator for the division of a matrix by a scalar value
   //        (\f$ A/=s \f$).
   //
   // \param rhs The right-hand side scalar value for the division.
   // \return Reference to this SMatTransposer.
   //
   // \note A division by zero is only checked by an user assert.
   */
   template< typename Other >  // Data type of the right-hand side scalar
   inline EnableIf_< IsNumeric<Other>, SMatTransposer >& operator/=( Other rhs )
   {
      BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

      (~sm_) /= rhs;
      return *this;
   }
   //**********************************************************************************************

   //**Rows function*******************************************************************************
   /*!\brief Returns the current number of rows of the matrix.
   //
   // \return The number of rows of the matrix.
   */
   inline size_t rows() const noexcept {
      return sm_.columns();
   }
   //**********************************************************************************************

   //**Columns function****************************************************************************
   /*!\brief Returns the current number of columns of the matrix.
   //
   // \return The number of columns of the matrix.
   */
   inline size_t columns() const noexcept {
      return sm_.rows();
   }
   //**********************************************************************************************

   //**Capacity function***************************************************************************
   /*!\brief Returns the maximum capacity of the matrix.
   //
   // \return The capacity of the matrix.
   */
   inline size_t capacity() const noexcept {
      return sm_.capacity();
   }
   //**********************************************************************************************

   //**Capacity function***************************************************************************
   /*!\brief Returns the current capacity of the specified row/column.
   //
   // \param i The index of the row/column.
   // \return The current capacity of row/column \a i.
   */
   inline size_t capacity( size_t i ) const noexcept {
      return sm_.capacity( i );
   }
   //**********************************************************************************************

   //**NonZeros function***************************************************************************
   /*!\brief Returns the number of non-zero elements in the matrix
   //
   // \return The number of non-zero elements in the matrix.
   */
   inline size_t nonZeros() const {
      return sm_.nonZeros();
   }
   //**********************************************************************************************

   //**NonZeros function***************************************************************************
   /*!\brief Returns the number of non-zero elements in the specified row/column.
   //
   // \param i The index of the row/column.
   // \return The number of non-zero elements of row/column \a i.
   */
   inline size_t nonZeros( size_t i ) const {
      return sm_.nonZeros( i );
   }
   //**********************************************************************************************

   //**Reset function******************************************************************************
   /*!\brief Resets the matrix elements.
   //
   // \return void
   */
   inline void reset() {
      return sm_.reset();
   }
   //**********************************************************************************************

   //**Insert function*****************************************************************************
   /*!\brief Inserting an element into the sparse matrix.
   //
   // \param i The row index of the new element. The index has to be in the range \f$[0..M-1]\f$.
   // \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
   // \param value The value of the element to be inserted.
   // \return Iterator to the newly inserted element.
   // \exception std::invalid_argument Invalid sparse matrix access index.
   //
   // This function insert a new element into the sparse matrix. However, duplicate elements are
   // not allowed. In case the sparse matrix already contains an element with row index \a i and
   // column index \a j, a \a std::invalid_argument exception is thrown.
   */
   inline Iterator insert( size_t i, size_t j, const ElementType& value ) {
      return sm_.insert( j, i, value );
   }
   //**********************************************************************************************

   //**Reserve function****************************************************************************
   /*!\brief Setting the minimum capacity of the sparse matrix.
   //
   // \param nonzeros The new minimum capacity of the sparse matrix.
   // \return void
   //
   // This function increases the capacity of the sparse matrix to at least \a nonzeros elements.
   // The current values of the matrix elements and the individual capacities of the matrix rows
   // are preserved.
   */
   inline void reserve( size_t nonzeros ) {
      sm_.reserve( nonzeros );
   }
   //**********************************************************************************************

   //**Reserve function****************************************************************************
   /*!\brief Setting the minimum capacity of a specific row/column of the sparse matrix.
   //
   // \param i The row/column index of the new element \f$[0..M-1]\f$ or \f$[0..N-1]\f$.
   // \param nonzeros The new minimum capacity of the specified row.
   // \return void
   //
   // This function increases the capacity of row/column \a i of the sparse matrix to at least
   // \a nonzeros elements. The current values of the sparse matrix and all other individual
   // row/column capacities are preserved. In case the storage order is set to \a rowMajor, the
   // function reserves capacity for row \a i and the index has to be in the range \f$[0..M-1]\f$.
   // In case the storage order is set to \a columnMajor, the function reserves capacity for column
   // \a i and the index has to be in the range \f$[0..N-1]\f$.
   */
   inline void reserve( size_t i, size_t nonzeros ) {
      sm_.reserve( i, nonzeros );
   }
   //**********************************************************************************************

   //**Append function*****************************************************************************
   /*!\brief Appending an element to the specified row/column of the sparse matrix.
   //
   // \param i The row index of the new element. The index has to be in the range \f$[0..M-1]\f$.
   // \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
   // \param value The value of the element to be appended.
   // \param check \a true if the new value should be checked for default values, \a false if not.
   // \return void
   //
   // This function provides a very efficient way to fill a sparse matrix with elements. It
   // appends a new element to the end of the specified row/column without any additional
   // memory allocation. Therefore it is strictly necessary to keep the following preconditions
   // in mind:
   //
   //  - the index of the new element must be strictly larger than the largest index of
   //    non-zero elements in the specified row/column of the sparse matrix
   //  - the current number of non-zero elements in row/column \a i must be smaller than
   //    the capacity of row/column \a i.
   //
   // Ignoring these preconditions might result in undefined behavior! The optional \a check
   // parameter specifies whether the new value should be tested for a default value. If the new
   // value is a default value (for instance 0 in case of an integral element type) the value is
   // not appended. Per default the values are not tested.
   //
   // \note Although append() does not allocate new memory, it still invalidates all iterators
   // returned by the end() functions!
   */
   inline void append( size_t i, size_t j, const ElementType& value, bool check=false ) {
      sm_.append( j, i, value, check );
   }
   //**********************************************************************************************

   //**Finalize function***************************************************************************
   /*!\brief Finalizing the element insertion of a row/column.
   //
   // \param i The index of the row/column to be finalized \f$[0..M-1]\f$.
   // \return void
   //
   // This function is part of the low-level interface to efficiently fill the matrix with elements.
   // After completion of row/column \a i via the append() function, this function can be called to
   // finalize row/column \a i and prepare the next row/column for insertion process via append().
   //
   // \note Although finalize() does not allocate new memory, it still invalidates all iterators
   // returned by the end() functions!
   */
   inline void finalize( size_t i ) {
      sm_.finalize( i );
   }
   //**********************************************************************************************

   //**IsIntact function***************************************************************************
   /*!\brief Returns whether the invariants of the matrix are intact.
   //
   // \return \a true in case the matrix's invariants are intact, \a false otherwise.
   */
   inline bool isIntact() const noexcept {
      return isIntact( sm_ );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the matrix can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the alias corresponds to this matrix, \a false if not.
   */
   template< typename Other >  // Data type of the foreign expression
   inline bool canAlias( const Other* alias ) const noexcept
   {
      return sm_.canAlias( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the matrix is aliased with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the alias corresponds to this matrix, \a false if not.
   */
   template< typename Other >  // Data type of the foreign expression
   inline bool isAliased( const Other* alias ) const noexcept
   {
      return sm_.isAliased( alias );
   }
   //**********************************************************************************************

   //**CanSMPAssign function***********************************************************************
   /*!\brief Returns whether the matrix can be used in SMP assignments.
   //
   // \return \a true in case the matrix can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept
   {
      return sm_.canSMPAssign();
   }
   //**********************************************************************************************

   //**Transpose assignment of row-major sparse matrices*******************************************
   /*!\brief Implementation of the transpose assignment of a row-major sparse matrix.
   //
   // \param rhs The right-hand side sparse matrix to be assigned.
   // \return void
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors. Instead of using this function use the
   // assignment operator.
   */
   template< typename MT2 >  // Type of the right-hand side sparse matrix
   inline void assign( const SparseMatrix<MT2,false>& rhs )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT2 );

      BLAZE_INTERNAL_ASSERT( sm_.columns() == (~rhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( sm_.rows() == (~rhs).columns()     , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( sm_.capacity() >= (~rhs).nonZeros(), "Capacity not sufficient"   );

      typedef ConstIterator_<MT2>  RhsIterator;

      const size_t m( rows() );

      for( size_t i=0UL; i<m; ++i ) {
         for( RhsIterator element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
            sm_.append( element->index(), i, element->value() );
         finalize( i );
      }
   }
   //**********************************************************************************************

   //**Transpose assignment of column-major sparse matrices****************************************
   /*!\brief Implementation of the transpose assignment of a column-major sparse matrix.
   //
   // \param rhs The right-hand side sparse matrix to be assigned.
   // \return void
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors. Instead of using this function use the
   // assignment operator.
   */
   template< typename MT2 >  // Type of the right-hand side sparse matrix
   inline void assign( const SparseMatrix<MT2,true>& rhs )
   {
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( MT2 );

      BLAZE_INTERNAL_ASSERT( sm_.columns() == (~rhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( sm_.rows() == (~rhs).columns()     , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( sm_.capacity() >= (~rhs).nonZeros(), "Capacity not sufficient"   );

      typedef ConstIterator_<MT2>  RhsIterator;

      const size_t m( rows() );
      const size_t n( columns() );

      // Counting the number of elements per row
      std::vector<size_t> rowLengths( m, 0UL );
      for( size_t j=0UL; j<n; ++j ) {
         for( RhsIterator element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
            ++rowLengths[element->index()];
      }

      // Resizing the sparse matrix
      for( size_t i=0UL; i<m; ++i ) {
         sm_.reserve( i, rowLengths[i] );
      }

      // Appending the elements to the rows of the sparse matrix
      for( size_t j=0UL; j<n; ++j ) {
         for( RhsIterator element=(~rhs).begin(j); element!=(~rhs).end(j); ++element ) {
            sm_.append( j, element->index(), element->value() );
         }
      }
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   MT& sm_;  //!< The sparse matrix operand.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ROW-MAJOR MATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of SMatTransposer for row-major matrices.
// \ingroup sparse_matrix_expression
//
// This specialization of SMatTransposer adapts the class template to the requirements of
// row-major matrices.
*/
template< typename MT >  // Type of the sparse matrix
class SMatTransposer<MT,true> : public SparseMatrix< SMatTransposer<MT,true>, true >
{
 public:
   //**Type definitions****************************************************************************
   typedef SMatTransposer<MT,true>  This;            //!< Type of this SMatTransposer instance.
   typedef TransposeType_<MT>       ResultType;      //!< Result type for expression template evaluations.
   typedef MT                       OppositeType;    //!< Result type with opposite storage order for expression template evaluations.
   typedef ResultType_<MT>          TransposeType;   //!< Transpose type for expression template evaluations.
   typedef ElementType_<MT>         ElementType;     //!< Resulting element type.
   typedef ReturnType_<MT>          ReturnType;      //!< Return type for expression template evaluations.
   typedef const This&              CompositeType;   //!< Data type for composite expression templates.
   typedef Reference_<MT>           Reference;       //!< Reference to a non-constant matrix value.
   typedef ConstReference_<MT>      ConstReference;  //!< Reference to a constant matrix value.
   typedef Iterator_<MT>            Iterator;        //!< Iterator over non-constant elements.
   typedef ConstIterator_<MT>       ConstIterator;   //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SMP assignments.
   /*! The \a smpAssignable compilation flag indicates whether the matrix can be used in SMP
       (shared memory parallel) assignments (both on the left-hand and right-hand side of the
       assignment). */
   enum : bool { smpAssignable = MT::smpAssignable };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SMatTransposer class.
   //
   // \param sm The sparse matrix operand.
   */
   explicit inline SMatTransposer( MT& sm ) noexcept
      : sm_( sm )  // The sparse matrix operand
   {}
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 2D-access to the matrix elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return Reference to the accessed value.
   */
   inline ConstReference operator()( size_t i, size_t j ) const {
      BLAZE_INTERNAL_ASSERT( i < sm_.columns(), "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < sm_.row()    , "Invalid column access index" );
      return sm_(j,i);
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
   inline ConstReference at( size_t i, size_t j ) const {
      if( i >= sm_.columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( j >= sm_.rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
      }
      return (*this)(i,j);
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of column \a j.
   //
   // \param j The column index.
   // \return Iterator to the first non-zero element of column \a j.
   */
   inline Iterator begin( size_t j ) {
      return sm_.begin(j);
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of column \a j.
   //
   // \param j The column index.
   // \return Iterator to the first non-zero element of column \a j.
   */
   inline ConstIterator begin( size_t j ) const {
      return sm_.cbegin(j);
   }
   //**********************************************************************************************

   //**Cbegin function*****************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of column \a j.
   //
   // \param j The column index.
   // \return Iterator to the first non-zero element of column \a j.
   */
   inline ConstIterator cbegin( size_t j ) const {
      return sm_.cbegin(j);
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of column \a j.
   //
   // \param j The column index.
   // \return Iterator just past the last non-zero element of column \a j.
   */
   inline Iterator end( size_t j ) {
      return sm_.end(j);
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of column \a j.
   //
   // \param j The column index.
   // \return Iterator just past the last non-zero element of column \a j.
   */
   inline ConstIterator end( size_t j ) const {
      return sm_.cend(j);
   }
   //**********************************************************************************************

   //**Cend function*******************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of column \a j.
   //
   // \param j The column index.
   // \return Iterator just past the last non-zero element of column \a j.
   */
   inline ConstIterator cend( size_t j ) const {
      return sm_.cend(j);
   }
   //**********************************************************************************************

   //**Multiplication assignment operator**********************************************************
   /*!\brief Multiplication assignment operator for the multiplication between a matrix and
   //        a scalar value (\f$ A*=s \f$).
   //
   // \param rhs The right-hand side scalar value for the multiplication.
   // \return Reference to this SMatTransposer.
   */
   template< typename Other >  // Data type of the right-hand side scalar
   inline EnableIf_< IsNumeric<Other>, SMatTransposer >& operator*=( Other rhs )
   {
      (~sm_) *= rhs;
      return *this;
   }
   //**********************************************************************************************

   //**Division assignment operator****************************************************************
   /*!\brief Division assignment operator for the division of a matrix by a scalar value
   //        (\f$ A/=s \f$).
   //
   // \param rhs The right-hand side scalar value for the division.
   // \return Reference to this SMatTransposer.
   //
   // \note A division by zero is only checked by an user assert.
   */
   template< typename Other >  // Data type of the right-hand side scalar
   inline EnableIf_< IsNumeric<Other>, SMatTransposer >& operator/=( Other rhs )
   {
      BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

      (~sm_) /= rhs;
      return *this;
   }
   //**********************************************************************************************

   //**Rows function*******************************************************************************
   /*!\brief Returns the current number of rows of the matrix.
   //
   // \return The number of rows of the matrix.
   */
   inline size_t rows() const noexcept {
      return sm_.columns();
   }
   //**********************************************************************************************

   //**Columns function****************************************************************************
   /*!\brief Returns the current number of columns of the matrix.
   //
   // \return The number of columns of the matrix.
   */
   inline size_t columns() const noexcept {
      return sm_.rows();
   }
   //**********************************************************************************************

   //**Capacity function***************************************************************************
   /*!\brief Returns the maximum capacity of the matrix.
   //
   // \return The capacity of the matrix.
   */
   inline size_t capacity() const noexcept {
      return sm_.capacity();
   }
   //**********************************************************************************************

   //**Capacity function***************************************************************************
   /*!\brief Returns the current capacity of the specified column.
   //
   // \param j The index of the column.
   // \return The current capacity of column \a j.
   */
   inline size_t capacity( size_t j ) const noexcept {
      return sm_.capacity( j );
   }
   //**********************************************************************************************

   //**NonZeros function***************************************************************************
   /*!\brief Returns the number of non-zero elements in the matrix
   //
   // \return The number of non-zero elements in the matrix.
   */
   inline size_t nonZeros() const {
      return sm_.nonZeros();
   }
   //**********************************************************************************************

   //**NonZeros function***************************************************************************
   /*!\brief Returns the number of non-zero elements in the specified column.
   //
   // \param j The index of the column.
   // \return The number of non-zero elements of column \a j.
   */
   inline size_t nonZeros( size_t j ) const {
      return sm_.nonZeros( j );
   }
   //**********************************************************************************************

   //**Reset function******************************************************************************
   /*!\brief Resets the matrix elements.
   //
   // \return void
   */
   inline void reset() {
      return sm_.reset();
   }
   //**********************************************************************************************

   //**Insert function*****************************************************************************
   /*!\brief Inserting an element into the sparse matrix.
   //
   // \param i The row index of the new element. The index has to be in the range \f$[0..M-1]\f$.
   // \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
   // \param value The value of the element to be inserted.
   // \return Iterator to the newly inserted element.
   // \exception std::invalid_argument Invalid sparse matrix access index.
   //
   // This function insert a new element into the sparse matrix. However, duplicate elements are
   // not allowed. In case the sparse matrix already contains an element with row index \a i and
   // column index \a j, a \a std::invalid_argument exception is thrown.
   */
   inline Iterator insert( size_t i, size_t j, const ElementType& value ) {
      return sm_.insert( j, i, value );
   }
   //**********************************************************************************************

   //**Reserve function****************************************************************************
   /*!\brief Setting the minimum capacity of the sparse matrix.
   //
   // \param nonzeros The new minimum capacity of the sparse matrix.
   // \return void
   //
   // This function increases the capacity of the sparse matrix to at least \a nonzeros elements.
   // The current values of the matrix elements and the individual capacities of the matrix rows
   // are preserved.
   */
   inline void reserve( size_t nonzeros ) {
      sm_.reserve( nonzeros );
   }
   //**********************************************************************************************

   //**Reserve function****************************************************************************
   /*!\brief Setting the minimum capacity of a specific column of the sparse matrix.
   //
   // \param j The column index of the new element \f$[0..N-1]\f$.
   // \param nonzeros The new minimum capacity of the specified row.
   // \return void
   //
   // This function increases the capacity of column \a j of the sparse matrix to at least
   // \a nonzeros elements. The current values of the sparse matrix and all other individual
   // column capacities are preserved.
   */
   inline void reserve( size_t i, size_t nonzeros ) {
      sm_.reserve( i, nonzeros );
   }
   //**********************************************************************************************

   //**Append function*****************************************************************************
   /*!\brief Appending an element to the specified column of the sparse matrix.
   //
   // \param i The row index of the new element. The index has to be in the range \f$[0..M-1]\f$.
   // \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
   // \param value The value of the element to be appended.
   // \param check \a true if the new value should be checked for default values, \a false if not.
   // \return void
   //
   // This function provides a very efficient way to fill a sparse matrix with elements. It
   // appends a new element to the end of the specified column without any additional memory
   // allocation. Therefore it is strictly necessary to keep the following preconditions in
   // mind:
   //
   //  - the index of the new element must be strictly larger than the largest index of non-zero
   //    elements in the specified column of the sparse matrix
   //  - the current number of non-zero elements in column \a j must be smaller than the capacity
   //    of column \a j.
   //
   // Ignoring these preconditions might result in undefined behavior! The optional \a check
   // parameter specifies whether the new value should be tested for a default value. If the new
   // value is a default value (for instance 0 in case of an integral element type) the value is
   // not appended. Per default the values are not tested.
   //
   // \note Although append() does not allocate new memory, it still invalidates all iterators
   // returned by the end() functions!
   */
   void append( size_t i, size_t j, const ElementType& value, bool check=false ) {
      sm_.append( j, i, value, check );
   }
   //**********************************************************************************************

   //**Finalize function***************************************************************************
   /*!\brief Finalizing the element insertion of a column.
   //
   // \param i The index of the column to be finalized \f$[0..M-1]\f$.
   // \return void
   //
   // This function is part of the low-level interface to efficiently fill the matrix with elements.
   // After completion of column \a i via the append() function, this function can be called to
   // finalize column \a i and prepare the next row/column for insertion process via append().
   //
   // \note Although finalize() does not allocate new memory, it still invalidates all iterators
   // returned by the end() functions!
   */
   inline void finalize( size_t j ) {
      sm_.finalize( j );
   }
   //**********************************************************************************************

   //**IsIntact function***************************************************************************
   /*!\brief Returns whether the invariants of the matrix are intact.
   //
   // \return \a true in case the matrix's invariants are intact, \a false otherwise.
   */
   inline bool isIntact() const noexcept {
      return isIntact( sm_ );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the matrix can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the alias corresponds to this matrix, \a false if not.
   */
   template< typename Other >  // Data type of the foreign expression
   inline bool canAlias( const Other* alias ) const noexcept
   {
      return sm_.canAlias( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the matrix is aliased with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the alias corresponds to this matrix, \a false if not.
   */
   template< typename Other >  // Data type of the foreign expression
   inline bool isAliased( const Other* alias ) const noexcept
   {
      return sm_.isAliased( alias );
   }
   //**********************************************************************************************

   //**CanSMPAssign function***********************************************************************
   /*!\brief Returns whether the matrix can be used in SMP assignments.
   //
   // \return \a true in case the matrix can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept
   {
      return sm_.canSMPAssign();
   }
   //**********************************************************************************************

   //**Transpose assignment of row-major sparse matrices*******************************************
   /*!\brief Implementation of the transpose assignment of a row-major sparse matrix.
   //
   // \param rhs The right-hand side sparse matrix to be assigned.
   // \return void
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors. Instead of using this function use the
   // assignment operator.
   */
   template< typename MT2 >  // Type of the right-hand side sparse matrix
   inline void assign( const SparseMatrix<MT2,false>& rhs )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT2 );

      BLAZE_INTERNAL_ASSERT( sm_.columns() == (~rhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( sm_.rows() == (~rhs).columns()     , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( sm_.capacity() >= (~rhs).nonZeros(), "Capacity not sufficient"   );

      typedef ConstIterator_<MT2>  RhsIterator;

      const size_t m( rows() );
      const size_t n( columns() );

      // Counting the number of elements per row
      std::vector<size_t> columnLengths( n, 0UL );
      for( size_t i=0UL; i<m; ++i ) {
         for( RhsIterator element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
            ++columnLengths[element->index()];
      }

      // Resizing the sparse matrix
      for( size_t j=0UL; j<n; ++j ) {
         sm_.reserve( j, columnLengths[j] );
      }

      // Appending the elements to the columns of the sparse matrix
      for( size_t i=0UL; i<m; ++i ) {
         for( RhsIterator element=(~rhs).begin(i); element!=(~rhs).end(i); ++element ) {
            sm_.append( element->index(), i, element->value() );
         }
      }
   }
   //**********************************************************************************************

   //**Transpose assignment of column-major sparse matrices****************************************
   /*!\brief Implementation of the transpose assignment of a column-major sparse matrix.
   //
   // \param rhs The right-hand side sparse matrix to be assigned.
   // \return void
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors. Instead of using this function use the
   // assignment operator.
   */
   template< typename MT2 >  // Type of the right-hand side sparse matrix
   inline void assign( const SparseMatrix<MT2,true>& rhs )
   {
      BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( MT2 );

      BLAZE_INTERNAL_ASSERT( sm_.columns() == (~rhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( sm_.rows() == (~rhs).columns()     , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( sm_.capacity() >= (~rhs).nonZeros(), "Capacity not sufficient"   );

      typedef ConstIterator_<MT2>  RhsIterator;

      const size_t n( columns() );

      for( size_t j=0UL; j<n; ++j ) {
         for( RhsIterator element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
            sm_.append( j, element->index(), element->value() );
         finalize( j );
      }
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   MT& sm_;  //!< The sparse matrix operand.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT );
   /*! \endcond */
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the sparse matrix contained in a SMatTransposer.
// \ingroup sparse_matrix_expression
//
// \param m The sparse matrix to be resetted.
// \return void
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline void reset( SMatTransposer<MT,SO>& m )
{
   m.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the given SMatTransposer are intact.
// \ingroup sparse_matrix_expression
//
// \param m The sparse matrix to be tested.
// \return \a true in caes the given matrix's invariants are intact, \a false otherwise.
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
inline bool isIntact( const SMatTransposer<MT,SO>& m ) noexcept
{
   return m.isIntact();
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBMATRIXTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO >
struct SubmatrixTrait< SMatTransposer<MT,SO> >
{
   using Type = SubmatrixTrait_< ResultType_< SMatTransposer<MT,SO> > >;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
