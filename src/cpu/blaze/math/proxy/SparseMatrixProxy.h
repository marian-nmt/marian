//=================================================================================================
/*!
//  \file blaze/math/proxy/SparseMatrixProxy.h
//  \brief Header file for the SparseMatrixProxy class
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

#ifndef _BLAZE_MATH_PROXY_SPARSEMATRIXPROXY_H_
#define _BLAZE_MATH_PROXY_SPARSEMATRIXPROXY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/typetraits/IsSquare.h>
#include <blaze/math/typetraits/IsColumnMajorMatrix.h>
#include <blaze/system/Inline.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Proxy backend for sparse matrix types.
// \ingroup math
//
// The SparseMatrixProxy class serves as a backend for the Proxy class. It is used in case the
// data type represented by the proxy is a sparse matrix and augments the Proxy interface by
// the complete interface required of sparse matrices.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
class SparseMatrixProxy : public SparseMatrix< PT, IsColumnMajorMatrix<MT>::value >
{
 public:
   //**Type definitions****************************************************************************
   typedef ResultType_<MT>      ResultType;      //!< Result type for expression template evaluations.
   typedef OppositeType_<MT>    OppositeType;    //!< Result type with opposite storage order for expression template evaluations.
   typedef TransposeType_<MT>   TransposeType;   //!< Transpose type for expression template evaluations.
   typedef ElementType_<MT>     ElementType;     //!< Type of the sparse matrix elements.
   typedef ReturnType_<MT>      ReturnType;      //!< Return type for expression template evaluations.
   typedef CompositeType_<MT>   CompositeType;   //!< Data type for composite expression templates.
   typedef Reference_<MT>       Reference;       //!< Reference to a non-constant matrix value.
   typedef ConstReference_<MT>  ConstReference;  //!< Reference to a constant matrix value.
   typedef Iterator_<MT>        Iterator;        //!< Iterator over non-constant elements.
   typedef ConstIterator_<MT>   ConstIterator;   //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SMP assignments.
   enum : bool { smpAssignable = MT::smpAssignable };
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference operator()( size_t i, size_t j ) const;
   inline Reference at( size_t i, size_t j ) const;

   inline Iterator      begin ( size_t i ) const;
   inline ConstIterator cbegin( size_t i ) const;
   inline Iterator      end   ( size_t i ) const;
   inline ConstIterator cend  ( size_t i ) const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t   rows() const;
   inline size_t   columns() const;
   inline size_t   capacity() const;
   inline size_t   capacity( size_t i ) const;
   inline size_t   nonZeros() const;
   inline size_t   nonZeros( size_t i ) const;
   inline void     reset() const;
   inline void     reset( size_t i ) const;
   inline void     clear() const;
   inline Iterator set( size_t i, size_t j, const ElementType& value ) const;
   inline Iterator insert( size_t i, size_t j, const ElementType& value ) const;
   inline void     append( size_t i, size_t j, const ElementType& value, bool check=false ) const;
   inline void     finalize( size_t i ) const;
   inline void     erase( size_t i, size_t j ) const;
   inline Iterator erase( size_t i, Iterator pos ) const;
   inline Iterator erase( size_t i, Iterator first, Iterator last ) const;
   inline void     resize( size_t m, size_t n, bool preserve=true ) const;
   inline void     reserve( size_t n ) const;
   inline void     reserve( size_t i, size_t n ) const;
   inline void     trim() const;
   inline void     trim( size_t i ) const;
   inline void     transpose() const;
   inline void     ctranspose() const;

   template< typename Other > inline void scale( const Other& scalar ) const;
   //@}
   //**********************************************************************************************

   //**Lookup functions****************************************************************************
   /*!\name Lookup functions */
   //@{
   inline Iterator find      ( size_t i, size_t j ) const;
   inline Iterator lowerBound( size_t i, size_t j ) const;
   inline Iterator upperBound( size_t i, size_t j ) const;
   //@}
   //**********************************************************************************************

 private:
   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( MT );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Function call operator for the direct access to matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function returns a reference to the accessed value at position (\a i,\a j). In case
// the sparse matrix does not yet store an element at position (\a i,\a j) , a new element is
// inserted into the sparse matrix. Note that this function only performs an index check in
// case BLAZE_USER_ASSERT() is active. In contrast, the at() function is guaranteed to perform
// a check of the given access indices.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline typename SparseMatrixProxy<PT,MT>::Reference
   SparseMatrixProxy<PT,MT>::operator()( size_t i, size_t j ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get()(i,j);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::invalid_argument Invalid access to restricted element.
// \exception std::out_of_range Invalid matrix access index.
//
// This function returns a reference to the accessed value at position (\a i,\a j). In case
// the sparse matrix does not yet store an element at position (\a i,\a j) , a new element is
// inserted into the sparse matrix. In contrast to the subscript operator this function always
// performs a check of the given access indices.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline typename SparseMatrixProxy<PT,MT>::Reference
   SparseMatrixProxy<PT,MT>::at( size_t i, size_t j ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().at(i,j);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i of the represented matrix.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first non-zero element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator to the first
// non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator to the first non-zero element of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline typename SparseMatrixProxy<PT,MT>::Iterator
   SparseMatrixProxy<PT,MT>::begin( size_t i ) const
{
   return (~*this).get().begin(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i of the represented matrix.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first non-zero element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator to the first
// non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator to the first non-zero element of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline typename SparseMatrixProxy<PT,MT>::ConstIterator
   SparseMatrixProxy<PT,MT>::cbegin( size_t i ) const
{
   return (~*this).get().cbegin(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i of the represented matrix.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last non-zero element of row/column
// \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
// past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
// the function returns an iterator just past the last non-zero element of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline typename SparseMatrixProxy<PT,MT>::Iterator
   SparseMatrixProxy<PT,MT>::end( size_t i ) const
{
   return (~*this).get().end(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i of the represented matrix.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last non-zero element of row/column
// \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
// past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
// the function returns an iterator just past the last non-zero element of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline typename SparseMatrixProxy<PT,MT>::ConstIterator
   SparseMatrixProxy<PT,MT>::cend( size_t i ) const
{
   return (~*this).get().cend(i);
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the current number of rows of the represented matrix.
//
// \return The number of rows of the matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline size_t SparseMatrixProxy<PT,MT>::rows() const
{
   return (~*this).get().rows();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of columns of the represented matrix.
//
// \return The number of columns of the matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline size_t SparseMatrixProxy<PT,MT>::columns() const
{
   return (~*this).get().columns();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the represented matrix.
//
// \return The capacity of the matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline size_t SparseMatrixProxy<PT,MT>::capacity() const
{
   return (~*this).get().capacity();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current capacity of the specified row/column of the represented matrix.
//
// \param i The index of the row/column.
// \return The current capacity of row/column \a i.
//
// This function returns the current capacity of the specified row/column. In case the
// storage order is set to \a rowMajor the function returns the capacity of row \a i,
// in case the storage flag is set to \a columnMajor the function returns the capacity
// of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline size_t SparseMatrixProxy<PT,MT>::capacity( size_t i ) const
{
   return (~*this).get().capacity(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the represented matrix.
//
// \return The number of non-zero elements in the matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline size_t SparseMatrixProxy<PT,MT>::nonZeros() const
{
   return (~*this).get().nonZeros();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the specified row/column of the represented matrix.
//
// \param i The index of the row/column.
// \return The number of non-zero elements of row/column \a i.
//
// This function returns the current number of non-zero elements in the specified row/column.
// In case the storage order is set to \a rowMajor the function returns the number of non-zero
// elements in row \a i, in case the storage flag is set to \a columnMajor the function returns
// the number of non-zero elements in column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline size_t SparseMatrixProxy<PT,MT>::nonZeros( size_t i ) const
{
   return (~*this).get().nonZeros(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset to the default initial value.
//
// \return void
//
// This function resets all elements of the matrix to the default initial values.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline void SparseMatrixProxy<PT,MT>::reset() const
{
   using blaze::reset;

   reset( (~*this).get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset the specified row/column to the default initial values.
//
// \param i The index of the row/column.
// \return void
//
// This function resets the values in the specified row/column to their default value. In case
// the storage order is set to \a rowMajor the function resets the values in row \a i, in case
// the storage order is set to \a columnMajor the function resets the values in column \a i.
// Note that the capacity of the row/column remains unchanged.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline void SparseMatrixProxy<PT,MT>::reset( size_t i ) const
{
   using blaze::reset;

   reset( (~*this).get(), i );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the represented vector.
//
// \return void
//
// This function clears the matrix to its default initial state.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline void SparseMatrixProxy<PT,MT>::clear() const
{
   using blaze::clear;

   clear( (~*this).get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting an element of the represented sparse matrix.
//
// \param i The row index of the new element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be set.
// \return Iterator to the set element.
// \exception std::invalid_argument Invalid access to restricted element.
// \exception std::invalid_argument Invalid sparse matrix access index.
//
// This function sets the value of an element of the sparse matrix. In case the sparse matrix
// already contains an element with row index \a i and column index \a j its value is modified,
// else a new element with the given \a value is inserted.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline typename SparseMatrixProxy<PT,MT>::Iterator
   SparseMatrixProxy<PT,MT>::set( size_t i, size_t j, const ElementType& value ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().set( i, j, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inserting an element into the represented sparse matrix.
//
// \param i The row index of the new element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be inserted.
// \return Iterator to the newly inserted element.
// \exception std::invalid_argument Invalid access to restricted element.
// \exception std::invalid_argument Invalid sparse matrix access index.
//
// This function inserts a new element into the sparse matrix. However, duplicate elements are
// not allowed. In case the sparse matrix already contains an element with row index \a i and
// column index \a j, a \a std::invalid_argument exception is thrown.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline typename SparseMatrixProxy<PT,MT>::Iterator
   SparseMatrixProxy<PT,MT>::insert( size_t i, size_t j, const ElementType& value ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().insert( i, j, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Appending an element to the specified row/column of the sparse matrix.
//
// \param i The row index of the new element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be appended.
// \param check \a true if the new value should be checked for default values, \a false if not.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function provides a very efficient way to fill a sparse matrix with elements. It appends
// a new element to the end of the specified row/column without any additional memory allocation.
// Therefore it is strictly necessary to keep the following preconditions in mind:
//
//  - the index of the new element must be strictly larger than the largest index of non-zero
//    elements in the specified row/column of the sparse matrix
//  - the current number of non-zero elements in the matrix must be smaller than the capacity
//    of the matrix
//
// Ignoring these preconditions might result in undefined behavior! The optional \a check
// parameter specifies whether the new value should be tested for a default value. If the new
// value is a default value (for instance 0 in case of an integral element type) the value is
// not appended. Per default the values are not tested.
//
// \note Although append() does not allocate new memory, it still invalidates all iterators
// returned by the end() functions!
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline void SparseMatrixProxy<PT,MT>::append( size_t i, size_t j, const ElementType& value, bool check ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().append( i, j, value, check );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Finalizing the element insertion of a row/column.
//
// \param i The index of the row/column to be finalized \f$[0..M-1]\f$.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function is part of the low-level interface to efficiently fill a matrix with elements.
// After completion of row/column \a i via the append() function, this function can be called to
// finalize row/column \a i and prepare the next row/column for insertion process via append().
//
// \note Although finalize() does not allocate new memory, it still invalidates all iterators
// returned by the end() functions!
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline void SparseMatrixProxy<PT,MT>::finalize( size_t i ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().finalize( i );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Erasing an element from the sparse matrix.
//
// \param i The row index of the element to be erased. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the element to be erased. The index has to be in the range \f$[0..N-1]\f$.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function erases an element from the sparse matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline void SparseMatrixProxy<PT,MT>::erase( size_t i, size_t j ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().erase( i, j );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Erasing an element from the sparse matrix.
//
// \param i The row/column index of the element to be erased. The index has to be in the range \f$[0..M-1]\f$.
// \param pos Iterator to the element to be erased.
// \return Iterator to the element after the erased element.
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function erases an element from the sparse matrix. In case the storage order is set to
// \a rowMajor the function erases an element from row \a i, in case the storage flag is set to
// \a columnMajor the function erases an element from column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline typename SparseMatrixProxy<PT,MT>::Iterator
   SparseMatrixProxy<PT,MT>::erase( size_t i, Iterator pos ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().erase( i, pos );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Erasing a range of elements from the sparse matrix.
//
// \param i The row/column index of the element to be erased. The index has to be in the range \f$[0..M-1]\f$.
// \param first Iterator to first element to be erased.
// \param last Iterator just past the last element to be erased.
// \return Iterator to the element after the erased element.
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function erases a range of element from the sparse matrix. In case the storage order is
// set to \a rowMajor the function erases a range of elements from row \a i, in case the storage
// flag is set to \a columnMajor the function erases a range of elements from column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline typename SparseMatrixProxy<PT,MT>::Iterator
   SparseMatrixProxy<PT,MT>::erase( size_t i, Iterator first, Iterator last ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().erase( i, first, last );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the represented matrix.
//
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function resizes the matrix using the given size to \f$ m \times n \f$. Depending on
// the type of the matrix, during this operation new dynamic memory may be allocated in case
// the capacity of the matrix is too small. Note that this function may invalidate all existing
// views (submatrices, rows, columns, ...) on the matrix if it is used to shrink the matrix.
// Additionally, the resize operation potentially changes all matrix elements. In order to
// preserve the old matrix values, the \a preserve flag can be set to \a true.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline void SparseMatrixProxy<PT,MT>::resize( size_t m, size_t n, bool preserve ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().resize( m, n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the minimum capacity of the represented matrix.
//
// \param n The new minimum capacity of the matrix.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function increases the capacity of the sparse matrix to at least \a nonzeros elements.
// The current values of the matrix elements and the individual capacities of the matrix rows
// are preserved.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline void SparseMatrixProxy<PT,MT>::reserve( size_t n ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().reserve( n );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the minimum capacity of a specific row/column of the sparse matrix.
//
// \param i The row/column index of the new element \f$[0..M-1]\f$ or \f$[0..N-1]\f$.
// \param n The new minimum capacity of the specified row/column.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function increases the capacity of row/column \a i of the sparse matrix to at least
// \a nonzeros elements. The current values of the sparse matrix and all other individual
// row/column capacities are preserved. In case the storage order is set to \a rowMajor, the
// function reserves capacity for row \a i and the index has to be in the range \f$[0..M-1]\f$.
// In case the storage order is set to \a columnMajor, the function reserves capacity for column
// \a i and the index has to be in the range \f$[0..N-1]\f$.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline void SparseMatrixProxy<PT,MT>::reserve( size_t i, size_t n ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().reserve( i, n );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Removing all excessive capacity from all rows/columns.
//
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// The trim() function can be used to reverse the effect of all row/column-specific reserve()
// calls. The function removes all excessive capacity from all rows (in case of a rowMajor
// matrix) or columns (in case of a columnMajor matrix). Note that this function does not
// remove the overall capacity but only reduces the capacity per row/column.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline void SparseMatrixProxy<PT,MT>::trim() const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().trim();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Removing all excessive capacity of a specific row/column of the sparse matrix.
//
// \param i The index of the row/column to be trimmed (\f$[0..M-1]\f$ or \f$[0..N-1]\f$).
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function can be used to reverse the effect of a row/column-specific reserve() call.
// It removes all excessive capacity from the specified row (in case of a rowMajor matrix)
// or column (in case of a columnMajor matrix). The excessive capacity is assigned to the
// subsequent row/column.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline void SparseMatrixProxy<PT,MT>::trim( size_t i ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().trim( i );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the represented matrix.
//
// \return Reference to the transposed matrix.
// \exception std::invalid_argument Invalid access to restricted element.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline void SparseMatrixProxy<PT,MT>::transpose() const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().transpose();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the represented matrix.
//
// \return Reference to the transposed matrix.
// \exception std::invalid_argument Invalid access to restricted element.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline void SparseMatrixProxy<PT,MT>::ctranspose() const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().ctranspose();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the sparse matrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the matrix scaling.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
*/
template< typename PT       // Type of the proxy
        , typename MT >     // Type of the sparse matrix
template< typename Other >  // Data type of the scalar value
inline void SparseMatrixProxy<PT,MT>::scale( const Other& scalar ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().scale( scalar );
}
//*************************************************************************************************




//=================================================================================================
//
//  LOOKUP FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Searches for a specific matrix element.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the element in case the index is found, end() iterator otherwise.
//
// This function can be used to check whether a specific element is contained in the sparse
// matrix. It specifically searches for the element with row index \a i and column index \a j.
// In case the element is found, the function returns an row/column iterator to the element.
// Otherwise an iterator just past the last non-zero element of row \a i or column \a j (the
// end() iterator) is returned. Note that the returned sparse matrix iterator is subject to
// invalidation due to inserting operations via the function call operator or the insert()
// function!
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline typename SparseMatrixProxy<PT,MT>::Iterator
   SparseMatrixProxy<PT,MT>::find( size_t i, size_t j ) const
{
   return (~*this).get().find( i, j );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first index not less then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index not less then the given index, end() iterator otherwise.
//
// In case of a row-major matrix, this function returns a row iterator to the first element with
// an index not less then the given column index. In case of a column-major matrix, the function
// returns a column iterator to the first element with an index not less then the given row
// index. In combination with the upperBound() function this function can be used to create a
// pair of iterators specifying a range of indices. Note that the returned compressed matrix
// iterator is subject to invalidation due to inserting operations via the function call operator
// or the insert() function!
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline typename SparseMatrixProxy<PT,MT>::Iterator
   SparseMatrixProxy<PT,MT>::lowerBound( size_t i, size_t j ) const
{
   return (~*this).get().lowerBound( i, j );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first index greater then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index greater then the given index, end() iterator otherwise.
//
// In case of a row-major matrix, this function returns a row iterator to the first element with
// an index greater then the given column index. In case of a column-major matrix, the function
// returns a column iterator to the first element with an index greater then the given row
// index. In combination with the upperBound() function this function can be used to create a
// pair of iterators specifying a range of indices. Note that the returned compressed matrix
// iterator is subject to invalidation due to inserting operations via the function call operator
// or the insert() function!
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
inline typename SparseMatrixProxy<PT,MT>::Iterator
   SparseMatrixProxy<PT,MT>::upperBound( size_t i, size_t j ) const
{
   return (~*this).get().upperBound( i, j );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name SparseMatrixProxy global functions */
//@{
template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE typename SparseMatrixProxy<PT,MT>::Iterator
   begin( const SparseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE typename SparseMatrixProxy<PT,MT>::ConstIterator
   cbegin( const SparseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE typename SparseMatrixProxy<PT,MT>::Iterator
   end( const SparseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE typename SparseMatrixProxy<PT,MT>::ConstIterator
   cend( const SparseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE size_t rows( const SparseMatrixProxy<PT,MT>& proxy );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE size_t columns( const SparseMatrixProxy<PT,MT>& proxy );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE size_t capacity( const SparseMatrixProxy<PT,MT>& proxy );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE size_t capacity( const SparseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE size_t nonZeros( const SparseMatrixProxy<PT,MT>& proxy );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE size_t nonZeros( const SparseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE void resize( const SparseMatrixProxy<PT,MT>& proxy, size_t m, size_t n, bool preserve=true );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE void reset( const SparseMatrixProxy<PT,MT>& proxy );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE void reset( const SparseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE void clear( const SparseMatrixProxy<PT,MT>& proxy );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i of the represented matrix.
// \ingroup math
//
// \param proxy The given access proxy.
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the given matrix is a row-major matrix the function returns an iterator to the first element
// of row \a i, in case it is a column-major matrix the function returns an iterator to the first
// element of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE typename SparseMatrixProxy<PT,MT>::Iterator
   begin( const SparseMatrixProxy<PT,MT>& proxy, size_t i )
{
   return proxy.begin(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i of the represented matrix.
// \ingroup math
//
// \param proxy The given access proxy.
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the given matrix is a row-major matrix the function returns an iterator to the first element
// of row \a i, in case it is a column-major matrix the function returns an iterator to the first
// element of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE typename SparseMatrixProxy<PT,MT>::ConstIterator
   cbegin( const SparseMatrixProxy<PT,MT>& proxy, size_t i )
{
   return proxy.cbegin(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i of the represented matrix.
// \ingroup math
//
// \param proxy The given access proxy.
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// In case the access proxy represents a matrix-like data structure that provides an end()
// function, this function returns an iterator just past the last element of row/column \a i of
// the matrix. In case the given matrix is a row-major matrix the function returns an iterator
// just past the last element of row \a i, in case it is a column-major matrix the function
// returns an iterator just past the last element of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE typename SparseMatrixProxy<PT,MT>::Iterator
   end( const SparseMatrixProxy<PT,MT>& proxy, size_t i )
{
   return proxy.end(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i of the represented matrix.
// \ingroup math
//
// \param proxy The given access proxy.
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// In case the access proxy represents a matrix-like data structure that provides a cend()
// function, this function returns an iterator just past the last element of row/column \a i of
// the matrix. In case the given matrix is a row-major matrix the function returns an iterator
// just past the last element of row \a i, in case it is a column-major matrix the function
// returns an iterator just past the last element of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE typename SparseMatrixProxy<PT,MT>::ConstIterator
   cend( const SparseMatrixProxy<PT,MT>& proxy, size_t i )
{
   return proxy.cend(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of rows of the represented matrix.
// \ingroup math
//
// \param proxy The given access proxy.
// \return The number of rows of the matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE size_t rows( const SparseMatrixProxy<PT,MT>& proxy )
{
   return proxy.rows();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of columns of the represented matrix.
// \ingroup math
//
// \param proxy The given access proxy.
// \return The number of columns of the matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE size_t columns( const SparseMatrixProxy<PT,MT>& proxy )
{
   return proxy.columns();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the represented matrix.
// \ingroup math
//
// \param proxy The given access proxy.
// \return The capacity of the matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE size_t capacity( const SparseMatrixProxy<PT,MT>& proxy )
{
   return proxy.capacity();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current capacity of the specified row/column of the represented matrix.
// \ingroup math
//
// \param proxy The given access proxy.
// \param i The index of the row/column.
// \return The current capacity of row/column \a i.
//
// This function returns the current capacity of the specified row/column. In case the
// storage order is set to \a rowMajor the function returns the capacity of row \a i,
// in case the storage flag is set to \a columnMajor the function returns the capacity
// of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE size_t capacity( const SparseMatrixProxy<PT,MT>& proxy, size_t i )
{
   return proxy.capacity(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the represented matrix.
// \ingroup math
//
// \param proxy The given access proxy.
// \return The number of non-zero elements in the matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE size_t nonZeros( const SparseMatrixProxy<PT,MT>& proxy )
{
   return proxy.nonZeros();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the specified row/column of the represented matrix.
// \ingroup math
//
// \param proxy The given access proxy.
// \param i The index of the row/column.
// \return The number of non-zero elements of row/column \a i.
//
// This function returns the current number of non-zero elements in the specified row/column.
// In case the storage order is set to \a rowMajor the function returns the number of non-zero
// elements in row \a i, in case the storage flag is set to \a columnMajor the function returns
// the number of non-zero elements in column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE size_t nonZeros( const SparseMatrixProxy<PT,MT>& proxy, size_t i )
{
   return proxy.nonZeros(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for non-square matrices.
// \ingroup math
//
// \param proxy The given access proxy.
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function changes the number of rows and columns of the given non-square matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE DisableIf_< IsSquare<MT> >
   resize_backend( const SparseMatrixProxy<PT,MT>& proxy, size_t m, size_t n, bool preserve )
{
   proxy.resize( m, n, preserve );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for square matrices.
// \ingroup math
//
// \param proxy The given access proxy.
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Invalid resize arguments for square matrix.
//
// This function changes the number of rows and columns of the given square matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE EnableIf_< IsSquare<MT> >
   resize_backend( const SparseMatrixProxy<PT,MT>& proxy, size_t m, size_t n, bool preserve )
{
   if( m != n ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid resize arguments for square matrix" );
   }

   proxy.resize( m, preserve );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the represented matrix.
// \ingroup math
//
// \param proxy The given access proxy.
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Invalid resize arguments for square matrix.
//
// This function resizes the represented matrix to the specified dimensions. Note that in case
// the matrix is a compile time square matrix (as for instance the blaze::SymmetricMatrix adaptor,
// ...) the specified number of rows must be identical to the number of columns. Otherwise a
// \a std::invalid_argument exception is thrown.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE void resize( const SparseMatrixProxy<PT,MT>& proxy, size_t m, size_t n, bool preserve )
{
   resize_backend( proxy, m, n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the represented element to the default initial values.
// \ingroup math
//
// \param proxy The given access proxy.
// \return void
//
// This function resets all elements of the matrix to the default initial values.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE void reset( const SparseMatrixProxy<PT,MT>& proxy )
{
   proxy.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset the specified row/column of the represented matrix.
// \ingroup math
//
// \param proxy The given access proxy.
// \param i The index of the row/column to be resetted.
// \return void
//
// This function resets all elements in the specified row/column of the given matrix to their
// default value. In case the given matrix is a \a rowMajor matrix the function resets the values
// in row \a i, if it is a \a columnMajor matrix the function resets the values in column \a i.
// Note that the capacity of the row/column remains unchanged.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE void reset( const SparseMatrixProxy<PT,MT>& proxy, size_t i )
{
   proxy.reset(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the represented matrix.
// \ingroup math
//
// \param proxy The given access proxy.
// \return void
//
// This function clears the matrix to its default initial state.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the sparse matrix
BLAZE_ALWAYS_INLINE void clear( const SparseMatrixProxy<PT,MT>& proxy )
{
   proxy.clear();
}
//*************************************************************************************************

} // namespace blaze

#endif
