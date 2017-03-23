//=================================================================================================
/*!
//  \file blaze/math/proxy/DenseMatrixProxy.h
//  \brief Header file for the DenseMatrixProxy class
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

#ifndef _BLAZE_MATH_PROXY_DENSEMATRIXPROXY_H_
#define _BLAZE_MATH_PROXY_DENSEMATRIXPROXY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/typetraits/IsColumnMajorMatrix.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsSquare.h>
#include <blaze/system/Inline.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/Types.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Proxy backend for dense matrix types.
// \ingroup math
//
// The DenseMatrixProxy class serves as a backend for the Proxy class. It is used in case the
// data type represented by the proxy is a dense matrix and augments the Proxy interface by
// the complete interface required of dense matrices.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
class DenseMatrixProxy : public DenseMatrix< PT, IsColumnMajorMatrix<MT>::value >
{
 public:
   //**Type definitions****************************************************************************
   typedef ResultType_<MT>      ResultType;      //!< Result type for expression template evaluations.
   typedef OppositeType_<MT>    OppositeType;    //!< Result type with opposite storage order for expression template evaluations.
   typedef TransposeType_<MT>   TransposeType;   //!< Transpose type for expression template evaluations.
   typedef ElementType_<MT>     ElementType;     //!< Type of the matrix elements.
   typedef ReturnType_<MT>      ReturnType;      //!< Return type for expression template evaluations.
   typedef CompositeType_<MT>   CompositeType;   //!< Data type for composite expression templates.
   typedef Reference_<MT>       Reference;       //!< Reference to a non-constant matrix value.
   typedef ConstReference_<MT>  ConstReference;  //!< Reference to a constant matrix value.
   typedef Pointer_<MT>         Pointer;         //!< Pointer to a non-constant matrix value.
   typedef ConstPointer_<MT>    ConstPointer;    //!< Pointer to a constant matrix value.
   typedef Iterator_<MT>        Iterator;        //!< Iterator over non-constant elements.
   typedef ConstIterator_<MT>   ConstIterator;   //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SIMD optimization.
   enum : bool { simdEnabled = MT::simdEnabled };

   //! Compilation flag for SMP assignments.
   enum : bool { smpAssignable = MT::smpAssignable };
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference operator()( size_t i, size_t j ) const;
   inline Reference at( size_t i, size_t j ) const;

   inline Pointer       data  () const;
   inline Pointer       data  ( size_t i ) const;
   inline Iterator      begin ( size_t i ) const;
   inline ConstIterator cbegin( size_t i ) const;
   inline Iterator      end   ( size_t i ) const;
   inline ConstIterator cend  ( size_t i ) const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t rows() const;
   inline size_t columns() const;
   inline size_t spacing() const;
   inline size_t capacity() const;
   inline size_t capacity( size_t i ) const;
   inline size_t nonZeros() const;
   inline size_t nonZeros( size_t i ) const;
   inline void   reset() const;
   inline void   reset( size_t i ) const;
   inline void   clear() const;
   inline void   resize( size_t m, size_t n, bool preserve=true ) const;
   inline void   extend( size_t m, size_t n, bool preserve=true ) const;
   inline void   reserve( size_t n ) const;
   inline void   transpose() const;
   inline void   ctranspose() const;

   template< typename Other > inline void scale( const Other& scalar ) const;
   //@}
   //**********************************************************************************************

 private:
   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT );
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
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline typename DenseMatrixProxy<PT,MT>::Reference
   DenseMatrixProxy<PT,MT>::operator()( size_t i, size_t j ) const
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
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline typename DenseMatrixProxy<PT,MT>::Reference
   DenseMatrixProxy<PT,MT>::at( size_t i, size_t j ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().at(i,j);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to matrix elements.
//
// \return Pointer to the internal element storage.
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function returns a pointer to the internal storage of the dense matrix. Note that you can
// NOT assume that all matrix elements lie adjacent to each other! The matrix may use techniques
// such as padding to improve the alignment of the data. Whereas the number of elements within a
// row/column are given by the \c rows() and \c columns() member functions, respectively, the
// total number of elements including padding is given by the \c spacing() member function.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline typename DenseMatrixProxy<PT,MT>::Pointer DenseMatrixProxy<PT,MT>::data() const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().data();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to matrix elements of row/column \a i.
//
// \return Pointer to the internal element storage.
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline typename DenseMatrixProxy<PT,MT>::Pointer DenseMatrixProxy<PT,MT>::data( size_t i ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().data(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i of the represented matrix.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline typename DenseMatrixProxy<PT,MT>::Iterator
   DenseMatrixProxy<PT,MT>::begin( size_t i ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().begin(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i of the represented matrix.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline typename DenseMatrixProxy<PT,MT>::ConstIterator
   DenseMatrixProxy<PT,MT>::cbegin( size_t i ) const
{
   return (~*this).get().cbegin(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i of the represented matrix.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline typename DenseMatrixProxy<PT,MT>::Iterator
   DenseMatrixProxy<PT,MT>::end( size_t i ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   return (~*this).get().end(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i of the represented matrix.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline typename DenseMatrixProxy<PT,MT>::ConstIterator
   DenseMatrixProxy<PT,MT>::cend( size_t i ) const
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
        , typename MT >  // Type of the dense matrix
inline size_t DenseMatrixProxy<PT,MT>::rows() const
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
        , typename MT >  // Type of the dense matrix
inline size_t DenseMatrixProxy<PT,MT>::columns() const
{
   return (~*this).get().columns();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the spacing between the beginning of two rows/columns of the represented matrix.
//
// \return The spacing between the beginning of two rows/columns.
//
// This function returns the spacing between the beginning of two rows/columns, i.e. the
// total number of elements of a row/column. In case the storage order is set to \a rowMajor
// the function returns the spacing between two rows, in case the storage flag is set to
// \a columnMajor the function returns the spacing between two columns.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline size_t DenseMatrixProxy<PT,MT>::spacing() const
{
   return (~*this).get().spacing();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the represented matrix.
//
// \return The capacity of the matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline size_t DenseMatrixProxy<PT,MT>::capacity() const
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
        , typename MT >  // Type of the dense matrix
inline size_t DenseMatrixProxy<PT,MT>::capacity( size_t i ) const
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
        , typename MT >  // Type of the dense matrix
inline size_t DenseMatrixProxy<PT,MT>::nonZeros() const
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
        , typename MT >  // Type of the dense matrix
inline size_t DenseMatrixProxy<PT,MT>::nonZeros( size_t i ) const
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
        , typename MT >  // Type of the dense matrix
inline void DenseMatrixProxy<PT,MT>::reset() const
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
        , typename MT >  // Type of the dense matrix
inline void DenseMatrixProxy<PT,MT>::reset( size_t i ) const
{
   using blaze::reset;

   reset( (~*this).get(), i );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the represented matrix.
//
// \return void
//
// This function clears the matrix to its default initial state.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline void DenseMatrixProxy<PT,MT>::clear() const
{
   using blaze::clear;

   clear( (~*this).get() );
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
// preserve the old matrix values, the \a preserve flag can be set to \a true. However, note
// that depending on the type of the matrix new matrix elements may not initialized!
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline void DenseMatrixProxy<PT,MT>::resize( size_t m, size_t n, bool preserve ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().resize( m, n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Extending the size of the represented matrix.
//
// \param m Number of additional rows.
// \param n Number of additional columns.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function increases the matrix size by \a m rows and \a n columns. Depending on the type
// of the matrix, during this operation new dynamic memory may be allocated in case the capacity
// of the matrix is too small. Therefore this function potentially changes all matrix elements.
// In order to preserve the old matrix values, the \a preserve flag can be set to \a true.
// However, note that depending on the type of the matrix new matrix elements may not
// initialized!
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline void DenseMatrixProxy<PT,MT>::extend( size_t m, size_t n, bool preserve ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().extend( m, n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the minimum capacity of the represented matrix.
//
// \param n The new minimum capacity of the matrix.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
//
// This function increases the capacity of the dense matrix to at least \a n elements. The
// current values of the matrix elements are preserved.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline void DenseMatrixProxy<PT,MT>::reserve( size_t n ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().reserve( n );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the represented matrix.
//
// \return Reference to the transposed matrix.
// \exception std::invalid_argument Invalid access to restricted element.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
inline void DenseMatrixProxy<PT,MT>::transpose() const
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
        , typename MT >  // Type of the dense matrix
inline void DenseMatrixProxy<PT,MT>::ctranspose() const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().ctranspose();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the matrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the matrix scaling.
// \return void
// \exception std::invalid_argument Invalid access to restricted element.
*/
template< typename PT       // Type of the proxy
        , typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the scalar value
inline void DenseMatrixProxy<PT,MT>::scale( const Other& scalar ) const
{
   if( (~*this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (~*this).get().scale( scalar );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DenseMatrixProxy global functions */
//@{
template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE typename DenseMatrixProxy<PT,MT>::Iterator
   begin( const DenseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE typename DenseMatrixProxy<PT,MT>::ConstIterator
   cbegin( const DenseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE typename DenseMatrixProxy<PT,MT>::Iterator
   end( const DenseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE typename DenseMatrixProxy<PT,MT>::ConstIterator
   cend( const DenseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE size_t rows( const DenseMatrixProxy<PT,MT>& proxy );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE size_t columns( const DenseMatrixProxy<PT,MT>& proxy );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE size_t capacity( const DenseMatrixProxy<PT,MT>& proxy );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE size_t capacity( const DenseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE size_t nonZeros( const DenseMatrixProxy<PT,MT>& proxy );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE size_t nonZeros( const DenseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE void resize( const DenseMatrixProxy<PT,MT>& proxy, size_t m, size_t n, bool preserve=true );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE void reset( const DenseMatrixProxy<PT,MT>& proxy );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE void reset( const DenseMatrixProxy<PT,MT>& proxy, size_t i );

template< typename PT, typename MT >
BLAZE_ALWAYS_INLINE void clear( const DenseMatrixProxy<PT,MT>& proxy );
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
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename DenseMatrixProxy<PT,MT>::Iterator
   begin( const DenseMatrixProxy<PT,MT>& proxy, size_t i )
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
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename DenseMatrixProxy<PT,MT>::ConstIterator
   cbegin( const DenseMatrixProxy<PT,MT>& proxy, size_t i )
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
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename DenseMatrixProxy<PT,MT>::Iterator
   end( const DenseMatrixProxy<PT,MT>& proxy, size_t i )
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
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename DenseMatrixProxy<PT,MT>::ConstIterator
   cend( const DenseMatrixProxy<PT,MT>& proxy, size_t i )
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
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE size_t rows( const DenseMatrixProxy<PT,MT>& proxy )
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
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE size_t columns( const DenseMatrixProxy<PT,MT>& proxy )
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
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE size_t capacity( const DenseMatrixProxy<PT,MT>& proxy )
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
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE size_t capacity( const DenseMatrixProxy<PT,MT>& proxy, size_t i )
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
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE size_t nonZeros( const DenseMatrixProxy<PT,MT>& proxy )
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
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE size_t nonZeros( const DenseMatrixProxy<PT,MT>& proxy, size_t i )
{
   return proxy.nonZeros(i);
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for non-resizable matrices.
// \ingroup math
//
// \param proxy The given access proxy.
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
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE DisableIf_< IsResizable<MT> >
   resize_backend( const DenseMatrixProxy<PT,MT>& proxy, size_t m, size_t n, bool preserve )
{
   UNUSED_PARAMETER( preserve );

   if( proxy.rows() != m || proxy.columns() != n ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix cannot be resized" );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for resizable, non-square matrices.
// \ingroup math
//
// \param proxy The given access proxy.
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function changes the number of rows and columns of the given resizable, non-square matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE EnableIf_< And< IsResizable<MT>, Not< IsSquare<MT> > > >
   resize_backend( const DenseMatrixProxy<PT,MT>& proxy, size_t m, size_t n, bool preserve )
{
   proxy.resize( m, n, preserve );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the \c resize() function for resizable, square matrices.
// \ingroup math
//
// \param proxy The given access proxy.
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
// \exception std::invalid_argument Invalid resize arguments for square matrix.
//
// This function changes the number of rows and columns of the given resizable, square matrix.
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE EnableIf_< And< IsResizable<MT>, IsSquare<MT> > >
   resize_backend( const DenseMatrixProxy<PT,MT>& proxy, size_t m, size_t n, bool preserve )
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
// \exception std::invalid_argument Matrix cannot be resized.
//
// This function resizes the represented matrix to the specified dimensions. In contrast to
// the \c resize() member function, which is only available on resizable matrix types, this
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
*/
template< typename PT    // Type of the proxy
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void resize( const DenseMatrixProxy<PT,MT>& proxy, size_t m, size_t n, bool preserve )
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
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void reset( const DenseMatrixProxy<PT,MT>& proxy )
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
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void reset( const DenseMatrixProxy<PT,MT>& proxy, size_t i )
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
        , typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void clear( const DenseMatrixProxy<PT,MT>& proxy )
{
   proxy.clear();
}
//*************************************************************************************************

} // namespace blaze

#endif
