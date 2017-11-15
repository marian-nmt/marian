//=================================================================================================
/*!
//  \file blaze/math/adaptors/symmetricmatrix/DenseNonNumeric.h
//  \brief SymmetricMatrix specialization for dense matrices with non-numeric element type
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

#ifndef _BLAZE_MATH_ADAPTORS_SYMMETRICMATRIX_DENSENONNUMERIC_H_
#define _BLAZE_MATH_ADAPTORS_SYMMETRICMATRIX_DENSENONNUMERIC_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <utility>
#include <blaze/math/adaptors/symmetricmatrix/BaseTemplate.h>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/Expression.h>
#include <blaze/math/constraints/Hermitian.h>
#include <blaze/math/constraints/Lower.h>
#include <blaze/math/constraints/Resizable.h>
#include <blaze/math/constraints/StorageOrder.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/Upper.h>
#include <blaze/math/dense/DenseMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/Conjugate.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/typetraits/Columns.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsCustom.h>
#include <blaze/math/typetraits/IsSparseMatrix.h>
#include <blaze/math/typetraits/IsSquare.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/RemoveAdaptor.h>
#include <blaze/math/typetraits/Rows.h>
#include <blaze/math/views/Submatrix.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Numeric.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/RemoveReference.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR DENSE MATRICES WITH NON-NUMERIC ELEMENT TYPE
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of SymmetricMatrix for dense matrices with non-numeric element type.
// \ingroup symmetric_matrix
//
// This specialization of SymmetricMatrix adapts the class template to the requirements of dense
// matrices with non-numeric data type.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
class SymmetricMatrix<MT,SO,true,false>
   : public DenseMatrix< SymmetricMatrix<MT,SO,true,false>, SO >
{
 private:
   //**Type definitions****************************************************************************
   typedef OppositeType_<MT>   OT;  //!< Opposite type of the dense matrix.
   typedef TransposeType_<MT>  TT;  //!< Transpose type of the dense matrix.
   typedef ElementType_<MT>    ET;  //!< Element type of the dense matrix.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef SymmetricMatrix<MT,SO,true,false>   This;            //!< Type of this SymmetricMatrix instance.
   typedef DenseMatrix<This,SO>                BaseType;        //!< Base type of this SymmetricMatrix instance.
   typedef This                                ResultType;      //!< Result type for expression template evaluations.
   typedef SymmetricMatrix<OT,!SO,true,false>  OppositeType;    //!< Result type with opposite storage order for expression template evaluations.
   typedef SymmetricMatrix<TT,!SO,true,false>  TransposeType;   //!< Transpose type for expression template evaluations.
   typedef ET                                  ElementType;     //!< Type of the matrix elements.
   typedef ReturnType_<MT>                     ReturnType;      //!< Return type for expression template evaluations.
   typedef const This&                         CompositeType;   //!< Data type for composite expression templates.
   typedef Reference_<MT>                      Reference;       //!< Reference to a non-constant matrix value.
   typedef ConstReference_<MT>                 ConstReference;  //!< Reference to a constant matrix value.
   typedef Pointer_<MT>                        Pointer;         //!< Pointer to a non-constant matrix value.
   typedef ConstPointer_<MT>                   ConstPointer;    //!< Pointer to a constant matrix value.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a SymmetricMatrix with different data/element type.
   */
   template< typename ET >  // Data type of the other matrix
   struct Rebind {
      //! The type of the other SymmetricMatrix.
      typedef SymmetricMatrix< typename MT::template Rebind<ET>::Other >  Other;
   };
   //**********************************************************************************************

   //**MatrixIterator class definition*************************************************************
   /*!\brief Iterator over the elements of the dense symmetric matrix.
   */
   template< typename MatrixType >  // Type of the adapted dense matrix
   class MatrixIterator
   {
    public:
      //**Type definitions*************************************************************************
      //! Return type for the access to the value of a dense matrix element.
      typedef If_< IsConst<MatrixType>
                 , ConstReference_<MatrixType>
                 , Reference_<MatrixType> >  Reference;

      typedef std::random_access_iterator_tag  IteratorCategory;  //!< The iterator category.
      typedef RemoveReference_<Reference>      ValueType;         //!< Type of the underlying elements.
      typedef ValueType*                       PointerType;       //!< Pointer return type.
      typedef Reference                        ReferenceType;     //!< Reference return type.
      typedef ptrdiff_t                        DifferenceType;    //!< Difference between two iterators.

      // STL iterator requirements
      typedef IteratorCategory  iterator_category;  //!< The iterator category.
      typedef ValueType         value_type;         //!< Type of the underlying elements.
      typedef PointerType       pointer;            //!< Pointer return type.
      typedef ReferenceType     reference;          //!< Reference return type.
      typedef DifferenceType    difference_type;    //!< Difference between two iterators.
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Default constructor of the MatrixIterator class.
      */
      inline MatrixIterator() noexcept
         : matrix_( nullptr )  // Reference to the adapted dense matrix
         , row_   ( 0UL )      // The current row index of the iterator
         , column_( 0UL )      // The current column index of the iterator
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the MatrixIterator class.
      //
      // \param matrix The adapted matrix.
      // \param row Initial row index of the iterator.
      // \param column Initial column index of the iterator.
      */
      inline MatrixIterator( MatrixType& matrix, size_t row, size_t column ) noexcept
         : matrix_( &matrix )  // Reference to the adapted dense matrix
         , row_   ( row     )  // The current row index of the iterator
         , column_( column  )  // The current column index of the iterator
      {}
      //*******************************************************************************************

      //**Conversion constructor*******************************************************************
      /*!\brief Conversion constructor from different MatrixIterator instances.
      //
      // \param it The row iterator to be copied.
      */
      template< typename MatrixType2 >
      inline MatrixIterator( const MatrixIterator<MatrixType2>& it ) noexcept
         : matrix_( it.matrix_ )  // Reference to the adapted dense matrix
         , row_   ( it.row_    )  // The current row index of the iterator
         , column_( it.column_ )  // The current column index of the iterator
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline MatrixIterator& operator+=( size_t inc ) noexcept {
         ( SO )?( row_ += inc ):( column_ += inc );
         return *this;
      }
      //*******************************************************************************************

      //**Subtraction assignment operator**********************************************************
      /*!\brief Subtraction assignment operator.
      //
      // \param dec The decrement of the iterator.
      // \return The decremented iterator.
      */
      inline MatrixIterator& operator-=( size_t dec ) noexcept {
         ( SO )?( row_ -= dec ):( column_ -= dec );
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline MatrixIterator& operator++() noexcept {
         ( SO )?( ++row_ ):( ++column_ );
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const MatrixIterator operator++( int ) noexcept {
         const MatrixIterator tmp( *this );
         ++(*this);
         return tmp;
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline MatrixIterator& operator--() noexcept {
         ( SO )?( --row_ ):( --column_ );
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const MatrixIterator operator--( int ) noexcept {
         const MatrixIterator tmp( *this );
         --(*this);
         return tmp;
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReferenceType operator*() const {
         if( ( SO && row_ < column_ ) || ( !SO && row_ > column_ ) )
            return (*matrix_)(row_,column_);
         else
            return (*matrix_)(column_,row_);
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline PointerType operator->() const {
         if( ( SO && row_ < column_ ) || ( !SO && row_ > column_ ) )
            return &(*matrix_)(row_,column_);
         else
            return &(*matrix_)(column_,row_);
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two MatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      template< typename MatrixType2 >
      inline bool operator==( const MatrixIterator<MatrixType2>& rhs ) const noexcept {
         return ( SO )?( row_ == rhs.row_ ):( column_ == rhs.column_ );
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two MatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      template< typename MatrixType2 >
      inline bool operator!=( const MatrixIterator<MatrixType2>& rhs ) const noexcept {
         return ( SO )?( row_ != rhs.row_ ):( column_ != rhs.column_ );
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two MatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      template< typename MatrixType2 >
      inline bool operator<( const MatrixIterator<MatrixType2>& rhs ) const noexcept {
         return ( SO )?( row_ < rhs.row_ ):( column_ < rhs.column_ );
         return ( column_ < rhs.column_ );
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two MatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      template< typename MatrixType2 >
      inline bool operator>( const MatrixIterator<MatrixType2>& rhs ) const noexcept {
         return ( SO )?( row_ > rhs.row_ ):( column_ > rhs.column_ );
         return ( column_ > rhs.column_ );
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two MatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      template< typename MatrixType2 >
      inline bool operator<=( const MatrixIterator<MatrixType2>& rhs ) const noexcept {
         return ( SO )?( row_ <= rhs.row_ ):( column_ <= rhs.column_ );
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two MatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      template< typename MatrixType2 >
      inline bool operator>=( const MatrixIterator<MatrixType2>& rhs ) const noexcept {
         return ( SO )?( row_ >= rhs.row_ ):( column_ >= rhs.column_ );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const MatrixIterator& rhs ) const noexcept {
         return ( SO )?( row_ - rhs.row_ ):( column_ - rhs.column_ );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between a MatrixIterator and an integral value.
      //
      // \param it The iterator to be incremented.
      // \param inc The number of elements the iterator is incremented.
      // \return The incremented iterator.
      */
      friend inline const MatrixIterator operator+( const MatrixIterator& it, size_t inc ) noexcept {
         if( SO )
            return MatrixIterator( *it.matrix_, it.row_ + inc, it.column_ );
         else
            return MatrixIterator( *it.matrix_, it.row_, it.column_ + inc );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between an integral value and a MatrixIterator.
      //
      // \param inc The number of elements the iterator is incremented.
      // \param it The iterator to be incremented.
      // \return The incremented iterator.
      */
      friend inline const MatrixIterator operator+( size_t inc, const MatrixIterator& it ) noexcept {
         if( SO )
            return MatrixIterator( *it.matrix_, it.row_ + inc, it.column_ );
         else
            return MatrixIterator( *it.matrix_, it.row_, it.column_ + inc );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Subtraction between a MatrixIterator and an integral value.
      //
      // \param it The iterator to be decremented.
      // \param dec The number of elements the iterator is decremented.
      // \return The decremented iterator.
      */
      friend inline const MatrixIterator operator-( const MatrixIterator& it, size_t dec ) noexcept {
         if( SO )
            return MatrixIterator( *it.matrix_, it.row_ - dec, it.column_ );
         else
            return MatrixIterator( *it.matrix_, it.row_, it.column_ - dec );
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      MatrixType* matrix_;  //!< Reference to the adapted dense matrix.
      size_t      row_;     //!< The current row index of the iterator.
      size_t      column_;  //!< The current column index of the iterator.
      //*******************************************************************************************

      //**Friend declarations**********************************************************************
      template< typename MatrixType2 > friend class MatrixIterator;
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   typedef MatrixIterator<MT>        Iterator;       //!< Iterator over non-constant elements.
   typedef MatrixIterator<const MT>  ConstIterator;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = false };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = MT::smpAssignable && !IsSMPAssignable<ET>::value };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline SymmetricMatrix();
   explicit inline SymmetricMatrix( size_t n );

   explicit inline SymmetricMatrix( ElementType* ptr, size_t n );
   explicit inline SymmetricMatrix( ElementType* ptr, size_t n, size_t nn );

   template< typename Deleter >
   explicit inline SymmetricMatrix( ElementType* ptr, size_t n, Deleter d );

   template< typename Deleter >
   explicit inline SymmetricMatrix( ElementType* ptr, size_t n, size_t nn, Deleter d );

   inline SymmetricMatrix( const SymmetricMatrix& m );
   inline SymmetricMatrix( SymmetricMatrix&& m ) noexcept;

   template< typename MT2 > inline SymmetricMatrix( const Matrix<MT2,SO>&  m );
   template< typename MT2 > inline SymmetricMatrix( const Matrix<MT2,!SO>& m );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator()( size_t i, size_t j );
   inline ConstReference operator()( size_t i, size_t j ) const;
   inline Reference      at( size_t i, size_t j );
   inline ConstReference at( size_t i, size_t j ) const;
   inline ConstPointer   data  () const noexcept;
   inline ConstPointer   data  ( size_t i ) const noexcept;
   inline Iterator       begin ( size_t i );
   inline ConstIterator  begin ( size_t i ) const;
   inline ConstIterator  cbegin( size_t i ) const;
   inline Iterator       end   ( size_t i );
   inline ConstIterator  end   ( size_t i ) const;
   inline ConstIterator  cend  ( size_t i ) const;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline SymmetricMatrix& operator=( const SymmetricMatrix& rhs );
   inline SymmetricMatrix& operator=( SymmetricMatrix&& rhs ) noexcept;

   template< typename MT2 >
   inline DisableIf_< IsComputation<MT2>, SymmetricMatrix& > operator=( const Matrix<MT2,SO>& rhs );

   template< typename MT2 >
   inline EnableIf_< IsComputation<MT2>, SymmetricMatrix& > operator=( const Matrix<MT2,SO>& rhs );

   template< typename MT2 >
   inline SymmetricMatrix& operator=( const Matrix<MT2,!SO>& rhs );

   template< typename MT2 >
   inline DisableIf_< IsComputation<MT2>, SymmetricMatrix& > operator+=( const Matrix<MT2,SO>& rhs );

   template< typename MT2 >
   inline EnableIf_< IsComputation<MT2>, SymmetricMatrix& > operator+=( const Matrix<MT2,SO>& rhs );

   template< typename MT2 >
   inline SymmetricMatrix& operator+=( const Matrix<MT2,!SO>& rhs );

   template< typename MT2 >
   inline DisableIf_< IsComputation<MT2>, SymmetricMatrix& > operator-=( const Matrix<MT2,SO>& rhs );

   template< typename MT2 >
   inline EnableIf_< IsComputation<MT2>, SymmetricMatrix& > operator-=( const Matrix<MT2,SO>& rhs );

   template< typename MT2 >
   inline SymmetricMatrix& operator-=( const Matrix<MT2,!SO>& rhs );

   template< typename MT2, bool SO2 >
   inline SymmetricMatrix& operator*=( const Matrix<MT2,SO2>& rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, SymmetricMatrix >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, SymmetricMatrix >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t           rows() const noexcept;
                              inline size_t           columns() const noexcept;
                              inline size_t           spacing() const noexcept;
                              inline size_t           capacity() const noexcept;
                              inline size_t           capacity( size_t i ) const noexcept;
                              inline size_t           nonZeros() const;
                              inline size_t           nonZeros( size_t i ) const;
                              inline void             reset();
                              inline void             reset( size_t i );
                              inline void             clear();
                                     void             resize ( size_t n, bool preserve=true );
                              inline void             extend ( size_t n, bool preserve=true );
                              inline void             reserve( size_t elements );
                              inline SymmetricMatrix& transpose();
                              inline SymmetricMatrix& ctranspose();
   template< typename Other > inline SymmetricMatrix& scale( const Other& scalar );
                              inline void             swap( SymmetricMatrix& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Debugging functions*************************************************************************
   /*!\name Utility functions */
   //@{
   inline bool isIntact() const noexcept;
   //@}
   //**********************************************************************************************

   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other > inline bool canAlias ( const Other* alias ) const noexcept;
   template< typename Other > inline bool isAliased( const Other* alias ) const noexcept;

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename MT2 > inline void assign   (       DenseMatrix <MT2,SO>& rhs );
   template< typename MT2 > inline void assign   ( const DenseMatrix <MT2,SO>& rhs );
   template< typename MT2 > inline void assign   ( const SparseMatrix<MT2,SO>& rhs );
   template< typename MT2 > inline void addAssign( const DenseMatrix <MT2,SO>& rhs );
   template< typename MT2 > inline void addAssign( const SparseMatrix<MT2,SO>& rhs );
   template< typename MT2 > inline void subAssign( const DenseMatrix <MT2,SO>& rhs );
   template< typename MT2 > inline void subAssign( const SparseMatrix<MT2,SO>& rhs );
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   MT matrix_;  //!< The adapted dense matrix.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename MT2, bool SO2, bool DF2, bool NF2 >
   friend bool isDefault( const SymmetricMatrix<MT2,SO2,DF2,NF2>& m );
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE        ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE       ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE         ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST                ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE             ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_EXPRESSION_TYPE      ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_HERMITIAN_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_LOWER_MATRIX_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_UPPER_MATRIX_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( OT, !SO );
   BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( TT, !SO );
   BLAZE_CONSTRAINT_MUST_NOT_BE_NUMERIC_TYPE         ( ElementType );
   BLAZE_STATIC_ASSERT( Rows<MT>::value == Columns<MT>::value );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The default constructor for SymmetricMatrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline SymmetricMatrix<MT,SO,true,false>::SymmetricMatrix()
   : matrix_()  // The adapted dense matrix
{
   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a matrix of size \f$ n \times n \f$.
//
// \param n The number of rows and columns of the matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline SymmetricMatrix<MT,SO,true,false>::SymmetricMatrix( size_t n )
   : matrix_( n, n )  // The adapted dense matrix
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE( MT );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a symmetric custom matrix of size \f$ n \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param n The number of rows and columns of the array of elements.
// \exception std::invalid_argument Invalid setup of symmetric custom matrix.
//
// This constructor creates an unpadded symmetric custom matrix of size \f$ n \times n \f$. The
// construction fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the values in the given array do not represent a symmetric matrix.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This constructor is \b NOT available for padded symmetric custom matrices!
// \note The matrix does \b NOT take responsibility for the given array of elements!
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline SymmetricMatrix<MT,SO,true,false>::SymmetricMatrix( ElementType* ptr, size_t n )
   : matrix_( ptr, n, n )  // The adapted dense matrix
{
   if( !isSymmetric( matrix_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of symmetric matrix" );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a symmetric custom matrix of size \f$ n \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param n The number of rows and columns of the array of elements.
// \param nn The total number of elements between two rows/columns.
// \exception std::invalid_argument Invalid setup of symmetric custom matrix.
//
// This constructor creates a symmetric custom matrix of size \f$ n \times n \f$. The construction
// fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified spacing \a nn is insufficient for the given data type \a Type and the
//    available instruction set;
//  - ... the values in the given array do not represent a symmetric matrix.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note The matrix does \b NOT take responsibility for the given array of elements!
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline SymmetricMatrix<MT,SO,true,false>::SymmetricMatrix( ElementType* ptr, size_t n, size_t nn )
   : matrix_( ptr, n, n, nn )  // The adapted dense matrix
{
   if( !isSymmetric( matrix_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of symmetric matrix" );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a symmetric custom matrix of size \f$ n \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param n The number of rows and columns of the array of elements.
// \param d The deleter to destroy the array of elements.
// \exception std::invalid_argument Invalid setup of symmetric custom matrix.
//
// This constructor creates an unpadded symmetric custom matrix of size \f$ n \times n \f$. The
// construction fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the values in the given array do not represent a symmetric matrix.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This constructor is \b NOT available for padded symmetric custom matrices!
*/
template< typename MT         // Type of the adapted dense matrix
        , bool SO >           // Storage order of the adapted dense matrix
template< typename Deleter >  // Type of the custom deleter
inline SymmetricMatrix<MT,SO,true,false>::SymmetricMatrix( ElementType* ptr, size_t n, Deleter d )
   : matrix_( ptr, n, n, d )  // The adapted dense matrix
{
   if( !isSymmetric( matrix_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of symmetric matrix" );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a symmetric custom matrix of size \f$ n \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param n The number of rows and columns of the array of elements.
// \param nn The total number of elements between two rows/columns.
// \param d The deleter to destroy the array of elements.
// \exception std::invalid_argument Invalid setup of symmetric custom matrix.
//
// This constructor creates a symmetric custom matrix of size \f$ n \times n \f$. The construction
// fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified spacing \a nn is insufficient for the given data type \a Type and the
//    available instruction set;
//  - ... the values in the given array do not represent a symmetric matrix.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
*/
template< typename MT         // Type of the adapted dense matrix
        , bool SO >           // Storage order of the adapted dense matrix
template< typename Deleter >  // Type of the custom deleter
inline SymmetricMatrix<MT,SO,true,false>::SymmetricMatrix( ElementType* ptr, size_t n, size_t nn, Deleter d )
   : matrix_( ptr, n, n, nn, d )  // The adapted dense matrix
{
   if( !isSymmetric( matrix_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of symmetric matrix" );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The copy constructor for SymmetricMatrix.
//
// \param m The symmetric matrix to be copied.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline SymmetricMatrix<MT,SO,true,false>::SymmetricMatrix( const SymmetricMatrix& m )
   : matrix_( m.matrix_ )  // The adapted dense matrix
{
   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The move constructor for SymmetricMatrix.
//
// \param m The symmetric matrix to be moved into this instance.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline SymmetricMatrix<MT,SO,true,false>::SymmetricMatrix( SymmetricMatrix&& m ) noexcept
   : matrix_( std::move( m.matrix_ ) )  // The adapted dense matrix
{
   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Conversion constructor from different matrices with the same storage order.
//
// \param m Matrix to be copied.
// \exception std::invalid_argument Invalid setup of symmetric matrix.
//
// This constructor initializes the symmetric matrix as a copy of the given matrix. In case the
// given matrix is not a symmetric matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the foreign matrix
inline SymmetricMatrix<MT,SO,true,false>::SymmetricMatrix( const Matrix<MT2,SO>& m )
   : matrix_()  // The adapted dense matrix
{
   using blaze::resize;

   typedef RemoveAdaptor_<ResultType_<MT2> >          RT;
   typedef If_< IsComputation<MT2>, RT, const MT2& >  Tmp;

   if( IsSymmetric<MT2>::value ) {
      resize( matrix_, (~m).rows(), (~m).columns() );
      assign( ~m );
   }
   else {
      Tmp tmp( ~m );

      if( !isSymmetric( tmp ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of symmetric matrix" );
      }

      resize( matrix_, tmp.rows(), tmp.rows() );
      assign( tmp );
   }

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Conversion constructor from different matrices with opposite storage order.
//
// \param m Matrix to be copied.
// \exception std::invalid_argument Invalid setup of symmetric matrix.
//
// This constructor initializes the symmetric matrix as a copy of the given matrix. In case the
// given matrix is not a symmetric matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the foreign matrix
inline SymmetricMatrix<MT,SO,true,false>::SymmetricMatrix( const Matrix<MT2,!SO>& m )
   : matrix_()  // The adapted dense matrix
{
   using blaze::resize;

   typedef RemoveAdaptor_< ResultType_<MT2> >         RT;
   typedef If_< IsComputation<MT2>, RT, const MT2& >  Tmp;

   if( IsSymmetric<MT2>::value ) {
      resize( matrix_, (~m).rows(), (~m).columns() );
      assign( trans( ~m ) );
   }
   else {
      Tmp tmp( ~m );

      if( !isSymmetric( tmp ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of symmetric matrix" );
      }

      resize( matrix_, tmp.rows(), tmp.rows() );
      assign( trans( tmp ) );
   }

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..N-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// The function call operator provides access to both the elements at position (i,j) and (j,i).
// In order to preserve the symmetry of the matrix, any modification to one of the elements will
// also be applied to the other element.
//
// Note that this function only performs an index check in case BLAZE_USER_ASSERT() is active. In
// contrast, the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename SymmetricMatrix<MT,SO,true,false>::Reference
   SymmetricMatrix<MT,SO,true,false>::operator()( size_t i, size_t j )
{
   BLAZE_USER_ASSERT( i<rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<columns(), "Invalid column access index" );

   if( ( !SO && i > j ) || ( SO && i < j ) )
      return matrix_(i,j);
   else
      return matrix_(j,i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..N-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// The function call operator provides access to both the elements at position (i,j) and (j,i).
// In order to preserve the symmetry of the matrix, any modification to one of the elements will
// also be applied to the other element.
//
// Note that this function only performs an index check in case BLAZE_USER_ASSERT() is active. In
// contrast, the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename SymmetricMatrix<MT,SO,true,false>::ConstReference
   SymmetricMatrix<MT,SO,true,false>::operator()( size_t i, size_t j ) const
{
   BLAZE_USER_ASSERT( i<rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<columns(), "Invalid column access index" );

   if( ( !SO && i > j ) || ( SO && i < j ) )
      return matrix_(i,j);
   else
      return matrix_(j,i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..N-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// The function call operator provides access to both the elements at position (i,j) and (j,i).
// In order to preserve the symmetry of the matrix, any modification to one of the elements will
// also be applied to the other element.
//
// Note that in contrast to the subscript operator this function always performs a check of the
// given access indices.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename SymmetricMatrix<MT,SO,true,false>::Reference
   SymmetricMatrix<MT,SO,true,false>::at( size_t i, size_t j )
{
   if( i >= rows() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= columns() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i,j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..N-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// The function call operator provides access to both the elements at position (i,j) and (j,i).
// In order to preserve the symmetry of the matrix, any modification to one of the elements will
// also be applied to the other element.
//
// Note that in contrast to the subscript operator this function always performs a check of the
// given access indices.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename SymmetricMatrix<MT,SO,true,false>::ConstReference
   SymmetricMatrix<MT,SO,true,false>::at( size_t i, size_t j ) const
{
   if( i >= rows() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= columns() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i,j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the matrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the symmetric matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The symmetric matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function. Also note that you can NOT assume that the symmetric matrix stores all its
// elements. It may choose to store its elements in a lower or upper triangular matrix fashion.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename SymmetricMatrix<MT,SO,true,false>::ConstPointer
   SymmetricMatrix<MT,SO,true,false>::data() const noexcept
{
   return matrix_.data();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the matrix elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
// Note that you can NOT assume that the symmetric matrix stores all its elements. It may choose
// to store its elements in a lower or upper triangular matrix fashion.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename SymmetricMatrix<MT,SO,true,false>::ConstPointer
   SymmetricMatrix<MT,SO,true,false>::data( size_t i ) const noexcept
{
   return matrix_.data(i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the symmetric matrix adapts a \a rowMajor dense matrix the function returns an iterator to
// the first element of row \a i, in case it adapts a \a columnMajor dense matrix the function
// returns an iterator to the first element of column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename SymmetricMatrix<MT,SO,true,false>::Iterator
   SymmetricMatrix<MT,SO,true,false>::begin( size_t i )
{
   if( SO )
      return Iterator( matrix_, 0UL, i );
   else
      return Iterator( matrix_, i, 0UL );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the symmetric matrix adapts a \a rowMajor dense matrix the function returns an iterator to
// the first element of row \a i, in case it adapts a \a columnMajor dense matrix the function
// returns an iterator to the first element of column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename SymmetricMatrix<MT,SO,true,false>::ConstIterator
   SymmetricMatrix<MT,SO,true,false>::begin( size_t i ) const
{
   if( SO )
      return ConstIterator( matrix_, 0UL, i );
   else
      return ConstIterator( matrix_, i, 0UL );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the symmetric matrix adapts a \a rowMajor dense matrix the function returns an iterator to
// the first element of row \a i, in case it adapts a \a columnMajor dense matrix the function
// returns an iterator to the first element of column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename SymmetricMatrix<MT,SO,true,false>::ConstIterator
   SymmetricMatrix<MT,SO,true,false>::cbegin( size_t i ) const
{
   if( SO )
      return ConstIterator( matrix_, 0UL, i );
   else
      return ConstIterator( matrix_, i, 0UL );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the symmetric matrix adapts a \a rowMajor dense matrix the function returns an iterator
// just past the last element of row \a i, in case it adapts a \a columnMajor dense matrix the
// function returns an iterator just past the last element of column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename SymmetricMatrix<MT,SO,true,false>::Iterator
   SymmetricMatrix<MT,SO,true,false>::end( size_t i )
{
   if( SO )
      return Iterator( matrix_, rows(), i );
   else
      return Iterator( matrix_, i, columns() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the symmetric matrix adapts a \a rowMajor dense matrix the function returns an iterator
// just past the last element of row \a i, in case it adapts a \a columnMajor dense matrix the
// function returns an iterator just past the last element of column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename SymmetricMatrix<MT,SO,true,false>::ConstIterator
   SymmetricMatrix<MT,SO,true,false>::end( size_t i ) const
{
   if( SO )
      return ConstIterator( matrix_, rows(), i );
   else
      return ConstIterator( matrix_, i, columns() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the symmetric matrix adapts a \a rowMajor dense matrix the function returns an iterator
// just past the last element of row \a i, in case it adapts a \a columnMajor dense matrix the
// function returns an iterator just past the last element of column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename SymmetricMatrix<MT,SO,true,false>::ConstIterator
   SymmetricMatrix<MT,SO,true,false>::cend( size_t i ) const
{
   if( SO )
      return ConstIterator( matrix_, rows(), i );
   else
      return ConstIterator( matrix_, i, columns() );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for SymmetricMatrix.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
//
// If possible and necessary, the matrix is resized according to the given \f$ N \times N \f$
// matrix and initialized as a copy of this matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline SymmetricMatrix<MT,SO,true,false>&
   SymmetricMatrix<MT,SO,true,false>::operator=( const SymmetricMatrix& rhs )
{
   using blaze::resize;

   if( &rhs == this ) return *this;

   resize( matrix_, rhs.rows(), rhs.columns() );
   assign( rhs );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Move assignment operator for SymmetricMatrix.
//
// \param rhs The matrix to be moved into this instance.
// \return Reference to the assigned matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline SymmetricMatrix<MT,SO,true,false>&
   SymmetricMatrix<MT,SO,true,false>::operator=( SymmetricMatrix&& rhs ) noexcept
{
   matrix_ = std::move( rhs.matrix_ );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for general matrices.
//
// \param rhs The general matrix to be copied.
// \return Reference to the assigned matrix.
// \exception std::invalid_argument Invalid assignment to symmetric matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// If possible and necessary, the matrix is resized according to the given \f$ N \times N \f$
// matrix and initialized as a copy of this matrix. If the matrix cannot be resized accordingly,
// a \a std::invalid_argument exception is thrown. Also note that the given matrix must be a
// symmetric matrix. Otherwise, a \a std::invalid_argument exception is thrown.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side matrix
inline DisableIf_< IsComputation<MT2>, SymmetricMatrix<MT,SO,true,false>& >
   SymmetricMatrix<MT,SO,true,false>::operator=( const Matrix<MT2,SO>& rhs )
{
   using blaze::resize;

   if( !IsSymmetric<MT2>::value && !isSymmetric( ~rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to symmetric matrix" );
   }

   if( (~rhs).isAliased( this ) ) {
      SymmetricMatrix tmp( ~rhs );
      swap( tmp );
   }
   else {
      resize( matrix_, (~rhs).rows(), (~rhs).columns() );
      if( IsSparseMatrix<MT2>::value )
         reset();
      assign( ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for matrix computations.
//
// \param rhs The matrix computation to be copied.
// \return Reference to the assigned matrix.
// \exception std::invalid_argument Invalid assignment to symmetric matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// If possible and necessary, the matrix is resized according to the given \f$ N \times N \f$
// matrix and initialized as a copy of this matrix. If the matrix cannot be resized accordingly,
// a \a std::invalid_argument exception is thrown. Also note that the given matrix must be a
// symmetric matrix. Otherwise, a \a std::invalid_argument exception is thrown.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side matrix
inline EnableIf_< IsComputation<MT2>, SymmetricMatrix<MT,SO,true,false>& >
   SymmetricMatrix<MT,SO,true,false>::operator=( const Matrix<MT2,SO>& rhs )
{
   using blaze::resize;

   typedef If_< IsSymmetric<MT2>, CompositeType_<MT2>, ResultType_<MT2> >  Tmp;

   if( !IsSquare<MT2>::value && !isSquare( ~rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to symmetric matrix" );
   }

   Tmp tmp( ~rhs );

   if( !IsSymmetric<Tmp>::value && !isSymmetric( tmp ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to symmetric matrix" );
   }

   BLAZE_INTERNAL_ASSERT( !tmp.canAlias( this ), "Aliasing detected" );

   resize( matrix_, tmp.rows(), tmp.columns() );
   if( IsSparseMatrix<Tmp>::value )
      reset();
   assign( tmp );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for matrices with opposite storage order.
//
// \param rhs The right-hand side matrix to be copied.
// \return Reference to the assigned matrix.
// \exception std::invalid_argument Invalid assignment to symmetric matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// If possible and necessary, the matrix is resized according to the given \f$ N \times N \f$
// matrix and initialized as a copy of this matrix. If the matrix cannot be resized accordingly,
// a \a std::invalid_argument exception is thrown. Also note that the given matrix must be a
// symmetric matrix. Otherwise, a \a std::invalid_argument exception is thrown.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side matrix
inline SymmetricMatrix<MT,SO,true,false>&
   SymmetricMatrix<MT,SO,true,false>::operator=( const Matrix<MT2,!SO>& rhs )
{
   return this->operator=( trans( ~rhs ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a general matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side general matrix to be added.
// \return Reference to the matrix.
// \exception std::invalid_argument Invalid assignment to symmetric matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also note that the result of the addition operation must be a symmetric matrix, i.e.
// the given matrix must be a symmetric matrix. In case the result is not a symmetric matrix, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side matrix
inline DisableIf_< IsComputation<MT2>, SymmetricMatrix<MT,SO,true,false>& >
   SymmetricMatrix<MT,SO,true,false>::operator+=( const Matrix<MT2,SO>& rhs )
{
   if( !IsSymmetric<MT2>::value && !isSymmetric( ~rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to symmetric matrix" );
   }

   addAssign( ~rhs );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix computation (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix computation to be added.
// \return Reference to the matrix.
// \exception std::invalid_argument Invalid assignment to symmetric matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also note that the result of the addition operation must be a symmetric matrix, i.e.
// the given matrix must be a symmetric matrix. In case the result is not a symmetric matrix, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side matrix
inline EnableIf_< IsComputation<MT2>, SymmetricMatrix<MT,SO,true,false>& >
   SymmetricMatrix<MT,SO,true,false>::operator+=( const Matrix<MT2,SO>& rhs )
{
   typedef If_< IsSymmetric<MT2>, CompositeType_<MT2>, ResultType_<MT2> >  Tmp;

   if( !IsSquare<MT2>::value && !isSquare( ~rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to symmetric matrix" );
   }

   Tmp tmp( ~rhs );

   if( !IsSymmetric<Tmp>::value && !isSymmetric( tmp ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to symmetric matrix" );
   }

   BLAZE_INTERNAL_ASSERT( !tmp.canAlias( this ), "Aliasing detected" );

   addAssign( tmp );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix with opposite storage order
//        (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added.
// \return Reference to the matrix.
// \exception std::invalid_argument Invalid assignment to symmetric matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also note that the result of the addition operation must be a symmetric matrix, i.e.
// the given matrix must be a symmetric matrix. In case the result is not a symmetric matrix, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side matrix
inline SymmetricMatrix<MT,SO,true,false>&
   SymmetricMatrix<MT,SO,true,false>::operator+=( const Matrix<MT2,!SO>& rhs )
{
   return this->operator+=( trans( ~rhs ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a general matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side general matrix to be subtracted.
// \return Reference to the matrix.
// \exception std::invalid_argument Invalid assignment to symmetric matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also note that the result of the subtraction operation must be a symmetric matrix,
// i.e. the given matrix must be a symmetric matrix. In case the result is not a symmetric matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side matrix
inline DisableIf_< IsComputation<MT2>, SymmetricMatrix<MT,SO,true,false>& >
   SymmetricMatrix<MT,SO,true,false>::operator-=( const Matrix<MT2,SO>& rhs )
{
   if( !IsSymmetric<MT2>::value && !isSymmetric( ~rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to symmetric matrix" );
   }

   subAssign( ~rhs );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix computation (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix computation to be subtracted.
// \return Reference to the matrix.
// \exception std::invalid_argument Invalid assignment to symmetric matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also note that the result of the subtraction operation must be a symmetric matrix,
// i.e. the given matrix must be a symmetric matrix. In case the result is not a symmetric matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side matrix
inline EnableIf_< IsComputation<MT2>, SymmetricMatrix<MT,SO,true,false>& >
   SymmetricMatrix<MT,SO,true,false>::operator-=( const Matrix<MT2,SO>& rhs )
{
   typedef If_< IsSymmetric<MT2>, CompositeType_<MT2>, ResultType_<MT2> >  Tmp;

   if( !IsSquare<MT2>::value && !isSquare( ~rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to symmetric matrix" );
   }

   Tmp tmp( ~rhs );

   if( !IsSymmetric<Tmp>::value && !isSymmetric( tmp ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to symmetric matrix" );
   }

   BLAZE_INTERNAL_ASSERT( !tmp.canAlias( this ), "Aliasing detected" );

   subAssign( tmp );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix with opposite storage
//        order (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted.
// \return Reference to the matrix.
// \exception std::invalid_argument Invalid assignment to symmetric matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also note that the result of the subtraction operation must be a symmetric matrix,
// i.e. the given matrix must be a symmetric matrix. In case the result is not a symmetric matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side matrix
inline SymmetricMatrix<MT,SO,true,false>&
   SymmetricMatrix<MT,SO,true,false>::operator-=( const Matrix<MT2,!SO>& rhs )
{
   return this->operator-=( trans( ~rhs ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of a matrix (\f$ A*=B \f$).
//
// \param rhs The right-hand side matrix for the multiplication.
// \return Reference to the matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also note that the result of the multiplication operation must be a symmetric matrix.
// In case it is not, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted dense matrix
        , bool SO >     // Storage order of the adapted dense matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline SymmetricMatrix<MT,SO,true,false>&
   SymmetricMatrix<MT,SO,true,false>::operator*=( const Matrix<MT2,SO2>& rhs )
{
   using blaze::resize;

   typedef MultTrait_< MT, ResultType_<MT2> >  Tmp;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( Tmp );

   if( matrix_.rows() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to symmetric matrix" );
   }

   Tmp tmp( (*this) * ~rhs );

   if( !isSymmetric( tmp ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to symmetric matrix" );
   }

   resize( matrix_, tmp.rows(), tmp.columns() );
   assign( tmp );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication between a matrix and
//        a scalar value (\f$ A*=s \f$).
//
// \param rhs The right-hand side scalar value for the multiplication.
// \return Reference to the matrix.
*/
template< typename MT       // Type of the adapted dense matrix
        , bool SO >         // Storage order of the adapted dense matrix
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, SymmetricMatrix<MT,SO,true,false> >&
   SymmetricMatrix<MT,SO,true,false>::operator*=( Other rhs )
{
   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j )
         for( size_t i=0UL; i<=j; ++i )
            matrix_(i,j) *= rhs;
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         for( size_t j=0UL; j<=i; ++j )
            matrix_(i,j) *= rhs;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division assignment operator for the division of a matrix by a scalar value
//        (\f$ A/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the matrix.
*/
template< typename MT       // Type of the adapted dense matrix
        , bool SO >         // Storage order of the adapted dense matrix
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, SymmetricMatrix<MT,SO,true,false> >&
   SymmetricMatrix<MT,SO,true,false>::operator/=( Other rhs )
{
   BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j )
         for( size_t i=0UL; i<=j; ++i )
            matrix_(i,j) /= rhs;
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         for( size_t j=0UL; j<=i; ++j )
            matrix_(i,j) /= rhs;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current number of rows of the matrix.
//
// \return The number of rows of the matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t SymmetricMatrix<MT,SO,true,false>::rows() const noexcept
{
   return matrix_.rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current number of columns of the matrix.
//
// \return The number of columns of the matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t SymmetricMatrix<MT,SO,true,false>::columns() const noexcept
{
   return matrix_.columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the spacing between the beginning of two rows/columns.
//
// \return The spacing between the beginning of two rows/columns.
//
// This function returns the spacing between the beginning of two rows/columns, i.e.
// the total number of elements of a row/column. In case the symmetric matrix adapts a
// \a rowMajor dense matrix the function returns the spacing between two rows, in case
// it adapts a \a columnMajor dense matrix the function returns the spacing between two
// columns.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t SymmetricMatrix<MT,SO,true,false>::spacing() const noexcept
{
   return matrix_.spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the matrix.
//
// \return The capacity of the matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t SymmetricMatrix<MT,SO,true,false>::capacity() const noexcept
{
   return matrix_.capacity();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current capacity of the specified row/column.
//
// \param i The index of the row/column.
// \return The current capacity of row/column \a i.
//
// This function returns the current capacity of the specified row/column. In case the symmetric
// matrix adapts a \a rowMajor dense matrix the function returns the capacity of row \a i, in
// case it adapts a \a columnMajor dense matrix the function returns the capacity of column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t SymmetricMatrix<MT,SO,true,false>::capacity( size_t i ) const noexcept
{
   return matrix_.capacity(i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the total number of non-zero elements in the matrix
//
// \return The number of non-zero elements in the symmetric matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t SymmetricMatrix<MT,SO,true,false>::nonZeros() const
{
   size_t nonzeros( 0UL );

   if( SO )
   {
      for( size_t j=0UL; j<columns(); ++j ) {
         for( size_t i=0UL; i<j; ++i ) {
            if( !isDefault( matrix_(i,j) ) )
               nonzeros += 2UL;
         }
         if( !isDefault( matrix_(j,j) ) )
            ++nonzeros;
      }
   }
   else
   {
      for( size_t i=0UL; i<rows(); ++i ) {
         for( size_t j=0UL; j<i; ++j ) {
            if( !isDefault( matrix_(i,j) ) )
               nonzeros += 2UL;
         }
         if( !isDefault( matrix_(i,i) ) )
            ++nonzeros;
      }
   }

   return nonzeros;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the specified row/column.
//
// \param i The index of the row/column.
// \return The number of non-zero elements of row/column \a i.
//
// This function returns the current number of non-zero elements in the specified row/column. In
// case the symmetric matrix adapts a \a rowMajor dense matrix the function returns the number of
// non-zero elements in row \a i, in case it adapts a to \a columnMajor dense matrix the function
// returns the number of non-zero elements in column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t SymmetricMatrix<MT,SO,true,false>::nonZeros( size_t i ) const
{
   size_t nonzeros( 0UL );

   if( SO )
   {
      for( size_t j=0UL; j<i; ++j ) {
         if( !isDefault( matrix_(j,i) ) )
            ++nonzeros;
      }
      for( size_t j=i; j<rows(); ++j ) {
         if( !isDefault( matrix_(i,j) ) )
            ++nonzeros;
      }
   }
   else
   {
      for( size_t j=0UL; j<i; ++j ) {
         if( !isDefault( matrix_(i,j) ) )
            ++nonzeros;
      }
      for( size_t j=i; j<rows(); ++j ) {
         if( !isDefault( matrix_(j,i) ) )
            ++nonzeros;
      }
   }

   return nonzeros;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void SymmetricMatrix<MT,SO,true,false>::reset()
{
   using blaze::clear;

   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j )
         for( size_t i=0UL; i<=j; ++i )
            clear( matrix_(i,j) );
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         for( size_t j=0UL; j<=i; ++j )
            clear( matrix_(i,j) );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset the specified row \b and column to the default initial values.
//
// \param i The index of the row/column.
// \return void
// \exception std::invalid_argument Invalid row/column access index.
//
// This function resets the values in the specified row \b and column to their default value.
// The following example demonstrates this by means of a \f$ 5 \times 5 \f$ symmetric matrix:

   \code
   using blaze::DynamicMatrix;
   using blaze::StaticVector;
   using blaze::SymmetricMatrix;

   SymmetricMatrix< DynamicMatrix< StaticVector<int,1UL> > > A;

   // Initializing the symmetric matrix A to
   //
   //      ( (    ) (  2 ) (  5 ) ( -4 ) (    ) )
   //      ( (  2 ) (  1 ) ( -3 ) (  7 ) (    ) )
   //  A = ( (  5 ) ( -3 ) (  8 ) ( -1 ) ( -2 ) )
   //      ( ( -4 ) (  7 ) ( -1 ) (    ) ( -6 ) )
   //      ( (    ) (  0 ) ( -2 ) ( -6 ) (  1 ) )
   // ...

   // Resetting the 1st row/column results in the matrix
   //
   //      ( (    ) (    ) (  5 ) ( -4 ) (    ) )
   //      ( (    ) (    ) (    ) (    ) (    ) )
   //  A = ( (  5 ) (    ) (  8 ) ( -1 ) ( -2 ) )
   //      ( ( -4 ) (    ) ( -1 ) (    ) ( -6 ) )
   //      ( (    ) (    ) ( -2 ) ( -6 ) (  1 ) )
   //
   A.reset( 1UL );
   \endcode

// Note that the reset() function has no impact on the capacity of the matrix or row/column.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void SymmetricMatrix<MT,SO,true,false>::reset( size_t i )
{
   using blaze::clear;

   for( Iterator element=begin(i); element!=end(i); ++element )
      clear( *element );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the symmetric matrix.
//
// \return void
//
// This function clears the symmetric matrix and returns it to its default state. The function has
// the same effect as calling clear() on the adapted matrix of type \a MT: In case of a resizable
// matrix (for instance DynamicMatrix or HybridMatrix) the number of rows and columns will be set
// to 0, whereas in case of a fixed-size matrix (for instance StaticMatrix) only the elements will
// be reset to their default state.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void SymmetricMatrix<MT,SO,true,false>::clear()
{
   using blaze::clear;

   clear( matrix_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Changing the size of the symmetric matrix.
//
// \param n The new number of rows and columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// In case the symmetric matrix adapts a resizable matrix, this function resizes the matrix using
// the given size to \f$ n \times n \f$. During this operation, new dynamic memory may be allocated
// in case the capacity of the matrix is too small. Note that this function may invalidate all
// existing views (submatrices, rows, columns, ...) on the matrix if it is used to shrink the
// matrix. Additionally, the resize operation potentially changes all matrix elements. In order
// to preserve the old matrix values, the \a preserve flag can be set to \a true. In case the
// size of the matrix is increased, new elements are default initialized.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
void SymmetricMatrix<MT,SO,true,false>::resize( size_t n, bool preserve )
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE( MT );

   UNUSED_PARAMETER( preserve );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square symmetric matrix detected" );

   const size_t oldsize( matrix_.rows() );

   matrix_.resize( n, n, true );

   if( n > oldsize ) {
      const size_t increment( n - oldsize );
      submatrix( matrix_, 0UL, oldsize, oldsize, increment ).reset();
      submatrix( matrix_, oldsize, 0UL, increment, n ).reset();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Extending the size of the matrix.
//
// \param n Number of additional rows and columns.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function increases the matrix size by \a n rows and \a n columns. During this operation,
// new dynamic memory may be allocated in case the capacity of the matrix is too small. Therefore
// this function potentially changes all matrix elements. In order to preserve the old matrix
// values, the \a preserve flag can be set to \a true. The new elements are default initialized.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void SymmetricMatrix<MT,SO,true,false>::extend( size_t n, bool preserve )
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE( MT );

   UNUSED_PARAMETER( preserve );

   resize( rows() + n, true );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setting the minimum capacity of the matrix.
//
// \param elements The new minimum capacity of the symmetric matrix.
// \return void
//
// This function increases the capacity of the symmetric matrix to at least \a elements elements.
// The current values of the matrix elements are preserved.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void SymmetricMatrix<MT,SO,true,false>::reserve( size_t elements )
{
   matrix_.reserve( elements );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transpose of the symmetric matrix.
//
// \return Reference to the transposed matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline SymmetricMatrix<MT,SO,true,false>& SymmetricMatrix<MT,SO,true,false>::transpose()
{
   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place conjugate transpose of the symmetric matrix.
//
// \return Reference to the transposed matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline SymmetricMatrix<MT,SO,true,false>& SymmetricMatrix<MT,SO,true,false>::ctranspose()
{
   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j )
         for( size_t i=0UL; i<=j; ++i )
            conjugate( matrix_(i,j) );
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         for( size_t j=0UL; j<=i; ++j )
            conjugate( matrix_(i,j) );
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the matrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the matrix scaling.
// \return Reference to the matrix.
*/
template< typename MT       // Type of the adapted dense matrix
        , bool SO >         // Storage order of the adapted dense matrix
template< typename Other >  // Data type of the scalar value
inline SymmetricMatrix<MT,SO,true,false>&
   SymmetricMatrix<MT,SO,true,false>::scale( const Other& scalar )
{
   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j )
         for( size_t i=0UL; i<=j; ++i )
            matrix_(i,j) *= scalar;
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         for( size_t j=0UL; j<=i; ++j )
            matrix_(i,j) *= scalar;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Swapping the contents of two matrices.
//
// \param m The matrix to be swapped.
// \return void
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void SymmetricMatrix<MT,SO,true,false>::swap( SymmetricMatrix& m ) noexcept
{
   using std::swap;

   swap( matrix_, m.matrix_ );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DEBUGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the symmetric matrix are intact.
//
// \return \a true in case the symmetric matrix's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the symmetric matrix are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline bool SymmetricMatrix<MT,SO,true,false>::isIntact() const noexcept
{
   using blaze::isIntact;

   return isIntact( matrix_ ) &&
          ( IsCustom<MT>::value || ( SO ? isUpper( matrix_ ) : isLower( matrix_ ) ) );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  EXPRESSION TEMPLATE EVALUATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the matrix can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this matrix, \a false if not.
//
// This function returns whether the given address can alias with the matrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename MT       // Type of the adapted dense matrix
        , bool SO >         // Storage order of the adapted dense matrix
template< typename Other >  // Data type of the foreign expression
inline bool SymmetricMatrix<MT,SO,true,false>::canAlias( const Other* alias ) const noexcept
{
   return matrix_.canAlias( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the matrix is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this matrix, \a false if not.
//
// This function returns whether the given address is aliased with the matrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename MT       // Type of the adapted dense matrix
        , bool SO >         // Storage order of the adapted dense matrix
template< typename Other >  // Data type of the foreign expression
inline bool SymmetricMatrix<MT,SO,true,false>::isAliased( const Other* alias ) const noexcept
{
   return matrix_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the matrix is properly aligned in memory.
//
// \return \a true in case the matrix is aligned, \a false if not.
//
// This function returns whether the matrix is guaranteed to be properly aligned in memory, i.e.
// whether the beginning and the end of each row/column of the matrix are guaranteed to conform
// to the alignment restrictions of the element type \a Type.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline bool SymmetricMatrix<MT,SO,true,false>::isAligned() const noexcept
{
   return matrix_.isAligned();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the matrix can be used in SMP assignments.
//
// \return \a true in case the matrix can be used in SMP assignments, \a false if not.
//
// This function returns whether the matrix can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the matrix).
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline bool SymmetricMatrix<MT,SO,true,false>::canSMPAssign() const noexcept
{
   return matrix_.canSMPAssign();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Optimized implementation of the assignment of a temporary dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void SymmetricMatrix<MT,SO,true,false>::assign( DenseMatrix<MT2,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j )
         for( size_t i=0UL; i<=j; ++i )
            matrix_(i,j) = std::move( (~rhs)(i,j) );
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         for( size_t j=0UL; j<=i; ++j )
            matrix_(i,j) = std::move( (~rhs)(i,j) );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void SymmetricMatrix<MT,SO,true,false>::assign( const DenseMatrix<MT2,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j )
         for( size_t i=0UL; i<=j; ++i )
            matrix_(i,j) = (~rhs)(i,j);
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         for( size_t j=0UL; j<=i; ++j )
            matrix_(i,j) = (~rhs)(i,j);
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void SymmetricMatrix<MT,SO,true,false>::assign( const SparseMatrix<MT2,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   typedef ConstIterator_<MT2>  ConstIterator;

   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j ) {
         const ConstIterator last( (~rhs).upperBound(j,j) );
         for( ConstIterator element=(~rhs).begin(j); element!=last; ++element )
            matrix_(element->index(),j) = element->value();
      }
   }
   else {
      for( size_t i=0UL; i<rows(); ++i ) {
         const ConstIterator last( (~rhs).upperBound(i,i) );
         for( ConstIterator element=(~rhs).begin(i); element!=last; ++element )
            matrix_(i,element->index()) = element->value();
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void SymmetricMatrix<MT,SO,true,false>::addAssign( const DenseMatrix<MT2,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j )
         for( size_t i=0UL; i<=j; ++i )
            matrix_(i,j) += (~rhs)(i,j);
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         for( size_t j=0UL; j<=i; ++j )
            matrix_(i,j) += (~rhs)(i,j);
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void SymmetricMatrix<MT,SO,true,false>::addAssign( const SparseMatrix<MT2,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   typedef ConstIterator_<MT2>  ConstIterator;

   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j ) {
         const ConstIterator last( (~rhs).upperBound(j,j) );
         for( ConstIterator element=(~rhs).begin(j); element!=last; ++element )
            matrix_(element->index(),j) += element->value();
      }
   }
   else {
      for( size_t i=0UL; i<rows(); ++i ) {
         const ConstIterator last( (~rhs).upperBound(i,i) );
         for( ConstIterator element=(~rhs).begin(i); element!=last; ++element )
            matrix_(i,element->index()) += element->value();
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void SymmetricMatrix<MT,SO,true,false>::subAssign( const DenseMatrix<MT2,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j )
         for( size_t i=0UL; i<=j; ++i )
            matrix_(i,j) -= (~rhs)(i,j);
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         for( size_t j=0UL; j<=i; ++j )
            matrix_(i,j) -= (~rhs)(i,j);
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void SymmetricMatrix<MT,SO,true,false>::subAssign( const SparseMatrix<MT2,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   typedef ConstIterator_<MT2>  ConstIterator;

   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j ) {
         const ConstIterator last( (~rhs).upperBound(j,j) );
         for( ConstIterator element=(~rhs).begin(j); element!=last; ++element )
            matrix_(element->index(),j) -= element->value();
      }
   }
   else {
      for( size_t i=0UL; i<rows(); ++i ) {
         const ConstIterator last( (~rhs).upperBound(i,i) );
         for( ConstIterator element=(~rhs).begin(i); element!=last; ++element )
            matrix_(i,element->index()) -= element->value();
      }
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
