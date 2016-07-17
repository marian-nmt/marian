//=================================================================================================
/*!
//  \file blaze/math/views/submatrix/Dense.h
//  \brief Submatrix specialization for dense matrices
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

#ifndef _BLAZE_MATH_VIEWS_SUBMATRIX_DENSE_H_
#define _BLAZE_MATH_VIEWS_SUBMATRIX_DENSE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <iterator>
#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/constraints/ColumnMajorMatrix.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/RowMajorMatrix.h>
#include <blaze/math/constraints/Submatrix.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/TransExpr.h>
#include <blaze/math/constraints/UniTriangular.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/View.h>
#include <blaze/math/Functions.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/StorageOrder.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/DerestrictTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/SubmatrixTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/typetraits/AreSIMDCombinable.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDSub.h>
#include <blaze/math/typetraits/IsDiagonal.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsHermitian.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsRestricted.h>
#include <blaze/math/typetraits/IsSparseMatrix.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/math/typetraits/IsUniUpper.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/math/views/submatrix/BaseTemplate.h>
#include <blaze/system/Blocking.h>
#include <blaze/system/CacheSize.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/AlignmentCheck.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Vectorizable.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Not.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/Template.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsReference.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR UNALIGNED ROW-MAJOR DENSE SUBMATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Submatrix for unaligned row-major dense submatrices.
// \ingroup views
//
// This specialization of Submatrix adapts the class template to the requirements of unaligned
// row-major dense submatrices.
*/
template< typename MT >  // Type of the dense matrix
class Submatrix<MT,unaligned,false,true>
   : public DenseMatrix< Submatrix<MT,unaligned,false,true>, false >
   , private View
{
 private:
   //**Type definitions****************************************************************************
   //! Composite data type of the dense matrix expression.
   typedef If_< IsExpression<MT>, MT, MT& >  Operand;
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef Submatrix<MT,unaligned,false,true>  This;           //!< Type of this Submatrix instance.
   typedef DenseMatrix<This,false>             BaseType;       //!< Base type of this Submatrix instance.
   typedef SubmatrixTrait_<MT>                 ResultType;     //!< Result type for expression template evaluations.
   typedef OppositeType_<ResultType>           OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef TransposeType_<ResultType>          TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<MT>                    ElementType;    //!< Type of the submatrix elements.
   typedef SIMDTrait_<ElementType>             SIMDType;       //!< SIMD type of the submatrix elements.
   typedef ReturnType_<MT>                     ReturnType;     //!< Return type for expression template evaluations
   typedef const Submatrix&                    CompositeType;  //!< Data type for composite expression templates.

   //! Reference to a constant submatrix value.
   typedef ConstReference_<MT>  ConstReference;

   //! Reference to a non-constant submatrix value.
   typedef If_< IsConst<MT>, ConstReference, Reference_<MT> >  Reference;

   //! Pointer to a constant submatrix value.
   typedef const ElementType*  ConstPointer;

   //! Pointer to a non-constant submatrix value.
   typedef If_< Or< IsConst<MT>, Not< HasMutableDataAccess<MT> > >, ConstPointer, ElementType* >  Pointer;
   //**********************************************************************************************

   //**SubmatrixIterator class definition**********************************************************
   /*!\brief Iterator over the elements of the sparse submatrix.
   */
   template< typename IteratorType >  // Type of the dense matrix iterator
   class SubmatrixIterator
   {
    public:
      //**Type definitions*************************************************************************
      //! The iterator category.
      typedef typename std::iterator_traits<IteratorType>::iterator_category  IteratorCategory;

      //! Type of the underlying elements.
      typedef typename std::iterator_traits<IteratorType>::value_type  ValueType;

      //! Pointer return type.
      typedef typename std::iterator_traits<IteratorType>::pointer  PointerType;

      //! Reference return type.
      typedef typename std::iterator_traits<IteratorType>::reference  ReferenceType;

      //! Difference between two iterators.
      typedef typename std::iterator_traits<IteratorType>::difference_type  DifferenceType;

      // STL iterator requirements
      typedef IteratorCategory  iterator_category;  //!< The iterator category.
      typedef ValueType         value_type;         //!< Type of the underlying elements.
      typedef PointerType       pointer;            //!< Pointer return type.
      typedef ReferenceType     reference;          //!< Reference return type.
      typedef DifferenceType    difference_type;    //!< Difference between two iterators.
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Default constructor of the SubmatrixIterator class.
      */
      inline SubmatrixIterator()
         : iterator_ (       )  // Iterator to the current submatrix element
         , isAligned_( false )  // Memory alignment flag
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor of the SubmatrixIterator class.
      //
      // \param iterator Iterator to the initial element.
      // \param isMemoryAligned Memory alignment flag.
      */
      inline SubmatrixIterator( IteratorType iterator, bool isMemoryAligned )
         : iterator_ ( iterator        )  // Iterator to the current submatrix element
         , isAligned_( isMemoryAligned )  // Memory alignment flag
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Conversion constructor from different SubmatrixIterator instances.
      //
      // \param it The submatrix iterator to be copied.
      */
      template< typename IteratorType2 >
      inline SubmatrixIterator( const SubmatrixIterator<IteratorType2>& it )
         : iterator_ ( it.base()      )  // Iterator to the current submatrix element
         , isAligned_( it.isAligned() )  // Memory alignment flag
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline SubmatrixIterator& operator+=( size_t inc ) {
         iterator_ += inc;
         return *this;
      }
      //*******************************************************************************************

      //**Subtraction assignment operator**********************************************************
      /*!\brief Subtraction assignment operator.
      //
      // \param dec The decrement of the iterator.
      // \return The decremented iterator.
      */
      inline SubmatrixIterator& operator-=( size_t dec ) {
         iterator_ -= dec;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline SubmatrixIterator& operator++() {
         ++iterator_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const SubmatrixIterator operator++( int ) {
         return SubmatrixIterator( iterator_++, isAligned_ );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline SubmatrixIterator& operator--() {
         --iterator_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const SubmatrixIterator operator--( int ) {
         return SubmatrixIterator( iterator_--, isAligned_ );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReferenceType operator*() const {
         return *iterator_;
      }
      //*******************************************************************************************

      //**Load function****************************************************************************
      /*!\brief Load of a SIMD element of the dense submatrix.
      //
      // \return The loaded SIMD element.
      //
      // This function performs a load of the current SIMD element of the submatrix iterator.
      // This function must \b NOT be called explicitly! It is used internally for the performance
      // optimized evaluation of expression templates. Calling this function explicitly might
      // result in erroneous results and/or in compilation errors.
      */
      inline SIMDType load() const noexcept {
         if( isAligned_ )
            return loada();
         else
            return loadu();
      }
      //*******************************************************************************************

      //**Loada function***************************************************************************
      /*!\brief Aligned load of a SIMD element of the dense submatrix.
      //
      // \return The loaded SIMD element.
      //
      // This function performs an aligned load of the current SIMD element of the submatrix
      // iterator. This function must \b NOT be called explicitly! It is used internally for
      // the performance optimized evaluation of expression templates. Calling this function
      // explicitly might result in erroneous results and/or in compilation errors.
      */
      inline SIMDType loada() const noexcept {
         return iterator_.loada();
      }
      //*******************************************************************************************

      //**Loadu function***************************************************************************
      /*!\brief Unaligned load of a SIMD element of the dense submatrix.
      //
      // \return The loaded SIMD element.
      //
      // This function performs an unaligned load of the current SIMD element of the submatrix
      // iterator. This function must \b NOT be called explicitly! It is used internally for the
      // performance optimized evaluation of expression templates. Calling this function explicitly
      // might result in erroneous results and/or in compilation errors.
      */
      inline SIMDType loadu() const noexcept {
         return iterator_.loadu();
      }
      //*******************************************************************************************

      //**Store function***************************************************************************
      /*!\brief Store of a SIMD element of the dense submatrix.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs a store of the current SIMD element of the submatrix iterator.
      // This function must \b NOT be called explicitly! It is used internally for the performance
      // optimized evaluation of expression templates. Calling this function explicitly might
      // result in erroneous results and/or in compilation errors.
      */
      inline void store( const SIMDType& value ) const {
         storeu( value );
      }
      //*******************************************************************************************

      //**Storea function**************************************************************************
      /*!\brief Aligned store of a SIMD element of the dense submatrix.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs an aligned store of the current SIMD element of the submatrix
      // iterator. This function must \b NOT be called explicitly! It is used internally for the
      // performance optimized evaluation of expression templates. Calling this function explicitly
      // might result in erroneous results and/or in compilation errors.
      */
      inline void storea( const SIMDType& value ) const {
         iterator_.storea( value );
      }
      //*******************************************************************************************

      //**Storeu function**************************************************************************
      /*!\brief Unaligned store of a SIMD element of the dense submatrix.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs an unaligned store of the current SIMD element of the submatrix
      // iterator. This function must \b NOT be called explicitly! It is used internally for the
      // performance optimized evaluation of expression templates. Calling this function explicitly
      // might result in erroneous results and/or in compilation errors.
      */
      inline void storeu( const SIMDType& value ) const {
         if( isAligned_ ) {
            iterator_.storea( value );
         }
         else {
            iterator_.storeu( value );
         }
      }
      //*******************************************************************************************

      //**Stream function**************************************************************************
      /*!\brief Aligned, non-temporal store of a SIMD element of the dense submatrix.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs an aligned, non-temporal store of the current SIMD element of the
      // submatrix iterator. This function must \b NOT be called explicitly! It is used internally
      // for the performance optimized evaluation of expression templates. Calling this function
      // explicitly might result in erroneous results and/or in compilation errors.
      */
      inline void stream( const SIMDType& value ) const {
         iterator_.stream( value );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const SubmatrixIterator& rhs ) const {
         return iterator_ == rhs.iterator_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const SubmatrixIterator& rhs ) const {
         return iterator_ != rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const SubmatrixIterator& rhs ) const {
         return iterator_ < rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const SubmatrixIterator& rhs ) const {
         return iterator_ > rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const SubmatrixIterator& rhs ) const {
         return iterator_ <= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const SubmatrixIterator& rhs ) const {
         return iterator_ >= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const SubmatrixIterator& rhs ) const {
         return iterator_ - rhs.iterator_;
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between a SubmatrixIterator and an integral value.
      //
      // \param it The iterator to be incremented.
      // \param inc The number of elements the iterator is incremented.
      // \return The incremented iterator.
      */
      friend inline const SubmatrixIterator operator+( const SubmatrixIterator& it, size_t inc ) {
         return SubmatrixIterator( it.iterator_ + inc, it.isAligned_ );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between an integral value and a SubmatrixIterator.
      //
      // \param inc The number of elements the iterator is incremented.
      // \param it The iterator to be incremented.
      // \return The incremented iterator.
      */
      friend inline const SubmatrixIterator operator+( size_t inc, const SubmatrixIterator& it ) {
         return SubmatrixIterator( it.iterator_ + inc, it.isAligned_ );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Subtraction between a SubmatrixIterator and an integral value.
      //
      // \param it The iterator to be decremented.
      // \param dec The number of elements the iterator is decremented.
      // \return The decremented iterator.
      */
      friend inline const SubmatrixIterator operator-( const SubmatrixIterator& it, size_t dec ) {
         return SubmatrixIterator( it.iterator_ - dec, it.isAligned_ );
      }
      //*******************************************************************************************

      //**Base function****************************************************************************
      /*!\brief Access to the current position of the submatrix iterator.
      //
      // \return The current position of the submatrix iterator.
      */
      inline IteratorType base() const {
         return iterator_;
      }
      //*******************************************************************************************

      //**IsAligned function***********************************************************************
      /*!\brief Access to the iterator's memory alignment flag.
      //
      // \return \a true in case the iterator is aligned, \a false if it is not.
      */
      inline bool isAligned() const noexcept {
         return isAligned_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType iterator_;   //!< Iterator to the current submatrix element.
      bool         isAligned_;  //!< Memory alignment flag.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   //! Iterator over constant elements.
   typedef SubmatrixIterator< ConstIterator_<MT> >  ConstIterator;

   //! Iterator over non-constant elements.
   typedef If_< IsConst<MT>, ConstIterator, SubmatrixIterator< Iterator_<MT> > >  Iterator;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = MT::simdEnabled };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = MT::smpAssignable };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline Submatrix( Operand matrix, size_t rindex, size_t cindex, size_t m, size_t n );
   // No explicitly declared copy constructor.
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
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Pointer        data  ( size_t i ) noexcept;
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
   inline Submatrix& operator=( const ElementType& rhs );
   inline Submatrix& operator=( initializer_list< initializer_list<ElementType> > list );
   inline Submatrix& operator=( const Submatrix& rhs );

   template< typename MT2, bool SO2 >
   inline Submatrix& operator=( const Matrix<MT2,SO2>& rhs );

   template< typename MT2, bool SO2 >
   inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator+=( const Matrix<MT2,SO2>& rhs );

   template< typename MT2, bool SO2 >
   inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator+=( const Matrix<MT2,SO2>& rhs );

   template< typename MT2, bool SO2 >
   inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator-=( const Matrix<MT2,SO2>& rhs );

   template< typename MT2, bool SO2 >
   inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator-=( const Matrix<MT2,SO2>& rhs );

   template< typename MT2, bool SO2 >
   inline Submatrix& operator*=( const Matrix<MT2,SO2>& rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, Submatrix >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, Submatrix >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t     row() const noexcept;
                              inline size_t     rows() const noexcept;
                              inline size_t     column() const noexcept;
                              inline size_t     columns() const noexcept;
                              inline size_t     spacing() const noexcept;
                              inline size_t     capacity() const noexcept;
                              inline size_t     capacity( size_t i ) const noexcept;
                              inline size_t     nonZeros() const;
                              inline size_t     nonZeros( size_t i ) const;
                              inline void       reset();
                              inline void       reset( size_t i );
                              inline Submatrix& transpose();
                              inline Submatrix& ctranspose();
   template< typename Other > inline Submatrix& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT2 >
   struct VectorizedAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<MT2> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT2 >
   struct VectorizedAddAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<MT2> >::value &&
                            HasSIMDAdd< ElementType, ElementType_<MT2> >::value &&
                            !IsDiagonal<MT2>::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT2 >
   struct VectorizedSubAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<MT2> >::value &&
                            HasSIMDSub< ElementType, ElementType_<MT2> >::value &&
                            !IsDiagonal<MT2>::value };
   };
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   enum : size_t { SIMDSIZE = SIMDTrait<ElementType>::size };
   //**********************************************************************************************

 public:
   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other >
   inline bool canAlias( const Other* alias ) const noexcept;

   template< typename MT2, bool AF2, bool SO2 >
   inline bool canAlias( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< typename MT2, bool AF2, bool SO2 >
   inline bool isAliased( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept;

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t i, size_t j ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t i, size_t j, const SIMDType& value ) noexcept;

   template< typename MT2 >
   inline DisableIf_< VectorizedAssign<MT2> > assign( const DenseMatrix<MT2,false>& rhs );

   template< typename MT2 >
   inline EnableIf_< VectorizedAssign<MT2> > assign( const DenseMatrix<MT2,false>& rhs );

   template< typename MT2 > inline void assign( const DenseMatrix<MT2,true>&  rhs );
   template< typename MT2 > inline void assign( const SparseMatrix<MT2,false>&  rhs );
   template< typename MT2 > inline void assign( const SparseMatrix<MT2,true>& rhs );

   template< typename MT2 >
   inline DisableIf_< VectorizedAddAssign<MT2> > addAssign( const DenseMatrix<MT2,false>& rhs );

   template< typename MT2 >
   inline EnableIf_< VectorizedAddAssign<MT2> > addAssign( const DenseMatrix<MT2,false>& rhs );

   template< typename MT2 > inline void addAssign( const DenseMatrix<MT2,true>&  rhs );
   template< typename MT2 > inline void addAssign( const SparseMatrix<MT2,false>&  rhs );
   template< typename MT2 > inline void addAssign( const SparseMatrix<MT2,true>& rhs );

   template< typename MT2 >
   inline DisableIf_< VectorizedSubAssign<MT2> > subAssign( const DenseMatrix<MT2,false>& rhs );

   template< typename MT2 >
   inline EnableIf_< VectorizedSubAssign<MT2> > subAssign( const DenseMatrix<MT2,false>& rhs );

   template< typename MT2 > inline void subAssign( const DenseMatrix<MT2,true>&  rhs );
   template< typename MT2 > inline void subAssign( const SparseMatrix<MT2,false>&  rhs );
   template< typename MT2 > inline void subAssign( const SparseMatrix<MT2,true>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline bool hasOverlap() const noexcept;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand      matrix_;   //!< The dense matrix containing the submatrix.
   const size_t row_;      //!< The first row of the submatrix.
   const size_t column_;   //!< The first column of the submatrix.
   const size_t m_;        //!< The number of rows of the submatrix.
   const size_t n_;        //!< The number of columns of the submatrix.
   const bool isAligned_;  //!< Memory alignment flag.
                           /*!< The alignment flag indicates whether the submatrix is fully aligned
                                with respect to the given element type and the available instruction
                                set. In case the submatrix is fully aligned it is possible to use
                                aligned loads and stores instead of unaligned loads and stores. In
                                order to be aligned, the first element of each row/column must be
                                aligned. */
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename MT2, bool AF2, bool SO2, bool DF2 > friend class Submatrix;

   template< bool AF1, typename MT2, bool AF2, bool SO2, bool DF2 >
   friend const Submatrix<MT2,AF1,SO2,DF2>
      submatrix( const Submatrix<MT2,AF2,SO2,DF2>& sm, size_t row, size_t column, size_t m, size_t n );

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isIntact( const Submatrix<MT2,AF2,SO2,DF2>& sm ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isSame( const Submatrix<MT2,AF2,SO2,DF2>& a, const Matrix<MT2,SO2>& b ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isSame( const Matrix<MT2,SO2>& a, const Submatrix<MT2,AF2,SO2,DF2>& b ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isSame( const Submatrix<MT2,AF2,SO2,DF2>& a, const Submatrix<MT2,AF2,SO2,DF2>& b ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool tryAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                          size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename MT3, bool SO3 >
   friend bool tryAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Matrix<MT3,SO3>& rhs,
                          size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool tryAddAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename MT3, bool SO3 >
   friend bool tryAddAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Matrix<MT3,SO3>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool trySubAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename MT3, bool SO3 >
   friend bool trySubAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Matrix<MT3,SO3>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool tryMultAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                              size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend DerestrictTrait_< Submatrix<MT2,AF2,SO2,DF2> > derestrict( Submatrix<MT2,AF2,SO2,DF2>& sm );
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE   ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SUBMATRIX_TYPE   ( MT );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE     ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE   ( MT );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for Submatrix.
//
// \param matrix The dense matrix containing the submatrix.
// \param rindex The index of the first row of the submatrix in the given dense matrix.
// \param cindex The index of the first column of the submatrix in the given dense matrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \exception std::invalid_argument Invalid submatrix specification.
//
// In case the submatrix is not properly specified (i.e. if the specified submatrix is not
// contained in the given dense matrix) a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,unaligned,false,true>::Submatrix( Operand matrix, size_t rindex, size_t cindex, size_t m, size_t n )
   : matrix_   ( matrix )  // The dense matrix containing the submatrix
   , row_      ( rindex )  // The first row of the submatrix
   , column_   ( cindex )  // The first column of the submatrix
   , m_        ( m      )  // The number of rows of the submatrix
   , n_        ( n      )  // The number of columns of the submatrix
   , isAligned_( simdEnabled && matrix.data() != nullptr && checkAlignment( data() ) &&
                 ( m < 2UL || ( matrix.spacing() & size_t(-SIMDSIZE) ) == 0UL ) )
{
   if( ( row_ + m_ > matrix_.rows() ) || ( column_ + n_ > matrix_.columns() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid submatrix specification" );
   }
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
/*!\brief 2D-access to the dense submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::Reference
   Submatrix<MT,unaligned,false,true>::operator()( size_t i, size_t j )
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return matrix_(row_+i,column_+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the dense submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::ConstReference
   Submatrix<MT,unaligned,false,true>::operator()( size_t i, size_t j ) const
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return const_cast<const MT&>( matrix_ )(row_+i,column_+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::Reference
   Submatrix<MT,unaligned,false,true>::at( size_t i, size_t j )
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
/*!\brief Checked access to the submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::ConstReference
   Submatrix<MT,unaligned,false,true>::at( size_t i, size_t j ) const
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
/*!\brief Low-level data access to the submatrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense submatrix. Note that
// you can NOT assume that all matrix elements lie adjacent to each other! The dense submatrix
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::Pointer
   Submatrix<MT,unaligned,false,true>::data() noexcept
{
   return matrix_.data() + row_*spacing() + column_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the submatrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense submatrix. Note that
// you can NOT assume that all matrix elements lie adjacent to each other! The dense submatrix
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::ConstPointer
   Submatrix<MT,unaligned,false,true>::data() const noexcept
{
   return matrix_.data() + row_*spacing() + column_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the submatrix elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::Pointer
   Submatrix<MT,unaligned,false,true>::data( size_t i ) noexcept
{
   return matrix_.data() + (row_+i)*spacing() + column_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the submatrix elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::ConstPointer
   Submatrix<MT,unaligned,false,true>::data( size_t i ) const noexcept
{
   return matrix_.data() + (row_+i)*spacing() + column_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::Iterator
   Submatrix<MT,unaligned,false,true>::begin( size_t i )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense submatrix row access index" );
   return Iterator( matrix_.begin( row_ + i ) + column_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::ConstIterator
   Submatrix<MT,unaligned,false,true>::begin( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense submatrix row access index" );
   return ConstIterator( matrix_.cbegin( row_ + i ) + column_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::ConstIterator
   Submatrix<MT,unaligned,false,true>::cbegin( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense submatrix row access index" );
   return ConstIterator( matrix_.cbegin( row_ + i ) + column_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::Iterator
   Submatrix<MT,unaligned,false,true>::end( size_t i )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense submatrix row access index" );
   return Iterator( matrix_.begin( row_ + i ) + column_ + n_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::ConstIterator
   Submatrix<MT,unaligned,false,true>::end( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense submatrix row access index" );
   return ConstIterator( matrix_.cbegin( row_ + i ) + column_ + n_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,false,true>::ConstIterator
   Submatrix<MT,unaligned,false,true>::cend( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense submatrix row access index" );
   return ConstIterator( matrix_.cbegin( row_ + i ) + column_ + n_, isAligned_ );
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
/*!\brief Homogenous assignment to all submatrix elements.
//
// \param rhs Scalar value to be assigned to all submatrix elements.
// \return Reference to the assigned submatrix.
//
// This function homogeneously assigns the given value to all dense matrix elements. Note that in
// case the underlying dense matrix is a lower/upper matrix only lower/upper and diagonal elements
// of the underlying matrix are modified.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,unaligned,false,true>&
   Submatrix<MT,unaligned,false,true>::operator=( const ElementType& rhs )
{
   const size_t iend( row_ + m_ );

   for( size_t i=row_; i<iend; ++i )
   {
      const size_t jbegin( ( IsUpper<MT>::value )
                           ?( ( IsUniUpper<MT>::value || IsStrictlyUpper<MT>::value )
                              ?( max( i+1UL, column_ ) )
                              :( max( i, column_ ) ) )
                           :( column_ ) );
      const size_t jend  ( ( IsLower<MT>::value )
                           ?( ( IsUniLower<MT>::value || IsStrictlyLower<MT>::value )
                              ?( min( i, column_+n_ ) )
                              :( min( i+1UL, column_+n_ ) ) )
                           :( column_+n_ ) );

      for( size_t j=jbegin; j<jend; ++j )
         matrix_(i,j) = rhs;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all submatrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to submatrix.
//
// This assignment operator offers the option to directly assign to all elements of the submatrix
// by means of an initializer list. The submatrix elements are assigned the values from the given
// initializer list. Missing values are initialized as default. Note that in case the size
// of the top-level initializer list exceeds the number of rows or the size of any nested list
// exceeds the number of columns, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,unaligned,false,true>&
   Submatrix<MT,unaligned,false,true>::operator=( initializer_list< initializer_list<ElementType> > list )
{
   if( list.size() != rows() || determineColumns( list ) > columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to submatrix" );
   }

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      std::fill( std::copy( rowList.begin(), rowList.end(), begin(i) ), end(i), ElementType() );
      ++i;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for Submatrix.
//
// \param rhs Sparse submatrix to be copied.
// \return Reference to the assigned submatrix.
// \exception std::invalid_argument Submatrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The dense submatrix is initialized as a copy of the given dense submatrix. In case the current
// sizes of the two submatrices don't match, a \a std::invalid_argument exception is thrown. Also,
// if the underlying matrix \a MT is a lower triangular, upper triangular, or symmetric matrix
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,unaligned,false,true>&
   Submatrix<MT,unaligned,false,true>::operator=( const Submatrix& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( this == &rhs || ( &matrix_ == &rhs.matrix_ && row_ == rhs.row_ && column_ == rhs.column_ ) )
      return *this;

   if( rows() != rhs.rows() || columns() != rhs.columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Submatrix sizes do not match" );
   }

   if( !tryAssign( matrix_, rhs, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( rhs.canAlias( &matrix_ ) ) {
      const ResultType tmp( rhs );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different matrices.
//
// \param rhs Matrix to be assigned.
// \return Reference to the assigned submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The dense submatrix is initialized as a copy of the given matrix. In case the current sizes
// of the two matrices don't match, a \a std::invalid_argument exception is thrown. Also, if
// the underlying matrix \a MT is a lower triangular, upper triangular, or symmetric matrix
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline Submatrix<MT,unaligned,false,true>&
   Submatrix<MT,unaligned,false,true>::operator=( const Matrix<MT2,SO2>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   typedef If_< IsRestricted<MT>, CompositeType_<MT2>, const MT2& >  Right;
   Right right( ~rhs );

   if( !tryAssign( matrix_, right, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   if( IsSparseMatrix<MT2>::value )
      reset();

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &matrix_ ) ) {
      const ResultType_<MT2> tmp( right );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,unaligned,false,true>& >
   Submatrix<MT,unaligned,false,true>::operator+=( const Matrix<MT2,SO2>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef AddTrait_< ResultType, ResultType_<MT2> >  AddType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( !tryAddAssign( matrix_, ~rhs, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( ( ( IsSymmetric<MT>::value || IsHermitian<MT>::value ) && hasOverlap() ) ||
       (~rhs).canAlias( &matrix_ ) ) {
      const AddType tmp( *this + (~rhs) );
      smpAssign( left, tmp );
   }
   else {
      smpAddAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,unaligned,false,true>& >
   Submatrix<MT,unaligned,false,true>::operator+=( const Matrix<MT2,SO2>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef AddTrait_< ResultType, ResultType_<MT2> >  AddType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const AddType tmp( *this + (~rhs) );

   if( !tryAssign( matrix_, tmp, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,unaligned,false,true>& >
   Submatrix<MT,unaligned,false,true>::operator-=( const Matrix<MT2,SO2>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef SubTrait_< ResultType, ResultType_<MT2> >  SubType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( !trySubAssign( matrix_, ~rhs, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( ( ( IsSymmetric<MT>::value || IsHermitian<MT>::value ) && hasOverlap() ) ||
       (~rhs).canAlias( &matrix_ ) ) {
      const SubType tmp( *this - (~rhs ) );
      smpAssign( left, tmp );
   }
   else {
      smpSubAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,unaligned,false,true>& >
   Submatrix<MT,unaligned,false,true>::operator-=( const Matrix<MT2,SO2>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef SubTrait_< ResultType, ResultType_<MT2> >  SubType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const SubType tmp( *this - (~rhs) );

   if( !tryAssign( matrix_, tmp, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of a matrix (\f$ A*=B \f$).
//
// \param rhs The right-hand side matrix for the multiplication.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline Submatrix<MT,unaligned,false,true>&
   Submatrix<MT,unaligned,false,true>::operator*=( const Matrix<MT2,SO2>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef MultTrait_< ResultType, ResultType_<MT2> >  MultType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( MultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MultType );

   if( columns() != (~rhs).rows() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const MultType tmp( *this * (~rhs) );

   if( !tryAssign( matrix_, tmp, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication between a dense submatrix
//        and a scalar value (\f$ A*=s \f$).
//
// \param rhs The right-hand side scalar value for the multiplication.
// \return Reference to the dense submatrix.
//
// This operator cannot be used for submatrices on lower or upper unitriangular matrices. The
// attempt to scale such a submatrix results in a compilation error!
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, Submatrix<MT,unaligned,false,true> >&
   Submatrix<MT,unaligned,false,true>::operator*=( Other rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   DerestrictTrait_<This> left( derestrict( *this ) );
   smpAssign( left, (*this) * rhs );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division assignment operator for the division of a dense submatrix by a scalar value
//        (\f$ A/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the dense submatrix.
//
// This operator cannot be used for submatrices on lower or upper unitriangular matrices. The
// attempt to scale such a submatrix results in a compilation error!
//
// \note A division by zero is only checked by an user assert.
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, Submatrix<MT,unaligned,false,true> >&
   Submatrix<MT,unaligned,false,true>::operator/=( Other rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

   DerestrictTrait_<This> left( derestrict( *this ) );
   smpAssign( left, (*this) / rhs );

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
/*!\brief Returns the index of the first row of the submatrix in the underlying dense matrix.
//
// \return The index of the first row.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,false,true>::row() const noexcept
{
   return row_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of rows of the dense submatrix.
//
// \return The number of rows of the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,false,true>::rows() const noexcept
{
   return m_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first column of the submatrix in the underlying dense matrix.
//
// \return The index of the first column.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,false,true>::column() const noexcept
{
   return column_;
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of columns of the dense submatrix.
//
// \return The number of columns of the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,false,true>::columns() const noexcept
{
   return n_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the spacing between the beginning of two rows/columns.
//
// \return The spacing between the beginning of two rows/columns.
//
// This function returns the spacing between the beginning of two rows/columns, i.e. the
// total number of elements of a row/column. In case the storage order is set to \a rowMajor
// the function returns the spacing between two rows, in case the storage flag is set to
// \a columnMajor the function returns the spacing between two columns.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,false,true>::spacing() const noexcept
{
   return matrix_.spacing();
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense submatrix.
//
// \return The capacity of the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,false,true>::capacity() const noexcept
{
   return rows() * columns();
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
// This function returns the current capacity of the specified row/column. In case the
// storage order is set to \a rowMajor the function returns the capacity of row \a i,
// in case the storage flag is set to \a columnMajor the function returns the capacity
// of column \a i.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,false,true>::capacity( size_t i ) const noexcept
{
   UNUSED_PARAMETER( i );

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   return columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the dense submatrix
//
// \return The number of non-zero elements in the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,false,true>::nonZeros() const
{
   const size_t iend( row_ + m_ );
   const size_t jend( column_ + n_ );
   size_t nonzeros( 0UL );

   for( size_t i=row_; i<iend; ++i )
      for( size_t j=column_; j<jend; ++j )
         if( !isDefault( matrix_(i,j) ) )
            ++nonzeros;

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
// This function returns the current number of non-zero elements in the specified row/column.
// In case the storage order is set to \a rowMajor the function returns the number of non-zero
// elements in row \a i, in case the storage flag is set to \a columnMajor the function returns
// the number of non-zero elements in column \a i.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,false,true>::nonZeros( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   const size_t jend( column_ + n_ );
   size_t nonzeros( 0UL );

   for( size_t j=column_; j<jend; ++j )
      if( !isDefault( matrix_(row_+i,j) ) )
         ++nonzeros;

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
template< typename MT >  // Type of the dense matrix
inline void Submatrix<MT,unaligned,false,true>::reset()
{
   using blaze::clear;

   for( size_t i=row_; i<row_+m_; ++i )
   {
      const size_t jbegin( ( IsUpper<MT>::value )
                           ?( ( IsUniUpper<MT>::value || IsStrictlyUpper<MT>::value )
                              ?( max( i+1UL, column_ ) )
                              :( max( i, column_ ) ) )
                           :( column_ ) );
      const size_t jend  ( ( IsLower<MT>::value )
                           ?( ( IsUniLower<MT>::value || IsStrictlyLower<MT>::value )
                              ?( min( i, column_+n_ ) )
                              :( min( i+1UL, column_+n_ ) ) )
                           :( column_+n_ ) );

      for( size_t j=jbegin; j<jend; ++j )
         clear( matrix_(i,j) );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline void Submatrix<MT,unaligned,false,true>::reset( size_t i )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   const size_t jbegin( ( IsUpper<MT>::value )
                        ?( ( IsUniUpper<MT>::value || IsStrictlyUpper<MT>::value )
                           ?( max( i+1UL, column_ ) )
                           :( max( i, column_ ) ) )
                        :( column_ ) );
   const size_t jend  ( ( IsLower<MT>::value )
                        ?( ( IsUniLower<MT>::value || IsStrictlyLower<MT>::value )
                           ?( min( i, column_+n_ ) )
                           :( min( i+1UL, column_+n_ ) ) )
                        :( column_+n_ ) );

   for( size_t j=jbegin; j<jend; ++j )
      clear( matrix_(row_+i,j) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transpose of the submatrix.
//
// \return Reference to the transposed submatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic submatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense submatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the submatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the submatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,unaligned,false,true>& Submatrix<MT,unaligned,false,true>::transpose()
{
   if( m_ != n_ ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic submatrix" );
   }

   if( !tryAssign( matrix_, trans( *this ), row_, column_ ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );
   const ResultType tmp( trans( *this ) );
   smpAssign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place conjugate transpose of the submatrix.
//
// \return Reference to the transposed submatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic submatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense submatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the submatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the submatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,unaligned,false,true>& Submatrix<MT,unaligned,false,true>::ctranspose()
{
   if( m_ != n_ ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic submatrix" );
   }

   if( !tryAssign( matrix_, ctrans( *this ), row_, column_ ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );
   const ResultType tmp( ctrans( *this ) );
   smpAssign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the dense submatrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the submatrix scaling.
// \return Reference to the dense submatrix.
//
// This function scales all elements of the submatrix by the given scalar value \a scalar. Note
// that the function cannot be used to scale a submatrix on a lower or upper unitriangular matrix.
// The attempt to scale such a submatrix results in a compile time error!
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the scalar value
inline Submatrix<MT,unaligned,false,true>&
   Submatrix<MT,unaligned,false,true>::scale( const Other& scalar )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   const size_t iend( row_ + m_ );

   for( size_t i=row_; i<iend; ++i )
   {
      const size_t jbegin( ( IsUpper<MT>::value )
                           ?( ( IsStrictlyUpper<MT>::value )
                              ?( max( i+1UL, column_ ) )
                              :( max( i, column_ ) ) )
                           :( column_ ) );
      const size_t jend  ( ( IsLower<MT>::value )
                           ?( ( IsStrictlyLower<MT>::value )
                              ?( min( i, column_+n_ ) )
                              :( min( i+1UL, column_+n_ ) ) )
                           :( column_+n_ ) );

      for( size_t j=jbegin; j<jend; ++j )
         matrix_(i,j) *= scalar;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checking whether there exists an overlap in the context of a symmetric matrix.
//
// \return \a true in case an overlap exists, \a false if not.
//
// This function checks if in the context of a symmetric matrix the submatrix has an overlap with
// its counterpart. In case an overlap exists, the function return \a true, otherwise it returns
// \a false.
*/
template< typename MT >  // Type of the dense matrix
inline bool Submatrix<MT,unaligned,false,true>::hasOverlap() const noexcept
{
   BLAZE_INTERNAL_ASSERT( IsSymmetric<MT>::value || IsHermitian<MT>::value, "Invalid matrix detected" );

   if( ( row_ + m_ <= column_ ) || ( column_ + n_ <= row_ ) )
      return false;
   else return true;
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
/*!\brief Returns whether the submatrix can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address can alias with the submatrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the foreign expression
inline bool Submatrix<MT,unaligned,false,true>::canAlias( const Other* alias ) const noexcept
{
   return matrix_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix can alias with the given dense submatrix \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address can alias with the submatrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Data type of the foreign dense submatrix
        , bool AF2       // Alignment flag of the foreign dense submatrix
        , bool SO2 >     // Storage order of the foreign dense submatrix
inline bool Submatrix<MT,unaligned,false,true>::canAlias( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept
{
   return ( matrix_.isAliased( &alias->matrix_ ) &&
            ( row_    + m_ > alias->row_    ) && ( row_    < alias->row_    + alias->m_ ) &&
            ( column_ + n_ > alias->column_ ) && ( column_ < alias->column_ + alias->n_ ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address is aliased with the submatrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the foreign expression
inline bool Submatrix<MT,unaligned,false,true>::isAliased( const Other* alias ) const noexcept
{
   return matrix_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is aliased with the given dense submatrix \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address is aliased with the submatrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Data type of the foreign dense submatrix
        , bool AF2       // Alignment flag of the foreign dense submatrix
        , bool SO2 >     // Storage order of the foreign dense submatrix
inline bool Submatrix<MT,unaligned,false,true>::isAliased( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept
{
   return ( matrix_.isAliased( &alias->matrix_ ) &&
            ( row_    + m_ > alias->row_    ) && ( row_    < alias->row_    + alias->m_ ) &&
            ( column_ + n_ > alias->column_ ) && ( column_ < alias->column_ + alias->n_ ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is properly aligned in memory.
//
// \return \a true in case the submatrix is aligned, \a false if not.
//
// This function returns whether the submatrix is guaranteed to be properly aligned in memory,
// i.e. whether the beginning and the end of each row/column of the submatrix are guaranteed to
// conform to the alignment restrictions of the underlying element type.
*/
template< typename MT >  // Type of the dense matrix
inline bool Submatrix<MT,unaligned,false,true>::isAligned() const noexcept
{
   return isAligned_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix can be used in SMP assignments.
//
// \return \a true in case the submatrix can be used in SMP assignments, \a false if not.
//
// This function returns whether the submatrix can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the submatrix).
*/
template< typename MT >  // Type of the dense matrix
inline bool Submatrix<MT,unaligned,false,true>::canSMPAssign() const noexcept
{
   return ( rows() > SMP_DMATASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Load of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense submatrix. The row
// index must be smaller than the number of rows and the column index must be smaller than
// the number of columns. Additionally, the column index (in case of a row-major matrix) or
// the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is
// used internally for the performance optimized evaluation of expression templates. Calling
// this function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename Submatrix<MT,unaligned,false,true>::SIMDType
   Submatrix<MT,unaligned,false,true>::load( size_t i, size_t j ) const noexcept
{
   if( isAligned_ )
      return loada( i, j );
   else
      return loadu( i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major matrix)
// or the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename Submatrix<MT,unaligned,false,true>::SIMDType
   Submatrix<MT,unaligned,false,true>::loada( size_t i, size_t j ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   return matrix_.loada( row_+i, column_+j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned load of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major matrix)
// or the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename Submatrix<MT,unaligned,false,true>::SIMDType
   Submatrix<MT,unaligned,false,true>::loadu( size_t i, size_t j ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   return matrix_.loadu( row_+i, column_+j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store of a specific SIMD element of the dense submatrix. The row
// index must be smaller than the number of rows and the column index must be smaller than the
// number of columns. Additionally, the column index (in case of a row-major matrix) or the row
// index (in case of a column-major matrix) must be a multiple of the number of values inside
// the SIMD element. This function must \b NOT be called explicitly! It is used internally
// for the performance optimized evaluation of expression templates. Calling this function
// explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,unaligned,false,true>::store( size_t i, size_t j, const SIMDType& value ) noexcept
{
   if( isAligned_ )
      storea( i, j, value );
   else
      storeu( i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major matrix)
// or the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,unaligned,false,true>::storea( size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   matrix_.storea( row_+i, column_+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major matrix)
// or the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,unaligned,false,true>::storeu( size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   matrix_.storeu( row_+i, column_+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned, non-temporal store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store of a specific SIMD element of the dense
// submatrix. The row index must be smaller than the number of rows and the column index must be
// smaller than the number of columns. Additionally, the column index (in case of a row-major
// matrix) or the row index (in case of a column-major matrix) must be a multiple of the number
// of values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,unaligned,false,true>::stream( size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   if( isAligned_ )
      matrix_.stream( row_+i, column_+j, value );
   else
      matrix_.storeu( row_+i, column_+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline DisableIf_< typename Submatrix<MT,unaligned,false,true>::BLAZE_TEMPLATE VectorizedAssign<MT2> >
   Submatrix<MT,unaligned,false,true>::assign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( n_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<m_; ++i ) {
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         matrix_(row_+i,column_+j    ) = (~rhs)(i,j    );
         matrix_(row_+i,column_+j+1UL) = (~rhs)(i,j+1UL);
      }
      if( jpos < n_ ) {
         matrix_(row_+i,column_+jpos) = (~rhs)(i,jpos);
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline EnableIf_< typename Submatrix<MT,unaligned,false,true>::BLAZE_TEMPLATE VectorizedAssign<MT2> >
   Submatrix<MT,unaligned,false,true>::assign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( n_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

   if( useStreaming && isAligned_ &&
       m_*n_ > ( cacheSize / ( sizeof(ElementType) * 3UL ) ) &&
       !(~rhs).isAliased( &matrix_ ) )
   {
      for( size_t i=0UL; i<m_; ++i )
      {
         size_t j( 0UL );
         Iterator left( begin(i) );
         ConstIterator_<MT2> right( (~rhs).begin(i) );

         for( ; j<jpos; j+=SIMDSIZE ) {
            left.stream( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; j<n_; ++j ) {
            *left = *right;
         }
      }
   }
   else
   {
      for( size_t i=0UL; i<m_; ++i )
      {
         size_t j( 0UL );
         Iterator left( begin(i) );
         ConstIterator_<MT2> right( (~rhs).begin(i) );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; j<jpos; j+=SIMDSIZE ) {
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; j<n_; ++j ) {
            *left = *right; ++left; ++right;
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void Submatrix<MT,unaligned,false,true>::assign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<m_; ii+=block ) {
      const size_t iend( ( m_<(ii+block) )?( m_ ):( ii+block ) );
      for( size_t jj=0UL; jj<n_; jj+=block ) {
         const size_t jend( ( n_<(jj+block) )?( n_ ):( jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               matrix_(row_+i,column_+j) = (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,unaligned,false,true>::assign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT2> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         matrix_(row_+i,column_+element->index()) = element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,unaligned,false,true>::assign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT2> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         matrix_(row_+element->index(),column_+j) = element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline DisableIf_< typename Submatrix<MT,unaligned,false,true>::BLAZE_TEMPLATE VectorizedAddAssign<MT2> >
   Submatrix<MT,unaligned,false,true>::addAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( n_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<m_; ++i )
   {
      if( IsDiagonal<MT2>::value ) {
         matrix_(row_+i,column_+i) += (~rhs)(i,i);
      }
      else {
         for( size_t j=0UL; j<jpos; j+=2UL ) {
            matrix_(row_+i,column_+j    ) += (~rhs)(i,j    );
            matrix_(row_+i,column_+j+1UL) += (~rhs)(i,j+1UL);
         }
         if( jpos < n_ ) {
            matrix_(row_+i,column_+jpos) += (~rhs)(i,jpos);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the addition assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline EnableIf_< typename Submatrix<MT,unaligned,false,true>::BLAZE_TEMPLATE VectorizedAddAssign<MT2> >
   Submatrix<MT,unaligned,false,true>::addAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
   {
      const size_t jbegin( ( IsUpper<MT2>::value )
                           ?( ( IsStrictlyUpper<MT2>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                           :( 0UL ) );
      const size_t jend  ( ( IsLower<MT2>::value )
                           ?( IsStrictlyLower<MT2>::value ? i : i+1UL )
                           :( n_ ) );
      BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

      const size_t jpos( jend & size_t(-SIMDSIZE) );
      BLAZE_INTERNAL_ASSERT( ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

      size_t j( jbegin );
      Iterator left( begin(i) + jbegin );
      ConstIterator_<MT2> right( (~rhs).begin(i) + jbegin );

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; j<jpos; j+=SIMDSIZE ) {
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; j<jend; ++j ) {
         *left += *right; ++left; ++right;
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void Submatrix<MT,unaligned,false,true>::addAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<m_; ii+=block ) {
      const size_t iend( ( m_<(ii+block) )?( m_ ):( ii+block ) );
      for( size_t jj=0UL; jj<n_; jj+=block ) {
         const size_t jend( ( n_<(jj+block) )?( n_ ):( jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               matrix_(row_+i,column_+j) += (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,unaligned,false,true>::addAssign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT2> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         matrix_(row_+i,column_+element->index()) += element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,unaligned,false,true>::addAssign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT2> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         matrix_(row_+element->index(),column_+j) += element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline DisableIf_< typename Submatrix<MT,unaligned,false,true>::BLAZE_TEMPLATE VectorizedSubAssign<MT2> >
   Submatrix<MT,unaligned,false,true>::subAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( n_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<m_; ++i )
   {
      if( IsDiagonal<MT2>::value ) {
         matrix_(row_+i,column_+i) -= (~rhs)(i,i);
      }
      else {
         for( size_t j=0UL; j<jpos; j+=2UL ) {
            matrix_(row_+i,column_+j    ) -= (~rhs)(i,j    );
            matrix_(row_+i,column_+j+1UL) -= (~rhs)(i,j+1UL);
         }
         if( jpos < n_ ) {
            matrix_(row_+i,column_+jpos) -= (~rhs)(i,jpos);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the subtraction assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline EnableIf_< typename Submatrix<MT,unaligned,false,true>::BLAZE_TEMPLATE VectorizedSubAssign<MT2> >
   Submatrix<MT,unaligned,false,true>::subAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
   {
      const size_t jbegin( ( IsUpper<MT2>::value )
                           ?( ( IsStrictlyUpper<MT2>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                           :( 0UL ) );
      const size_t jend  ( ( IsLower<MT2>::value )
                           ?( IsStrictlyLower<MT2>::value ? i : i+1UL )
                           :( n_ ) );
      BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

      const size_t jpos( jend & size_t(-SIMDSIZE) );
      BLAZE_INTERNAL_ASSERT( ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

      size_t j( jbegin );
      Iterator left( begin(i) + jbegin );
      ConstIterator_<MT2> right( (~rhs).begin(i) + jbegin );

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; j<jpos; j+=SIMDSIZE ) {
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; j<jend; ++j ) {
         *left -= *right; ++left; ++right;
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void Submatrix<MT,unaligned,false,true>::subAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<m_; ii+=block ) {
      const size_t iend( ( m_<(ii+block) )?( m_ ):( ii+block ) );
      for( size_t jj=0UL; jj<n_; jj+=block ) {
         const size_t jend( ( n_<(jj+block) )?( n_ ):( jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               matrix_(row_+i,column_+j) -= (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,unaligned,false,true>::subAssign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT2> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         matrix_(row_+i,column_+element->index()) -= element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,unaligned,false,true>::subAssign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT2> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         matrix_(row_+element->index(),column_+j) -= element->value();
}
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR UNALIGNED COLUMN-MAJOR DENSE SUBMATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Submatrix for unaligned column-major dense submatrices.
// \ingroup submatrix
//
// This specialization of Submatrix adapts the class template to the requirements of unaligned
// column-major dense submatrices.
*/
template< typename MT >  // Type of the dense matrix
class Submatrix<MT,unaligned,true,true>
   : public DenseMatrix< Submatrix<MT,unaligned,true,true>, true >
   , private View
{
 private:
   //**Type definitions****************************************************************************
   //! Composite data type of the dense matrix expression.
   typedef If_< IsExpression<MT>, MT, MT& >  Operand;
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef Submatrix<MT,unaligned,true,true>  This;           //!< Type of this Submatrix instance.
   typedef DenseMatrix<This,true>             BaseType;       //!< Base type of this Submatrix instance.
   typedef SubmatrixTrait_<MT>                ResultType;     //!< Result type for expression template evaluations.
   typedef OppositeType_<ResultType>          OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef TransposeType_<ResultType>         TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<MT>                   ElementType;    //!< Type of the submatrix elements.
   typedef SIMDTrait_<ElementType>            SIMDType;       //!< SIMD type of the submatrix elements.
   typedef ReturnType_<MT>                    ReturnType;     //!< Return type for expression template evaluations
   typedef const Submatrix&                   CompositeType;  //!< Data type for composite expression templates.

   //! Reference to a constant submatrix value.
   typedef ConstReference_<MT>  ConstReference;

   //! Reference to a non-constant submatrix value.
   typedef If_< IsConst<MT>, ConstReference, Reference_<MT> >  Reference;

   //! Pointer to a constant submatrix value.
   typedef const ElementType*  ConstPointer;

   //! Pointer to a non-constant submatrix value.
   typedef If_< Or< IsConst<MT>, Not< HasMutableDataAccess<MT> > >, ConstPointer, ElementType* >  Pointer;
   //**********************************************************************************************

   //**SubmatrixIterator class definition**********************************************************
   /*!\brief Iterator over the elements of the sparse submatrix.
   */
   template< typename IteratorType >  // Type of the dense matrix iterator
   class SubmatrixIterator
   {
    public:
      //**Type definitions*************************************************************************
      //! The iterator category.
      typedef typename std::iterator_traits<IteratorType>::iterator_category  IteratorCategory;

      //! Type of the underlying elements.
      typedef typename std::iterator_traits<IteratorType>::value_type  ValueType;

      //! Pointer return type.
      typedef typename std::iterator_traits<IteratorType>::pointer  PointerType;

      //! Reference return type.
      typedef typename std::iterator_traits<IteratorType>::reference  ReferenceType;

      //! Difference between two iterators.
      typedef typename std::iterator_traits<IteratorType>::difference_type  DifferenceType;

      // STL iterator requirements
      typedef IteratorCategory  iterator_category;  //!< The iterator category.
      typedef ValueType         value_type;         //!< Type of the underlying elements.
      typedef PointerType       pointer;            //!< Pointer return type.
      typedef ReferenceType     reference;          //!< Reference return type.
      typedef DifferenceType    difference_type;    //!< Difference between two iterators.
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Default constructor of the SubmatrixIterator class.
      */
      inline SubmatrixIterator()
         : iterator_ (       )  // Iterator to the current submatrix element
         , isAligned_( false )  // Memory alignment flag
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor of the SubmatrixIterator class.
      //
      // \param iterator Iterator to the initial element.
      // \param finalIterator The final iterator for SIMD operations.
      // \param remainingElements The number of remaining elements beyond the final iterator.
      // \param isMemoryAligned Memory alignment flag.
      */
      inline SubmatrixIterator( IteratorType iterator, bool isMemoryAligned )
         : iterator_ ( iterator        )  // Iterator to the current submatrix element
         , isAligned_( isMemoryAligned )  // Memory alignment flag
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Conversion constructor from different SubmatrixIterator instances.
      //
      // \param it The submatrix iterator to be copied.
      */
      template< typename IteratorType2 >
      inline SubmatrixIterator( const SubmatrixIterator<IteratorType2>& it )
         : iterator_ ( it.base()      )  // Iterator to the current submatrix element
         , isAligned_( it.isAligned() )  // Memory alignment flag
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline SubmatrixIterator& operator+=( size_t inc ) {
         iterator_ += inc;
         return *this;
      }
      //*******************************************************************************************

      //**Subtraction assignment operator**********************************************************
      /*!\brief Subtraction assignment operator.
      //
      // \param dec The decrement of the iterator.
      // \return The decremented iterator.
      */
      inline SubmatrixIterator& operator-=( size_t dec ) {
         iterator_ -= dec;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline SubmatrixIterator& operator++() {
         ++iterator_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const SubmatrixIterator operator++( int ) {
         return SubmatrixIterator( iterator_++, isAligned_ );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline SubmatrixIterator& operator--() {
         --iterator_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const SubmatrixIterator operator--( int ) {
         return SubmatrixIterator( iterator_--, isAligned_ );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReferenceType operator*() const {
         return *iterator_;
      }
      //*******************************************************************************************

      //**Load function****************************************************************************
      /*!\brief Load of a SIMD element of the dense submatrix.
      //
      // \return The loaded SIMD element.
      //
      // This function performs a load of the current SIMD element of the submatrix iterator.
      // This function must \b NOT be called explicitly! It is used internally for the performance
      // optimized evaluation of expression templates. Calling this function explicitly might
      // result in erroneous results and/or in compilation errors.
      */
      inline SIMDType load() const noexcept {
         if( isAligned_ )
            return loada();
         else
            return loadu();
      }
      //*******************************************************************************************

      //**Loada function***************************************************************************
      /*!\brief Aligned load of a SIMD element of the dense submatrix.
      //
      // \return The loaded SIMD element.
      //
      // This function performs an aligned load of the current SIMD element of the submatrix
      // iterator. This function must \b NOT be called explicitly! It is used internally for the
      // performance optimized evaluation of expression templates. Calling this function explicitly
      // might result in erroneous results and/or in compilation errors.
      */
      inline SIMDType loada() const noexcept {
         return iterator_.loada();
      }
      //*******************************************************************************************

      //**Loadu function***************************************************************************
      /*!\brief Unaligned load of a SIMD element of the dense submatrix.
      //
      // \return The loaded SIMD element.
      //
      // This function performs an unaligned load of the current SIMD element of the submatrix
      // iterator. This function must \b NOT be called explicitly! It is used internally for the
      // performance optimized evaluation of expression templates. Calling this function explicitly
      // might result in erroneous results and/or in compilation errors.
      */
      inline SIMDType loadu() const noexcept {
         return iterator_.loadu();
      }
      //*******************************************************************************************

      //**Store function***************************************************************************
      /*!\brief Store of a SIMD element of the dense submatrix.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs a store of the current SIMD element of the submatrix iterator.
      // This function must \b NOT be called explicitly! It is used internally for the performance
      // optimized evaluation of expression templates. Calling this function explicitly might
      // result in erroneous results and/or in compilation errors.
      */
      inline void store( const SIMDType& value ) const {
         storeu( value );
      }
      //*******************************************************************************************

      //**Storea function**************************************************************************
      /*!\brief Aligned store of a SIMD element of the dense submatrix.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs an aligned store of the current SIMD element of the submatrix
      // iterator. This function must \b NOT be called explicitly! It is used internally for the
      // performance optimized evaluation of expression templates. Calling this function explicitly
      // might result in erroneous results and/or in compilation errors.
      */
      inline void storea( const SIMDType& value ) const {
         iterator_.storea( value );
      }
      //*******************************************************************************************

      //**Storeu function**************************************************************************
      /*!\brief Unaligned store of a SIMD element of the dense submatrix.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs an unaligned store of the current SIMD element of the submatrix
      // iterator. This function must \b NOT be called explicitly! It is used internally for the
      // performance optimized evaluation of expression templates. Calling this function explicitly
      // might result in erroneous results and/or in compilation errors.
      */
      inline void storeu( const SIMDType& value ) const {
         if( isAligned_ ) {
            iterator_.storea( value );
         }
         else {
            iterator_.storeu( value );
         }
      }
      //*******************************************************************************************

      //**Stream function**************************************************************************
      /*!\brief Aligned, non-temporal store of a SIMD element of the dense submatrix.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs an aligned, non-temporal store of the current SIMD element of the
      // submatrix iterator. This function must \b NOT be called explicitly! It is used internally
      // for the performance optimized evaluation of expression templates. Calling this function
      // explicitly might result in erroneous results and/or in compilation errors.
      */
      inline void stream( const SIMDType& value ) const {
         iterator_.stream( value );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const SubmatrixIterator& rhs ) const {
         return iterator_ == rhs.iterator_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const SubmatrixIterator& rhs ) const {
         return iterator_ != rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const SubmatrixIterator& rhs ) const {
         return iterator_ < rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const SubmatrixIterator& rhs ) const {
         return iterator_ > rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const SubmatrixIterator& rhs ) const {
         return iterator_ <= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const SubmatrixIterator& rhs ) const {
         return iterator_ >= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const SubmatrixIterator& rhs ) const {
         return iterator_ - rhs.iterator_;
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between a SubmatrixIterator and an integral value.
      //
      // \param it The iterator to be incremented.
      // \param inc The number of elements the iterator is incremented.
      // \return The incremented iterator.
      */
      friend inline const SubmatrixIterator operator+( const SubmatrixIterator& it, size_t inc ) {
         return SubmatrixIterator( it.iterator_ + inc, it.isAligned_ );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between an integral value and a SubmatrixIterator.
      //
      // \param inc The number of elements the iterator is incremented.
      // \param it The iterator to be incremented.
      // \return The incremented iterator.
      */
      friend inline const SubmatrixIterator operator+( size_t inc, const SubmatrixIterator& it ) {
         return SubmatrixIterator( it.iterator_ + inc, it.isAligned_ );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Subtraction between a SubmatrixIterator and an integral value.
      //
      // \param it The iterator to be decremented.
      // \param inc The number of elements the iterator is decremented.
      // \return The decremented iterator.
      */
      friend inline const SubmatrixIterator operator-( const SubmatrixIterator& it, size_t dec ) {
         return SubmatrixIterator( it.iterator_ - dec, it.isAligned_ );
      }
      //*******************************************************************************************

      //**Base function****************************************************************************
      /*!\brief Access to the current position of the submatrix iterator.
      //
      // \return The current position of the submatrix iterator.
      */
      inline IteratorType base() const {
         return iterator_;
      }
      //*******************************************************************************************

      //**IsAligned function***********************************************************************
      /*!\brief Access to the iterator's memory alignment flag.
      //
      // \return \a true in case the iterator is aligned, \a false if it is not.
      */
      inline bool isAligned() const noexcept {
         return isAligned_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType iterator_;   //!< Iterator to the current submatrix element.
      bool         isAligned_;  //!< Memory alignment flag.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   //! Iterator over constant elements.
   typedef SubmatrixIterator< ConstIterator_<MT> >  ConstIterator;

   //! Iterator over non-constant elements.
   typedef If_< IsConst<MT>, ConstIterator, SubmatrixIterator< Iterator_<MT> > >  Iterator;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = MT::simdEnabled };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = MT::smpAssignable };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline Submatrix( Operand matrix, size_t rindex, size_t cindex, size_t m, size_t n );
   // No explicitly declared copy constructor.
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
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Pointer        data  ( size_t j ) noexcept;
   inline ConstPointer   data  ( size_t j ) const noexcept;
   inline Iterator       begin ( size_t j );
   inline ConstIterator  begin ( size_t j ) const;
   inline ConstIterator  cbegin( size_t j ) const;
   inline Iterator       end   ( size_t j );
   inline ConstIterator  end   ( size_t j ) const;
   inline ConstIterator  cend  ( size_t j ) const;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline Submatrix& operator=( const ElementType& rhs );
   inline Submatrix& operator=( initializer_list< initializer_list<ElementType> > list );
   inline Submatrix& operator=( const Submatrix& rhs );

   template< typename MT2, bool SO >
   inline Submatrix& operator=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator+=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator+=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator-=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator-=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline Submatrix& operator*=( const Matrix<MT2,SO>& rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, Submatrix >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, Submatrix >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t     row() const noexcept;
                              inline size_t     rows() const noexcept;
                              inline size_t     column() const noexcept;
                              inline size_t     columns() const noexcept;
                              inline size_t     spacing() const noexcept;
                              inline size_t     capacity() const noexcept;
                              inline size_t     capacity( size_t i ) const noexcept;
                              inline size_t     nonZeros() const;
                              inline size_t     nonZeros( size_t i ) const;
                              inline void       reset();
                              inline void       reset( size_t i );
                              inline Submatrix& transpose();
                              inline Submatrix& ctranspose();
   template< typename Other > inline Submatrix& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT2 >
   struct VectorizedAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<MT2> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT2 >
   struct VectorizedAddAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<MT2> >::value &&
                            HasSIMDAdd< ElementType, ElementType_<MT2> >::value &&
                            !IsDiagonal<MT2>::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT2 >
   struct VectorizedSubAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<MT2> >::value &&
                            HasSIMDSub< ElementType, ElementType_<MT2> >::value &&
                            !IsDiagonal<MT2>::value };
   };
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   enum : size_t { SIMDSIZE = SIMDTrait<ElementType>::size };
   //**********************************************************************************************

 public:
   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other >
   inline bool canAlias( const Other* alias ) const noexcept;

   template< typename MT2, bool AF2, bool SO2 >
   inline bool canAlias( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< typename MT2, bool AF2, bool SO2 >
   inline bool isAliased( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept;

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t i, size_t j ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t i, size_t j, const SIMDType& value ) noexcept;

   template< typename MT2 >
   inline DisableIf_< VectorizedAssign<MT2> > assign( const DenseMatrix<MT2,true>& rhs );

   template< typename MT2 >
   inline EnableIf_< VectorizedAssign<MT2> > assign( const DenseMatrix<MT2,true>& rhs );

   template< typename MT2 > inline void assign( const DenseMatrix<MT2,false>&  rhs );
   template< typename MT2 > inline void assign( const SparseMatrix<MT2,true>&  rhs );
   template< typename MT2 > inline void assign( const SparseMatrix<MT2,false>& rhs );

   template< typename MT2 >
   inline DisableIf_< VectorizedAddAssign<MT2> > addAssign( const DenseMatrix<MT2,true>& rhs );

   template< typename MT2 >
   inline EnableIf_< VectorizedAddAssign<MT2> > addAssign( const DenseMatrix<MT2,true>& rhs );

   template< typename MT2 > inline void addAssign( const DenseMatrix<MT2,false>&  rhs );
   template< typename MT2 > inline void addAssign( const SparseMatrix<MT2,true>&  rhs );
   template< typename MT2 > inline void addAssign( const SparseMatrix<MT2,false>& rhs );

   template< typename MT2 >
   inline DisableIf_< VectorizedSubAssign<MT2> > subAssign( const DenseMatrix<MT2,true>& rhs );

   template< typename MT2 >
   inline EnableIf_< VectorizedSubAssign<MT2> > subAssign( const DenseMatrix<MT2,true>& rhs );

   template< typename MT2 > inline void subAssign( const DenseMatrix<MT2,false>&  rhs );
   template< typename MT2 > inline void subAssign( const SparseMatrix<MT2,true>&  rhs );
   template< typename MT2 > inline void subAssign( const SparseMatrix<MT2,false>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline bool hasOverlap() const noexcept;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand      matrix_;   //!< The dense matrix containing the submatrix.
   const size_t row_;      //!< The first row of the submatrix.
   const size_t column_;   //!< The first column of the submatrix.
   const size_t m_;        //!< The number of rows of the submatrix.
   const size_t n_;        //!< The number of columns of the submatrix.
   const bool isAligned_;  //!< Memory alignment flag.
                           /*!< The alignment flag indicates whether the submatrix is fully aligned
                                with respect to the given element type and the available instruction
                                set. In case the submatrix is fully aligned it is possible to use
                                aligned loads and stores instead of unaligned loads and stores. In
                                order to be aligned, the first element of each row/column must be
                                aligned. */
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename MT2, bool AF2, bool SO2, bool DF2 > friend class Submatrix;

   template< bool AF1, typename MT2, bool AF2, bool SO2, bool DF2 >
   friend const Submatrix<MT2,AF1,SO2,DF2>
      submatrix( const Submatrix<MT2,AF2,SO2,DF2>& sm, size_t row, size_t column, size_t m, size_t n );

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isIntact( const Submatrix<MT2,AF2,SO2,DF2>& sm ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isSame( const Submatrix<MT2,AF2,SO2,DF2>& a, const Matrix<MT2,SO2>& b ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isSame( const Matrix<MT2,SO2>& a, const Submatrix<MT2,AF2,SO2,DF2>& b ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isSame( const Submatrix<MT2,AF2,SO2,DF2>& a, const Submatrix<MT2,AF2,SO2,DF2>& b ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool tryAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                          size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename MT3, bool SO3 >
   friend bool tryAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Matrix<MT3,SO3>& rhs,
                          size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool tryAddAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename MT3, bool SO3 >
   friend bool tryAddAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Matrix<MT3,SO3>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool trySubAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename MT3, bool SO3 >
   friend bool trySubAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Matrix<MT3,SO3>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool tryMultAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                              size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend DerestrictTrait_< Submatrix<MT2,AF2,SO2,DF2> > derestrict( Submatrix<MT2,AF2,SO2,DF2>& sm );
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE       ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE      ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SUBMATRIX_TYPE      ( MT );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE        ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE      ( MT );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for Submatrix.
//
// \param matrix The dense matrix containing the submatrix.
// \param rindex The index of the first row of the submatrix in the given dense matrix.
// \param cindex The index of the first column of the submatrix in the given dense matrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \exception std::invalid_argument Invalid submatrix specification.
//
// In case the submatrix is not properly specified (i.e. if the specified submatrix is not
// contained in the given dense matrix) a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,unaligned,true,true>::Submatrix( Operand matrix, size_t rindex, size_t cindex, size_t m, size_t n )
   : matrix_   ( matrix )  // The dense matrix containing the submatrix
   , row_      ( rindex )  // The first row of the submatrix
   , column_   ( cindex )  // The first column of the submatrix
   , m_        ( m      )  // The number of rows of the submatrix
   , n_        ( n      )  // The number of columns of the submatrix
   , isAligned_( simdEnabled && matrix.data() != nullptr && checkAlignment( data() ) &&
                 ( n < 2UL || ( matrix.spacing() & size_t(-SIMDSIZE) ) == 0UL ) )
{
   if( ( row_ + m_ > matrix_.rows() ) || ( column_ + n_ > matrix_.columns() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid submatrix specification" );
   }
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
/*!\brief 2D-access to the dense submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::Reference
   Submatrix<MT,unaligned,true,true>::operator()( size_t i, size_t j )
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return matrix_(row_+i,column_+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the dense submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::ConstReference
   Submatrix<MT,unaligned,true,true>::operator()( size_t i, size_t j ) const
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return const_cast<const MT&>( matrix_ )(row_+i,column_+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::Reference
   Submatrix<MT,unaligned,true,true>::at( size_t i, size_t j )
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
/*!\brief Checked access to the submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::ConstReference
   Submatrix<MT,unaligned,true,true>::at( size_t i, size_t j ) const
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
/*!\brief Low-level data access to the submatrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense submatrix. Note that
// you can NOT assume that all matrix elements lie adjacent to each other! The dense submatrix
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::Pointer
   Submatrix<MT,unaligned,true,true>::data() noexcept
{
   return matrix_.data() + row_ + column_*spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the submatrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense submatrix. Note that
// you can NOT assume that all matrix elements lie adjacent to each other! The dense submatrix
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::ConstPointer
   Submatrix<MT,unaligned,true,true>::data() const noexcept
{
   return matrix_.data() + row_ + column_*spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the submatrix elements of column \a j.
//
// \param j The column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::Pointer
   Submatrix<MT,unaligned,true,true>::data( size_t j ) noexcept
{
   return matrix_.data() + row_ + (column_+j)*spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the submatrix elements of column \a j.
//
// \param j The column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::ConstPointer
   Submatrix<MT,unaligned,true,true>::data( size_t j ) const noexcept
{
   return matrix_.data() + row_ + (column_+j)*spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator to the first non-zero element of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::Iterator
   Submatrix<MT,unaligned,true,true>::begin( size_t j )
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid dense submatrix column access index" );
   return Iterator( matrix_.begin( column_ + j ) + row_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator to the first non-zero element of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::ConstIterator
   Submatrix<MT,unaligned,true,true>::begin( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid dense submatrix column access index" );
   return ConstIterator( matrix_.cbegin( column_ + j ) + row_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator to the first non-zero element of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::ConstIterator
   Submatrix<MT,unaligned,true,true>::cbegin( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid dense submatrix column access index" );
   return ConstIterator( matrix_.cbegin( column_ + j ) + row_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last non-zero element of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::Iterator
   Submatrix<MT,unaligned,true,true>::end( size_t j )
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid dense submatrix column access index" );
   return Iterator( matrix_.begin( column_ + j ) + row_ + m_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last non-zero element of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::ConstIterator
   Submatrix<MT,unaligned,true,true>::end( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid dense submatrix column access index" );
   return ConstIterator( matrix_.cbegin( column_ + j ) + row_ + m_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last non-zero element of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,unaligned,true,true>::ConstIterator
   Submatrix<MT,unaligned,true,true>::cend( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid dense submatrix column access index" );
   return ConstIterator( matrix_.cbegin( column_ + j ) + row_ + m_, isAligned_ );
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
/*!\brief Homogenous assignment to all submatrix elements.
//
// \param rhs Scalar value to be assigned to all submatrix elements.
// \return Reference to the assigned submatrix.
//
// This function homogeneously assigns the given value to all dense matrix elements. Note that in
// case the underlying dense matrix is a lower/upper matrix only lower/upper and diagonal elements
// of the underlying matrix are modified.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,unaligned,true,true>&
   Submatrix<MT,unaligned,true,true>::operator=( const ElementType& rhs )
{
   const size_t jend( column_ + n_ );

   for( size_t j=column_; j<jend; ++j )
   {
      const size_t ibegin( ( IsLower<MT>::value )
                           ?( ( IsUniLower<MT>::value || IsStrictlyLower<MT>::value )
                              ?( max( j+1UL, row_ ) )
                              :( max( j, row_ ) ) )
                           :( row_ ) );
      const size_t iend  ( ( IsUpper<MT>::value )
                           ?( ( IsUniUpper<MT>::value || IsStrictlyUpper<MT>::value )
                              ?( min( j, row_+m_ ) )
                              :( min( j+1UL, row_+m_ ) ) )
                           :( row_+m_ ) );

      for( size_t i=ibegin; i<iend; ++i )
         matrix_(i,j) = rhs;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all submatrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to submatrix.
//
// This assignment operator offers the option to directly assign to all elements of the submatrix
// by means of an initializer list. The submatrix elements are assigned the values from the given
// initializer list. Missing values are initialized as default. Note that in case the size
// of the top-level initializer list exceeds the number of rows or the size of any nested list
// exceeds the number of columns, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,unaligned,true,true>&
   Submatrix<MT,unaligned,true,true>::operator=( initializer_list< initializer_list<ElementType> > list )
{
   if( list.size() != rows() || determineColumns( list ) > columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to submatrix" );
   }

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      size_t j( 0UL );
      for( const auto& element : rowList ) {
         matrix_(row_+i,column_+j) = element;
         ++j;
      }
      for( ; j<n_; ++j ) {
         matrix_(row_+i,column_+j) = ElementType();
      }
      ++i;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for Submatrix.
//
// \param rhs Sparse submatrix to be copied.
// \return Reference to the assigned submatrix.
// \exception std::invalid_argument Submatrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The dense submatrix is initialized as a copy of the given dense submatrix. In case the current
// sizes of the two submatrices don't match, a \a std::invalid_argument exception is thrown. Also,
// if the underlying matrix \a MT is a lower triangular, upper triangular, or symmetric matrix
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,unaligned,true,true>&
   Submatrix<MT,unaligned,true,true>::operator=( const Submatrix& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( this == &rhs || ( &matrix_ == &rhs.matrix_ && row_ == rhs.row_ && column_ == rhs.column_ ) )
      return *this;

   if( rows() != rhs.rows() || columns() != rhs.columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Submatrix sizes do not match" );
   }

   if( !tryAssign( matrix_, rhs, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( rhs.canAlias( &matrix_ ) ) {
      const ResultType tmp( rhs );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different matrices.
//
// \param rhs Matrix to be assigned.
// \return Reference to the assigned submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The dense submatrix is initialized as a copy of the given matrix. In case the current sizes
// of the two matrices don't match, a \a std::invalid_argument exception is thrown. Also, if
// the underlying matrix \a MT is a lower triangular, upper triangular, or symmetric matrix
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline Submatrix<MT,unaligned,true,true>&
   Submatrix<MT,unaligned,true,true>::operator=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   typedef If_< IsRestricted<MT>, CompositeType_<MT2>, const MT2& >  Right;
   Right right( ~rhs );

   if( !tryAssign( matrix_, right, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   if( IsSparseMatrix<MT2>::value )
      reset();

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &matrix_ ) ) {
      const ResultType_<MT2> tmp( right );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO  >     // Storage order of the right-hand side matrix
inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,unaligned,true,true>& >
   Submatrix<MT,unaligned,true,true>::operator+=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef AddTrait_< ResultType, ResultType_<MT2> >  AddType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( !tryAddAssign( matrix_, ~rhs, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( ( ( IsSymmetric<MT>::value || IsHermitian<MT>::value ) && hasOverlap() ) ||
       (~rhs).canAlias( &matrix_ ) ) {
      const AddType tmp( *this + (~rhs) );
      smpAssign( left, tmp );
   }
   else {
      smpAddAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO  >     // Storage order of the right-hand side matrix
inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,unaligned,true,true>& >
   Submatrix<MT,unaligned,true,true>::operator+=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef AddTrait_< ResultType, ResultType_<MT2> >  AddType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const AddType tmp( *this + (~rhs) );

   if( !tryAssign( matrix_, tmp, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,unaligned,true,true>& >
   Submatrix<MT,unaligned,true,true>::operator-=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef SubTrait_< ResultType, ResultType_<MT2> >  SubType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( !trySubAssign( matrix_, ~rhs, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( ( ( IsSymmetric<MT>::value || IsHermitian<MT>::value ) && hasOverlap() ) ||
       (~rhs).canAlias( &matrix_ ) ) {
      const SubType tmp( *this - (~rhs ) );
      smpAssign( left, tmp );
   }
   else {
      smpSubAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,unaligned,true,true>& >
   Submatrix<MT,unaligned,true,true>::operator-=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef SubTrait_< ResultType, ResultType_<MT2> >  SubType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const SubType tmp( *this - (~rhs) );

   if( !tryAssign( matrix_, tmp, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of a matrix (\f$ A*=B \f$).
//
// \param rhs The right-hand side matrix for the multiplication.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline Submatrix<MT,unaligned,true,true>&
   Submatrix<MT,unaligned,true,true>::operator*=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef MultTrait_< ResultType, ResultType_<MT2> >  MultType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( MultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MultType );

   if( columns() != (~rhs).rows() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const MultType tmp( *this * (~rhs) );

   if( !tryAssign( matrix_, tmp, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication between a dense submatrix
//        and a scalar value (\f$ A*=s \f$).
//
// \param rhs The right-hand side scalar value for the multiplication.
// \return Reference to the dense submatrix.
//
// This operator cannot be used for submatrices on lower or upper unitriangular matrices. The
// attempt to scale such a submatrix results in a compilation error!
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, Submatrix<MT,unaligned,true,true> >&
   Submatrix<MT,unaligned,true,true>::operator*=( Other rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   DerestrictTrait_<This> left( derestrict( *this ) );
   smpAssign( left, (*this) * rhs );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division assignment operator for the division of a dense submatrix by a scalar value
//        (\f$ A/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the dense submatrix.
//
// This operator cannot be used for submatrices on lower or upper unitriangular matrices. The
// attempt to scale such a submatrix results in a compilation error!
//
// \note A division by zero is only checked by an user assert.
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, Submatrix<MT,unaligned,true,true> >&
   Submatrix<MT,unaligned,true,true>::operator/=( Other rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

   DerestrictTrait_<This> left( derestrict( *this ) );
   smpAssign( left, (*this) / rhs );

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
/*!\brief Returns the index of the first row of the submatrix in the underlying dense matrix.
//
// \return The index of the first row.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,true,true>::row() const noexcept
{
   return row_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of rows of the dense submatrix.
//
// \return The number of rows of the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,true,true>::rows() const noexcept
{
   return m_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first column of the submatrix in the underlying dense matrix.
//
// \return The index of the first column.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,true,true>::column() const noexcept
{
   return column_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of columns of the dense submatrix.
//
// \return The number of columns of the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,true,true>::columns() const noexcept
{
   return n_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the spacing between the beginning of two columns.
//
// \return The spacing between the beginning of two columns.
//
// This function returns the spacing between the beginning of two columns, i.e. the total
// number of elements of a column.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,true,true>::spacing() const noexcept
{
   return matrix_.spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense submatrix.
//
// \return The capacity of the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,true,true>::capacity() const noexcept
{
   return rows() * columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current capacity of the specified column.
//
// \param j The index of the column.
// \return The current capacity of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,true,true>::capacity( size_t j ) const noexcept
{
   UNUSED_PARAMETER( j );

   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the dense submatrix
//
// \return The number of non-zero elements in the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,true,true>::nonZeros() const
{
   const size_t iend( row_ + m_ );
   const size_t jend( column_ + n_ );
   size_t nonzeros( 0UL );

   for( size_t j=column_; j<jend; ++j )
      for( size_t i=row_; i<iend; ++i )
         if( !isDefault( matrix_(i,j) ) )
            ++nonzeros;

   return nonzeros;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the specified column.
//
// \param j The index of the column.
// \return The number of non-zero elements of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,unaligned,true,true>::nonZeros( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   const size_t iend( row_ + m_ );
   size_t nonzeros( 0UL );

   for( size_t i=row_; i<iend; ++i )
      if( !isDefault( matrix_(i,column_+j) ) )
         ++nonzeros;

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
template< typename MT >  // Type of the dense matrix
inline void Submatrix<MT,unaligned,true,true>::reset()
{
   using blaze::clear;

   for( size_t j=column_; j<column_+n_; ++j )
   {
      const size_t ibegin( ( IsLower<MT>::value )
                           ?( ( IsUniLower<MT>::value || IsStrictlyLower<MT>::value )
                              ?( max( j+1UL, row_ ) )
                              :( max( j, row_ ) ) )
                           :( row_ ) );
      const size_t iend  ( ( IsUpper<MT>::value )
                           ?( ( IsUniUpper<MT>::value || IsStrictlyUpper<MT>::value )
                              ?( min( j, row_+m_ ) )
                              :( min( j+1UL, row_+m_ ) ) )
                           :( row_+m_ ) );

      for( size_t i=ibegin; i<iend; ++i )
         clear( matrix_(i,j) );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset the specified column to the default initial values.
//
// \param j The index of the column.
// \return void
*/
template< typename MT >  // Type of the dense matrix
inline void Submatrix<MT,unaligned,true,true>::reset( size_t j )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   const size_t ibegin( ( IsLower<MT>::value )
                        ?( ( IsUniLower<MT>::value || IsStrictlyLower<MT>::value )
                           ?( max( j+1UL, row_ ) )
                           :( max( j, row_ ) ) )
                        :( row_ ) );
   const size_t iend  ( ( IsUpper<MT>::value )
                        ?( ( IsUniUpper<MT>::value || IsStrictlyUpper<MT>::value )
                           ?( min( j, row_+m_ ) )
                           :( min( j+1UL, row_+m_ ) ) )
                        :( row_+m_ ) );

   for( size_t i=ibegin; i<iend; ++i )
      clear( matrix_(i,column_+j) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transpose of the submatrix.
//
// \return Reference to the transposed submatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic submatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense submatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the submatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the submatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,unaligned,true,true>& Submatrix<MT,unaligned,true,true>::transpose()
{
   if( m_ != n_ ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic submatrix" );
   }

   if( !tryAssign( matrix_, trans( *this ), row_, column_ ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );
   const ResultType tmp( trans( *this ) );
   smpAssign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place conjugate transpose of the submatrix.
//
// \return Reference to the transposed submatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic submatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense submatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the submatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the submatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,unaligned,true,true>& Submatrix<MT,unaligned,true,true>::ctranspose()
{
   if( m_ != n_ ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic submatrix" );
   }

   if( !tryAssign( matrix_, ctrans( *this ), row_, column_ ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );
   const ResultType tmp( ctrans( *this ) );
   smpAssign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the dense submatrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the submatrix scaling.
// \return Reference to the dense submatrix.
//
// This function scales all elements of the submatrix by the given scalar value \a scalar. Note
// that the function cannot be used to scale a submatrix on a lower or upper unitriangular matrix.
// The attempt to scale such a submatrix results in a compile time error!
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the scalar value
inline Submatrix<MT,unaligned,true,true>& Submatrix<MT,unaligned,true,true>::scale( const Other& scalar )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   const size_t jend( column_ + n_ );

   for( size_t j=column_; j<jend; ++j )
   {
      const size_t ibegin( ( IsLower<MT>::value )
                           ?( ( IsStrictlyLower<MT>::value )
                              ?( max( j+1UL, row_ ) )
                              :( max( j, row_ ) ) )
                           :( row_ ) );
      const size_t iend  ( ( IsUpper<MT>::value )
                           ?( ( IsStrictlyUpper<MT>::value )
                              ?( min( j, row_+m_ ) )
                              :( min( j+1UL, row_+m_ ) ) )
                           :( row_+m_ ) );

      for( size_t i=ibegin; i<iend; ++i )
         matrix_(i,j) *= scalar;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checking whether there exists an overlap in the context of a symmetric matrix.
//
// \return \a true in case an overlap exists, \a false if not.
//
// This function checks if in the context of a symmetric matrix the submatrix has an overlap with
// its counterpart. In case an overlap exists, the function return \a true, otherwise it returns
// \a false.
*/
template< typename MT >  // Type of the dense matrix
inline bool Submatrix<MT,unaligned,true,true>::hasOverlap() const noexcept
{
   BLAZE_INTERNAL_ASSERT( IsSymmetric<MT>::value || IsHermitian<MT>::value, "Invalid matrix detected" );

   if( ( row_ + m_ <= column_ ) || ( column_ + n_ <= row_ ) )
      return false;
   else return true;
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
/*!\brief Returns whether the submatrix can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address can alias with the submatrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the foreign expression
inline bool Submatrix<MT,unaligned,true,true>::canAlias( const Other* alias ) const noexcept
{
   return matrix_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix can alias with the given dense submatrix \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address can alias with the submatrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Data type of the foreign dense submatrix
        , bool AF2       // Alignment flag of the foreign dense submatrix
        , bool SO2 >     // Storage order of the foreign dense submatrix
inline bool Submatrix<MT,unaligned,true,true>::canAlias( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept
{
   return ( matrix_.isAliased( &alias->matrix_ ) &&
            ( row_    + m_ > alias->row_    ) && ( row_    < alias->row_    + alias->m_ ) &&
            ( column_ + n_ > alias->column_ ) && ( column_ < alias->column_ + alias->n_ ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address is aliased with the submatrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the foreign expression
inline bool Submatrix<MT,unaligned,true,true>::isAliased( const Other* alias ) const noexcept
{
   return matrix_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is aliased with the given dense submatrix \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address is aliased with the submatrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Data type of the foreign dense submatrix
        , bool AF2       // Alignment flag of the foreign dense submatrix
        , bool SO2 >     // Storage order of the foreign dense submatrix
inline bool Submatrix<MT,unaligned,true,true>::isAliased( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept
{
   return ( matrix_.isAliased( &alias->matrix_ ) &&
            ( row_    + m_ > alias->row_    ) && ( row_    < alias->row_    + alias->m_ ) &&
            ( column_ + n_ > alias->column_ ) && ( column_ < alias->column_ + alias->n_ ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is properly aligned in memory.
//
// \return \a true in case the submatrix is aligned, \a false if not.
//
// This function returns whether the submatrix is guaranteed to be properly aligned in memory,
// i.e. whether the beginning and the end of each column of the submatrix are guaranteed to
// conform to the alignment restrictions of the underlying element type.
*/
template< typename MT >  // Type of the dense matrix
inline bool Submatrix<MT,unaligned,true,true>::isAligned() const noexcept
{
   return isAligned_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix can be used in SMP assignments.
//
// \return \a true in case the submatrix can be used in SMP assignments, \a false if not.
//
// This function returns whether the submatrix can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the submatrix).
*/
template< typename MT >  // Type of the dense matrix
inline bool Submatrix<MT,unaligned,true,true>::canSMPAssign() const noexcept
{
   return ( columns() > SMP_DMATASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Load of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense submatrix. The row
// index must be smaller than the number of rows and the column index must be smaller than
// the number of columns. Additionally, the row index must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is
// used internally for the performance optimized evaluation of expression templates. Calling
// this function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename Submatrix<MT,unaligned,true,true>::SIMDType
   Submatrix<MT,unaligned,true,true>::load( size_t i, size_t j ) const noexcept
{
   if( isAligned_ )
      return loada( i, j );
   else
      return loadu( i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the row index must be a multiple of the number
// of values inside the SIMD element. This function must \b NOT be called explicitly! It is
// used internally for the performance optimized evaluation of expression templates. Calling
// this function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename Submatrix<MT,unaligned,true,true>::SIMDType
   Submatrix<MT,unaligned,true,true>::loada( size_t i, size_t j ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );

   return matrix_.loada( row_+i, column_+j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned load of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the row index must be a multiple of the number
// of values inside the SIMD element. This function must \b NOT be called explicitly! It is
// used internally for the performance optimized evaluation of expression templates. Calling
// this function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename Submatrix<MT,unaligned,true,true>::SIMDType
   Submatrix<MT,unaligned,true,true>::loadu( size_t i, size_t j ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );

   return matrix_.loadu( row_+i, column_+j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store of a specific SIMD element of the dense submatrix. The
// row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the row index must be a multiple of the number
// of values inside the SIMD element. This function must \b NOT be called explicitly! It is
// used internally for the performance optimized evaluation of expression templates. Calling
// this function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,unaligned,true,true>::store( size_t i, size_t j, const SIMDType& value ) noexcept
{
   if( isAligned_ )
      storea( i, j, value );
   else
      storeu( i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the row index must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,unaligned,true,true>::storea( size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );

   matrix_.storea( row_+i, column_+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the row index must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,unaligned,true,true>::storeu( size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );

   matrix_.storeu( row_+i, column_+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned, non-temporal store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store of a specific SIMD element of the
// dense submatrix. The row index must be smaller than the number of rows and the column
// index must be smaller than the number of columns. Additionally, the row index must be a
// multiple of the number of values inside the SIMD element. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,unaligned,true,true>::stream( size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );

   if( isAligned_ )
      matrix_.stream( row_+i, column_+j, value );
   else
      matrix_.storeu( row_+i, column_+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline DisableIf_< typename Submatrix<MT,unaligned,true,true>::BLAZE_TEMPLATE VectorizedAssign<MT2> >
   Submatrix<MT,unaligned,true,true>::assign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t ipos( m_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( m_ - ( m_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t j=0UL; j<n_; ++j ) {
      for( size_t i=0UL; i<ipos; i+=2UL ) {
         matrix_(row_+i    ,column_+j) = (~rhs)(i    ,j);
         matrix_(row_+i+1UL,column_+j) = (~rhs)(i+1UL,j);
      }
      if( ipos < m_ ) {
         matrix_(row_+ipos,column_+j) = (~rhs)(ipos,j);
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline EnableIf_< typename Submatrix<MT,unaligned,true,true>::BLAZE_TEMPLATE VectorizedAssign<MT2> >
   Submatrix<MT,unaligned,true,true>::assign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t ipos( m_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( m_ - ( m_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   if( useStreaming && isAligned_ &&
       m_*n_ > ( cacheSize / ( sizeof(ElementType) * 3UL ) ) &&
       !(~rhs).isAliased( &matrix_ ) )
   {
      for( size_t j=0UL; j<n_; ++j )
      {
         size_t i( 0UL );
         Iterator left( begin(j) );
         ConstIterator_<MT2> right( (~rhs).begin(j) );

         for( ; i<ipos; i+=SIMDSIZE ) {
            left.stream( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; i<m_; ++i ) {
            *left = *right; ++left; ++right;
         }
      }
   }
   else
   {
      for( size_t j=0UL; j<n_; ++j )
      {
         size_t i( 0UL );
         Iterator left( begin(j) );
         ConstIterator_<MT2> right( (~rhs).begin(j) );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; i<ipos; i+=SIMDSIZE ) {
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; i<m_; ++i ) {
            *left = *right; ++left; ++right;
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void Submatrix<MT,unaligned,true,true>::assign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t jj=0UL; jj<n_; jj+=block ) {
      const size_t jend( ( n_<(jj+block) )?( n_ ):( jj+block ) );
      for( size_t ii=0UL; ii<m_; ii+=block ) {
         const size_t iend( ( m_<(ii+block) )?( m_ ):( ii+block ) );
         for( size_t j=jj; j<jend; ++j ) {
            for( size_t i=ii; i<iend; ++i ) {
               matrix_(row_+i,column_+j) = (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,unaligned,true,true>::assign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT2> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         matrix_(row_+element->index(),column_+j) = element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,unaligned,true,true>::assign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT2> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         matrix_(row_+i,column_+element->index()) = element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline DisableIf_< typename Submatrix<MT,unaligned,true,true>::BLAZE_TEMPLATE VectorizedAddAssign<MT2> >
   Submatrix<MT,unaligned,true,true>::addAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t ipos( m_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( m_ - ( m_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t j=0UL; j<n_; ++j )
   {
      if( IsDiagonal<MT2>::value ) {
         matrix_(row_+j,column_+j) += (~rhs)(j,j);
      }
      else {
         for( size_t i=0UL; i<ipos; i+=2UL ) {
            matrix_(row_+i    ,column_+j) += (~rhs)(i    ,j);
            matrix_(row_+i+1UL,column_+j) += (~rhs)(i+1UL,j);
         }
         if( ipos < m_ ) {
            matrix_(row_+ipos,column_+j) += (~rhs)(ipos,j);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the addition assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline EnableIf_< typename Submatrix<MT,unaligned,true,true>::BLAZE_TEMPLATE VectorizedAddAssign<MT2> >
   Submatrix<MT,unaligned,true,true>::addAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
   {
      const size_t ibegin( ( IsLower<MT>::value )
                           ?( ( IsStrictlyLower<MT>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                           :( 0UL ) );
      const size_t iend  ( ( IsUpper<MT>::value )
                           ?( IsStrictlyUpper<MT>::value ? j : j+1UL )
                           :( m_ ) );
      BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

      const size_t ipos( iend & size_t(-SIMDSIZE) );
      BLAZE_INTERNAL_ASSERT( ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

      size_t i( ibegin );
      Iterator left( begin(j) + ibegin );
      ConstIterator_<MT2> right( (~rhs).begin(j) + ibegin );

      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<ipos; i+=SIMDSIZE ) {
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<iend; ++i ) {
         *left += *right; ++left; ++right;
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void Submatrix<MT,unaligned,true,true>::addAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t jj=0UL; jj<n_; jj+=block ) {
      const size_t jend( ( n_<(jj+block) )?( n_ ):( jj+block ) );
      for( size_t ii=0UL; ii<m_; ii+=block ) {
         const size_t iend( ( m_<(ii+block) )?( m_ ):( ii+block ) );
         for( size_t j=jj; j<jend; ++j ) {
            for( size_t i=ii; i<iend; ++i ) {
               matrix_(row_+i,column_+j) += (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,unaligned,true,true>::addAssign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT2> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         matrix_(row_+element->index(),column_+j) += element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,unaligned,true,true>::addAssign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT2> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         matrix_(row_+i,column_+element->index()) += element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline DisableIf_< typename Submatrix<MT,unaligned,true,true>::BLAZE_TEMPLATE VectorizedSubAssign<MT2> >
   Submatrix<MT,unaligned,true,true>::subAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t ipos( m_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( m_ - ( m_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t j=0UL; j<n_; ++j )
   {
      if( IsDiagonal<MT2>::value ) {
         matrix_(row_+j,column_+j) -= (~rhs)(j,j);
      }
      else {
         for( size_t i=0UL; i<ipos; i+=2UL ) {
            matrix_(row_+i    ,column_+j) -= (~rhs)(i    ,j);
            matrix_(row_+i+1UL,column_+j) -= (~rhs)(i+1UL,j);
         }
         if( ipos < m_ ) {
            matrix_(row_+ipos,column_+j) -= (~rhs)(ipos,j);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the subtraction assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline EnableIf_< typename Submatrix<MT,unaligned,true,true>::BLAZE_TEMPLATE VectorizedSubAssign<MT2> >
   Submatrix<MT,unaligned,true,true>::subAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
   {
      const size_t ibegin( ( IsLower<MT>::value )
                           ?( ( IsStrictlyLower<MT>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                           :( 0UL ) );
      const size_t iend  ( ( IsUpper<MT>::value )
                           ?( IsStrictlyUpper<MT>::value ? j : j+1UL )
                           :( m_ ) );
      BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

      const size_t ipos( iend & size_t(-SIMDSIZE) );
      BLAZE_INTERNAL_ASSERT( ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

      size_t i( ibegin );
      Iterator left( begin(j) + ibegin );
      ConstIterator_<MT2> right( (~rhs).begin(j) + ibegin );

      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<ipos; i+=SIMDSIZE ) {
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<iend; ++i ) {
         *left -= *right; ++left; ++right;
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void Submatrix<MT,unaligned,true,true>::subAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t jj=0UL; jj<n_; jj+=block ) {
      const size_t jend( ( n_<(jj+block) )?( n_ ):( jj+block ) );
      for( size_t ii=0UL; ii<m_; ii+=block ) {
         const size_t iend( ( m_<(ii+block) )?( m_ ):( ii+block ) );
         for( size_t j=jj; j<jend; ++j ) {
            for( size_t i=ii; i<iend; ++i ) {
               matrix_(row_+i,column_+j) -= (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,unaligned,true,true>::subAssign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT2> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         matrix_(row_+element->index(),column_+j) -= element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,unaligned,true,true>::subAssign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT2> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         matrix_(row_+i,column_+element->index()) -= element->value();
}
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ALIGNED ROW-MAJOR DENSE SUBMATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Submatrix for aligned row-major dense submatrices.
// \ingroup views
//
// This specialization of Submatrix adapts the class template to the requirements of aligned
// row-major dense submatrices.
*/
template< typename MT >  // Type of the dense matrix
class Submatrix<MT,aligned,false,true>
   : public DenseMatrix< Submatrix<MT,aligned,false,true>, false >
   , private View
{
 private:
   //**Type definitions****************************************************************************
   //! Composite data type of the dense matrix expression.
   typedef If_< IsExpression<MT>, MT, MT& >  Operand;
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef Submatrix<MT,aligned,false,true>  This;           //!< Type of this Submatrix instance.
   typedef DenseMatrix<This,false>           BaseType;       //!< Base type of this Submatrix instance.
   typedef SubmatrixTrait_<MT>               ResultType;     //!< Result type for expression template evaluations.
   typedef OppositeType_<ResultType>         OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef TransposeType_<ResultType>        TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<MT>                  ElementType;    //!< Type of the submatrix elements.
   typedef SIMDTrait_<ElementType>           SIMDType;       //!< SIMD type of the submatrix elements.
   typedef ReturnType_<MT>                   ReturnType;     //!< Return type for expression template evaluations
   typedef const Submatrix&                  CompositeType;  //!< Data type for composite expression templates.

   //! Reference to a constant submatrix value.
   typedef ConstReference_<MT>  ConstReference;

   //! Reference to a non-constant submatrix value.
   typedef If_< IsConst<MT>, ConstReference, Reference_<MT> >  Reference;

   //! Pointer to a constant submatrix value.
   typedef const ElementType*  ConstPointer;

   //! Pointer to a non-constant submatrix value.
   typedef If_< Or< IsConst<MT>, Not< HasMutableDataAccess<MT> > >, ConstPointer, ElementType* >  Pointer;

   //! Iterator over constant elements.
   typedef ConstIterator_<MT>  ConstIterator;

   //! Iterator over non-constant elements.
   typedef If_< IsConst<MT>, ConstIterator, Iterator_<MT> >  Iterator;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = MT::simdEnabled };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = MT::smpAssignable };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline Submatrix( Operand matrix, size_t rindex, size_t cindex, size_t m, size_t n );
   // No explicitly declared copy constructor.
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
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Pointer        data  ( size_t i ) noexcept;
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
   inline Submatrix& operator=( const ElementType& rhs );
   inline Submatrix& operator=( initializer_list< initializer_list<ElementType> > list );
   inline Submatrix& operator=( const Submatrix& rhs );

   template< typename MT2, bool SO >
   inline Submatrix& operator=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator+=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator+=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator-=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator-=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline Submatrix& operator*=( const Matrix<MT2,SO>& rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, Submatrix >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, Submatrix >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t     row() const noexcept;
                              inline size_t     rows() const noexcept;
                              inline size_t     column() const noexcept;
                              inline size_t     columns() const noexcept;
                              inline size_t     spacing() const noexcept;
                              inline size_t     capacity() const noexcept;
                              inline size_t     capacity( size_t i ) const noexcept;
                              inline size_t     nonZeros() const;
                              inline size_t     nonZeros( size_t i ) const;
                              inline void       reset();
                              inline void       reset( size_t i );
                              inline Submatrix& transpose();
                              inline Submatrix& ctranspose();
   template< typename Other > inline Submatrix& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT2 >
   struct VectorizedAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<MT2> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT2 >
   struct VectorizedAddAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<MT2> >::value &&
                            HasSIMDAdd< ElementType, ElementType_<MT2> >::value &&
                            !IsDiagonal<MT2>::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT2 >
   struct VectorizedSubAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<MT2> >::value &&
                            HasSIMDSub< ElementType, ElementType_<MT2> >::value &&
                            !IsDiagonal<MT2>::value };
   };
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   enum : size_t { SIMDSIZE = SIMDTrait<ElementType>::size };
   //**********************************************************************************************

 public:
   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other >
   inline bool canAlias( const Other* alias ) const noexcept;

   template< typename MT2, bool AF2, bool SO2 >
   inline bool canAlias( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< typename MT2, bool AF2, bool SO2 >
   inline bool isAliased( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept;

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t i, size_t j ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t i, size_t j, const SIMDType& value ) noexcept;

   template< typename MT2 >
   inline DisableIf_< VectorizedAssign<MT2> > assign( const DenseMatrix<MT2,false>& rhs );

   template< typename MT2 >
   inline EnableIf_< VectorizedAssign<MT2> > assign( const DenseMatrix<MT2,false>& rhs );

   template< typename MT2 > inline void assign( const DenseMatrix<MT2,true>&  rhs );
   template< typename MT2 > inline void assign( const SparseMatrix<MT2,false>&  rhs );
   template< typename MT2 > inline void assign( const SparseMatrix<MT2,true>& rhs );

   template< typename MT2 >
   inline DisableIf_< VectorizedAddAssign<MT2> > addAssign( const DenseMatrix<MT2,false>& rhs );

   template< typename MT2 >
   inline EnableIf_< VectorizedAddAssign<MT2> > addAssign( const DenseMatrix<MT2,false>& rhs );

   template< typename MT2 > inline void addAssign( const DenseMatrix<MT2,true>&  rhs );
   template< typename MT2 > inline void addAssign( const SparseMatrix<MT2,false>&  rhs );
   template< typename MT2 > inline void addAssign( const SparseMatrix<MT2,true>& rhs );

   template< typename MT2 >
   inline DisableIf_< VectorizedSubAssign<MT2> > subAssign( const DenseMatrix<MT2,false>& rhs );

   template< typename MT2 >
   inline EnableIf_< VectorizedSubAssign<MT2> > subAssign( const DenseMatrix<MT2,false>& rhs );

   template< typename MT2 > inline void subAssign( const DenseMatrix<MT2,true>&  rhs );
   template< typename MT2 > inline void subAssign( const SparseMatrix<MT2,false>&  rhs );
   template< typename MT2 > inline void subAssign( const SparseMatrix<MT2,true>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline bool hasOverlap() const noexcept;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand      matrix_;  //!< The dense matrix containing the submatrix.
   const size_t row_;     //!< The first row of the submatrix.
   const size_t column_;  //!< The first column of the submatrix.
   const size_t m_;       //!< The number of rows of the submatrix.
   const size_t n_;       //!< The number of columns of the submatrix.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename MT2, bool AF2, bool SO2, bool DF2 > friend class Submatrix;

   template< bool AF1, typename MT2, bool AF2, bool SO2, bool DF2 >
   friend const Submatrix<MT2,AF1,SO2,DF2>
      submatrix( const Submatrix<MT2,AF2,SO2,DF2>& sm, size_t row, size_t column, size_t m, size_t n );

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isIntact( const Submatrix<MT2,AF2,SO2,DF2>& sm ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isSame( const Submatrix<MT2,AF2,SO2,DF2>& a, const Matrix<MT2,SO2>& b ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isSame( const Matrix<MT2,SO2>& a, const Submatrix<MT2,AF2,SO2,DF2>& b ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isSame( const Submatrix<MT2,AF2,SO2,DF2>& a, const Submatrix<MT2,AF2,SO2,DF2>& b ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool tryAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                          size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename MT3, bool SO3 >
   friend bool tryAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Matrix<MT3,SO3>& rhs,
                          size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool tryAddAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename MT3, bool SO3 >
   friend bool tryAddAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Matrix<MT3,SO3>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool trySubAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename MT3, bool SO3 >
   friend bool trySubAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Matrix<MT3,SO3>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool tryMultAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                              size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend DerestrictTrait_< Submatrix<MT2,AF2,SO2,DF2> > derestrict( Submatrix<MT2,AF2,SO2,DF2>& sm );
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE   ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SUBMATRIX_TYPE   ( MT );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE     ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE   ( MT );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for Submatrix.
//
// \param matrix The dense matrix containing the submatrix.
// \param rindex The index of the first row of the submatrix in the given dense matrix.
// \param cindex The index of the first column of the submatrix in the given dense matrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \exception std::invalid_argument Invalid submatrix specification.
//
// In case the submatrix is not properly specified (i.e. if the specified submatrix is not
// contained in the given dense matrix) a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,aligned,false,true>::Submatrix( Operand matrix, size_t rindex, size_t cindex, size_t m, size_t n )
   : matrix_( matrix )  // The dense matrix containing the submatrix
   , row_   ( rindex    )  // The first row of the submatrix
   , column_( cindex )  // The first column of the submatrix
   , m_     ( m      )  // The number of rows of the submatrix
   , n_     ( n      )  // The number of columns of the submatrix
{
   if( ( row_ + m_ > matrix_.rows() ) || ( column_ + n_ > matrix_.columns() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid submatrix specification" );
   }

   if( ( simdEnabled && matrix_.data() != nullptr && !checkAlignment( data() ) ) ||
       ( m_ > 1UL && matrix_.spacing() % SIMDSIZE != 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid submatrix alignment" );
   }
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
/*!\brief 2D-access to the dense submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::Reference
   Submatrix<MT,aligned,false,true>::operator()( size_t i, size_t j )
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return matrix_(row_+i,column_+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the dense submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::ConstReference
   Submatrix<MT,aligned,false,true>::operator()( size_t i, size_t j ) const
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return const_cast<const MT&>( matrix_ )(row_+i,column_+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::Reference
   Submatrix<MT,aligned,false,true>::at( size_t i, size_t j )
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
/*!\brief Checked access to the submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::ConstReference
   Submatrix<MT,aligned,false,true>::at( size_t i, size_t j ) const
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
/*!\brief Low-level data access to the submatrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense submatrix. Note that
// you can NOT assume that all matrix elements lie adjacent to each other! The dense submatrix
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::Pointer
   Submatrix<MT,aligned,false,true>::data() noexcept
{
   return matrix_.data() + row_*spacing() + column_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the submatrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense submatrix. Note that
// you can NOT assume that all matrix elements lie adjacent to each other! The dense submatrix
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::ConstPointer
   Submatrix<MT,aligned,false,true>::data() const noexcept
{
   return matrix_.data() + row_*spacing() + column_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the submatrix elements of row \a i.
//
// \param i The row index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense submatrix in row \a i.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::Pointer
   Submatrix<MT,aligned,false,true>::data( size_t i ) noexcept
{
   return matrix_.data() + (row_+i)*spacing() + column_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the submatrix elements of row \a i.
//
// \param i The row index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense submatrix in row \a i.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::ConstPointer
   Submatrix<MT,aligned,false,true>::data( size_t i ) const noexcept
{
   return matrix_.data() + (row_+i)*spacing() + column_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::Iterator
   Submatrix<MT,aligned,false,true>::begin( size_t i )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense submatrix row access index" );
   return ( matrix_.begin( row_ + i ) + column_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::ConstIterator
   Submatrix<MT,aligned,false,true>::begin( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense submatrix row access index" );
   return ( matrix_.cbegin( row_ + i ) + column_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::ConstIterator
   Submatrix<MT,aligned,false,true>::cbegin( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense submatrix row access index" );
   return ( matrix_.cbegin( row_ + i ) + column_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::Iterator
   Submatrix<MT,aligned,false,true>::end( size_t i )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense submatrix row access index" );
   return ( matrix_.begin( row_ + i ) + column_ + n_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::ConstIterator
   Submatrix<MT,aligned,false,true>::end( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense submatrix row access index" );
   return ( matrix_.cbegin( row_ + i ) + column_ + n_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,false,true>::ConstIterator
   Submatrix<MT,aligned,false,true>::cend( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense submatrix row access index" );
   return ( matrix_.cbegin( row_ + i ) + column_ + n_ );
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
/*!\brief Homogenous assignment to all submatrix elements.
//
// \param rhs Scalar value to be assigned to all submatrix elements.
// \return Reference to the assigned submatrix.
//
// This function homogeneously assigns the given value to all dense matrix elements. Note that in
// case the underlying dense matrix is a lower/upper matrix only lower/upper and diagonal elements
// of the underlying matrix are modified.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,aligned,false,true>&
   Submatrix<MT,aligned,false,true>::operator=( const ElementType& rhs )
{
   const size_t iend( row_ + m_ );

   for( size_t i=row_; i<iend; ++i )
   {
      const size_t jbegin( ( IsUpper<MT>::value )
                           ?( ( IsUniUpper<MT>::value || IsStrictlyUpper<MT>::value )
                              ?( max( i+1UL, column_ ) )
                              :( max( i, column_ ) ) )
                           :( column_ ) );
      const size_t jend  ( ( IsLower<MT>::value )
                           ?( ( IsUniLower<MT>::value || IsStrictlyLower<MT>::value )
                              ?( min( i, column_+n_ ) )
                              :( min( i+1UL, column_+n_ ) ) )
                           :( column_+n_ ) );

      for( size_t j=jbegin; j<jend; ++j )
         matrix_(i,j) = rhs;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all submatrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to submatrix.
//
// This assignment operator offers the option to directly assign to all elements of the submatrix
// by means of an initializer list. The submatrix elements are assigned the values from the given
// initializer list. Missing values are initialized as default. Note that in case the size
// of the top-level initializer list exceeds the number of rows or the size of any nested list
// exceeds the number of columns, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,aligned,false,true>&
   Submatrix<MT,aligned,false,true>::operator=( initializer_list< initializer_list<ElementType> > list )
{
   if( list.size() != rows() || determineColumns( list ) > columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to submatrix" );
   }

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      std::fill( std::copy( rowList.begin(), rowList.end(), begin(i) ), end(i), ElementType() );
      ++i;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for Submatrix.
//
// \param rhs Sparse submatrix to be copied.
// \return Reference to the assigned submatrix.
// \exception std::invalid_argument Submatrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The dense submatrix is initialized as a copy of the given dense submatrix. In case the current
// sizes of the two submatrices don't match, a \a std::invalid_argument exception is thrown. Also,
// if the underlying matrix \a MT is a lower triangular, upper triangular, or symmetric matrix
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,aligned,false,true>&
   Submatrix<MT,aligned,false,true>::operator=( const Submatrix& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( this == &rhs || ( &matrix_ == &rhs.matrix_ && row_ == rhs.row_ && column_ == rhs.column_ ) )
      return *this;

   if( rows() != rhs.rows() || columns() != rhs.columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Submatrix sizes do not match" );
   }

   if( !tryAssign( matrix_, rhs, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( rhs.canAlias( &matrix_ ) ) {
      const ResultType tmp( rhs );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different matrices.
//
// \param rhs Matrix to be assigned.
// \return Reference to the assigned submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The dense submatrix is initialized as a copy of the given matrix. In case the current sizes
// of the two matrices don't match, a \a std::invalid_argument exception is thrown. Also, if
// the underlying matrix \a MT is a lower triangular, upper triangular, or symmetric matrix
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline Submatrix<MT,aligned,false,true>&
   Submatrix<MT,aligned,false,true>::operator=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   typedef If_< IsRestricted<MT>, CompositeType_<MT2>, const MT2& >  Right;
   Right right( ~rhs );

   if( !tryAssign( matrix_, right, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   if( IsSparseMatrix<MT2>::value )
      reset();

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &matrix_ ) ) {
      const ResultType_<MT2> tmp( right );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,aligned,false,true>& >
   Submatrix<MT,aligned,false,true>::operator+=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef AddTrait_< ResultType, ResultType_<MT2> >  AddType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( !tryAddAssign( matrix_, ~rhs, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( ( ( IsSymmetric<MT>::value || IsHermitian<MT>::value ) && hasOverlap() ) ||
       (~rhs).canAlias( &matrix_ ) ) {
      const AddType tmp( *this + (~rhs) );
      smpAssign( left, tmp );
   }
   else {
      smpAddAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,aligned,false,true>& >
   Submatrix<MT,aligned,false,true>::operator+=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef AddTrait_< ResultType, ResultType_<MT2> >  AddType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const AddType tmp( *this + (~rhs) );

   if( !tryAssign( matrix_, tmp, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,aligned,false,true>& >
   Submatrix<MT,aligned,false,true>::operator-=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef SubTrait_< ResultType, ResultType_<MT2> >  SubType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( !trySubAssign( matrix_, ~rhs, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( ( ( IsSymmetric<MT>::value || IsHermitian<MT>::value ) && hasOverlap() ) ||
       (~rhs).canAlias( &matrix_ ) ) {
      const SubType tmp( *this - (~rhs ) );
      smpAssign( left, tmp );
   }
   else {
      smpSubAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,aligned,false,true>& >
   Submatrix<MT,aligned,false,true>::operator-=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef SubTrait_< ResultType, ResultType_<MT2> >  SubType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const SubType tmp( *this - (~rhs) );

   if( !tryAssign( matrix_, tmp, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of a matrix (\f$ A*=B \f$).
//
// \param rhs The right-hand side matrix for the multiplication.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline Submatrix<MT,aligned,false,true>&
   Submatrix<MT,aligned,false,true>::operator*=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef MultTrait_< ResultType, ResultType_<MT2> >  MultType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( MultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MultType );

   if( columns() != (~rhs).rows() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const MultType tmp( *this * (~rhs) );

   if( !tryAssign( matrix_, tmp, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication between a dense submatrix
//        and a scalar value (\f$ A*=s \f$).
//
// \param rhs The right-hand side scalar value for the multiplication.
// \return Reference to the dense submatrix.
//
// This operator cannot be used for submatrices on lower or upper unitriangular matrices. The
// attempt to scale such a submatrix results in a compilation error!
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, Submatrix<MT,aligned,false,true> >&
   Submatrix<MT,aligned,false,true>::operator*=( Other rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   DerestrictTrait_<This> left( derestrict( *this ) );
   smpAssign( left, (*this) * rhs );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division assignment operator for the division of a dense submatrix by a scalar value
//        (\f$ A/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the dense submatrix.
//
// This operator cannot be used for submatrices on lower or upper unitriangular matrices. The
// attempt to scale such a submatrix results in a compilation error!
//
// \note A division by zero is only checked by an user assert.
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, Submatrix<MT,aligned,false,true> >&
   Submatrix<MT,aligned,false,true>::operator/=( Other rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

   DerestrictTrait_<This> left( derestrict( *this ) );
   smpAssign( left, (*this) / rhs );

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
/*!\brief Returns the index of the first row of the submatrix in the underlying dense matrix.
//
// \return The index of the first row.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,false,true>::row() const noexcept
{
   return row_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of rows of the dense submatrix.
//
// \return The number of rows of the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,false,true>::rows() const noexcept
{
   return m_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first column of the submatrix in the underlying dense matrix.
//
// \return The index of the first column.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,false,true>::column() const noexcept
{
   return column_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of columns of the dense submatrix.
//
// \return The number of columns of the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,false,true>::columns() const noexcept
{
   return n_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the spacing between the beginning of two rows/columns.
//
// \return The spacing between the beginning of two rows/columns.
//
// This function returns the spacing between the beginning of two rows/columns, i.e. the
// total number of elements of a row/column. In case the storage order is set to \a rowMajor
// the function returns the spacing between two rows, in case the storage flag is set to
// \a columnMajor the function returns the spacing between two columns.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,false,true>::spacing() const noexcept
{
   return matrix_.spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense submatrix.
//
// \return The capacity of the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,false,true>::capacity() const noexcept
{
   return rows() * columns();
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
// This function returns the current capacity of the specified row/column. In case the
// storage order is set to \a rowMajor the function returns the capacity of row \a i,
// in case the storage flag is set to \a columnMajor the function returns the capacity
// of column \a i.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,false,true>::capacity( size_t i ) const noexcept
{
   UNUSED_PARAMETER( i );

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   return columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the dense submatrix
//
// \return The number of non-zero elements in the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,false,true>::nonZeros() const
{
   const size_t iend( row_ + m_ );
   const size_t jend( column_ + n_ );
   size_t nonzeros( 0UL );

   for( size_t i=row_; i<iend; ++i )
      for( size_t j=column_; j<jend; ++j )
         if( !isDefault( matrix_(i,j) ) )
            ++nonzeros;

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
// This function returns the current number of non-zero elements in the specified row/column.
// In case the storage order is set to \a rowMajor the function returns the number of non-zero
// elements in row \a i, in case the storage flag is set to \a columnMajor the function returns
// the number of non-zero elements in column \a i.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,false,true>::nonZeros( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   const size_t jend( column_ + n_ );
   size_t nonzeros( 0UL );

   for( size_t j=column_; j<jend; ++j )
      if( !isDefault( matrix_(row_+i,j) ) )
         ++nonzeros;

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
template< typename MT >  // Type of the dense matrix
inline void Submatrix<MT,aligned,false,true>::reset()
{
   using blaze::clear;

   for( size_t i=row_; i<row_+m_; ++i )
   {
      const size_t jbegin( ( IsUpper<MT>::value )
                           ?( ( IsUniUpper<MT>::value || IsStrictlyUpper<MT>::value )
                              ?( max( i+1UL, column_ ) )
                              :( max( i, column_ ) ) )
                           :( column_ ) );
      const size_t jend  ( ( IsLower<MT>::value )
                           ?( ( IsUniLower<MT>::value || IsStrictlyLower<MT>::value )
                              ?( min( i, column_+n_ ) )
                              :( min( i+1UL, column_+n_ ) ) )
                           :( column_+n_ ) );

      for( size_t j=jbegin; j<jend; ++j )
         clear( matrix_(i,j) );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
template< typename MT >  // Type of the dense matrix
inline void Submatrix<MT,aligned,false,true>::reset( size_t i )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   const size_t jbegin( ( IsUpper<MT>::value )
                        ?( ( IsUniUpper<MT>::value || IsStrictlyUpper<MT>::value )
                           ?( max( i+1UL, column_ ) )
                           :( max( i, column_ ) ) )
                        :( column_ ) );
   const size_t jend  ( ( IsLower<MT>::value )
                        ?( ( IsUniLower<MT>::value || IsStrictlyLower<MT>::value )
                           ?( min( i, column_+n_ ) )
                           :( min( i+1UL, column_+n_ ) ) )
                        :( column_+n_ ) );

   for( size_t j=jbegin; j<jend; ++j )
      clear( matrix_(row_+i,j) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transpose of the submatrix.
//
// \return Reference to the transposed submatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic submatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense submatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the submatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the submatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,aligned,false,true>& Submatrix<MT,aligned,false,true>::transpose()
{
   if( m_ != n_ ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic submatrix" );
   }

   if( !tryAssign( matrix_, trans( *this ), row_, column_ ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );
   const ResultType tmp( trans( *this ) );
   smpAssign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place conjugate transpose of the submatrix.
//
// \return Reference to the transposed submatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic submatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense submatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the submatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the submatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,aligned,false,true>& Submatrix<MT,aligned,false,true>::ctranspose()
{
   if( m_ != n_ ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic submatrix" );
   }

   if( !tryAssign( matrix_, ctrans( *this ), row_, column_ ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );
   const ResultType tmp( ctrans( *this ) );
   smpAssign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the dense submatrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the submatrix scaling.
// \return Reference to the dense submatrix.
//
// This function scales all elements of the submatrix by the given scalar value \a scalar. Note
// that the function cannot be used to scale a submatrix on a lower or upper unitriangular matrix.
// The attempt to scale such a submatrix results in a compile time error!
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the scalar value
inline Submatrix<MT,aligned,false,true>& Submatrix<MT,aligned,false,true>::scale( const Other& scalar )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   const size_t iend( row_ + m_ );

   for( size_t i=row_; i<iend; ++i )
   {
      const size_t jbegin( ( IsUpper<MT>::value )
                           ?( ( IsStrictlyUpper<MT>::value )
                              ?( max( i+1UL, column_ ) )
                              :( max( i, column_ ) ) )
                           :( column_ ) );
      const size_t jend  ( ( IsLower<MT>::value )
                           ?( ( IsStrictlyLower<MT>::value )
                              ?( min( i, column_+n_ ) )
                              :( min( i+1UL, column_+n_ ) ) )
                           :( column_+n_ ) );

      for( size_t j=jbegin; j<jend; ++j )
         matrix_(i,j) *= scalar;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checking whether there exists an overlap in the context of a symmetric matrix.
//
// \return \a true in case an overlap exists, \a false if not.
//
// This function checks if in the context of a symmetric matrix the submatrix has an overlap with
// its counterpart. In case an overlap exists, the function return \a true, otherwise it returns
// \a false.
*/
template< typename MT >  // Type of the dense matrix
inline bool Submatrix<MT,aligned,false,true>::hasOverlap() const noexcept
{
   BLAZE_INTERNAL_ASSERT( IsSymmetric<MT>::value || IsHermitian<MT>::value, "Invalid matrix detected" );

   if( ( row_ + m_ <= column_ ) || ( column_ + n_ <= row_ ) )
      return false;
   else return true;
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
/*!\brief Returns whether the submatrix can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address can alias with the submatrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the foreign expression
inline bool Submatrix<MT,aligned,false,true>::canAlias( const Other* alias ) const noexcept
{
   return matrix_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix can alias with the given dense submatrix \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address can alias with the submatrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Data type of the foreign dense submatrix
        , bool AF2       // Alignment flag of the foreign dense submatrix
        , bool SO2 >     // Storage order of the foreign dense submatrix
inline bool Submatrix<MT,aligned,false,true>::canAlias( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept
{
   return ( matrix_.isAliased( &alias->matrix_ ) &&
            ( row_    + m_ > alias->row_    ) && ( row_    < alias->row_    + alias->m_ ) &&
            ( column_ + n_ > alias->column_ ) && ( column_ < alias->column_ + alias->n_ ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address is aliased with the submatrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the foreign expression
inline bool Submatrix<MT,aligned,false,true>::isAliased( const Other* alias ) const noexcept
{
   return matrix_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is aliased with the given dense submatrix \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address is aliased with the submatrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Data type of the foreign dense submatrix
        , bool AF2       // Alignment flag of the foreign dense submatrix
        , bool SO2 >     // Storage order of the foreign dense submatrix
inline bool Submatrix<MT,aligned,false,true>::isAliased( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept
{
   return ( matrix_.isAliased( &alias->matrix_ ) &&
            ( row_    + m_ > alias->row_    ) && ( row_    < alias->row_    + alias->m_ ) &&
            ( column_ + n_ > alias->column_ ) && ( column_ < alias->column_ + alias->n_ ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is properly aligned in memory.
//
// \return \a true in case the submatrix is aligned, \a false if not.
//
// This function returns whether the submatrix is guaranteed to be properly aligned in memory,
// i.e. whether the beginning and the end of each row of the submatrix are guaranteed to conform
// to the alignment restrictions of the underlying element type.
*/
template< typename MT >  // Type of the dense matrix
inline bool Submatrix<MT,aligned,false,true>::isAligned() const noexcept
{
   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix can be used in SMP assignments.
//
// \return \a true in case the submatrix can be used in SMP assignments, \a false if not.
//
// This function returns whether the submatrix can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the submatrix).
*/
template< typename MT >  // Type of the dense matrix
inline bool Submatrix<MT,aligned,false,true>::canSMPAssign() const noexcept
{
   return ( rows() > SMP_DMATASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Load of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense submatrix. The row
// index must be smaller than the number of rows and the column index must be smaller than
// the number of columns. Additionally, the column index (in case of a row-major matrix) or
// the row index (in case of a column-major matrix) must be a multiple of the number of values
// inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename Submatrix<MT,aligned,false,true>::SIMDType
   Submatrix<MT,aligned,false,true>::load( size_t i, size_t j ) const noexcept
{
   return loada( i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major matrix)
// or the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename Submatrix<MT,aligned,false,true>::SIMDType
   Submatrix<MT,aligned,false,true>::loada( size_t i, size_t j ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   return matrix_.loada( row_+i, column_+j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned load of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major matrix)
// or the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename Submatrix<MT,aligned,false,true>::SIMDType
   Submatrix<MT,aligned,false,true>::loadu( size_t i, size_t j ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   return matrix_.loadu( row_+i, column_+j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store of a specific SIMD element of the dense submatrix. The
// row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major matrix)
// or the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,aligned,false,true>::store( size_t i, size_t j, const SIMDType& value ) noexcept
{
   return storea( i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller than
// the number of columns. Additionally, the column index (in case of a row-major matrix) or the
// row index (in case of a column-major matrix) must be a multiple of the number of values inside
// the SIMD element. This function must \b NOT be called explicitly! It is used internally for
// the performance optimized evaluation of expression templates. Calling this function explicitly
// might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,aligned,false,true>::storea( size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   return matrix_.storea( row_+i, column_+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store of a specific SIMD element of the dense
// submatrix. The row index must be smaller than the number of rows and the column index must
// be smaller than the number of columns. Additionally, the column index (in case of a row-major
// matrix) or the row index (in case of a column-major matrix) must be a multiple of the number
// of values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,aligned,false,true>::storeu( size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   matrix_.storeu( row_+i, column_+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned, non-temporal store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store of a specific SIMD element of the
// dense submatrix. The row index must be smaller than the number of rows and the column index
// must be smaller than the number of columns. Additionally, the column index (in case of a
// row-major matrix) or the row index (in case of a column-major matrix) must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,aligned,false,true>::stream( size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   matrix_.stream( row_+i, column_+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline DisableIf_< typename Submatrix<MT,aligned,false,true>::BLAZE_TEMPLATE VectorizedAssign<MT2> >
   Submatrix<MT,aligned,false,true>::assign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( n_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<m_; ++i ) {
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         matrix_(row_+i,column_+j    ) = (~rhs)(i,j    );
         matrix_(row_+i,column_+j+1UL) = (~rhs)(i,j+1UL);
      }
      if( jpos < n_ ) {
         matrix_(row_+i,column_+jpos) = (~rhs)(i,jpos);
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline EnableIf_< typename Submatrix<MT,aligned,false,true>::BLAZE_TEMPLATE VectorizedAssign<MT2> >
   Submatrix<MT,aligned,false,true>::assign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( n_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

   if( useStreaming &&
       m_*n_ > ( cacheSize / ( sizeof(ElementType) * 3UL ) ) &&
       !(~rhs).isAliased( &matrix_ ) )
   {
      for( size_t i=0UL; i<m_; ++i )
      {
         size_t j( 0UL );
         Iterator left( begin(i) );
         ConstIterator_<MT2> right( (~rhs).begin(i) );

         for( ; j<jpos; j+=SIMDSIZE ) {
            left.stream( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; j<n_; ++j ) {
            *left = *right; ++left; ++right;
         }
      }
   }
   else
   {
      for( size_t i=0UL; i<m_; ++i )
      {
         size_t j( 0UL );
         Iterator left( begin(i) );
         ConstIterator_<MT2> right( (~rhs).begin(i) );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; j<jpos; j+=SIMDSIZE ) {
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; j<n_; ++j ) {
            *left = *right; ++left; ++right;
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void Submatrix<MT,aligned,false,true>::assign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<m_; ii+=block ) {
      const size_t iend( ( m_<(ii+block) )?( m_ ):( ii+block ) );
      for( size_t jj=0UL; jj<n_; jj+=block ) {
         const size_t jend( ( n_<(jj+block) )?( n_ ):( jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               matrix_(row_+i,column_+j) = (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,aligned,false,true>::assign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT2> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         matrix_(row_+i,column_+element->index()) = element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,aligned,false,true>::assign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT2> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         matrix_(row_+element->index(),column_+j) = element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline DisableIf_< typename Submatrix<MT,aligned,false,true>::BLAZE_TEMPLATE VectorizedAddAssign<MT2> >
   Submatrix<MT,aligned,false,true>::addAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( n_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<m_; ++i )
   {
      if( IsDiagonal<MT2>::value ) {
         matrix_(row_+i,column_+i) += (~rhs)(i,i);
      }
      else {
         for( size_t j=0UL; j<jpos; j+=2UL ) {
            matrix_(row_+i,column_+j    ) += (~rhs)(i,j    );
            matrix_(row_+i,column_+j+1UL) += (~rhs)(i,j+1UL);
         }
         if( jpos < n_ ) {
            matrix_(row_+i,column_+jpos) += (~rhs)(i,jpos);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the addition assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline EnableIf_< typename Submatrix<MT,aligned,false,true>::BLAZE_TEMPLATE VectorizedAddAssign<MT2> >
   Submatrix<MT,aligned,false,true>::addAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
   {
      const size_t jbegin( ( IsUpper<MT2>::value )
                           ?( ( IsStrictlyUpper<MT2>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                           :( 0UL ) );
      const size_t jend  ( ( IsLower<MT2>::value )
                           ?( IsStrictlyLower<MT2>::value ? i : i+1UL )
                           :( n_ ) );
      BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

      const size_t jpos( jend & size_t(-SIMDSIZE) );
      BLAZE_INTERNAL_ASSERT( ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

      size_t j( jbegin );
      Iterator left( begin(i) + jbegin );
      ConstIterator_<MT2> right( (~rhs).begin(i) + jbegin );

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; j<jpos; j+=SIMDSIZE ) {
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; j<jend; ++j ) {
         *left += *right; ++left; ++right;
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void Submatrix<MT,aligned,false,true>::addAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<m_; ii+=block ) {
      const size_t iend( ( m_<(ii+block) )?( m_ ):( ii+block ) );
      for( size_t jj=0UL; jj<n_; jj+=block ) {
         const size_t jend( ( n_<(jj+block) )?( n_ ):( jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               matrix_(row_+i,column_+j) += (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,aligned,false,true>::addAssign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT2> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         matrix_(row_+i,column_+element->index()) += element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,aligned,false,true>::addAssign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT2> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         matrix_(row_+element->index(),column_+j) += element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline DisableIf_< typename Submatrix<MT,aligned,false,true>::BLAZE_TEMPLATE VectorizedSubAssign<MT2> >
   Submatrix<MT,aligned,false,true>::subAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( n_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<m_; ++i )
   {
      if( IsDiagonal<MT2>::value ) {
         matrix_(row_+i,column_+i) -= (~rhs)(i,i);
      }
      else {
         for( size_t j=0UL; j<jpos; j+=2UL ) {
            matrix_(row_+i,column_+j    ) -= (~rhs)(i,j    );
            matrix_(row_+i,column_+j+1UL) -= (~rhs)(i,j+1UL);
         }
         if( jpos < n_ ) {
            matrix_(row_+i,column_+jpos) -= (~rhs)(i,jpos);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the subtraction assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline EnableIf_< typename Submatrix<MT,aligned,false,true>::BLAZE_TEMPLATE VectorizedSubAssign<MT2> >
   Submatrix<MT,aligned,false,true>::subAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
   {
      const size_t jbegin( ( IsUpper<MT2>::value )
                           ?( ( IsStrictlyUpper<MT2>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                           :( 0UL ) );
      const size_t jend  ( ( IsLower<MT2>::value )
                           ?( IsStrictlyLower<MT2>::value ? i : i+1UL )
                           :( n_ ) );
      BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

      const size_t jpos( jend & size_t(-SIMDSIZE) );
      BLAZE_INTERNAL_ASSERT( ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

      size_t j( jbegin );
      Iterator left( begin(i) + jbegin );
      ConstIterator_<MT2> right( (~rhs).begin(i) + jbegin );

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; j<jpos; j+=SIMDSIZE ) {
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; j<jend; ++j ) {
         *left -= *right; ++left; ++right;
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void Submatrix<MT,aligned,false,true>::subAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<m_; ii+=block ) {
      const size_t iend( ( m_<(ii+block) )?( m_ ):( ii+block ) );
      for( size_t jj=0UL; jj<n_; jj+=block ) {
         const size_t jend( ( n_<(jj+block) )?( n_ ):( jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               matrix_(row_+i,column_+j) -= (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,aligned,false,true>::subAssign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT2> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         matrix_(row_+i,column_+element->index()) -= element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,aligned,false,true>::subAssign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT2> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         matrix_(row_+element->index(),column_+j) -= element->value();
}
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ALIGNED COLUMN-MAJOR DENSE SUBMATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Submatrix for aligned column-major dense submatrices.
// \ingroup views
//
// This specialization of Submatrix adapts the class template to the requirements of aligned
// column-major dense submatrices.
*/
template< typename MT >  // Type of the dense matrix
class Submatrix<MT,aligned,true,true>
   : public DenseMatrix< Submatrix<MT,aligned,true,true>, true >
   , private View
{
 private:
   //**Type definitions****************************************************************************
   //! Composite data type of the dense matrix expression.
   typedef If_< IsExpression<MT>, MT, MT& >  Operand;
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef Submatrix<MT,aligned,true,true>  This;           //!< Type of this Submatrix instance.
   typedef DenseMatrix<This,true>           BaseType;       //!< Base type of this Submatrix instance.
   typedef SubmatrixTrait_<MT>              ResultType;     //!< Result type for expression template evaluations.
   typedef OppositeType_<ResultType>        OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef TransposeType_<ResultType>       TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<MT>                 ElementType;    //!< Type of the submatrix elements.
   typedef SIMDTrait_<ElementType>          SIMDType;       //!< SIMD type of the submatrix elements.
   typedef ReturnType_<MT>                  ReturnType;     //!< Return type for expression template evaluations
   typedef const Submatrix&                 CompositeType;  //!< Data type for composite expression templates.

   //! Reference to a constant submatrix value.
   typedef ConstReference_<MT>  ConstReference;

   //! Reference to a non-constant submatrix value.
   typedef If_< IsConst<MT>, ConstReference, Reference_<MT> >  Reference;

   //! Pointer to a constant submatrix value.
   typedef const ElementType*  ConstPointer;

   //! Pointer to a non-constant submatrix value.
   typedef If_< Or< IsConst<MT>, Not< HasMutableDataAccess<MT> > >, ConstPointer, ElementType* >  Pointer;

   //! Iterator over constant elements.
   typedef ConstIterator_<MT>  ConstIterator;

   //! Iterator over non-constant elements.
   typedef If_< IsConst<MT>, ConstIterator, Iterator_<MT> >  Iterator;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = MT::simdEnabled };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = MT::smpAssignable };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline Submatrix( Operand matrix, size_t rindex, size_t cindex, size_t m, size_t n );
   // No explicitly declared copy constructor.
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
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Pointer        data  ( size_t j ) noexcept;
   inline ConstPointer   data  ( size_t j ) const noexcept;
   inline Iterator       begin ( size_t j );
   inline ConstIterator  begin ( size_t j ) const;
   inline ConstIterator  cbegin( size_t j ) const;
   inline Iterator       end   ( size_t j );
   inline ConstIterator  end   ( size_t j ) const;
   inline ConstIterator  cend  ( size_t j ) const;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline Submatrix& operator=( const ElementType& rhs );
   inline Submatrix& operator=( initializer_list< initializer_list<ElementType> > list );
   inline Submatrix& operator=( const Submatrix& rhs );

   template< typename MT2, bool SO >
   inline Submatrix& operator=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator+=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator+=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator-=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix& >
      operator-=( const Matrix<MT2,SO>& rhs );

   template< typename MT2, bool SO >
   inline Submatrix& operator*=( const Matrix<MT2,SO>& rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, Submatrix >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, Submatrix >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t     row() const noexcept;
                              inline size_t     rows() const noexcept;
                              inline size_t     column() const noexcept;
                              inline size_t     columns() const noexcept;
                              inline size_t     spacing() const noexcept;
                              inline size_t     capacity() const noexcept;
                              inline size_t     capacity( size_t i ) const noexcept;
                              inline size_t     nonZeros() const;
                              inline size_t     nonZeros( size_t i ) const;
                              inline void       reset();
                              inline void       reset( size_t i );
                              inline Submatrix& transpose();
                              inline Submatrix& ctranspose();
   template< typename Other > inline Submatrix& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT2 >
   struct VectorizedAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<MT2> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT2 >
   struct VectorizedAddAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<MT2> >::value &&
                            HasSIMDAdd< ElementType, ElementType_<MT2> >::value &&
                            !IsDiagonal<MT2>::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT2 >
   struct VectorizedSubAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<MT2> >::value &&
                            HasSIMDSub< ElementType, ElementType_<MT2> >::value &&
                            !IsDiagonal<MT2>::value };
   };
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   enum : size_t { SIMDSIZE = SIMDTrait<ElementType>::size };
   //**********************************************************************************************

 public:
   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other >
   inline bool canAlias( const Other* alias ) const noexcept;

   template< typename MT2, bool AF2, bool SO2 >
   inline bool canAlias( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< typename MT2, bool AF2, bool SO2 >
   inline bool isAliased( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept;

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t i, size_t j ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t i, size_t j, const SIMDType& value ) noexcept;

   template< typename MT2 >
   inline DisableIf_< VectorizedAssign<MT2> > assign( const DenseMatrix<MT2,true>& rhs );

   template< typename MT2 >
   inline EnableIf_< VectorizedAssign<MT2> > assign( const DenseMatrix<MT2,true>& rhs );

   template< typename MT2 > inline void assign( const DenseMatrix<MT2,false>&  rhs );
   template< typename MT2 > inline void assign( const SparseMatrix<MT2,true>&  rhs );
   template< typename MT2 > inline void assign( const SparseMatrix<MT2,false>& rhs );

   template< typename MT2 >
   inline DisableIf_< VectorizedAddAssign<MT2> > addAssign( const DenseMatrix<MT2,true>& rhs );

   template< typename MT2 >
   inline EnableIf_< VectorizedAddAssign<MT2> > addAssign( const DenseMatrix<MT2,true>& rhs );

   template< typename MT2 > inline void addAssign( const DenseMatrix<MT2,false>&  rhs );
   template< typename MT2 > inline void addAssign( const SparseMatrix<MT2,true>&  rhs );
   template< typename MT2 > inline void addAssign( const SparseMatrix<MT2,false>& rhs );

   template< typename MT2 >
   inline DisableIf_< VectorizedSubAssign<MT2> > subAssign( const DenseMatrix<MT2,true>& rhs );

   template< typename MT2 >
   inline EnableIf_< VectorizedSubAssign<MT2> > subAssign( const DenseMatrix<MT2,true>& rhs );

   template< typename MT2 > inline void subAssign( const DenseMatrix<MT2,false>&  rhs );
   template< typename MT2 > inline void subAssign( const SparseMatrix<MT2,true>&  rhs );
   template< typename MT2 > inline void subAssign( const SparseMatrix<MT2,false>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline bool hasOverlap() const noexcept;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand      matrix_;  //!< The dense matrix containing the submatrix.
   const size_t row_;     //!< The first row of the submatrix.
   const size_t column_;  //!< The first column of the submatrix.
   const size_t m_;       //!< The number of rows of the submatrix.
   const size_t n_;       //!< The number of columns of the submatrix.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename MT2, bool AF2, bool SO2, bool DF2 > friend class Submatrix;

   template< bool AF1, typename MT2, bool AF2, bool SO2, bool DF2 >
   friend const Submatrix<MT2,AF1,SO2,DF2>
      submatrix( const Submatrix<MT2,AF2,SO2,DF2>& sm, size_t row, size_t column, size_t m, size_t n );

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isIntact( const Submatrix<MT2,AF2,SO2,DF2>& sm ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isSame( const Submatrix<MT2,AF2,SO2,DF2>& a, const Matrix<MT2,SO2>& b ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isSame( const Matrix<MT2,SO2>& a, const Submatrix<MT2,AF2,SO2,DF2>& b ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend bool isSame( const Submatrix<MT2,AF2,SO2,DF2>& a, const Submatrix<MT2,AF2,SO2,DF2>& b ) noexcept;

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool tryAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                          size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename MT3, bool SO3 >
   friend bool tryAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Matrix<MT3,SO3>& rhs,
                          size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool tryAddAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename MT3, bool SO3 >
   friend bool tryAddAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Matrix<MT3,SO3>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool trySubAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename MT3, bool SO3 >
   friend bool trySubAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Matrix<MT3,SO3>& rhs,
                             size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2, typename VT, bool TF >
   friend bool tryMultAssign( const Submatrix<MT2,AF2,SO2,DF2>& lhs, const Vector<VT,TF>& rhs,
                              size_t row, size_t column );

   template< typename MT2, bool AF2, bool SO2, bool DF2 >
   friend DerestrictTrait_< Submatrix<MT2,AF2,SO2,DF2> > derestrict( Submatrix<MT2,AF2,SO2,DF2>& sm );
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE       ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE      ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SUBMATRIX_TYPE      ( MT );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE        ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE      ( MT );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for Submatrix.
//
// \param matrix The dense matrix containing the submatrix.
// \param rindex The index of the first row of the submatrix in the given dense matrix.
// \param cindex The index of the first column of the submatrix in the given dense matrix.
// \param m The number of rows of the submatrix.
// \param n The number of columns of the submatrix.
// \exception std::invalid_argument Invalid submatrix specification.
//
// In case the submatrix is not properly specified (i.e. if the specified submatrix is not
// contained in the given dense matrix) a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,aligned,true,true>::Submatrix( Operand matrix, size_t rindex, size_t cindex, size_t m, size_t n )
   : matrix_( matrix )  // The dense matrix containing the submatrix
   , row_   ( rindex )  // The first row of the submatrix
   , column_( cindex )  // The first column of the submatrix
   , m_     ( m      )  // The number of rows of the submatrix
   , n_     ( n      )  // The number of columns of the submatrix
{
   if( ( row_ + m_ > matrix_.rows() ) || ( column_ + n_ > matrix_.columns() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid submatrix specification" );
   }

   if( ( simdEnabled && matrix_.data() != nullptr && !checkAlignment( data() ) ) ||
       ( n_ > 1UL && matrix_.spacing() % SIMDSIZE != 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid submatrix alignment" );
   }
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
/*!\brief 2D-access to the dense submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::Reference
   Submatrix<MT,aligned,true,true>::operator()( size_t i, size_t j )
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return matrix_(row_+i,column_+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the dense submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::ConstReference
   Submatrix<MT,aligned,true,true>::operator()( size_t i, size_t j ) const
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return const_cast<const MT&>( matrix_ )(row_+i,column_+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::Reference
   Submatrix<MT,aligned,true,true>::at( size_t i, size_t j )
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
/*!\brief Checked access to the submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::ConstReference
   Submatrix<MT,aligned,true,true>::at( size_t i, size_t j ) const
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
/*!\brief Low-level data access to the submatrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense submatrix. Note that
// you can NOT assume that all matrix elements lie adjacent to each other! The dense submatrix
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::Pointer
   Submatrix<MT,aligned,true,true>::data() noexcept
{
   return matrix_.data() + row_ + column_*spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the submatrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense submatrix. Note that
// you can NOT assume that all matrix elements lie adjacent to each other! The dense submatrix
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::ConstPointer
   Submatrix<MT,aligned,true,true>::data() const noexcept
{
   return matrix_.data() + row_ + column_*spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the submatrix elements of column \a j.
//
// \param j The column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::Pointer
   Submatrix<MT,aligned,true,true>::data( size_t j ) noexcept
{
   return matrix_.data() + row_ + (column_+j)*spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the submatrix elements of column \a j.
//
// \param j The column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::ConstPointer
   Submatrix<MT,aligned,true,true>::data( size_t j ) const noexcept
{
   return matrix_.data() + row_ + (column_+j)*spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator to the first non-zero element of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::Iterator
   Submatrix<MT,aligned,true,true>::begin( size_t j )
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid dense submatrix column access index" );
   return ( matrix_.begin( column_ + j ) + row_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator to the first non-zero element of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::ConstIterator
   Submatrix<MT,aligned,true,true>::begin( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid dense submatrix column access index" );
   return ( matrix_.cbegin( column_ + j ) + row_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator to the first non-zero element of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::ConstIterator
   Submatrix<MT,aligned,true,true>::cbegin( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid dense submatrix column access index" );
   return ( matrix_.cbegin( column_ + j ) + row_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last non-zero element of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::Iterator
   Submatrix<MT,aligned,true,true>::end( size_t j )
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid dense submatrix column access index" );
   return ( matrix_.begin( column_ + j ) + row_ + m_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last non-zero element of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::ConstIterator
   Submatrix<MT,aligned,true,true>::end( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid dense submatrix column access index" );
   return ( matrix_.cbegin( column_ + j ) + row_ + m_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last non-zero element of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline typename Submatrix<MT,aligned,true,true>::ConstIterator
   Submatrix<MT,aligned,true,true>::cend( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid dense submatrix column access index" );
   return ( matrix_.cbegin( column_ + j ) + row_ + m_ );
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
/*!\brief Homogenous assignment to all submatrix elements.
//
// \param rhs Scalar value to be assigned to all submatrix elements.
// \return Reference to the assigned submatrix.
//
// This function homogeneously assigns the given value to all dense matrix elements. Note that in
// case the underlying dense matrix is a lower/upper matrix only lower/upper and diagonal elements
// of the underlying matrix are modified.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,aligned,true,true>&
   Submatrix<MT,aligned,true,true>::operator=( const ElementType& rhs )
{
   const size_t jend( column_ + n_ );

   for( size_t j=column_; j<jend; ++j )
   {
      const size_t ibegin( ( IsLower<MT>::value )
                           ?( ( IsUniLower<MT>::value || IsStrictlyLower<MT>::value )
                              ?( max( j+1UL, row_ ) )
                              :( max( j, row_ ) ) )
                           :( row_ ) );
      const size_t iend  ( ( IsUpper<MT>::value )
                           ?( ( IsUniUpper<MT>::value || IsStrictlyUpper<MT>::value )
                              ?( min( j, row_+m_ ) )
                              :( min( j+1UL, row_+m_ ) ) )
                           :( row_+m_ ) );

      for( size_t i=ibegin; i<iend; ++i )
         matrix_(i,j) = rhs;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all submatrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to submatrix.
//
// This assignment operator offers the option to directly assign to all elements of the submatrix
// by means of an initializer list. The submatrix elements are assigned the values from the given
// initializer list. Missing values are initialized as default. Note that in case the size
// of the top-level initializer list exceeds the number of rows or the size of any nested list
// exceeds the number of columns, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,aligned,true,true>&
   Submatrix<MT,aligned,true,true>::operator=( initializer_list< initializer_list<ElementType> > list )
{
   if( list.size() != rows() || determineColumns( list ) > columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to submatrix" );
   }

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      size_t j( 0UL );
      for( const auto& element : rowList ) {
         matrix_(row_+i,column_+j) = element;
         ++j;
      }
      for( ; j<n_; ++j ) {
         matrix_(row_+i,column_+j) = ElementType();
      }
      ++i;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for Submatrix.
//
// \param rhs Sparse submatrix to be copied.
// \return Reference to the assigned submatrix.
// \exception std::invalid_argument Submatrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The dense submatrix is initialized as a copy of the given dense submatrix. In case the current
// sizes of the two submatrices don't match, a \a std::invalid_argument exception is thrown. Also,
// if the underlying matrix \a MT is a lower triangular, upper triangular, or symmetric matrix
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,aligned,true,true>&
   Submatrix<MT,aligned,true,true>::operator=( const Submatrix& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( this == &rhs || ( &matrix_ == &rhs.matrix_ && row_ == rhs.row_ && column_ == rhs.column_ ) )
      return *this;

   if( rows() != rhs.rows() || columns() != rhs.columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Submatrix sizes do not match" );
   }

   if( !tryAssign( matrix_, rhs, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( rhs.canAlias( &matrix_ ) ) {
      const ResultType tmp( rhs );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different matrices.
//
// \param rhs Matrix to be assigned.
// \return Reference to the assigned submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The dense submatrix is initialized as a copy of the given dense submatrix. In case the
// current sizes of the two matrices don't match, a \a std::invalid_argument exception is
// thrown. Also, if the underlying matrix \a MT is a symmetric matrix and the assignment
// would violate its symmetry, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline Submatrix<MT,aligned,true,true>&
   Submatrix<MT,aligned,true,true>::operator=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   typedef If_< IsRestricted<MT>, CompositeType_<MT2>, const MT2& >  Right;
   Right right( ~rhs );

   if( !tryAssign( matrix_, right, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   if( IsSparseMatrix<MT2>::value )
      reset();

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &matrix_ ) ) {
      const ResultType_<MT2> tmp( right );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO  >     // Storage order of the right-hand side matrix
inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,aligned,true,true>& >
   Submatrix<MT,aligned,true,true>::operator+=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef AddTrait_< ResultType, ResultType_<MT2> >  AddType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( !tryAddAssign( matrix_, ~rhs, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( ( ( IsSymmetric<MT>::value || IsHermitian<MT>::value ) && hasOverlap() ) ||
       (~rhs).canAlias( &matrix_ ) ) {
      const AddType tmp( *this + (~rhs) );
      smpAssign( left, tmp );
   }
   else {
      smpAddAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO  >     // Storage order of the right-hand side matrix
inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,aligned,true,true>& >
   Submatrix<MT,aligned,true,true>::operator+=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef AddTrait_< ResultType, ResultType_<MT2> >  AddType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const AddType tmp( *this + (~rhs) );

   if( !tryAssign( matrix_, tmp, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline DisableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,aligned,true,true>& >
   Submatrix<MT,aligned,true,true>::operator-=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef SubTrait_< ResultType, ResultType_<MT2> >  SubType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( !trySubAssign( matrix_, ~rhs, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( ( ( IsSymmetric<MT>::value || IsHermitian<MT>::value ) && hasOverlap() ) ||
       (~rhs).canAlias( &matrix_ ) ) {
      const SubType tmp( *this - (~rhs ) );
      smpAssign( left, tmp );
   }
   else {
      smpSubAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the submatrix.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline EnableIf_< And< IsRestricted<MT>, RequiresEvaluation<MT2> >, Submatrix<MT,aligned,true,true>& >
   Submatrix<MT,aligned,true,true>::operator-=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef SubTrait_< ResultType, ResultType_<MT2> >  SubType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const SubType tmp( *this - (~rhs) );

   if( !tryAssign( matrix_, tmp, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of a matrix (\f$ A*=B \f$).
//
// \param rhs The right-hand side matrix for the multiplication.
// \return Reference to the dense submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline Submatrix<MT,aligned,true,true>&
   Submatrix<MT,aligned,true,true>::operator*=( const Matrix<MT2,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<MT2> );

   typedef MultTrait_< ResultType, ResultType_<MT2> >  MultType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( MultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MultType );

   if( columns() != (~rhs).rows() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const MultType tmp( *this * (~rhs) );

   if( !tryAssign( matrix_, tmp, row_, column_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication between a dense submatrix
//        and a scalar value (\f$ A*=s \f$).
//
// \param rhs The right-hand side scalar value for the multiplication.
// \return Reference to the dense submatrix.
//
// This operator cannot be used for submatrices on lower or upper unitriangular matrices. The
// attempt to scale such a submatrix results in a compilation error!
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, Submatrix<MT,aligned,true,true> >&
   Submatrix<MT,aligned,true,true>::operator*=( Other rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   DerestrictTrait_<This> left( derestrict( *this ) );
   smpAssign( left, (*this) * rhs );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division assignment operator for the division of a dense submatrix by a scalar value
//        (\f$ A/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the dense submatrix.
//
// This operator cannot be used for submatrices on lower or upper unitriangular matrices. The
// attempt to scale such a submatrix results in a compilation error!
//
// \note A division by zero is only checked by an user assert.
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, Submatrix<MT,aligned,true,true> >&
   Submatrix<MT,aligned,true,true>::operator/=( Other rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

   DerestrictTrait_<This> left( derestrict( *this ) );
   smpAssign( left, (*this) / rhs );

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
/*!\brief Returns the index of the first row of the submatrix in the underlying dense matrix.
//
// \return The index of the first row.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,true,true>::row() const noexcept
{
   return row_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of rows of the dense submatrix.
//
// \return The number of rows of the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,true,true>::rows() const noexcept
{
   return m_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first column of the submatrix in the underlying dense matrix.
//
// \return The index of the first column.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,true,true>::column() const noexcept
{
   return column_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of columns of the dense submatrix.
//
// \return The number of columns of the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,true,true>::columns() const noexcept
{
   return n_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the spacing between the beginning of two columns.
//
// \return The spacing between the beginning of two columns.
//
// This function returns the spacing between the beginning of two columns, i.e. the total
// number of elements of a column.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,true,true>::spacing() const noexcept
{
   return matrix_.spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense submatrix.
//
// \return The capacity of the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,true,true>::capacity() const noexcept
{
   return rows() * columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current capacity of the specified column.
//
// \param j The index of the column.
// \return The current capacity of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,true,true>::capacity( size_t j ) const noexcept
{
   UNUSED_PARAMETER( j );

   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the dense submatrix
//
// \return The number of non-zero elements in the dense submatrix.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,true,true>::nonZeros() const
{
   const size_t iend( row_ + m_ );
   const size_t jend( column_ + n_ );
   size_t nonzeros( 0UL );

   for( size_t j=column_; j<jend; ++j )
      for( size_t i=row_; i<iend; ++i )
         if( !isDefault( matrix_(i,j) ) )
            ++nonzeros;

   return nonzeros;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the specified column.
//
// \param j The index of the column.
// \return The number of non-zero elements of column \a j.
*/
template< typename MT >  // Type of the dense matrix
inline size_t Submatrix<MT,aligned,true,true>::nonZeros( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   const size_t iend( row_ + m_ );
   size_t nonzeros( 0UL );

   for( size_t i=row_; i<iend; ++i )
      if( !isDefault( matrix_(i,column_+j) ) )
         ++nonzeros;

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
template< typename MT >  // Type of the dense matrix
inline void Submatrix<MT,aligned,true,true>::reset()
{
   using blaze::clear;

   for( size_t j=column_; j<column_+n_; ++j )
   {
      const size_t ibegin( ( IsLower<MT>::value )
                           ?( ( IsUniLower<MT>::value || IsStrictlyLower<MT>::value )
                              ?( max( j+1UL, row_ ) )
                              :( max( j, row_ ) ) )
                           :( row_ ) );
      const size_t iend  ( ( IsUpper<MT>::value )
                           ?( ( IsUniUpper<MT>::value || IsStrictlyUpper<MT>::value )
                              ?( min( j, row_+m_ ) )
                              :( min( j+1UL, row_+m_ ) ) )
                           :( row_+m_ ) );

      for( size_t i=ibegin; i<iend; ++i )
         clear( matrix_(i,j) );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset the specified column to the default initial values.
//
// \param j The index of the column.
// \return void
*/
template< typename MT >  // Type of the dense matrix
inline void Submatrix<MT,aligned,true,true>::reset( size_t j )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   const size_t ibegin( ( IsLower<MT>::value )
                        ?( ( IsUniLower<MT>::value || IsStrictlyLower<MT>::value )
                           ?( max( j+1UL, row_ ) )
                           :( max( j, row_ ) ) )
                        :( row_ ) );
   const size_t iend  ( ( IsUpper<MT>::value )
                        ?( ( IsUniUpper<MT>::value || IsStrictlyUpper<MT>::value )
                           ?( min( j, row_+m_ ) )
                           :( min( j+1UL, row_+m_ ) ) )
                        :( row_+m_ ) );

   for( size_t i=ibegin; i<iend; ++i )
      clear( matrix_(i,column_+j) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transpose of the submatrix.
//
// \return Reference to the transposed submatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic submatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense submatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the submatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the submatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,aligned,true,true>& Submatrix<MT,aligned,true,true>::transpose()
{
   if( m_ != n_ ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic submatrix" );
   }

   if( !tryAssign( matrix_, trans( *this ), row_, column_ ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );
   const ResultType tmp( trans( *this ) );
   smpAssign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place conjugate transpose of the submatrix.
//
// \return Reference to the transposed submatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic submatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense submatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the submatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the submatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT >  // Type of the dense matrix
inline Submatrix<MT,aligned,true,true>& Submatrix<MT,aligned,true,true>::ctranspose()
{
   if( m_ != n_ ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic submatrix" );
   }

   if( !tryAssign( matrix_, ctrans( *this ), row_, column_ ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );
   const ResultType tmp( ctrans( *this ) );
   smpAssign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the dense submatrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the submatrix scaling.
// \return Reference to the dense submatrix.
//
// This function scales all elements of the submatrix by the given scalar value \a scalar. Note
// that the function cannot be used to scale a submatrix on a lower or upper unitriangular matrix.
// The attempt to scale such a submatrix results in a compile time error!
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the scalar value
inline Submatrix<MT,aligned,true,true>& Submatrix<MT,aligned,true,true>::scale( const Other& scalar )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   const size_t jend( column_ + n_ );

   for( size_t j=column_; j<jend; ++j )
   {
      const size_t ibegin( ( IsLower<MT>::value )
                           ?( ( IsStrictlyLower<MT>::value )
                              ?( max( j+1UL, row_ ) )
                              :( max( j, row_ ) ) )
                           :( row_ ) );
      const size_t iend  ( ( IsUpper<MT>::value )
                           ?( ( IsStrictlyUpper<MT>::value )
                              ?( min( j, row_+m_ ) )
                              :( min( j+1UL, row_+m_ ) ) )
                           :( row_+m_ ) );

      for( size_t i=ibegin; i<iend; ++i )
         matrix_(i,j) *= scalar;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checking whether there exists an overlap in the context of a symmetric matrix.
//
// \return \a true in case an overlap exists, \a false if not.
//
// This function checks if in the context of a symmetric matrix the submatrix has an overlap with
// its counterpart. In case an overlap exists, the function return \a true, otherwise it returns
// \a false.
*/
template< typename MT >  // Type of the dense matrix
inline bool Submatrix<MT,aligned,true,true>::hasOverlap() const noexcept
{
   BLAZE_INTERNAL_ASSERT( IsSymmetric<MT>::value || IsHermitian<MT>::value, "Invalid matrix detected" );

   if( ( row_ + m_ <= column_ ) || ( column_ + n_ <= row_ ) )
      return false;
   else return true;
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
/*!\brief Returns whether the submatrix can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address can alias with the submatrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the foreign expression
inline bool Submatrix<MT,aligned,true,true>::canAlias( const Other* alias ) const noexcept
{
   return matrix_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix can alias with the given dense submatrix \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address can alias with the submatrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Data type of the foreign dense submatrix
        , bool AF2       // Alignment flag of the foreign dense submatrix
        , bool SO2 >     // Storage order of the foreign dense submatrix
inline bool Submatrix<MT,aligned,true,true>::canAlias( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept
{
   return ( matrix_.isAliased( &alias->matrix_ ) &&
            ( row_    + m_ > alias->row_    ) && ( row_    < alias->row_    + alias->m_ ) &&
            ( column_ + n_ > alias->column_ ) && ( column_ < alias->column_ + alias->n_ ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address is aliased with the submatrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >     // Type of the dense matrix
template< typename Other >  // Data type of the foreign expression
inline bool Submatrix<MT,aligned,true,true>::isAliased( const Other* alias ) const noexcept
{
   return matrix_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is aliased with the given dense submatrix \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address is aliased with the submatrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT >  // Type of the dense matrix
template< typename MT2   // Data type of the foreign dense submatrix
        , bool AF2       // Alignment flag of the foreign dense submatrix
        , bool SO2 >     // Storage order of the foreign dense submatrix
inline bool Submatrix<MT,aligned,true,true>::isAliased( const Submatrix<MT2,AF2,SO2,true>* alias ) const noexcept
{
   return ( matrix_.isAliased( &alias->matrix_ ) &&
            ( row_    + m_ > alias->row_    ) && ( row_    < alias->row_    + alias->m_ ) &&
            ( column_ + n_ > alias->column_ ) && ( column_ < alias->column_ + alias->n_ ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is properly aligned in memory.
//
// \return \a true in case the submatrix is aligned, \a false if not.
//
// This function returns whether the submatrix is guaranteed to be properly aligned in memory,
// i.e. whether the beginning and the end of each column of the submatrix are guaranteed to
// conform to the alignment restrictions of the underlying element type.
*/
template< typename MT >  // Type of the dense matrix
inline bool Submatrix<MT,aligned,true,true>::isAligned() const noexcept
{
   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix can be used in SMP assignments.
//
// \return \a true in case the submatrix can be used in SMP assignments, \a false if not.
//
// This function returns whether the submatrix can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the submatrix).
*/
template< typename MT >  // Type of the dense matrix
inline bool Submatrix<MT,aligned,true,true>::canSMPAssign() const noexcept
{
   return ( columns() > SMP_DMATASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Load of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense submatrix. The row
// index must be smaller than the number of rows and the column index must be smaller than
// the number of columns. Additionally, the row index must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is
// used internally for the performance optimized evaluation of expression templates. Calling
// this function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename Submatrix<MT,aligned,true,true>::SIMDType
   Submatrix<MT,aligned,true,true>::load( size_t i, size_t j ) const noexcept
{
   return loada( i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the row index must be a multiple of the number
// of values inside the SIMD element. This function must \b NOT be called explicitly! It is
// used internally for the performance optimized evaluation of expression templates. Calling
// this function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename Submatrix<MT,aligned,true,true>::SIMDType
   Submatrix<MT,aligned,true,true>::loada( size_t i, size_t j ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );

   return matrix_.loada( row_+i, column_+j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned load of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the row index must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE typename Submatrix<MT,aligned,true,true>::SIMDType
   Submatrix<MT,aligned,true,true>::loadu( size_t i, size_t j ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );

   return matrix_.loadu( row_+i, column_+j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store of a specific SIMD element of the dense submatrix. The row
// index must be smaller than the number of rows and the column index must be smaller than
// the number of columns. Additionally, the row index must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,aligned,true,true>::store( size_t i, size_t j, const SIMDType& value ) noexcept
{
   storea( i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the row index must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,aligned,true,true>::storea( size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );

   matrix_.storea( row_+i, column_+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store of a specific SIMD element of the dense submatrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the row index must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,aligned,true,true>::storeu( size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );

   matrix_.storeu( row_+i, column_+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned, non-temporal store of a SIMD element of the submatrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store of a specific SIMD element of the
// dense submatrix. The row index must be smaller than the number of rows and the column
// index must be smaller than the number of columns. Additionally, the row index must be
// a multiple of the number of values inside the SIMD element. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename MT >  // Type of the dense matrix
BLAZE_ALWAYS_INLINE void
   Submatrix<MT,aligned,true,true>::stream( size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );

   matrix_.stream( row_+i, column_+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline DisableIf_< typename Submatrix<MT,aligned,true,true>::BLAZE_TEMPLATE VectorizedAssign<MT2> >
   Submatrix<MT,aligned,true,true>::assign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t ipos( m_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( m_ - ( m_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t j=0UL; j<n_; ++j ) {
      for( size_t i=0UL; i<ipos; i+=2UL ) {
         matrix_(row_+i    ,column_+j) = (~rhs)(i    ,j);
         matrix_(row_+i+1UL,column_+j) = (~rhs)(i+1UL,j);
      }
      if( ipos < m_ ) {
         matrix_(row_+ipos,column_+j) = (~rhs)(ipos,j);
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline EnableIf_< typename Submatrix<MT,aligned,true,true>::BLAZE_TEMPLATE VectorizedAssign<MT2> >
   Submatrix<MT,aligned,true,true>::assign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t ipos( m_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( m_ - ( m_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   if( useStreaming &&
       m_*n_ > ( cacheSize / ( sizeof(ElementType) * 3UL ) ) &&
       !(~rhs).isAliased( &matrix_ ) )
   {
      for( size_t j=0UL; j<n_; ++j )
      {
         size_t i( 0UL );
         Iterator left( begin(j) );
         ConstIterator_<MT2> right( (~rhs).begin(j) );

         for( ; i<ipos; i+=SIMDSIZE ) {
            left.stream( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; i<m_; ++i ) {
            *left = *right; ++left; ++right;
         }
      }
   }
   else
   {
      for( size_t j=0UL; j<n_; ++j )
      {
         size_t i( 0UL );
         Iterator left( begin(j) );
         ConstIterator_<MT2> right( (~rhs).begin(j) );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; i<ipos; i+=SIMDSIZE ) {
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; i<m_; ++i ) {
            *left = *right; ++left; ++right;
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void Submatrix<MT,aligned,true,true>::assign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t jj=0UL; jj<n_; jj+=block ) {
      const size_t jend( ( n_<(jj+block) )?( n_ ):( jj+block ) );
      for( size_t ii=0UL; ii<m_; ii+=block ) {
         const size_t iend( ( m_<(ii+block) )?( m_ ):( ii+block ) );
         for( size_t j=jj; j<jend; ++j ) {
            for( size_t i=ii; i<iend; ++i ) {
               matrix_(row_+i,column_+j) = (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,aligned,true,true>::assign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT2> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         matrix_(row_+element->index(),column_+j) = element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,aligned,true,true>::assign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT2> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         matrix_(row_+i,column_+element->index()) = element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline DisableIf_< typename Submatrix<MT,aligned,true,true>::BLAZE_TEMPLATE VectorizedAddAssign<MT2> >
   Submatrix<MT,aligned,true,true>::addAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t ipos( m_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( m_ - ( m_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t j=0UL; j<n_; ++j )
   {
      if( IsDiagonal<MT2>::value ) {
         matrix_(row_+j,column_+j) += (~rhs)(j,j);
      }
      else {
         for( size_t i=0UL; i<ipos; i+=2UL ) {
            matrix_(row_+i    ,column_+j) += (~rhs)(i    ,j);
            matrix_(row_+i+1UL,column_+j) += (~rhs)(i+1UL,j);
         }
         if( ipos < m_ ) {
            matrix_(row_+ipos,column_+j) += (~rhs)(ipos,j);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the addition assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline EnableIf_< typename Submatrix<MT,aligned,true,true>::BLAZE_TEMPLATE VectorizedAddAssign<MT2> >
   Submatrix<MT,aligned,true,true>::addAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
   {
      const size_t ibegin( ( IsLower<MT>::value )
                           ?( ( IsStrictlyLower<MT>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                           :( 0UL ) );
      const size_t iend  ( ( IsUpper<MT>::value )
                           ?( IsStrictlyUpper<MT>::value ? j : j+1UL )
                           :( m_ ) );
      BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

      const size_t ipos( iend & size_t(-SIMDSIZE) );
      BLAZE_INTERNAL_ASSERT( ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

      size_t i( ibegin );
      Iterator left( begin(j) + ibegin );
      ConstIterator_<MT2> right( (~rhs).begin(j) + ibegin );

      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<ipos; i+=SIMDSIZE ) {
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<iend; ++i ) {
         *left += *right; ++left; ++right;
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void Submatrix<MT,aligned,true,true>::addAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t jj=0UL; jj<n_; jj+=block ) {
      const size_t jend( ( n_<(jj+block) )?( n_ ):( jj+block ) );
      for( size_t ii=0UL; ii<m_; ii+=block ) {
         const size_t iend( ( m_<(ii+block) )?( m_ ):( ii+block ) );
         for( size_t j=jj; j<jend; ++j ) {
            for( size_t i=ii; i<iend; ++i ) {
               matrix_(row_+i,column_+j) += (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,aligned,true,true>::addAssign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT2> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         matrix_(row_+element->index(),column_+j) += element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,aligned,true,true>::addAssign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT2> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         matrix_(row_+i,column_+element->index()) += element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline DisableIf_< typename Submatrix<MT,aligned,true,true>::BLAZE_TEMPLATE VectorizedSubAssign<MT2> >
   Submatrix<MT,aligned,true,true>::subAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t ipos( m_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( m_ - ( m_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t j=0UL; j<n_; ++j )
   {
      if( IsDiagonal<MT2>::value ) {
         matrix_(row_+j,column_+j) -= (~rhs)(j,j);
      }
      else {
         for( size_t i=0UL; i<ipos; i+=2UL ) {
            matrix_(row_+i    ,column_+j) -= (~rhs)(i    ,j);
            matrix_(row_+i+1UL,column_+j) -= (~rhs)(i+1UL,j);
         }
         if( ipos < m_ ) {
            matrix_(row_+ipos,column_+j) -= (~rhs)(ipos,j);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the subtraction assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline EnableIf_< typename Submatrix<MT,aligned,true,true>::BLAZE_TEMPLATE VectorizedSubAssign<MT2> >
   Submatrix<MT,aligned,true,true>::subAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
   {
      const size_t ibegin( ( IsLower<MT>::value )
                           ?( ( IsStrictlyLower<MT>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                           :( 0UL ) );
      const size_t iend  ( ( IsUpper<MT>::value )
                           ?( IsStrictlyUpper<MT>::value ? j : j+1UL )
                           :( m_ ) );
      BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

      const size_t ipos( iend & size_t(-SIMDSIZE) );
      BLAZE_INTERNAL_ASSERT( ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

      size_t i( ibegin );
      Iterator left( begin(j) + ibegin );
      ConstIterator_<MT2> right( (~rhs).begin(j) + ibegin );

      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<ipos; i+=SIMDSIZE ) {
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<iend; ++i ) {
         *left -= *right; ++left; ++right;
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side dense matrix
inline void Submatrix<MT,aligned,true,true>::subAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t jj=0UL; jj<n_; jj+=block ) {
      const size_t jend( ( n_<(jj+block) )?( n_ ):( jj+block ) );
      for( size_t ii=0UL; ii<m_; ii+=block ) {
         const size_t iend( ( m_<(ii+block) )?( m_ ):( ii+block ) );
         for( size_t j=jj; j<jend; ++j ) {
            for( size_t i=ii; i<iend; ++i ) {
               matrix_(row_+i,column_+j) -= (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,aligned,true,true>::subAssign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT2> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         matrix_(row_+element->index(),column_+j) -= element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT >   // Type of the dense matrix
template< typename MT2 >  // Type of the right-hand side sparse matrix
inline void Submatrix<MT,aligned,true,true>::subAssign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT2> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         matrix_(row_+i,column_+element->index()) -= element->value();
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
