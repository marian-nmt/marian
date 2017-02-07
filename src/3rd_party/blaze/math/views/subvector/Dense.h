//=================================================================================================
/*!
//  \file blaze/math/views/subvector/Dense.h
//  \brief Subvector specialization for dense vectors
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

#ifndef _BLAZE_MATH_VIEWS_SUBVECTOR_DENSE_H_
#define _BLAZE_MATH_VIEWS_SUBVECTOR_DENSE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/Subvector.h>
#include <blaze/math/constraints/TransExpr.h>
#include <blaze/math/constraints/TransposeFlag.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/CrossExpr.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/View.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/traits/DerestrictTrait.h>
#include <blaze/math/typetraits/AreSIMDCombinable.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDDiv.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/HasSIMDSub.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsRestricted.h>
#include <blaze/math/typetraits/IsSparseVector.h>
#include <blaze/math/views/subvector/BaseTemplate.h>
#include <blaze/system/CacheSize.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/AlignmentCheck.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Vectorizable.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Not.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/Template.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/typetraits/IsNumeric.h>


namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR UNALIGNED DENSE SUBVECTORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Subvector for unaligned dense subvectors.
// \ingroup views
//
// This specialization of Subvector adapts the class template to the requirements of unaligned
// dense subvectors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
class Subvector<VT,unaligned,TF,true>
   : public DenseVector< Subvector<VT,unaligned,TF,true>, TF >
   , private View
{
 private:
   //**Type definitions****************************************************************************
   //! Composite data type of the dense vector expression.
   typedef If_< IsExpression<VT>, VT, VT& >  Operand;
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef Subvector<VT,unaligned,TF,true>  This;           //!< Type of this Subvector instance.
   typedef DenseVector<This,TF>             BaseType;       //!< Base type of this Subvector instance.
   typedef SubvectorTrait_<VT>              ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>       TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<VT>                 ElementType;    //!< Type of the subvector elements.
   typedef SIMDTrait_<ElementType>          SIMDType;       //!< SIMD type of the subvector elements.
   typedef ReturnType_<VT>                  ReturnType;     //!< Return type for expression template evaluations
   typedef const Subvector&                 CompositeType;  //!< Data type for composite expression templates.

   //! Reference to a constant subvector value.
   typedef ConstReference_<VT>  ConstReference;

   //! Reference to a non-constant subvector value.
   typedef If_< IsConst<VT>, ConstReference, Reference_<VT> >  Reference;

   //! Pointer to a constant subvector value.
   typedef const ElementType*  ConstPointer;

   //! Pointer to a non-constant subvector value.
   typedef If_< Or< IsConst<VT>, Not< HasMutableDataAccess<VT> > >, ConstPointer, ElementType* >  Pointer;
   //**********************************************************************************************

   //**SubvectorIterator class definition**********************************************************
   /*!\brief Iterator over the elements of the sparse subvector.
   */
   template< typename IteratorType >  // Type of the dense vector iterator
   class SubvectorIterator
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
      /*!\brief Default constructor of the SubvectorIterator class.
      */
      inline SubvectorIterator()
         : iterator_ (       )  // Iterator to the current subvector element
         , isAligned_( false )  // Memory alignment flag
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor of the SubvectorIterator class.
      //
      // \param iterator Iterator to the initial element.
      // \param isMemoryAligned Memory alignment flag.
      */
      inline SubvectorIterator( IteratorType iterator, bool isMemoryAligned )
         : iterator_ ( iterator        )  // Iterator to the current subvector element
         , isAligned_( isMemoryAligned )  // Memory alignment flag
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Conversion constructor from different SubvectorIterator instances.
      //
      // \param it The subvector iterator to be copied
      */
      template< typename IteratorType2 >
      inline SubvectorIterator( const SubvectorIterator<IteratorType2>& it )
         : iterator_ ( it.base()      )  // Iterator to the current subvector element
         , isAligned_( it.isAligned() )  // Memory alignment flag
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline SubvectorIterator& operator+=( size_t inc ) {
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
      inline SubvectorIterator& operator-=( size_t dec ) {
         iterator_ -= dec;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline SubvectorIterator& operator++() {
         ++iterator_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const SubvectorIterator operator++( int ) {
         return SubvectorIterator( iterator_++, isAligned_ );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline SubvectorIterator& operator--() {
         --iterator_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const SubvectorIterator operator--( int ) {
         return SubvectorIterator( iterator_--, isAligned_ );
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
      /*!\brief Load of a SIMD element of the dense subvector.
      //
      // \return The loaded SIMD element.
      //
      // This function performs a load of the current SIMD element of the subvector iterator.
      // This function must \b NOT be called explicitly! It is used internally for the performance
      // optimized evaluation of expression templates. Calling this function explicitly might
      // result in erroneous results and/or in compilation errors.
      */
      inline SIMDType load() const {
         return loadu();
      }
      //*******************************************************************************************

      //**Loada function***************************************************************************
      /*!\brief Aligned load of a SIMD element of the dense subvector.
      //
      // \return The loaded SIMD element.
      //
      // This function performs an aligned load of the current SIMD element of the subvector
      // iterator. This function must \b NOT be called explicitly! It is used internally for the
      // performance optimized evaluation of expression templates. Calling this function explicitly
      // might result in erroneous results and/or in compilation errors.
      */
      inline SIMDType loada() const {
         return iterator_.loada();
      }
      //*******************************************************************************************

      //**Loadu function***************************************************************************
      /*!\brief Unaligned load of a SIMD element of the dense subvector.
      //
      // \return The loaded SIMD element.
      //
      // This function performs an unaligned load of the current SIMD element of the subvector
      // iterator. This function must \b NOT be called explicitly! It is used internally for the
      // performance optimized evaluation of expression templates. Calling this function explicitly
      // might result in erroneous results and/or in compilation errors.
      */
      inline SIMDType loadu() const {
         if( isAligned_ ) {
            return iterator_.loada();
         }
         else {
            return iterator_.loadu();
         }
      }
      //*******************************************************************************************

      //**Store function***************************************************************************
      /*!\brief Store of a SIMD element of the dense subvector.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs a store of the current SIMD element of the subvector iterator.
      // This function must \b NOT be called explicitly! It is used internally for the performance
      // optimized evaluation of expression templates. Calling this function explicitly might
      // result in erroneous results and/or in compilation errors.
      */
      inline void store( const SIMDType& value ) const {
         storeu( value );
      }
      //*******************************************************************************************

      //**Storea function**************************************************************************
      /*!\brief Aligned store of a SIMD element of the dense subvector.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs an aligned store of the current SIMD element of the subvector
      // iterator. This function must \b NOT be called explicitly! It is used internally for the
      // performance optimized evaluation of expression templates. Calling this function explicitly
      // might result in erroneous results and/or in compilation errors.
      */
      inline void storea( const SIMDType& value ) const {
         iterator_.storea( value );
      }
      //*******************************************************************************************

      //**Storeu function**************************************************************************
      /*!\brief Unaligned store of a SIMD element of the dense subvector.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs an unaligned store of the current SIMD element of the subvector
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
      /*!\brief Aligned, non-temporal store of a SIMD element of the dense subvector.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs an aligned, non-temporal store of the current SIMD element of the
      // subvector iterator. This function must \b NOT be called explicitly! It is used internally
      // for the performance optimized evaluation of expression templates. Calling this function
      // explicitly might result in erroneous results and/or in compilation errors.
      */
      inline void stream( const SIMDType& value ) const {
         iterator_.stream( value );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two SubvectorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const SubvectorIterator& rhs ) const {
         return iterator_ == rhs.iterator_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two SubvectorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const SubvectorIterator& rhs ) const {
         return iterator_ != rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two SubvectorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const SubvectorIterator& rhs ) const {
         return iterator_ < rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two SubvectorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const SubvectorIterator& rhs ) const {
         return iterator_ > rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two SubvectorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const SubvectorIterator& rhs ) const {
         return iterator_ <= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two SubvectorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const SubvectorIterator& rhs ) const {
         return iterator_ >= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const SubvectorIterator& rhs ) const {
         return iterator_ - rhs.iterator_;
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between a SubvectorIterator and an integral value.
      //
      // \param it The iterator to be incremented.
      // \param inc The number of elements the iterator is incremented.
      // \return The incremented iterator.
      */
      friend inline const SubvectorIterator operator+( const SubvectorIterator& it, size_t inc ) {
         return SubvectorIterator( it.iterator_ + inc, it.isAligned_ );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between an integral value and a SubvectorIterator.
      //
      // \param inc The number of elements the iterator is incremented.
      // \param it The iterator to be incremented.
      // \return The incremented iterator.
      */
      friend inline const SubvectorIterator operator+( size_t inc, const SubvectorIterator& it ) {
         return SubvectorIterator( it.iterator_ + inc, it.isAligned_ );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Subtraction between a SubvectorIterator and an integral value.
      //
      // \param it The iterator to be decremented.
      // \param dec The number of elements the iterator is decremented.
      // \return The decremented iterator.
      */
      friend inline const SubvectorIterator operator-( const SubvectorIterator& it, size_t dec ) {
         return SubvectorIterator( it.iterator_ - dec, it.isAligned_ );
      }
      //*******************************************************************************************

      //**Base function****************************************************************************
      /*!\brief Access to the current position of the subvector iterator.
      //
      // \return The current position of the subvector iterator.
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
      inline bool isAligned() const {
         return isAligned_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType iterator_;   //!< Iterator to the current subvector element.
      bool         isAligned_;  //!< Memory alignment flag.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   //! Iterator over constant elements.
   typedef SubvectorIterator< ConstIterator_<VT> >  ConstIterator;

   //! Iterator over non-constant elements.
   typedef If_< IsConst<VT>, ConstIterator, SubvectorIterator< Iterator_<VT> > >  Iterator;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = VT::simdEnabled };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = VT::smpAssignable };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline Subvector( Operand vector, size_t index, size_t n );
   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator[]( size_t index );
   inline ConstReference operator[]( size_t index ) const;
   inline Reference      at( size_t index );
   inline ConstReference at( size_t index ) const;
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Iterator       begin ();
   inline ConstIterator  begin () const;
   inline ConstIterator  cbegin() const;
   inline Iterator       end   ();
   inline ConstIterator  end   () const;
   inline ConstIterator  cend  () const;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
                            inline Subvector& operator= ( const ElementType& rhs );
                            inline Subvector& operator= ( initializer_list<ElementType> list );
                            inline Subvector& operator= ( const Subvector& rhs );
   template< typename VT2 > inline Subvector& operator= ( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline Subvector& operator+=( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline Subvector& operator-=( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline Subvector& operator*=( const DenseVector<VT2,TF>&  rhs );
   template< typename VT2 > inline Subvector& operator*=( const SparseVector<VT2,TF>& rhs );
   template< typename VT2 > inline Subvector& operator/=( const DenseVector<VT2,TF>&  rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, Subvector >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, Subvector >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t     size() const noexcept;
                              inline size_t     capacity() const noexcept;
                              inline size_t     nonZeros() const;
                              inline void       reset();
   template< typename Other > inline Subvector& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT2 >
   struct VectorizedAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<VT2> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT2 >
   struct VectorizedAddAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<VT2> >::value &&
                            HasSIMDAdd< ElementType, ElementType_<VT2> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT2 >
   struct VectorizedSubAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<VT2> >::value &&
                            HasSIMDSub< ElementType, ElementType_<VT2> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT2 >
   struct VectorizedMultAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<VT2> >::value &&
                            HasSIMDMult< ElementType, ElementType_<VT2> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT2 >
   struct VectorizedDivAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<VT2> >::value &&
                            HasSIMDDiv< ElementType, ElementType_<VT2> >::value };
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

   template< typename VT2, bool AF2, bool TF2 >
   inline bool canAlias( const Subvector<VT2,AF2,TF2,true>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< typename VT2, bool AF2, bool TF2 >
   inline bool isAliased( const Subvector<VT2,AF2,TF2,true>* alias ) const noexcept;

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t index ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t index ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t index ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t index, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t index, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t index, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t index, const SIMDType& value ) noexcept;

   template< typename VT2 >
   inline DisableIf_< VectorizedAssign<VT2> > assign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 >
   inline EnableIf_< VectorizedAssign<VT2> > assign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 > inline void assign( const SparseVector<VT2,TF>& rhs );

   template< typename VT2 >
   inline DisableIf_< VectorizedAddAssign<VT2> > addAssign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 >
   inline EnableIf_< VectorizedAddAssign<VT2> > addAssign ( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 > inline void addAssign( const SparseVector<VT2,TF>& rhs );

   template< typename VT2 >
   inline DisableIf_< VectorizedSubAssign<VT2> > subAssign ( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 >
   inline EnableIf_< VectorizedSubAssign<VT2> > subAssign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 > inline void subAssign( const SparseVector<VT2,TF>& rhs );

   template< typename VT2 >
   inline DisableIf_< VectorizedMultAssign<VT2> > multAssign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 >
   inline EnableIf_< VectorizedMultAssign<VT2> > multAssign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 > inline void multAssign( const SparseVector<VT2,TF>& rhs );

   template< typename VT2 >
   inline DisableIf_< VectorizedDivAssign<VT2> > divAssign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 >
   inline EnableIf_< VectorizedDivAssign<VT2> > divAssign( const DenseVector <VT2,TF>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand      vector_;   //!< The dense vector containing the subvector.
   const size_t offset_;   //!< The offset of the subvector within the dense vector.
   const size_t size_;     //!< The size of the subvector.
   const bool isAligned_;  //!< Memory alignment flag.
                           /*!< The alignment flag indicates whether the subvector is fully aligned
                                with respect to the given element type and the available instruction
                                set. In case the subvector is fully aligned it is possible to use
                                aligned loads and stores instead of unaligned loads and stores. In
                                order to be aligned, the first element of the subvector must be
                                aligned. */
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename VT2, bool AF2, bool TF2, bool DF2 > friend class Subvector;

   template< bool AF1, typename VT2, bool AF2, bool TF2, bool DF2 >
   friend const Subvector<VT2,AF1,TF2,DF2>
      subvector( const Subvector<VT2,AF2,TF2,DF2>& sv, size_t index, size_t size );

   template< typename VT2, bool AF2, bool TF2, bool DF2 >
   friend bool isIntact( const Subvector<VT2,AF2,TF2,DF2>& sv ) noexcept;

   template< typename VT2, bool AF2, bool TF2, bool DF2 >
   friend bool isSame( const Subvector<VT2,AF2,TF2,DF2>& a, const Vector<VT2,TF2>& b ) noexcept;

   template< typename VT2, bool AF2, bool TF2, bool DF2 >
   friend bool isSame( const Vector<VT2,TF2>& a, const Subvector<VT2,AF2,TF2,DF2>& b ) noexcept;

   template< typename VT2, bool AF2, bool TF2, bool DF2 >
   friend bool isSame( const Subvector<VT2,AF2,TF2,DF2>& a, const Subvector<VT2,AF2,TF2,DF2>& b ) noexcept;

   template< typename VT2, bool AF2, bool TF2, bool DF2, typename VT3 >
   friend bool tryAssign( const Subvector<VT2,AF2,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT2, bool AF2, bool TF2, bool DF2, typename VT3 >
   friend bool tryAddAssign( const Subvector<VT2,AF2,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT2, bool AF2, bool TF2, bool DF2, typename VT3 >
   friend bool trySubAssign( const Subvector<VT2,AF2,DF2,TF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT2, bool AF2, bool TF2, bool DF2, typename VT3 >
   friend bool tryMultAssign( const Subvector<VT2,AF2,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT2, bool AF2, bool TF2, bool DF2 >
   friend DerestrictTrait_< Subvector<VT2,AF2,TF2,DF2> > derestrict( Subvector<VT2,AF2,TF2,DF2>& sv );
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE   ( VT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( VT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE  ( VT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SUBVECTOR_TYPE  ( VT );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( VT, TF );
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
/*!\brief The constructor for Subvector.
//
// \param vector The dense vector containing the subvector.
// \param index The first index of the subvector in the given vector.
// \param n The size of the subvector.
// \exception std::invalid_argument Invalid subvector specification.
//
// In case the subvector is not properly specified (i.e. if the specified first index is larger
// than the size of the given vector or the subvector is specified beyond the size of the vector)
// a \a std::invalid_argument exception is thrown.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline Subvector<VT,unaligned,TF,true>::Subvector( Operand vector, size_t index, size_t n )
   : vector_   ( vector )  // The vector containing the subvector
   , offset_   ( index  )  // The offset of the subvector within the dense vector
   , size_     ( n      )  // The size of the subvector
   , isAligned_( simdEnabled && vector.data() != nullptr && checkAlignment( data() ) )
{
   if( index + n > vector.size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid subvector specification" );
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
/*!\brief Subscript operator for the direct access to the subvector elements.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,unaligned,TF,true>::Reference
   Subvector<VT,unaligned,TF,true>::operator[]( size_t index )
{
   BLAZE_USER_ASSERT( index < size(), "Invalid subvector access index" );
   return vector_[offset_+index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subscript operator for the direct access to the subvector elements.
//
// \param index Access index. The index must be smaller than the number of subvector columns.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,unaligned,TF,true>::ConstReference
   Subvector<VT,unaligned,TF,true>::operator[]( size_t index ) const
{
   BLAZE_USER_ASSERT( index < size(), "Invalid subvector access index" );
   return const_cast<const VT&>( vector_ )[offset_+index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the subvector elements.
//
// \param index Access index. The index must be smaller than the number of subvector columns.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid subvector access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,unaligned,TF,true>::Reference
   Subvector<VT,unaligned,TF,true>::at( size_t index )
{
   if( index >= size() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid subvector access index" );
   }
   return (*this)[index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the subvector elements.
//
// \param index Access index. The index must be smaller than the number of subvector columns.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid subvector access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,unaligned,TF,true>::ConstReference
   Subvector<VT,unaligned,TF,true>::at( size_t index ) const
{
   if( index >= size() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid subvector access index" );
   }
   return (*this)[index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the subvector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,unaligned,TF,true>::Pointer
   Subvector<VT,unaligned,TF,true>::data() noexcept
{
   return vector_.data() + offset_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the subvector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,unaligned,TF,true>::ConstPointer
   Subvector<VT,unaligned,TF,true>::data() const noexcept
{
   return vector_.data() + offset_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the subvector.
//
// \return Iterator to the first element of the subvector.
//
// This function returns an iterator to the first element of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,unaligned,TF,true>::Iterator
   Subvector<VT,unaligned,TF,true>::begin()
{
   return Iterator( vector_.begin() + offset_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the subvector.
//
// \return Iterator to the first element of the subvector.
//
// This function returns an iterator to the first element of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,unaligned,TF,true>::ConstIterator
   Subvector<VT,unaligned,TF,true>::begin() const
{
   return ConstIterator( vector_.cbegin() + offset_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the subvector.
//
// \return Iterator to the first element of the subvector.
//
// This function returns an iterator to the first element of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,unaligned,TF,true>::ConstIterator
   Subvector<VT,unaligned,TF,true>::cbegin() const
{
   return ConstIterator( vector_.cbegin() + offset_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the subvector.
//
// \return Iterator just past the last element of the subvector.
//
// This function returns an iterator just past the last element of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,unaligned,TF,true>::Iterator
   Subvector<VT,unaligned,TF,true>::end()
{
   return Iterator( vector_.begin() + offset_ + size_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the subvector.
//
// \return Iterator just past the last element of the subvector.
//
// This function returns an iterator just past the last element of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,unaligned,TF,true>::ConstIterator
   Subvector<VT,unaligned,TF,true>::end() const
{
   return ConstIterator( vector_.cbegin() + offset_ + size_, isAligned_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the subvector.
//
// \return Iterator just past the last element of the subvector.
//
// This function returns an iterator just past the last element of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,unaligned,TF,true>::ConstIterator
   Subvector<VT,unaligned,TF,true>::cend() const
{
   return ConstIterator( vector_.cbegin() + offset_ + size_, isAligned_ );
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
/*!\brief Homogenous assignment to all subvector elements.
//
// \param rhs Scalar value to be assigned to all subvector elements.
// \return Reference to the assigned subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline Subvector<VT,unaligned,TF,true>&
   Subvector<VT,unaligned,TF,true>::operator=( const ElementType& rhs )
{
   const size_t iend( offset_ + size_ );

   for( size_t i=offset_; i<iend; ++i )
      vector_[i] = rhs;

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all subvector elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to subvector.
//
// This assignment operator offers the option to directly assign to all elements of the subvector
// by means of an initializer list. The subvector elements are assigned the values from the given
// initializer list. Missing values are reset to their default state. Note that in case the size
// of the initializer list exceeds the size of the subvector, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline Subvector<VT,unaligned,TF,true>&
   Subvector<VT,unaligned,TF,true>::operator=( initializer_list<ElementType> list )
{
   if( list.size() > size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to subvector" );
   }

   std::fill( std::copy( list.begin(), list.end(), begin() ), end(), ElementType() );

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for Subvector.
//
// \param rhs Dense subvector to be copied.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Subvector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two subvectors don't match, a \a std::invalid_argument
// exception is thrown.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline Subvector<VT,unaligned,TF,true>&
   Subvector<VT,unaligned,TF,true>::operator=( const Subvector& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( &rhs == this || ( &vector_ == &rhs.vector_ && offset_ == rhs.offset_ ) )
      return *this;

   if( size() != rhs.size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Subvector sizes do not match" );
   }

   if( !tryAssign( vector_, rhs, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( rhs.canAlias( &vector_ ) ) {
      const ResultType tmp( rhs );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different vectors.
//
// \param rhs Vector to be assigned.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument
// exception is thrown.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side vector
inline Subvector<VT,unaligned,TF,true>&
   Subvector<VT,unaligned,TF,true>::operator=( const Vector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   typedef If_< IsRestricted<VT>, CompositeType_<VT2>, const VT2& >  Right;
   Right right( ~rhs );

   if( !tryAssign( vector_, right, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &vector_ ) ) {
      const ResultType_<VT2> tmp( right );
      smpAssign( left, tmp );
   }
   else {
      if( IsSparseVector<VT2>::value )
         reset();
      smpAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a vector (\f$ \vec{a}+=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be added to the dense subvector.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side vector
inline Subvector<VT,unaligned,TF,true>&
   Subvector<VT,unaligned,TF,true>::operator+=( const Vector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   typedef If_< IsRestricted<VT>, CompositeType_<VT2>, const VT2& >  Right;
   Right right( ~rhs );

   if( !tryAddAssign( vector_, right, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &vector_ ) ) {
      const ResultType_<VT2> tmp( right );
      smpAddAssign( left, tmp );
   }
   else {
      smpAddAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a vector (\f$ \vec{a}-=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be subtracted from the dense subvector.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side vector
inline Subvector<VT,unaligned,TF,true>&
   Subvector<VT,unaligned,TF,true>::operator-=( const Vector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   typedef If_< IsRestricted<VT>, CompositeType_<VT2>, const VT2& >  Right;
   Right right( ~rhs );

   if( !trySubAssign( vector_, right, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &vector_ ) ) {
      const ResultType_<VT2> tmp( right );
      smpSubAssign( left, tmp );
   }
   else {
      smpSubAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of a dense vector
//        (\f$ \vec{a}*=\vec{b} \f$).
//
// \param rhs The right-hand side dense vector to be multiplied with the dense subvector.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline Subvector<VT,unaligned,TF,true>&
   Subvector<VT,unaligned,TF,true>::operator*=( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   typedef If_< IsRestricted<VT>, CompositeType_<VT2>, const VT2& >  Right;
   Right right( ~rhs );

   if( !tryMultAssign( vector_, right, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &vector_ ) ) {
      const ResultType_<VT2> tmp( right );
      smpMultAssign( left, tmp );
   }
   else {
      smpMultAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of a sparse vector
//        (\f$ \vec{a}*=\vec{b} \f$).
//
// \param rhs The right-hand side sparse vector to be multiplied with the dense subvector.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side sparse vector
inline Subvector<VT,unaligned,TF,true>&
   Subvector<VT,unaligned,TF,true>::operator*=( const SparseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   const ResultType tmp( *this * (~rhs) );

   if( !tryAssign( vector_, tmp, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division assignment operator for the division of a dense vector (\f$ \vec{a}/=\vec{b} \f$).
//
// \param rhs The right-hand side dense vector divisor.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline Subvector<VT,unaligned,TF,true>&
   Subvector<VT,unaligned,TF,true>::operator/=( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   typedef If_< IsRestricted<VT>, CompositeType_<VT2>, const VT2& >  Right;
   Right right( ~rhs );

   if( !tryDivAssign( vector_, right, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &vector_ ) ) {
      const ResultType_<VT2> tmp( right );
      smpDivAssign( left, tmp );
   }
   else {
      smpDivAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication between a subvector and
//        a scalar value (\f$ \vec{a}*=s \f$).
//
// \param rhs The right-hand side scalar value for the multiplication.
// \return Reference to the assigned subvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, Subvector<VT,unaligned,TF,true> >&
   Subvector<VT,unaligned,TF,true>::operator*=( Other rhs )
{
   DerestrictTrait_<This> left( derestrict( *this ) );
   smpAssign( left, (*this) * rhs );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division assignment operator for the division of a subvector by a scalar value
//        (\f$ \vec{a}/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the assigned subvector.
//
// \note A division by zero is only checked by an user assert.
*/
template< typename VT       // Type of the dense vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, Subvector<VT,unaligned,TF,true> >&
   Subvector<VT,unaligned,TF,true>::operator/=( Other rhs )
{
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
/*!\brief Returns the current size/dimension of the dense subvector.
//
// \return The size of the dense subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline size_t Subvector<VT,unaligned,TF,true>::size() const noexcept
{
   return size_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense subvector.
//
// \return The capacity of the dense subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline size_t Subvector<VT,unaligned,TF,true>::capacity() const noexcept
{
   return vector_.capacity() - offset_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the subvector.
//
// \return The number of non-zero elements in the subvector.
//
// Note that the number of non-zero elements is always less than or equal to the current size
// of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline size_t Subvector<VT,unaligned,TF,true>::nonZeros() const
{
   size_t nonzeros( 0 );

   const size_t iend( offset_ + size_ );
   for( size_t i=offset_; i<iend; ++i ) {
      if( !isDefault( vector_[i] ) )
         ++nonzeros;
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
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline void Subvector<VT,unaligned,TF,true>::reset()
{
   using blaze::clear;

   const size_t iend( offset_ + size_ );
   for( size_t i=offset_; i<iend; ++i )
      clear( vector_[i] );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the dense subvector by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the subvector scaling.
// \return Reference to the dense subvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the scalar value
inline Subvector<VT,unaligned,TF,true>&
   Subvector<VT,unaligned,TF,true>::scale( const Other& scalar )
{
   const size_t iend( offset_ + size_ );
   for( size_t i=offset_; i<iend; ++i )
      vector_[i] *= scalar;
   return *this;
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
/*!\brief Returns whether the dense subvector can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense subvector, \a false if not.
//
// This function returns whether the given address can alias with the dense subvector.
// In contrast to the isAliased() function this function is allowed to use compile time
// expressions to optimize the evaluation.
*/
template< typename VT       // Type of the dense vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool Subvector<VT,unaligned,TF,true>::canAlias( const Other* alias ) const noexcept
{
   return vector_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense subvector can alias with the given dense subvector \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense subvector, \a false if not.
//
// This function returns whether the given address can alias with the dense subvector.
// In contrast to the isAliased() function this function is allowed to use compile time
// expressions to optimize the evaluation.
*/
template< typename VT   // Type of the dense vector
        , bool TF >     // Transpose flag
template< typename VT2  // Data type of the foreign dense subvector
        , bool AF2      // Alignment flag of the foreign dense subvector
        , bool TF2 >    // Transpose flag of the foreign dense subvector
inline bool Subvector<VT,unaligned,TF,true>::canAlias( const Subvector<VT2,AF2,TF2,true>* alias ) const noexcept
{
   return ( vector_.isAliased( &alias->vector_ ) &&
            ( offset_ + size_ > alias->offset_ ) && ( offset_ < alias->offset_ + alias->size_ ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense subvector is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense subvector, \a false if not.
//
// This function returns whether the given address is aliased with the dense subvector.
// In contrast to the canAlias() function this function is not allowed to use compile time
// expressions to optimize the evaluation.
*/
template< typename VT       // Type of the dense vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool Subvector<VT,unaligned,TF,true>::isAliased( const Other* alias ) const noexcept
{
   return vector_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense subvector is aliased with the given dense subvector \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense subvector, \a false if not.
//
// This function returns whether the given address is aliased with the dense subvector.
// In contrast to the canAlias() function this function is not allowed to use compile time
// expressions to optimize the evaluation.
*/
template< typename VT   // Type of the dense vector
        , bool TF >     // Transpose flag
template< typename VT2  // Data type of the foreign dense subvector
        , bool AF2      // Alignment flag of the foreign dense subvector
        , bool TF2 >    // Transpose flag of the foreign dense subvector
inline bool Subvector<VT,unaligned,TF,true>::isAliased( const Subvector<VT2,AF2,TF2,true>* alias ) const noexcept
{
   return ( vector_.isAliased( &alias->vector_ ) &&
            ( offset_ + size_ > alias->offset_ ) && ( offset_ < alias->offset_ + alias->size_ ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the subvector is properly aligned in memory.
//
// \return \a true in case the subvector is aligned, \a false if not.
//
// This function returns whether the subvector is guaranteed to be properly aligned in memory,
// i.e. whether the beginning and the end of the subvector are guaranteed to conform to the
// alignment restrictions of the underlying element type.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline bool Subvector<VT,unaligned,TF,true>::isAligned() const noexcept
{
   return isAligned_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the subvector can be used in SMP assignments.
//
// \return \a true in case the subvector can be used in SMP assignments, \a false if not.
//
// This function returns whether the subvector can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current size of the
// subvector).
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline bool Subvector<VT,unaligned,TF,true>::canSMPAssign() const noexcept
{
   return ( size() > SMP_DVECASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the dense subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense subvector.
// The index must be smaller than the number of subvector elements and it must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
BLAZE_ALWAYS_INLINE typename Subvector<VT,unaligned,TF,true>::SIMDType
   Subvector<VT,unaligned,TF,true>::load( size_t index ) const noexcept
{
   if( isAligned_ )
      return loada( index );
   else
      return loadu( index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the dense subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense subvector.
// The index must be smaller than the number of subvector elements and it must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
BLAZE_ALWAYS_INLINE typename Subvector<VT,unaligned,TF,true>::SIMDType
   Subvector<VT,unaligned,TF,true>::loada( size_t index ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( index < size()            , "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size(), "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL   , "Invalid subvector access index" );

   return vector_.loada( offset_+index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned load of a SIMD element of the dense subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense
// subvector. The index must be smaller than the number of subvector elements and it must be
// a multiple of the number of values inside the SIMD element. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
BLAZE_ALWAYS_INLINE typename Subvector<VT,unaligned,TF,true>::SIMDType
   Subvector<VT,unaligned,TF,true>::loadu( size_t index ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( index < size()            , "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size(), "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL   , "Invalid subvector access index" );

   return vector_.loadu( offset_+index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned store of a SIMD element of the subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store a specific SIMD element of the dense subvector.
// The index must be smaller than the number of subvector elements and it must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
BLAZE_ALWAYS_INLINE void
   Subvector<VT,unaligned,TF,true>::store( size_t index, const SIMDType& value ) noexcept
{
   if( isAligned_ )
      storea( index, value );
   else
      storeu( index, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned store of a SIMD element of the subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store a specific SIMD element of the dense subvector.
// The index must be smaller than the number of subvector elements and it must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
BLAZE_ALWAYS_INLINE void
   Subvector<VT,unaligned,TF,true>::storea( size_t index, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( index < size()            , "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size(), "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL   , "Invalid subvector access index" );

   vector_.storea( offset_+index, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned store of a SIMD element of the subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store a specific SIMD element of the dense subvector.
// The index must be smaller than the number of subvector elements and it must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
BLAZE_ALWAYS_INLINE void
   Subvector<VT,unaligned,TF,true>::storeu( size_t index, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( index < size()            , "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size(), "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL   , "Invalid subvector access index" );

   vector_.storeu( offset_+index, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned, non-temporal store of a SIMD element of the subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store a specific SIMD element of the
// dense subvector. The index must be smaller than the number of subvector elements and it
// must be a multiple of the number of values inside the SIMD element. This function
// must \b NOT be called explicitly! It is used internally for the performance optimized
// evaluation of expression templates. Calling this function explicitly might result in
// erroneous results and/or in compilation errors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
BLAZE_ALWAYS_INLINE void
   Subvector<VT,unaligned,TF,true>::stream( size_t index, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( index < size()            , "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size(), "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL   , "Invalid subvector access index" );

   if( isAligned_ )
      vector_.stream( offset_+index, value );
   else
      vector_.storeu( offset_+index, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline DisableIf_< typename Subvector<VT,unaligned,TF,true>::BLAZE_TEMPLATE VectorizedAssign<VT2> >
   Subvector<VT,unaligned,TF,true>::assign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      vector_[offset_+i    ] = (~rhs)[i    ];
      vector_[offset_+i+1UL] = (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      vector_[offset_+ipos] = (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline EnableIf_< typename Subvector<VT,unaligned,TF,true>::BLAZE_TEMPLATE VectorizedAssign<VT2> >
   Subvector<VT,unaligned,TF,true>::assign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT2> right( (~rhs).begin() );

   if( useStreaming && isAligned_ &&
       ( size_ > ( cacheSize/( sizeof(ElementType) * 3UL ) ) ) &&
       !(~rhs).isAliased( &vector_ ) )
   {
      for( ; i<ipos; i+=SIMDSIZE ) {
         left.stream( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<size_; ++i ) {
         *left = *right;
      }
   }
   else
   {
      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
         left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<ipos; i+=SIMDSIZE ) {
         left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<size_; ++i ) {
         *left = *right; ++left; ++right;
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side sparse vector
inline void Subvector<VT,unaligned,TF,true>::assign( const SparseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   for( ConstIterator_<VT2> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      vector_[offset_+element->index()] = element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline DisableIf_< typename Subvector<VT,unaligned,TF,true>::BLAZE_TEMPLATE VectorizedAddAssign<VT2> >
   Subvector<VT,unaligned,TF,true>::addAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      vector_[offset_+i    ] += (~rhs)[i    ];
      vector_[offset_+i+1UL] += (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      vector_[offset_+ipos] += (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the addition assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline EnableIf_< typename Subvector<VT,unaligned,TF,true>::BLAZE_TEMPLATE VectorizedAddAssign<VT2> >
   Subvector<VT,unaligned,TF,true>::addAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT2> right( (~rhs).begin() );

   for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE ) {
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<size_; ++i ) {
      *left += *right; ++left; ++right;
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side sparse vector
inline void Subvector<VT,unaligned,TF,true>::addAssign( const SparseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   for( ConstIterator_<VT2> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      vector_[offset_+element->index()] += element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline DisableIf_< typename Subvector<VT,unaligned,TF,true>::BLAZE_TEMPLATE VectorizedSubAssign<VT2> >
   Subvector<VT,unaligned,TF,true>::subAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      vector_[offset_+i    ] -= (~rhs)[i    ];
      vector_[offset_+i+1UL] -= (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      vector_[offset_+ipos] -= (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the subtraction assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline EnableIf_< typename Subvector<VT,unaligned,TF,true>::BLAZE_TEMPLATE VectorizedSubAssign<VT2> >
   Subvector<VT,unaligned,TF,true>::subAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT2> right( (~rhs).begin() );

   for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE ) {
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<size_; ++i ) {
      *left -= *right; ++left; ++right;
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side sparse vector
inline void Subvector<VT,unaligned,TF,true>::subAssign( const SparseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   for( ConstIterator_<VT2> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      vector_[offset_+element->index()] -= element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the multiplication assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be multiplied.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline DisableIf_< typename Subvector<VT,unaligned,TF,true>::BLAZE_TEMPLATE VectorizedMultAssign<VT2> >
   Subvector<VT,unaligned,TF,true>::multAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      vector_[offset_+i    ] *= (~rhs)[i    ];
      vector_[offset_+i+1UL] *= (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      vector_[offset_+ipos] *= (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the multiplication assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be multiplied.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline EnableIf_< typename Subvector<VT,unaligned,TF,true>::BLAZE_TEMPLATE VectorizedMultAssign<VT2> >
   Subvector<VT,unaligned,TF,true>::multAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT2> right( (~rhs).begin() );

   for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE ) {
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<size_; ++i ) {
      *left *= *right; ++left; ++right;
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the multiplication assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be multiplied.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side sparse vector
inline void Subvector<VT,unaligned,TF,true>::multAssign( const SparseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const ResultType tmp( serial( *this ) );

   reset();

   for( ConstIterator_<VT2> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      vector_[offset_+element->index()] = tmp[element->index()] * element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the division assignment of a dense vector.
//
// \param rhs The right-hand side dense vector divisor.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline DisableIf_< typename Subvector<VT,unaligned,TF,true>::BLAZE_TEMPLATE VectorizedDivAssign<VT2> >
   Subvector<VT,unaligned,TF,true>::divAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      vector_[offset_+i    ] /= (~rhs)[i    ];
      vector_[offset_+i+1UL] /= (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      vector_[offset_+ipos] /= (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the division assignment of a dense vector.
//
// \param rhs The right-hand side dense vector divisor.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline EnableIf_< typename Subvector<VT,unaligned,TF,true>::BLAZE_TEMPLATE VectorizedDivAssign<VT2> >
   Subvector<VT,unaligned,TF,true>::divAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT2> right( (~rhs).begin() );

   for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
      left.store( left.load() / right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() / right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() / right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() / right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE ) {
      left.store( left.load() / right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<size_; ++i ) {
      *left /= *right; ++left; ++right;
   }
}
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ALIGNED DENSE SUBVECTORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Subvector for aligned dense subvectors.
// \ingroup subvector
//
// This specialization of Subvector adapts the class template to the requirements of aligned
// dense subvectors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
class Subvector<VT,aligned,TF,true>
   : public DenseVector< Subvector<VT,aligned,TF,true>, TF >
   , private View
{
 private:
   //**Type definitions****************************************************************************
   //! Composite data type of the dense vector expression.
   typedef If_< IsExpression<VT>, VT, VT& >  Operand;
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef Subvector<VT,aligned,TF,true>  This;           //!< Type of this Subvector instance.
   typedef DenseVector<This,TF>           BaseType;       //!< Base type of this Subvector instance.
   typedef SubvectorTrait_<VT>            ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>     TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<VT>               ElementType;    //!< Type of the subvector elements.
   typedef SIMDTrait_<ElementType>        SIMDType;       //!< SIMD type of the subvector elements.
   typedef ReturnType_<VT>                ReturnType;     //!< Return type for expression template evaluations
   typedef const Subvector&               CompositeType;  //!< Data type for composite expression templates.

   //! Reference to a constant subvector value.
   typedef ConstReference_<VT>  ConstReference;

   //! Reference to a non-constant subvector value.
   typedef If_< IsConst<VT>, ConstReference, Reference_<VT> >  Reference;

   //! Pointer to a constant subvector value.
   typedef const ElementType*  ConstPointer;

   //! Pointer to a non-constant subvector value.
   typedef If_< Or< IsConst<VT>, Not< HasMutableDataAccess<VT> > >, ConstPointer, ElementType* >  Pointer;

   //! Iterator over constant elements.
   typedef ConstIterator_<VT>  ConstIterator;

   //! Iterator over non-constant elements.
   typedef If_< IsConst<VT>, ConstIterator, Iterator_<VT> >  Iterator;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = VT::simdEnabled };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = VT::smpAssignable };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline Subvector( Operand vector, size_t index, size_t n );
   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator[]( size_t index );
   inline ConstReference operator[]( size_t index ) const;
   inline Reference      at( size_t index );
   inline ConstReference at( size_t index ) const;
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Iterator       begin ();
   inline ConstIterator  begin () const;
   inline ConstIterator  cbegin() const;
   inline Iterator       end   ();
   inline ConstIterator  end   () const;
   inline ConstIterator  cend  () const;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
                            inline Subvector& operator= ( const ElementType& rhs );
                            inline Subvector& operator= ( initializer_list<ElementType> list );
                            inline Subvector& operator= ( const Subvector& rhs );
   template< typename VT2 > inline Subvector& operator= ( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline Subvector& operator+=( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline Subvector& operator-=( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline Subvector& operator*=( const DenseVector<VT2,TF>&  rhs );
   template< typename VT2 > inline Subvector& operator*=( const SparseVector<VT2,TF>& rhs );
   template< typename VT2 > inline Subvector& operator/=( const DenseVector<VT2,TF>&  rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, Subvector >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, Subvector >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t     size() const noexcept;
                              inline size_t     capacity() const noexcept;
                              inline size_t     nonZeros() const;
                              inline void       reset();
   template< typename Other > inline Subvector& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT2 >
   struct VectorizedAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<VT2> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT2 >
   struct VectorizedAddAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<VT2> >::value &&
                            HasSIMDAdd< ElementType, ElementType_<VT2> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT2 >
   struct VectorizedSubAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<VT2> >::value &&
                            HasSIMDSub< ElementType, ElementType_<VT2> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT2 >
   struct VectorizedMultAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<VT2> >::value &&
                            HasSIMDMult< ElementType, ElementType_<VT2> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT2 >
   struct VectorizedDivAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT2::simdEnabled &&
                            AreSIMDCombinable< ElementType, ElementType_<VT2> >::value &&
                            HasSIMDDiv< ElementType, ElementType_<VT2> >::value };
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

   template< typename VT2, bool AF2, bool TF2 >
   inline bool canAlias( const Subvector<VT2,AF2,TF2,true>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< typename VT2, bool AF2, bool TF2 >
   inline bool isAliased( const Subvector<VT2,AF2,TF2,true>* alias ) const noexcept;

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t index ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t index ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t index ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t index, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t index, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t index, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t index, const SIMDType& value ) noexcept;

   template< typename VT2 >
   inline DisableIf_< VectorizedAssign<VT2> > assign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 >
   inline EnableIf_< VectorizedAssign<VT2> > assign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 > inline void assign( const SparseVector<VT2,TF>& rhs );

   template< typename VT2 >
   inline DisableIf_< VectorizedAddAssign<VT2> > addAssign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 >
   inline EnableIf_< VectorizedAddAssign<VT2> > addAssign ( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 > inline void addAssign( const SparseVector<VT2,TF>& rhs );

   template< typename VT2 >
   inline DisableIf_< VectorizedSubAssign<VT2> > subAssign ( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 >
   inline EnableIf_< VectorizedSubAssign<VT2> > subAssign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 > inline void subAssign( const SparseVector<VT2,TF>& rhs );

   template< typename VT2 >
   inline DisableIf_< VectorizedMultAssign<VT2> > multAssign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 >
   inline EnableIf_< VectorizedMultAssign<VT2> > multAssign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 > inline void multAssign( const SparseVector<VT2,TF>& rhs );

   template< typename VT2 >
   inline DisableIf_< VectorizedDivAssign<VT2> > divAssign( const DenseVector <VT2,TF>& rhs );

   template< typename VT2 >
   inline EnableIf_< VectorizedDivAssign<VT2> > divAssign( const DenseVector <VT2,TF>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand      vector_;  //!< The dense vector containing the subvector.
   const size_t offset_;  //!< The offset of the subvector within the dense vector.
   const size_t size_;    //!< The size of the subvector.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename VT2, bool AF2, bool TF2, bool DF2 > friend class Subvector;

   template< bool AF1, typename VT2, bool AF2, bool TF2, bool DF2 >
   friend const Subvector<VT2,AF1,TF2,DF2>
      subvector( const Subvector<VT2,AF2,TF2,DF2>& sv, size_t index, size_t size );

   template< typename VT2, bool AF2, bool TF2, bool DF2 >
   friend bool isIntact( const Subvector<VT2,AF2,TF2,DF2>& sv ) noexcept;

   template< typename VT2, bool AF2, bool TF2, bool DF2 >
   friend bool isSame( const Subvector<VT2,AF2,TF2,DF2>& a, const Vector<VT2,TF2>& b ) noexcept;

   template< typename VT2, bool AF2, bool TF2, bool DF2 >
   friend bool isSame( const Vector<VT2,TF2>& a, const Subvector<VT2,AF2,TF2,DF2>& b ) noexcept;

   template< typename VT2, bool AF2, bool TF2, bool DF2 >
   friend bool isSame( const Subvector<VT2,AF2,TF2,DF2>& a, const Subvector<VT2,AF2,TF2,DF2>& b ) noexcept;

   template< typename VT2, bool AF2, bool TF2, bool DF2, typename VT3 >
   friend bool tryAssign( const Subvector<VT2,AF2,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT2, bool AF2, bool TF2, bool DF2, typename VT3 >
   friend bool tryAddAssign( const Subvector<VT2,AF2,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT2, bool AF2, bool TF2, bool DF2, typename VT3 >
   friend bool trySubAssign( const Subvector<VT2,AF2,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT2, bool AF2, bool TF2, bool DF2, typename VT3 >
   friend bool tryMultAssign( const Subvector<VT2,AF2,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT2, bool AF2, bool TF2, bool DF2 >
   friend DerestrictTrait_< Subvector<VT2,AF2,TF2,DF2> > derestrict( Subvector<VT2,AF2,TF2,DF2>& sv );
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE   ( VT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( VT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE  ( VT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SUBVECTOR_TYPE  ( VT );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( VT, TF );
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
/*!\brief The constructor for Subvector.
//
// \param vector The dense vector containing the subvector.
// \param index The first index of the subvector in the given vector.
// \param n The size of the subvector.
// \exception std::invalid_argument Invalid subvector specification.
//
// In case the subvector is not properly specified (i.e. if the specified first index is larger
// than the size of the given vector or the subvector is specified beyond the size of the vector)
// a \a std::invalid_argument exception is thrown.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline Subvector<VT,aligned,TF,true>::Subvector( Operand vector, size_t index, size_t n )
   : vector_( vector )  // The vector containing the subvector
   , offset_( index  )  // The offset of the subvector within the dense vector
   , size_  ( n      )  // The size of the subvector
{
   if( index + n > vector.size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid subvector specification" );
   }

   if( simdEnabled && vector_.data() != nullptr && !checkAlignment( data() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid subvector alignment" );
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
/*!\brief Subscript operator for the direct access to the subvector elements.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \return Reference to the accessed value.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,aligned,TF,true>::Reference
   Subvector<VT,aligned,TF,true>::operator[]( size_t index )
{
   BLAZE_USER_ASSERT( index < size(), "Invalid subvector access index" );
   return vector_[offset_+index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subscript operator for the direct access to the subvector elements.
//
// \param index Access index. The index must be smaller than the number of subvector columns.
// \return Reference to the accessed value.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,aligned,TF,true>::ConstReference
   Subvector<VT,aligned,TF,true>::operator[]( size_t index ) const
{
   BLAZE_USER_ASSERT( index < size(), "Invalid subvector access index" );
   return const_cast<const VT&>( vector_ )[offset_+index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the subvector elements.
//
// \param index Access index. The index must be smaller than the number of subvector columns.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid subvector access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,aligned,TF,true>::Reference
   Subvector<VT,aligned,TF,true>::at( size_t index )
{
   if( index >= size() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid subvector access index" );
   }
   return (*this)[index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the subvector elements.
//
// \param index Access index. The index must be smaller than the number of subvector columns.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid subvector access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,aligned,TF,true>::ConstReference
   Subvector<VT,aligned,TF,true>::at( size_t index ) const
{
   if( index >= size() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid subvector access index" );
   }
   return (*this)[index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the subvector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,aligned,TF,true>::Pointer Subvector<VT,aligned,TF,true>::data() noexcept
{
   return vector_.data() + offset_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the subvector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,aligned,TF,true>::ConstPointer
   Subvector<VT,aligned,TF,true>::data() const noexcept
{
   return vector_.data() + offset_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the subvector.
//
// \return Iterator to the first element of the subvector.
//
// This function returns an iterator to the first element of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,aligned,TF,true>::Iterator Subvector<VT,aligned,TF,true>::begin()
{
   return ( vector_.begin() + offset_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the subvector.
//
// \return Iterator to the first element of the subvector.
//
// This function returns an iterator to the first element of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,aligned,TF,true>::ConstIterator
   Subvector<VT,aligned,TF,true>::begin() const
{
   return ( vector_.cbegin() + offset_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the subvector.
//
// \return Iterator to the first element of the subvector.
//
// This function returns an iterator to the first element of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,aligned,TF,true>::ConstIterator
   Subvector<VT,aligned,TF,true>::cbegin() const
{
   return ( vector_.cbegin() + offset_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the subvector.
//
// \return Iterator just past the last element of the subvector.
//
// This function returns an iterator just past the last element of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,aligned,TF,true>::Iterator Subvector<VT,aligned,TF,true>::end()
{
   return ( vector_.begin() + offset_ + size_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the subvector.
//
// \return Iterator just past the last element of the subvector.
//
// This function returns an iterator just past the last element of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,aligned,TF,true>::ConstIterator
   Subvector<VT,aligned,TF,true>::end() const
{
   return ( vector_.cbegin() + offset_ + size_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the subvector.
//
// \return Iterator just past the last element of the subvector.
//
// This function returns an iterator just past the last element of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline typename Subvector<VT,aligned,TF,true>::ConstIterator
   Subvector<VT,aligned,TF,true>::cend() const
{
   return ( vector_.cbegin() + offset_ + size_ );
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
/*!\brief Homogenous assignment to all subvector elements.
//
// \param rhs Scalar value to be assigned to all subvector elements.
// \return Reference to the assigned subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline Subvector<VT,aligned,TF,true>&
   Subvector<VT,aligned,TF,true>::operator=( const ElementType& rhs )
{
   const size_t iend( offset_ + size_ );

   for( size_t i=offset_; i<iend; ++i )
      vector_[i] = rhs;

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all subvector elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to subvector.
//
// This assignment operator offers the option to directly assign to all elements of the subvector
// by means of an initializer list. The subvector elements are assigned the values from the given
// initializer list. Missing values are reset to their default state. Note that in case the size
// of the initializer list exceeds the size of the subvector, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline Subvector<VT,aligned,TF,true>&
   Subvector<VT,aligned,TF,true>::operator=( initializer_list<ElementType> list )
{
   if( list.size() > size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to subvector" );
   }

   std::fill( std::copy( list.begin(), list.end(), begin() ), end(), ElementType() );

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for Subvector.
//
// \param rhs Dense subvector to be copied.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Subvector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two subvectors don't match, a \a std::invalid_argument
// exception is thrown.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline Subvector<VT,aligned,TF,true>&
   Subvector<VT,aligned,TF,true>::operator=( const Subvector& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( &rhs == this || ( &vector_ == &rhs.vector_ && offset_ == rhs.offset_ ) )
      return *this;

   if( size() != rhs.size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Subvector sizes do not match" );
   }

   if( !tryAssign( vector_, rhs, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( rhs.canAlias( &vector_ ) ) {
      const ResultType tmp( ~rhs );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different vectors.
//
// \param rhs Vector to be assigned.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument
// exception is thrown.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side vector
inline Subvector<VT,aligned,TF,true>&
   Subvector<VT,aligned,TF,true>::operator=( const Vector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   typedef If_< IsRestricted<VT>, CompositeType_<VT2>, const VT2& >  Right;
   Right right( ~rhs );

   if( !tryAssign( vector_, right, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &vector_ ) ) {
      const ResultType_<VT2> tmp( right );
      smpAssign( left, tmp );
   }
   else {
      if( IsSparseVector<VT2>::value )
         reset();
      smpAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a vector (\f$ \vec{a}+=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be added to the dense subvector.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side vector
inline Subvector<VT,aligned,TF,true>&
   Subvector<VT,aligned,TF,true>::operator+=( const Vector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   typedef If_< IsRestricted<VT>, CompositeType_<VT2>, const VT2& >  Right;
   Right right( ~rhs );

   if( !tryAddAssign( vector_, right, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &vector_ ) ) {
      const ResultType_<VT2> tmp( right );
      smpAddAssign( left, tmp );
   }
   else {
      smpAddAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a vector (\f$ \vec{a}-=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be subtracted from the dense subvector.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side vector
inline Subvector<VT,aligned,TF,true>&
   Subvector<VT,aligned,TF,true>::operator-=( const Vector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   typedef If_< IsRestricted<VT>, CompositeType_<VT2>, const VT2& >  Right;
   Right right( ~rhs );

   if( !trySubAssign( vector_, right, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &vector_ ) ) {
      const ResultType_<VT2> tmp( right );
      smpSubAssign( left, tmp );
   }
   else {
      smpSubAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of a dense vector
//        (\f$ \vec{a}*=\vec{b} \f$).
//
// \param rhs The right-hand side dense vector to be multiplied with the dense subvector.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline Subvector<VT,aligned,TF,true>&
   Subvector<VT,aligned,TF,true>::operator*=( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   typedef If_< IsRestricted<VT>, CompositeType_<VT2>, const VT2& >  Right;
   Right right( ~rhs );

   if( !tryMultAssign( vector_, right, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &vector_ ) ) {
      const ResultType_<VT2> tmp( right );
      smpMultAssign( left, tmp );
   }
   else {
      smpMultAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of a sparse vector
//        (\f$ \vec{a}*=\vec{b} \f$).
//
// \param rhs The right-hand side sparse vector to be multiplied with the dense subvector.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side sparse vector
inline Subvector<VT,aligned,TF,true>&
   Subvector<VT,aligned,TF,true>::operator*=( const SparseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   const ResultType tmp( *this * (~rhs) );

   if( !tryAssign( vector_, tmp, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division assignment operator for the division of a dense vector (\f$ \vec{a}/=\vec{b} \f$).
//
// \param rhs The right-hand side dense vector divisor.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline Subvector<VT,aligned,TF,true>&
   Subvector<VT,aligned,TF,true>::operator/=( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   typedef If_< IsRestricted<VT>, CompositeType_<VT2>, const VT2& >  Right;
   Right right( ~rhs );

   if( !tryDivAssign( vector_, right, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( IsReference<Right>::value && right.canAlias( &vector_ ) ) {
      const ResultType_<VT2> tmp( right );
      smpDivAssign( left, tmp );
   }
   else {
      smpDivAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication between a subvector and
//        a scalar value (\f$ \vec{a}*=s \f$).
//
// \param rhs The right-hand side scalar value for the multiplication.
// \return Reference to the assigned subvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, Subvector<VT,aligned,TF,true> >&
   Subvector<VT,aligned,TF,true>::operator*=( Other rhs )
{
   DerestrictTrait_<This> left( derestrict( *this ) );
   smpAssign( left, (*this) * rhs );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division assignment operator for the division of a subvector by a scalar value
//        (\f$ \vec{a}/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the assigned subvector.
//
// \note A division by zero is only checked by an user assert.
*/
template< typename VT       // Type of the dense vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, Subvector<VT,aligned,TF,true> >&
   Subvector<VT,aligned,TF,true>::operator/=( Other rhs )
{
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
/*!\brief Returns the current size/dimension of the dense subvector.
//
// \return The size of the dense subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline size_t Subvector<VT,aligned,TF,true>::size() const noexcept
{
   return size_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense subvector.
//
// \return The capacity of the dense subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline size_t Subvector<VT,aligned,TF,true>::capacity() const noexcept
{
   return vector_.capacity() - offset_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the subvector.
//
// \return The number of non-zero elements in the subvector.
//
// Note that the number of non-zero elements is always less than or equal to the current size
// of the subvector.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline size_t Subvector<VT,aligned,TF,true>::nonZeros() const
{
   size_t nonzeros( 0 );

   const size_t iend( offset_ + size_ );
   for( size_t i=offset_; i<iend; ++i ) {
      if( !isDefault( vector_[i] ) )
         ++nonzeros;
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
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline void Subvector<VT,aligned,TF,true>::reset()
{
   using blaze::clear;

   const size_t iend( offset_ + size_ );
   for( size_t i=offset_; i<iend; ++i )
      clear( vector_[i] );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the dense subvector by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the subvector scaling.
// \return Reference to the dense subvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the scalar value
inline Subvector<VT,aligned,TF,true>& Subvector<VT,aligned,TF,true>::scale( const Other& scalar )
{
   const size_t iend( offset_ + size_ );
   for( size_t i=offset_; i<iend; ++i )
      vector_[i] *= scalar;
   return *this;
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
/*!\brief Returns whether the dense subvector can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense subvector, \a false if not.
//
// This function returns whether the given address can alias with the dense subvector.
// In contrast to the isAliased() function this function is allowed to use compile time
// expressions to optimize the evaluation.
*/
template< typename VT       // Type of the dense vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool Subvector<VT,aligned,TF,true>::canAlias( const Other* alias ) const noexcept
{
   return vector_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense subvector can alias with the given dense subvector \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense subvector, \a false if not.
//
// This function returns whether the given address can alias with the dense subvector.
// In contrast to the isAliased() function this function is allowed to use compile time
// expressions to optimize the evaluation.
*/
template< typename VT   // Type of the dense vector
        , bool TF >     // Transpose flag
template< typename VT2  // Data type of the foreign dense subvector
        , bool AF2      // Alignment flag of the foreign dense subvector
        , bool TF2 >    // Transpose flag of the foreign dense subvector
inline bool Subvector<VT,aligned,TF,true>::canAlias( const Subvector<VT2,AF2,TF2,true>* alias ) const noexcept
{
   return ( vector_.isAliased( &alias->vector_ ) &&
            ( offset_ + size_ > alias->offset_ ) && ( offset_ < alias->offset_ + alias->size_ ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense subvector is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense subvector, \a false if not.
//
// This function returns whether the given address is aliased with the dense subvector.
// In contrast to the canAlias() function this function is not allowed to use compile time
// expressions to optimize the evaluation.
*/
template< typename VT       // Type of the dense vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool Subvector<VT,aligned,TF,true>::isAliased( const Other* alias ) const noexcept
{
   return vector_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense subvector is aliased with the given dense subvector \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense subvector, \a false if not.
//
// This function returns whether the given address is aliased with the dense subvector.
// In contrast to the canAlias() function this function is not allowed to use compile time
// expressions to optimize the evaluation.
*/
template< typename VT   // Type of the dense vector
        , bool TF >     // Transpose flag
template< typename VT2  // Data type of the foreign dense subvector
        , bool AF2      // Alignment flag of the foreign dense subvector
        , bool TF2 >    // Transpose flag of the foreign dense subvector
inline bool Subvector<VT,aligned,TF,true>::isAliased( const Subvector<VT2,AF2,TF2,true>* alias ) const noexcept
{
   return ( vector_.isAliased( &alias->vector_ ) &&
            ( offset_ + size_ > alias->offset_ ) && ( offset_ < alias->offset_ + alias->size_ ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the subvector is properly aligned in memory.
//
// \return \a true in case the subvector is aligned, \a false if not.
//
// This function returns whether the subvector is guaranteed to be properly aligned in memory,
// i.e. whether the beginning and the end of the subvector are guaranteed to conform to the
// alignment restrictions of the underlying element type.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline bool Subvector<VT,aligned,TF,true>::isAligned() const noexcept
{
   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the subvector can be used in SMP assignments.
//
// \return \a true in case the subvector can be used in SMP assignments, \a false if not.
//
// This function returns whether the subvector can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current size of the
// subvector).
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline bool Subvector<VT,aligned,TF,true>::canSMPAssign() const noexcept
{
   return ( size() > SMP_DVECASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Load of a SIMD element of the dense subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense subvector. The index
// must be smaller than the number of subvector elements and it must be a multiple of the
// number of values inside the SIMD element. This function must \b NOT be called explicitly!
// It is used internally for the performance optimized evaluation of expression templates.
// Calling this function explicitly might result in erroneous results and/or in compilation
// errors.
*/
template< typename VT       // Type of the dense vector
        , bool TF >         // Transpose flag
BLAZE_ALWAYS_INLINE typename Subvector<VT,aligned,TF,true>::SIMDType
   Subvector<VT,aligned,TF,true>::load( size_t index ) const noexcept
{
   return loada( index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the dense subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense subvector.
// The index must be smaller than the number of subvector elements and it must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename VT       // Type of the dense vector
        , bool TF >         // Transpose flag
BLAZE_ALWAYS_INLINE typename Subvector<VT,aligned,TF,true>::SIMDType
   Subvector<VT,aligned,TF,true>::loada( size_t index ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( index < size()            , "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size(), "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL   , "Invalid subvector access index" );

   return vector_.loada( offset_+index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned load of a SIMD element of the dense subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense subvector.
// The index must be smaller than the number of subvector elements and it must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
BLAZE_ALWAYS_INLINE typename Subvector<VT,aligned,TF,true>::SIMDType
   Subvector<VT,aligned,TF,true>::loadu( size_t index ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( index < size()            , "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size(), "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL   , "Invalid subvector access index" );

   return vector_.loadu( offset_+index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Store of a SIMD element of the subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store a specific SIMD element of the dense subvector. The index
// must be smaller than the number of subvector elements and it must be a multiple of the
// number of values inside the SIMD element. This function must \b NOT be called explicitly!
// It is used internally for the performance optimized evaluation of expression templates.
// Calling this function explicitly might result in erroneous results and/or in compilation
// errors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
BLAZE_ALWAYS_INLINE void
   Subvector<VT,aligned,TF,true>::store( size_t index, const SIMDType& value ) noexcept
{
   storea( index, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned store of a SIMD element of the subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store a specific SIMD element of the dense subvector.
// The index must be smaller than the number of subvector elements and it must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
BLAZE_ALWAYS_INLINE void
   Subvector<VT,aligned,TF,true>::storea( size_t index, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( index < size()            , "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size(), "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL   , "Invalid subvector access index" );

   vector_.storea( offset_+index, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned store of a SIMD element of the subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store a specific SIMD element of the dense subvector.
// The index must be smaller than the number of subvector elements and it must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
BLAZE_ALWAYS_INLINE void
   Subvector<VT,aligned,TF,true>::storeu( size_t index, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( index < size()            , "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size(), "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL   , "Invalid subvector access index" );

   vector_.storeu( offset_+index, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned, non-temporal store of a SIMD element of the subvector.
//
// \param index Access index. The index must be smaller than the number of subvector elements.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store a specific SIMD element of the
// dense subvector. The index must be smaller than the number of subvector elements and it
// must be a multiple of the number of values inside the SIMD element. This function
// must \b NOT be called explicitly! It is used internally for the performance optimized
// evaluation of expression templates. Calling this function explicitly might result in
// erroneous results and/or in compilation errors.
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
BLAZE_ALWAYS_INLINE void
   Subvector<VT,aligned,TF,true>::stream( size_t index, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( index < size()            , "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size(), "Invalid subvector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL   , "Invalid subvector access index" );

   vector_.stream( offset_+index, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline DisableIf_< typename Subvector<VT,aligned,TF,true>::BLAZE_TEMPLATE VectorizedAssign<VT2> >
   Subvector<VT,aligned,TF,true>::assign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      vector_[offset_+i    ] = (~rhs)[i    ];
      vector_[offset_+i+1UL] = (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      vector_[offset_+ipos] = (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline EnableIf_< typename Subvector<VT,aligned,TF,true>::BLAZE_TEMPLATE VectorizedAssign<VT2> >
   Subvector<VT,aligned,TF,true>::assign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT2> right( (~rhs).begin() );

   if( useStreaming && size_ > ( cacheSize/( sizeof(ElementType) * 3UL ) ) && !(~rhs).isAliased( &vector_ ) )
   {
      for( ; i<ipos; i+=SIMDSIZE ) {
         left.stream( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<size_; ++i ) {
         *left = *right; ++left; ++right;
      }
   }
   else
   {
      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
         left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<ipos; i+=SIMDSIZE ) {
         left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; i<size_; ++i ) {
         *left = *right; ++left; ++right;
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side sparse vector
inline void Subvector<VT,aligned,TF,true>::assign( const SparseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   for( ConstIterator_<VT2> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      vector_[offset_+element->index()] = element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline DisableIf_< typename Subvector<VT,aligned,TF,true>::BLAZE_TEMPLATE VectorizedAddAssign<VT2> >
   Subvector<VT,aligned,TF,true>::addAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      vector_[offset_+i    ] += (~rhs)[i    ];
      vector_[offset_+i+1UL] += (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      vector_[offset_+ipos] += (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the addition assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline EnableIf_< typename Subvector<VT,aligned,TF,true>::BLAZE_TEMPLATE VectorizedAddAssign<VT2> >
   Subvector<VT,aligned,TF,true>::addAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT2> right( (~rhs).begin() );

   for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE ) {
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<size_; ++i ) {
      *left += *right; ++left; ++right;
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side sparse vector
inline void Subvector<VT,aligned,TF,true>::addAssign( const SparseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   for( ConstIterator_<VT2> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      vector_[offset_+element->index()] += element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline DisableIf_< typename Subvector<VT,aligned,TF,true>::BLAZE_TEMPLATE VectorizedSubAssign<VT2> >
   Subvector<VT,aligned,TF,true>::subAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      vector_[offset_+i    ] -= (~rhs)[i    ];
      vector_[offset_+i+1UL] -= (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      vector_[offset_+ipos] -= (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the subtraction assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline EnableIf_< typename Subvector<VT,aligned,TF,true>::BLAZE_TEMPLATE VectorizedSubAssign<VT2> >
   Subvector<VT,aligned,TF,true>::subAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT2> right( (~rhs).begin() );

   for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE ) {
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<size_; ++i ) {
      *left -= *right; ++left; ++right;
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side sparse vector
inline void Subvector<VT,aligned,TF,true>::subAssign( const SparseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   for( ConstIterator_<VT2> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      vector_[offset_+element->index()] -= element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the multiplication assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be multiplied.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline DisableIf_< typename Subvector<VT,aligned,TF,true>::BLAZE_TEMPLATE VectorizedMultAssign<VT2> >
   Subvector<VT,aligned,TF,true>::multAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      vector_[offset_+i    ] *= (~rhs)[i    ];
      vector_[offset_+i+1UL] *= (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      vector_[offset_+ipos] *= (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the multiplication assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be multiplied.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline EnableIf_< typename Subvector<VT,aligned,TF,true>::BLAZE_TEMPLATE VectorizedMultAssign<VT2> >
   Subvector<VT,aligned,TF,true>::multAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT2> right( (~rhs).begin() );

   for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE ) {
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<size_; ++i ) {
      *left *= *right; ++left; ++right;
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the multiplication assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be multiplied.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side sparse vector
inline void Subvector<VT,aligned,TF,true>::multAssign( const SparseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const ResultType tmp( serial( *this ) );

   reset();

   for( ConstIterator_<VT2> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      vector_[offset_+element->index()] = tmp[element->index()] * element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the division assignment of a dense vector.
//
// \param rhs The right-hand side dense vector divisor.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline DisableIf_< typename Subvector<VT,aligned,TF,true>::BLAZE_TEMPLATE VectorizedDivAssign<VT2> >
   Subvector<VT,aligned,TF,true>::divAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      vector_[offset_+i    ] /= (~rhs)[i    ];
      vector_[offset_+i+1UL] /= (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      vector_[offset_+ipos] /= (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the division assignment of a dense vector.
//
// \param rhs The right-hand side dense vector divisor.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT     // Type of the dense vector
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline EnableIf_< typename Subvector<VT,aligned,TF,true>::BLAZE_TEMPLATE VectorizedDivAssign<VT2> >
   Subvector<VT,aligned,TF,true>::divAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT2> right( (~rhs).begin() );

   for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
      left.store( left.load() / right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() / right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() / right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() / right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE ) {
      left.store( left.load() / right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<size_; ++i ) {
      *left /= *right; ++left; ++right;
   }
}
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR DVECDVECCROSSEXPR
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Subvector for dense vector/dense vector cross products.
// \ingroup subvector
//
// This specialization of Subvector adapts the class template to the special case of dense
// vector/dense vector cross products.
*/
template< typename VT1  // Type of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side dense vector
        , bool TF >     // Transpose flag
class Subvector< DVecDVecCrossExpr<VT1,VT2,TF>, unaligned, TF, true >
   : public DenseVector< Subvector< DVecDVecCrossExpr<VT1,VT2,TF>, unaligned, TF, true >, TF >
   , private View
{
 private:
   //**Type definitions****************************************************************************
   typedef DVecDVecCrossExpr<VT1,VT2,TF>  CPE;  //!< Type of the cross product expression.
   typedef ResultType_<CPE>               RT;   //!< Result type of the cross product expression.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef Subvector<CPE,unaligned,TF,true>  This;           //!< Type of this Subvector instance.
   typedef DenseVector<This,TF>              BaseType;       //!< Base type of this Subvector instance.
   typedef SubvectorTrait_<RT>               ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>        TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<CPE>                 ElementType;    //!< Type of the subvector elements.
   typedef ReturnType_<CPE>                  ReturnType;     //!< Return type for expression template evaluations
   typedef const ResultType                  CompositeType;  //!< Data type for composite expression templates.
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = false };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = false };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the Subvector specialization class.
   //
   // \param vector The dense vector/dense vector cross product expression.
   // \param index The first index of the subvector in the given expression.
   // \param n The size of the subvector.
   */
   explicit inline Subvector( const CPE& vector, size_t index, size_t n ) noexcept
      : vector_( vector )  // The dense vector/dense vector cross product expression
      , offset_( index  )  // The offset of the subvector within the cross product expression
      , size_  ( n      )  // The size of the subvector
   {}
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the vector elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator[]( size_t index ) const {
      BLAZE_INTERNAL_ASSERT( index < size(), "Invalid vector access index" );
      return vector_[offset_+index];
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
      if( index >= size() ) {
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
   inline size_t size() const noexcept {
      return size_;
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
      return vector_.canAlias( alias );
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
      return vector_.isAliased( alias );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   CPE          vector_;  //!< The dense vector/dense vector cross product expression.
   const size_t offset_;  //!< The offset of the subvector within the cross product expression.
   const size_t size_;    //!< The size of the subvector.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< bool AF1, typename VT, bool AF2, bool TF2, bool DF2 >
   friend const Subvector<VT,AF1,TF2,DF2>
      subvector( const Subvector<VT,AF2,TF2,DF2>& sv, size_t index, size_t size );

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isIntact( const Subvector<VT3,AF,TF2,DF2>& sv ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isSame( const Subvector<VT3,AF,TF2,DF2>& a, const Vector<VT3,TF2>& b ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isSame( const Vector<VT3,TF2>& a, const Subvector<VT3,AF,TF2,DF2>& b ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isSame( const Subvector<VT3,AF,TF2,DF2>& a, const Subvector<VT3,AF,TF2,DF2>& b ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool tryAssign( const Subvector<VT3,AF,TF2,DF2>& lhs, const Vector<VT4,TF2>& rhs, size_t index );

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool tryAddAssign( const Subvector<VT2,AF,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool trySubAssign( const Subvector<VT2,AF,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool tryMultAssign( const Subvector<VT3,AF,TF2,DF2>& lhs, const Vector<VT4,TF2>& rhs, size_t index );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR DVECSVECCROSSEXPR
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Subvector for dense vector/sparse vector cross products.
// \ingroup subvector
//
// This specialization of Subvector adapts the class template to the special case of dense
// vector/sparse vector cross products.
*/
template< typename VT1  // Type of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side sparse vector
        , bool TF >     // Transpose flag
class Subvector< DVecSVecCrossExpr<VT1,VT2,TF>, unaligned, TF, true >
   : public DenseVector< Subvector< DVecSVecCrossExpr<VT1,VT2,TF>, unaligned, TF, true >, TF >
   , private View
{
 private:
   //**Type definitions****************************************************************************
   typedef DVecSVecCrossExpr<VT1,VT2,TF>  CPE;  //!< Type of the cross product expression.
   typedef ResultType_<CPE>               RT;   //!< Result type of the cross product expression.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef Subvector<CPE,unaligned,TF,true>  This;           //!< Type of this Subvector instance.
   typedef DenseVector<This,TF>              BaseType;       //!< Base type of this Subvector instance.
   typedef SubvectorTrait_<RT>               ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>        TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<CPE>                 ElementType;    //!< Type of the subvector elements.
   typedef ReturnType_<CPE>                  ReturnType;     //!< Return type for expression template evaluations
   typedef const ResultType                  CompositeType;  //!< Data type for composite expression templates.
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = false };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = false };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the Subvector specialization class.
   //
   // \param vector The dense vector/sparse vector cross product expression.
   // \param index The first index of the subvector in the given expression.
   // \param n The size of the subvector.
   */
   explicit inline Subvector( const CPE& vector, size_t index, size_t n ) noexcept
      : vector_( vector )  // The dense vector/sparse vector cross product expression
      , offset_( index  )  // The offset of the subvector within the cross product expression
      , size_  ( n      )  // The size of the subvector
   {}
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the vector elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator[]( size_t index ) const {
      BLAZE_INTERNAL_ASSERT( index < size(), "Invalid vector access index" );
      return vector_[offset_+index];
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
      if( index >= size() ) {
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
   inline size_t size() const noexcept {
      return size_;
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
      return vector_.canAlias( alias );
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
      return vector_.isAliased( alias );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   CPE          vector_;  //!< The dense vector/sparse vector cross product expression.
   const size_t offset_;  //!< The offset of the subvector within the cross product expression.
   const size_t size_;    //!< The size of the subvector.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< bool AF1, typename VT, bool AF2, bool TF2, bool DF2 >
   friend const Subvector<VT,AF1,TF2,DF2>
      subvector( const Subvector<VT,AF2,TF2,DF2>& sv, size_t index, size_t size );

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isIntact( const Subvector<VT3,AF,TF2,DF2>& sv ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isSame( const Subvector<VT3,AF,TF2,DF2>& a, const Vector<VT3,TF2>& b ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isSame( const Vector<VT3,TF2>& a, const Subvector<VT3,AF,TF2,DF2>& b ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isSame( const Subvector<VT3,AF,TF2,DF2>& a, const Subvector<VT3,AF,TF2,DF2>& b ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool tryAssign( const Subvector<VT3,AF,TF2,DF2>& lhs, const Vector<VT4,TF2>& rhs, size_t index );

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool tryAddAssign( const Subvector<VT2,AF,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool trySubAssign( const Subvector<VT2,AF,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool tryMultAssign( const Subvector<VT3,AF,TF2,DF2>& lhs, const Vector<VT4,TF2>& rhs, size_t index );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR SVECDVECCROSSEXPR
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Subvector for sparse vector/dense vector cross products.
// \ingroup subvector
//
// This specialization of Subvector adapts the class template to the special case of sparse
// vector/dense vector cross products.
*/
template< typename VT1  // Type of the left-hand side sparse vector
        , typename VT2  // Type of the right-hand side dense vector
        , bool TF >     // Transpose flag
class Subvector< SVecDVecCrossExpr<VT1,VT2,TF>, unaligned, TF, true >
   : public DenseVector< Subvector< SVecDVecCrossExpr<VT1,VT2,TF>, unaligned, TF, true >, TF >
   , private View
{
 private:
   //**Type definitions****************************************************************************
   typedef SVecDVecCrossExpr<VT1,VT2,TF>  CPE;  //!< Type of the cross product expression.
   typedef ResultType_<CPE>               RT;   //!< Result type of the cross product expression.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef Subvector<CPE,unaligned,TF,true>  This;           //!< Type of this Subvector instance.
   typedef DenseVector<This,TF>              BaseType;       //!< Base type of this Subvector instance.
   typedef SubvectorTrait_<RT>               ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>        TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<CPE>                 ElementType;    //!< Type of the subvector elements.
   typedef ReturnType_<CPE>                  ReturnType;     //!< Return type for expression template evaluations
   typedef const ResultType                  CompositeType;  //!< Data type for composite expression templates.
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = false };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = false };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the Subvector specialization class.
   //
   // \param vector The sparse vector/dense vector cross product expression.
   // \param index The first index of the subvector in the given expression.
   // \param n The size of the subvector.
   */
   explicit inline Subvector( const CPE& vector, size_t index, size_t n ) noexcept
      : vector_( vector )  // The sparse vector/dense vector cross product expression
      , offset_( index  )  // The offset of the subvector within the cross product expression
      , size_  ( n      )  // The size of the subvector
   {}
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the vector elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator[]( size_t index ) const {
      BLAZE_INTERNAL_ASSERT( index < size(), "Invalid vector access index" );
      return vector_[offset_+index];
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
      if( index >= size() ) {
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
   inline size_t size() const noexcept {
      return size_;
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
      return vector_.canAlias( alias );
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
      return vector_.isAliased( alias );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   CPE          vector_;  //!< The sparse vector/dense vector cross product expression.
   const size_t offset_;  //!< The offset of the subvector within the cross product expression.
   const size_t size_;    //!< The size of the subvector.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< bool AF1, typename VT, bool AF2, bool TF2, bool DF2 >
   friend const Subvector<VT,AF1,TF2,DF2>
      subvector( const Subvector<VT,AF2,TF2,DF2>& sv, size_t index, size_t size );

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isIntact( const Subvector<VT3,AF,TF2,DF2>& sv ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isSame( const Subvector<VT3,AF,TF2,DF2>& a, const Vector<VT3,TF2>& b ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isSame( const Vector<VT3,TF2>& a, const Subvector<VT3,AF,TF2,DF2>& b ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isSame( const Subvector<VT3,AF,TF2,DF2>& a, const Subvector<VT3,AF,TF2,DF2>& b ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool tryAssign( const Subvector<VT3,AF,TF2,DF2>& lhs, const Vector<VT4,TF2>& rhs, size_t index );

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool tryAddAssign( const Subvector<VT2,AF,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool trySubAssign( const Subvector<VT2,AF,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool tryMultAssign( const Subvector<VT3,AF,TF2,DF2>& lhs, const Vector<VT4,TF2>& rhs, size_t index );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR SVECSVECCROSSEXPR
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Subvector for sparse vector/sparse vector cross products.
// \ingroup subvector
//
// This specialization of Subvector adapts the class template to the special case of sparse
// vector/sparse vector cross products.
*/
template< typename VT1  // Type of the left-hand side sparse vector
        , typename VT2  // Type of the right-hand side sparse vector
        , bool TF >     // Transpose flag
class Subvector< SVecSVecCrossExpr<VT1,VT2,TF>, unaligned, TF, true >
   : public DenseVector< Subvector< SVecSVecCrossExpr<VT1,VT2,TF>, unaligned, TF, true >, TF >
   , private View
{
 private:
   //**Type definitions****************************************************************************
   typedef SVecSVecCrossExpr<VT1,VT2,TF>  CPE;  //!< Type of the cross product expression.
   typedef ResultType_<CPE>               RT;   //!< Result type of the cross product expression.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef Subvector<CPE,unaligned,TF,true>  This;           //!< Type of this Subvector instance.
   typedef DenseVector<This,TF>              BaseType;       //!< Base type of this Subvector instance.
   typedef SubvectorTrait_<RT>               ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>        TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<CPE>                 ElementType;    //!< Type of the subvector elements.
   typedef ReturnType_<CPE>                  ReturnType;     //!< Return type for expression template evaluations
   typedef const ResultType                  CompositeType;  //!< Data type for composite expression templates.
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   enum : bool { simdEnabled = false };

   //! Compilation switch for the expression template assignment strategy.
   enum : bool { smpAssignable = false };
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the Subvector specialization class.
   //
   // \param vector The sparse vector/sparse vector cross product expression.
   // \param index The first index of the subvector in the given expression.
   // \param n The size of the subvector.
   */
   explicit inline Subvector( const CPE& vector, size_t index, size_t n ) noexcept
      : vector_( vector )  // The sparse vector/sparse vector cross product expression
      , offset_( index  )  // The offset of the subvector within the cross product expression
      , size_  ( n      )  // The size of the subvector
   {}
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the vector elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator[]( size_t index ) const {
      BLAZE_INTERNAL_ASSERT( index < size(), "Invalid vector access index" );
      return vector_[offset_+index];
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
      if( index >= size() ) {
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
   inline size_t size() const noexcept {
      return size_;
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
      return vector_.canAlias( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression is aliased with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case an alias effect is detected, \a false otherwise.
   */
   template< typename T >
   inline bool isAliased( const T* alias ) const {
      return vector_.isAliased( alias );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   CPE          vector_;  //!< The sparse vector/sparse vector cross product expression.
   const size_t offset_;  //!< The offset of the subvector within the cross product expression.
   const size_t size_;    //!< The size of the subvector.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< bool AF1, typename VT, bool AF2, bool TF2, bool DF2 >
   friend const Subvector<VT,AF1,TF2,DF2>
      subvector( const Subvector<VT,AF2,TF2,DF2>& sv, size_t index, size_t size );

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isIntact( const Subvector<VT3,AF,TF2,DF2>& sv ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isSame( const Subvector<VT3,AF,TF2,DF2>& a, const Vector<VT3,TF2>& b ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isSame( const Vector<VT3,TF2>& a, const Subvector<VT3,AF,TF2,DF2>& b ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2 >
   friend bool isSame( const Subvector<VT3,AF,TF2,DF2>& a, const Subvector<VT3,AF,TF2,DF2>& b ) noexcept;

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool tryAssign( const Subvector<VT3,AF,TF2,DF2>& lhs, const Vector<VT4,TF2>& rhs, size_t index );

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool tryAddAssign( const Subvector<VT2,AF,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool trySubAssign( const Subvector<VT2,AF,TF2,DF2>& lhs, const Vector<VT3,TF2>& rhs, size_t index );

   template< typename VT3, bool AF, bool TF2, bool DF2, typename VT4 >
   friend bool tryMultAssign( const Subvector<VT3,AF,TF2,DF2>& lhs, const Vector<VT4,TF2>& rhs, size_t index );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
