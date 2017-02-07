//=================================================================================================
/*!
//  \file blaze/math/views/subvector/Sparse.h
//  \brief Subvector specialization for sparse vectors
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

#ifndef _BLAZE_MATH_VIEWS_SUBVECTOR_SPARSE_H_
#define _BLAZE_MATH_VIEWS_SUBVECTOR_SPARSE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/SparseVector.h>
#include <blaze/math/constraints/Subvector.h>
#include <blaze/math/constraints/TransExpr.h>
#include <blaze/math/constraints/TransposeFlag.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/SparseVector.h>
#include <blaze/math/expressions/View.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/sparse/SparseElement.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/DerestrictTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/traits/SubvectorTrait.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsRestricted.h>
#include <blaze/math/views/subvector/BaseTemplate.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/typetraits/IsFloatingPoint.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsReference.h>


namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR SPARSE SUBVECTORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Subvector for sparse subvectors.
// \ingroup views
//
// This specialization of Subvector adapts the class template to the requirements of sparse
// subvectors.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
class Subvector<VT,AF,TF,false>
   : public SparseVector< Subvector<VT,AF,TF,false>, TF >
   , private View
{
 private:
   //**Type definitions****************************************************************************
   //! Composite data type of the sparse vector expression.
   typedef If_< IsExpression<VT>, VT, VT& >  Operand;
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef Subvector<VT,AF,TF,false>   This;           //!< Type of this Subvector instance.
   typedef SparseVector<This,TF>       BaseType;       //!< Base type of this Subvector instance.
   typedef SubvectorTrait_<VT>         ResultType;     //!< Result type for expression template evaluations.
   typedef TransposeType_<ResultType>  TransposeType;  //!< Transpose type for expression template evaluations.
   typedef ElementType_<VT>            ElementType;    //!< Type of the subvector elements.
   typedef ReturnType_<VT>             ReturnType;     //!< Return type for expression template evaluations
   typedef const Subvector&            CompositeType;  //!< Data type for composite expression templates.

   //! Reference to a constant subvector value.
   typedef ConstReference_<VT>  ConstReference;

   //! Reference to a non-constant subvector value.
   typedef If_< IsConst<VT>, ConstReference, Reference_<VT> >  Reference;
   //**********************************************************************************************

   //**SubvectorElement class definition***********************************************************
   /*!\brief Access proxy for a specific element of the sparse subvector.
   */
   template< typename VectorType      // Type of the sparse vector
           , typename IteratorType >  // Type of the sparse vector iterator
   class SubvectorElement : private SparseElement
   {
    private:
      //*******************************************************************************************
      //! Compilation switch for the return type of the value member function.
      /*! The \a returnConst compile time constant expression represents a compilation switch for
          the return type of the value member function. In case the given vector type \a VectorType
          is const qualified, \a returnConst will be set to 1 and the value member function will
          return a reference to const. Otherwise \a returnConst will be set to 0 and the value
          member function will offer write access to the sparse vector elements. */
      enum : bool { returnConst = IsConst<VectorType>::value };
      //*******************************************************************************************

      //**Type definitions*************************************************************************
      //! Type of the underlying sparse elements.
      typedef typename std::iterator_traits<IteratorType>::value_type  SET;

      typedef Reference_<SET>       RT;   //!< Reference type of the underlying sparse element.
      typedef ConstReference_<SET>  CRT;  //!< Reference-to-const type of the underlying sparse element.
      //*******************************************************************************************

    public:
      //**Type definitions*************************************************************************
      typedef ValueType_<SET>              ValueType;       //!< The value type of the row element.
      typedef size_t                       IndexType;       //!< The index type of the row element.
      typedef IfTrue_<returnConst,CRT,RT>  Reference;       //!< Reference return type
      typedef CRT                          ConstReference;  //!< Reference-to-const return type.
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the SubvectorElement class.
      //
      // \param pos Iterator to the current position within the sparse subvector.
      // \param offset The offset within the according sparse vector.
      */
      inline SubvectorElement( IteratorType pos, size_t offset )
         : pos_   ( pos    )  // Iterator to the current position within the sparse subvector
         , offset_( offset )  // Offset within the according sparse vector
      {}
      //*******************************************************************************************

      //**Assignment operator**********************************************************************
      /*!\brief Assignment to the accessed sparse subvector element.
      //
      // \param v The new value of the sparse subvector element.
      // \return Reference to the sparse subvector element.
      */
      template< typename T > inline SubvectorElement& operator=( const T& v ) {
         *pos_ = v;
         return *this;
      }
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment to the accessed sparse subvector element.
      //
      // \param v The right-hand side value for the addition.
      // \return Reference to the sparse subvector element.
      */
      template< typename T > inline SubvectorElement& operator+=( const T& v ) {
         *pos_ += v;
         return *this;
      }
      //*******************************************************************************************

      //**Subtraction assignment operator**********************************************************
      /*!\brief Subtraction assignment to the accessed sparse subvector element.
      //
      // \param v The right-hand side value for the subtraction.
      // \return Reference to the sparse subvector element.
      */
      template< typename T > inline SubvectorElement& operator-=( const T& v ) {
         *pos_ -= v;
         return *this;
      }
      //*******************************************************************************************

      //**Multiplication assignment operator*******************************************************
      /*!\brief Multiplication assignment to the accessed sparse subvector element.
      //
      // \param v The right-hand side value for the multiplication.
      // \return Reference to the sparse subvector element.
      */
      template< typename T > inline SubvectorElement& operator*=( const T& v ) {
         *pos_ *= v;
         return *this;
      }
      //*******************************************************************************************

      //**Division assignment operator*************************************************************
      /*!\brief Division assignment to the accessed sparse subvector element.
      //
      // \param v The right-hand side value for the division.
      // \return Reference to the sparse subvector element.
      */
      template< typename T > inline SubvectorElement& operator/=( const T& v ) {
         *pos_ /= v;
         return *this;
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the sparse subvector element at the current iterator position.
      //
      // \return Reference to the sparse subvector element at the current iterator position.
      */
      inline const SubvectorElement* operator->() const {
         return this;
      }
      //*******************************************************************************************

      //**Value function***************************************************************************
      /*!\brief Access to the current value of the sparse subvector element.
      //
      // \return The current value of the sparse subvector element.
      */
      inline Reference value() const {
         return pos_->value();
      }
      //*******************************************************************************************

      //**Index function***************************************************************************
      /*!\brief Access to the current index of the sparse element.
      //
      // \return The current index of the sparse element.
      */
      inline IndexType index() const {
         return pos_->index() - offset_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType pos_;  //!< Iterator to the current position within the sparse subvector.
      size_t offset_;     //!< Offset within the according sparse vector.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**SubvectorIterator class definition**********************************************************
   /*!\brief Iterator over the elements of the sparse subvector.
   */
   template< typename VectorType      // Type of the sparse vector
           , typename IteratorType >  // Type of the sparse vector iterator
   class SubvectorIterator
   {
    public:
      //**Type definitions*************************************************************************
      typedef std::forward_iterator_tag                  IteratorCategory;  //!< The iterator category.
      typedef SubvectorElement<VectorType,IteratorType>  ValueType;         //!< Type of the underlying elements.
      typedef ValueType                                  PointerType;       //!< Pointer return type.
      typedef ValueType                                  ReferenceType;     //!< Reference return type.
      typedef ptrdiff_t                                  DifferenceType;    //!< Difference between two iterators.

      // STL iterator requirements
      typedef IteratorCategory  iterator_category;  //!< The iterator category.
      typedef ValueType         value_type;         //!< Type of the underlying elements.
      typedef PointerType       pointer;            //!< Pointer return type.
      typedef ReferenceType     reference;          //!< Reference return type.
      typedef DifferenceType    difference_type;    //!< Difference between two iterators.
      //*******************************************************************************************

      //**Default constructor**********************************************************************
      /*!\brief Default constructor for the SubvectorIterator class.
      */
      inline SubvectorIterator()
         : pos_   ()  // Iterator to the current sparse element
         , offset_()  // The offset of the subvector within the sparse vector
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the SubvectorIterator class.
      //
      // \param iterator Iterator to the current sparse element.
      // \param index The starting index of the subvector within the sparse vector.
      */
      inline SubvectorIterator( IteratorType iterator, size_t index )
         : pos_   ( iterator )  // Iterator to the current sparse element
         , offset_( index    )  // The offset of the subvector within the sparse vector
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Conversion constructor from different SubvectorIterator instances.
      //
      // \param it The subvector iterator to be copied.
      */
      template< typename VectorType2, typename IteratorType2 >
      inline SubvectorIterator( const SubvectorIterator<VectorType2,IteratorType2>& it )
         : pos_   ( it.base()   )  // Iterator to the current sparse element.
         , offset_( it.offset() )  // The offset of the subvector within the sparse vector
      {}
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline SubvectorIterator& operator++() {
         ++pos_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const SubvectorIterator operator++( int ) {
         const SubvectorIterator tmp( *this );
         ++(*this);
         return tmp;
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the current sparse subvector element.
      //
      // \return Reference to the sparse subvector element.
      */
      inline ReferenceType operator*() const {
         return ReferenceType( pos_, offset_ );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the current sparse subvector element.
      //
      // \return Pointer to the sparse subvector element.
      */
      inline PointerType operator->() const {
         return PointerType( pos_, offset_ );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two SubvectorIterator objects.
      //
      // \param rhs The right-hand side subvector iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      template< typename VectorType2, typename IteratorType2 >
      inline bool operator==( const SubvectorIterator<VectorType2,IteratorType2>& rhs ) const {
         return base() == rhs.base();
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two SubvectorIterator objects.
      //
      // \param rhs The right-hand side subvector iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      template< typename VectorType2, typename IteratorType2 >
      inline bool operator!=( const SubvectorIterator<VectorType2,IteratorType2>& rhs ) const {
         return !( *this == rhs );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two subvector iterators.
      //
      // \param rhs The right-hand side subvector iterator.
      // \return The number of elements between the two subvector iterators.
      */
      inline DifferenceType operator-( const SubvectorIterator& rhs ) const {
         return pos_ - rhs.pos_;
      }
      //*******************************************************************************************

      //**Base function****************************************************************************
      /*!\brief Access to the current position of the subvector iterator.
      //
      // \return The current position of the subvector iterator.
      */
      inline IteratorType base() const {
         return pos_;
      }
      //*******************************************************************************************

      //**Offset function**************************************************************************
      /*!\brief Access to the offset of the subvector iterator.
      //
      // \return The offset of the subvector iterator.
      */
      inline size_t offset() const noexcept {
         return offset_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType pos_;     //!< Iterator to the current sparse element.
      size_t       offset_;  //!< The offset of the subvector within the sparse vector.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   //! Iterator over constant elements.
   typedef SubvectorIterator< const VT, ConstIterator_<VT> >  ConstIterator;

   //! Iterator over non-constant elements.
   typedef If_< IsConst<VT>, ConstIterator, SubvectorIterator< VT, Iterator_<VT> > >  Iterator;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
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
                            inline Subvector& operator= ( const Subvector& rhs );
   template< typename VT2 > inline Subvector& operator= ( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline Subvector& operator+=( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline Subvector& operator-=( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline Subvector& operator*=( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline Subvector& operator/=( const DenseVector<VT2,TF>& rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, Subvector >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, Subvector >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t     size() const noexcept;
                              inline size_t     capacity() const noexcept;
                              inline size_t     nonZeros() const;
                              inline void       reset();
                              inline Iterator   set    ( size_t index, const ElementType& value );
                              inline Iterator   insert ( size_t index, const ElementType& value );
                              inline void       erase  ( size_t index );
                              inline Iterator   erase  ( Iterator pos );
                              inline Iterator   erase  ( Iterator first, Iterator last );
                              inline void       reserve( size_t n );
   template< typename Other > inline Subvector& scale  ( const Other& scalar );
   //@}
   //**********************************************************************************************

   //**Lookup functions****************************************************************************
   /*!\name Lookup functions */
   //@{
   inline Iterator      find      ( size_t index );
   inline ConstIterator find      ( size_t index ) const;
   inline Iterator      lowerBound( size_t index );
   inline ConstIterator lowerBound( size_t index ) const;
   inline Iterator      upperBound( size_t index );
   inline ConstIterator upperBound( size_t index ) const;
   //@}
   //**********************************************************************************************

   //**Low-level utility functions*****************************************************************
   /*!\name Low-level utility functions */
   //@{
   inline void append( size_t index, const ElementType& value, bool check=false );
   //@}
   //**********************************************************************************************

   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other > inline bool canAlias ( const Other* alias ) const noexcept;
   template< typename Other > inline bool isAliased( const Other* alias ) const noexcept;

   inline bool canSMPAssign() const noexcept;

   template< typename VT2 >   inline void assign   ( const DenseVector <VT2,TF>& rhs );
   template< typename VT2 >   inline void assign   ( const SparseVector<VT2,TF>& rhs );
   template< typename VT2 >   inline void addAssign( const DenseVector <VT2,TF>& rhs );
   template< typename VT2 >   inline void addAssign( const SparseVector<VT2,TF>& rhs );
   template< typename VT2 >   inline void subAssign( const DenseVector <VT2,TF>& rhs );
   template< typename VT2 >   inline void subAssign( const SparseVector<VT2,TF>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand      vector_;  //!< The sparse vector containing the subvector.
   const size_t offset_;  //!< The offset of the subvector within the sparse vector.
   const size_t size_;    //!< The size of the subvector.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
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
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE  ( VT );
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
// \param vector The sparse vector containing the subvector.
// \param index The index of the first element of the subvector.
// \param n The size of the subvector.
// \exception std::invalid_argument Invalid subvector specification.
//
// In case the subvector is not properly specified (i.e. if the specified first index is larger
// than the size of the given vector or the subvector is specified beyond the size of the vector)
// a \a std::invalid_argument exception is thrown.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline Subvector<VT,AF,TF,false>::Subvector( Operand vector, size_t index, size_t n )
   : vector_( vector )  // The sparse vector containing the subvector
   , offset_( index  )  // The offset of the subvector within the sparse vector
   , size_  ( n      )  // The size of the subvector
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
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::Reference
   Subvector<VT,AF,TF,false>::operator[]( size_t index )
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
// \param index Access index. The index must be smaller than the number of subvector elements.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::ConstReference
   Subvector<VT,AF,TF,false>::operator[]( size_t index ) const
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
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::Reference
   Subvector<VT,AF,TF,false>::at( size_t index )
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
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::ConstReference
   Subvector<VT,AF,TF,false>::at( size_t index ) const
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
/*!\brief Returns an iterator to the first element of the subvector.
//
// \return Iterator to the first element of the subvector.
//
// This function returns an iterator to the first element of the subvector.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::Iterator Subvector<VT,AF,TF,false>::begin()
{
   if( offset_ == 0UL )
      return Iterator( vector_.begin(), offset_ );
   else
      return Iterator( vector_.lowerBound( offset_ ), offset_ );
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
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::ConstIterator Subvector<VT,AF,TF,false>::begin() const
{
   if( offset_ == 0UL )
      return ConstIterator( vector_.cbegin(), offset_ );
   else
      return ConstIterator( vector_.lowerBound( offset_ ), offset_ );
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
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::ConstIterator Subvector<VT,AF,TF,false>::cbegin() const
{
   if( offset_ == 0UL )
      return ConstIterator( vector_.cbegin(), offset_ );
   else
      return ConstIterator( vector_.lowerBound( offset_ ), offset_ );
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
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::Iterator Subvector<VT,AF,TF,false>::end()
{
   if( offset_ + size_ == vector_.size() )
      return Iterator( vector_.end(), offset_ );
   else
      return Iterator( vector_.lowerBound( offset_ + size_ ), offset_ );
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
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::ConstIterator Subvector<VT,AF,TF,false>::end() const
{
   if( offset_ + size_ == vector_.size() )
      return ConstIterator( vector_.cend(), offset_ );
   else
      return ConstIterator( vector_.lowerBound( offset_ + size_ ), offset_ );
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
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::ConstIterator Subvector<VT,AF,TF,false>::cend() const
{
   if( offset_ + size_ == vector_.size() )
      return ConstIterator( vector_.cend(), offset_ );
   else
      return ConstIterator( vector_.lowerBound( offset_ + size_ ), offset_ );
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
/*!\brief Copy assignment operator for Subvector.
//
// \param rhs Sparse subvector to be copied.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Subvector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two subvectors don't match, a \a std::invalid_argument
// exception is thrown.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline Subvector<VT,AF,TF,false>&
   Subvector<VT,AF,TF,false>::operator=( const Subvector& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( this == &rhs || ( &vector_ == &rhs.vector_ && offset_ == rhs.offset_ ) )
      return *this;

   if( size() != rhs.size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( !tryAssign( vector_, rhs, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   if( rhs.canAlias( &vector_ ) ) {
      const ResultType tmp( rhs );
      reset();
      assign( left, tmp );
   }
   else {
      reset();
      assign( left, rhs );
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
// \param rhs Dense vector to be assigned.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument
// exception is thrown.
*/
template< typename VT     // Type of the sparse vector
        , bool AF         // Alignment flag
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side vector
inline Subvector<VT,AF,TF,false>&
   Subvector<VT,AF,TF,false>::operator=( const Vector<VT2,TF>& rhs )
{
   using blaze::assign;

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

   if( IsReference<Right>::value || right.canAlias( &vector_ ) ) {
      const ResultType_<VT2> tmp( right );
      reset();
      assign( left, tmp );
   }
   else {
      reset();
      assign( left, right );
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
// \param rhs The right-hand side vector to be added to the sparse subvector.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT     // Type of the sparse vector
        , bool AF         // Alignment flag
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side vector
inline Subvector<VT,AF,TF,false>&
   Subvector<VT,AF,TF,false>::operator+=( const Vector<VT2,TF>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   typedef AddTrait_< ResultType, ResultType_<VT2> >  AddType;

   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( AddType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   const AddType tmp( *this + (~rhs) );

   if( !tryAssign( vector_, tmp, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   left.reset();
   assign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a vector (\f$ \vec{a}-=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be subtracted from the sparse subvector.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT     // Type of the sparse vector
        , bool AF         // Alignment flag
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side vector
inline Subvector<VT,AF,TF,false>&
   Subvector<VT,AF,TF,false>::operator-=( const Vector<VT2,TF>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   typedef SubTrait_< ResultType, ResultType_<VT2> >  SubType;

   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( SubType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   const SubType tmp( *this - (~rhs) );

   if( !tryAssign( vector_, tmp, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   left.reset();
   assign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of a vector
//        (\f$ \vec{a}*=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be multiplied with the sparse subvector.
// \return Reference to the assigned subvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT     // Type of the sparse vector
        , bool AF         // Alignment flag
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side vector
inline Subvector<VT,AF,TF,false>&
   Subvector<VT,AF,TF,false>::operator*=( const Vector<VT2,TF>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   typedef MultTrait_< ResultType, ResultType_<VT2> >  MultType;

   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( MultType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MultType );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   const MultType tmp( *this * (~rhs) );

   if( !tryAssign( vector_, tmp, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   left.reset();
   assign( left, tmp );

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
template< typename VT     // Type of the sparse vector
        , bool AF         // Alignment flag
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline Subvector<VT,AF,TF,false>&
   Subvector<VT,AF,TF,false>::operator/=( const DenseVector<VT2,TF>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE  ( ResultType_<VT2> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT2> );

   typedef DivTrait_< ResultType, ResultType_<VT2> >  DivType;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( DivType );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( DivType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( DivType );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   const DivType tmp( *this / (~rhs) );

   if( !tryAssign( vector_, tmp, offset_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   DerestrictTrait_<This> left( derestrict( *this ) );

   left.reset();
   assign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication between a sparse subvector
//        and a scalar value (\f$ \vec{a}*=s \f$).
//
// \param rhs The right-hand side scalar value for the multiplication.
// \return Reference to the assigned subvector.
//
// This operator can only be used for built-in data types. Additionally, the elements of
// the sparse subvector must support the multiplication assignment operator for the given
// scalar built-in data type.
*/
template< typename VT       // Type of the sparse vector
        , bool AF           // Alignment flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, Subvector<VT,AF,TF,false> >&
   Subvector<VT,AF,TF,false>::operator*=( Other rhs )
{
   const Iterator last( end() );
   for( Iterator element=begin(); element!=last; ++element )
      element->value() *= rhs;
   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division assignment operator for the division of a sparse subvector by a scalar value
//        (\f$ \vec{a}/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the assigned subvector.
//
// This operator can only be used for built-in data types. Additionally, the elements of the
// sparse subvector must either support the multiplication assignment operator for the given
// floating point data type or the division assignment operator for the given integral data
// type.
*/
template< typename VT       // Type of the sparse vector
        , bool AF           // Alignment flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, Subvector<VT,AF,TF,false> >&
   Subvector<VT,AF,TF,false>::operator/=( Other rhs )
{
   BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

   typedef DivTrait_<ElementType,Other>     DT;
   typedef If_< IsNumeric<DT>, DT, Other >  Tmp;

   const Iterator last( end() );

   // Depending on the two involved data types, an integer division is applied or a
   // floating point division is selected.
   if( IsNumeric<DT>::value && IsFloatingPoint<DT>::value ) {
      const Tmp tmp( Tmp(1)/static_cast<Tmp>( rhs ) );
      for( Iterator element=begin(); element!=last; ++element )
         element->value() *= tmp;
   }
   else {
      for( Iterator element=begin(); element!=last; ++element )
         element->value() /= rhs;
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
/*!\brief Returns the size/dimension of the sparse subvector.
//
// \return The size of the sparse subvector.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline size_t Subvector<VT,AF,TF,false>::size() const noexcept
{
   return size_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the sparse subvector.
//
// \return The capacity of the sparse subvector.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline size_t Subvector<VT,AF,TF,false>::capacity() const noexcept
{
   return nonZeros() + vector_.capacity() - vector_.nonZeros();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the subvector.
//
// \return The number of non-zero elements in the subvector.
//
// Note that the number of non-zero elements is always smaller than the size of the subvector.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline size_t Subvector<VT,AF,TF,false>::nonZeros() const
{
   return end() - begin();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline void Subvector<VT,AF,TF,false>::reset()
{
   vector_.erase( vector_.lowerBound( offset_ ), vector_.lowerBound( offset_ + size_ ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setting an element of the sparse subvector.
//
// \param index The index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be set.
// \return Reference to the set value.
//
// This function sets the value of an element of the sparse subvector. In case the sparse subvector
// already contains an element with index \a index its value is modified, else a new element with
// the given \a value is inserted.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::Iterator
   Subvector<VT,AF,TF,false>::set( size_t index, const ElementType& value )
{
   return Iterator( vector_.set( offset_ + index, value ), offset_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Inserting an element into the sparse subvector.
//
// \param index The index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be inserted.
// \return Reference to the inserted value.
// \exception std::invalid_argument Invalid sparse subvector access index.
//
// This function inserts a new element into the sparse subvector. However, duplicate elements
// are not allowed. In case the sparse subvector already contains an element at index \a index,
// a \a std::invalid_argument exception is thrown.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::Iterator
   Subvector<VT,AF,TF,false>::insert( size_t index, const ElementType& value )
{
   return Iterator( vector_.insert( offset_ + index, value ), offset_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing an element from the sparse subvector.
//
// \param index The index of the element to be erased. The index has to be in the range \f$[0..N-1]\f$.
// \return void
//
// This function erases an element from the sparse subvector.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline void Subvector<VT,AF,TF,false>::erase( size_t index )
{
   vector_.erase( offset_ + index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing an element from the sparse subvector.
//
// \param pos Iterator to the element to be erased.
// \return void
//
// This function erases an element from the sparse subvector.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::Iterator Subvector<VT,AF,TF,false>::erase( Iterator pos )
{
   return Iterator( vector_.erase( pos.base() ), offset_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing a range of elements from the sparse subvector.
//
// \param first Iterator to first element to be erased.
// \param last Iterator just past the last element to be erased.
// \return Iterator to the element after the erased element.
//
// This function erases a range of elements from the sparse subvector.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::Iterator
   Subvector<VT,AF,TF,false>::erase( Iterator first, Iterator last )
{
   return Iterator( vector_.erase( first.base(), last.base() ), offset_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setting the minimum capacity of the sparse subvector.
//
// \param n The new minimum capacity of the sparse subvector.
// \return void
//
// This function increases the capacity of the sparse subvector to at least \a n elements. The
// current values of the subvector elements are preserved.
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
void Subvector<VT,AF,TF,false>::reserve( size_t n )
{
   const size_t current( capacity() );

   if( n > current ) {
      vector_.reserve( vector_.capacity() + n - current );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the sparse subvector by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the subvector scaling.
// \return Reference to the sparse subvector.
*/
template< typename VT       // Type of the sparse vector
        , bool AF           // Alignment flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the scalar value
inline Subvector<VT,AF,TF,false>& Subvector<VT,AF,TF,false>::scale( const Other& scalar )
{
   for( Iterator element=begin(); element!=end(); ++element )
      element->value() *= scalar;
   return *this;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LOOKUP FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Searches for a specific subvector element.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the element in case the index is found, end() iterator otherwise.
//
// This function can be used to check whether a specific element is contained in the sparse
// subvector. It specifically searches for the element with index \a index. In case the element
// is found, the function returns an iterator to the element. Otherwise an iterator just past
// the last non-zero element of the sparse subvector (the end() iterator) is returned. Note that
// the returned sparse subvector iterator is subject to invalidation due to inserting operations
// via the subscript operator or the insert() function!
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::Iterator
   Subvector<VT,AF,TF,false>::find( size_t index )
{
   const Iterator_<VT> pos( vector_.find( offset_ + index ) );

   if( pos != vector_.end() )
      return Iterator( pos, offset_ );
   else
      return end();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Searches for a specific subvector element.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the element in case the index is found, end() iterator otherwise.
//
// This function can be used to check whether a specific element is contained in the sparse
// subvector. It specifically searches for the element with index \a index. In case the element
// is found, the function returns an iterator to the element. Otherwise an iterator just past
// the last non-zero element of the sparse subvector (the end() iterator) is returned. Note that
// the returned sparse subvector iterator is subject to invalidation due to inserting operations
// via the subscript operator or the insert() function!
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::ConstIterator
   Subvector<VT,AF,TF,false>::find( size_t index ) const
{
   const ConstIterator_<VT> pos( vector_.find( offset_ + index ) );

   if( pos != vector_.end() )
      return Iterator( pos, offset_ );
   else
      return end();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index not less then the given index.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index not less then the given index, end() iterator otherwise.
//
// This function returns an iterator to the first element with an index not less then the given
// index. In combination with the upperBound() function this function can be used to create a
// pair of iterators specifying a range of indices. Note that the returned sparse subvector
// iterator is subject to invalidation due to inserting operations via the subscript operator
// or the insert() function!
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::Iterator
   Subvector<VT,AF,TF,false>::lowerBound( size_t index )
{
   return Iterator( vector_.lowerBound( offset_ + index ), offset_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index not less then the given index.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index not less then the given index, end() iterator otherwise.
//
// This function returns an iterator to the first element with an index not less then the given
// index. In combination with the upperBound() function this function can be used to create a
// pair of iterators specifying a range of indices. Note that the returned sparse subvector
// iterator is subject to invalidation due to inserting operations via the subscript operator
// or the insert() function!
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::ConstIterator
   Subvector<VT,AF,TF,false>::lowerBound( size_t index ) const
{
   return ConstIterator( vector_.lowerBound( offset_ + index ), offset_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index greater then the given index.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index greater then the given index, end() iterator otherwise.
//
// This function returns an iterator to the first element with an index greater then the given
// index. In combination with the upperBound() function this function can be used to create a
// pair of iterators specifying a range of indices. Note that the returned sparse subvector
// iterator is subject to invalidation due to inserting operations via the subscript operator
// or the insert() function!
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::Iterator
   Subvector<VT,AF,TF,false>::upperBound( size_t index )
{
   return Iterator( vector_.upperBound( offset_ + index ), offset_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index greater then the given index.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index greater then the given index, end() iterator otherwise.
//
// This function returns an iterator to the first element with an index greater then the given
// index. In combination with the upperBound() function this function can be used to create a
// pair of iterators specifying a range of indices. Note that the returned sparse subvector
// iterator is subject to invalidation due to inserting operations via the subscript operator
// or the insert() function!
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline typename Subvector<VT,AF,TF,false>::ConstIterator
   Subvector<VT,AF,TF,false>::upperBound( size_t index ) const
{
   return ConstIterator( vector_.upperBound( offset_ + index ), offset_ );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LOW-LEVEL UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Appending an element to the sparse subvector.
//
// \param index The index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be appended.
// \param check \a true if the new value should be checked for default values, \a false if not.
// \return void
//
// This function provides a very efficient way to fill a sparse subvector with elements. It
// appends a new element to the end of the sparse subvector without any memory allocation.
// Therefore it is strictly necessary to keep the following preconditions in mind:
//
//  - the index of the new element must be strictly larger than the largest index of non-zero
//    elements in the sparse subvector
//  - the current number of non-zero elements must be smaller than the capacity of the subvector
//
// Ignoring these preconditions might result in undefined behavior! The optional \a check
// parameter specifies whether the new value should be tested for a default value. If the new
// value is a default value (for instance 0 in case of an integral element type) the value is
// not appended. Per default the values are not tested.
//
// \note Although append() does not allocate new memory, it still invalidates all iterators
// returned by the end() functions!
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline void Subvector<VT,AF,TF,false>::append( size_t index, const ElementType& value, bool check )
{
   if( offset_ + size_ == vector_.size() )
      vector_.append( offset_ + index, value, check );
   else if( !check || !isDefault( value ) )
      vector_.insert( offset_ + index, value );
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
/*!\brief Returns whether the sparse subvector can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this sparse subvector, \a false if not.
//
// This function returns whether the given address can alias with the sparse subvector. In
// contrast to the isAliased() function this function is allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename VT       // Type of the sparse vector
        , bool AF           // Alignment flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool Subvector<VT,AF,TF,false>::canAlias( const Other* alias ) const noexcept
{
   return vector_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the sparse subvector is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this sparse subvector, \a false if not.
//
// This function returns whether the given address is aliased with the sparse subvector.
// In contrast to the canAlias() function this function is not allowed to use compile time
// expressions to optimize the evaluation.
*/
template< typename VT       // Type of the sparse vector
        , bool AF           // Alignment flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool Subvector<VT,AF,TF,false>::isAliased( const Other* alias ) const noexcept
{
   return vector_.isAliased( alias );
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
// vector).
*/
template< typename VT  // Type of the sparse vector
        , bool AF      // Alignment flag
        , bool TF >    // Transpose flag
inline bool Subvector<VT,AF,TF,false>::canSMPAssign() const noexcept
{
   return false;
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
template< typename VT     // Type of the sparse vector
        , bool AF         // Alignment flag
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline void Subvector<VT,AF,TF,false>::assign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );
   BLAZE_INTERNAL_ASSERT( nonZeros() == 0UL, "Invalid non-zero elements detected" );

   reserve( (~rhs).size() );

   for( size_t i=0UL; i<size(); ++i ) {
      append( i, (~rhs)[i], true );
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
template< typename VT     // Type of the sparse vector
        , bool AF         // Alignment flag
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side sparse vector
inline void Subvector<VT,AF,TF,false>::assign( const SparseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );
   BLAZE_INTERNAL_ASSERT( nonZeros() == 0UL, "Invalid non-zero elements detected" );

   reserve( (~rhs).nonZeros() );

   for( ConstIterator_<VT2> element=(~rhs).begin(); element!=(~rhs).end(); ++element ) {
      append( element->index(), element->value(), true );
   }
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
template< typename VT     // Type of the sparse vector
        , bool AF         // Alignment flag
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline void Subvector<VT,AF,TF,false>::addAssign( const DenseVector<VT2,TF>& rhs )
{
   typedef AddTrait_< ResultType, ResultType_<VT2> >  AddType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( AddType );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( AddType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const AddType tmp( serial( *this + (~rhs) ) );
   reset();
   assign( tmp );
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
template< typename VT     // Type of the sparse vector
        , bool AF         // Alignment flag
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side sparse vector
inline void Subvector<VT,AF,TF,false>::addAssign( const SparseVector<VT2,TF>& rhs )
{
   typedef AddTrait_< ResultType, ResultType_<VT2> >  AddType;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( AddType );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( AddType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const AddType tmp( serial( *this + (~rhs) ) );
   reset();
   assign( tmp );
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
template< typename VT     // Type of the sparse vector
        , bool AF         // Alignment flag
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side dense vector
inline void Subvector<VT,AF,TF,false>::subAssign( const DenseVector<VT2,TF>& rhs )
{
   typedef SubTrait_< ResultType, ResultType_<VT2> >  SubType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( SubType );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( SubType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const SubType tmp( serial( *this - (~rhs) ) );
   reset();
   assign( tmp );
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
template< typename VT     // Type of the sparse vector
        , bool AF         // Alignment flag
        , bool TF >       // Transpose flag
template< typename VT2 >  // Type of the right-hand side sparse vector
inline void Subvector<VT,AF,TF,false>::subAssign( const SparseVector<VT2,TF>& rhs )
{
   typedef SubTrait_< ResultType, ResultType_<VT2> >  SubType;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( SubType );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( SubType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const SubType tmp( serial( *this - (~rhs) ) );
   reset();
   assign( tmp );
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
