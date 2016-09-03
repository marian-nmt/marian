//=================================================================================================
/*!
//  \file blaze/math/sparse/CompressedVector.h
//  \brief Implementation of an arbitrarily sized compressed vector
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

#ifndef _BLAZE_MATH_SPARSE_COMPRESSEDVECTOR_H_
#define _BLAZE_MATH_SPARSE_COMPRESSEDVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <utility>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/TransposeFlag.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/SparseVector.h>
#include <blaze/math/Forward.h>
#include <blaze/math/Functions.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/sparse/ValueIndexPair.h>
#include <blaze/math/sparse/VectorAccessProxy.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/CrossTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/MathTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/traits/SubvectorTrait.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsSMPAssignable.h>
#include <blaze/system/Thresholds.h>
#include <blaze/system/TransposeFlag.h>
#include <blaze/util/Algorithm.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/SameSize.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/Memory.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsFloatingPoint.h>
#include <blaze/util/typetraits/IsNumeric.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup compressed_vector CompressedVector
// \ingroup sparse_vector
*/
/*!\brief Efficient implementation of an arbitrary sized sparse vector.
// \ingroup compressed_vector
//
// The CompressedVector class is the representation of an arbitrarily sized sparse vector,
// which stores only non-zero elements of arbitrary type. The type of the elements and the
// transpose flag of the vector can be specified via the two template parameters:

   \code
   template< typename Type, bool TF >
   class CompressedVector;
   \endcode

//  - Type: specifies the type of the vector elements. CompressedVector can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
//  - TF  : specifies whether the vector is a row vector (\a blaze::rowVector) or a column
//          vector (\a blaze::columnVector). The default value is \a blaze::columnVector.
//
// Inserting/accessing elements in a compressed vector can be done by several alternative
// functions. The following example demonstrates all options:

   \code
   // Creating a compressed column vector of size 100
   CompressedVector<double,columnVector> a( 100 );

   // The subscript operator provides access to all possible elements of the compressed vector,
   // including the zero elements. In case the subscript operator is used to access an element
   // that is currently not stored in the sparse vector, the element is inserted into the vector.
   a[42] = 2.0;

   // The second operation for inserting elements is the set() function. In case the element
   // is not contained in the vector it is inserted into the vector, if it is already contained
   // in the vector its value is modified.
   a.set( 45, -1.2 );

   // An alternative for inserting elements into the vector is the insert() function. However,
   // it inserts the element only in case the element is not already contained in the vector.
   a.insert( 50, 3.7 );

   // A very efficient way to add new elements to a sparse vector is the append() function.
   // Note that append() requires that the appended element's index is strictly larger than
   // the currently largest non-zero index of the vector and that the vector's capacity
   // is large enough to hold the new element.
   a.reserve( 10 );
   a.append( 51, -2.1 );

   // In order to traverse all non-zero elements currently stored in the vector, the begin()
   // and end() functions can be used. In the example, all non-zero elements of the vector are
   // traversed.
   for( CompressedVector<double,false>::Iterator it=a.begin(); it!=a.end(); ++it ) {
      ... = it->value();  // Access to the value of the non-zero element
      ... = it->index();  // Access to the index of the non-zero element
   }
   \endcode

// The use of CompressedVector is very natural and intuitive. All operations (addition, subtraction,
// multiplication, scaling, ...) can be performed on all possible combinations of dense and sparse
// vectors with fitting element types. The following example gives an impression of the use of
// CompressedVector:

   \code
   using blaze::CompressedVector;
   using blaze::DynamicVector;
   using blaze::CompressedMatrix;

   CompressedVector<double> a( 2 );  // Default constructed, non-initialized 2D vectors
   a[0] = 1.0;                       // Initialization of the first element
   a[1] = 2.0;                       // Initialization of the second element

   CompressedVector<double> b( 2 );        // Empty sparse vector
   DynamicVector<float>     c( 2, 2.0F );  // Directly, homogeneously initialized dense vector
   CompressedVector<double> d;             // Default constructed dynamic vector
   CompressedMatrix<double> A;             // Default constructed row-major matrix

   d = a + b;  // Vector addition between vectors of equal element type
   d = a - c;  // Vector subtraction between a sparse and dense vector with different element types
   d = a * b;  // Component-wise vector multiplication

   a *= 2.0;      // In-place scaling of vector
   d  = a * 2.0;  // Scaling of vector a
   d  = 2.0 * a;  // Scaling of vector a

   d += a - b;  // Addition assignment
   d -= a + c;  // Subtraction assignment
   d *= a * b;  // Multiplication assignment

   double scalar = trans( a ) * b;  // Scalar/dot/inner product between two vectors

   A = a * trans( b );  // Outer product between two vectors
   \endcode
*/
template< typename Type                     // Data type of the vector
        , bool TF = defaultTransposeFlag >  // Transpose flag
class CompressedVector : public SparseVector< CompressedVector<Type,TF>, TF >
{
 private:
   //**Type definitions****************************************************************************
   typedef ValueIndexPair<Type>  ElementBase;  //!< Base class for the compressed vector element.
   //**********************************************************************************************

   //**Private class Element***********************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Value-index-pair for the CompressedVector class.
   */
   struct Element : public ElementBase
   {
      // This operator is required due to a bug in all versions of the the MSVC compiler.
      // A simple 'using ElementBase::operator=;' statement results in ambiguity problems.
      template< typename Other >
      inline Element& operator=( const Other& rhs )
      {
         ElementBase::operator=( rhs );
         return *this;
      }

      friend class CompressedVector;
   };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef CompressedVector<Type,TF>   This;            //!< Type of this CompressedVector instance.
   typedef SparseVector<This,TF>       BaseType;        //!< Base type of this CompressedVector instance.
   typedef This                        ResultType;      //!< Result type for expression template evaluations.
   typedef CompressedVector<Type,!TF>  TransposeType;   //!< Transpose type for expression template evaluations.
   typedef Type                        ElementType;     //!< Type of the compressed vector elements.
   typedef const Type&                 ReturnType;      //!< Return type for expression template evaluations.
   typedef const CompressedVector&     CompositeType;   //!< Data type for composite expression templates.
   typedef VectorAccessProxy<This>     Reference;       //!< Reference to a non-constant vector value.
   typedef const Type&                 ConstReference;  //!< Reference to a constant vector value.
   typedef Element*                    Iterator;        //!< Iterator over non-constant elements.
   typedef const Element*              ConstIterator;   //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a CompressedVector with different data/element type.
   */
   template< typename ET >  // Data type of the other vector
   struct Rebind {
      typedef CompressedVector<ET,TF>  Other;  //!< The type of the other CompressedVector.
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SMP assignments.
   /*! The \a smpAssignable compilation flag indicates whether the vector can be used in SMP
       (shared memory parallel) assignments (both on the left-hand and right-hand side of the
       assignment). */
   enum : bool { smpAssignable = !IsSMPAssignable<Type>::value };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
                           explicit inline CompressedVector() noexcept;
                           explicit inline CompressedVector( size_t size ) noexcept;
                           explicit inline CompressedVector( size_t size, size_t nonzeros );
                                    inline CompressedVector( const CompressedVector& sv );
                                    inline CompressedVector( CompressedVector&& sv ) noexcept;
   template< typename VT >          inline CompressedVector( const DenseVector<VT,TF>&  dv );
   template< typename VT >          inline CompressedVector( const SparseVector<VT,TF>& sv );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~CompressedVector();
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator[]( size_t index ) noexcept;
   inline ConstReference operator[]( size_t index ) const noexcept;
   inline Reference      at( size_t index );
   inline ConstReference at( size_t index ) const;
   inline Iterator       begin () noexcept;
   inline ConstIterator  begin () const noexcept;
   inline ConstIterator  cbegin() const noexcept;
   inline Iterator       end   () noexcept;
   inline ConstIterator  end   () const noexcept;
   inline ConstIterator  cend  () const noexcept;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline CompressedVector& operator=( const CompressedVector& rhs );
   inline CompressedVector& operator=( CompressedVector&& rhs ) noexcept;

   template< typename VT > inline CompressedVector& operator= ( const DenseVector<VT,TF>& rhs );
   template< typename VT > inline CompressedVector& operator= ( const SparseVector<VT,TF>& rhs );
   template< typename VT > inline CompressedVector& operator+=( const Vector<VT,TF>& rhs );
   template< typename VT > inline CompressedVector& operator-=( const Vector<VT,TF>& rhs );
   template< typename VT > inline CompressedVector& operator*=( const DenseVector<VT,TF>& rhs );
   template< typename VT > inline CompressedVector& operator*=( const SparseVector<VT,TF>& rhs );
   template< typename VT > inline CompressedVector& operator/=( const DenseVector<VT,TF>& rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, CompressedVector >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_< IsNumeric<Other>, CompressedVector >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t            size() const noexcept;
                              inline size_t            capacity() const noexcept;
                              inline size_t            nonZeros() const;
                              inline void              reset();
                              inline void              clear();
                              inline Iterator          set   ( size_t index, const Type& value );
                              inline Iterator          insert( size_t index, const Type& value );
                              inline void              erase ( size_t index );
                              inline Iterator          erase ( Iterator pos );
                              inline Iterator          erase ( Iterator first, Iterator last );
                              inline void              resize( size_t n, bool preserve=true );
                                     void              reserve( size_t n );
   template< typename Other > inline CompressedVector& scale( const Other& scalar );
                              inline void              swap( CompressedVector& sv ) noexcept;
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
   inline void append( size_t index, const Type& value, bool check=false );
   //@}
   //**********************************************************************************************

   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other > inline bool canAlias ( const Other* alias ) const noexcept;
   template< typename Other > inline bool isAliased( const Other* alias ) const noexcept;

   inline bool canSMPAssign() const noexcept;

   template< typename VT > inline void assign    ( const DenseVector <VT,TF>& rhs );
   template< typename VT > inline void assign    ( const SparseVector<VT,TF>& rhs );
   template< typename VT > inline void addAssign ( const DenseVector <VT,TF>& rhs );
   template< typename VT > inline void addAssign ( const SparseVector<VT,TF>& rhs );
   template< typename VT > inline void subAssign ( const DenseVector <VT,TF>& rhs );
   template< typename VT > inline void subAssign ( const SparseVector<VT,TF>& rhs );
   template< typename VT > inline void multAssign( const DenseVector <VT,TF>& rhs );
   template< typename VT > inline void divAssign ( const DenseVector <VT,TF>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
          Iterator insert( Iterator pos, size_t index, const Type& value );
   inline size_t   extendCapacity() const noexcept;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t size_;             //!< The current size/dimension of the compressed vector.
   size_t capacity_;         //!< The maximum capacity of the compressed vector.
   Iterator begin_;          //!< Pointer to the first non-zero element of the compressed vector.
   Iterator end_;            //!< Pointer one past the last non-zero element of the compressed vector.

   static const Type zero_;  //!< Neutral element for accesses to zero elements.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE  ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST         ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE      ( Type );
   BLAZE_CONSTRAINT_MUST_HAVE_SAME_SIZE( ElementBase, Element );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  DEFINITION AND INITIALIZATION OF THE STATIC MEMBER VARIABLES
//
//=================================================================================================

template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
const Type CompressedVector<Type,TF>::zero_ = Type();




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The default constructor for CompressedVector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CompressedVector<Type,TF>::CompressedVector() noexcept
   : size_    ( 0UL )      // The current size/dimension of the compressed vector
   , capacity_( 0UL )      // The maximum capacity of the compressed vector
   , begin_   ( nullptr )  // Pointer to the first non-zero element of the compressed vector
   , end_     ( nullptr )  // Pointer to the last non-zero element of the compressed vector
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a compressed vector of size \a n.
//
// \param n The size of the vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CompressedVector<Type,TF>::CompressedVector( size_t n ) noexcept
   : size_    ( n   )      // The current size/dimension of the compressed vector
   , capacity_( 0UL )      // The maximum capacity of the compressed vector
   , begin_   ( nullptr )  // Pointer to the first non-zero element of the compressed vector
   , end_     ( nullptr )  // Pointer to the last non-zero element of the compressed vector
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a compressed vector of size \a n.
//
// \param n The size of the vector.
// \param nonzeros The number of expected non-zero elements.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CompressedVector<Type,TF>::CompressedVector( size_t n, size_t nonzeros )
   : size_    ( n )                               // The current size/dimension of the compressed vector
   , capacity_( nonzeros )                        // The maximum capacity of the compressed vector
   , begin_   ( allocate<Element>( capacity_ ) )  // Pointer to the first non-zero element of the compressed vector
   , end_     ( begin_ )                          // Pointer to the last non-zero element of the compressed vector
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for CompressedVector.
//
// \param sv Compressed vector to be copied.
//
// The copy constructor is explicitly defined due to the required dynamic memory management
// and in order to enable/facilitate NRV optimization.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CompressedVector<Type,TF>::CompressedVector( const CompressedVector& sv )
   : size_    ( sv.size_ )                        // The current size/dimension of the compressed vector
   , capacity_( sv.nonZeros() )                   // The maximum capacity of the compressed vector
   , begin_   ( allocate<Element>( capacity_ ) )  // Pointer to the first non-zero element of the compressed vector
   , end_     ( begin_+capacity_ )                // Pointer to the last non-zero element of the compressed vector
{
   std::copy( sv.begin_, sv.end_, begin_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The move constructor for CompressedVector.
//
// \param sv The compressed vector to be moved into this instance.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CompressedVector<Type,TF>::CompressedVector( CompressedVector&& sv ) noexcept
   : size_    ( sv.size_ )      // The current size/dimension of the compressed vector
   , capacity_( sv.capacity_ )  // The maximum capacity of the compressed vector
   , begin_   ( sv.begin_ )     // Pointer to the first non-zero element of the compressed vector
   , end_     ( sv.end_ )       // Pointer to the last non-zero element of the compressed vector
{
   sv.size_     = 0UL;
   sv.capacity_ = 0UL;
   sv.begin_    = nullptr;
   sv.end_      = nullptr;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from dense vectors.
//
// \param dv Dense vector to be copied.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the foreign dense vector
inline CompressedVector<Type,TF>::CompressedVector( const DenseVector<VT,TF>& dv )
   : size_    ( (~dv).size() )  // The current size/dimension of the compressed vector
   , capacity_( 0UL )           // The maximum capacity of the compressed vector
   , begin_   ( nullptr )       // Pointer to the first non-zero element of the compressed vector
   , end_     ( nullptr )       // Pointer to the last non-zero element of the compressed vector
{
   using blaze::assign;
   assign( *this, ~dv );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different sparse vectors.
//
// \param sv Sparse vector to be copied.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the foreign sparse vector
inline CompressedVector<Type,TF>::CompressedVector( const SparseVector<VT,TF>& sv )
   : size_    ( (~sv).size() )                    // The current size/dimension of the compressed vector
   , capacity_( (~sv).nonZeros() )                // The maximum capacity of the compressed vector
   , begin_   ( allocate<Element>( capacity_ ) )  // Pointer to the first non-zero element of the compressed vector
   , end_     ( begin_ )                          // Pointer to the last non-zero element of the compressed vector
{
   using blaze::assign;
   assign( *this, ~sv );
}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The destructor for CompressedVector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CompressedVector<Type,TF>::~CompressedVector()
{
   deallocate( begin_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Subscript operator for the direct access to the compressed vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function returns a reference to the accessed value at position \a index. In case the
// compressed vector does not yet store an element for index \a index, a new element is inserted
// into the compressed vector. An alternative for traversing the non-zero elements of the sparse
// vector are the begin() and end() functions.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::Reference
   CompressedVector<Type,TF>::operator[]( size_t index ) noexcept
{
   BLAZE_USER_ASSERT( index < size_, "Invalid compressed vector access index" );

   return Reference( *this, index );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subscript operator for the direct access to the compressed vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::ConstReference
   CompressedVector<Type,TF>::operator[]( size_t index ) const noexcept
{
   BLAZE_USER_ASSERT( index < size_, "Invalid compressed vector access index" );

   const ConstIterator pos( lowerBound( index ) );

   if( pos == end_ || pos->index_ != index )
      return zero_;
   else
      return pos->value_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the compressed vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid compressed vector access index.
//
// This function returns a reference to the accessed value at position \a index. In case the
// compressed vector does not yet store an element for index \a index, a new element is inserted
// into the compressed vector. In contrast to the subscript operator this function always
// performs a check of the given access index.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::Reference
   CompressedVector<Type,TF>::at( size_t index )
{
   if( index >= size_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid compressed vector access index" );
   }
   return (*this)[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the compressed vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid compressed vector access index.
//
// This function returns a reference to the accessed value at position \a index. In case the
// compressed vector does not yet store an element for index \a index, a new element is inserted
// into the compressed vector. In contrast to the subscript operator this function always
// performs a check of the given access index.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::ConstReference
   CompressedVector<Type,TF>::at( size_t index ) const
{
   if( index >= size_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid compressed vector access index" );
   }
   return (*this)[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first non-zero element of the compressed vector.
//
// \return Iterator to the first non-zero element of the compressed vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::Iterator CompressedVector<Type,TF>::begin() noexcept
{
   return Iterator( begin_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first non-zero element of the compressed vector.
//
// \return Iterator to the first non-zero element of the compressed vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::ConstIterator
   CompressedVector<Type,TF>::begin() const noexcept
{
   return ConstIterator( begin_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first non-zero element of the compressed vector.
//
// \return Iterator to the first non-zero element of the compressed vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::ConstIterator
   CompressedVector<Type,TF>::cbegin() const noexcept
{
   return ConstIterator( begin_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last non-zero element of the compressed vector.
//
// \return Iterator just past the last non-zero element of the compressed vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::Iterator CompressedVector<Type,TF>::end() noexcept
{
   return Iterator( end_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last non-zero element of the compressed vector.
//
// \return Iterator just past the last non-zero element of the compressed vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::ConstIterator
   CompressedVector<Type,TF>::end() const noexcept
{
   return ConstIterator( end_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last non-zero element of the compressed vector.
//
// \return Iterator just past the last non-zero element of the compressed vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::ConstIterator
   CompressedVector<Type,TF>::cend() const noexcept
{
   return ConstIterator( end_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Copy assignment operator for CompressedVector.
//
// \param rhs Compressed vector to be copied.
// \return Reference to the assigned compressed vector.
//
// The compressed vector is resized according to the given compressed vector and initialized
// as a copy of this vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CompressedVector<Type,TF>&
   CompressedVector<Type,TF>::operator=( const CompressedVector& rhs )
{
   if( &rhs == this ) return *this;

   const size_t nonzeros( rhs.nonZeros() );

   if( nonzeros > capacity_ ) {
      Iterator newBegin( allocate<Element>( nonzeros ) );
      end_ = std::copy( rhs.begin_, rhs.end_, newBegin );
      std::swap( begin_, newBegin );
      deallocate( newBegin );

      size_     = rhs.size_;
      capacity_ = nonzeros;
   }
   else {
      end_  = std::copy( rhs.begin_, rhs.end_, begin_ );
      size_ = rhs.size_;
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Move assignment operator for CompressedVector.
//
// \param rhs The compressed vector to be moved into this instance.
// \return Reference to the assigned compressed vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CompressedVector<Type,TF>&
   CompressedVector<Type,TF>::operator=( CompressedVector&& rhs ) noexcept
{
   deallocate( begin_ );

   size_     = rhs.size_;
   capacity_ = rhs.capacity_;
   begin_    = rhs.begin_;
   end_      = rhs.end_;

   rhs.size_     = 0UL;
   rhs.capacity_ = 0UL;
   rhs.begin_    = nullptr;
   rhs.end_      = nullptr;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for dense vectors.
//
// \param rhs Dense vector to be copied.
// \return Reference to the assigned compressed vector.
//
// The vector is resized according to the given dense vector and initialized as a copy of
// this vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline CompressedVector<Type,TF>&
   CompressedVector<Type,TF>::operator=( const DenseVector<VT,TF>& rhs )
{
   using blaze::assign;

   if( (~rhs).canAlias( this ) ) {
      CompressedVector tmp( ~rhs );
      swap( tmp );
   }
   else {
      size_ = (~rhs).size();
      end_  = begin_;
      assign( *this, ~rhs );
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different sparse vectors.
//
// \param rhs Sparse vector to be copied.
// \return Reference to the assigned compressed vector.
//
// The vector is resized according to the given sparse vector and initialized as a copy of
// this vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline CompressedVector<Type,TF>&
   CompressedVector<Type,TF>::operator=( const SparseVector<VT,TF>& rhs )
{
   using blaze::assign;

   if( (~rhs).canAlias( this ) || (~rhs).nonZeros() > capacity_ ) {
      CompressedVector tmp( ~rhs );
      swap( tmp );
   }
   else {
      size_ = (~rhs).size();
      end_  = begin_;
      assign( *this, ~rhs );
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment operator for the addition of a vector (\f$ \vec{a}+=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be added to the compressed vector.
// \return Reference to the compressed vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CompressedVector<Type,TF>& CompressedVector<Type,TF>::operator+=( const Vector<VT,TF>& rhs )
{
   using blaze::addAssign;

   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   addAssign( *this, ~rhs );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment operator for the subtraction of a vector (\f$ \vec{a}-=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be subtracted from the compressed vector.
// \return Reference to the compressed vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CompressedVector<Type,TF>& CompressedVector<Type,TF>::operator-=( const Vector<VT,TF>& rhs )
{
   using blaze::subAssign;

   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   subAssign( *this, ~rhs );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication of a dense vector
//        (\f$ \vec{a}*=\vec{b} \f$).
//
// \param rhs The right-hand side dense vector to be multiplied with the compressed vector.
// \return Reference to the compressed vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CompressedVector<Type,TF>&
   CompressedVector<Type,TF>::operator*=( const DenseVector<VT,TF>& rhs )
{
   using blaze::multAssign;

   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      CompressedVector tmp( *this * (~rhs) );
      swap( tmp );
   }
   else {
      CompositeType_<VT> tmp( ~rhs );
      multAssign( *this, tmp );
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication of a sparse vector
//        (\f$ \vec{a}*=\vec{b} \f$).
//
// \param rhs The right-hand side sparse vector to be multiplied with the compressed vector.
// \return Reference to the compressed vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CompressedVector<Type,TF>&
   CompressedVector<Type,TF>::operator*=( const SparseVector<VT,TF>& rhs )
{
   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   CompressedVector tmp( *this * (~rhs) );
   swap( tmp );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment operator for the division of a dense vector (\f$ \vec{a}/=\vec{b} \f$).
//
// \param rhs The right-hand side dense vector divisor.
// \return Reference to the compressed vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CompressedVector<Type,TF>& CompressedVector<Type,TF>::operator/=( const DenseVector<VT,TF>& rhs )
{
   using blaze::divAssign;

   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      CompressedVector tmp( *this / (~rhs) );
      swap( tmp );
   }
   else {
      CompositeType_<VT> tmp( ~rhs );
      divAssign( *this, tmp );
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication between a compressed vector
//        and a scalar value (\f$ \vec{a}*=s \f$).
//
// \param rhs The right-hand side scalar value for the multiplication.
// \return Reference to the compressed vector.
//
// This operator can only be used for built-in data types. Additionally, the elements of the
// compressed vector must support the multiplication assignment operator for the given scalar
// built-in data type.
*/
template< typename Type     // Data type of the vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, CompressedVector<Type,TF> >&
   CompressedVector<Type,TF>::operator*=( Other rhs )
{
   for( Iterator element=begin_; element!=end_; ++element )
      element->value_ *= rhs;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment operator for the division of a compressed vector by a scalar value
//        (\f$ \vec{a}/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the compressed vector.
//
// This operator can only be used for built-in data types. Additionally, the elements of the
// compressed vector must either support the multiplication assignment operator for the given
// floating point data type or the division assignment operator for the given integral data
// type.
*/
template< typename Type     // Data type of the vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_< IsNumeric<Other>, CompressedVector<Type,TF> >&
   CompressedVector<Type,TF>::operator/=( Other rhs )
{
   BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

   typedef DivTrait_<Type,Other>            DT;
   typedef If_< IsNumeric<DT>, DT, Other >  Tmp;

   // Depending on the two involved data types, an integer division is applied or a
   // floating point division is selected.
   if( IsNumeric<DT>::value && IsFloatingPoint<DT>::value ) {
      const Tmp tmp( Tmp(1)/static_cast<Tmp>( rhs ) );
      for( Iterator element=begin_; element!=end_; ++element )
         element->value_ *= tmp;
   }
   else {
      for( Iterator element=begin_; element!=end_; ++element )
         element->value_ /= rhs;
   }

   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the current size/dimension of the compressed vector.
//
// \return The size of the compressed vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline size_t CompressedVector<Type,TF>::size() const noexcept
{
   return size_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the compressed vector.
//
// \return The capacity of the compressed vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline size_t CompressedVector<Type,TF>::capacity() const noexcept
{
   return capacity_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the compressed vector.
//
// \return The number of non-zero elements in the compressed vector.
//
// Note that the number of non-zero elements is always smaller than the current size of the
// compressed vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline size_t CompressedVector<Type,TF>::nonZeros() const
{
   return end_ - begin_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void CompressedVector<Type,TF>::reset()
{
   end_ = begin_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the compressed vector.
//
// \return void
//
// After the clear() function, the size of the compressed vector is 0.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void CompressedVector<Type,TF>::clear()
{
   size_ = 0UL;
   end_  = begin_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting an element of the compressed vector.
//
// \param index The index of the element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be set.
// \return Reference to the set value.
//
// This function sets the value of an element of the compressed vector. In case the sparse vector
// already contains an element with index \a index its value is modified, else a new element with
// the given \a value is inserted.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::Iterator
   CompressedVector<Type,TF>::set( size_t index, const Type& value )
{
   BLAZE_USER_ASSERT( index < size_, "Invalid compressed vector access index" );

   const Iterator pos( lowerBound( index ) );

   if( pos != end_ && pos->index_ == index ) {
      pos->value() = value;
      return pos;
   }
   else return insert( pos, index, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inserting an element into the compressed vector.
//
// \param index The index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be inserted.
// \return Reference to the inserted value.
// \exception std::invalid_argument Invalid compressed vector access index.
//
// This function inserts a new element into the compressed vector. However, duplicate elements
// are not allowed. In case the sparse vector already contains an element with index \a index,
// a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
typename CompressedVector<Type,TF>::Iterator
   CompressedVector<Type,TF>::insert( size_t index, const Type& value )
{
   BLAZE_USER_ASSERT( index < size_, "Invalid compressed vector access index" );

   const Iterator pos( lowerBound( index ) );

   if( pos != end_ && pos->index_ == index ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Bad access index" );
   }

   return insert( pos, index, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inserting an element into the compressed vector.
//
// \param pos The position of the new element.
// \param index The index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be inserted.
// \return Reference to the inserted value.
// \exception std::invalid_argument Invalid compressed vector access index.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
typename CompressedVector<Type,TF>::Iterator
   CompressedVector<Type,TF>::insert( Iterator pos, size_t index, const Type& value )
{
    if( nonZeros() != capacity_ ) {
      std::move_backward( pos, end_, end_+1 );
      pos->value_ = value;
      pos->index_ = index;
      ++end_;

      return pos;
   }
   else {
      size_t newCapacity( extendCapacity() );

      Iterator newBegin = allocate<Element>( newCapacity );
      Iterator tmp      = std::move( begin_, pos, newBegin );
      tmp->value_ = value;
      tmp->index_ = index;
      end_ = std::move( pos, end_, tmp+1 );

      std::swap( newBegin, begin_ );
      deallocate( newBegin );
      capacity_ = newCapacity;

      return tmp;
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Erasing an element from the compressed vector.
//
// \param index The index of the element to be erased. The index has to be in the range \f$[0..N-1]\f$.
// \return void
//
// This function erases an element from the compressed vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void CompressedVector<Type,TF>::erase( size_t index )
{
   BLAZE_USER_ASSERT( index < size_, "Invalid compressed vector access index" );

   const Iterator pos( find( index ) );
   if( pos != end_ )
      end_ = std::move( pos+1, end_, pos );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Erasing an element from the compressed vector.
//
// \param pos Iterator to the element to be erased.
// \return Iterator to the element after the erased element.
//
// This function erases an element from the compressed vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::Iterator CompressedVector<Type,TF>::erase( Iterator pos )
{
   BLAZE_USER_ASSERT( pos >= begin_ && pos <= end_, "Invalid compressed vector iterator" );

   if( pos != end_ )
      end_ = std::move( pos+1, end_, pos );
   return pos;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Erasing a range of elements from the compressed vector.
//
// \param first Iterator to first element to be erased.
// \param last Iterator just past the last element to be erased.
// \return Iterator to the element after the erased element.
//
// This function erases a range of elements from the compressed vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::Iterator
   CompressedVector<Type,TF>::erase( Iterator first, Iterator last )
{
   BLAZE_USER_ASSERT( first <= last, "Invalid iterator range" );
   BLAZE_USER_ASSERT( first >= begin_ && first <= end_, "Invalid compressed vector iterator" );
   BLAZE_USER_ASSERT( last  >= begin_ && last  <= end_, "Invalid compressed vector iterator" );

   if( first != last )
      end_ = std::move( last, end_, first );
   return first;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the compressed vector.
//
// \param n The new size of the compressed vector.
// \param preserve \a true if the old values of the vector should be preserved, \a false if not.
// \return void
//
// This function resizes the compressed vector using the given size to \a n. During this
// operation, new dynamic memory may be allocated in case the capacity of the compressed
// vector is too small. Note that this function may invalidate all existing views (subvectors,
// ...) on the vector if it is used to shrink the vector. Additionally, the resize operation
// potentially changes all vector elements. In order to preserve the old vector values, the
// \a preserve flag can be set to \a true.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void CompressedVector<Type,TF>::resize( size_t n, bool preserve )
{
   if( preserve ) {
      end_ = lowerBound( n );
   }
   else {
      end_ = begin_;
   }

   size_ = n;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the minimum capacity of the compressed vector.
//
// \param n The new minimum capacity of the compressed vector.
// \return void
//
// This function increases the capacity of the compressed vector to at least \a n elements. The
// current values of the vector elements are preserved.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
void CompressedVector<Type,TF>::reserve( size_t n )
{
   if( n > capacity_ ) {
      const size_t newCapacity( n );

      // Allocating a new data and index array
      Iterator newBegin  = allocate<Element>( newCapacity );

      // Replacing the old data and index array
      end_ = transfer( begin_, end_, newBegin );
      std::swap( newBegin, begin_ );
      capacity_ = newCapacity;
      deallocate( newBegin );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the compressed vector by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the vector scaling.
// \return Reference to the compressed vector.
*/
template< typename Type     // Data type of the vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the scalar value
inline CompressedVector<Type,TF>& CompressedVector<Type,TF>::scale( const Other& scalar )
{
   for( Iterator element=begin_; element!=end_; ++element )
      element->value_ *= scalar;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two compressed vectors.
//
// \param sv The compressed vector to be swapped.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void CompressedVector<Type,TF>::swap( CompressedVector& sv ) noexcept
{
   std::swap( size_, sv.size_ );
   std::swap( capacity_, sv.capacity_ );
   std::swap( begin_, sv.begin_ );
   std::swap( end_, sv.end_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculating a new vector capacity.
//
// \return The new compressed vector capacity.
//
// This function calculates a new vector capacity based on the current capacity of the sparse
// vector. Note that the new capacity is restricted to the interval \f$[7..size]\f$.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline size_t CompressedVector<Type,TF>::extendCapacity() const noexcept
{
   using blaze::max;
   using blaze::min;

   size_t nonzeros( 2UL*capacity_+1UL );
   nonzeros = max( nonzeros, 7UL   );
   nonzeros = min( nonzeros, size_ );

   BLAZE_INTERNAL_ASSERT( nonzeros > capacity_, "Invalid capacity value" );

   return nonzeros;
}
//*************************************************************************************************




//=================================================================================================
//
//  LOOKUP FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Searches for a specific vector element.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the element in case the index is found, end() iterator otherwise.
//
// This function can be used to check whether a specific element is contained in the sparse
// vector. It specifically searches for the element with index \a index. In case the element
// is found, the function returns an iterator to the element. Otherwise an iterator just past
// the last non-zero element of the compressed vector (the end() iterator) is returned. Note
// that the returned compressed vector iterator is subject to invalidation due to inserting
// operations via the subscript operator or the insert() function!
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::Iterator CompressedVector<Type,TF>::find( size_t index )
{
   return const_cast<Iterator>( const_cast<const This&>( *this ).find( index ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Searches for a specific vector element.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the element in case the index is found, end() iterator otherwise.
//
// This function can be used to check whether a specific element is contained in the sparse
// vector. It specifically searches for the element with index \a index. In case the element
// is found, the function returns an iterator to the element. Otherwise an iterator just past
// the last non-zero element of the compressed vector (the end() iterator) is returned. Note
// that the returned compressed vector iterator is subject to invalidation due to inserting
// operations via the subscript operator or the insert() function!
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::ConstIterator CompressedVector<Type,TF>::find( size_t index ) const
{
   const ConstIterator pos( lowerBound( index ) );
   if( pos != end_ && pos->index_ == index )
      return pos;
   else return end_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first index not less then the given index.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index not less then the given index, end() iterator otherwise.
//
// This function returns an iterator to the first element with an index not less then the given
// index. In combination with the upperBound() function this function can be used to create a
// pair of iterators specifying a range of indices. Note that the returned compressed vector
// iterator is subject to invalidation due to inserting operations via the subscript operator
// or the insert() function!
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::Iterator
   CompressedVector<Type,TF>::lowerBound( size_t index )
{
   return const_cast<Iterator>( const_cast<const This&>( *this ).lowerBound( index ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first index not less then the given index.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index not less then the given index, end() iterator otherwise.
//
// This function returns an iterator to the first element with an index not less then the given
// index. In combination with the upperBound() function this function can be used to create a
// pair of iterators specifying a range of indices. Note that the returned compressed vector
// iterator is subject to invalidation due to inserting operations via the subscript operator
// or the insert() function!
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::ConstIterator
   CompressedVector<Type,TF>::lowerBound( size_t index ) const
{
   return std::lower_bound( begin_, end_, index,
                            []( const Element& element, size_t i )
                            {
                               return element.index() < i;
                            } );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first index greater then the given index.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index greater then the given index, end() iterator otherwise.
//
// This function returns an iterator to the first element with an index greater then the given
// index. In combination with the lowerBound() function this function can be used to create a
// pair of iterators specifying a range of indices. Note that the returned compressed vector
// iterator is subject to invalidation due to inserting operations via the subscript operator
// or the insert() function!
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::Iterator
   CompressedVector<Type,TF>::upperBound( size_t index )
{
   return const_cast<Iterator>( const_cast<const This&>( *this ).upperBound( index ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first index greater then the given index.
//
// \param index The index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index greater then the given index, end() iterator otherwise.
//
// This function returns an iterator to the first element with an index greater then the given
// index. In combination with the lowerBound() function this function can be used to create a
// pair of iterators specifying a range of indices. Note that the returned compressed vector
// iterator is subject to invalidation due to inserting operations via the subscript operator
// or the insert() function!
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename CompressedVector<Type,TF>::ConstIterator
   CompressedVector<Type,TF>::upperBound( size_t index ) const
{
   return std::upper_bound( begin_, end_, index,
                            []( size_t i, const Element& element )
                            {
                               return i < element.index();
                            } );
}
//*************************************************************************************************




//=================================================================================================
//
//  LOW-LEVEL UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Appending an element to the compressed vector.
//
// \param index The index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be appended.
// \param check \a true if the new value should be checked for default values, \a false if not.
// \return void
//
// This function provides a very efficient way to fill a compressed vector with elements. It
// appends a new element to the end of the compressed vector without any memory allocation.
// Therefore it is strictly necessary to keep the following preconditions in mind:
//
//  - the index of the new element must be strictly larger than the largest index of non-zero
//    elements in the compressed vector
//  - the current number of non-zero elements must be smaller than the capacity of the vector
//
// Ignoring these preconditions might result in undefined behavior! The optional \a check
// parameter specifies whether the new value should be tested for a default value. If the new
// value is a default value (for instance 0 in case of an integral element type) the value is
// not appended. Per default the values are not tested.
//
// \note Although append() does not allocate new memory, it still invalidates all iterators
// returned by the end() functions!
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void CompressedVector<Type,TF>::append( size_t index, const Type& value, bool check )
{
   BLAZE_USER_ASSERT( index < size_, "Invalid compressed vector access index" );
   BLAZE_USER_ASSERT( nonZeros() < capacity(), "Not enough reserved capacity" );
   BLAZE_USER_ASSERT( begin_ == end_ || (end_-1UL)->index_ < index, "Index is not strictly increasing" );

   end_->value_ = value;

   if( !check || !isDefault( end_->value_ ) ) {
      end_->index_ = index;
      ++end_;
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  EXPRESSION TEMPLATE EVALUATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the vector can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this vector, \a false if not.
//
// This function returns whether the given address can alias with the vector. In contrast
// to the isAliased() function this function is allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type     // Data type of the vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool CompressedVector<Type,TF>::canAlias( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the vector is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this vector, \a false if not.
//
// This function returns whether the given address is aliased with the vector. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type     // Data type of the vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool CompressedVector<Type,TF>::isAliased( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the vector can be used in SMP assignments.
//
// \return \a true in case the vector can be used in SMP assignments, \a false if not.
//
// This function returns whether the vector can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current size of the
// vector).
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline bool CompressedVector<Type,TF>::canSMPAssign() const noexcept
{
   return false;
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline void CompressedVector<Type,TF>::assign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );
   BLAZE_INTERNAL_ASSERT( nonZeros() == 0UL, "Invalid non-zero elements detected" );

   size_t nonzeros( 0UL );

   for( size_t i=0UL; i<size_; ++i )
   {
      if( nonzeros == capacity_ )
         reserve( extendCapacity() );

      end_->value_ = (~rhs)[i];

      if( !isDefault( end_->value_ ) ) {
         end_->index_ = i;
         ++end_;
         ++nonzeros;
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CompressedVector<Type,TF>::assign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );
   BLAZE_INTERNAL_ASSERT( nonZeros() == 0UL, "Invalid non-zero elements detected" );

   // Using the following formulation instead of a std::copy function call of the form
   //
   //          end_ = std::copy( (~rhs).begin(), (~rhs).end(), begin_ );
   //
   // results in much less requirements on the ConstIterator type provided from the right-hand
   // sparse vector type
   for( ConstIterator_<VT> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      append( element->index(), element->value() );
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline void CompressedVector<Type,TF>::addAssign( const DenseVector<VT,TF>& rhs )
{
   typedef AddTrait_< This, ResultType_<VT> >  AddType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( AddType );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( AddType, TF );
   BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<AddType> );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const AddType tmp( serial( *this + (~rhs) ) );
   reset();
   assign( tmp );
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CompressedVector<Type,TF>::addAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   CompressedVector tmp( serial( *this + (~rhs) ) );
   swap( tmp );
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline void CompressedVector<Type,TF>::subAssign( const DenseVector<VT,TF>& rhs )
{
   typedef SubTrait_< This, ResultType_<VT> >  SubType;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( SubType );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( SubType, TF );
   BLAZE_CONSTRAINT_MUST_BE_REFERENCE_TYPE( CompositeType_<SubType> );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const SubType tmp( serial( *this - (~rhs) ) );
   reset();
   assign( tmp );
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CompressedVector<Type,TF>::subAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   CompressedVector tmp( serial( *this - (~rhs) ) );
   swap( tmp );
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline void CompressedVector<Type,TF>::multAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );

   for( Iterator element=begin_; element!=end_; ++element ) {
      element->value_ *= (~rhs)[element->index_];
   }
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline void CompressedVector<Type,TF>::divAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );

   for( Iterator element=begin_; element!=end_; ++element ) {
      element->value_ /= (~rhs)[element->index_];
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  COMPRESSEDVECTOR OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name CompressedVector operators */
//@{
template< typename Type, bool TF >
inline void reset( CompressedVector<Type,TF>& v );

template< typename Type, bool TF >
inline void clear( CompressedVector<Type,TF>& v );

template< typename Type, bool TF >
inline bool isDefault( const CompressedVector<Type,TF>& v );

template< typename Type, bool TF >
inline bool isIntact( const CompressedVector<Type,TF>& v ) noexcept;

template< typename Type, bool TF >
inline void swap( CompressedVector<Type,TF>& a, CompressedVector<Type,TF>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given compressed vector.
// \ingroup compressed_vector
//
// \param v The compressed vector to be resetted.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void reset( CompressedVector<Type,TF>& v )
{
   v.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given compressed vector.
// \ingroup compressed_vector
//
// \param v The compressed vector to be cleared.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void clear( CompressedVector<Type,TF>& v )
{
   v.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given compressed vector is in default state.
// \ingroup compressed_vector
//
// \param v The compressed vector to be tested for its default state.
// \return \a true in case the given vector's size is zero, \a false otherwise.
//
// This function checks whether the compressed vector is in default (constructed) state, i.e. if
// it's size is 0. In case it is in default state, the function returns \a true, else it will
// return \a false. The following example demonstrates the use of the \a isDefault() function:

   \code
   blaze::CompressedVector<double> a;
   // ... Resizing and initialization
   if( isDefault( a ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline bool isDefault( const CompressedVector<Type,TF>& v )
{
   return ( v.size() == 0UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given compressed vector are intact.
// \ingroup compressed_vector
//
// \param v The compressed vector to be tested.
// \return \a true in case the given vector's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the compressed vector are intact, i.e. if
// its state is valid. In case the invariants are intact, the function returns \a true, else
// it will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   blaze::CompressedVector<double> a;
   // ... Resizing and initialization
   if( isIntact( a ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline bool isIntact( const CompressedVector<Type,TF>& v ) noexcept
{
   return ( v.nonZeros() <= v.capacity() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two compressed vectors.
// \ingroup compressed_vector
//
// \param a The first compressed vector to be swapped.
// \param b The second compressed vector to be swapped.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void swap( CompressedVector<Type,TF>& a, CompressedVector<Type,TF>& b ) noexcept
{
   a.swap( b );
}
//*************************************************************************************************




//=================================================================================================
//
//  ISRESIZABLE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool TF >
struct IsResizable< CompressedVector<T,TF> > : public TrueType
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
template< typename T1, bool TF, typename T2, size_t N >
struct AddTrait< CompressedVector<T1,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< AddTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct AddTrait< StaticVector<T1,N,TF>, CompressedVector<T2,TF> >
{
   using Type = StaticVector< AddTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool TF, typename T2, size_t N >
struct AddTrait< CompressedVector<T1,TF>, HybridVector<T2,N,TF> >
{
   using Type = HybridVector< AddTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct AddTrait< HybridVector<T1,N,TF>, CompressedVector<T2,TF> >
{
   using Type = HybridVector< AddTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool TF, typename T2 >
struct AddTrait< CompressedVector<T1,TF>, DynamicVector<T2,TF> >
{
   using Type = DynamicVector< AddTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2 >
struct AddTrait< DynamicVector<T1,TF>, CompressedVector<T2,TF> >
{
   using Type = DynamicVector< AddTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2, bool AF, bool PF >
struct AddTrait< CompressedVector<T1,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = DynamicVector< AddTrait_<T1,T2>, TF >;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2 >
struct AddTrait< CustomVector<T1,AF,PF,TF>, CompressedVector<T2,TF> >
{
   using Type = DynamicVector< AddTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2 >
struct AddTrait< CompressedVector<T1,TF>, CompressedVector<T2,TF> >
{
   using Type = CompressedVector< AddTrait_<T1,T2>, TF >;
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
template< typename T1, bool TF, typename T2, size_t N >
struct SubTrait< CompressedVector<T1,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< SubTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct SubTrait< StaticVector<T1,N,TF>, CompressedVector<T2,TF> >
{
   using Type = StaticVector< SubTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool TF, typename T2, size_t N >
struct SubTrait< CompressedVector<T1,TF>, HybridVector<T2,N,TF> >
{
   using Type = HybridVector< SubTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct SubTrait< HybridVector<T1,N,TF>, CompressedVector<T2,TF> >
{
   using Type = HybridVector< SubTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool TF, typename T2 >
struct SubTrait< CompressedVector<T1,TF>, DynamicVector<T2,TF> >
{
   using Type = DynamicVector< SubTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2 >
struct SubTrait< DynamicVector<T1,TF>, CompressedVector<T2,TF> >
{
   using Type = DynamicVector< SubTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2, bool AF, bool PF >
struct SubTrait< CompressedVector<T1,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = DynamicVector< SubTrait_<T1,T2>, TF >;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2 >
struct SubTrait< CustomVector<T1,AF,PF,TF>, CompressedVector<T2,TF> >
{
   using Type = DynamicVector< SubTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2 >
struct SubTrait< CompressedVector<T1,TF>, CompressedVector<T2,TF> >
{
   using Type = CompressedVector< SubTrait_<T1,T2>, TF >;
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
template< typename T1, bool TF, typename T2 >
struct MultTrait< CompressedVector<T1,TF>, T2, EnableIf_< IsNumeric<T2> > >
{
   using Type = CompressedVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, typename T2, bool TF >
struct MultTrait< T1, CompressedVector<T2,TF>, EnableIf_< IsNumeric<T1> > >
{
   using Type = CompressedVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2, size_t N >
struct MultTrait< CompressedVector<T1,TF>, StaticVector<T2,N,TF> >
{
   using Type = CompressedVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, typename T2, size_t N >
struct MultTrait< CompressedVector<T1,false>, StaticVector<T2,N,true> >
{
   using Type = CompressedMatrix< MultTrait_<T1,T2>, true >;
};

template< typename T1, typename T2, size_t N >
struct MultTrait< CompressedVector<T1,true>, StaticVector<T2,N,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct MultTrait< StaticVector<T1,N,TF>, CompressedVector<T2,TF> >
{
   using Type = CompressedVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, size_t N, typename T2 >
struct MultTrait< StaticVector<T1,N,false>, CompressedVector<T2,true> >
{
   using Type = CompressedMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, size_t N, typename T2 >
struct MultTrait< StaticVector<T1,N,true>, CompressedVector<T2,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, bool TF, typename T2, size_t N >
struct MultTrait< CompressedVector<T1,TF>, HybridVector<T2,N,TF> >
{
   using Type = CompressedVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, typename T2, size_t N >
struct MultTrait< CompressedVector<T1,false>, HybridVector<T2,N,true> >
{
   using Type = CompressedMatrix< MultTrait_<T1,T2>, true >;
};

template< typename T1, typename T2, size_t N >
struct MultTrait< CompressedVector<T1,true>, HybridVector<T2,N,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct MultTrait< HybridVector<T1,N,TF>, CompressedVector<T2,TF> >
{
   using Type = CompressedVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, size_t N, typename T2 >
struct MultTrait< HybridVector<T1,N,false>, CompressedVector<T2,true> >
{
   using Type = CompressedMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, size_t N, typename T2 >
struct MultTrait< HybridVector<T1,N,true>, CompressedVector<T2,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, bool TF, typename T2 >
struct MultTrait< CompressedVector<T1,TF>, DynamicVector<T2,TF> >
{
   using Type = CompressedVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, typename T2 >
struct MultTrait< CompressedVector<T1,false>, DynamicVector<T2,true> >
{
   using Type = CompressedMatrix< MultTrait_<T1,T2>, true >;
};

template< typename T1, typename T2 >
struct MultTrait< CompressedVector<T1,true>, DynamicVector<T2,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, bool TF, typename T2 >
struct MultTrait< DynamicVector<T1,TF>, CompressedVector<T2,TF> >
{
   using Type = CompressedVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, typename T2 >
struct MultTrait< DynamicVector<T1,false>, CompressedVector<T2,true> >
{
   using Type = CompressedMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, typename T2 >
struct MultTrait< DynamicVector<T1,true>, CompressedVector<T2,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, bool TF, typename T2, bool AF, bool PF >
struct MultTrait< CompressedVector<T1,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = CompressedVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, typename T2, bool AF, bool PF >
struct MultTrait< CompressedVector<T1,false>, CustomVector<T2,AF,PF,true> >
{
   using Type = CompressedMatrix< MultTrait_<T1,T2>, true >;
};

template< typename T1, typename T2, bool AF, bool PF >
struct MultTrait< CompressedVector<T1,true>, CustomVector<T2,AF,PF,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2 >
struct MultTrait< CustomVector<T1,AF,PF,TF>, CompressedVector<T2,TF> >
{
   using Type = CompressedVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, bool AF, bool PF, typename T2 >
struct MultTrait< CustomVector<T1,AF,PF,false>, CompressedVector<T2,true> >
{
   using Type = CompressedMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, bool AF, bool PF, typename T2 >
struct MultTrait< CustomVector<T1,AF,PF,true>, CompressedVector<T2,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, bool TF, typename T2 >
struct MultTrait< CompressedVector<T1,TF>, CompressedVector<T2,TF> >
{
   using Type = CompressedVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, typename T2 >
struct MultTrait< CompressedVector<T1,false>, CompressedVector<T2,true> >
{
   using Type = CompressedMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, typename T2 >
struct MultTrait< CompressedVector<T1,true>, CompressedVector<T2,false> >
{
   using Type = MultTrait_<T1,T2>;
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
template< typename T1, bool TF, typename T2 >
struct CrossTrait< CompressedVector<T1,TF>, StaticVector<T2,3UL,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, bool TF, typename T2 >
struct CrossTrait< StaticVector<T1,3UL,TF>, CompressedVector<T2,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, bool TF, typename T2, size_t N >
struct CrossTrait< CompressedVector<T1,TF>, HybridVector<T2,N,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct CrossTrait< HybridVector<T1,N,TF>, CompressedVector<T2,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, bool TF, typename T2 >
struct CrossTrait< CompressedVector<T1,TF>, DynamicVector<T2,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, bool TF, typename T2 >
struct CrossTrait< DynamicVector<T1,TF>, CompressedVector<T2,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, bool TF, typename T2 >
struct CrossTrait< CompressedVector<T1,TF>, CompressedVector<T2,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
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
template< typename T1, bool TF, typename T2 >
struct DivTrait< CompressedVector<T1,TF>, T2, EnableIf_< IsNumeric<T2> > >
{
   using Type = CompressedVector< DivTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2, size_t N >
struct DivTrait< CompressedVector<T1,TF>, StaticVector<T2,N,TF> >
{
   using Type = CompressedVector< DivTrait_<T1,T2>, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct DivTrait< StaticVector<T1,N,TF>, CompressedVector<T2,TF> >
{
   using Type = CompressedVector< DivTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2, size_t N >
struct DivTrait< CompressedVector<T1,TF>, HybridVector<T2,N,TF> >
{
   using Type = CompressedVector< DivTrait_<T1,T2>, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct DivTrait< HybridVector<T1,N,TF>, CompressedVector<T2,TF> >
{
   using Type = CompressedVector< DivTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2 >
struct DivTrait< CompressedVector<T1,TF>, DynamicVector<T2,TF> >
{
   using Type = CompressedVector< DivTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2 >
struct DivTrait< DynamicVector<T1,TF>, CompressedVector<T2,TF> >
{
   using Type = CompressedVector< DivTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2, bool AF, bool PF >
struct DivTrait< CompressedVector<T1,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = CompressedVector< DivTrait_<T1,T2>, TF >;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2 >
struct DivTrait< CustomVector<T1,AF,PF,TF>, CompressedVector<T2,TF> >
{
   using Type = CompressedVector< DivTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2 >
struct DivTrait< CompressedVector<T1,TF>, CompressedVector<T2,TF> >
{
   using Type = CompressedVector< DivTrait_<T1,T2>, TF >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MATHTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, bool TF, typename T2 >
struct MathTrait< CompressedVector<T1,TF>, CompressedVector<T2,TF> >
{
   using HighType = CompressedVector< typename MathTrait<T1,T2>::HighType, TF >;
   using LowType  = CompressedVector< typename MathTrait<T1,T2>::LowType , TF >;
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
template< typename T1, bool TF >
struct SubvectorTrait< CompressedVector<T1,TF> >
{
   using Type = CompressedVector<T1,TF>;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
