//=================================================================================================
/*!
//  \file blaze/math/dense/DynamicVector.h
//  \brief Header file for the implementation of an arbitrarily sized vector
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

#ifndef _BLAZE_MATH_DENSE_DYNAMICVECTOR_H_
#define _BLAZE_MATH_DENSE_DYNAMICVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <utility>
#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/dense/DenseIterator.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/SparseVector.h>
#include <blaze/math/Forward.h>
#include <blaze/math/Functions.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/CrossTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/MathTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/traits/SubvectorTrait.h>
#include <blaze/math/typetraits/AreSIMDCombinable.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDDiv.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/HasSIMDSub.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsSMPAssignable.h>
#include <blaze/math/typetraits/IsSparseVector.h>
#include <blaze/system/CacheSize.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Restrict.h>
#include <blaze/system/Thresholds.h>
#include <blaze/system/TransposeFlag.h>
#include <blaze/util/Algorithm.h>
#include <blaze/util/AlignmentCheck.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Vectorizable.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/Memory.h>
#include <blaze/util/Template.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsVectorizable.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup dynamic_vector DynamicVector
// \ingroup dense_vector
*/
/*!\brief Efficient implementation of an arbitrary sized vector.
// \ingroup dynamic_vector
//
// The DynamicVector class template is the representation of an arbitrary sized vector with
// dynamically allocated elements of arbitrary type. The type of the elements and the transpose
// flag of the vector can be specified via the two template parameters:

   \code
   template< typename Type, bool TF >
   class DynamicVector;
   \endcode

//  - Type: specifies the type of the vector elements. DynamicVector can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
//  - TF  : specifies whether the vector is a row vector (\a blaze::rowVector) or a column
//          vector (\a blaze::columnVector). The default value is \a blaze::columnVector.
//
// These contiguously stored elements can be directly accessed with the subscript operator. The
// numbering of the vector elements is

                             \f[\left(\begin{array}{*{5}{c}}
                             0 & 1 & 2 & \cdots & N-1 \\
                             \end{array}\right)\f]

// The use of DynamicVector is very natural and intuitive. All operations (addition, subtraction,
// multiplication, scaling, ...) can be performed on all possible combinations of dense and sparse
// vectors with fitting element types. The following example gives an impression of the use of
// DynamicVector:

   \code
   using blaze::DynamicVector;
   using blaze::CompressedVector;
   using blaze::DynamicMatrix;

   DynamicVector<double> a( 2 );  // Non-initialized 2D vector of size 2
   a[0] = 1.0;                    // Initialization of the first element
   a[1] = 2.0;                    // Initialization of the second element

   DynamicVector<double>   b( 2, 2.0  );  // Directly, homogeneously initialized 2D vector
   CompressedVector<float> c( 2 );        // Empty sparse single precision vector
   DynamicVector<double>   d;             // Default constructed dynamic vector
   DynamicMatrix<double>   A;             // Default constructed row-major matrix

   d = a + b;  // Vector addition between vectors of equal element type
   d = a - c;  // Vector subtraction between a dense and sparse vector with different element types
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
class DynamicVector : public DenseVector< DynamicVector<Type,TF>, TF >
{
 public:
   //**Type definitions****************************************************************************
   typedef DynamicVector<Type,TF>   This;           //!< Type of this DynamicVector instance.
   typedef DenseVector<This,TF>     BaseType;       //!< Base type of this DynamicVector instance.
   typedef This                     ResultType;     //!< Result type for expression template evaluations.
   typedef DynamicVector<Type,!TF>  TransposeType;  //!< Transpose type for expression template evaluations.
   typedef Type                     ElementType;    //!< Type of the vector elements.
   typedef SIMDTrait_<ElementType>  SIMDType;       //!< SIMD type of the vector elements.
   typedef const Type&              ReturnType;     //!< Return type for expression template evaluations
   typedef const DynamicVector&     CompositeType;  //!< Data type for composite expression templates.

   typedef Type&        Reference;       //!< Reference to a non-constant vector value.
   typedef const Type&  ConstReference;  //!< Reference to a constant vector value.
   typedef Type*        Pointer;         //!< Pointer to a non-constant vector value.
   typedef const Type*  ConstPointer;    //!< Pointer to a constant vector value.

   typedef DenseIterator<Type,aligned>        Iterator;       //!< Iterator over non-constant elements.
   typedef DenseIterator<const Type,aligned>  ConstIterator;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a DynamicVector with different data/element type.
   */
   template< typename ET >  // Data type of the other vector
   struct Rebind {
      typedef DynamicVector<ET,TF>  Other;  //!< The type of the other DynamicVector.
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SIMD optimization.
   /*! The \a simdEnabled compilation flag indicates whether expressions the vector is involved
       in can be optimized via SIMD operationss. In case the element type of the vector is a
       vectorizable data type, the \a simdEnabled compilation flag is set to \a true, otherwise
       it is set to \a false. */
   enum : bool { simdEnabled = IsVectorizable<Type>::value };

   //! Compilation flag for SMP assignments.
   /*! The \a smpAssignable compilation flag indicates whether the vector can be used in SMP
       (shared memory parallel) assignments (both on the left-hand and right-hand side of the
       assignment). */
   enum : bool { smpAssignable = !IsSMPAssignable<Type>::value };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline DynamicVector() noexcept;
   explicit inline DynamicVector( size_t n );
   explicit inline DynamicVector( size_t n, const Type& init );
   explicit inline DynamicVector( initializer_list<Type> list );

   template< typename Other >
   explicit inline DynamicVector( size_t n, const Other* array );

   template< typename Other, size_t N >
   explicit inline DynamicVector( const Other (&array)[N] );

                           inline DynamicVector( const DynamicVector& v );
                           inline DynamicVector( DynamicVector&& v ) noexcept;
   template< typename VT > inline DynamicVector( const Vector<VT,TF>& v );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~DynamicVector();
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator[]( size_t index ) noexcept;
   inline ConstReference operator[]( size_t index ) const noexcept;
   inline Reference      at( size_t index );
   inline ConstReference at( size_t index ) const;
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
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
   inline DynamicVector& operator=( const Type& rhs );
   inline DynamicVector& operator=( initializer_list<Type> list );

   template< typename Other, size_t N >
   inline DynamicVector& operator=( const Other (&array)[N] );

   inline DynamicVector& operator=( const DynamicVector& rhs );
   inline DynamicVector& operator=( DynamicVector&& rhs ) noexcept;

   template< typename VT > inline DynamicVector& operator= ( const Vector<VT,TF>& rhs );
   template< typename VT > inline DynamicVector& operator+=( const Vector<VT,TF>& rhs );
   template< typename VT > inline DynamicVector& operator-=( const Vector<VT,TF>& rhs );
   template< typename VT > inline DynamicVector& operator*=( const Vector<VT,TF>& rhs );
   template< typename VT > inline DynamicVector& operator/=( const DenseVector<VT,TF>& rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, DynamicVector >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, DynamicVector >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t         size() const noexcept;
                              inline size_t         capacity() const noexcept;
                              inline size_t         nonZeros() const;
                              inline void           reset();
                              inline void           clear();
                              inline void           resize( size_t n, bool preserve=true );
                              inline void           extend( size_t n, bool preserve=true );
                              inline void           reserve( size_t n );
   template< typename Other > inline DynamicVector& scale( const Other& scalar );
                              inline void           swap( DynamicVector& v ) noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT >
   struct VectorizedAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<VT> >::value };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT >
   struct VectorizedAddAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<VT> >::value &&
                            HasSIMDAdd< Type, ElementType_<VT> >::value };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT >
   struct VectorizedSubAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<VT> >::value &&
                            HasSIMDSub< Type, ElementType_<VT> >::value };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT >
   struct VectorizedMultAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<VT> >::value &&
                            HasSIMDMult< Type, ElementType_<VT> >::value };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT >
   struct VectorizedDivAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<VT> >::value &&
                            HasSIMDDiv< Type, ElementType_<VT> >::value };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   //! The number of elements packed within a single SIMD element.
   enum : size_t { SIMDSIZE = SIMDTrait<ElementType>::size };
   //**********************************************************************************************

 public:
   //**Debugging functions*************************************************************************
   /*!\name Debugging functions */
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

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t index ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t index ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t index ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t index, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t index, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t index, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t index, const SIMDType& value ) noexcept;

   template< typename VT >
   inline DisableIf_<VectorizedAssign<VT> > assign( const DenseVector<VT,TF>& rhs );

   template< typename VT >
   inline EnableIf_<VectorizedAssign<VT> > assign( const DenseVector<VT,TF>& rhs );

   template< typename VT > inline void assign( const SparseVector<VT,TF>& rhs );

   template< typename VT >
   inline DisableIf_<VectorizedAddAssign<VT> > addAssign( const DenseVector<VT,TF>& rhs );

   template< typename VT >
   inline EnableIf_<VectorizedAddAssign<VT> > addAssign( const DenseVector<VT,TF>& rhs );

   template< typename VT > inline void addAssign( const SparseVector<VT,TF>& rhs );

   template< typename VT >
   inline DisableIf_<VectorizedSubAssign<VT> > subAssign( const DenseVector<VT,TF>& rhs );

   template< typename VT >
   inline EnableIf_<VectorizedSubAssign<VT> > subAssign( const DenseVector<VT,TF>& rhs );

   template< typename VT > inline void subAssign( const SparseVector<VT,TF>& rhs );

   template< typename VT >
   inline DisableIf_<VectorizedMultAssign<VT> > multAssign( const DenseVector<VT,TF>& rhs );

   template< typename VT >
   inline EnableIf_<VectorizedMultAssign<VT> > multAssign( const DenseVector<VT,TF>& rhs );

   template< typename VT > inline void multAssign( const SparseVector<VT,TF>& rhs );

   template< typename VT >
   inline DisableIf_<VectorizedDivAssign<VT> > divAssign( const DenseVector<VT,TF>& rhs );

   template< typename VT >
   inline EnableIf_<VectorizedDivAssign<VT> > divAssign( const DenseVector<VT,TF>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t adjustCapacity( size_t minCapacity ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t size_;             //!< The current size/dimension of the vector.
   size_t capacity_;         //!< The maximum capacity of the vector.
   Type* BLAZE_RESTRICT v_;  //!< The dynamically allocated vector elements.
                             /*!< Access to the vector elements is gained via the subscript operator.
                                  The order of the elements is
                                  \f[\left(\begin{array}{*{5}{c}}
                                  0 & 1 & 2 & \cdots & N-1 \\
                                  \end{array}\right)\f] */
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE  ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST         ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE      ( Type );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The default constructor for DynamicVector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline DynamicVector<Type,TF>::DynamicVector() noexcept
   : size_    ( 0UL )      // The current size/dimension of the vector
   , capacity_( 0UL )      // The maximum capacity of the vector
   , v_       ( nullptr )  // The vector elements
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a vector of size \a n. No element initialization is performed!
//
// \param n The size of the vector.
//
// \note This constructor is only responsible to allocate the required dynamic memory. No
// element initialization is performed!
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline DynamicVector<Type,TF>::DynamicVector( size_t n )
   : size_    ( n )                            // The current size/dimension of the vector
   , capacity_( adjustCapacity( n ) )          // The maximum capacity of the vector
   , v_       ( allocate<Type>( capacity_ ) )  // The vector elements
{
   if( IsVectorizable<Type>::value ) {
      for( size_t i=size_; i<capacity_; ++i )
         v_[i] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a homogeneous initialization of all \a n vector elements.
//
// \param n The size of the vector.
// \param init The initial value of the vector elements.
//
// All vector elements are initialized with the specified value.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline DynamicVector<Type,TF>::DynamicVector( size_t n, const Type& init )
   : size_    ( n )                            // The current size/dimension of the vector
   , capacity_( adjustCapacity( n ) )          // The maximum capacity of the vector
   , v_       ( allocate<Type>( capacity_ ) )  // The vector elements
{
   for( size_t i=0UL; i<size_; ++i )
      v_[i] = init;

   if( IsVectorizable<Type>::value ) {
      for( size_t i=size_; i<capacity_; ++i )
         v_[i] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List initialization of all vector elements.
//
// \param list The initializer list.
//
// This assignment operator provides the option to explicitly initialize the elements of the
// vector within a constructor call:

   \code
   blaze::DynamicVector<double> v1{ 4.2, 6.3, -1.2 };
   \endcode

// The vector is sized according to the size of the initializer list and all its elements are
// initialized by the values of the given initializer list.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline DynamicVector<Type,TF>::DynamicVector( initializer_list<Type> list )
   : size_    ( list.size() )                  // The current size/dimension of the vector
   , capacity_( adjustCapacity( size_ ) )      // The maximum capacity of the vector
   , v_       ( allocate<Type>( capacity_ ) )  // The vector elements
{
   std::fill( std::copy( list.begin(), list.end(), v_ ), v_+capacity_, Type() );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array initialization of all vector elements.
//
// \param n The size of the vector.
// \param array Dynamic array for the initialization.
//
// This assignment operator offers the option to directly initialize the elements of the vector
// with a dynamic array:

   \code
   double* array = new double[4];
   // ... Initialization of the dynamic array
   blaze::DynamicVector<double> v( array, 4UL );
   delete[] array;
   \endcode

// The vector is sized according to the specified size of the array and initialized with the
// values from the given array. Note that it is expected that the given \a array has at least
// \a n elements. Providing an array with less elements results in undefined behavior!
*/
template< typename Type     // Data type of the vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the initialization array
inline DynamicVector<Type,TF>::DynamicVector( size_t n, const Other* array )
   : size_    ( n )                            // The current size/dimension of the vector
   , capacity_( adjustCapacity( n ) )          // The maximum capacity of the vector
   , v_       ( allocate<Type>( capacity_ ) )  // The vector elements
{
   for( size_t i=0UL; i<n; ++i )
      v_[i] = array[i];

   if( IsVectorizable<Type>::value ) {
      for( size_t i=n; i<capacity_; ++i )
         v_[i] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array initialization of all vector elements.
//
// \param array N-dimensional array for the initialization.
//
// This constructor offers the option to directly initialize the elements of the vector with a
// static array:

   \code
   const int init[4] = { 1, 2, 3 };
   blaze::DynamicVector<int> v( init );
   \endcode

// The vector is sized according to the size of the array and initialized with the values from the
// given array. Missing values are initialized with default values (as e.g. the fourth element in
// the example).
*/
template< typename Type   // Data type of the vector
        , bool TF >       // Transpose flag
template< typename Other  // Data type of the initialization array
        , size_t N >      // Dimension of the initialization array
inline DynamicVector<Type,TF>::DynamicVector( const Other (&array)[N] )
   : size_    ( N )                            // The current size/dimension of the vector
   , capacity_( adjustCapacity( N ) )          // The maximum capacity of the vector
   , v_       ( allocate<Type>( capacity_ ) )  // The vector elements
{
   for( size_t i=0UL; i<N; ++i )
      v_[i] = array[i];

   if( IsVectorizable<Type>::value ) {
      for( size_t i=N; i<capacity_; ++i )
         v_[i] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for DynamicVector.
//
// \param v Vector to be copied.
//
// The copy constructor is explicitly defined due to the required dynamic memory management
// and in order to enable/facilitate NRV optimization.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline DynamicVector<Type,TF>::DynamicVector( const DynamicVector& v )
   : size_    ( v.size_ )                      // The current size/dimension of the vector
   , capacity_( adjustCapacity( v.size_ ) )    // The maximum capacity of the vector
   , v_       ( allocate<Type>( capacity_ ) )  // The vector elements
{
   BLAZE_INTERNAL_ASSERT( capacity_ <= v.capacity_, "Invalid capacity estimation" );

   for( size_t i=0UL; i<capacity_; ++i )
      v_[i] = v.v_[i];

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The move constructor for DynamicVector.
//
// \param v The vector to be moved into this instance.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline DynamicVector<Type,TF>::DynamicVector( DynamicVector&& v ) noexcept
   : size_    ( v.size_     )  // The current size/dimension of the vector
   , capacity_( v.capacity_ )  // The maximum capacity of the vector
   , v_       ( v.v_        )  // The vector elements
{
   v.size_     = 0UL;
   v.capacity_ = 0UL;
   v.v_        = nullptr;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different vectors.
//
// \param v Vector to be copied.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the foreign vector
inline DynamicVector<Type,TF>::DynamicVector( const Vector<VT,TF>& v )
   : size_    ( (~v).size() )                  // The current size/dimension of the vector
   , capacity_( adjustCapacity( size_ ) )      // The maximum capacity of the vector
   , v_       ( allocate<Type>( capacity_ ) )  // The vector elements
{
   for( size_t i=( IsSparseVector<VT>::value   ? 0UL       : size_ );
               i<( IsVectorizable<Type>::value ? capacity_ : size_ ); ++i ) {
      v_[i] = Type();
   }

   smpAssign( *this, ~v );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The destructor for DynamicVector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline DynamicVector<Type,TF>::~DynamicVector()
{
   deallocate( v_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Subscript operator for the direct access to the vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename DynamicVector<Type,TF>::Reference
   DynamicVector<Type,TF>::operator[]( size_t index ) noexcept
{
   BLAZE_USER_ASSERT( index < size_, "Invalid vector access index" );
   return v_[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subscript operator for the direct access to the vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename DynamicVector<Type,TF>::ConstReference
   DynamicVector<Type,TF>::operator[]( size_t index ) const noexcept
{
   BLAZE_USER_ASSERT( index < size_, "Invalid vector access index" );
   return v_[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid vector access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename DynamicVector<Type,TF>::Reference
   DynamicVector<Type,TF>::at( size_t index )
{
   if( index >= size_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid vector access index" );
   }
   return (*this)[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid vector access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename DynamicVector<Type,TF>::ConstReference
   DynamicVector<Type,TF>::at( size_t index ) const
{
   if( index >= size_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid vector access index" );
   }
   return (*this)[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the vector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename DynamicVector<Type,TF>::Pointer DynamicVector<Type,TF>::data() noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the vector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename DynamicVector<Type,TF>::ConstPointer DynamicVector<Type,TF>::data() const noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the dynamic vector.
//
// \return Iterator to the first element of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename DynamicVector<Type,TF>::Iterator DynamicVector<Type,TF>::begin() noexcept
{
   return Iterator( v_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the dynamic vector.
//
// \return Iterator to the first element of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename DynamicVector<Type,TF>::ConstIterator DynamicVector<Type,TF>::begin() const noexcept
{
   return ConstIterator( v_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the dynamic vector.
//
// \return Iterator to the first element of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename DynamicVector<Type,TF>::ConstIterator DynamicVector<Type,TF>::cbegin() const noexcept
{
   return ConstIterator( v_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the dynamic vector.
//
// \return Iterator just past the last element of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename DynamicVector<Type,TF>::Iterator DynamicVector<Type,TF>::end() noexcept
{
   return Iterator( v_ + size_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the dynamic vector.
//
// \return Iterator just past the last element of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename DynamicVector<Type,TF>::ConstIterator DynamicVector<Type,TF>::end() const noexcept
{
   return ConstIterator( v_ + size_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the dynamic vector.
//
// \return Iterator just past the last element of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline typename DynamicVector<Type,TF>::ConstIterator DynamicVector<Type,TF>::cend() const noexcept
{
   return ConstIterator( v_ + size_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Homogenous assignment to all vector elements.
//
// \param rhs Scalar value to be assigned to all vector elements.
// \return Reference to the assigned vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline DynamicVector<Type,TF>& DynamicVector<Type,TF>::operator=( const Type& rhs )
{
   for( size_t i=0UL; i<size_; ++i )
      v_[i] = rhs;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List assignment to all vector elements.
//
// \param list The initializer list.
//
// This assignment operator offers the option to directly assign to all elements of the vector
// by means of an initializer list:

   \code
   blaze::DynamicVector<double> v;
   v = { 4.2, 6.3, -1.2 };
   \endcode

// The vector is resized according to the size of the initializer list and all its elements are
// assigned the values from the given initializer list.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline DynamicVector<Type,TF>& DynamicVector<Type,TF>::operator=( initializer_list<Type> list )
{
   resize( list.size(), false );
   std::copy( list.begin(), list.end(), v_ );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array assignment to all vector elements.
//
// \param array N-dimensional array for the assignment.
// \return Reference to the assigned vector.
//
// This assignment operator offers the option to directly set all elements of the vector:

   \code
   const int init[4] = { 1, 2, 3 };
   blaze::DynamicVector<int> v;
   v = init;
   \endcode

// The vector is resized according to the size of the array and assigned the values from the given
// array. Missing values are initialized with default values (as e.g. the fourth element in the
// example).
*/
template< typename Type   // Data type of the vector
        , bool TF >       // Transpose flag
template< typename Other  // Data type of the initialization array
        , size_t N >      // Dimension of the initialization array
inline DynamicVector<Type,TF>& DynamicVector<Type,TF>::operator=( const Other (&array)[N] )
{
   resize( N, false );

   for( size_t i=0UL; i<N; ++i )
      v_[i] = array[i];

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Copy assignment operator for DynamicVector.
//
// \param rhs Vector to be copied.
// \return Reference to the assigned vector.
//
// The vector is resized according to the given N-dimensional vector and initialized as a
// copy of this vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline DynamicVector<Type,TF>& DynamicVector<Type,TF>::operator=( const DynamicVector& rhs )
{
   if( &rhs == this ) return *this;

   resize( rhs.size_, false );
   smpAssign( *this, ~rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Move assignment operator for DynamicVector.
//
// \param rhs The vector to be moved into this instance.
// \return Reference to the assigned vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline DynamicVector<Type,TF>& DynamicVector<Type,TF>::operator=( DynamicVector&& rhs ) noexcept
{
   deallocate( v_ );

   size_     = rhs.size_;
   capacity_ = rhs.capacity_;
   v_        = rhs.v_;

   rhs.size_     = 0UL;
   rhs.capacity_ = 0UL;
   rhs.v_        = nullptr;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different vectors.
//
// \param rhs Vector to be copied.
// \return Reference to the assigned vector.
//
// The vector is resized according to the given vector and initialized as a copy of this vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline DynamicVector<Type,TF>& DynamicVector<Type,TF>::operator=( const Vector<VT,TF>& rhs )
{
   if( (~rhs).canAlias( this ) ) {
      DynamicVector tmp( ~rhs );
      swap( tmp );
   }
   else {
      resize( (~rhs).size(), false );
      if( IsSparseVector<VT>::value )
         reset();
      smpAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment operator for the addition of a vector (\f$ \vec{a}+=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be added to the vector.
// \return Reference to the vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline DynamicVector<Type,TF>& DynamicVector<Type,TF>::operator+=( const Vector<VT,TF>& rhs )
{
   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_<VT> tmp( ~rhs );
      smpAddAssign( *this, tmp );
   }
   else {
      smpAddAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment operator for the subtraction of a vector
//        (\f$ \vec{a}-=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be subtracted from the vector.
// \return Reference to the vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline DynamicVector<Type,TF>& DynamicVector<Type,TF>::operator-=( const Vector<VT,TF>& rhs )
{
   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_<VT> tmp( ~rhs );
      smpSubAssign( *this, tmp );
   }
   else {
      smpSubAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication of a vector
//        (\f$ \vec{a}*=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be multiplied with the vector.
// \return Reference to the vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline DynamicVector<Type,TF>& DynamicVector<Type,TF>::operator*=( const Vector<VT,TF>& rhs )
{
   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( IsSparseVector<VT>::value || (~rhs).canAlias( this ) ) {
      DynamicVector<Type,TF> tmp( *this * (~rhs) );
      swap( tmp );
   }
   else {
      smpMultAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment operator for the division of a dense vector (\f$ \vec{a}/=\vec{b} \f$).
//
// \param rhs The right-hand side dense vector divisor.
// \return Reference to the vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline DynamicVector<Type,TF>& DynamicVector<Type,TF>::operator/=( const DenseVector<VT,TF>& rhs )
{
   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      DynamicVector<Type,TF> tmp( *this / (~rhs) );
      swap( tmp );
   }
   else {
      smpDivAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication between a vector and
//        a scalar value (\f$ \vec{a}*=s \f$).
//
// \param rhs The right-hand side scalar value for the multiplication.
// \return Reference to the vector.
*/
template< typename Type     // Data type of the vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, DynamicVector<Type,TF> >&
   DynamicVector<Type,TF>::operator*=( Other rhs )
{
   smpAssign( *this, (*this) * rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment operator for the division of a vector by a scalar value
//        (\f$ \vec{a}/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the vector.
//
// \note A division by zero is only checked by an user assert.
*/
template< typename Type     // Data type of the vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, DynamicVector<Type,TF> >&
   DynamicVector<Type,TF>::operator/=( Other rhs )
{
   BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

   smpAssign( *this, (*this) / rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the current size/dimension of the vector.
//
// \return The size of the vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline size_t DynamicVector<Type,TF>::size() const noexcept
{
   return size_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the vector.
//
// \return The capacity of the vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline size_t DynamicVector<Type,TF>::capacity() const noexcept
{
   return capacity_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the vector.
//
// \return The number of non-zero elements in the vector.
//
// Note that the number of non-zero elements is always less than or equal to the current size
// of the vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline size_t DynamicVector<Type,TF>::nonZeros() const
{
   size_t nonzeros( 0 );

   for( size_t i=0UL; i<size_; ++i ) {
      if( !isDefault( v_[i] ) )
         ++nonzeros;
   }

   return nonzeros;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void DynamicVector<Type,TF>::reset()
{
   using blaze::clear;
   for( size_t i=0UL; i<size_; ++i )
      clear( v_[i] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the vector.
//
// \return void
//
// After the clear() function, the size of the vector is 0.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void DynamicVector<Type,TF>::clear()
{
   resize( 0UL, false );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the vector.
//
// \param n The new size of the vector.
// \param preserve \a true if the old values of the vector should be preserved, \a false if not.
// \return void
//
// This function resizes the vector using the given size to \a n. During this operation, new
// dynamic memory may be allocated in case the capacity of the vector is too small. Note that
// this function may invalidate all existing views (subvectors, ...) on the vector if it is
// used to shrink the vector. Additionally, the resize operation potentially changes all vector
// elements. In order to preserve the old vector values, the \a preserve flag can be set to
// \a true. However, new vector elements are not initialized!
//
// The following example illustrates the resize operation of a vector of size 2 to a vector of
// size 4. The new, uninitialized elements are marked with \a x:

                              \f[
                              \left(\begin{array}{*{2}{c}}
                              1 & 2 \\
                              \end{array}\right)

                              \Longrightarrow

                              \left(\begin{array}{*{4}{c}}
                              1 & 2 & x & x \\
                              \end{array}\right)
                              \f]
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void DynamicVector<Type,TF>::resize( size_t n, bool preserve )
{
   if( n > capacity_ )
   {
      // Allocating a new array
      const size_t newCapacity( adjustCapacity( n ) );
      Type* BLAZE_RESTRICT tmp = allocate<Type>( newCapacity );

      // Initializing the new array
      if( preserve ) {
         transfer( v_, v_+size_, tmp );
      }

      if( IsVectorizable<Type>::value ) {
         for( size_t i=size_; i<newCapacity; ++i )
            tmp[i] = Type();
      }

      // Replacing the old array
      std::swap( v_, tmp );
      deallocate( tmp );
      capacity_ = newCapacity;
   }
   else if( IsVectorizable<Type>::value && n < size_ )
   {
      for( size_t i=n; i<size_; ++i )
         v_[i] = Type();
   }

   size_ = n;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Extending the size of the vector.
//
// \param n Number of additional vector elements.
// \param preserve \a true if the old values of the vector should be preserved, \a false if not.
// \return void
//
// This function increases the vector size by \a n elements. During this operation, new dynamic
// memory may be allocated in case the capacity of the vector is too small. Therefore this
// function potentially changes all vector elements. In order to preserve the old vector values,
// the \a preserve flag can be set to \a true. However, new vector elements are not initialized!
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void DynamicVector<Type,TF>::extend( size_t n, bool preserve )
{
   resize( size_+n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the minimum capacity of the vector.
//
// \param n The new minimum capacity of the vector.
// \return void
//
// This function increases the capacity of the vector to at least \a n elements. The current
// values of the vector elements are preserved.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void DynamicVector<Type,TF>::reserve( size_t n )
{
   if( n > capacity_ )
   {
      // Allocating a new array
      const size_t newCapacity( adjustCapacity( n ) );
      Type* BLAZE_RESTRICT tmp = allocate<Type>( newCapacity );

      // Initializing the new array
      transfer( v_, v_+size_, tmp );

      if( IsVectorizable<Type>::value ) {
         for( size_t i=size_; i<newCapacity; ++i )
            tmp[i] = Type();
      }

      // Replacing the old array
      std::swap( tmp, v_ );
      deallocate( tmp );
      capacity_ = newCapacity;
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the vector by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the vector scaling.
// \return Reference to the vector.
*/
template< typename Type     // Data type of the vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the scalar value
inline DynamicVector<Type,TF>& DynamicVector<Type,TF>::scale( const Other& scalar )
{
   for( size_t i=0UL; i<size_; ++i )
      v_[i] *= scalar;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two vectors.
//
// \param v The vector to be swapped.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void DynamicVector<Type,TF>::swap( DynamicVector& v ) noexcept
{
   std::swap( size_, v.size_ );
   std::swap( capacity_, v.capacity_ );
   std::swap( v_, v.v_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Adjusting the new capacity of the vector according to its data type \a Type.
//
// \param minCapacity The minimum necessary capacity.
// \return The new capacity.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline size_t DynamicVector<Type,TF>::adjustCapacity( size_t minCapacity ) const noexcept
{
   if( usePadding && IsVectorizable<Type>::value )
      return nextMultiple<size_t>( minCapacity, SIMDSIZE );
   else return minCapacity;
}
//*************************************************************************************************




//=================================================================================================
//
//  DEBUGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the invariants of the dynamic vector are intact.
//
// \return \a true in case the dynamic vector's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the dynamic vector are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline bool DynamicVector<Type,TF>::isIntact() const noexcept
{
   if( size_ > capacity_ )
      return false;

   if( IsNumeric<Type>::value ) {
      for( size_t i=size_; i<capacity_; ++i ) {
         if( v_[i] != Type() )
            return false;
      }
   }

   return true;
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
inline bool DynamicVector<Type,TF>::canAlias( const Other* alias ) const noexcept
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
inline bool DynamicVector<Type,TF>::isAliased( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the vector is properly aligned in memory.
//
// \return \a true in case the vector is aligned, \a false if not.
//
// This function returns whether the vector is guaranteed to be properly aligned in memory, i.e.
// whether the beginning and the end of the vector are guaranteed to conform to the alignment
// restrictions of the element type \a Type.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline bool DynamicVector<Type,TF>::isAligned() const noexcept
{
   return true;
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
inline bool DynamicVector<Type,TF>::canSMPAssign() const noexcept
{
   return ( size() > SMP_DVECASSIGN_THRESHOLD );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Load of a SIMD element of the vector.
//
// \param index Access index. The index must be smaller than the number of vector elements.
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense vector. The index
// must be smaller than the number of vector elements and it must be a multiple of the number
// of values inside the SIMD element. This function must \b NOT be called explicitly! It is
// used internally for the performance optimized evaluation of expression templates. Calling
// this function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE typename DynamicVector<Type,TF>::SIMDType
   DynamicVector<Type,TF>::load( size_t index ) const noexcept
{
   return loada( index );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned load of a SIMD element of the vector.
//
// \param index Access index. The index must be smaller than the number of vector elements.
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense vector. The
// index must be smaller than the number of vector elements and it must be a multiple of the
// number of values inside the SIMD element. This function must \b NOT be called explicitly!
// It is used internally for the performance optimized evaluation of expression templates.
// Calling this function explicitly might result in erroneous results and/or in compilation
// errors.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE typename DynamicVector<Type,TF>::SIMDType
   DynamicVector<Type,TF>::loada( size_t index ) const noexcept
{
   using blaze::loada;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= capacity_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_+index ), "Invalid alignment detected" );

   return loada( v_+index );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Unaligned load of a SIMD element of the vector.
//
// \param index Access index. The index must be smaller than the number of vector elements.
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense vector. The
// index must be smaller than the number of vector elements and it must be a multiple of the
// number of values inside the SIMD element. This function must \b NOT be called explicitly!
// It is used internally for the performance optimized evaluation of expression templates.
// Calling this function explicitly might result in erroneous results and/or in compilation
// errors.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE typename DynamicVector<Type,TF>::SIMDType
   DynamicVector<Type,TF>::loadu( size_t index ) const noexcept
{
   using blaze::loadu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= capacity_, "Invalid vector access index" );

   return loadu( v_+index );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Store of a SIMD element of the vector.
//
// \param index Access index. The index must be smaller than the number of vector elements.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store of a specific SIMD element of the dense vector. The index
// must be smaller than the number of vector elements and it must be a multiple of the number
// of values inside the SIMD element. This function must \b NOT be called explicitly! It is
// used internally for the performance optimized evaluation of expression templates. Calling
// this function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void
   DynamicVector<Type,TF>::store( size_t index, const SIMDType& value ) noexcept
{
   storea( index, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned store of a SIMD element of the vector.
//
// \param index Access index. The index must be smaller than the number of vector elements.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store of a specific SIMD element of the dense vector. The
// index must be smaller than the number of vector elements and it must be a multiple of the
// number of values inside the SIMD element. This function must \b NOT be called explicitly! It
// is used internally for the performance optimized evaluation of expression templates. Calling
// this function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void
   DynamicVector<Type,TF>::storea( size_t index, const SIMDType& value ) noexcept
{
   using blaze::storea;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= capacity_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_+index ), "Invalid alignment detected" );

   storea( v_+index, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Unaligned store of a SIMD element of the vector.
//
// \param index Access index. The index must be smaller than the number of vector elements.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store of a specific SIMD element of the dense vector.
// The index must be smaller than the number of vector elements and it must be a multiple of the
// number of values inside the SIMD element. This function must \b NOT be called explicitly! It
// is used internally for the performance optimized evaluation of expression templates. Calling
// this function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void
   DynamicVector<Type,TF>::storeu( size_t index, const SIMDType& value ) noexcept
{
   using blaze::storeu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= capacity_, "Invalid vector access index" );

   storeu( v_+index, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned, non-temporal store of a SIMD element of the vector.
//
// \param index Access index. The index must be smaller than the number of vector elements.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store of a specific SIMD element of the
// dense vector. The index must be smaller than the number of vector elements and it must be
// a multiple of the number of values inside the SIMD element. This function must \b NOT be
// called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void
   DynamicVector<Type,TF>::stream( size_t index, const SIMDType& value ) noexcept
{
   using blaze::stream;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= capacity_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_+index ), "Invalid alignment detected" );

   stream( v_+index, value );
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
inline DisableIf_<typename DynamicVector<Type,TF>::BLAZE_TEMPLATE VectorizedAssign<VT> >
   DynamicVector<Type,TF>::assign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t i=0UL; i<ipos; i+=2UL ) {
      v_[i    ] = (~rhs)[i    ];
      v_[i+1UL] = (~rhs)[i+1UL];
   }
   if( ipos < (~rhs).size() )
      v_[ipos] = (~rhs)[ipos];
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename DynamicVector<Type,TF>::BLAZE_TEMPLATE VectorizedAssign<VT> >
   DynamicVector<Type,TF>::assign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const bool remainder( !usePadding || !IsPadded<VT>::value );

   const size_t ipos( ( remainder )?( size_ & size_t(-SIMDSIZE) ):( size_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i=0UL;
   Iterator left( begin() );
   ConstIterator_<VT> right( (~rhs).begin() );

   if( useStreaming && size_ > ( cacheSize/( sizeof(Type) * 3UL ) ) && !(~rhs).isAliased( this ) )
   {
      for( ; i<ipos; i+=SIMDSIZE ) {
         left.stream( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; remainder && i<size_; ++i ) {
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
      for( ; remainder && i<size_; ++i ) {
         *left = *right; ++left; ++right;
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
inline void DynamicVector<Type,TF>::assign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   for( ConstIterator_<VT> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      v_[element->index()] = element->value();
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
inline DisableIf_<typename DynamicVector<Type,TF>::BLAZE_TEMPLATE VectorizedAddAssign<VT> >
   DynamicVector<Type,TF>::addAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t i=0UL; i<ipos; i+=2UL ) {
      v_[i    ] += (~rhs)[i    ];
      v_[i+1UL] += (~rhs)[i+1UL];
   }
   if( ipos < (~rhs).size() )
      v_[ipos] += (~rhs)[ipos];
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename DynamicVector<Type,TF>::BLAZE_TEMPLATE VectorizedAddAssign<VT> >
   DynamicVector<Type,TF>::addAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const bool remainder( !usePadding || !IsPadded<VT>::value );

   const size_t ipos( ( remainder )?( size_ & size_t(-SIMDSIZE) ):( size_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT> right( (~rhs).begin() );

   for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE ) {
      left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; remainder && i<size_; ++i ) {
      *left += *right; ++left; ++right;
   }
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
inline void DynamicVector<Type,TF>::addAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   for( ConstIterator_<VT> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      v_[element->index()] += element->value();
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
inline DisableIf_<typename DynamicVector<Type,TF>::BLAZE_TEMPLATE VectorizedSubAssign<VT> >
   DynamicVector<Type,TF>::subAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t i=0UL; i<ipos; i+=2UL ) {
      v_[i    ] -= (~rhs)[i    ];
      v_[i+1UL] -= (~rhs)[i+1UL];
   }
   if( ipos < (~rhs).size() )
      v_[ipos] -= (~rhs)[ipos];
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename DynamicVector<Type,TF>::BLAZE_TEMPLATE VectorizedSubAssign<VT> >
   DynamicVector<Type,TF>::subAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const bool remainder( !usePadding || !IsPadded<VT>::value );

   const size_t ipos( ( remainder )?( size_ & size_t(-SIMDSIZE) ):( size_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT> right( (~rhs).begin() );

   for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE ) {
      left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; remainder && i<size_; ++i ) {
      *left -= *right; ++left; ++right;
   }
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
inline void DynamicVector<Type,TF>::subAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   for( ConstIterator_<VT> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      v_[element->index()] -= element->value();
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
inline DisableIf_<typename DynamicVector<Type,TF>::BLAZE_TEMPLATE VectorizedMultAssign<VT> >
   DynamicVector<Type,TF>::multAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t i=0UL; i<ipos; i+=2UL ) {
      v_[i    ] *= (~rhs)[i    ];
      v_[i+1UL] *= (~rhs)[i+1UL];
   }
   if( ipos < (~rhs).size() )
      v_[ipos] *= (~rhs)[ipos];
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename DynamicVector<Type,TF>::BLAZE_TEMPLATE VectorizedMultAssign<VT> >
   DynamicVector<Type,TF>::multAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const bool remainder( !usePadding || !IsPadded<VT>::value );

   const size_t ipos( ( remainder )?( size_ & size_t(-SIMDSIZE) ):( size_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT> right( (~rhs).begin() );

   for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE ) {
      left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
   }
   for( ; remainder && i<size_; ++i ) {
      *left *= *right; ++left; ++right;
   }
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void DynamicVector<Type,TF>::multAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const DynamicVector tmp( serial( *this ) );

   reset();

   for( ConstIterator_<VT> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      v_[element->index()] = tmp[element->index()] * element->value();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the division assignment of a dense vector.
//
// \param rhs The right-hand side dense vector divisior.
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
inline DisableIf_<typename DynamicVector<Type,TF>::BLAZE_TEMPLATE VectorizedDivAssign<VT> >
   DynamicVector<Type,TF>::divAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t i=0UL; i<ipos; i+=2UL ) {
      v_[i    ] /= (~rhs)[i    ];
      v_[i+1UL] /= (~rhs)[i+1UL];
   }
   if( ipos < (~rhs).size() )
      v_[ipos] /= (~rhs)[ipos];
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename DynamicVector<Type,TF>::BLAZE_TEMPLATE VectorizedDivAssign<VT> >
   DynamicVector<Type,TF>::divAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );
   Iterator left( begin() );
   ConstIterator_<VT> right( (~rhs).begin() );

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
//*************************************************************************************************




//=================================================================================================
//
//  DYNAMICVECTOR OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DynamicVector operators */
//@{
template< typename Type, bool TF >
inline void reset( DynamicVector<Type,TF>& v );

template< typename Type, bool TF >
inline void clear( DynamicVector<Type,TF>& v );

template< typename Type, bool TF >
inline bool isDefault( const DynamicVector<Type,TF>& v );

template< typename Type, bool TF >
inline bool isIntact( const DynamicVector<Type,TF>& v ) noexcept;

template< typename Type, bool TF >
inline void swap( DynamicVector<Type,TF>& a, DynamicVector<Type,TF>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given dynamic vector.
// \ingroup dynamic_vector
//
// \param v The dynamic vector to be resetted.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void reset( DynamicVector<Type,TF>& v )
{
   v.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given dynamic vector.
// \ingroup dynamic_vector
//
// \param v The dynamic vector to be cleared.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void clear( DynamicVector<Type,TF>& v )
{
   v.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given dynamic vector is in default state.
// \ingroup dynamic_vector
//
// \param v The dynamic vector to be tested for its default state.
// \return \a true in case the given vector's size is zero, \a false otherwise.
//
// This function checks whether the dynamic vector is in default (constructed) state, i.e. if
// it's size is 0. In case it is in default state, the function returns \a true, else it will
// return \a false. The following example demonstrates the use of the \a isDefault() function:

   \code
   blaze::DynamicVector<int> a;
   // ... Resizing and initialization
   if( isDefault( a ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline bool isDefault( const DynamicVector<Type,TF>& v )
{
   return ( v.size() == 0UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given dynamic vector are intact.
// \ingroup dynamic_vector
//
// \param v The dynamic vector to be tested.
// \return \a true in case the given vector's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the dynamic vector are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   blaze::DynamicVector<int> a;
   // ... Resizing and initialization
   if( isIntact( a ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline bool isIntact( const DynamicVector<Type,TF>& v ) noexcept
{
   return v.isIntact();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two vectors.
// \ingroup dynamic_vector
//
// \param a The first vector to be swapped.
// \param b The second vector to be swapped.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void swap( DynamicVector<Type,TF>& a, DynamicVector<Type,TF>& b ) noexcept
{
   a.swap( b );
}
//*************************************************************************************************




//=================================================================================================
//
//  HASCONSTDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool TF >
struct HasConstDataAccess< DynamicVector<T,TF> > : public TrueType
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
template< typename T, bool TF >
struct HasMutableDataAccess< DynamicVector<T,TF> > : public TrueType
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
template< typename T, bool TF >
struct IsAligned< DynamicVector<T,TF> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISPADDED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool TF >
struct IsPadded< DynamicVector<T,TF> > : public BoolConstant<usePadding>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISRESIZABLE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool TF >
struct IsResizable< DynamicVector<T,TF> > : public TrueType
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
struct AddTrait< DynamicVector<T1,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< AddTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct AddTrait< StaticVector<T1,N,TF>, DynamicVector<T2,TF> >
{
   using Type = StaticVector< AddTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool TF, typename T2, size_t N >
struct AddTrait< DynamicVector<T1,TF>, HybridVector<T2,N,TF> >
{
   using Type = HybridVector< AddTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct AddTrait< HybridVector<T1,N,TF>, DynamicVector<T2,TF> >
{
   using Type = HybridVector< AddTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool TF, typename T2 >
struct AddTrait< DynamicVector<T1,TF>, DynamicVector<T2,TF> >
{
   using Type = DynamicVector< AddTrait_<T1,T2>, TF >;
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
struct SubTrait< DynamicVector<T1,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< SubTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct SubTrait< StaticVector<T1,N,TF>, DynamicVector<T2,TF> >
{
   using Type = StaticVector< SubTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool TF, typename T2, size_t N >
struct SubTrait< DynamicVector<T1,TF>, HybridVector<T2,N,TF> >
{
   using Type = HybridVector< SubTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct SubTrait< HybridVector<T1,N,TF>, DynamicVector<T2,TF> >
{
   using Type = HybridVector< SubTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool TF, typename T2 >
struct SubTrait< DynamicVector<T1,TF>, DynamicVector<T2,TF> >
{
   using Type = DynamicVector< SubTrait_<T1,T2>, TF >;
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
struct MultTrait< DynamicVector<T1,TF>, T2, EnableIf_<IsNumeric<T2> > >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, typename T2, bool TF >
struct MultTrait< T1, DynamicVector<T2,TF>, EnableIf_<IsNumeric<T1> > >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2, size_t N >
struct MultTrait< DynamicVector<T1,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< MultTrait_<T1,T2>, N, TF >;
};

template< typename T1, typename T2, size_t N >
struct MultTrait< DynamicVector<T1,false>, StaticVector<T2,N,true> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, typename T2, size_t N >
struct MultTrait< DynamicVector<T1,true>, StaticVector<T2,N,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct MultTrait< StaticVector<T1,N,TF>, DynamicVector<T2,TF> >
{
   using Type = StaticVector< MultTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, typename T2 >
struct MultTrait< StaticVector<T1,N,false>, DynamicVector<T2,true> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, size_t N, typename T2 >
struct MultTrait< StaticVector<T1,N,true>, DynamicVector<T2,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, bool TF, typename T2, size_t N >
struct MultTrait< DynamicVector<T1,TF>, HybridVector<T2,N,TF> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, N, TF >;
};

template< typename T1, typename T2, size_t N >
struct MultTrait< DynamicVector<T1,false>, HybridVector<T2,N,true> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, typename T2, size_t N >
struct MultTrait< DynamicVector<T1,true>, HybridVector<T2,N,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct MultTrait< HybridVector<T1,N,TF>, DynamicVector<T2,TF> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, typename T2 >
struct MultTrait< HybridVector<T1,N,false>, DynamicVector<T2,true> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, size_t N, typename T2 >
struct MultTrait< HybridVector<T1,N,true>, DynamicVector<T2,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, bool TF, typename T2 >
struct MultTrait< DynamicVector<T1,TF>, DynamicVector<T2,TF> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, typename T2 >
struct MultTrait< DynamicVector<T1,false>, DynamicVector<T2,true> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, typename T2 >
struct MultTrait< DynamicVector<T1,true>, DynamicVector<T2,false> >
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
struct CrossTrait< DynamicVector<T1,TF>, StaticVector<T2,3UL,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, bool TF, typename T2 >
struct CrossTrait< StaticVector<T1,3UL,TF>, DynamicVector<T2,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, bool TF, typename T2, size_t N >
struct CrossTrait< DynamicVector<T1,TF>, HybridVector<T2,N,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct CrossTrait< HybridVector<T1,N,TF>, DynamicVector<T2,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, bool TF, typename T2 >
struct CrossTrait< DynamicVector<T1,TF>, DynamicVector<T2,TF> >
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
struct DivTrait< DynamicVector<T1,TF>, T2, EnableIf_<IsNumeric<T2> > >
{
   using Type = DynamicVector< DivTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2, size_t N >
struct DivTrait< DynamicVector<T1,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< DivTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct DivTrait< StaticVector<T1,N,TF>, DynamicVector<T2,TF> >
{
   using Type = StaticVector< DivTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool TF, typename T2, size_t N >
struct DivTrait< DynamicVector<T1,TF>, HybridVector<T2,N,TF> >
{
   using Type = HybridVector< DivTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct DivTrait< HybridVector<T1,N,TF>, DynamicVector<T2,TF> >
{
   using Type = HybridVector< DivTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool TF, typename T2 >
struct DivTrait< DynamicVector<T1,TF>, DynamicVector<T2,TF> >
{
   using Type = DynamicVector< DivTrait_<T1,T2>, TF >;
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
struct MathTrait< DynamicVector<T1,TF>, DynamicVector<T2,TF> >
{
   using HighType = DynamicVector< typename MathTrait<T1,T2>::HighType, TF >;
   using LowType  = DynamicVector< typename MathTrait<T1,T2>::LowType , TF >;
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
struct SubvectorTrait< DynamicVector<T1,TF> >
{
   using Type = DynamicVector<T1,TF>;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
