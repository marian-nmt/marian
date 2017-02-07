//=================================================================================================
/*!
//  \file blaze/math/dense/StaticVector.h
//  \brief Header file for the implementation of a fixed-size vector
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

#ifndef _BLAZE_MATH_DENSE_STATICVECTOR_H_
#define _BLAZE_MATH_DENSE_STATICVECTOR_H_


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
#include <blaze/math/typetraits/IsSparseVector.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/TransposeFlag.h>
#include <blaze/util/AlignedArray.h>
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
#include <blaze/util/mpl/NextMultiple.h>
#include <blaze/util/mpl/SizeT.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Template.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsVectorizable.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup static_vector StaticVector
// \ingroup dense_vector
*/
/*!\brief Efficient implementation of a fixed-sized vector.
// \ingroup static_vector
//
// The StaticVector class template is the representation of a fixed-size vector with statically
// allocated elements of arbitrary type. The type of the elements, the number of elements and
// the transpose flag of the vector can be specified via the three template parameters:

   \code
   template< typename Type, size_t N, bool TF >
   class StaticVector;
   \endcode

//  - Type: specifies the type of the vector elements. StaticVector can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
//  - N   : specifies the total number of vector elements. It is expected that StaticVector is
//          only used for tiny and small vectors.
//  - TF  : specifies whether the vector is a row vector (\a blaze::rowVector) or a column
//          vector (\a blaze::columnVector). The default value is \a blaze::columnVector.
//
// These contiguously stored elements can be directly accessed with the subscript operator. The
// numbering of the vector elements is

                             \f[\left(\begin{array}{*{4}{c}}
                             0 & 1 & \cdots & N-1 \\
                             \end{array}\right)\f]

// The use of StaticVector is very natural and intuitive. All operations (addition, subtraction,
// multiplication, scaling, ...) can be performed on all possible combinations of dense and sparse
// vectors with fitting element types. The following example gives an impression of the use of a
// 2-dimensional StaticVector:

   \code
   using blaze::StaticVector;
   using blaze::CompressedVector;
   using blaze::StaticMatrix;

   StaticVector<double,2UL> a;  // Default initialized 2D vector
   a[0] = 1.0;                  // Initialization of the first element
   a[1] = 2.0;                  // Initialization of the second element

   StaticVector<double,2UL> b( 3.0, 2.0 );  // Directly initialized 2D vector
   CompressedVector<float>  c( 2 );         // Empty sparse single precision vector
   StaticVector<double,2UL> d;              // Default constructed static vector
   StaticMatrix<double,2UL,2UL> A;          // Default constructed static row-major matrix

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
        , size_t N                          // Number of elements
        , bool TF = defaultTransposeFlag >  // Transpose flag
class StaticVector : public DenseVector< StaticVector<Type,N,TF>, TF >
{
 public:
   //**Type definitions****************************************************************************
   typedef StaticVector<Type,N,TF>   This;           //!< Type of this StaticVector instance.
   typedef DenseVector<This,TF>      BaseType;       //!< Base type of this StaticVector instance.
   typedef This                      ResultType;     //!< Result type for expression template evaluations.
   typedef StaticVector<Type,N,!TF>  TransposeType;  //!< Transpose type for expression template evaluations.
   typedef Type                      ElementType;    //!< Type of the vector elements.
   typedef SIMDTrait_<ElementType>   SIMDType;       //!< SIMD type of the vector elements.
   typedef const Type&               ReturnType;     //!< Return type for expression template evaluations.
   typedef const StaticVector&       CompositeType;  //!< Data type for composite expression templates.

   typedef Type&        Reference;       //!< Reference to a non-constant vector value.
   typedef const Type&  ConstReference;  //!< Reference to a constant vector value.
   typedef Type*        Pointer;         //!< Pointer to a non-constant vector value.
   typedef const Type*  ConstPointer;    //!< Pointer to a constant vector value.

   typedef DenseIterator<Type,aligned>        Iterator;       //!< Iterator over non-constant elements.
   typedef DenseIterator<const Type,aligned>  ConstIterator;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a StaticVector with different data/element type.
   */
   template< typename ET >  // Data type of the other vector
   struct Rebind {
      typedef StaticVector<ET,N,TF>  Other;  //!< The type of the other StaticVector.
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SIMD optimization.
   /*! The \a simdEnabled compilation flag indicates whether expressions the vector is involved
       in can be optimized via SIMD operations. In case the element type of the vector is a
       vectorizable data type, the \a simdEnabled compilation flag is set to \a true, otherwise
       it is set to \a false. */
   enum : bool { simdEnabled = IsVectorizable<Type>::value };

   //! Compilation flag for SMP assignments.
   /*! The \a smpAssignable compilation flag indicates whether the vector can be used in SMP
       (shared memory parallel) assignments (both on the left-hand and right-hand side of the
       assignment). */
   enum : bool { smpAssignable = false };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline StaticVector();
   explicit inline StaticVector( const Type& init );
   explicit inline StaticVector( initializer_list<Type> list );

   template< typename Other >
   explicit inline StaticVector( size_t n, const Other* array );

   template< typename Other >
   explicit inline StaticVector( const Other (&array)[N] );

                              inline StaticVector( const StaticVector& v );
   template< typename Other > inline StaticVector( const StaticVector<Other,N,TF>& v );
   template< typename VT >    inline StaticVector( const Vector<VT,TF>& v );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
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
   inline StaticVector& operator=( const Type& rhs );
   inline StaticVector& operator=( initializer_list<Type> list );

   template< typename Other >
   inline StaticVector& operator=( const Other (&array)[N] );

                              inline StaticVector& operator=( const StaticVector& rhs );
   template< typename Other > inline StaticVector& operator=( const StaticVector<Other,N,TF>& rhs );

   template< typename VT > inline StaticVector& operator= ( const Vector<VT,TF>& rhs );
   template< typename VT > inline StaticVector& operator+=( const Vector<VT,TF>& rhs );
   template< typename VT > inline StaticVector& operator-=( const Vector<VT,TF>& rhs );
   template< typename VT > inline StaticVector& operator*=( const Vector<VT,TF>& rhs );
   template< typename VT > inline StaticVector& operator/=( const DenseVector<VT,TF>& rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, StaticVector >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, StaticVector >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline constexpr size_t size() const noexcept;
                              inline constexpr size_t capacity() const noexcept;
                              inline size_t           nonZeros() const;
                              inline void             reset();
   template< typename Other > inline StaticVector&    scale( const Other& scalar );
                              inline void             swap( StaticVector& v ) noexcept;
   //@}
   //**********************************************************************************************

   //**Memory functions****************************************************************************
   /*!\name Memory functions */
   //@{
   static inline void* operator new  ( std::size_t size );
   static inline void* operator new[]( std::size_t size );
   static inline void* operator new  ( std::size_t size, const std::nothrow_t& );
   static inline void* operator new[]( std::size_t size, const std::nothrow_t& );

   static inline void operator delete  ( void* ptr );
   static inline void operator delete[]( void* ptr );
   static inline void operator delete  ( void* ptr, const std::nothrow_t& );
   static inline void operator delete[]( void* ptr, const std::nothrow_t& );
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
   //! The number of elements packed within a single SIMD vector.
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

   inline bool isAligned() const noexcept;

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t index ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t index ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t index ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t index, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t index, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t index, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t index, const SIMDType& value ) noexcept;

   template< typename VT >
   inline DisableIf_< VectorizedAssign<VT> > assign( const DenseVector<VT,TF>& rhs );

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
   //**********************************************************************************************
   //! Alignment adjustment.
   enum : size_t { NN = ( usePadding )?( NextMultiple< SizeT<N>, SizeT<SIMDSIZE> >::value ):( N ) };
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   AlignedArray<Type,NN> v_;  //!< The statically allocated vector elements.
                              /*!< Access to the vector values is gained via the subscript operator.
                                   The order of the elements is
                                   \f[\left(\begin{array}{*{4}{c}}
                                   0 & 1 & \cdots & N-1 \\
                                   \end{array}\right)\f] */
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE  ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST         ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE      ( Type );
   BLAZE_STATIC_ASSERT( !usePadding || ( NN % SIMDSIZE == 0UL ) );
   BLAZE_STATIC_ASSERT( NN >= N );
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
/*!\brief The default constructor for StaticVector.
//
// All vector elements are initialized to the default value (i.e. 0 for integral data types).
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline StaticVector<Type,N,TF>::StaticVector()
   : v_()  // The statically allocated vector elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   if( IsNumeric<Type>::value ) {
      for( size_t i=0UL; i<NN; ++i )
         v_[i] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a homogenous initialization of all elements.
//
// \param init Initial value for all vector elements.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline StaticVector<Type,N,TF>::StaticVector( const Type& init )
   : v_()  // The statically allocated vector elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   for( size_t i=0UL; i<N; ++i )
      v_[i] = init;

   for( size_t i=N; i<NN; ++i )
      v_[i] = Type();

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List initialization of all vector elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid setup of static vector.
//
// This constructor provides the option to explicitly initialize the elements of the vector by
// means of an initializer list:

   \code
   blaze::StaticVector<double,3UL> v1{ 4.2, 6.3, -1.2 };
   \endcode

// The vector elements are initialized by the values of the given initializer list. Missing values
// are initialized as default. Note that in case the size of the initializer list exceeds the size
// of the vector, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline StaticVector<Type,N,TF>::StaticVector( initializer_list<Type> list )
   : v_()  // The statically allocated vector elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   if( list.size() > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of static vector" );
   }

   std::fill( std::copy( list.begin(), list.end(), v_.data() ), v_.data()+NN, Type() );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array initialization of all vector elements.
//
// \param n The size of the vector.
// \param array Dynamic array for the initialization.
//
// This constructor offers the option to directly initialize the elements of the vector with a
// dynamic array:

   \code
   const double array* = new double[2];
   // ... Initialization of the array
   blaze::StaticVector<double,2> v( array, 2UL );
   delete[] array;
   \endcode

// The vector is initialized with the values from the given array. Missing values are initialized
// with default values. In case the size of the given vector exceeds the maximum size of the
// static vector (i.e. is larger than N), a \a std::invalid_argument exception is thrown.\n
// Note that it is expected that the given \a array has at least \a n elements. Providing an
// array with less elements results in undefined behavior!
*/
template< typename Type     // Data type of the vector
        , size_t N          // Number of elements
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the initialization array
inline StaticVector<Type,N,TF>::StaticVector( size_t n, const Other* array )
   : v_()  // The statically allocated vector elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   if( n > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of static vector" );
   }

   for( size_t i=0UL; i<n; ++i )
      v_[i] = array[i];

   if( IsNumeric<Type>::value ) {
      for( size_t i=n; i<NN; ++i )
         v_[i] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array initialization of all vector elements.
//
// \param array M-dimensional array for the initialization.
//
// This assignment operator offers the option to directly initialize the elements of the vector
// with a static array:

   \code
   const double init[3] = { 1.0, 2.0 };
   blaze::StaticVector<double,3> v( init );
   \endcode

// The vector is initialized with the values from the given array. Missing values are initialized
// with default values (as e.g. the third value in the example).
*/
template< typename Type     // Data type of the vector
        , size_t N          // Number of elements
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the initialization array
inline StaticVector<Type,N,TF>::StaticVector( const Other (&array)[N] )
   : v_()  // The statically allocated vector elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   for( size_t i=0UL; i<N; ++i )
      v_[i] = array[i];

   for( size_t i=N; i<NN; ++i )
      v_[i] = Type();

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for StaticVector.
//
// \param v Vector to be copied.
//
// The copy constructor is explicitly defined in order to enable/facilitate NRV optimization.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline StaticVector<Type,N,TF>::StaticVector( const StaticVector& v )
   : v_()  // The statically allocated vector elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   for( size_t i=0UL; i<NN; ++i )
      v_[i] = v.v_[i];

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different StaticVector instances.
//
// \param v Vector to be copied.
*/
template< typename Type     // Data type of the vector
        , size_t N          // Number of elements
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign vector
inline StaticVector<Type,N,TF>::StaticVector( const StaticVector<Other,N,TF>& v )
   : v_()  // The statically allocated vector elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   for( size_t i=0UL; i<N; ++i )
      v_[i] = v[i];

   for( size_t i=N; i<NN; ++i )
      v_[i] = Type();

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different vectors.
//
// \param v Vector to be copied.
// \exception std::invalid_argument Invalid setup of static vector.
//
// This constructor initializes the static vector from the given vector. In case the size
// of the given vector does not match the size of the static vector (i.e. is not N), a
// \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the foreign vector
inline StaticVector<Type,N,TF>::StaticVector( const Vector<VT,TF>& v )
   : v_()  // The statically allocated vector elements
{
   using blaze::assign;

   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   if( (~v).size() != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of static vector" );
   }

   for( size_t i=( IsSparseVector<VT>::value ? 0UL : N ); i<NN; ++i ) {
      v_[i] = Type();
   }

   assign( *this, ~v );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline typename StaticVector<Type,N,TF>::Reference
   StaticVector<Type,N,TF>::operator[]( size_t index ) noexcept
{
   BLAZE_USER_ASSERT( index < N, "Invalid vector access index" );
   return v_[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subscript operator for the direct access to the vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference-to-const to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline typename StaticVector<Type,N,TF>::ConstReference
   StaticVector<Type,N,TF>::operator[]( size_t index ) const noexcept
{
   BLAZE_USER_ASSERT( index < N, "Invalid vector access index" );
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline typename StaticVector<Type,N,TF>::Reference
   StaticVector<Type,N,TF>::at( size_t index )
{
   if( index >= N ) {
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline typename StaticVector<Type,N,TF>::ConstReference
   StaticVector<Type,N,TF>::at( size_t index ) const
{
   if( index >= N ) {
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
// This function returns a pointer to the internal storage of the static vector.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline typename StaticVector<Type,N,TF>::Pointer StaticVector<Type,N,TF>::data() noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the vector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the static vector.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline typename StaticVector<Type,N,TF>::ConstPointer StaticVector<Type,N,TF>::data() const noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the static vector.
//
// \return Iterator to the first element of the static vector.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline typename StaticVector<Type,N,TF>::Iterator StaticVector<Type,N,TF>::begin() noexcept
{
   return Iterator( v_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the static vector.
//
// \return Iterator to the first element of the static vector.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline typename StaticVector<Type,N,TF>::ConstIterator StaticVector<Type,N,TF>::begin() const noexcept
{
   return ConstIterator( v_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the static vector.
//
// \return Iterator to the first element of the static vector.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline typename StaticVector<Type,N,TF>::ConstIterator StaticVector<Type,N,TF>::cbegin() const noexcept
{
   return ConstIterator( v_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the static vector.
//
// \return Iterator just past the last element of the static vector.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline typename StaticVector<Type,N,TF>::Iterator StaticVector<Type,N,TF>::end() noexcept
{
   return Iterator( v_ + N );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the static vector.
//
// \return Iterator just past the last element of the static vector.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline typename StaticVector<Type,N,TF>::ConstIterator StaticVector<Type,N,TF>::end() const noexcept
{
   return ConstIterator( v_ + N );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the static vector.
//
// \return Iterator just past the last element of the static vector.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline typename StaticVector<Type,N,TF>::ConstIterator StaticVector<Type,N,TF>::cend() const noexcept
{
   return ConstIterator( v_ + N );
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline StaticVector<Type,N,TF>& StaticVector<Type,N,TF>::operator=( const Type& rhs )
{
   for( size_t i=0UL; i<N; ++i )
      v_[i] = rhs;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List assignment to all vector elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to static vector.
//
// This assignment operator offers the option to directly assign to all elements of the vector
// by means of an initializer list:

   \code
   blaze::StaticVector<double,3UL> v;
   v = { 4.2, 6.3, -1.2 };
   \endcode

// The vector elements are assigned the values from the given initializer list. Missing values
// are reset to their default state. Note that in case the size of the initializer list exceeds
// the size of the vector, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline StaticVector<Type,N,TF>& StaticVector<Type,N,TF>::operator=( initializer_list<Type> list )
{
   if( list.size() > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to static vector" );
   }

   std::fill( std::copy( list.begin(), list.end(), v_.data() ), v_.data()+N, Type() );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array assignment to all vector elements.
//
// \param array M-dimensional array for the assignment.
// \return Reference to the assigned vector.
//
// This assignment operator offers the option to directly set all elements of the vector:

   \code
   const double init[3] = { 1.0, 2.0 };
   blaze::StaticVector<double,3> v;
   v = init;
   \endcode

// The vector is assigned the values from the given array. Missing values are initialized with
// default values (as e.g. the third value in the example).
*/
template< typename Type     // Data type of the vector
        , size_t N          // Number of elements
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the initialization array
inline StaticVector<Type,N,TF>& StaticVector<Type,N,TF>::operator=( const Other (&array)[N] )
{
   for( size_t i=0UL; i<N; ++i )
      v_[i] = array[i];
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Copy assignment operator for StaticVector.
//
// \param rhs Vector to be copied.
// \return Reference to the assigned vector.
//
// Explicit definition of a copy assignment operator for performance reasons.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline StaticVector<Type,N,TF>& StaticVector<Type,N,TF>::operator=( const StaticVector& rhs )
{
   using blaze::assign;

   assign( *this, ~rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different StaticVector instances.
//
// \param rhs Vector to be copied.
// \return Reference to the assigned vector.
*/
template< typename Type     // Data type of the vector
        , size_t N          // Number of elements
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign vector
inline StaticVector<Type,N,TF>& StaticVector<Type,N,TF>::operator=( const StaticVector<Other,N,TF>& rhs )
{
   using blaze::assign;

   assign( *this, ~rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different vectors.
//
// \param rhs Vector to be copied.
// \return Reference to the assigned vector.
// \exception std::invalid_argument Invalid assignment to static vector.
//
// This constructor initializes the vector as a copy of the given vector. In case the
// size of the given vector is not N, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline StaticVector<Type,N,TF>& StaticVector<Type,N,TF>::operator=( const Vector<VT,TF>& rhs )
{
   using blaze::assign;

   if( (~rhs).size() != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to static vector" );
   }

   if( (~rhs).canAlias( this ) ) {
      StaticVector tmp( ~rhs );
      swap( tmp );
   }
   else {
      if( IsSparseVector<VT>::value )
         reset();
      assign( *this, ~rhs );
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline StaticVector<Type,N,TF>& StaticVector<Type,N,TF>::operator+=( const Vector<VT,TF>& rhs )
{
   using blaze::addAssign;

   if( (~rhs).size() != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      StaticVector tmp( ~rhs );
      addAssign( *this, tmp );
   }
   else {
      addAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment operator for the subtraction of a vector (\f$ \vec{a}-=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be subtracted from the vector.
// \return Reference to the vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline StaticVector<Type,N,TF>& StaticVector<Type,N,TF>::operator-=( const Vector<VT,TF>& rhs )
{
   using blaze::subAssign;

   if( (~rhs).size() != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      StaticVector tmp( ~rhs );
      subAssign( *this, tmp );
   }
   else {
      subAssign( *this, ~rhs );
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline StaticVector<Type,N,TF>& StaticVector<Type,N,TF>::operator*=( const Vector<VT,TF>& rhs )
{
   using blaze::multAssign;

   if( (~rhs).size() != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( IsSparseVector<VT>::value || (~rhs).canAlias( this ) ) {
      const StaticVector tmp( *this * (~rhs) );
      this->operator=( tmp );
   }
   else {
      multAssign( *this, ~rhs );
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline StaticVector<Type,N,TF>& StaticVector<Type,N,TF>::operator/=( const DenseVector<VT,TF>& rhs )
{
   using blaze::divAssign;

   if( (~rhs).size() != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const StaticVector tmp( *this / (~rhs) );
      this->operator=( tmp );
   }
   else {
      divAssign( *this, ~rhs );
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
        , size_t N          // Number of elements
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, StaticVector<Type,N,TF> >&
   StaticVector<Type,N,TF>::operator*=( Other rhs )
{
   using blaze::assign;

   assign( *this, (*this) * rhs );

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
        , size_t N          // Number of elements
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, StaticVector<Type,N,TF> >&
   StaticVector<Type,N,TF>::operator/=( Other rhs )
{
   using blaze::assign;

   BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

   assign( *this, (*this) / rhs );

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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline constexpr size_t StaticVector<Type,N,TF>::size() const noexcept
{
   return N;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the vector.
//
// \return The capacity of the vector.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline constexpr size_t StaticVector<Type,N,TF>::capacity() const noexcept
{
   return NN;
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline size_t StaticVector<Type,N,TF>::nonZeros() const
{
   size_t nonzeros( 0 );

   for( size_t i=0UL; i<N; ++i ) {
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline void StaticVector<Type,N,TF>::reset()
{
   using blaze::clear;
   for( size_t i=0UL; i<N; ++i )
      clear( v_[i] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the vector by the scalar value \a scalar (\f$ \vec{a}*=s \f$).
//
// \param scalar The scalar value for the vector scaling.
// \return Reference to the vector.
*/
template< typename Type     // Data type of the vector
        , size_t N          // Number of elements
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the scalar value
inline StaticVector<Type,N,TF>& StaticVector<Type,N,TF>::scale( const Other& scalar )
{
   for( size_t i=0; i<N; ++i )
      v_[i] *= scalar;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two static vectors.
//
// \param v The vector to be swapped.
// \return void
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline void StaticVector<Type,N,TF>::swap( StaticVector& v ) noexcept
{
   using std::swap;

   for( size_t i=0UL; i<N; ++i )
      swap( v_[i], v.v_[i] );
}
//*************************************************************************************************




//=================================================================================================
//
//  MEMORY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Class specific implementation of operator new.
//
// \param size The total number of bytes to be allocated.
// \return Pointer to the newly allocated memory.
// \exception std::bad_alloc Allocation failed.
//
// This class-specific implementation of operator new provides the functionality to allocate
// dynamic memory based on the alignment restrictions of the StaticVector class template.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline void* StaticVector<Type,N,TF>::operator new( std::size_t size )
{
   UNUSED_PARAMETER( size );

   BLAZE_INTERNAL_ASSERT( size == sizeof( StaticVector ), "Invalid number of bytes detected" );

   return allocate<StaticVector>( 1UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of operator new[].
//
// \param size The total number of bytes to be allocated.
// \return Pointer to the newly allocated memory.
// \exception std::bad_alloc Allocation failed.
//
// This class-specific implementation of operator new provides the functionality to allocate
// dynamic memory based on the alignment restrictions of the StaticVector class template.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline void* StaticVector<Type,N,TF>::operator new[]( std::size_t size )
{
   BLAZE_INTERNAL_ASSERT( size >= sizeof( StaticVector )       , "Invalid number of bytes detected" );
   BLAZE_INTERNAL_ASSERT( size %  sizeof( StaticVector ) == 0UL, "Invalid number of bytes detected" );

   return allocate<StaticVector>( size/sizeof(StaticVector) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of the no-throw operator new.
//
// \param size The total number of bytes to be allocated.
// \return Pointer to the newly allocated memory.
// \exception std::bad_alloc Allocation failed.
//
// This class-specific implementation of operator new provides the functionality to allocate
// dynamic memory based on the alignment restrictions of the StaticVector class template.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline void* StaticVector<Type,N,TF>::operator new( std::size_t size, const std::nothrow_t& )
{
   UNUSED_PARAMETER( size );

   BLAZE_INTERNAL_ASSERT( size == sizeof( StaticVector ), "Invalid number of bytes detected" );

   return allocate<StaticVector>( 1UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of the no-throw operator new[].
//
// \param size The total number of bytes to be allocated.
// \return Pointer to the newly allocated memory.
// \exception std::bad_alloc Allocation failed.
//
// This class-specific implementation of operator new provides the functionality to allocate
// dynamic memory based on the alignment restrictions of the StaticVector class template.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline void* StaticVector<Type,N,TF>::operator new[]( std::size_t size, const std::nothrow_t& )
{
   BLAZE_INTERNAL_ASSERT( size >= sizeof( StaticVector )       , "Invalid number of bytes detected" );
   BLAZE_INTERNAL_ASSERT( size %  sizeof( StaticVector ) == 0UL, "Invalid number of bytes detected" );

   return allocate<StaticVector>( size/sizeof(StaticVector) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of operator delete.
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline void StaticVector<Type,N,TF>::operator delete( void* ptr )
{
   deallocate( static_cast<StaticVector*>( ptr ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of operator delete[].
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline void StaticVector<Type,N,TF>::operator delete[]( void* ptr )
{
   deallocate( static_cast<StaticVector*>( ptr ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of no-throw operator delete.
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline void StaticVector<Type,N,TF>::operator delete( void* ptr, const std::nothrow_t& )
{
   deallocate( static_cast<StaticVector*>( ptr ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of no-throw operator delete[].
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline void StaticVector<Type,N,TF>::operator delete[]( void* ptr, const std::nothrow_t& )
{
   deallocate( static_cast<StaticVector*>( ptr ) );
}
//*************************************************************************************************




//=================================================================================================
//
//  DEBUGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the invariants of the static vector are intact.
//
// \return \a true in case the static vector's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the static vector are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline bool StaticVector<Type,N,TF>::isIntact() const noexcept
{
   if( IsNumeric<Type>::value ) {
      for( size_t i=N; i<NN; ++i ) {
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
        , size_t N          // Number of elements
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool StaticVector<Type,N,TF>::canAlias( const Other* alias ) const noexcept
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
        , size_t N          // Number of elements
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool StaticVector<Type,N,TF>::isAliased( const Other* alias ) const noexcept
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline bool StaticVector<Type,N,TF>::isAligned() const noexcept
{
   return true;
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE typename StaticVector<Type,N,TF>::SIMDType
   StaticVector<Type,N,TF>::load( size_t index ) const noexcept
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE typename StaticVector<Type,N,TF>::SIMDType
   StaticVector<Type,N,TF>::loada( size_t index ) const noexcept
{
   using blaze::loada;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < N, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= NN, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( &v_[index] ), "Invalid alignment detected" );

   return loada( &v_[index] );
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE typename StaticVector<Type,N,TF>::SIMDType
   StaticVector<Type,N,TF>::loadu( size_t index ) const noexcept
{
   using blaze::loadu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < N, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= NN, "Invalid vector access index" );

   return loadu( &v_[index] );
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void
   StaticVector<Type,N,TF>::store( size_t index, const SIMDType& value ) noexcept
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void
   StaticVector<Type,N,TF>::storea( size_t index, const SIMDType& value ) noexcept
{
   using blaze::storea;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < N, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= NN , "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( &v_[index] ), "Invalid alignment detected" );

   storea( &v_[index], value );
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void
   StaticVector<Type,N,TF>::storeu( size_t index, const SIMDType& value ) noexcept
{
   using blaze::storeu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < N, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= NN, "Invalid vector access index" );

   storeu( &v_[index], value );
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void
   StaticVector<Type,N,TF>::stream( size_t index, const SIMDType& value ) noexcept
{
   using blaze::stream;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < N, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= NN, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index % SIMDSIZE == 0UL, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( &v_[index] ), "Invalid alignment detected" );

   stream( &v_[index], value );
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename StaticVector<Type,N,TF>::BLAZE_TEMPLATE VectorizedAssign<VT> >
   StaticVector<Type,N,TF>::assign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

   for( size_t i=0UL; i<N; ++i )
      v_[i] = (~rhs)[i];
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename StaticVector<Type,N,TF>::BLAZE_TEMPLATE VectorizedAssign<VT> >
   StaticVector<Type,N,TF>::assign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

   const bool remainder( !usePadding || !IsPadded<VT>::value );

   const size_t ipos( ( remainder )?( N & size_t(-SIMDSIZE) ):( N ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );

   for( ; i<ipos; i+=SIMDSIZE ) {
      store( i, (~rhs).load(i) );
   }
   for( ; remainder && i<N; ++i ) {
      v_[i] = (~rhs)[i];
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void StaticVector<Type,N,TF>::assign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename StaticVector<Type,N,TF>::BLAZE_TEMPLATE VectorizedAddAssign<VT> >
   StaticVector<Type,N,TF>::addAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

   for( size_t i=0UL; i<N; ++i )
      v_[i] += (~rhs)[i];
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename StaticVector<Type,N,TF>::BLAZE_TEMPLATE VectorizedAddAssign<VT> >
   StaticVector<Type,N,TF>::addAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

   const bool remainder( !usePadding || !IsPadded<VT>::value );

   const size_t ipos( ( remainder )?( N & size_t(-SIMDSIZE) ):( N ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );

   for( ; i<ipos; i+=SIMDSIZE ) {
      store( i, load(i) + (~rhs).load(i) );
   }
   for( ; remainder && i<N; ++i ) {
      v_[i] += (~rhs)[i];
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void StaticVector<Type,N,TF>::addAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename StaticVector<Type,N,TF>::BLAZE_TEMPLATE VectorizedSubAssign<VT> >
   StaticVector<Type,N,TF>::subAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

   for( size_t i=0UL; i<N; ++i )
      v_[i] -= (~rhs)[i];
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename StaticVector<Type,N,TF>::BLAZE_TEMPLATE VectorizedSubAssign<VT> >
   StaticVector<Type,N,TF>::subAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

   const bool remainder( !usePadding || !IsPadded<VT>::value );

   const size_t ipos( ( remainder )?( N & size_t(-SIMDSIZE) ):( N ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );

   for( ; i<ipos; i+=SIMDSIZE ) {
      store( i, load(i) - (~rhs).load(i) );
   }
   for( ; remainder && i<N; ++i ) {
      v_[i] -= (~rhs)[i];
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void StaticVector<Type,N,TF>::subAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename StaticVector<Type,N,TF>::BLAZE_TEMPLATE VectorizedMultAssign<VT> >
   StaticVector<Type,N,TF>::multAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

   for( size_t i=0UL; i<N; ++i )
      v_[i] *= (~rhs)[i];
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename StaticVector<Type,N,TF>::BLAZE_TEMPLATE VectorizedMultAssign<VT> >
   StaticVector<Type,N,TF>::multAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

   const bool remainder( !usePadding || !IsPadded<VT>::value );

   const size_t ipos( ( remainder )?( N & size_t(-SIMDSIZE) ):( N ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );

   for( ; i<ipos; i+=SIMDSIZE ) {
      store( i, load(i) * (~rhs).load(i) );
   }
   for( ; remainder && i<N; ++i ) {
      v_[i] *= (~rhs)[i];
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void StaticVector<Type,N,TF>::multAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

   const StaticVector tmp( serial( *this ) );

   reset();

   for( ConstIterator_<VT> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      v_[element->index()] = tmp[element->index()] * element->value();
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename StaticVector<Type,N,TF>::BLAZE_TEMPLATE VectorizedDivAssign<VT> >
   StaticVector<Type,N,TF>::divAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

   for( size_t i=0UL; i<N; ++i )
      v_[i] /= (~rhs)[i];
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
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename StaticVector<Type,N,TF>::BLAZE_TEMPLATE VectorizedDivAssign<VT> >
   StaticVector<Type,N,TF>::divAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( (~rhs).size() == N, "Invalid vector sizes" );

   const size_t ipos( N & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( N - ( N % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   size_t i( 0UL );

   for( ; i<ipos; i+=SIMDSIZE ) {
      store( i, load(i) / (~rhs).load(i) );
   }
   for( ; i<N; ++i ) {
      v_[i] /= (~rhs)[i];
   }
}
//*************************************************************************************************








//=================================================================================================
//
//  UNDEFINED CLASS TEMPLATE SPECIALIZATION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of StaticVector for 0 elements.
// \ingroup static_vector
//
// This specialization of the StaticVector class template is left undefined and therefore
// prevents the instantiation for 0 elements.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
class StaticVector<Type,0UL,TF>;
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  STATICVECTOR OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name StaticVector operators */
//@{
template< typename Type, size_t N, bool TF >
inline void reset( StaticVector<Type,N,TF>& v );

template< typename Type, size_t N, bool TF >
inline void clear( StaticVector<Type,N,TF>& v );

template< typename Type, size_t N, bool TF >
inline bool isDefault( const StaticVector<Type,N,TF>& v );

template< typename Type, size_t N, bool TF >
inline bool isIntact( const StaticVector<Type,N,TF>& v ) noexcept;

template< typename Type, bool TF >
inline const StaticVector<Type,2UL,TF> perp( const StaticVector<Type,2UL,TF>& v );

template< typename Type, bool TF >
inline const StaticVector<Type,3UL,TF> perp( const StaticVector<Type,3UL,TF>& v );

template< typename Type, size_t N, bool TF >
inline void swap( StaticVector<Type,N,TF>& a, StaticVector<Type,N,TF>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given static vector.
// \ingroup static_vector
//
// \param v The vector to be resetted.
// \return void
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline void reset( StaticVector<Type,N,TF>& v )
{
   v.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given static vector.
// \ingroup static_vector
//
// \param v The vector to be cleared.
// \return void
//
// Clearing a static vector is equivalent to resetting it via the reset() function.
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline void clear( StaticVector<Type,N,TF>& v )
{
   v.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given static vector is in default state.
// \ingroup static_vector
//
// \param v The vector to be tested for its default state.
// \return \a true in case the given vector is component-wise zero, \a false otherwise.
//
// This function checks whether the static vector is in default state. For instance, in case
// the static vector is instantiated for a built-in integral or floating point data type, the
// function returns \a true in case all vector elements are 0 and \a false in case any vector
// element is not 0. Following example demonstrates the use of the \a isDefault function:

   \code
   blaze::StaticVector<double,3> a;
   // ... Initialization
   if( isDefault( a ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline bool isDefault( const StaticVector<Type,N,TF>& v )
{
   for( size_t i=0UL; i<N; ++i )
      if( !isDefault( v[i] ) ) return false;
   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given static vector are intact.
// \ingroup static_vector
//
// \param v The static vector to be tested.
// \return \a true in case the given vector's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the static vector are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   blaze::StaticVector<double,3> a;
   // ... Resizing and initialization
   if( isIntact( a ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline bool isIntact( const StaticVector<Type,N,TF>& v ) noexcept
{
   return v.isIntact();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Unary perp dot product operator for the calculation of a perpendicular vector
//        (\f$ \vec{a}=\vec{b}^\perp \f$).
//
// \param v The vector to be rotated.
// \return The perpendicular vector.
//
// The "perp dot product" \f$ \vec{a}^\perp \cdot b \f$ for the vectors \f$ \vec{a} \f$ and
// \f$ \vec{b} \f$ is a modification of the two-dimensional dot product in which \f$ \vec{a} \f$
// is replaced by the perpendicular vector rotated 90 degrees to the left defined by Hill (1994).
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline const StaticVector<Type,2UL,TF> perp( const StaticVector<Type,2UL,TF>& v )
{
   return StaticVector<Type,2UL,TF>( -v[1UL], v[0UL] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creates a perpendicular vector b which satisfies \f$ \vec{a} \cdot \vec{b} = 0 \f$.
//
// \param v The vector to be rotated.
// \return The perpendicular vector.
//
// \note The perpendicular vector may have any length!
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline const StaticVector<Type,3UL,TF> perp( const StaticVector<Type,3UL,TF>& v )
{
   if( v[0] != Type() || v[1] != Type() )
      return StaticVector<Type,3UL,TF>( v[1UL], -v[0UL], Type() );
   else
      return StaticVector<Type,3UL,TF>( Type(), v[2UL], -v[1UL] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two static vectors.
// \ingroup static_vector
//
// \param a The first vector to be swapped.
// \param b The second vector to be swapped.
// \return void
*/
template< typename Type  // Data type of the vector
        , size_t N       // Number of elements
        , bool TF >      // Transpose flag
inline void swap( StaticVector<Type,N,TF>& a, StaticVector<Type,N,TF>& b ) noexcept
{
   a.swap( b );
}
//*************************************************************************************************




//=================================================================================================
//
//  SIZE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, size_t N, bool TF >
struct Size< StaticVector<T,N,TF> >
{
   static const size_t value = N;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HASCONSTDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, size_t N, bool TF >
struct HasConstDataAccess< StaticVector<T,N,TF> > : public TrueType
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
template< typename T, size_t N, bool TF >
struct HasMutableDataAccess< StaticVector<T,N,TF> > : public TrueType
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
template< typename T, size_t N, bool TF >
struct IsAligned< StaticVector<T,N,TF> > : public TrueType
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
template< typename T, size_t N, bool TF >
struct IsPadded< StaticVector<T,N,TF> > : public BoolConstant<usePadding>
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
template< typename T1, size_t N, bool TF, typename T2 >
struct AddTrait< StaticVector<T1,N,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< AddTrait_<T1,T2>, N, TF >;
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
template< typename T1, size_t N, bool TF, typename T2 >
struct SubTrait< StaticVector<T1,N,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< SubTrait_<T1,T2>, N, TF >;
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
template< typename T1, size_t N, bool TF, typename T2 >
struct MultTrait< StaticVector<T1,N,TF>, T2, EnableIf_<IsNumeric<T2> > >
{
   using Type = StaticVector< MultTrait_<T1,T2>, N, TF >;
};

template< typename T1, typename T2, size_t N, bool TF >
struct MultTrait< T1, StaticVector<T2,N,TF>, EnableIf_<IsNumeric<T1> > >
{
   using Type = StaticVector< MultTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct MultTrait< StaticVector<T1,N,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< MultTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t M, typename T2, size_t N >
struct MultTrait< StaticVector<T1,M,false>, StaticVector<T2,N,true> >
{
   using Type = StaticMatrix< MultTrait_<T1,T2>, M, N, false >;
};

template< typename T1, size_t N, typename T2 >
struct MultTrait< StaticVector<T1,N,true>, StaticVector<T2,N,false> >
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
template< typename T1, typename T2, bool TF >
struct CrossTrait< StaticVector<T1,3UL,TF>, StaticVector<T2,3UL,TF> >
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
template< typename T1, size_t N, bool TF, typename T2 >
struct DivTrait< StaticVector<T1,N,TF>, T2, EnableIf_<IsNumeric<T2> > >
{
   using Type = StaticVector< DivTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2 >
struct DivTrait< StaticVector<T1,N,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< DivTrait_<T1,T2>, N, TF >;
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
template< typename T1, size_t N, bool TF, typename T2 >
struct MathTrait< StaticVector<T1,N,TF>, StaticVector<T2,N,TF> >
{
   using HighType = StaticVector< typename MathTrait<T1,T2>::HighType, N, TF >;
   using LowType  = StaticVector< typename MathTrait<T1,T2>::LowType , N, TF >;
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
template< typename T1, size_t N, bool TF >
struct SubvectorTrait< StaticVector<T1,N,TF> >
{
   using Type = HybridVector<T1,N,TF>;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
