//=================================================================================================
/*!
//  \file blaze/math/dense/CustomVector.h
//  \brief Header file for the implementation of a customizable vector
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

#ifndef _BLAZE_MATH_DENSE_CUSTOMVECTOR_H_
#define _BLAZE_MATH_DENSE_CUSTOMVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <utility>
#include <boost/smart_ptr/shared_array.hpp>
#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/TransposeFlag.h>
#include <blaze/math/dense/DenseIterator.h>
#include <blaze/math/dense/DynamicVector.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/SparseVector.h>
#include <blaze/math/Forward.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/PaddingFlag.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/CrossTrait.h>
#include <blaze/math/traits/DivTrait.h>
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
#include <blaze/math/typetraits/IsCustom.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsSMPAssignable.h>
#include <blaze/math/typetraits/IsSparseVector.h>
#include <blaze/system/CacheSize.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Thresholds.h>
#include <blaze/system/TransposeFlag.h>
#include <blaze/util/AlignmentCheck.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Vectorizable.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/policies/NoDelete.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Template.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blaze/util/typetraits/IsClass.h>
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
/*!\defgroup custom_vector CustomVector
// \ingroup dense_vector
*/
/*!\brief Efficient implementation of a customizable vector.
// \ingroup custom_vector
//
// \section customvector_general General
//
// The CustomVector class template provides the functionality to represent an external array of
// elements of arbitrary type and a fixed size as a native \b Blaze dense vector data structure.
// Thus in contrast to all other dense vector types a custom vector does not perform any kind
// of memory allocation by itself, but it is provided with an existing array of element during
// construction. A custom vector can therefore be considered an alias to the existing array.
//
// The type of the elements, the properties of the given array of elements and the transpose
// flag of the vector can be specified via the following four template parameters:

   \code
   template< typename Type, bool AF, bool PF, bool TF >
   class CustomVector;
   \endcode

//  - Type: specifies the type of the vector elements. CustomVector can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
//  - AF  : specifies whether the represented, external arrays are properly aligned with
//          respect to the available instruction set (SSE, AVX, ...) or not.
//  - PF  : specified whether the represented, external arrays are properly padded with
//          respect to the available instruction set (SSE, AVX, ...) or not.
//  - TF  : specifies whether the vector is a row vector (\a blaze::rowVector) or a column
//          vector (\a blaze::columnVector). The default value is \a blaze::columnVector.
//
// The following examples give an impression of several possible types of custom vectors:

   \code
   using blaze::CustomVector;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;

   // Definition of a custom column vector for unaligned, unpadded integer arrays
   typedef CustomVector<int,unaligned,unpadded,columnVector>  UnalignedUnpadded;

   // Definition of a custom column vector for unaligned but padded 'float' arrays
   typedef CustomVector<float,unaligned,padded,columnVector>  UnalignedPadded;

   // Definition of a custom row vector for aligned, unpadded 'double' arrays
   typedef CustomVector<double,aligned,unpadded,rowVector>  AlignedUnpadded;

   // Definition of a custom row vector for aligned, padded 'complex<double>' arrays
   typedef CustomVector<complex<double>,aligned,padded,rowVector>  AlignedPadded;
   \endcode

// \n \section customvector_special_properties Special Properties of Custom Vectors
//
// In comparison with the remaining \b Blaze dense vector types CustomVector has several special
// characteristics. All of these result from the fact that a custom vector is not performing any
// kind of memory allocation, but instead is given an existing array of elements. The following
// sections discuss all of these characteristics:
//
//  -# <b>\ref customvector_memory_management</b>
//  -# <b>\ref customvector_copy_operations</b>
//  -# <b>\ref customvector_alignment</b>
//  -# <b>\ref customvector_padding</b>
//
// \n \subsection customvector_memory_management Memory Management
//
// The CustomVector class template acts as an adaptor for an existing array of elements. As such
// it provides everything that is required to use the array just like a native \b Blaze dense
// vector data structure. However, this flexibility comes with the price that the user of a custom
// vector is responsible for the resource management.
//
// When constructing a custom vector there are two choices: Either a user manually manages the
// array of elements outside the custom vector, or alternatively passes the responsibility for
// the memory management to an instance of CustomVector. In the second case the CustomVector
// class employs shared ownership between all copies of the custom vector, which reference the
// same array.
//
// The following examples give an impression of several possible types of custom vectors:

   \code
   using blaze::CustomVector;
   using blaze::ArrayDelete;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::columnVector;
   using blaze::rowVector;

   // Definition of a 3-dimensional custom vector with unaligned, unpadded and externally
   // managed integer array. Note that the std::vector must be guaranteed to outlive the
   // custom vector!
   std::vector<int> vec( 3UL );
   CustomVector<int,unaligned,unpadded> a( &vec[0], 3UL );

   // Definition of a custom row vector with size 3 for unaligned, unpadded integer arrays.
   // The responsibility for the memory management is passed to the custom vector by
   // providing a deleter of type 'blaze::ArrayDelete' that is used during the destruction
   // of the custom vector.
   CustomVector<int,unaligned,unpadded,rowVector> b( new int[3], 3UL, ArrayDelete() );

   // Definition of a custom vector with size 3 and capacity 16 with aligned and padded
   // integer array. The memory management is passed to the custom vector by providing a
   // deleter of type 'blaze::Deallocate'.
   CustomVector<int,aligned,padded> c( allocate<int>( 16UL ), 3UL, 16UL, Deallocate() );
   \endcode

// It is possible to pass any type of deleter to the constructor. The deleter is only required
// to provide a function call operator that can be passed the pointer to the managed array. As
// an example the following code snipped shows the implementation of two native \b Blaze deleters
// blaze::ArrayDelete and blaze::Deallocate:

   \code
   namespace blaze {

   struct ArrayDelete
   {
      template< typename Type >
      inline void operator()( Type ptr ) const { boost::checked_array_delete( ptr ); }
   };

   struct Deallocate
   {
      template< typename Type >
      inline void operator()( Type ptr ) const { deallocate( ptr ); }
   };

   } // namespace blaze
   \endcode

// \n \subsection customvector_copy_operations Copy Operations
//
// As with all dense vectors it is possible to copy construct a custom vector:

   \code
   using blaze::CustomVector;
   using blaze::unaligned;
   using blaze::unpadded;

   typedef CustomVector<int,unaligned,unpadded>  CustomType;

   std::vector<int> vec( 5UL, 10 );  // Vector of 5 integers of the value 10
   CustomType a( &vec[0], 5UL );     // Represent the std::vector as Blaze dense vector
   a[1] = 20;                        // Also modifies the std::vector

   CustomType b( a );  // Creating a copy of vector a
   b[2] = 20;          // Also affect vector a and the std::vector
   \endcode

// It is important to note that a custom vector acts as a reference to the specified array. Thus
// the result of the copy constructor is a new custom vector that is referencing and representing
// the same array as the original custom vector. In case a deleter has been provided to the first
// custom vector, both vectors share the responsibility to destroy the array when the last vector
// goes out of scope.
//
// In contrast to copy construction, just as with references, copy assignment does not change
// which array is referenced by the custom vector, but modifies the values of the array:

   \code
   std::vector<int> vec2( 5UL, 4 );  // Vector of 5 integers of the value 4
   CustomType c( &vec2[0], 5UL );    // Represent the std::vector as Blaze dense vector

   a = c;  // Copy assignment: Set all values of vector a and b to 4.
   \endcode

// \n \subsection customvector_alignment Alignment
//
// In case the custom vector is specified as \a aligned the passed array must be guaranteed to
// be aligned according to the requirements of the used instruction set (SSE, AVX, ...). For
// instance, if AVX is active an array of integers must be 32-bit aligned:

   \code
   using blaze::CustomVector;
   using blaze::Deallocate;
   using blaze::aligned;
   using blaze::unpadded;

   int* array = blaze::allocate<int>( 5UL );  // Needs to be 32-bit aligned
   CustomVector<int,aligned,unpadded> a( array, 5UL, Deallocate() );
   \endcode

// In case the alignment requirements are violated, a \a std::invalid_argument exception is
// thrown.
//
// \n \subsection customvector_padding Padding
//
// Adding padding elements to the end of an array can have a significant impact on performance.
// For instance, assuming that AVX is available, then two aligned, padded, 3-dimensional vectors
// of double precision values can be added via a single SIMD addition operations:

   \code
   using blaze::CustomVector;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::padded;

   typedef CustomVector<double,aligned,padded>  CustomType;

   // Creating padded custom vectors of size 3 and a capacity of 4
   CustomType a( allocate<double>( 4UL ), 3UL, 4UL, Deallocate() );
   CustomType b( allocate<double>( 4UL ), 3UL, 4UL, Deallocate() );
   CustomType c( allocate<double>( 4UL ), 3UL, 4UL, Deallocate() );

   // ... Initialization

   c = a + b;  // AVX-based vector addition
   \endcode

// In this example, maximum performance is possible. However, in case no padding elements are
// inserted, a scalar addition has to be used:

   \code
   using blaze::CustomVector;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::unpadded;

   typedef CustomVector<double,aligned,unpadded>  CustomType;

   // Creating unpadded custom vector of size 3
   CustomType a( allocate<double>( 3UL ), 3UL, Deallocate() );
   CustomType b( allocate<double>( 3UL ), 3UL, Deallocate() );
   CustomType c( allocate<double>( 3UL ), 3UL, Deallocate() );

   // ... Initialization

   c = a + b;  // Scalar vector addition
   \endcode

// Note the different number of constructor parameters for unpadded and padded custom vectors:
// In contrast to unpadded vectors, where during the construction only the size of the array
// has to be specified, during the construction of a padded custom vector it is additionally
// necessary to explicitly specify the capacity of the array.
//
// The number of padding elements is required to be sufficient with respect to the available
// instruction set: In case of an aligned padded custom vector the added padding elements must
// guarantee that the capacity is greater or equal than the size and a multiple of the SIMD vector
// width. In case of unaligned padded vectors the number of padding elements can be greater or
// equal the number of padding elements of an aligned padded custom vector. In case the padding
// is insufficient with respect to the available instruction set, a \a std::invalid_argument
// exception is thrown.
//
// Please also note that \b Blaze will zero initialize the padding elements in order to achieve
// maximum performance!
//
//
// \n \section customvector_arithmetic_operations Arithmetic Operations
//
// The use of custom vectors in arithmetic operations is designed to be as natural and intuitive
// as possible. All operations (addition, subtraction, multiplication, scaling, ...) can be
// expressed similar to a text book representation. Also, custom vectors can be combined with all
// other dense and sparse vectors and matrices. The following example gives an impression of the
// use of CustomVector:

   \code
   using blaze::CustomVector;
   using blaze::CompressedVector;
   using blaze::DynamicMatrix;
   using blaze::ArrayDelete;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;

   // Non-initialized custom column vector of size 2. All given arrays are considered to be
   // unaligned and unpadded. The memory is managed via 'ArrayDelete'.
   CustomVector<double,unaligned,unpadded> a( new double[2], 2UL, ArrayDelete() );

   a[0] = 1.0;  // Initialization of the first element
   a[1] = 2.0;  // Initialization of the second element

   // Non-initialized custom column vector of size 2 and capacity 4. All given arrays are required
   // to be properly aligned and padded. The memory is managed via 'Deallocate'.
   CustomVector<double,aligned,padded> b( allocate<double>( 4UL ), 2UL, 4UL, Deallocate() );

   b = 2.0;  // Homogeneous initialization of all elements

   CompressedVector<float> c( 2 );  // Empty sparse single precision vector
   DynamicVector<double>   d;       // Default constructed dynamic vector
   DynamicMatrix<double>   A;       // Default constructed row-major matrix

   d = a + b;  // Vector addition between custom vectors of equal element type
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
        , bool AF                           // Alignment flag
        , bool PF                           // Padding flag
        , bool TF = defaultTransposeFlag >  // Transpose flag
class CustomVector : public DenseVector< CustomVector<Type,AF,PF,TF>, TF >
{
 public:
   //**Type definitions****************************************************************************
   typedef CustomVector<Type,AF,PF,TF>  This;           //!< Type of this CustomVector instance.
   typedef DenseVector<This,TF>         BaseType;       //!< Base type of this CustomVector instance.
   typedef DynamicVector<Type,TF>       ResultType;     //!< Result type for expression template evaluations.
   typedef DynamicVector<Type,!TF>      TransposeType;  //!< Transpose type for expression template evaluations.
   typedef Type                         ElementType;    //!< Type of the vector elements.
   typedef SIMDTrait_<ElementType>      SIMDType;       //!< SIMD type of the vector elements.
   typedef const Type&                  ReturnType;     //!< Return type for expression template evaluations
   typedef const CustomVector&          CompositeType;  //!< Data type for composite expression templates.

   typedef Type&        Reference;       //!< Reference to a non-constant vector value.
   typedef const Type&  ConstReference;  //!< Reference to a constant vector value.
   typedef Type*        Pointer;         //!< Pointer to a non-constant vector value.
   typedef const Type*  ConstPointer;    //!< Pointer to a constant vector value.

   typedef DenseIterator<Type,AF>        Iterator;       //!< Iterator over non-constant elements.
   typedef DenseIterator<const Type,AF>  ConstIterator;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a CustomVector with different data/element type.
   */
   template< typename ET >  // Data type of the other vector
   struct Rebind {
      typedef CustomVector<ET,AF,PF,TF>  Other;  //!< The type of the other CustomVector.
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
   enum : bool { smpAssignable = !IsSMPAssignable<Type>::value };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline CustomVector();
   explicit inline CustomVector( Type* ptr, size_t n );
   explicit inline CustomVector( Type* ptr, size_t n, size_t nn );

   template< typename Deleter, typename = EnableIf_<IsClass<Deleter> > >
   explicit inline CustomVector( Type* ptr, size_t n, Deleter d );

   template< typename Deleter >
   explicit inline CustomVector( Type* ptr, size_t n, size_t nn, Deleter d );

   inline CustomVector( const CustomVector& v );
   inline CustomVector( CustomVector&& v ) noexcept;
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
   inline CustomVector& operator=( const Type& rhs );
   inline CustomVector& operator=( initializer_list<Type> list );

   template< typename Other, size_t N >
   inline CustomVector& operator=( const Other (&array)[N] );

   inline CustomVector& operator=( const CustomVector& rhs );
   inline CustomVector& operator=( CustomVector&& rhs ) noexcept;

   template< typename VT > inline CustomVector& operator= ( const Vector<VT,TF>& rhs );
   template< typename VT > inline CustomVector& operator+=( const Vector<VT,TF>& rhs );
   template< typename VT > inline CustomVector& operator-=( const Vector<VT,TF>& rhs );
   template< typename VT > inline CustomVector& operator*=( const Vector<VT,TF>& rhs );
   template< typename VT > inline CustomVector& operator/=( const DenseVector<VT,TF>& rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, CustomVector >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, CustomVector >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                                inline size_t        size() const noexcept;
                                inline size_t        capacity() const noexcept;
                                inline size_t        nonZeros() const;
                                inline void          reset();
                                inline void          clear();
   template< typename Other   > inline CustomVector& scale( const Other& scalar );
                                inline void          swap( CustomVector& v ) noexcept;
   //@}
   //**********************************************************************************************

   //**Resource management functions***************************************************************
   /*!\name Resource management functions */
   //@{
   inline void reset( Type* ptr, size_t n );
   inline void reset( Type* ptr, size_t n, size_t nn );

   template< typename Deleter, typename = EnableIf_<IsClass<Deleter> > >
   inline void reset( Type* ptr, size_t n, Deleter d );

   template< typename Deleter >
   inline void reset( Type* ptr, size_t n, size_t nn, Deleter d );
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

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   enum : size_t { SIMDSIZE = SIMDTrait<ElementType>::size };
   //**********************************************************************************************

 public:
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
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t size_;                  //!< The size/dimension of the custom vector.
   boost::shared_array<Type> v_;  //!< The custom array of elements.
                                  /*!< Access to the array of elements is gained via the
                                       subscript operator. The order of the elements is
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
/*!\brief The default constructor for CustomVector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,PF,TF>::CustomVector()
   : size_( 0UL )  // The size/dimension of the vector
   , v_   (     )  // The custom array of elements
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for an unpadded custom vector of size \a n.
//
// \param ptr The array of elements to be used by the vector.
// \param n The number of array elements to be used by the custom vector.
// \exception std::invalid_argument Invalid setup of custom vector.
//
// This constructor creates an unpadded custom vector of size \a n. The construction fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This constructor is \b NOT available for padded custom vectors!
// \note The custom vector does \b NOT take responsibility for the given array of elements!
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,PF,TF>::CustomVector( Type* ptr, size_t n )
   : size_( n )  // The size/dimension of the vector
   , v_   (   )  // The custom array of elements
{
   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && !checkAlignment( ptr ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   v_.reset( ptr, NoDelete() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a padded custom vector of size \a n and capacity \a nn.
//
// \param ptr The array of elements to be used by the vector.
// \param n The number of array elements to be used by the custom vector.
// \param nn The maximum size of the given array.
// \exception std::invalid_argument Invalid setup of custom vector.
//
// This constructor creates a padded custom vector of size \a n and capacity \a nn. The
// construction fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified capacity \a nn is insufficient for the given data type \a Type and the
//    available instruction set.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This constructor is \b NOT available for unpadded custom vectors!
// \note The custom vector does \b NOT take responsibility for the given array of elements!
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,PF,TF>::CustomVector( Type* ptr, size_t n, size_t nn )
   : size_( 0UL )  // The size/dimension of the vector
   , v_   (     )  // The custom array of elements
{
   BLAZE_STATIC_ASSERT( PF == padded );

   UNUSED_PARAMETER( ptr, n, nn );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for an unpadded custom vector of size \a n.
//
// \param ptr The array of elements to be used by the vector.
// \param n The number of array elements to be used by the custom vector.
// \param d The deleter to destroy the array of elements.
// \exception std::invalid_argument Invalid setup of custom vector.
//
// This constructor creates an unpadded custom vector of size \a n. The construction fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This constructor is \b NOT available for padded custom vectors!
*/
template< typename Type     // Data type of the vector
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool TF >         // Transpose flag
template< typename Deleter  // Type of the custom deleter
        , typename >        // Type restriction on the custom deleter
inline CustomVector<Type,AF,PF,TF>::CustomVector( Type* ptr, size_t n, Deleter d )
   : size_( n )  // The size/dimension of the vector
   , v_   (   )  // The custom array of elements
{
   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && !checkAlignment( ptr ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   v_.reset( ptr, d );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a padded custom vector of size \a n and capacity \a nn.
//
// \param ptr The array of elements to be used by the vector.
// \param n The number of array elements to be used by the custom vector.
// \param nn The maximum size of the given array.
// \param d The deleter to destroy the array of elements.
// \exception std::invalid_argument Invalid setup of custom vector.
//
// This constructor creates a padded custom vector of size \a n and capacity \a nn. The
// construction fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified capacity \a nn is insufficient for the given data type \a Type and the
//    available instruction set.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This constructor is \b NOT available for unpadded custom vectors!
*/
template< typename Type       // Data type of the vector
        , bool AF             // Alignment flag
        , bool PF             // Padding flag
        , bool TF >           // Transpose flag
template< typename Deleter >  // Type of the custom deleter
inline CustomVector<Type,AF,PF,TF>::CustomVector( Type* ptr, size_t n, size_t nn, Deleter d )
   : size_( 0UL )  // The size/dimension of the vector
   , v_   (     )  // The custom array of elements
{
   BLAZE_STATIC_ASSERT( PF == padded );

   UNUSED_PARAMETER( ptr, n, nn, d );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for CustomVector.
//
// \param v Vector to be copied.
//
// The copy constructor initializes the custom vector as an exact copy of the given custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,PF,TF>::CustomVector( const CustomVector& v )
   : size_( v.size_ )  // The size/dimension of the vector
   , v_   ( v.v_    )  // The custom array of elements
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The move constructor for CustomVector.
//
// \param v The vector to be moved into this instance.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,PF,TF>::CustomVector( CustomVector&& v ) noexcept
   : size_( v.size_ )            // The size/dimension of the vector
   , v_   ( std::move( v.v_ ) )  // The custom array of elements
{
   v.size_ = 0UL;

   BLAZE_INTERNAL_ASSERT( v.data() == nullptr, "Invalid data reference detected" );
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,PF,TF>::Reference
   CustomVector<Type,AF,PF,TF>::operator[]( size_t index ) noexcept
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,PF,TF>::ConstReference
   CustomVector<Type,AF,PF,TF>::operator[]( size_t index ) const noexcept
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,PF,TF>::Reference
   CustomVector<Type,AF,PF,TF>::at( size_t index )
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,PF,TF>::ConstReference
   CustomVector<Type,AF,PF,TF>::at( size_t index ) const
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
// This function returns a pointer to the internal storage of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,PF,TF>::Pointer
   CustomVector<Type,AF,PF,TF>::data() noexcept
{
   return v_.get();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the vector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,PF,TF>::ConstPointer
   CustomVector<Type,AF,PF,TF>::data() const noexcept
{
   return v_.get();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the custom vector.
//
// \return Iterator to the first element of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,PF,TF>::Iterator
   CustomVector<Type,AF,PF,TF>::begin() noexcept
{
   return Iterator( v_.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the custom vector.
//
// \return Iterator to the first element of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,PF,TF>::ConstIterator
   CustomVector<Type,AF,PF,TF>::begin() const noexcept
{
   return ConstIterator( v_.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the custom vector.
//
// \return Iterator to the first element of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,PF,TF>::ConstIterator
   CustomVector<Type,AF,PF,TF>::cbegin() const noexcept
{
   return ConstIterator( v_.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the custom vector.
//
// \return Iterator just past the last element of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,PF,TF>::Iterator
   CustomVector<Type,AF,PF,TF>::end() noexcept
{
   return Iterator( v_.get() + size_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the custom vector.
//
// \return Iterator just past the last element of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,PF,TF>::ConstIterator
   CustomVector<Type,AF,PF,TF>::end() const noexcept
{
   return ConstIterator( v_.get() + size_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the custom vector.
//
// \return Iterator just past the last element of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,PF,TF>::ConstIterator
   CustomVector<Type,AF,PF,TF>::cend() const noexcept
{
   return ConstIterator( v_.get() + size_ );
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,PF,TF>& CustomVector<Type,AF,PF,TF>::operator=( const Type& rhs )
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
// \exception std::invalid_argument Invalid assignment to custom vector.
//
// This assignment operator offers the option to directly assign to all elements of the vector
// by means of an initializer list:

   \code
   using blaze::CustomVector;
   using blaze::unaliged;
   using blaze::unpadded;

   const int array[4] = { 1, 2, 3, 4 };

   CustomVector<double,unaligned,unpadded> v( array, 4UL );
   v = { 5, 6, 7 };
   \endcode

// The vector elements are assigned the values from the given initializer list. Missing values
// are reset to their default state. Note that in case the size of the initializer list exceeds
// the size of the vector, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,PF,TF>& CustomVector<Type,AF,PF,TF>::operator=( initializer_list<Type> list )
{
   if( list.size() > size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to custom vector" );
   }

   std::fill( std::copy( list.begin(), list.end(), v_.get() ), v_.get()+size_, Type() );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array assignment to all vector elements.
//
// \param array N-dimensional array for the assignment.
// \return Reference to the assigned vector.
// \exception std::invalid_argument Invalid array size.
//
// This assignment operator offers the option to directly set all elements of the vector. The
// following example demonstrates this by means of an unaligned, unpadded custom vector:

   \code
   using blaze::CustomVector;
   using blaze::unaliged;
   using blaze::unpadded;

   const int array[4] = { 1, 2, 3, 4 };
   const int init[4]  = { 5, 6, 7 };

   CustomVector<double,unaligned,unpadded> v( array, 4UL );
   v = init;
   \endcode

// The vector is assigned the values from the given array. Missing values are initialized with
// default values (as e.g. the fourth element in the example). Note that the size of the array
// must match the size of the custom vector. Otherwise a \a std::invalid_argument exception is
// thrown. Also note that after the assignment \a array will have the same entries as \a init.
*/
template< typename Type   // Data type of the vector
        , bool AF         // Alignment flag
        , bool PF         // Padding flag
        , bool TF >       // Transpose flag
template< typename Other  // Data type of the initialization array
        , size_t N >      // Dimension of the initialization array
inline CustomVector<Type,AF,PF,TF>& CustomVector<Type,AF,PF,TF>::operator=( const Other (&array)[N] )
{
   if( size_ != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array size" );
   }

   for( size_t i=0UL; i<N; ++i )
      v_[i] = array[i];

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Copy assignment operator for CustomVector.
//
// \param rhs Vector to be copied.
// \return Reference to the assigned vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// The vector is initialized as a copy of the given vector. In case the current sizes of the two
// vectors don't match, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,PF,TF>& CustomVector<Type,AF,PF,TF>::operator=( const CustomVector& rhs )
{
   if( rhs.size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   smpAssign( *this, ~rhs );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Move assignment operator for CustomVector.
//
// \param rhs The vector to be moved into this instance.
// \return Reference to the assigned vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,PF,TF>&
   CustomVector<Type,AF,PF,TF>::operator=( CustomVector&& rhs ) noexcept
{
   size_ = rhs.size_;
   v_    = std::move( rhs.v_ );

   rhs.size_ = 0UL;

   BLAZE_INTERNAL_ASSERT( rhs.data() == nullptr, "Invalid data reference detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different vectors.
//
// \param rhs Vector to be copied.
// \return Reference to the assigned vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// The vector is initialized as a copy of the given vector. In case the current sizes of the two
// vectors don't match, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CustomVector<Type,AF,PF,TF>& CustomVector<Type,AF,PF,TF>::operator=( const Vector<VT,TF>& rhs )
{
   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_<VT> tmp( ~rhs );
      smpAssign( *this, tmp );
   }
   else {
      if( IsSparseVector<VT>::value )
         reset();
      smpAssign( *this, ~rhs );
   }

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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CustomVector<Type,AF,PF,TF>& CustomVector<Type,AF,PF,TF>::operator+=( const Vector<VT,TF>& rhs )
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CustomVector<Type,AF,PF,TF>& CustomVector<Type,AF,PF,TF>::operator-=( const Vector<VT,TF>& rhs )
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CustomVector<Type,AF,PF,TF>& CustomVector<Type,AF,PF,TF>::operator*=( const Vector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( VT, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT> );

   typedef MultTrait_< ResultType, ResultType_<VT> >  MultType;

   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( MultType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MultType );

   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( IsSparseVector<VT>::value || (~rhs).canAlias( this ) ) {
      const MultType tmp( *this * (~rhs) );
      this->operator=( tmp );
   }
   else {
      smpMultAssign( *this, ~rhs );
   }

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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CustomVector<Type,AF,PF,TF>&
   CustomVector<Type,AF,PF,TF>::operator/=( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( VT, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT> );

   typedef DivTrait_< ResultType, ResultType_<VT> >  DivType;

   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( DivType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( DivType );

   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const DivType tmp( *this / (~rhs) );
      this->operator=( tmp );
   }
   else {
      smpDivAssign( *this, ~rhs );
   }

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
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, CustomVector<Type,AF,PF,TF> >&
   CustomVector<Type,AF,PF,TF>::operator*=( Other rhs )
{
   smpAssign( *this, (*this) * rhs );
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
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, CustomVector<Type,AF,PF,TF> >&
   CustomVector<Type,AF,PF,TF>::operator/=( Other rhs )
{
   BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

   smpAssign( *this, (*this) / rhs );
   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the size/dimension of the vector.
//
// \return The size of the vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline size_t CustomVector<Type,AF,PF,TF>::size() const noexcept
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline size_t CustomVector<Type,AF,PF,TF>::capacity() const noexcept
{
   return size_;
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline size_t CustomVector<Type,AF,PF,TF>::nonZeros() const
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline void CustomVector<Type,AF,PF,TF>::reset()
{
   using blaze::clear;
   for( size_t i=0UL; i<size_; ++i )
      clear( v_[i] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the vector to its default state.
//
// \return void
//
// This function clears the vector to its default state. In case the vector has been passed the
// responsibility to manage the given array, it disposes the resource via the specified deleter.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline void CustomVector<Type,AF,PF,TF>::clear()
{
   size_ = 0UL;
   v_.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the vector by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the vector scaling.
// \return Reference to the vector.
*/
template< typename Type     // Data type of the vector
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the scalar value
inline CustomVector<Type,AF,PF,TF>& CustomVector<Type,AF,PF,TF>::scale( const Other& scalar )
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline void CustomVector<Type,AF,PF,TF>::swap( CustomVector& v ) noexcept
{
   using std::swap;

   swap( size_, v.size_ );
   swap( v_, v.v_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  RESOURCE MANAGEMENT FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Resets the custom vector and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the vector.
// \param n The number of array elements to be used by the custom vector.
// \return void
// \exception std::invalid_argument Invalid setup of custom vector.
//
// This function resets the custom vector to the given array of elements of size \a n. The
// function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This function is \b NOT available for padded custom vectors!
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom vector referencing the array goes out of scope.
// \note The custom vector does NOT take responsibility for the new array of elements!
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline void CustomVector<Type,AF,PF,TF>::reset( Type* ptr, size_t n )
{
   CustomVector tmp( ptr, n );
   swap( tmp );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resets the custom vector and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the vector.
// \param n The number of array elements to be used by the custom vector.
// \param nn The maximum size of the given array.
// \return void
// \exception std::invalid_argument Invalid setup of custom vector.
//
// This function resets the custom vector to the given array of elements of size \a n and
// capacity \a nn. The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified capacity \a nn is insufficient for the given data type \a Type and
//    the available instruction set.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This function is \a NOT available for unpadded custom vectors!
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom vector referencing the array goes out of scope.
// \note The custom vector does NOT take responsibility for the new array of elements!
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline void CustomVector<Type,AF,PF,TF>::reset( Type* ptr, size_t n, size_t nn )
{
   BLAZE_STATIC_ASSERT( PF == padded );

   UNUSED_PARAMETER( ptr, n, nn );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resets the custom vector and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the vector.
// \param n The number of array elements to be used by the custom vector.
// \param d The deleter to destroy the array of elements.
// \return void
// \exception std::invalid_argument Invalid setup of custom vector.
//
// This function resets the custom vector to the given array of elements of size \a n. The
// function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This function is \b NOT available for padded custom vectors!
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom vector referencing the array goes out of scope.
*/
template< typename Type     // Data type of the vector
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool TF >         // Transpose flag
template< typename Deleter  // Type of the custom deleter
        , typename >        // Type restriction on the custom deleter
inline void CustomVector<Type,AF,PF,TF>::reset( Type* ptr, size_t n, Deleter d )
{
   CustomVector tmp( ptr, n, d );
   swap( tmp );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resets the custom vector and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the vector.
// \param n The number of array elements to be used by the custom vector.
// \param nn The maximum size of the given array.
// \param d The deleter to destroy the array of elements.
// \return void
// \exception std::invalid_argument Invalid setup of custom vector.
//
// This function resets the custom vector to the given array of elements of size \a n and
// capacity \a nn. The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified capacity \a nn is insufficient for the given data type \a Type and
//    the available instruction set.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This function is \a NOT available for unpadded custom vectors!
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom vector referencing the array goes out of scope.
*/
template< typename Type       // Data type of the vector
        , bool AF             // Alignment flag
        , bool PF             // Padding flag
        , bool TF >           // Transpose flag
template< typename Deleter >  // Type of the custom deleter
inline void CustomVector<Type,AF,PF,TF>::reset( Type* ptr, size_t n, size_t nn, Deleter d )
{
   BLAZE_STATIC_ASSERT( PF == padded );

   UNUSED_PARAMETER( ptr, n, nn, d );
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
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool CustomVector<Type,AF,PF,TF>::canAlias( const Other* alias ) const noexcept
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
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool CustomVector<Type,AF,PF,TF>::isAliased( const Other* alias ) const noexcept
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline bool CustomVector<Type,AF,PF,TF>::isAligned() const noexcept
{
   return ( AF || checkAlignment( v_.get() ) );
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline bool CustomVector<Type,AF,PF,TF>::canSMPAssign() const noexcept
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE typename CustomVector<Type,AF,PF,TF>::SIMDType
   CustomVector<Type,AF,PF,TF>::load( size_t index ) const noexcept
{
   if( AF )
      return loada( index );
   else
      return loadu( index );
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE typename CustomVector<Type,AF,PF,TF>::SIMDType
   CustomVector<Type,AF,PF,TF>::loada( size_t index ) const noexcept
{
   using blaze::loada;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( !AF || index % SIMDSIZE == 0UL, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_.get()+index ), "Invalid vector access index" );

   return loada( v_.get()+index );
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE typename CustomVector<Type,AF,PF,TF>::SIMDType
   CustomVector<Type,AF,PF,TF>::loadu( size_t index ) const noexcept
{
   using blaze::loadu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index< size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size_, "Invalid vector access index" );

   return loadu( v_.get()+index );
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void CustomVector<Type,AF,PF,TF>::store( size_t index, const SIMDType& value ) noexcept
{
   if( AF )
      storea( index, value );
   else
      storeu( index, value );
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void CustomVector<Type,AF,PF,TF>::storea( size_t index, const SIMDType& value ) noexcept
{
   using blaze::storea;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( !AF || index % SIMDSIZE == 0UL, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_.get()+index ), "Invalid vector access index" );

   storea( v_.get()+index, value );
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void CustomVector<Type,AF,PF,TF>::storeu( size_t index, const SIMDType& value ) noexcept
{
   using blaze::storeu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size_, "Invalid vector access index" );

   storeu( v_.get()+index, value );
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void CustomVector<Type,AF,PF,TF>::stream( size_t index, const SIMDType& value ) noexcept
{
   using blaze::stream;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( !AF || index % SIMDSIZE == 0UL, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_.get()+index ), "Invalid vector access index" );

   stream( v_.get()+index, value );
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename CustomVector<Type,AF,PF,TF>::BLAZE_TEMPLATE VectorizedAssign<VT> >
   CustomVector<Type,AF,PF,TF>::assign( const DenseVector<VT,TF>& rhs )
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename CustomVector<Type,AF,PF,TF>::BLAZE_TEMPLATE VectorizedAssign<VT> >
   CustomVector<Type,AF,PF,TF>::assign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

   if( AF && useStreaming && size_ > ( cacheSize/( sizeof(Type) * 3UL ) ) && !(~rhs).isAliased( this ) )
   {
      size_t i( 0UL );

      for( ; i<ipos; i+=SIMDSIZE ) {
         stream( i, (~rhs).load(i) );
      }
      for( ; i<size_; ++i ) {
         v_[i] = (~rhs)[i];
      }
   }
   else
   {
      const size_t i4way( size_ & size_t(-SIMDSIZE*4) );
      BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE*4UL) ) ) == i4way, "Invalid end calculation" );
      BLAZE_INTERNAL_ASSERT( i4way <= ipos, "Invalid end calculation" );

      size_t i( 0UL );
      ConstIterator_<VT> it( (~rhs).begin() );

      for( ; i<i4way; i+=SIMDSIZE*4UL ) {
         store( i             , it.load() ); it += SIMDSIZE;
         store( i+SIMDSIZE    , it.load() ); it += SIMDSIZE;
         store( i+SIMDSIZE*2UL, it.load() ); it += SIMDSIZE;
         store( i+SIMDSIZE*3UL, it.load() ); it += SIMDSIZE;
      }
      for( ; i<ipos; i+=SIMDSIZE, it+=SIMDSIZE ) {
         store( i, it.load() );
      }
      for( ; i<size_; ++i, ++it ) {
         v_[i] = *it;
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CustomVector<Type,AF,PF,TF>::assign( const SparseVector<VT,TF>& rhs )
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename CustomVector<Type,AF,PF,TF>::BLAZE_TEMPLATE VectorizedAddAssign<VT> >
   CustomVector<Type,AF,PF,TF>::addAssign( const DenseVector<VT,TF>& rhs )
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename CustomVector<Type,AF,PF,TF>::BLAZE_TEMPLATE VectorizedAddAssign<VT> >
   CustomVector<Type,AF,PF,TF>::addAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

   const size_t i4way( size_ & size_t(-SIMDSIZE*4) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE*4UL) ) ) == i4way, "Invalid end calculation" );
   BLAZE_INTERNAL_ASSERT( i4way <= ipos, "Invalid end calculation" );

   size_t i( 0UL );
   ConstIterator_<VT> it( (~rhs).begin() );

   for( ; i<i4way; i+=SIMDSIZE*4UL ) {
      store( i             , load(i             ) + it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE    , load(i+SIMDSIZE    ) + it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*2UL, load(i+SIMDSIZE*2UL) + it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*3UL, load(i+SIMDSIZE*3UL) + it.load() ); it += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE, it+=SIMDSIZE ) {
      store( i, load(i) + it.load() );
   }
   for( ; i<size_; ++i, ++it ) {
      v_[i] += *it;
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CustomVector<Type,AF,PF,TF>::addAssign( const SparseVector<VT,TF>& rhs )
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename CustomVector<Type,AF,PF,TF>::BLAZE_TEMPLATE VectorizedSubAssign<VT> >
   CustomVector<Type,AF,PF,TF>::subAssign( const DenseVector<VT,TF>& rhs )
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename CustomVector<Type,AF,PF,TF>::BLAZE_TEMPLATE VectorizedSubAssign<VT> >
   CustomVector<Type,AF,PF,TF>::subAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

   const size_t i4way( size_ & size_t(-SIMDSIZE*4) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE*4UL) ) ) == i4way, "Invalid end calculation" );
   BLAZE_INTERNAL_ASSERT( i4way <= ipos, "Invalid end calculation" );

   size_t i( 0UL );
   ConstIterator_<VT> it( (~rhs).begin() );

   for( ; i<i4way; i+=SIMDSIZE*4UL ) {
      store( i             , load(i             ) - it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE    , load(i+SIMDSIZE    ) - it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*2UL, load(i+SIMDSIZE*2UL) - it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*3UL, load(i+SIMDSIZE*3UL) - it.load() ); it += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE, it+=SIMDSIZE ) {
      store( i, load(i) - it.load() );
   }
   for( ; i<size_; ++i, ++it ) {
      v_[i] -= *it;
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CustomVector<Type,AF,PF,TF>::subAssign( const SparseVector<VT,TF>& rhs )
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename CustomVector<Type,AF,PF,TF>::BLAZE_TEMPLATE VectorizedMultAssign<VT> >
   CustomVector<Type,AF,PF,TF>::multAssign( const DenseVector<VT,TF>& rhs )
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename CustomVector<Type,AF,PF,TF>::BLAZE_TEMPLATE VectorizedMultAssign<VT> >
   CustomVector<Type,AF,PF,TF>::multAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

   const size_t i4way( size_ & size_t(-SIMDSIZE*4) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE*4UL) ) ) == i4way, "Invalid end calculation" );
   BLAZE_INTERNAL_ASSERT( i4way <= ipos, "Invalid end calculation" );

   size_t i( 0UL );
   ConstIterator_<VT> it( (~rhs).begin() );

   for( ; i<i4way; i+=SIMDSIZE*4UL ) {
      store( i             , load(i             ) * it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE    , load(i+SIMDSIZE    ) * it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*2UL, load(i+SIMDSIZE*2UL) * it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*3UL, load(i+SIMDSIZE*3UL) * it.load() ); it += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE, it+=SIMDSIZE ) {
      store( i, load(i) * it.load() );
   }
   for( ; i<size_; ++i, ++it ) {
      v_[i] *= *it;
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CustomVector<Type,AF,PF,TF>::multAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const DynamicVector<Type,TF> tmp( serial( *this ) );

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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename CustomVector<Type,AF,PF,TF>::BLAZE_TEMPLATE VectorizedDivAssign<VT> >
   CustomVector<Type,AF,PF,TF>::divAssign( const DenseVector<VT,TF>& rhs )
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
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename CustomVector<Type,AF,PF,TF>::BLAZE_TEMPLATE VectorizedDivAssign<VT> >
   CustomVector<Type,AF,PF,TF>::divAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

   const size_t i4way( size_ & size_t(-SIMDSIZE*4) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE*4UL) ) ) == i4way, "Invalid end calculation" );
   BLAZE_INTERNAL_ASSERT( i4way <= ipos, "Invalid end calculation" );

   size_t i( 0UL );
   ConstIterator_<VT> it( (~rhs).begin() );

   for( ; i<i4way; i+=SIMDSIZE*4UL ) {
      store( i             , load(i             ) / it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE    , load(i+SIMDSIZE    ) / it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*2UL, load(i+SIMDSIZE*2UL) / it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*3UL, load(i+SIMDSIZE*3UL) / it.load() ); it += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE, it+=SIMDSIZE ) {
      store( i, load(i) / it.load() );
   }
   for( ; i<size_; ++i, ++it ) {
      v_[i] /= *it;
   }
}
//*************************************************************************************************








//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR PADDED VECTORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of CustomVector for padded vectors.
// \ingroup custom_vector
//
// This specialization of CustomVector adapts the class template to the requirements of padded
// vectors.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
class CustomVector<Type,AF,padded,TF>
   : public DenseVector< CustomVector<Type,AF,padded,TF>, TF >
{
 public:
   //**Type definitions****************************************************************************
   typedef CustomVector<Type,AF,padded,TF>  This;           //!< Type of this CustomVector instance.
   typedef DynamicVector<Type,TF>           ResultType;     //!< Result type for expression template evaluations.
   typedef DynamicVector<Type,!TF>          TransposeType;  //!< Transpose type for expression template evaluations.
   typedef Type                             ElementType;    //!< Type of the vector elements.
   typedef SIMDTrait_<ElementType>          SIMDType;       //!< SIMD type of the vector elements.
   typedef const Type&                      ReturnType;     //!< Return type for expression template evaluations
   typedef const CustomVector&              CompositeType;  //!< Data type for composite expression templates.

   typedef Type&        Reference;       //!< Reference to a non-constant vector value.
   typedef const Type&  ConstReference;  //!< Reference to a constant vector value.
   typedef Type*        Pointer;         //!< Pointer to a non-constant vector value.
   typedef const Type*  ConstPointer;    //!< Pointer to a constant vector value.

   typedef DenseIterator<Type,AF>        Iterator;       //!< Iterator over non-constant elements.
   typedef DenseIterator<const Type,AF>  ConstIterator;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a CustomVector with different data/element type.
   */
   template< typename ET >  // Data type of the other vector
   struct Rebind {
      typedef CustomVector<ET,AF,padded,TF>  Other;  //!< The type of the other CustomVector.
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
   enum : bool { smpAssignable = !IsSMPAssignable<Type>::value };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline CustomVector();
   explicit inline CustomVector( Type* ptr, size_t n, size_t nn );

   template< typename Deleter >
   explicit inline CustomVector( Type* ptr, size_t n, size_t nn, Deleter d );

   inline CustomVector( const CustomVector& v );
   inline CustomVector( CustomVector&& v ) noexcept;
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
   inline CustomVector& operator=( const Type& rhs );
   inline CustomVector& operator=( initializer_list<Type> list );

   template< typename Other, size_t N >
   inline CustomVector& operator=( const Other (&array)[N] );

   inline CustomVector& operator=( const CustomVector& rhs );
   inline CustomVector& operator=( CustomVector&& rhs ) noexcept;

   template< typename VT > inline CustomVector& operator= ( const Vector<VT,TF>& rhs );
   template< typename VT > inline CustomVector& operator+=( const Vector<VT,TF>& rhs );
   template< typename VT > inline CustomVector& operator-=( const Vector<VT,TF>& rhs );
   template< typename VT > inline CustomVector& operator*=( const Vector<VT,TF>& rhs );
   template< typename VT > inline CustomVector& operator/=( const DenseVector<VT,TF>& rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, CustomVector >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, CustomVector >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                                inline size_t        size() const noexcept;
                                inline size_t        capacity() const noexcept;
                                inline size_t        nonZeros() const;
                                inline void          reset();
                                inline void          clear();
   template< typename Other >   inline CustomVector& scale( const Other& scalar );
                                inline void          swap( CustomVector& v ) noexcept;
   //@}
   //**********************************************************************************************

   //**Resource management functions***************************************************************
   /*!\name Resource management functions */
   //@{
                                inline void reset( Type* ptr, size_t n, size_t nn );
   template< typename Deleter > inline void reset( Type* ptr, size_t n, size_t nn, Deleter d );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT >
   struct VectorizedAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<VT> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT >
   struct VectorizedAddAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<VT> >::value &&
                            HasSIMDAdd< Type, ElementType_<VT> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT >
   struct VectorizedSubAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<VT> >::value &&
                            HasSIMDSub< Type, ElementType_<VT> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT >
   struct VectorizedMultAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<VT> >::value &&
                            HasSIMDMult< Type, ElementType_<VT> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename VT >
   struct VectorizedDivAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && VT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<VT> >::value &&
                            HasSIMDDiv< Type, ElementType_<VT> >::value };
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
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t size_;                  //!< The size/dimension of the custom vector.
   size_t capacity_;              //!< The maximum capacity of the custom vector.
   boost::shared_array<Type> v_;  //!< The custom array of elements.
                                  /*!< Access to the array of elements is gained via the
                                       subscript operator. The order of the elements is
                                       \f[\left(\begin{array}{*{5}{c}}
                                       0 & 1 & 2 & \cdots & N-1 \\
                                       \end{array}\right)\f] */
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE  ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST         ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE      ( Type );
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
/*!\brief The default constructor for CustomVector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,padded,TF>::CustomVector()
   : size_    ( 0UL )  // The size/dimension of the vector
   , capacity_( 0UL )  // The maximum capacity of the vector
   , v_       (     )  // The custom array of elements
{}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a padded custom vector of size \a n and capacity \a nn.
//
// \param ptr The array of elements to be used by the vector.
// \param n The number of array elements to be used by the custom vector.
// \param nn The maximum size of the given array.
// \exception std::invalid_argument Invalid setup of custom vector.
//
// This constructor creates a padded custom vector of size \a n and capacity \a nn. The
// construction of the vector fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified capacity \a nn is insufficient for the given data type \a Type and
//    the available instruction set.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note The custom vector does NOT take responsibility for the given array of elements!
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,padded,TF>::CustomVector( Type* ptr, size_t n, size_t nn )
   : size_    ( n  )  // The size/dimension of the vector
   , capacity_( nn )  // The maximum capacity of the vector
   , v_       (    )  // The custom array of elements
{
   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && !checkAlignment( ptr ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   if( IsVectorizable<Type>::value && capacity_ < nextMultiple<size_t>( size_, SIMDSIZE ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Insufficient capacity for padded vector" );
   }

   v_.reset( ptr, NoDelete() );

   if( IsVectorizable<Type>::value ) {
      for( size_t i=size_; i<capacity_; ++i )
         v_[i] = Type();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a padded custom vector of size \a n and capacity \a nn.
//
// \param ptr The array of elements to be used by the vector.
// \param n The number of array elements to be used by the custom vector.
// \param nn The maximum size of the given array.
// \param d The deleter to destroy the array of elements.
// \exception std::invalid_argument Invalid setup of custom vector.
//
// This constructor creates a padded custom vector of size \a n and capacity \a nn. The
// construction of the vector fails if ...
//
//  - ... the passes pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified capacity \a nn is insufficient for the given data type \a Type and
//    the available instruction set.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
*/
template< typename Type       // Data type of the vector
        , bool AF             // Alignment flag
        , bool TF >           // Transpose flag
template< typename Deleter >  // Type of the custom deleter
inline CustomVector<Type,AF,padded,TF>::CustomVector( Type* ptr, size_t n, size_t nn, Deleter d )
   : size_    ( n  )  // The custom array of elements
   , capacity_( nn )  // The maximum capacity of the vector
   , v_       (    )  // The custom array of elements
{
   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && !checkAlignment( ptr ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   if( IsVectorizable<Type>::value && capacity_ < nextMultiple<size_t>( size_, SIMDSIZE ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Insufficient capacity for padded vector" );
   }

   v_.reset( ptr, d );

   if( IsVectorizable<Type>::value ) {
      for( size_t i=size_; i<capacity_; ++i )
         v_[i] = Type();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The copy constructor for CustomVector.
//
// \param v Vector to be copied.
//
// The copy constructor initializes the custom vector as an exact copy of the given custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,padded,TF>::CustomVector( const CustomVector& v )
   : size_    ( v.size_     )  // The size/dimension of the vector
   , capacity_( v.capacity_ )  // The maximum capacity of the vector
   , v_       ( v.v_        )  // The custom array of elements
{}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The move constructor for CustomVector.
//
// \param v The vector to be moved into this instance.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,padded,TF>::CustomVector( CustomVector&& v ) noexcept
   : size_    ( v.size_     )        // The size/dimension of the vector
   , capacity_( v.capacity_ )        // The maximum capacity of the vector
   , v_       ( std::move( v.v_ ) )  // The custom array of elements
{
   v.size_     = 0UL;
   v.capacity_ = 0UL;

   BLAZE_INTERNAL_ASSERT( v.data() == nullptr, "Invalid data reference detected" );
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
/*!\brief Subscript operator for the direct access to the vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,padded,TF>::Reference
   CustomVector<Type,AF,padded,TF>::operator[]( size_t index ) noexcept
{
   BLAZE_USER_ASSERT( index < size_, "Invalid vector access index" );
   return v_[index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subscript operator for the direct access to the vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,padded,TF>::ConstReference
   CustomVector<Type,AF,padded,TF>::operator[]( size_t index ) const noexcept
{
   BLAZE_USER_ASSERT( index < size_, "Invalid vector access index" );
   return v_[index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,padded,TF>::Reference
   CustomVector<Type,AF,padded,TF>::at( size_t index )
{
   if( index >= size_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid vector access index" );
   }
   return (*this)[index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,padded,TF>::ConstReference
   CustomVector<Type,AF,padded,TF>::at( size_t index ) const
{
   if( index >= size_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid vector access index" );
   }
   return (*this)[index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the vector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,padded,TF>::Pointer
   CustomVector<Type,AF,padded,TF>::data() noexcept
{
   return v_.get();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the vector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,padded,TF>::ConstPointer
   CustomVector<Type,AF,padded,TF>::data() const noexcept
{
   return v_.get();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the custom vector.
//
// \return Iterator to the first element of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,padded,TF>::Iterator
   CustomVector<Type,AF,padded,TF>::begin() noexcept
{
   return Iterator( v_.get() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the custom vector.
//
// \return Iterator to the first element of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,padded,TF>::ConstIterator
   CustomVector<Type,AF,padded,TF>::begin() const noexcept
{
   return ConstIterator( v_.get() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the custom vector.
//
// \return Iterator to the first element of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,padded,TF>::ConstIterator
   CustomVector<Type,AF,padded,TF>::cbegin() const noexcept
{
   return ConstIterator( v_.get() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the custom vector.
//
// \return Iterator just past the last element of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,padded,TF>::Iterator
   CustomVector<Type,AF,padded,TF>::end() noexcept
{
   return Iterator( v_.get() + size_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the custom vector.
//
// \return Iterator just past the last element of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,padded,TF>::ConstIterator
   CustomVector<Type,AF,padded,TF>::end() const noexcept
{
   return ConstIterator( v_.get() + size_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the custom vector.
//
// \return Iterator just past the last element of the custom vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline typename CustomVector<Type,AF,padded,TF>::ConstIterator
   CustomVector<Type,AF,padded,TF>::cend() const noexcept
{
   return ConstIterator( v_.get() + size_ );
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
/*!\brief Homogenous assignment to all vector elements.
//
// \param rhs Scalar value to be assigned to all vector elements.
// \return Reference to the assigned vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,padded,TF>&
   CustomVector<Type,AF,padded,TF>::operator=( const Type& rhs )
{
   for( size_t i=0UL; i<size_; ++i )
      v_[i] = rhs;
   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all vector elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to custom vector.
//
// This assignment operator offers the option to directly assign to all elements of the vector
// by means of an initializer list:

   \code
   using blaze::CustomVector;
   using blaze::unaliged;
   using blaze::padded;

   const int array[4] = { 1, 2, 3, 4 };

   CustomVector<double,unaligned,padded> v( array, 4UL, 8UL );
   v = { 5, 6, 7 };
   \endcode

// The vector elements are assigned the values from the given initializer list. Missing values
// are reset to their default state. Note that in case the size of the initializer list exceeds
// the size of the vector, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,padded,TF>&
   CustomVector<Type,AF,padded,TF>::operator=( initializer_list<Type> list )
{
   if( list.size() > size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to custom vector" );
   }

   std::fill( std::copy( list.begin(), list.end(), v_.get() ), v_.get()+capacity_, Type() );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Array assignment to all vector elements.
//
// \param array N-dimensional array for the assignment.
// \return Reference to the assigned vector.
// \exception std::invalid_argument Invalid array size.
//
// This assignment operator offers the option to directly set all elements of the vector. The
// following example demonstrates this by means of an unaligned, padded custom vector:

   \code
   using blaze::CustomVector;
   using blaze::unaliged;
   using blaze::padded;

   const int array[8] = { 1, 2, 3, 4, 0, 0, 0, 0 };
   const int init[4]  = { 5, 6, 7 };

   CustomVector<double,unaligned,padded> v( array, 4UL, 8UL );
   v = init;
   \endcode

// The vector is assigned the values from the given array. Missing values are initialized with
// default values (as e.g. the fourth element in the example). Note that the size of the array
// must match the size of the custom vector. Otherwise a \a std::invalid_argument exception is
// thrown. Also note that after the assignment \a array will have the same entries as \a init.
*/
template< typename Type   // Data type of the vector
        , bool AF         // Alignment flag
        , bool TF >       // Transpose flag
template< typename Other  // Data type of the initialization array
        , size_t N >      // Dimension of the initialization array
inline CustomVector<Type,AF,padded,TF>&
   CustomVector<Type,AF,padded,TF>::operator=( const Other (&array)[N] )
{
   if( size_ != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array size" );
   }

   for( size_t i=0UL; i<N; ++i )
      v_[i] = array[i];

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for CustomVector.
//
// \param rhs Vector to be copied.
// \return Reference to the assigned vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// The vector is initialized as a copy of the given vector. In case the current sizes of the two
// vectors don't match, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,padded,TF>&
   CustomVector<Type,AF,padded,TF>::operator=( const CustomVector& rhs )
{
   if( rhs.size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   smpAssign( *this, ~rhs );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Move assignment operator for CustomVector.
//
// \param rhs The vector to be moved into this instance.
// \return Reference to the assigned vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline CustomVector<Type,AF,padded,TF>&
   CustomVector<Type,AF,padded,TF>::operator=( CustomVector&& rhs ) noexcept
{
   size_     = rhs.size_;
   capacity_ = rhs.capacity_;
   v_        = std::move( rhs.v_ );

   rhs.size_     = 0UL;
   rhs.capacity_ = 0UL;

   BLAZE_INTERNAL_ASSERT( rhs.data() == nullptr, "Invalid data reference detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different vectors.
//
// \param rhs Vector to be copied.
// \return Reference to the assigned vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// The vector is initialized as a copy of the given vector. In case the current sizes of the two
// vectors don't match, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CustomVector<Type,AF,padded,TF>&
   CustomVector<Type,AF,padded,TF>::operator=( const Vector<VT,TF>& rhs )
{
   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_<VT> tmp( ~rhs );
      smpAssign( *this, tmp );
   }
   else {
      if( IsSparseVector<VT>::value )
         reset();
      smpAssign( *this, ~rhs );
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CustomVector<Type,AF,padded,TF>&
   CustomVector<Type,AF,padded,TF>::operator+=( const Vector<VT,TF>& rhs )
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

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CustomVector<Type,AF,padded,TF>&
   CustomVector<Type,AF,padded,TF>::operator-=( const Vector<VT,TF>& rhs )
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

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CustomVector<Type,AF,padded,TF>&
   CustomVector<Type,AF,padded,TF>::operator*=( const Vector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( VT, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT> );

   typedef MultTrait_< ResultType, ResultType_<VT> >  MultType;

   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( MultType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MultType );

   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( IsSparseVector<VT>::value || (~rhs).canAlias( this ) ) {
      const MultType tmp( *this * (~rhs) );
      this->operator=( tmp );
   }
   else {
      smpMultAssign( *this, ~rhs );
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CustomVector<Type,AF,padded,TF>&
   CustomVector<Type,AF,padded,TF>::operator/=( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( VT, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_<VT> );

   typedef DivTrait_< ResultType, ResultType_<VT> >  DivType;

   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( DivType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( DivType );

   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const DivType tmp( *this / (~rhs) );
      this->operator=( tmp );
   }
   else {
      smpDivAssign( *this, ~rhs );
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication between a vector and
//        a scalar value (\f$ \vec{a}*=s \f$).
//
// \param rhs The right-hand side scalar value for the multiplication.
// \return Reference to the vector.
*/
template< typename Type     // Data type of the vector
        , bool AF           // Alignment flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, CustomVector<Type,AF,padded,TF> >&
   CustomVector<Type,AF,padded,TF>::operator*=( Other rhs )
{
   smpAssign( *this, (*this) * rhs );
   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division assignment operator for the division of a vector by a scalar value
//        (\f$ \vec{a}/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the vector.
//
// \note A division by zero is only checked by an user assert.
*/
template< typename Type     // Data type of the vector
        , bool AF           // Alignment flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, CustomVector<Type,AF,padded,TF> >&
   CustomVector<Type,AF,padded,TF>::operator/=( Other rhs )
{
   BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

   smpAssign( *this, (*this) / rhs );
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
/*!\brief Returns the size/dimension of the vector.
//
// \return The size of the vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline size_t CustomVector<Type,AF,padded,TF>::size() const noexcept
{
   return size_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the vector.
//
// \return The capacity of the vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline size_t CustomVector<Type,AF,padded,TF>::capacity() const noexcept
{
   return capacity_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the vector.
//
// \return The number of non-zero elements in the vector.
//
// Note that the number of non-zero elements is always less than or equal to the current size
// of the vector.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline size_t CustomVector<Type,AF,padded,TF>::nonZeros() const
{
   size_t nonzeros( 0 );

   for( size_t i=0UL; i<size_; ++i ) {
      if( !isDefault( v_[i] ) )
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline void CustomVector<Type,AF,padded,TF>::reset()
{
   using blaze::clear;
   for( size_t i=0UL; i<size_; ++i )
      clear( v_[i] );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the vector to its default state.
//
// \return void
//
// This function clears the vector to its default state. In case the vector has been passed the
// responsibility to manage the given array, it disposes the resource via the specified deleter.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline void CustomVector<Type,AF,padded,TF>::clear()
{
   size_     = 0UL;
   capacity_ = 0UL;
   v_.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the vector by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the vector scaling.
// \return Reference to the vector.
*/
template< typename Type     // Data type of the vector
        , bool AF           // Alignment flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the scalar value
inline CustomVector<Type,AF,padded,TF>&
   CustomVector<Type,AF,padded,TF>::scale( const Other& scalar )
{
   for( size_t i=0UL; i<size_; ++i )
      v_[i] *= scalar;
   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Swapping the contents of two vectors.
//
// \param v The vector to be swapped.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline void CustomVector<Type,AF,padded,TF>::swap( CustomVector& v ) noexcept
{
   using std::swap;

   swap( size_, v.size_ );
   swap( capacity_, v.capacity_ );
   swap( v_, v.v_ );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RESOURCE MANAGEMENT FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resets the custom vector and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the vector.
// \param n The number of array elements to be used by the custom vector.
// \param nn The maximum size of the given array.
// \return void
// \exception std::invalid_argument Invalid setup of custom vector.
//
// This function resets the custom vector to the given array of elements of size \a n and capacity
// \a nn. The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the specified capacity \a nn is insufficient for the given data type \a Type and
//    the available instruction set.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom vector referencing the array goes out of scope.
// \note The custom vector does NOT take responsibility for the new array of elements!
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline void CustomVector<Type,AF,padded,TF>::reset( Type* ptr, size_t n, size_t nn )
{
   CustomVector tmp( ptr, n, nn );
   swap( tmp );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resets the custom vector and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the vector.
// \param n The number of array elements to be used by the custom vector.
// \param nn The maximum size of the given array.
// \param d The deleter to destroy the array of elements.
// \return void
// \exception std::invalid_argument Invalid setup of custom vector.
//
// This function resets the custom vector to the given array of elements of size \a n and capacity
// \a nn. The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the specified capacity \a nn is insufficient for the given data type \a Type and
//    the available instruction set.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom vector referencing the array goes out of scope.
*/
template< typename Type       // Data type of the vector
        , bool AF             // Alignment flag
        , bool TF >           // Transpose flag
template< typename Deleter >  // Type of the custom deleter
inline void CustomVector<Type,AF,padded,TF>::reset( Type* ptr, size_t n, size_t nn, Deleter d )
{
   CustomVector tmp( ptr, n, nn, d );
   swap( tmp );
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
        , bool AF           // Alignment flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool CustomVector<Type,AF,padded,TF>::canAlias( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF           // Alignment flag
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool CustomVector<Type,AF,padded,TF>::isAliased( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the vector is properly aligned in memory.
//
// \return \a true in case the vector is aligned, \a false if not.
//
// This function returns whether the vector is guaranteed to be properly aligned in memory, i.e.
// whether the beginning and the end of the vector are guaranteed to conform to the alignment
// restrictions of the element type \a Type.
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline bool CustomVector<Type,AF,padded,TF>::isAligned() const noexcept
{
   return ( AF || checkAlignment( v_.get() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
inline bool CustomVector<Type,AF,padded,TF>::canSMPAssign() const noexcept
{
   return ( size() > SMP_DVECASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE typename CustomVector<Type,AF,padded,TF>::SIMDType
   CustomVector<Type,AF,padded,TF>::load( size_t index ) const noexcept
{
   if( AF )
      return loada( index );
   else
      return loadu( index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE typename CustomVector<Type,AF,padded,TF>::SIMDType
   CustomVector<Type,AF,padded,TF>::loada( size_t index ) const noexcept
{
   using blaze::loada;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= capacity_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( !AF || index % SIMDSIZE == 0UL, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_.get()+index ), "Invalid vector access index" );

   return loada( v_.get()+index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE typename CustomVector<Type,AF,padded,TF>::SIMDType
   CustomVector<Type,AF,padded,TF>::loadu( size_t index ) const noexcept
{
   using blaze::loadu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= capacity_, "Invalid vector access index" );

   return loadu( v_.get()+index );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void
   CustomVector<Type,AF,padded,TF>::store( size_t index, const SIMDType& value ) noexcept
{
   if( AF )
      storea( index, value );
   else
      storeu( index, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void
   CustomVector<Type,AF,padded,TF>::storea( size_t index, const SIMDType& value ) noexcept
{
   using blaze::storea;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= capacity_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( !AF || index % SIMDSIZE == 0UL, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_.get()+index ), "Invalid vector access index" );

   storea( v_.get()+index, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void
   CustomVector<Type,AF,padded,TF>::storeu( size_t index, const SIMDType& value ) noexcept
{
   using blaze::storeu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= capacity_, "Invalid vector access index" );

   storeu( v_.get()+index, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
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
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
BLAZE_ALWAYS_INLINE void
   CustomVector<Type,AF,padded,TF>::stream( size_t index, const SIMDType& value ) noexcept
{
   using blaze::stream;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( index < size_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( index + SIMDSIZE <= capacity_, "Invalid vector access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_.get()+index ), "Invalid vector access index" );

   stream( v_.get()+index, value );
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename CustomVector<Type,AF,padded,TF>::BLAZE_TEMPLATE VectorizedAssign<VT> >
   CustomVector<Type,AF,padded,TF>::assign( const DenseVector<VT,TF>& rhs )
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename CustomVector<Type,AF,padded,TF>::BLAZE_TEMPLATE VectorizedAssign<VT> >
   CustomVector<Type,AF,padded,TF>::assign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const bool remainder( !IsPadded<VT>::value );

   const size_t ipos( ( remainder )?( size_ & size_t(-SIMDSIZE) ):( size_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( size_ - ( size_ % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

   if( AF && useStreaming && size_ > ( cacheSize/( sizeof(Type) * 3UL ) ) && !(~rhs).isAliased( this ) )
   {
      size_t i( 0UL );

      for( ; i<ipos; i+=SIMDSIZE ) {
         stream( i, (~rhs).load(i) );
      }
      for( ; remainder && i<size_; ++i ) {
         v_[i] = (~rhs)[i];
      }
   }
   else
   {
      const size_t i4way( size_ & size_t(-SIMDSIZE*4) );
      BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE*4UL) ) ) == i4way, "Invalid end calculation" );
      BLAZE_INTERNAL_ASSERT( i4way <= ipos, "Invalid end calculation" );

      size_t i( 0UL );
      ConstIterator_<VT> it( (~rhs).begin() );

      for( ; i<i4way; i+=SIMDSIZE*4UL ) {
         store( i             , it.load() ); it += SIMDSIZE;
         store( i+SIMDSIZE    , it.load() ); it += SIMDSIZE;
         store( i+SIMDSIZE*2UL, it.load() ); it += SIMDSIZE;
         store( i+SIMDSIZE*3UL, it.load() ); it += SIMDSIZE;
      }
      for( ; i<ipos; i+=SIMDSIZE, it+=SIMDSIZE ) {
         store( i, it.load() );
      }
      for( ; remainder && i<size_; ++i, ++it ) {
         v_[i] = *it;
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CustomVector<Type,AF,padded,TF>::assign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   for( ConstIterator_<VT> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      v_[element->index()] = element->value();
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename CustomVector<Type,AF,padded,TF>::BLAZE_TEMPLATE VectorizedAddAssign<VT> >
   CustomVector<Type,AF,padded,TF>::addAssign( const DenseVector<VT,TF>& rhs )
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename CustomVector<Type,AF,padded,TF>::BLAZE_TEMPLATE VectorizedAddAssign<VT> >
   CustomVector<Type,AF,padded,TF>::addAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const bool remainder( !IsPadded<VT>::value );

   const size_t ipos( ( remainder )?( size_ & size_t(-SIMDSIZE) ):( size_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( size_ - ( size_ % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

   const size_t i4way( size_ & size_t(-SIMDSIZE*4) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE*4UL) ) ) == i4way, "Invalid end calculation" );
   BLAZE_INTERNAL_ASSERT( i4way <= ipos, "Invalid end calculation" );

   size_t i( 0UL );
   ConstIterator_<VT> it( (~rhs).begin() );

   for( ; i<i4way; i+=SIMDSIZE*4UL ) {
      store( i             , load(i             ) + it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE    , load(i+SIMDSIZE    ) + it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*2UL, load(i+SIMDSIZE*2UL) + it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*3UL, load(i+SIMDSIZE*3UL) + it.load() ); it += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE, it+=SIMDSIZE ) {
      store( i, load(i) + it.load() );
   }
   for( ; remainder && i<size_; ++i, ++it ) {
      v_[i] += *it;
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CustomVector<Type,AF,padded,TF>::addAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   for( ConstIterator_<VT> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      v_[element->index()] += element->value();
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename CustomVector<Type,AF,padded,TF>::BLAZE_TEMPLATE VectorizedSubAssign<VT> >
   CustomVector<Type,AF,padded,TF>::subAssign( const DenseVector<VT,TF>& rhs )
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename CustomVector<Type,AF,padded,TF>::BLAZE_TEMPLATE VectorizedSubAssign<VT> >
   CustomVector<Type,AF,padded,TF>::subAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const bool remainder( !IsPadded<VT>::value );

   const size_t ipos( ( remainder )?( size_ & size_t(-SIMDSIZE) ):( size_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( size_ - ( size_ % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

   const size_t i4way( size_ & size_t(-SIMDSIZE*4) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE*4UL) ) ) == i4way, "Invalid end calculation" );
   BLAZE_INTERNAL_ASSERT( i4way <= ipos, "Invalid end calculation" );

   size_t i( 0UL );
   ConstIterator_<VT> it( (~rhs).begin() );

   for( ; i<i4way; i+=SIMDSIZE*4UL ) {
      store( i             , load(i             ) - it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE    , load(i+SIMDSIZE    ) - it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*2UL, load(i+SIMDSIZE*2UL) - it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*3UL, load(i+SIMDSIZE*3UL) - it.load() ); it += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE, it+=SIMDSIZE ) {
      store( i, load(i) - it.load() );
   }
   for( ; remainder && i<size_; ++i, ++it ) {
      v_[i] -= *it;
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CustomVector<Type,AF,padded,TF>::subAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   for( ConstIterator_<VT> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      v_[element->index()] -= element->value();
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename CustomVector<Type,AF,padded,TF>::BLAZE_TEMPLATE VectorizedMultAssign<VT> >
   CustomVector<Type,AF,padded,TF>::multAssign( const DenseVector<VT,TF>& rhs )
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename CustomVector<Type,AF,padded,TF>::BLAZE_TEMPLATE VectorizedMultAssign<VT> >
   CustomVector<Type,AF,padded,TF>::multAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const bool remainder( !IsPadded<VT>::value );

   const size_t ipos( ( remainder )?( size_ & size_t(-SIMDSIZE) ):( size_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( size_ - ( size_ % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

   const size_t i4way( size_ & size_t(-SIMDSIZE*4) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE*4UL) ) ) == i4way, "Invalid end calculation" );
   BLAZE_INTERNAL_ASSERT( i4way <= ipos, "Invalid end calculation" );

   size_t i( 0UL );
   ConstIterator_<VT> it( (~rhs).begin() );

   for( ; i<i4way; i+=SIMDSIZE*4UL ) {
      store( i             , load(i             ) * it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE    , load(i+SIMDSIZE    ) * it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*2UL, load(i+SIMDSIZE*2UL) * it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*3UL, load(i+SIMDSIZE*3UL) * it.load() ); it += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE, it+=SIMDSIZE ) {
      store( i, load(i) * it.load() );
   }
   for( ; remainder && i<size_; ++i, ++it ) {
      v_[i] *= *it;
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CustomVector<Type,AF,padded,TF>::multAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const DynamicVector<Type,TF> tmp( serial( *this ) );

   reset();

   for( ConstIterator_<VT> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      v_[element->index()] = tmp[element->index()] * element->value();
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline DisableIf_<typename CustomVector<Type,AF,padded,TF>::BLAZE_TEMPLATE VectorizedDivAssign<VT> >
   CustomVector<Type,AF,padded,TF>::divAssign( const DenseVector<VT,TF>& rhs )
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
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline EnableIf_<typename CustomVector<Type,AF,padded,TF>::BLAZE_TEMPLATE VectorizedDivAssign<VT> >
   CustomVector<Type,AF,padded,TF>::divAssign( const DenseVector<VT,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size_ & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % SIMDSIZE ) ) == ipos, "Invalid end calculation" );

   const size_t i4way( size_ & size_t(-SIMDSIZE*4) );
   BLAZE_INTERNAL_ASSERT( ( size_ - ( size_ % (SIMDSIZE*4UL) ) ) == i4way, "Invalid end calculation" );
   BLAZE_INTERNAL_ASSERT( i4way <= ipos, "Invalid end calculation" );

   size_t i( 0UL );
   ConstIterator_<VT> it( (~rhs).begin() );

   for( ; i<i4way; i+=SIMDSIZE*4UL ) {
      store( i             , load(i             ) / it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE    , load(i+SIMDSIZE    ) / it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*2UL, load(i+SIMDSIZE*2UL) / it.load() ); it += SIMDSIZE;
      store( i+SIMDSIZE*3UL, load(i+SIMDSIZE*3UL) / it.load() ); it += SIMDSIZE;
   }
   for( ; i<ipos; i+=SIMDSIZE, it+=SIMDSIZE ) {
      store( i, load(i) / it.load() );
   }
   for( ; i<size_; ++i, ++it ) {
      v_[i] /= *it;
   }
}
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  CUSTOMVECTOR OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name CustomVector operators */
//@{
template< typename Type, bool AF, bool PF, bool TF >
inline void reset( CustomVector<Type,AF,PF,TF>& v );

template< typename Type, bool AF, bool PF, bool TF >
inline void clear( CustomVector<Type,AF,PF,TF>& v );

template< typename Type, bool AF, bool PF, bool TF >
inline bool isDefault( const CustomVector<Type,AF,PF,TF>& v );

template< typename Type, bool AF, bool PF, bool TF >
inline bool isIntact( const CustomVector<Type,AF,PF,TF>& v ) noexcept;

template< typename Type, bool AF, bool PF, bool TF >
inline void swap( CustomVector<Type,AF,PF,TF>& a, CustomVector<Type,AF,PF,TF>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given custom vector.
// \ingroup custom_vector
//
// \param v The custom vector to be resetted.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline void reset( CustomVector<Type,AF,PF,TF>& v )
{
   v.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given custom vector.
// \ingroup custom_vector
//
// \param v The custom vector to be cleared.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline void clear( CustomVector<Type,AF,PF,TF>& v )
{
   v.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given custom vector is in default state.
// \ingroup custom_vector
//
// \param v The custom vector to be tested for its default state.
// \return \a true in case the given vector is component-wise zero, \a false otherwise.
//
// This function checks whether the custom vector is in default state. For instance, in case
// the static vector is instantiated for a built-in integral or floating point data type, the
// function returns \a true in case all vector elements are 0 and \a false in case any vector
// element is not 0. Following example demonstrates the use of the \a isDefault function:

   \code
   using blaze::aligned;
   using blaze::padded;

   blaze::CustomVector<int,aligned,padded> a( ... );
   // ... Resizing and initialization
   if( isDefault( a ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline bool isDefault( const CustomVector<Type,AF,PF,TF>& v )
{
   return ( v.size() == 0UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given custom vector are intact.
// \ingroup custom_vector
//
// \param v The custom vector to be tested.
// \return \a true in case the given vector's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the custom vector are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   using blaze::aligned;
   using blaze::padded;

   blaze::CustomVector<int,aligned,padded> a( ... );
   // ... Resizing and initialization
   if( isIntact( a ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline bool isIntact( const CustomVector<Type,AF,PF,TF>& v ) noexcept
{
   return ( v.size() <= v.capacity() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two vectors.
// \ingroup custom_vector
//
// \param a The first vector to be swapped.
// \param b The second vector to be swapped.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool TF >      // Transpose flag
inline void swap( CustomVector<Type,AF,PF,TF>& a, CustomVector<Type,AF,PF,TF>& b ) noexcept
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
template< typename T, bool AF, bool PF, bool TF >
struct HasConstDataAccess< CustomVector<T,AF,PF,TF> > : public TrueType
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
template< typename T, bool AF, bool PF, bool TF >
struct HasMutableDataAccess< CustomVector<T,AF,PF,TF> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISCUSTOM SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool AF, bool PF, bool TF >
struct IsCustom< CustomVector<T,AF,PF,TF> > : public TrueType
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
template< typename T, bool PF, bool TF >
struct IsAligned< CustomVector<T,aligned,PF,TF> > : public TrueType
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
template< typename T, bool AF, bool TF >
struct IsPadded< CustomVector<T,AF,padded,TF> > : public TrueType
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
template< typename T1, bool AF, bool PF, bool TF, typename T2, size_t N >
struct AddTrait< CustomVector<T1,AF,PF,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< AddTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2, bool AF, bool PF >
struct AddTrait< StaticVector<T1,N,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = StaticVector< AddTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2, size_t N >
struct AddTrait< CustomVector<T1,AF,PF,TF>, HybridVector<T2,N,TF> >
{
   using Type = HybridVector< AddTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2, bool AF, bool PF >
struct AddTrait< HybridVector<T1,N,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = HybridVector< AddTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2 >
struct AddTrait< CustomVector<T1,AF,PF,TF>, DynamicVector<T2,TF> >
{
   using Type = DynamicVector< AddTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2, bool AF, bool PF >
struct AddTrait< DynamicVector<T1,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = DynamicVector< AddTrait_<T1,T2>, TF >;
};

template< typename T1, bool AF1, bool PF1, bool TF, typename T2, bool AF2, bool PF2 >
struct AddTrait< CustomVector<T1,AF1,PF1,TF>, CustomVector<T2,AF2,PF2,TF> >
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
template< typename T1, bool AF, bool PF, bool TF, typename T2, size_t N >
struct SubTrait< CustomVector<T1,AF,PF,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< SubTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2, bool AF, bool PF >
struct SubTrait< StaticVector<T1,N,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = StaticVector< SubTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2, size_t N >
struct SubTrait< CustomVector<T1,AF,PF,TF>, HybridVector<T2,N,TF> >
{
   using Type = HybridVector< SubTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2, bool AF, bool PF >
struct SubTrait< HybridVector<T1,N,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = HybridVector< SubTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2 >
struct SubTrait< CustomVector<T1,AF,PF,TF>, DynamicVector<T2,TF> >
{
   using Type = DynamicVector< SubTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2, bool AF, bool PF >
struct SubTrait< DynamicVector<T1,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = DynamicVector< SubTrait_<T1,T2>, TF >;
};

template< typename T1, bool AF1, bool PF1, bool TF, typename T2, bool AF2, bool PF2 >
struct SubTrait< CustomVector<T1,AF1,PF1,TF>, CustomVector<T2,AF2,PF2,TF> >
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
template< typename T1, bool AF, bool PF, bool TF, typename T2 >
struct MultTrait< CustomVector<T1,AF,PF,TF>, T2, EnableIf_<IsNumeric<T2> > >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, typename T2, bool AF, bool PF, bool TF >
struct MultTrait< T1, CustomVector<T2,AF,PF,TF>, EnableIf_<IsNumeric<T1> > >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2, size_t N >
struct MultTrait< CustomVector<T1,AF,PF,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< MultTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool AF, bool PF, typename T2, size_t N >
struct MultTrait< CustomVector<T1,AF,PF,false>, StaticVector<T2,N,true> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, bool AF, bool PF, typename T2, size_t N >
struct MultTrait< CustomVector<T1,AF,PF,true>, StaticVector<T2,N,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, size_t N, bool TF, typename T2, bool AF, bool PF >
struct MultTrait< StaticVector<T1,N,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = StaticVector< MultTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, typename T2, bool AF, bool PF >
struct MultTrait< StaticVector<T1,N,false>, CustomVector<T2,AF,PF,true> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, size_t N, typename T2, bool AF, bool PF >
struct MultTrait< StaticVector<T1,N,true>, CustomVector<T2,AF,PF,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2, size_t N >
struct MultTrait< CustomVector<T1,AF,PF,TF>, HybridVector<T2,N,TF> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool AF, bool PF, typename T2, size_t N >
struct MultTrait< CustomVector<T1,AF,PF,false>, HybridVector<T2,N,true> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, bool AF, bool PF, typename T2, size_t N >
struct MultTrait< CustomVector<T1,AF,PF,true>, HybridVector<T2,N,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, size_t N, bool TF, typename T2, bool AF, bool PF >
struct MultTrait< HybridVector<T1,N,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, typename T2, bool AF, bool PF >
struct MultTrait< HybridVector<T1,N,false>, CustomVector<T2,AF,PF,true> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, size_t N, typename T2, bool AF, bool PF >
struct MultTrait< HybridVector<T1,N,true>, CustomVector<T2,AF,PF,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2 >
struct MultTrait< CustomVector<T1,AF,PF,TF>, DynamicVector<T2,TF> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, bool AF, bool PF, typename T2 >
struct MultTrait< CustomVector<T1,AF,PF,false>, DynamicVector<T2,true> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, bool AF, bool PF, typename T2 >
struct MultTrait< CustomVector<T1,AF,PF,true>, DynamicVector<T2,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, bool TF, typename T2, bool AF, bool PF >
struct MultTrait< DynamicVector<T1,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, typename T2, bool AF, bool PF >
struct MultTrait< DynamicVector<T1,false>, CustomVector<T2,AF,PF,true> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, typename T2, bool AF, bool PF >
struct MultTrait< DynamicVector<T1,true>, CustomVector<T2,AF,PF,false> >
{
   using Type = MultTrait_<T1,T2>;
};

template< typename T1, bool AF1, bool PF1, bool TF, typename T2, bool AF2, bool PF2 >
struct MultTrait< CustomVector<T1,AF1,PF1,TF>, CustomVector<T2,AF2,PF2,TF> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, TF >;
};

template< typename T1, bool AF1, bool PF1, typename T2, bool AF2, bool PF2 >
struct MultTrait< CustomVector<T1,AF1,PF1,false>, CustomVector<T2,AF2,PF2,true> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, false >;
};

template< typename T1, bool AF1, bool PF1, typename T2, bool AF2, bool PF2 >
struct MultTrait< CustomVector<T1,AF1,PF1,true>, CustomVector<T2,AF2,PF2,false> >
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
template< typename T1, bool AF, bool PF, bool TF, typename T2 >
struct CrossTrait< CustomVector<T1,AF,PF,TF>, StaticVector<T2,3UL,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, bool TF, typename T2, bool AF, bool PF >
struct CrossTrait< StaticVector<T1,3UL,TF>, CustomVector<T2,AF,PF,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2, size_t N >
struct CrossTrait< CustomVector<T1,AF,PF,TF>, HybridVector<T2,N,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, size_t N, bool TF, typename T2, bool AF, bool PF >
struct CrossTrait< HybridVector<T1,N,TF>, CustomVector<T2,AF,PF,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2 >
struct CrossTrait< CustomVector<T1,AF,PF,TF>, DynamicVector<T2,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, bool TF, typename T2, bool AF, bool PF >
struct CrossTrait< DynamicVector<T1,TF>, CustomVector<T2,AF,PF,TF> >
{
 private:
   using T = MultTrait_<T1,T2>;

 public:
   using Type = StaticVector< SubTrait_<T,T>, 3UL, TF >;
};

template< typename T1, bool AF1, bool PF1, bool TF, typename T2, bool AF2, bool PF2 >
struct CrossTrait< CustomVector<T1,AF1,PF1,TF>, CustomVector<T2,AF2,PF2,TF> >
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
template< typename T1, bool AF, bool PF, bool TF, typename T2 >
struct DivTrait< CustomVector<T1,AF,PF,TF>, T2, EnableIf_<IsNumeric<T2> > >
{
   using Type = DynamicVector< DivTrait_<T1,T2>, TF >;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2, size_t N >
struct DivTrait< CustomVector<T1,AF,PF,TF>, StaticVector<T2,N,TF> >
{
   using Type = StaticVector< DivTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2, bool AF, bool PF >
struct DivTrait< StaticVector<T1,N,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = StaticVector< DivTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2, size_t N >
struct DivTrait< CustomVector<T1,AF,PF,TF>, HybridVector<T2,N,TF> >
{
   using Type = HybridVector< DivTrait_<T1,T2>, N, TF >;
};

template< typename T1, size_t N, bool TF, typename T2, bool AF, bool PF >
struct DivTrait< HybridVector<T1,N,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = HybridVector< DivTrait_<T1,T2>, N, TF >;
};

template< typename T1, bool AF, bool PF, bool TF, typename T2 >
struct DivTrait< CustomVector<T1,AF,PF,TF>, DynamicVector<T2,TF> >
{
   using Type = DynamicVector< DivTrait_<T1,T2>, TF >;
};

template< typename T1, bool TF, typename T2, bool AF, bool PF >
struct DivTrait< DynamicVector<T1,TF>, CustomVector<T2,AF,PF,TF> >
{
   using Type = DynamicVector< DivTrait_<T1,T2>, TF >;
};

template< typename T1, bool AF1, bool PF1, bool TF, typename T2, bool AF2, bool PF2 >
struct DivTrait< CustomVector<T1,AF1,PF1,TF>, CustomVector<T2,AF2,PF2,TF> >
{
   using Type = DynamicVector< DivTrait_<T1,T2>, TF >;
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
template< typename T1, bool AF, bool PF, bool TF >
struct SubvectorTrait< CustomVector<T1,AF,PF,TF> >
{
   using Type = DynamicVector<T1,TF>;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
