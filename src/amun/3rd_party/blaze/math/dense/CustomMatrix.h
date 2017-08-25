//=================================================================================================
/*!
//  \file blaze/math/dense/CustomMatrix.h
//  \brief Header file for the implementation of a customizable matrix
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

#ifndef _BLAZE_MATH_DENSE_CUSTOMMATRIX_H_
#define _BLAZE_MATH_DENSE_CUSTOMMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <utility>
#include <boost/smart_ptr/shared_array.hpp>
#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/constraints/Diagonal.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/dense/DenseIterator.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/Forward.h>
#include <blaze/math/Functions.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/PaddingFlag.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/Conjugate.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/ColumnTrait.h>
#include <blaze/math/traits/CTransExprTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/InvExprTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/RowTrait.h>
#include <blaze/math/traits/SubmatrixTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/traits/TransExprTrait.h>
#include <blaze/math/typetraits/AreSIMDCombinable.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDSub.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsCustom.h>
#include <blaze/math/typetraits/IsDiagonal.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsSMPAssignable.h>
#include <blaze/math/typetraits/IsSparseMatrix.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/system/Blocking.h>
#include <blaze/system/CacheSize.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/StorageOrder.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/Assert.h>
#include <blaze/util/AlignmentCheck.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Vectorizable.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/Misalignment.h>
#include <blaze/util/policies/NoDelete.h>
#include <blaze/util/Template.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blaze/util/typetraits/IsClass.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsSame.h>
#include <blaze/util/typetraits/IsVectorizable.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup custom_matrix CustomMatrix
// \ingroup dense_matrix
*/
/*!\brief Efficient implementation of a customizable matrix.
// \ingroup custom_matrix
//
// The CustomMatrix class template provides the functionality to represent an external array of
// elements of arbitrary type and a fixed size as a native \b Blaze dense matrix data structure.
// Thus in contrast to all other dense matrix types a custom matrix does not perform any kind
// of memory allocation by itself, but it is provided with an existing array of element during
// construction. A custom matrix can therefore be considered an alias to the existing array.
//
// The type of the elements, the properties of the given array of elements and the storage order
// of the matrix can be specified via the following four template parameters:

   \code
   template< typename Type, bool AF, bool PF, bool SO >
   class CustomMatrix;
   \endcode

//  - Type: specifies the type of the matrix elements. CustomMatrix can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
//  - AF  : specifies whether the represented, external arrays are properly aligned with
//          respect to the available instruction set (SSE, AVX, ...) or not.
//  - PF  : specified whether the represented, external arrays are properly padded with
//          respect to the available instruction set (SSE, AVX, ...) or not.
//  - SO  : specifies the storage order (blaze::rowMajor, blaze::columnMajor) of the matrix.
//          The default value is blaze::rowMajor.
//
// The following examples give an impression of several possible types of custom matrices:

   \code
   using blaze::CustomMatrix;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;

   // Definition of a custom row-major matrix for unaligned, unpadded integer arrays
   typedef CustomMatrix<int,unaligned,unpadded,rowMajor>  UnalignedUnpadded;

   // Definition of a custom column-major matrix for unaligned but padded 'float' arrays
   typedef CustomMatrix<float,unaligned,padded,columnMajor>  UnalignedPadded;

   // Definition of a custom row-major matrix for aligned, unpadded 'double' arrays
   typedef CustomMatrix<double,aligned,unpadded,rowMajor>  AlignedUnpadded;

   // Definition of a custom column-major matrix for aligned, padded 'complex<double>' arrays
   typedef CustomMatrix<complex<double>,aligned,padded,columnMajor>  AlignedPadded;
   \endcode

// \n \section custommatrix_special_properties Special Properties of Custom Matrices
//
// In comparison with the remaining \b Blaze dense matrix types CustomMatrix has several special
// characteristics. All of these result from the fact that a custom matrix is not performing any
// kind of memory allocation, but instead is given an existing array of elements. The following
// sections discuss all of these characteristics:
//
//  -# <b>\ref custommatrix_memory_management</b>
//  -# <b>\ref custommatrix_copy_operations</b>
//  -# <b>\ref custommatrix_alignment</b>
//  -# <b>\ref custommatrix_padding</b>
//
// \n \subsection custommatrix_memory_management Memory Management
//
// The CustomMatrix class template acts as an adaptor for an existing array of elements. As such
// it provides everything that is required to use the array just like a native \b Blaze dense
// matrix data structure. However, this flexibility comes with the price that the user of a custom
// matrix is responsible for the resource management.
//
// When constructing a custom matrix there are two choices: Either a user manually manages the
// array of elements outside the custom matrix, or alternatively passes the responsibility for
// the memory management to an instance of CustomMatrix. In the second case the CustomMatrix
// class employs shared ownership between all copies of the custom matrix, which reference the
// same array.
//
// The following examples give an impression of several possible custom matrices:

   \code
   using blaze::CustomMatrix;
   using blaze::ArrayDelete;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowMajor;
   using blaze::columnMajor;

   // Definition of a 3x4 custom row-major matrix with unaligned, unpadded and externally
   // managed integer array. Note that the std::vector must be guaranteed to outlive the
   // custom matrix!
   std::vector<int> vec( 12UL );
   CustomMatrix<int,unaligned,unpadded> A( &vec[0], 3UL, 4UL );

   // Definition of a 3x4 custom row-major matrix for unaligned, unpadded integer arrays.
   // The responsibility for the memory management is passed to the custom matrix by
   // providing a deleter of type 'blaze::ArrayDelete' that is used during the destruction
   // of the custom matrix.
   CustomMatrix<int,unaligned,unpadded,rowMajor> B( new int[12], 3UL, 4UL, ArrayDelete() );

   // Definition of a custom 8x12 matrix for an aligned and padded integer array of
   // capacity 128 (including 8 padding elements per row). The memory management is passed
   // to the custom matrix by providing a deleter of type 'blaze::Deallocate'.
   CustomMatrix<int,aligned,padded> C( allocate<int>( 128UL ), 8UL, 12UL, 16UL, Deallocate() );
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

// \n \subsection custommatrix_copy_operations Copy Operations
//
// As with all dense matrices it is possible to copy construct a custom matrix:

   \code
   using blaze::CustomMatrix;
   using blaze::unaligned;
   using blaze::unpadded;

   typedef CustomMatrix<int,unaligned,unpadded>  CustomType;

   std::vector<int> vec( 6UL, 10 );    // Vector of 6 integers of the value 10
   CustomType A( &vec[0], 2UL, 3UL );  // Represent the std::vector as Blaze dense matrix
   a[1] = 20;                          // Also modifies the std::vector

   CustomType B( a );  // Creating a copy of vector a
   b[2] = 20;          // Also affect matrix A and the std::vector
   \endcode

// It is important to note that a custom matrix acts as a reference to the specified array. Thus
// the result of the copy constructor is a new custom matrix that is referencing and representing
// the same array as the original custom matrix. In case a deleter has been provided to the first
// custom matrix, both matrices share the responsibility to destroy the array when the last matrix
// goes out of scope.
//
// In contrast to copy construction, just as with references, copy assignment does not change
// which array is referenced by the custom matrices, but modifies the values of the array:

   \code
   std::vector<int> vec2( 6UL, 4 );     // Vector of 6 integers of the value 4
   CustomType C( &vec2[0], 2UL, 3UL );  // Represent the std::vector as Blaze dense matrix

   A = C;  // Copy assignment: Set all values of matrix A and B to 4.
   \endcode

// \n \subsection custommatrix_alignment Alignment
//
// In case the custom matrix is specified as \a aligned the passed array must adhere to some
// alignment restrictions based on the alignment requirements of the used data type and the
// used instruction set (SSE, AVX, ...). The restriction applies to the first element of each
// row/column: In case of a row-major matrix the first element of each row must be properly
// aligned, in case of a column-major matrix the first element of each column must be properly
// aligned. For instance, if a row-major matrix is used and AVX is active the first element of
// each row must be 32-bit aligned:

   \code
   using blaze::CustomMatrix;
   using blaze::Deallocate;
   using blaze::aligned;
   using blaze::padded;
   using blaze::rowMajor;

   int* array = blaze::allocate<int>( 40UL );  // Is guaranteed to be 32-bit aligned
   CustomMatrix<int,aligned,padded,rowMajor> A( array, 5UL, 6UL, 8UL, Deallocate() );
   \endcode

// In the example, the row-major matrix has six columns. However, since with AVX eight integer
// values are loaded together the matrix is padded with two additional elements. This guarantees
// that the first element of each row is 32-bit aligned. In case the alignment requirements are
// violated, a \a std::invalid_argument exception is thrown.
//
// \n \subsection custommatrix_padding Padding
//
// Adding padding elements to the end of an array can have a significant impact on performance.
// For instance, assuming that AVX is available, then two aligned, padded, 3x3 double precision
// matrices can be added via three SIMD addition operations:

   \code
   using blaze::CustomMatrix;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::padded;

   typedef CustomMatrix<double,aligned,padded>  CustomType;

   // Creating padded custom 3x3 matrix with an additional padding element in each row
   CustomType A( allocate<double>( 12UL ), 3UL, 3UL, 4UL, Deallocate() );
   CustomType B( allocate<double>( 12UL ), 3UL, 3UL, 4UL, Deallocate() );
   CustomType C( allocate<double>( 12UL ), 3UL, 3UL, 4UL, Deallocate() );

   // ... Initialization

   C = A + B;  // AVX-based matrix addition
   \endcode

// In this example, maximum performance is possible. However, in case no padding elements are
// inserted a scalar addition has to be used:

   \code
   using blaze::CustomMatrix;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::unpadded;

   typedef CustomMatrix<double,aligned,unpadded>  CustomType;

   // Creating unpadded custom 3x3 matrix
   CustomType A( allocate<double>( 12UL ), 3UL, 3UL, 4UL, Deallocate() );
   CustomType B( allocate<double>( 12UL ), 3UL, 3UL, 4UL, Deallocate() );
   CustomType C( allocate<double>( 12UL ), 3UL, 3UL, 4UL, Deallocate() );

   // ... Initialization

   C = A + B;  // Scalar matrix addition
   \endcode

// Note that the construction of padded and unpadded aligned matrices looks identical. However,
// in case of padded matrices, \b Blaze will zero initialize the padding element and use them
// in all computations in order to achieve maximum performance. In case of an unpadded matrix
// \b Blaze will ignore the elements with the downside that it is not possible to load a complete
// row to an AVX register, which makes it necessary to fall back to a scalar addition.
//
// The number of padding elements is required to be sufficient with respect to the available
// instruction set: In case of an aligned padded custom matrix the added padding elements must
// guarantee that the total number of elements in each row/column is a multiple of the SIMD
// vector width. In case of an unaligned padded matrix the number of padding elements can be
// greater or equal the number of padding elements of an aligned padded custom matrix. In case
// the padding is insufficient with respect to the available instruction set, a
// \a std::invalid_argument exception is thrown.
//
//
// \n \section custommatrix_arithmetic_operations Arithmetic Operations
//
// The use of custom matrices in arithmetic operations is designed to be as natural and intuitive
// as possible. All operations (addition, subtraction, multiplication, scaling, ...) can be
// expressed similar to a text book representation. Also, custom matrices can be combined with all
// other dense and sparse vectors and matrices. The following example gives an impression of the
// use of CustomMatrix:

   \code
   using blaze::CustomMatrix;
   using blaze::CompressedMatrix;
   using blaze::ArrayDelete;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowMajor;
   using blaze::columnMajor;

   // Non-initialized custom 2x3 row-major matrix. All given arrays are considered to be
   // unaligned and unpadded. The memory is managed via 'ArrayDelete'.
   CustomMatrix<double,unaligned,unpadded> A( new double[6], 2UL, 3UL, blaze::ArrayDelete() );

   A(0,0) = 1.0; A(0,1) = 2.0; A(0,2) = 3.0;  // Initialization of the first row
   A(1,0) = 4.0; A(1,1) = 5.0; A(1,2) = 6.0;  // Initialization of the second row

   // Non-initialized custom 2x3 row-major matrix with padding elements. All given arrays are
   // required to be properly aligned and padded. The memory is managed via 'Deallocate'.
   CustomMatrix<double,aligned,padded> B( allocate<double>( 16UL ), 2UL, 3UL, 8UL, Deallocate() );

   B(0,0) = 1.0; B(0,1) = 3.0; B(0,2) = 5.0;    // Initialization of the first row
   B(1,0) = 2.0; B(1,1) = 4.0; B(1,2) = 6.0;    // Initialization of the second row

   CompressedMatrix<float> C( 2, 3 );        // Empty row-major sparse single precision matrix
   DynamicMatrix<float>    D( 3, 2, 4.0F );  // Directly, homogeneously initialized single precision 3x2 matrix

   DynamicMatrix<double,rowMajor>    E( A );  // Creation of a new row-major matrix as a copy of A
   DynamicMatrix<double,columnMajor> F;       // Creation of a default column-major matrix

   E = A + B;     // Matrix addition and assignment to a row-major matrix
   F = A - C;     // Matrix subtraction and assignment to a column-major matrix
   F = A * D;     // Matrix multiplication between two matrices of different element types

   A *= 2.0;      // In-place scaling of matrix A
   E  = 2.0 * B;  // Scaling of matrix B
   F  = D * 2.0;  // Scaling of matrix D

   E += A - B;    // Addition assignment
   E -= A + C;    // Subtraction assignment
   F *= A * D;    // Multiplication assignment
   \endcode
*/
template< typename Type                    // Data type of the matrix
        , bool AF                          // Alignment flag
        , bool PF                          // Padding flag
        , bool SO = defaultStorageOrder >  // Storage order
class CustomMatrix : public DenseMatrix< CustomMatrix<Type,AF,PF,SO>, SO >
{
 public:
   //**Type definitions****************************************************************************
   typedef CustomMatrix<Type,AF,PF,SO>  This;           //!< Type of this CustomMatrix instance.
   typedef DenseMatrix<This,SO>         BaseType;       //!< Base type of this CustomMatrix instance.
   typedef DynamicMatrix<Type,SO>       ResultType;     //!< Result type for expression template evaluations.
   typedef DynamicMatrix<Type,!SO>      OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef DynamicMatrix<Type,!SO>      TransposeType;  //!< Transpose type for expression template evaluations.
   typedef Type                         ElementType;    //!< Type of the matrix elements.
   typedef SIMDTrait_<ElementType>      SIMDType;       //!< SIMD type of the matrix elements.
   typedef const Type&                  ReturnType;     //!< Return type for expression template evaluations.
   typedef const This&                  CompositeType;  //!< Data type for composite expression templates.

   typedef Type&        Reference;       //!< Reference to a non-constant matrix value.
   typedef const Type&  ConstReference;  //!< Reference to a constant matrix value.
   typedef Type*        Pointer;         //!< Pointer to a non-constant matrix value.
   typedef const Type*  ConstPointer;    //!< Pointer to a constant matrix value.

   typedef DenseIterator<Type,AF>        Iterator;       //!< Iterator over non-constant elements.
   typedef DenseIterator<const Type,AF>  ConstIterator;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a CustomMatrix with different data/element type.
   */
   template< typename ET >  // Data type of the other matrix
   struct Rebind {
      typedef CustomMatrix<ET,AF,PF,SO>  Other;  //!< The type of the other CustomMatrix.
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SIMD optimization.
   /*! The \a simdEnabled compilation flag indicates whether expressions the matrix is involved
       in can be optimized via SIMD operations. In case the element type of the matrix is a
       vectorizable data type, the \a simdEnabled compilation flag is set to \a true, otherwise
       it is set to \a false. */
   enum : bool { simdEnabled = IsVectorizable<Type>::value };

   //! Compilation flag for SMP assignments.
   /*! The \a smpAssignable compilation flag indicates whether the matrix can be used in SMP
       (shared memory parallel) assignments (both on the left-hand and right-hand side of the
       assignment). */
   enum : bool { smpAssignable = !IsSMPAssignable<Type>::value };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline CustomMatrix();
   explicit inline CustomMatrix( Type* ptr, size_t m, size_t n );
   explicit inline CustomMatrix( Type* ptr, size_t m, size_t n, size_t nn );

   template< typename Deleter, typename = EnableIf_< IsClass<Deleter> > >
   explicit inline CustomMatrix( Type* ptr, size_t m, size_t n, Deleter D );

   template< typename Deleter >
   explicit inline CustomMatrix( Type* ptr, size_t m, size_t n, size_t nn, Deleter D );

   inline CustomMatrix( const CustomMatrix& m );
   inline CustomMatrix( CustomMatrix&& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator()( size_t i, size_t j ) noexcept;
   inline ConstReference operator()( size_t i, size_t j ) const noexcept;
   inline Reference      at( size_t i, size_t j );
   inline ConstReference at( size_t i, size_t j ) const;
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Pointer        data  ( size_t i ) noexcept;
   inline ConstPointer   data  ( size_t i ) const noexcept;
   inline Iterator       begin ( size_t i ) noexcept;
   inline ConstIterator  begin ( size_t i ) const noexcept;
   inline ConstIterator  cbegin( size_t i ) const noexcept;
   inline Iterator       end   ( size_t i ) noexcept;
   inline ConstIterator  end   ( size_t i ) const noexcept;
   inline ConstIterator  cend  ( size_t i ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline CustomMatrix& operator=( const Type& set );
   inline CustomMatrix& operator=( initializer_list< initializer_list<Type> > list );

   template< typename Other, size_t M, size_t N >
   inline CustomMatrix& operator=( const Other (&array)[M][N] );

   inline CustomMatrix& operator=( const CustomMatrix& rhs );
   inline CustomMatrix& operator=( CustomMatrix&& rhs ) noexcept;

   template< typename MT, bool SO2 > inline CustomMatrix& operator= ( const Matrix<MT,SO2>& rhs );
   template< typename MT, bool SO2 > inline CustomMatrix& operator+=( const Matrix<MT,SO2>& rhs );
   template< typename MT, bool SO2 > inline CustomMatrix& operator-=( const Matrix<MT,SO2>& rhs );
   template< typename MT, bool SO2 > inline CustomMatrix& operator*=( const Matrix<MT,SO2>& rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, CustomMatrix >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, CustomMatrix >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t        rows() const noexcept;
                              inline size_t        columns() const noexcept;
                              inline size_t        spacing() const noexcept;
                              inline size_t        capacity() const noexcept;
                              inline size_t        capacity( size_t i ) const noexcept;
                              inline size_t        nonZeros() const;
                              inline size_t        nonZeros( size_t i ) const;
                              inline void          reset();
                              inline void          reset( size_t i );
                              inline void          clear();
                              inline CustomMatrix& transpose();
                              inline CustomMatrix& ctranspose();
   template< typename Other > inline CustomMatrix& scale( const Other& scalar );
                              inline void          swap( CustomMatrix& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Resource management functions***************************************************************
   /*!\name Resource management functions */
   //@{
   inline void reset( Type* ptr, size_t m, size_t n );
   inline void reset( Type* ptr, size_t m, size_t n, size_t nn );

   template< typename Deleter, typename = EnableIf_< IsClass<Deleter> > >
   inline void reset( Type* ptr, size_t m, size_t n, Deleter d );

   template< typename Deleter >
   inline void reset( Type* ptr, size_t m, size_t n, size_t nn, Deleter d );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT >
   struct VectorizedAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<MT> >::value };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT >
   struct VectorizedAddAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<MT> >::value &&
                            HasSIMDAdd< Type, ElementType_<MT> >::value &&
                            !IsDiagonal<MT>::value };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT >
   struct VectorizedSubAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<MT> >::value &&
                            HasSIMDSub< Type, ElementType_<MT> >::value &&
                            !IsDiagonal<MT>::value };
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

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t i, size_t j ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t i, size_t j, const SIMDType& value ) noexcept;

   template< typename MT >
   inline DisableIf_<VectorizedAssign<MT> > assign( const DenseMatrix<MT,SO>& rhs );

   template< typename MT >
   inline EnableIf_<VectorizedAssign<MT> > assign( const DenseMatrix<MT,SO>& rhs );

   template< typename MT > inline void assign( const DenseMatrix<MT,!SO>&  rhs );
   template< typename MT > inline void assign( const SparseMatrix<MT,SO>&  rhs );
   template< typename MT > inline void assign( const SparseMatrix<MT,!SO>& rhs );

   template< typename MT >
   inline DisableIf_<VectorizedAddAssign<MT> > addAssign( const DenseMatrix<MT,SO>& rhs );

   template< typename MT >
   inline EnableIf_<VectorizedAddAssign<MT> > addAssign( const DenseMatrix<MT,SO>& rhs );

   template< typename MT > inline void addAssign( const DenseMatrix<MT,!SO>&  rhs );
   template< typename MT > inline void addAssign( const SparseMatrix<MT,SO>&  rhs );
   template< typename MT > inline void addAssign( const SparseMatrix<MT,!SO>& rhs );

   template< typename MT >
   inline DisableIf_<VectorizedSubAssign<MT> > subAssign( const DenseMatrix<MT,SO>& rhs );

   template< typename MT >
   inline EnableIf_<VectorizedSubAssign<MT> > subAssign( const DenseMatrix<MT,SO>& rhs );

   template< typename MT > inline void subAssign( const DenseMatrix<MT,!SO>&  rhs );
   template< typename MT > inline void subAssign( const SparseMatrix<MT,SO>&  rhs );
   template< typename MT > inline void subAssign( const SparseMatrix<MT,!SO>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t m_;                     //!< The current number of rows of the matrix.
   size_t n_;                     //!< The current number of columns of the matrix.
   size_t nn_;                    //!< The number of elements between two rows.
   boost::shared_array<Type> v_;  //!< The dynamically allocated matrix elements.
                                  /*!< Access to the matrix elements is gained via the function call
                                       operator. In case of row-major order the memory layout of the
                                       elements is
                                       \f[\left(\begin{array}{*{5}{c}}
                                       0            & 1             & 2             & \cdots & N-1         \\
                                       N            & N+1           & N+2           & \cdots & 2 \cdot N-1 \\
                                       \vdots       & \vdots        & \vdots        & \ddots & \vdots      \\
                                       M \cdot N-N  & M \cdot N-N+1 & M \cdot N-N+2 & \cdots & M \cdot N-1 \\
                                       \end{array}\right)\f]. */
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
/*!\brief The default constructor for CustomMatrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline CustomMatrix<Type,AF,PF,SO>::CustomMatrix()
   : m_ ( 0UL )  // The current number of rows of the matrix
   , n_ ( 0UL )  // The current number of columns of the matrix
   , nn_( 0UL )  // The number of elements between two rows
   , v_ (     )  // The matrix elements
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a matrix of size \f$ m \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This constructor creates an unpadded custom matrix of size \f$ m \times n \f$. The construction
// fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This constructor is \b NOT available for padded custom matrices!
// \note The custom matrix does \b NOT take responsibility for the given array of elements!
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline CustomMatrix<Type,AF,PF,SO>::CustomMatrix( Type* ptr, size_t m, size_t n )
   : m_ ( m )  // The current number of rows of the matrix
   , n_ ( n )  // The current number of columns of the matrix
   , nn_( n )  // The number of elements between two rows
   , v_ (   )  // The matrix elements
{
   BLAZE_STATIC_ASSERT( PF == unpadded );

   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && ( !checkAlignment( ptr ) || nn_ % SIMDSIZE != 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   v_.reset( ptr, NoDelete() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a matrix of size \f$ m \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \param nn The total number of elements between two rows/columns.
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This constructor creates a custom matrix of size \f$ m \times n \f$. The construction fails
// if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified spacing \a nn is insufficient for the given data type \a Type and the
//    available instruction set.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note The custom matrix does \b NOT take responsibility for the given array of elements!
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline CustomMatrix<Type,AF,PF,SO>::CustomMatrix( Type* ptr, size_t m, size_t n, size_t nn )
   : m_ ( m  )  // The current number of rows of the matrix
   , n_ ( n  )  // The current number of columns of the matrix
   , nn_( nn )  // The number of elements between two rows
   , v_ (    )  // The matrix elements
{
   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && ( !checkAlignment( ptr ) || nn_ % SIMDSIZE != 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   if( PF && IsVectorizable<Type>::value && ( nn_ < nextMultiple<size_t>( n_, SIMDSIZE ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Insufficient capacity for padded matrix" );
   }

   v_.reset( ptr, NoDelete() );

   if( PF && IsVectorizable<Type>::value ) {
      for( size_t i=0UL; i<m_; ++i ) {
         for( size_t j=n_; j<nn_; ++j )
            v_[i*nn_+j] = Type();
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a matrix of size \f$ m \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \param d The deleter to destroy the array of elements.
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This constructor creates an unpadded custom matrix of size \f$ m \times n \f$. The construction
// fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This constructor is \b NOT available for padded custom matrices!
*/
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool SO >         // Storage order
template< typename Deleter  // Type of the custom deleter
        , typename >        // Type restriction on the custom deleter
inline CustomMatrix<Type,AF,PF,SO>::CustomMatrix( Type* ptr, size_t m, size_t n, Deleter d )
   : m_ ( m )  // The current number of rows of the matrix
   , n_ ( n )  // The current number of columns of the matrix
   , nn_( n )  // The number of elements between two rows
   , v_ (   )  // The matrix elements
{
   BLAZE_STATIC_ASSERT( PF == unpadded );

   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && ( !checkAlignment( ptr ) || nn_ % SIMDSIZE != 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   v_.reset( ptr, d );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a matrix of size \f$ m \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \param nn The total number of elements between two rows/columns.
// \param d The deleter to destroy the array of elements.
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This constructor creates a custom matrix of size \f$ m \times n \f$. The construction fails
// if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified spacing \a nn is insufficient for the given data type \a Type and the
//    available instruction set.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
*/
template< typename Type       // Data type of the matrix
        , bool AF             // Alignment flag
        , bool PF             // Padding flag
        , bool SO >           // Storage order
template< typename Deleter >  // Type of the custom deleter
inline CustomMatrix<Type,AF,PF,SO>::CustomMatrix( Type* ptr, size_t m, size_t n, size_t nn, Deleter d )
   : m_ ( m  )  // The current number of rows of the matrix
   , n_ ( n  )  // The current number of columns of the matrix
   , nn_( nn )  // The number of elements between two rows
   , v_ (    )  // The matrix elements
{
   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && ( !checkAlignment( ptr ) || nn_ % SIMDSIZE != 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   if( PF && IsVectorizable<Type>::value && ( nn_ < nextMultiple<size_t>( n_, SIMDSIZE ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Insufficient capacity for padded matrix" );
   }

   v_.reset( ptr, d );

   if( PF && IsVectorizable<Type>::value ) {
      for( size_t i=0UL; i<m_; ++i ) {
         for( size_t j=n_; j<nn_; ++j )
            v_[i*nn_+j] = Type();
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for CustomMatrix.
//
// \param m Matrix to be copied.
//
// The copy constructor initializes the custom matrix as an exact copy of the given custom matrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline CustomMatrix<Type,AF,PF,SO>::CustomMatrix( const CustomMatrix& m )
   : m_ ( m.m_  )  // The current number of rows of the matrix
   , n_ ( m.n_  )  // The current number of columns of the matrix
   , nn_( m.nn_ )  // The number of elements between two rows
   , v_ ( m.v_  )  // The matrix elements
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The move constructor for CustomMatrix.
//
// \param m The matrix to be moved into this instance.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline CustomMatrix<Type,AF,PF,SO>::CustomMatrix( CustomMatrix&& m ) noexcept
   : m_ ( m.m_  )              // The current number of rows of the matrix
   , n_ ( m.n_  )              // The current number of columns of the matrix
   , nn_( m.nn_ )              // The number of elements between two rows
   , v_ ( std::move( m.v_ ) )  // The matrix elements
{
   m.m_  = 0UL;
   m.n_  = 0UL;
   m.nn_ = 0UL;

   BLAZE_INTERNAL_ASSERT( m.data() == nullptr, "Invalid data reference detected" );
}
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief 2D-access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::Reference
   CustomMatrix<Type,AF,PF,SO>::operator()( size_t i, size_t j ) noexcept
{
   BLAZE_USER_ASSERT( i<m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<n_, "Invalid column access index" );
   return v_[i*nn_+j];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief 2D-access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::ConstReference
   CustomMatrix<Type,AF,PF,SO>::operator()( size_t i, size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( i<m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<n_, "Invalid column access index" );
   return v_[i*nn_+j];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::Reference
   CustomMatrix<Type,AF,PF,SO>::at( size_t i, size_t j )
{
   if( i >= m_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= n_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i,j);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::ConstReference
   CustomMatrix<Type,AF,PF,SO>::at( size_t i, size_t j ) const
{
   if( i >= m_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= n_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i,j);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the matrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The dynamic matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::Pointer
   CustomMatrix<Type,AF,PF,SO>::data() noexcept
{
   return v_.get();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the matrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The dynamic matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::ConstPointer
   CustomMatrix<Type,AF,PF,SO>::data() const noexcept
{
   return v_.get();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the matrix elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::Pointer
   CustomMatrix<Type,AF,PF,SO>::data( size_t i ) noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return v_.get() + i*nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the matrix elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::ConstPointer
   CustomMatrix<Type,AF,PF,SO>::data( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return v_.get() + i*nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::Iterator
   CustomMatrix<Type,AF,PF,SO>::begin( size_t i ) noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return Iterator( v_.get() + i*nn_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::ConstIterator
   CustomMatrix<Type,AF,PF,SO>::begin( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return ConstIterator( v_.get() + i*nn_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::ConstIterator
   CustomMatrix<Type,AF,PF,SO>::cbegin( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return ConstIterator( v_.get() + i*nn_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::Iterator
   CustomMatrix<Type,AF,PF,SO>::end( size_t i ) noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return Iterator( v_.get() + i*nn_ + n_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::ConstIterator
   CustomMatrix<Type,AF,PF,SO>::end( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return ConstIterator( v_.get() + i*nn_ + n_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline typename CustomMatrix<Type,AF,PF,SO>::ConstIterator
   CustomMatrix<Type,AF,PF,SO>::cend( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return ConstIterator( v_.get() + i*nn_ + n_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Homogenous assignment to all matrix elements.
//
// \param rhs Scalar value to be assigned to all matrix elements.
// \return Reference to the assigned matrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline CustomMatrix<Type,AF,PF,SO>& CustomMatrix<Type,AF,PF,SO>::operator=( const Type& rhs )
{
   for( size_t i=0UL; i<m_; ++i )
      for( size_t j=0UL; j<n_; ++j )
         v_[i*nn_+j] = rhs;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List assignment to all matrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to static matrix.
//
// This assignment operator offers the option to directly assign to all elements of the matrix
// by means of an initializer list:

   \code
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;

   const int array[9] = { 0, 0, 0,
                          0, 0, 0,
                          0, 0, 0 };
   blaze::CustomMatrix<int,unaligned,unpadded,rowMajor> A( array, 3UL, 3UL );
   A = { { 1, 2, 3 },
         { 4, 5 },
         { 7, 8, 9 } };
   \endcode

// The matrix elements are assigned the values from the given initializer list. Missing values
// are initialized as default (as e.g. the value 6 in the example). Note that in case the size
// of the top-level initializer list exceeds the number of rows or the size of any nested list
// exceeds the number of columns, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline CustomMatrix<Type,AF,PF,SO>&
   CustomMatrix<Type,AF,PF,SO>::operator=( initializer_list< initializer_list<Type> > list )
{
   if( list.size() != m_ || determineColumns( list ) > n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to custom matrix" );
   }

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      std::fill( std::copy( rowList.begin(), rowList.end(), v_.get()+i*nn_ ),
                 v_.get()+i*nn_+( PF ? nn_ : n_ ), Type() );
      ++i;
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array assignment to all matrix elements.
//
// \param array \f$ M \times N \f$ dimensional array for the assignment.
// \return Reference to the assigned matrix.
// \exception std::invalid_argument Invalid array size.
//
// This assignment operator offers the option to directly set all elements of the matrix:

   \code
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;

   const int array[9] = { 0, 0, 0,
                          0, 0, 0,
                          0, 0, 0 };
   const int init[3][3] = { { 1, 2, 3 },
                            { 4, 5 },
                            { 7, 8, 9 } };
   blaze::CustomMatrix<int,unaligned,unpadded,rowMajor> A( array, 3UL, 3UL );
   A = init;
   \endcode

// The matrix is assigned the values from the given array. Missing values are initialized with
// default values (as e.g. the value 6 in the example). Note that the size of the array must
// match the size of the custom matrix. Otherwise a \a std::invalid_argument exception is thrown.
// Also note that after the assignment \a array will have the same entries as \a init.
*/
template< typename Type   // Data type of the matrix
        , bool AF         // Alignment flag
        , bool PF         // Padding flag
        , bool SO >       // Storage order
template< typename Other  // Data type of the initialization array
        , size_t M        // Number of rows of the initialization array
        , size_t N >      // Number of columns of the initialization array
inline CustomMatrix<Type,AF,PF,SO>& CustomMatrix<Type,AF,PF,SO>::operator=( const Other (&array)[M][N] )
{
   if( m_ != M || n_ != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array size" );
   }

   for( size_t i=0UL; i<M; ++i )
      for( size_t j=0UL; j<N; ++j )
         v_[i*nn_+j] = array[i][j];

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Copy assignment operator for CustomMatrix.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// The matrix is initialized as a copy of the given matrix. In case the current sizes of the two
// matrices don't match, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline CustomMatrix<Type,AF,PF,SO>& CustomMatrix<Type,AF,PF,SO>::operator=( const CustomMatrix& rhs )
{
   if( rhs.rows() != m_ || rhs.columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   smpAssign( *this, ~rhs );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Move assignment operator for CustomMatrix.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline CustomMatrix<Type,AF,PF,SO>&
   CustomMatrix<Type,AF,PF,SO>::operator=( CustomMatrix&& rhs ) noexcept
{
   m_  = rhs.m_;
   n_  = rhs.n_;
   nn_ = rhs.nn_;
   v_  = std::move( rhs.v_ );

   rhs.m_  = 0UL;
   rhs.n_  = 0UL;
   rhs.nn_ = 0UL;

   BLAZE_INTERNAL_ASSERT( rhs.data() == nullptr, "Invalid data reference detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different matrices.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// The matrix is initialized as a copy of the given matrix. In case the current sizes of the two
// matrices don't match, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline CustomMatrix<Type,AF,PF,SO>& CustomMatrix<Type,AF,PF,SO>::operator=( const Matrix<MT,SO2>& rhs )
{
   typedef TransExprTrait_<This>   TT;
   typedef CTransExprTrait_<This>  CT;
   typedef InvExprTrait_<This>     IT;

   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( IsSame<MT,TT>::value && (~rhs).isAliased( this ) ) {
      transpose();
   }
   else if( IsSame<MT,CT>::value && (~rhs).isAliased( this ) ) {
      ctranspose();
   }
   else if( !IsSame<MT,IT>::value && (~rhs).canAlias( this ) ) {
      const ResultType_<MT> tmp( ~rhs );
      smpAssign( *this, tmp );
   }
   else {
      if( IsSparseMatrix<MT>::value )
         reset();
      smpAssign( *this, ~rhs );
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the matrix.
// \return Reference to the matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline CustomMatrix<Type,AF,PF,SO>& CustomMatrix<Type,AF,PF,SO>::operator+=( const Matrix<MT,SO2>& rhs )
{
   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_<MT> tmp( ~rhs );
      smpAddAssign( *this, tmp );
   }
   else {
      smpAddAssign( *this, ~rhs );
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the matrix.
// \return Reference to the matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline CustomMatrix<Type,AF,PF,SO>& CustomMatrix<Type,AF,PF,SO>::operator-=( const Matrix<MT,SO2>& rhs )
{
   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_<MT> tmp( ~rhs );
      smpSubAssign( *this, tmp );
   }
   else {
      smpSubAssign( *this, ~rhs );
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication of a matrix (\f$ A*=B \f$).
//
// \param rhs The right-hand side matrix for the multiplication.
// \return Reference to the matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// In case the current sizes of the two given matrices don't match, a \a std::invalid_argument
// is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline CustomMatrix<Type,AF,PF,SO>& CustomMatrix<Type,AF,PF,SO>::operator*=( const Matrix<MT,SO2>& rhs )
{
   if( m_ != n_ || (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const MultTrait_< ResultType, ResultType_<MT> > tmp( *this * (~rhs) );
   smpAssign( *this, tmp );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication between a matrix and
//        a scalar value (\f$ A*=s \f$).
//
// \param rhs The right-hand side scalar value for the multiplication.
// \return Reference to the matrix.
*/
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool SO >         // Storage order
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, CustomMatrix<Type,AF,PF,SO> >&
   CustomMatrix<Type,AF,PF,SO>::operator*=( Other rhs )
{
   smpAssign( *this, (*this) * rhs );
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment operator for the division of a matrix by a scalar value
//        (\f$ A/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the matrix.
*/
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool SO >         // Storage order
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, CustomMatrix<Type,AF,PF,SO> >&
   CustomMatrix<Type,AF,PF,SO>::operator/=( Other rhs )
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
/*!\brief Returns the current number of rows of the matrix.
//
// \return The number of rows of the matrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline size_t CustomMatrix<Type,AF,PF,SO>::rows() const noexcept
{
   return m_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of columns of the matrix.
//
// \return The number of columns of the matrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline size_t CustomMatrix<Type,AF,PF,SO>::columns() const noexcept
{
   return n_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the spacing between the beginning of two rows/columns.
//
// \return The spacing between the beginning of two rows/columns.
//
// This function returns the spacing between the beginning of two rows/columns, i.e. the
// total number of elements of a row/column. In case the storage order is set to \a rowMajor
// the function returns the spacing between two rows, in case the storage flag is set to
// \a columnMajor the function returns the spacing between two columns.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline size_t CustomMatrix<Type,AF,PF,SO>::spacing() const noexcept
{
   return nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the matrix.
//
// \return The capacity of the matrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline size_t CustomMatrix<Type,AF,PF,SO>::capacity() const noexcept
{
   return m_ * nn_;
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline size_t CustomMatrix<Type,AF,PF,SO>::capacity( size_t i ) const noexcept
{
   UNUSED_PARAMETER( i );
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   return nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of non-zero elements in the matrix
//
// \return The number of non-zero elements in the dense matrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline size_t CustomMatrix<Type,AF,PF,SO>::nonZeros() const
{
   size_t nonzeros( 0UL );

   for( size_t i=0UL; i<m_; ++i )
      for( size_t j=0UL; j<n_; ++j )
         if( !isDefault( v_[i*nn_+j] ) )
            ++nonzeros;

   return nonzeros;
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline size_t CustomMatrix<Type,AF,PF,SO>::nonZeros( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   const size_t jend( i*nn_ + n_ );
   size_t nonzeros( 0UL );

   for( size_t j=i*nn_; j<jend; ++j )
      if( !isDefault( v_[j] ) )
         ++nonzeros;

   return nonzeros;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline void CustomMatrix<Type,AF,PF,SO>::reset()
{
   using blaze::clear;

   for( size_t i=0UL; i<m_; ++i )
      for( size_t j=0UL; j<n_; ++j )
         clear( v_[i*nn_+j] );
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline void CustomMatrix<Type,AF,PF,SO>::reset( size_t i )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   for( size_t j=0UL; j<n_; ++j )
      clear( v_[i*nn_+j] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the \f$ M \times N \f$ matrix.
//
// \return void
//
// After the clear() function, the size of the matrix is 0.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline void CustomMatrix<Type,AF,PF,SO>::clear()
{
   m_  = 0UL;
   n_  = 0UL;
   nn_ = 0UL;
   v_.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the matrix.
//
// \return Reference to the transposed matrix.
// \exception std::logic_error Impossible transpose operation.
//
// In case the matrix is not a square matrix, a \a std::logic_error exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline CustomMatrix<Type,AF,PF,SO>& CustomMatrix<Type,AF,PF,SO>::transpose()
{
   using std::swap;

   if( m_ != n_ ) {
      BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
   }

   for( size_t i=1UL; i<m_; ++i )
      for( size_t j=0UL; j<i; ++j )
         swap( v_[i*nn_+j], v_[j*nn_+i] );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the matrix.
//
// \return Reference to the transposed matrix.
// \exception std::logic_error Impossible transpose operation.
//
// In case the matrix is not a square matrix, a \a std::logic_error exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline CustomMatrix<Type,AF,PF,SO>& CustomMatrix<Type,AF,PF,SO>::ctranspose()
{
   if( m_ != n_ ) {
      BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
   }

   for( size_t i=0UL; i<m_; ++i ) {
      for( size_t j=0UL; j<i; ++j ) {
         cswap( v_[i*nn_+j], v_[j*nn_+i] );
      }
      conjugate( v_[i*nn_+i] );
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the matrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the matrix scaling.
// \return Reference to the matrix.
*/
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool SO >         // Storage order
template< typename Other >  // Data type of the scalar value
inline CustomMatrix<Type,AF,PF,SO>& CustomMatrix<Type,AF,PF,SO>::scale( const Other& scalar )
{
   for( size_t i=0UL; i<m_; ++i )
      for( size_t j=0UL; j<n_; ++j )
         v_[i*nn_+j] *= scalar;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two matrices.
//
// \param m The matrix to be swapped.
// \return void
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline void CustomMatrix<Type,AF,PF,SO>::swap( CustomMatrix& m ) noexcept
{
   using std::swap;

   swap( m_ , m.m_  );
   swap( n_ , m.n_  );
   swap( nn_, m.nn_ );
   swap( v_ , m.v_  );
}
//*************************************************************************************************




//=================================================================================================
//
//  RESOURCE MANAGEMENT FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Resets the custom matrix and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \return void
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This function resets the custom matrix to the given array of elements of size \f$ m \times n \f$.
// The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This function is \b NOT available for padded custom matrices!
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom matrix referencing the array goes out of scope.
// \note The custom matrix does NOT take responsibility for the new array of elements!
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline void CustomMatrix<Type,AF,PF,SO>::reset( Type* ptr, size_t m, size_t n )
{
   BLAZE_STATIC_ASSERT( PF == unpadded );

   CustomMatrix tmp( ptr, m, n );
   swap( tmp );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resets the custom matrix and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \param nn The total number of elements between two rows/columns.
// \return void
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This function resets the custom matrix to the given array of elements of size \f$ m \times n \f$.
// The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom matrix referencing the array goes out of scope.
// \note The custom matrix does NOT take responsibility for the new array of elements!
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline void CustomMatrix<Type,AF,PF,SO>::reset( Type* ptr, size_t m, size_t n, size_t nn )
{
   CustomMatrix tmp( ptr, m, n, nn );
   swap( tmp );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resets the custom matrix and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \param d The deleter to destroy the array of elements.
// \return void
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This function resets the custom matrix to the given array of elements of size \f$ m \times n \f$.
// The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This function is \b NOT available for padded custom matrices!
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom matrix referencing the array goes out of scope.
*/
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool SO >         // Storage order
template< typename Deleter  // Type of the custom deleter
        , typename >        // Type restriction on the custom deleter
inline void CustomMatrix<Type,AF,PF,SO>::reset( Type* ptr, size_t m, size_t n, Deleter d )
{
   BLAZE_STATIC_ASSERT( !IsClass<Deleter>::value || PF == unpadded );

   CustomMatrix tmp( ptr, m, n, d );
   swap( tmp );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resets the custom matrix and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \param nn The total number of elements between two rows/columns.
// \param d The deleter to destroy the array of elements.
// \return void
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This function resets the custom matrix to the given array of elements of size \f$ m \times n \f$.
// The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom matrix referencing the array goes out of scope.
*/
template< typename Type       // Data type of the matrix
        , bool AF             // Alignment flag
        , bool PF             // Padding flag
        , bool SO >           // Storage order
template< typename Deleter >  // Type of the custom deleter
inline void CustomMatrix<Type,AF,PF,SO>::reset( Type* ptr, size_t m, size_t n, size_t nn, Deleter d )
{
   CustomMatrix tmp( ptr, m, n, nn, d );
   swap( tmp );
}
//*************************************************************************************************




//=================================================================================================
//
//  EXPRESSION TEMPLATE EVALUATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the matrix can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this matrix, \a false if not.
//
// This function returns whether the given address can alias with the matrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool SO >         // Storage order
template< typename Other >  // Data type of the foreign expression
inline bool CustomMatrix<Type,AF,PF,SO>::canAlias( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the matrix is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this matrix, \a false if not.
//
// This function returns whether the given address is aliased with the matrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , bool SO >         // Storage order
template< typename Other >  // Data type of the foreign expression
inline bool CustomMatrix<Type,AF,PF,SO>::isAliased( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the matrix is properly aligned in memory.
//
// \return \a true in case the matrix is aligned, \a false if not.
//
// This function returns whether the matrix is guaranteed to be properly aligned in memory, i.e.
// whether the beginning and the end of each row/column of the matrix are guaranteed to conform
// to the alignment restrictions of the element type \a Type.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline bool CustomMatrix<Type,AF,PF,SO>::isAligned() const noexcept
{
   return ( AF || ( checkAlignment( v_.get() ) && columns() % SIMDSIZE == 0UL ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the matrix can be used in SMP assignments.
//
// \return \a true in case the matrix can be used in SMP assignments, \a false if not.
//
// This function returns whether the matrix can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the matrix).
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline bool CustomMatrix<Type,AF,PF,SO>::canSMPAssign() const noexcept
{
   return ( rows() > SMP_DMATASSIGN_THRESHOLD );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Load of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense matrix. The row index
// must be smaller than the number of rows and the column index must be smaller then the number
// of columns. Additionally, the column index (in case of a row-major matrix) or the row index
// (in case of a column-major matrix) must be a multiple of the number of values inside the
// SIMD element. This function must \b NOT be called explicitly! It is used internally for the
// performance optimized evaluation of expression templates. Calling this function explicitly
// might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE typename CustomMatrix<Type,AF,PF,SO>::SIMDType
   CustomMatrix<Type,AF,PF,SO>::load( size_t i, size_t j ) const noexcept
{
   if( AF && PF )
      return loada( i, j );
   else
      return loadu( i, j );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned load of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense matrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major matrix)
// or the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE typename CustomMatrix<Type,AF,PF,SO>::SIMDType
   CustomMatrix<Type,AF,PF,SO>::loada( size_t i, size_t j ) const noexcept
{
   using blaze::loada;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !PF || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_.get()+i*nn_+j ), "Invalid alignment detected" );

   return loada( v_.get()+i*nn_+j );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Unaligned load of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense matrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major matrix)
// or the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE typename CustomMatrix<Type,AF,PF,SO>::SIMDType
   CustomMatrix<Type,AF,PF,SO>::loadu( size_t i, size_t j ) const noexcept
{
   using blaze::loadu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );

   return loadu( v_.get()+i*nn_+j );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Store of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store of a specific SIMD element of the dense matrix. The row index
// must be smaller than the number of rows and the column index must be smaller than the number
// of columns. Additionally, the column index (in case of a row-major matrix) or the row index
// (in case of a column-major matrix) must be a multiple of the number of values inside the
// SIMD element. This function must \b NOT be called explicitly! It is used internally for the
// performance optimized evaluation of expression templates. Calling this function explicitly
// might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE void
   CustomMatrix<Type,AF,PF,SO>::store( size_t i, size_t j, const SIMDType& value ) noexcept
{
   if( AF && PF )
      storea( i, j, value );
   else
      storeu( i, j, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned store of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store of a specific SIMD element of the dense matrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major matrix)
// or the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE void
   CustomMatrix<Type,AF,PF,SO>::storea( size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::storea;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !PF || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_.get()+i*nn_+j ), "Invalid alignment detected" );

   storea( v_.get()+i*nn_+j, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Unaligned store of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store of a specific SIMD element of the dense matrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major matrix)
// or the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE void
   CustomMatrix<Type,AF,PF,SO>::storeu( size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::storeu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );

   storeu( v_.get()+i*nn_+j, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned, non-temporal store of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store of a specific SIMD element of the
// dense matrix. The row index must be smaller than the number of rows and the column index
// must be smaller than the number of columns. Additionally, the column index (in case of a
// row-major matrix) or the row index (in case of a column-major matrix) must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE void
   CustomMatrix<Type,AF,PF,SO>::stream( size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::stream;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !PF || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_.get()+i*nn_+j ), "Invalid alignment detected" );

   stream( v_.get()+i*nn_+j, value );
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline DisableIf_<typename CustomMatrix<Type,AF,PF,SO>::BLAZE_TEMPLATE VectorizedAssign<MT> >
   CustomMatrix<Type,AF,PF,SO>::assign( const DenseMatrix<MT,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( n_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<m_; ++i ) {
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         v_[i*nn_+j    ] = (~rhs)(i,j    );
         v_[i*nn_+j+1UL] = (~rhs)(i,j+1UL);
      }
      if( jpos < n_ ) {
         v_[i*nn_+jpos] = (~rhs)(i,jpos);
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline EnableIf_<typename CustomMatrix<Type,AF,PF,SO>::BLAZE_TEMPLATE VectorizedAssign<MT> >
   CustomMatrix<Type,AF,PF,SO>::assign( const DenseMatrix<MT,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const bool remainder( !PF || !IsPadded<MT>::value );

   const size_t jpos( ( remainder )?( n_ & size_t(-SIMDSIZE) ):( n_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( n_ - ( n_ % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

   if( AF && PF && useStreaming &&
       ( m_*n_ > ( cacheSize / ( sizeof(Type) * 3UL ) ) ) && !(~rhs).isAliased( this ) )
   {
      for( size_t i=0UL; i<m_; ++i )
      {
         size_t j( 0UL );

         for( ; j<jpos; j+=SIMDSIZE ) {
            stream( i, j, (~rhs).load(i,j) );
         }
         for( ; remainder && j<n_; ++j ) {
            v_[i*nn_+j] = (~rhs)(i,j);
         }
      }
   }
   else
   {
      for( size_t i=0UL; i<m_; ++i )
      {
         size_t j( 0UL );
         ConstIterator_<MT> it( (~rhs).begin(i) );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            store( i, j             , it.load() ); it += SIMDSIZE;
            store( i, j+SIMDSIZE    , it.load() ); it += SIMDSIZE;
            store( i, j+SIMDSIZE*2UL, it.load() ); it += SIMDSIZE;
            store( i, j+SIMDSIZE*3UL, it.load() ); it += SIMDSIZE;
         }
         for( ; j<jpos; j+=SIMDSIZE, it+=SIMDSIZE ) {
            store( i, j, it.load() );
         }
         for( ; remainder && j<n_; ++j, ++it ) {
            v_[i*nn_+j] = *it;
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline void CustomMatrix<Type,AF,PF,SO>::assign( const DenseMatrix<MT,!SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<m_; ii+=block ) {
      const size_t iend( min( m_, ii+block ) );
      for( size_t jj=0UL; jj<n_; jj+=block ) {
         const size_t jend( min( n_, jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               v_[i*nn_+j] = (~rhs)(i,j);
            }
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline void CustomMatrix<Type,AF,PF,SO>::assign( const SparseMatrix<MT,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i*nn_+element->index()] = element->value();
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline void CustomMatrix<Type,AF,PF,SO>::assign( const SparseMatrix<MT,!SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()*nn_+j] = element->value();
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline DisableIf_<typename CustomMatrix<Type,AF,PF,SO>::BLAZE_TEMPLATE VectorizedAddAssign<MT> >
   CustomMatrix<Type,AF,PF,SO>::addAssign( const DenseMatrix<MT,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
   {
      if( IsDiagonal<MT>::value )
      {
         v_[i*nn_+i] += (~rhs)(i,i);
      }
      else
      {
         const size_t jbegin( ( IsUpper<MT>::value )
                              ?( IsStrictlyUpper<MT>::value ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend  ( ( IsLower<MT>::value )
                              ?( IsStrictlyLower<MT>::value ? i : i+1UL )
                              :( n_ ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         size_t j( jbegin );

         for( ; (j+2UL) <= jend; j+=2UL ) {
            v_[i*nn_+j    ] += (~rhs)(i,j    );
            v_[i*nn_+j+1UL] += (~rhs)(i,j+1UL);
         }
         if( j < jend ) {
            v_[i*nn_+j] += (~rhs)(i,j);
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline EnableIf_<typename CustomMatrix<Type,AF,PF,SO>::BLAZE_TEMPLATE VectorizedAddAssign<MT> >
   CustomMatrix<Type,AF,PF,SO>::addAssign( const DenseMatrix<MT,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_DIAGONAL_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const bool remainder( !PF || !IsPadded<MT>::value );

   for( size_t i=0UL; i<m_; ++i )
   {
      const size_t jbegin( ( IsUpper<MT>::value )
                           ?( ( IsStrictlyUpper<MT>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                           :( 0UL ) );
      const size_t jend  ( ( IsLower<MT>::value )
                           ?( IsStrictlyLower<MT>::value ? i : i+1UL )
                           :( n_ ) );
      BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

      const size_t jpos( ( remainder )?( jend & size_t(-SIMDSIZE) ):( jend ) );
      BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

      size_t j( jbegin );
      ConstIterator_<MT> it( (~rhs).begin(i) + jbegin );

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
         store( i, j             , load(i,j             ) + it.load() ); it += SIMDSIZE;
         store( i, j+SIMDSIZE    , load(i,j+SIMDSIZE    ) + it.load() ); it += SIMDSIZE;
         store( i, j+SIMDSIZE*2UL, load(i,j+SIMDSIZE*2UL) + it.load() ); it += SIMDSIZE;
         store( i, j+SIMDSIZE*3UL, load(i,j+SIMDSIZE*3UL) + it.load() ); it += SIMDSIZE;
      }
      for( ; j<jpos; j+=SIMDSIZE, it+=SIMDSIZE ) {
         store( i, j, load(i,j) + it.load() );
      }
      for( ; remainder && j<jend; ++j, ++it ) {
         v_[i*nn_+j] += *it;
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline void CustomMatrix<Type,AF,PF,SO>::addAssign( const DenseMatrix<MT,!SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<m_; ii+=block ) {
      const size_t iend( min( m_, ii+block ) );
      for( size_t jj=0UL; jj<n_; jj+=block )
      {
         if( IsLower<MT>::value && ii < jj ) break;
         if( IsUpper<MT>::value && ii > jj ) continue;

         for( size_t i=ii; i<iend; ++i )
         {
            const size_t jbegin( ( IsUpper<MT>::value )
                                 ?( max( ( IsStrictlyUpper<MT>::value ? i+1UL : i ), jj ) )
                                 :( jj ) );
            const size_t jend  ( ( IsLower<MT>::value )
                                 ?( min( ( IsStrictlyLower<MT>::value ? i : i+1UL ), n_, jj+block ) )
                                 :( min( n_, jj+block ) ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            for( size_t j=jbegin; j<jend; ++j ) {
               v_[i*nn_+j] += (~rhs)(i,j);
            }
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline void CustomMatrix<Type,AF,PF,SO>::addAssign( const SparseMatrix<MT,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i*nn_+element->index()] += element->value();
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline void CustomMatrix<Type,AF,PF,SO>::addAssign( const SparseMatrix<MT,!SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()*nn_+j] += element->value();
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline DisableIf_<typename CustomMatrix<Type,AF,PF,SO>::BLAZE_TEMPLATE VectorizedSubAssign<MT> >
   CustomMatrix<Type,AF,PF,SO>::subAssign( const DenseMatrix<MT,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
   {
      if( IsDiagonal<MT>::value )
      {
         v_[i*nn_+i] -= (~rhs)(i,i);
      }
      else
      {
         const size_t jbegin( ( IsUpper<MT>::value )
                              ?( IsStrictlyUpper<MT>::value ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend  ( ( IsLower<MT>::value )
                              ?( IsStrictlyLower<MT>::value ? i : i+1UL )
                              :( n_ ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         size_t j( jbegin );

         for( ; (j+2UL) <= jend; j+=2UL ) {
            v_[i*nn_+j    ] -= (~rhs)(i,j    );
            v_[i*nn_+j+1UL] -= (~rhs)(i,j+1UL);
         }
         if( j < jend ) {
            v_[i*nn_+j] -= (~rhs)(i,j);
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline EnableIf_<typename CustomMatrix<Type,AF,PF,SO>::BLAZE_TEMPLATE VectorizedSubAssign<MT> >
   CustomMatrix<Type,AF,PF,SO>::subAssign( const DenseMatrix<MT,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_DIAGONAL_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const bool remainder( !PF || !IsPadded<MT>::value );

   for( size_t i=0UL; i<m_; ++i )
   {
      const size_t jbegin( ( IsUpper<MT>::value )
                           ?( ( IsStrictlyUpper<MT>::value ? i+1UL : i ) & size_t(-SIMDSIZE) )
                           :( 0UL ) );
      const size_t jend  ( ( IsLower<MT>::value )
                           ?( IsStrictlyLower<MT>::value ? i : i+1UL )
                           :( n_ ) );
      BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

      const size_t jpos( ( remainder )?( jend & size_t(-SIMDSIZE) ):( jend ) );
      BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

      size_t j( jbegin );
      ConstIterator_<MT> it( (~rhs).begin(i) + jbegin );

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
         store( i, j             , load(i,j             ) - it.load() ); it += SIMDSIZE;
         store( i, j+SIMDSIZE    , load(i,j+SIMDSIZE    ) - it.load() ); it += SIMDSIZE;
         store( i, j+SIMDSIZE*2UL, load(i,j+SIMDSIZE*2UL) - it.load() ); it += SIMDSIZE;
         store( i, j+SIMDSIZE*3UL, load(i,j+SIMDSIZE*3UL) - it.load() ); it += SIMDSIZE;
      }
      for( ; j<jpos; j+=SIMDSIZE, it+=SIMDSIZE ) {
         store( i, j, load(i,j) - it.load() );
      }
      for( ; remainder && j<jend; ++j, ++it ) {
         v_[i*nn_+j] -= *it;
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline void CustomMatrix<Type,AF,PF,SO>::subAssign( const DenseMatrix<MT,!SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<m_; ii+=block ) {
      const size_t iend( min( m_, ii+block ) );
      for( size_t jj=0UL; jj<n_; jj+=block )
      {
         if( IsLower<MT>::value && ii < jj ) break;
         if( IsUpper<MT>::value && ii > jj ) continue;

         for( size_t i=ii; i<iend; ++i )
         {
            const size_t jbegin( ( IsUpper<MT>::value )
                                 ?( max( ( IsStrictlyUpper<MT>::value ? i+1UL : i ), jj ) )
                                 :( jj ) );
            const size_t jend  ( ( IsLower<MT>::value )
                                 ?( min( ( IsStrictlyLower<MT>::value ? i : i+1UL ), n_, jj+block ) )
                                 :( min( n_, jj+block ) ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            for( size_t j=jbegin; j<jend; ++j ) {
               v_[i*nn_+j] -= (~rhs)(i,j);
            }
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline void CustomMatrix<Type,AF,PF,SO>::subAssign( const SparseMatrix<MT,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i*nn_+element->index()] -= element->value();
}
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline void CustomMatrix<Type,AF,PF,SO>::subAssign( const SparseMatrix<MT,!SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()*nn_+j] -= element->value();
}
//*************************************************************************************************








//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR COLUMN-MAJOR MATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of CustomMatrix for column-major matrices.
// \ingroup custom_matrix
//
// This specialization of CustomMatrix adapts the class template to the requirements of
// column-major matrices.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
class CustomMatrix<Type,AF,PF,true> : public DenseMatrix< CustomMatrix<Type,AF,PF,true>, true >
{
 public:
   //**Type definitions****************************************************************************
   typedef CustomMatrix<Type,AF,PF,true>  This;           //!< Type of this CustomMatrix instance.
   typedef DenseMatrix<This,true>         BaseType;       //!< Base type of this CustomMatrix instance.
   typedef DynamicMatrix<Type,true>       ResultType;     //!< Result type for expression template evaluations.
   typedef DynamicMatrix<Type,false>      OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef DynamicMatrix<Type,false>      TransposeType;  //!< Transpose type for expression template evaluations.
   typedef Type                           ElementType;    //!< Type of the matrix elements.
   typedef SIMDTrait_<ElementType>        SIMDType;       //!< SIMD type of the matrix elements.
   typedef const Type&                    ReturnType;     //!< Return type for expression template evaluations.
   typedef const This&                    CompositeType;  //!< Data type for composite expression templates.

   typedef Type&        Reference;       //!< Reference to a non-constant matrix value.
   typedef const Type&  ConstReference;  //!< Reference to a constant matrix value.
   typedef Type*        Pointer;         //!< Pointer to a non-constant matrix value.
   typedef const Type*  ConstPointer;    //!< Pointer to a constant matrix value.

   typedef DenseIterator<Type,AF>        Iterator;       //!< Iterator over non-constant elements.
   typedef DenseIterator<const Type,AF>  ConstIterator;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a CustomMatrix with different data/element type.
   */
   template< typename ET >  // Data type of the other matrix
   struct Rebind {
      typedef CustomMatrix<ET,AF,PF,true>  Other;  //!< The type of the other CustomMatrix.
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SIMD optimization.
   /*! The \a simdEnabled compilation flag indicates whether expressions the matrix is involved
       in can be optimized via SIMD operations. In case the element type of the matrix is a
       vectorizable data type, the \a simdEnabled compilation flag is set to \a true, otherwise
       it is set to \a false. */
   enum : bool { simdEnabled = IsVectorizable<Type>::value };

   //! Compilation flag for SMP assignments.
   /*! The \a smpAssignable compilation flag indicates whether the matrix can be used in SMP
       (shared memory parallel) assignments (both on the left-hand and right-hand side of the
       assignment). */
   enum : bool { smpAssignable = !IsSMPAssignable<Type>::value };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline CustomMatrix();
   explicit inline CustomMatrix( Type* ptr, size_t m, size_t n );
   explicit inline CustomMatrix( Type* ptr, size_t m, size_t n, size_t mm );

   template< typename Deleter, typename = EnableIf_< IsClass<Deleter> > >
   explicit inline CustomMatrix( Type* ptr, size_t m, size_t n, Deleter d );

   template< typename Deleter >
   explicit inline CustomMatrix( Type* ptr, size_t m, size_t n, size_t mm, Deleter d );

   inline CustomMatrix( const CustomMatrix& m );
   inline CustomMatrix( CustomMatrix&& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator()( size_t i, size_t j ) noexcept;
   inline ConstReference operator()( size_t i, size_t j ) const noexcept;
   inline Reference      at( size_t i, size_t j );
   inline ConstReference at( size_t i, size_t j ) const;
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Pointer        data  ( size_t j ) noexcept;
   inline ConstPointer   data  ( size_t j ) const noexcept;
   inline Iterator       begin ( size_t j ) noexcept;
   inline ConstIterator  begin ( size_t j ) const noexcept;
   inline ConstIterator  cbegin( size_t j ) const noexcept;
   inline Iterator       end   ( size_t j ) noexcept;
   inline ConstIterator  end   ( size_t j ) const noexcept;
   inline ConstIterator  cend  ( size_t j ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline CustomMatrix& operator=( const Type& set );
   inline CustomMatrix& operator=( initializer_list< initializer_list<Type> > list );

   template< typename Other, size_t M, size_t N >
   inline CustomMatrix& operator=( const Other (&array)[M][N] );

   inline CustomMatrix& operator=( const CustomMatrix& rhs );
   inline CustomMatrix& operator=( CustomMatrix&& rhs ) noexcept;

   template< typename MT, bool SO > inline CustomMatrix& operator= ( const Matrix<MT,SO>& rhs );
   template< typename MT, bool SO > inline CustomMatrix& operator+=( const Matrix<MT,SO>& rhs );
   template< typename MT, bool SO > inline CustomMatrix& operator-=( const Matrix<MT,SO>& rhs );
   template< typename MT, bool SO > inline CustomMatrix& operator*=( const Matrix<MT,SO>& rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, CustomMatrix >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, CustomMatrix >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t        rows() const noexcept;
                              inline size_t        columns() const noexcept;
                              inline size_t        spacing() const noexcept;
                              inline size_t        capacity() const noexcept;
                              inline size_t        capacity( size_t j ) const noexcept;
                              inline size_t        nonZeros() const;
                              inline size_t        nonZeros( size_t j ) const;
                              inline void          reset();
                              inline void          reset( size_t j );
                              inline void          clear();
                              inline CustomMatrix& transpose();
                              inline CustomMatrix& ctranspose();
   template< typename Other > inline CustomMatrix& scale( const Other& scalar );
                              inline void          swap( CustomMatrix& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Resource management functions***************************************************************
   /*!\name Resource management functions */
   //@{
   inline void reset( Type* ptr, size_t m, size_t n );
   inline void reset( Type* ptr, size_t m, size_t n, size_t mm );

   template< typename Deleter, typename = EnableIf_< IsClass<Deleter> > >
   inline void reset( Type* ptr, size_t m, size_t n, Deleter d );

   template< typename Deleter >
   inline void reset( Type* ptr, size_t m, size_t n, size_t mm, Deleter d );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT >
   struct VectorizedAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<MT> >::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT >
   struct VectorizedAddAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<MT> >::value &&
                            HasSIMDAdd< Type, ElementType_<MT> >::value &&
                            !IsDiagonal<MT>::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT >
   struct VectorizedSubAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<MT> >::value &&
                            HasSIMDSub< Type, ElementType_<MT> >::value &&
                            !IsDiagonal<MT>::value };
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

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t i, size_t j ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t i, size_t j, const SIMDType& value ) noexcept;

   template< typename MT >
   inline DisableIf_<VectorizedAssign<MT> > assign( const DenseMatrix<MT,true>& rhs );

   template< typename MT >
   inline EnableIf_<VectorizedAssign<MT> > assign( const DenseMatrix<MT,true>& rhs );

   template< typename MT > inline void assign( const DenseMatrix<MT,false>&  rhs );
   template< typename MT > inline void assign( const SparseMatrix<MT,true>&  rhs );
   template< typename MT > inline void assign( const SparseMatrix<MT,false>& rhs );

   template< typename MT >
   inline DisableIf_<VectorizedAddAssign<MT> > addAssign( const DenseMatrix<MT,true>& rhs );

   template< typename MT >
   inline EnableIf_<VectorizedAddAssign<MT> > addAssign( const DenseMatrix<MT,true>& rhs );

   template< typename MT > inline void addAssign( const DenseMatrix<MT,false>&  rhs );
   template< typename MT > inline void addAssign( const SparseMatrix<MT,true>&  rhs );
   template< typename MT > inline void addAssign( const SparseMatrix<MT,false>& rhs );

   template< typename MT >
   inline DisableIf_<VectorizedSubAssign<MT> > subAssign ( const DenseMatrix<MT,true>& rhs );

   template< typename MT >
   inline EnableIf_<VectorizedSubAssign<MT> > subAssign ( const DenseMatrix<MT,true>& rhs );

   template< typename MT > inline void subAssign( const DenseMatrix<MT,false>&  rhs );
   template< typename MT > inline void subAssign( const SparseMatrix<MT,true>&  rhs );
   template< typename MT > inline void subAssign( const SparseMatrix<MT,false>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t m_;                     //!< The current number of rows of the matrix.
   size_t mm_;                    //!< The number of elements between two columns.
   size_t n_;                     //!< The current number of columns of the matrix.
   boost::shared_array<Type> v_;  //!< The dynamically allocated matrix elements.
                                  /*!< Access to the matrix elements is gained via the function
                                       call operator. */
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
/*!\brief The default constructor for CustomMatrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline CustomMatrix<Type,AF,PF,true>::CustomMatrix()
   : m_ ( 0UL )  // The current number of rows of the matrix
   , mm_( 0UL )  // The number of elements between two columns
   , n_ ( 0UL )  // The current number of columns of the matrix
   , v_ (     )  // The matrix elements
{}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a matrix of size \f$ m \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This constructor creates an unpadded custom matrix of size \f$ m \times n \f$. The construction
// fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This constructor is \b NOT available for padded custom matrices!
// \note The custom matrix does \b NOT take responsibility for the given array of elements!
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline CustomMatrix<Type,AF,PF,true>::CustomMatrix( Type* ptr, size_t m, size_t n )
   : m_ ( m )  // The current number of rows of the matrix
   , mm_( m )  // The number of elements between two columns
   , n_ ( n )  // The current number of columns of the matrix
   , v_ (   )  // The matrix elements
{
   BLAZE_STATIC_ASSERT( PF == unpadded );

   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && ( !checkAlignment( ptr ) || mm_ % SIMDSIZE != 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   v_.reset( ptr, NoDelete() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a matrix of size \f$ m \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \param mm The total number of elements between two columns.
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This constructor creates a custom matrix of size \f$ m \times n \f$. The construction fails
// if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified spacing \a mm is insufficient for the given data type \a Type and the
//    available instruction set.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note The custom matrix does \b NOT take responsibility for the given array of elements!
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline CustomMatrix<Type,AF,PF,true>::CustomMatrix( Type* ptr, size_t m, size_t n, size_t mm )
   : m_ ( m  )  // The current number of rows of the matrix
   , mm_( mm )  // The number of elements between two columns
   , n_ ( n  )  // The current number of columns of the matrix
   , v_ (    )  // The matrix elements
{
   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && ( !checkAlignment( ptr ) || mm_ % SIMDSIZE != 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   if( PF && IsVectorizable<Type>::value && ( mm_ < nextMultiple<size_t>( m_, SIMDSIZE ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Insufficient capacity for padded matrix" );
   }

   v_.reset( ptr, NoDelete() );

   if( PF && IsVectorizable<Type>::value ) {
      for( size_t j=0UL; j<n_; ++j )
         for( size_t i=m_; i<mm_; ++i ) {
            v_[i+j*mm_] = Type();
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a matrix of size \f$ m \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \param d The deleter to destroy the array of elements.
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This constructor creates an unpadded custom matrix of size \f$ m \times n \f$. The construction
// fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This constructor is \b NOT available for padded custom matrices!
*/
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF >         // Padding flag
template< typename Deleter  // Type of the custom deleter
        , typename >        // Type restriction on the custom deleter
inline CustomMatrix<Type,AF,PF,true>::CustomMatrix( Type* ptr, size_t m, size_t n, Deleter d )
   : m_ ( m )  // The current number of rows of the matrix
   , mm_( m )  // The number of elements between two columns
   , n_ ( n )  // The current number of columns of the matrix
   , v_ (   )  // The matrix elements
{
   BLAZE_STATIC_ASSERT( PF == unpadded );

   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && ( !checkAlignment( ptr ) || mm_ % SIMDSIZE != 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   v_.reset( ptr, d );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a matrix of size \f$ m \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \param mm The total number of elements between two columns.
// \param d The deleter to destroy the array of elements.
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This constructor creates a custom matrix of size \f$ m \times n \f$. The construction fails
// if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified spacing \a mm is insufficient for the given data type \a Type and the
//    available instruction set.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
*/
template< typename Type       // Data type of the matrix
        , bool AF             // Alignment flag
        , bool PF >           // Padding flag
template< typename Deleter >  // Type of the custom deleter
inline CustomMatrix<Type,AF,PF,true>::CustomMatrix( Type* ptr, size_t m, size_t n, size_t mm, Deleter d )
   : m_ ( m  )  // The current number of rows of the matrix
   , mm_( mm )  // The number of elements between two columns
   , n_ ( n  )  // The current number of columns of the matrix
   , v_ (    )  // The matrix elements
{
   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && ( !checkAlignment( ptr ) || mm_ % SIMDSIZE != 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   if( PF && IsVectorizable<Type>::value && ( mm_ < nextMultiple<size_t>( m_, SIMDSIZE ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Insufficient capacity for padded matrix" );
   }

   v_.reset( ptr, d );

   if( PF && IsVectorizable<Type>::value ) {
      for( size_t j=0UL; j<n_; ++j )
         for( size_t i=m_; i<mm_; ++i ) {
            v_[i+j*mm_] = Type();
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The copy constructor for CustomMatrix.
//
// \param m Matrix to be copied.
//
// The copy constructor is explicitly defined due to the required dynamic memory management
// and in order to enable/facilitate NRV optimization.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline CustomMatrix<Type,AF,PF,true>::CustomMatrix( const CustomMatrix& m )
   : m_ ( m.m_  )  // The current number of rows of the matrix
   , mm_( m.mm_ )  // The number of elements between two columns
   , n_ ( m.n_  )  // The current number of columns of the matrix
   , v_ ( m.v_  )  // The matrix elements
{}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The move constructor for CustomMatrix.
//
// \param m The matrix to be moved into this instance.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline CustomMatrix<Type,AF,PF,true>::CustomMatrix( CustomMatrix&& m ) noexcept
   : m_ ( m.m_  )              // The current number of rows of the matrix
   , mm_( m.mm_ )              // The number of elements between two columns
   , n_ ( m.n_  )              // The current number of columns of the matrix
   , v_ ( std::move( m.v_ ) )  // The matrix elements
{
   m.m_  = 0UL;
   m.mm_ = 0UL;
   m.n_  = 0UL;

   BLAZE_INTERNAL_ASSERT( m.data() == nullptr, "Invalid data reference detected" );
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
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::Reference
   CustomMatrix<Type,AF,PF,true>::operator()( size_t i, size_t j ) noexcept
{
   BLAZE_USER_ASSERT( i<m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<n_, "Invalid column access index" );
   return v_[i+j*mm_];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::ConstReference
   CustomMatrix<Type,AF,PF,true>::operator()( size_t i, size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( i<m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<n_, "Invalid column access index" );
   return v_[i+j*mm_];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::Reference
   CustomMatrix<Type,AF,PF,true>::at( size_t i, size_t j )
{
   if( i >= m_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= n_ ) {
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
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::ConstReference
   CustomMatrix<Type,AF,PF,true>::at( size_t i, size_t j ) const
{
   if( i >= m_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= n_ ) {
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
// This function returns a pointer to the internal storage of the dynamic matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The dynamic matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a column are given by the \c columns() member functions, the total number
// of elements including padding is given by the \c spacing() member function.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::Pointer
   CustomMatrix<Type,AF,PF,true>::data() noexcept
{
   return v_.get();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the matrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The dynamic matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a column are given by the \c columns() member functions, the total number
// of elements including padding is given by the \c spacing() member function.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::ConstPointer
   CustomMatrix<Type,AF,PF,true>::data() const noexcept
{
   return v_.get();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the matrix elements of column \a j.
//
// \param j The column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in column \a j.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::Pointer
   CustomMatrix<Type,AF,PF,true>::data( size_t j ) noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return v_.get() + j*mm_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the matrix elements of column \a j.
//
// \param j The column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in column \a j.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::ConstPointer
   CustomMatrix<Type,AF,PF,true>::data( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return v_.get() + j*mm_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of column \a j.
//
// \param j The column index.
// \return Iterator to the first element of column \a j.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::Iterator
   CustomMatrix<Type,AF,PF,true>::begin( size_t j ) noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return Iterator( v_.get() + j*mm_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of column \a j.
//
// \param j The column index.
// \return Iterator to the first element of column \a j.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::ConstIterator
   CustomMatrix<Type,AF,PF,true>::begin( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return ConstIterator( v_.get() + j*mm_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of column \a j.
//
// \param j The column index.
// \return Iterator to the first element of column \a j.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::ConstIterator
   CustomMatrix<Type,AF,PF,true>::cbegin( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return ConstIterator( v_.get() + j*mm_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last element of column \a j.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::Iterator
   CustomMatrix<Type,AF,PF,true>::end( size_t j ) noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return Iterator( v_.get() + j*mm_ + m_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last element of column \a j.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::ConstIterator
   CustomMatrix<Type,AF,PF,true>::end( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return ConstIterator( v_.get() + j*mm_ + m_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last element of column \a j.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline typename CustomMatrix<Type,AF,PF,true>::ConstIterator
   CustomMatrix<Type,AF,PF,true>::cend( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return ConstIterator( v_.get() + j*mm_ + m_ );
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
/*!\brief Homogenous assignment to all matrix elements.
//
// \param rhs Scalar value to be assigned to all matrix elements.
// \return Reference to the assigned matrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline CustomMatrix<Type,AF,PF,true>& CustomMatrix<Type,AF,PF,true>::operator=( const Type& rhs )
{
   for( size_t j=0UL; j<n_; ++j )
      for( size_t i=0UL; i<m_; ++i )
         v_[i+j*mm_] = rhs;

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all matrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to static matrix.
//
// This assignment operator offers the option to directly assign to all elements of the matrix
// by means of an initializer list:

   \code
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;

   const int array[9] = { 0, 0, 0,
                          0, 0, 0,
                          0, 0, 0 };
   blaze::CustomMatrix<int,unaligned,unpadded,rowMajor> A( array, 3UL, 3UL );
   A = { { 1, 2, 3 },
         { 4, 5 },
         { 7, 8, 9 } };
   \endcode

// The matrix elements are assigned the values from the given initializer list. Missing values
// are initialized as default (as e.g. the value 6 in the example). Note that in case the size
// of the top-level initializer list exceeds the number of rows or the size of any nested list
// exceeds the number of columns, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline CustomMatrix<Type,AF,PF,true>&
   CustomMatrix<Type,AF,PF,true>::operator=( initializer_list< initializer_list<Type> > list )
{
   if( list.size() != m_ || determineColumns( list ) > n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to custom matrix" );
   }

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      size_t j( 0UL );
      for( const auto& element : rowList ) {
         v_[i+j*mm_] = element;
         ++j;
      }
      for( ; j<n_; ++j ) {
         v_[i+j*mm_] = Type();
      }
      ++i;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Array assignment to all matrix elements.
//
// \param array \f$ M \times N \f$ dimensional array for the assignment.
// \return Reference to the assigned matrix.
// \exception std::invalid_argument Invalid array size.
//
// This assignment operator offers the option to directly set all elements of the matrix:

   \code
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::columnMajor;

   const int array[9] = { 0, 0, 0,
                          0, 0, 0,
                          0, 0, 0 };
   const int init[3][3] = { { 1, 2, 3 },
                            { 4, 5 },
                            { 7, 8, 9 } };
   blaze::CustomMatrix<int,unaligned,unpadded,columnMajor> A( array, 3UL, 3UL );
   A = init;
   \endcode

// The matrix is assigned the values from the given array. Missing values are initialized with
// default values (as e.g. the value 6 in the example). Note that the size of the array must
// match the size of the custom matrix. Otherwise a \a std::invalid_argument exception is thrown.
// Also note that after the assignment \a array will have the same entries as \a init.
*/
template< typename Type   // Data type of the matrix
        , bool AF         // Alignment flag
        , bool PF >       // Padding flag
template< typename Other  // Data type of the initialization array
        , size_t M        // Number of rows of the initialization array
        , size_t N >      // Number of columns of the initialization array
inline CustomMatrix<Type,AF,PF,true>&
   CustomMatrix<Type,AF,PF,true>::operator=( const Other (&array)[M][N] )
{
   if( m_ != M || n_ != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array size" );
   }

   for( size_t j=0UL; j<N; ++j )
      for( size_t i=0UL; i<M; ++i )
         v_[i+j*mm_] = array[i][j];

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for CustomMatrix.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// The matrix is initialized as a copy of the given matrix. In case the current sizes of the two
// matrices don't match, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline CustomMatrix<Type,AF,PF,true>&
   CustomMatrix<Type,AF,PF,true>::operator=( const CustomMatrix& rhs )
{
   if( rhs.rows() != m_ || rhs.columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   smpAssign( *this, ~rhs );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Move assignment operator for CustomMatrix.
//
// \param rhs The matrix to be moved into this instance.
// \return Reference to the assigned matrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline CustomMatrix<Type,AF,PF,true>&
   CustomMatrix<Type,AF,PF,true>::operator=( CustomMatrix&& rhs ) noexcept
{
   m_  = rhs.m_;
   mm_ = rhs.mm_;
   n_  = rhs.n_;
   v_  = std::move( rhs.v_ );

   rhs.m_  = 0UL;
   rhs.mm_ = 0UL;
   rhs.n_  = 0UL;

   BLAZE_INTERNAL_ASSERT( rhs.data() == nullptr, "Invalid data reference detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different matrices.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// The matrix is initialized as a copy of the given matrix. In case the current sizes of the two
// matrices don't match, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT    // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline CustomMatrix<Type,AF,PF,true>&
   CustomMatrix<Type,AF,PF,true>::operator=( const Matrix<MT,SO>& rhs )
{
   typedef TransExprTrait_<This>   TT;
   typedef CTransExprTrait_<This>  CT;
   typedef InvExprTrait_<This>     IT;

   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( IsSame<MT,TT>::value && (~rhs).isAliased( this ) ) {
      transpose();
   }
   else if( IsSame<MT,CT>::value && (~rhs).isAliased( this ) ) {
      ctranspose();
   }
   else if( !IsSame<MT,IT>::value && (~rhs).canAlias( this ) ) {
      const ResultType_<MT> tmp( ~rhs );
      smpAssign( *this, tmp );
   }
   else {
      if( IsSparseMatrix<MT>::value )
         reset();
      smpAssign( *this, ~rhs );
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the matrix.
// \return Reference to the matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT    // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline CustomMatrix<Type,AF,PF,true>&
   CustomMatrix<Type,AF,PF,true>::operator+=( const Matrix<MT,SO>& rhs )
{
   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_<MT> tmp( ~rhs );
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
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the matrix.
// \return Reference to the matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT    // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline CustomMatrix<Type,AF,PF,true>&
   CustomMatrix<Type,AF,PF,true>::operator-=( const Matrix<MT,SO>& rhs )
{
   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_<MT> tmp( ~rhs );
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
/*!\brief Multiplication assignment operator for the multiplication of a matrix (\f$ A*=B \f$).
//
// \param rhs The right-hand side matrix for the multiplication.
// \return Reference to the matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// In case the current sizes of the two given matrices don't match, a \a std::invalid_argument
// is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT    // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline CustomMatrix<Type,AF,PF,true>&
   CustomMatrix<Type,AF,PF,true>::operator*=( const Matrix<MT,SO>& rhs )
{
   if( m_ != n_ || (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const MultTrait_< ResultType, ResultType_<MT> > tmp( *this * (~rhs) );
   smpAssign( *this, tmp );

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
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF >         // Padding flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, CustomMatrix<Type,AF,PF,true> >&
   CustomMatrix<Type,AF,PF,true>::operator*=( Other rhs )
{
   smpAssign( *this, (*this) * rhs );
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
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF >         // Padding flag
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, CustomMatrix<Type,AF,PF,true> >&
   CustomMatrix<Type,AF,PF,true>::operator/=( Other rhs )
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
/*!\brief Returns the current number of rows of the matrix.
//
// \return The number of rows of the matrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline size_t CustomMatrix<Type,AF,PF,true>::rows() const noexcept
{
   return m_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current number of columns of the matrix.
//
// \return The number of columns of the matrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline size_t CustomMatrix<Type,AF,PF,true>::columns() const noexcept
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
// This function returns the spacing between the beginning of two column, i.e. the total number
// of elements of a column.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline size_t CustomMatrix<Type,AF,PF,true>::spacing() const noexcept
{
   return mm_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the matrix.
//
// \return The capacity of the matrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline size_t CustomMatrix<Type,AF,PF,true>::capacity() const noexcept
{
   return mm_ * n_;
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline size_t CustomMatrix<Type,AF,PF,true>::capacity( size_t j ) const noexcept
{
   UNUSED_PARAMETER( j );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );
   return mm_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the total number of non-zero elements in the matrix
//
// \return The number of non-zero elements in the dense matrix.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline size_t CustomMatrix<Type,AF,PF,true>::nonZeros() const
{
   size_t nonzeros( 0UL );

   for( size_t j=0UL; j<n_; ++j )
      for( size_t i=0UL; i<m_; ++i )
         if( !isDefault( v_[i+j*mm_] ) )
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline size_t CustomMatrix<Type,AF,PF,true>::nonZeros( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   const size_t iend( j*mm_ + m_ );
   size_t nonzeros( 0UL );

   for( size_t i=j*mm_; i<iend; ++i )
      if( !isDefault( v_[i] ) )
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline void CustomMatrix<Type,AF,PF,true>::reset()
{
   using blaze::clear;

   for( size_t j=0UL; j<n_; ++j )
      for( size_t i=0UL; i<m_; ++i )
         clear( v_[i+j*mm_] );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset the specified column to the default initial values.
//
// \param j The index of the column.
// \return void
//
// This function reset the values in the specified column to their default value. Note that
// the capacity of the column remains unchanged.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline void CustomMatrix<Type,AF,PF,true>::reset( size_t j )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );
   for( size_t i=0UL; i<m_; ++i )
      clear( v_[i+j*mm_] );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the \f$ M \times N \f$ matrix.
//
// \return void
//
// After the clear() function, the size of the matrix is 0.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline void CustomMatrix<Type,AF,PF,true>::clear()
{
   m_  = 0UL;
   mm_ = 0UL;
   n_  = 0UL;
   v_.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transpose of the matrix.
//
// \return Reference to the transposed matrix.
// \exception std::logic_error Impossible transpose operation.
//
// In case the matrix is not a square matrix, a \a std::logic_error exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline CustomMatrix<Type,AF,PF,true>& CustomMatrix<Type,AF,PF,true>::transpose()
{
   using std::swap;

   if( m_ != n_ ) {
      BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
   }

   for( size_t j=1UL; j<n_; ++j )
      for( size_t i=0UL; i<j; ++i )
         swap( v_[i+j*mm_], v_[j+i*mm_] );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place conjugate transpose of the matrix.
//
// \return Reference to the transposed matrix.
// \exception std::logic_error Impossible transpose operation.
//
// In case the matrix is not a square matrix, a \a std::logic_error exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline CustomMatrix<Type,AF,PF,true>& CustomMatrix<Type,AF,PF,true>::ctranspose()
{
   if( m_ != n_ ) {
      BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
   }

   for( size_t j=0UL; j<n_; ++j ) {
      for( size_t i=0UL; i<j; ++i ) {
         cswap( v_[i+j*mm_], v_[j+i*mm_] );
      }
      conjugate( v_[j+j*mm_] );
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
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF >         // Padding flag
template< typename Other >  // Data type of the scalar value
inline CustomMatrix<Type,AF,PF,true>& CustomMatrix<Type,AF,PF,true>::scale( const Other& scalar )
{
   for( size_t j=0UL; j<n_; ++j )
      for( size_t i=0UL; i<m_; ++i )
         v_[i+j*mm_] *= scalar;

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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline void CustomMatrix<Type,AF,PF,true>::swap( CustomMatrix& m ) noexcept
{
   using std::swap;

   swap( m_ , m.m_  );
   swap( mm_, m.mm_ );
   swap( n_ , m.n_  );
   swap( v_ , m.v_  );
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
/*!\brief Resets the custom matrix and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \return void
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This function resets the custom matrix to the given array of elements of size \f$ m \times n \f$.
// The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This function is \b NOT available for padded custom matrices!
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom matrix referencing the array goes out of scope.
// \note The custom matrix does NOT take responsibility for the new array of elements!
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline void CustomMatrix<Type,AF,PF,true>::reset( Type* ptr, size_t m, size_t n )
{
   BLAZE_STATIC_ASSERT( PF == unpadded );

   CustomMatrix tmp( ptr, m, n );
   swap( tmp );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resets the custom matrix and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \param mm The total number of elements between two columns.
// \return void
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This function resets the custom matrix to the given array of elements of size \f$ m \times n \f$.
// The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom matrix referencing the array goes out of scope.
// \note The custom matrix does NOT take responsibility for the new array of elements!
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline void CustomMatrix<Type,AF,PF,true>::reset( Type* ptr, size_t m, size_t n, size_t mm )
{
   CustomMatrix tmp( ptr, m, n, mm );
   swap( tmp );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resets the custom matrix and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \param d The deleter to destroy the array of elements.
// \return void
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This function resets the custom matrix to the given array of elements of size \f$ m \times n \f$.
// The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This function is \b NOT available for padded custom matrices!
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom matrix referencing the array goes out of scope.
*/
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF >         // Padding flag
template< typename Deleter  // Type of the custom deleter
        , typename >        // Type restriction on the custom deleter
inline void CustomMatrix<Type,AF,PF,true>::reset( Type* ptr, size_t m, size_t n, Deleter d )
{
   BLAZE_STATIC_ASSERT( !IsClass<Deleter>::value || PF == unpadded );

   CustomMatrix tmp( ptr, m, n, d );
   swap( tmp );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resets the custom matrix and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the matrix.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \param mm The total number of elements between two columns.
// \param d The deleter to destroy the array of elements.
// \return void
// \exception std::invalid_argument Invalid setup of custom matrix.
//
// This function resets the custom matrix to the given array of elements of size \f$ m \times n \f$.
// The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom matrix referencing the array goes out of scope.
*/
template< typename Type       // Data type of the matrix
        , bool AF             // Alignment flag
        , bool PF >           // Padding flag
template< typename Deleter >  // Type of the custom deleter
inline void CustomMatrix<Type,AF,PF,true>::reset( Type* ptr, size_t m, size_t n, size_t mm, Deleter d )
{
   CustomMatrix tmp( ptr, m, n, mm, d );
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
/*!\brief Returns whether the matrix can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this matrix, \a false if not.
//
// This function returns whether the given address can alias with the matrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF >         // Padding flag
template< typename Other >  // Data type of the foreign expression
inline bool CustomMatrix<Type,AF,PF,true>::canAlias( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
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
template< typename Type     // Data type of the matrix
        , bool AF           // Alignment flag
        , bool PF >         // Padding flag
template< typename Other >  // Data type of the foreign expression
inline bool CustomMatrix<Type,AF,PF,true>::isAliased( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
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
// whether the beginning and the end of each column of the matrix are guaranteed to conform to
// the alignment restrictions of the element type \a Type.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline bool CustomMatrix<Type,AF,PF,true>::isAligned() const noexcept
{
   return ( AF || ( checkAlignment( v_.get() ) && rows() % SIMDSIZE == 0UL ) );
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
inline bool CustomMatrix<Type,AF,PF,true>::canSMPAssign() const noexcept
{
   return ( columns() > SMP_DMATASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Load of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense matrix. The row index
// must be smaller than the number of rows and the column index must be smaller than the number
// of columns. Additionally, the row index must be a multiple of the number of values inside
// the SIMD element. This function must \b NOT be called explicitly! It is used internally
// for the performance optimized evaluation of expression templates. Calling this function
// explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
BLAZE_ALWAYS_INLINE typename CustomMatrix<Type,AF,PF,true>::SIMDType
   CustomMatrix<Type,AF,PF,true>::load( size_t i, size_t j ) const noexcept
{
   if( AF && PF )
      return loada( i, j );
   else
      return loadu( i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense matrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the row index must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
BLAZE_ALWAYS_INLINE typename CustomMatrix<Type,AF,PF,true>::SIMDType
   CustomMatrix<Type,AF,PF,true>::loada( size_t i, size_t j ) const noexcept
{
   using blaze::loada;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= ( PF ? mm_ : m_ ), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( !PF || i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_.get()+i+j*mm_ ), "Invalid alignment detected" );

   return loada( v_.get()+i+j*mm_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned load of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense matrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the row index must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
BLAZE_ALWAYS_INLINE typename CustomMatrix<Type,AF,PF,true>::SIMDType
   CustomMatrix<Type,AF,PF,true>::loadu( size_t i, size_t j ) const noexcept
{
   using blaze::loadu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= ( PF ? mm_ : m_ ), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );

   return loadu( v_.get()+i+j*mm_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Store of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store of a specific SIMD element of the dense matrix. The row index
// must be smaller than the number of rows and the column index must be smaller then the number
// of columns. Additionally, the row index must be a multiple of the number of values inside the
// SIMD element. This function must \b NOT be called explicitly! It is used internally for the
// performance optimized evaluation of expression templates. Calling this function explicitly
// might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
BLAZE_ALWAYS_INLINE void
   CustomMatrix<Type,AF,PF,true>::store( size_t i, size_t j, const SIMDType& value ) noexcept
{
   if( AF && PF )
      storea( i, j, value );
   else
      storeu( i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned store of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store of a specific SIMD element of the dense matrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the row index must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
BLAZE_ALWAYS_INLINE void
   CustomMatrix<Type,AF,PF,true>::storea( size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::storea;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= ( PF ? mm_ : m_ ), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( !PF || i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_.get()+i+j*mm_ ), "Invalid alignment detected" );

   storea( v_.get()+i+j*mm_, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned store of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store of a specific SIMD element of the dense matrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the row index must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
BLAZE_ALWAYS_INLINE void
   CustomMatrix<Type,AF,PF,true>::storeu( size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::storeu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= ( PF ? mm_ : m_ ), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );

   storeu( v_.get()+i+j*mm_, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned, non-temporal store of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store of a specific SIMD element of the
// dense matrix. The row index must be smaller than the number of rows and the column index
// must be smaller than the number of columns. Additionally, the row index must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
BLAZE_ALWAYS_INLINE void
   CustomMatrix<Type,AF,PF,true>::stream( size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::stream;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= ( PF ? mm_ : m_ ), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( !PF || i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_.get()+i+j*mm_ ), "Invalid alignment detected" );

   stream( v_.get()+i+j*mm_, value );
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side dense matrix
inline DisableIf_<typename CustomMatrix<Type,AF,PF,true>::BLAZE_TEMPLATE VectorizedAssign<MT> >
   CustomMatrix<Type,AF,PF,true>::assign( const DenseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t ipos( m_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( m_ - ( m_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t j=0UL; j<n_; ++j ) {
      for( size_t i=0UL; i<ipos; i+=2UL ) {
         v_[i    +j*mm_] = (~rhs)(i    ,j);
         v_[i+1UL+j*mm_] = (~rhs)(i+1UL,j);
      }
      if( ipos < m_ ) {
         v_[ipos+j*mm_] = (~rhs)(ipos,j);
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side dense matrix
inline EnableIf_<typename CustomMatrix<Type,AF,PF,true>::BLAZE_TEMPLATE VectorizedAssign<MT> >
   CustomMatrix<Type,AF,PF,true>::assign( const DenseMatrix<MT,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const bool remainder( !PF || !IsPadded<MT>::value );

   const size_t ipos( ( remainder )?( m_ & size_t(-SIMDSIZE) ):( m_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( m_ - ( m_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   if( AF && PF && useStreaming &&
       ( m_*n_ > ( cacheSize / ( sizeof(Type) * 3UL ) ) ) && !(~rhs).isAliased( this ) )
   {
      for( size_t j=0UL; j<n_; ++j )
      {
         size_t i( 0UL );

         for( ; i<ipos; i+=SIMDSIZE ) {
            stream( i, j, (~rhs).load(i,j) );
         }
         for( ; remainder && i<m_; ++i ) {
            v_[i+j*mm_] = (~rhs)(i,j);
         }
      }
   }
   else
   {
      for( size_t j=0UL; j<n_; ++j )
      {
         size_t i( 0UL );
         ConstIterator_<MT> it( (~rhs).begin(j) );

         for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
            store( i             , j, it.load() ); it += SIMDSIZE;
            store( i+SIMDSIZE    , j, it.load() ); it += SIMDSIZE;
            store( i+SIMDSIZE*2UL, j, it.load() ); it += SIMDSIZE;
            store( i+SIMDSIZE*3UL, j, it.load() ); it += SIMDSIZE;
         }
         for( ; i<ipos; i+=SIMDSIZE, it+=SIMDSIZE ) {
            store( i, j, it.load() );
         }
         for( ; remainder && i<m_; ++i, ++it ) {
            v_[i+j*mm_] = *it;
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side dense matrix
inline void CustomMatrix<Type,AF,PF,true>::assign( const DenseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t jj=0UL; jj<n_; jj+=block ) {
      const size_t jend( min( n_, jj+block ) );
      for( size_t ii=0UL; ii<m_; ii+=block ) {
         const size_t iend( min( m_, ii+block ) );
         for( size_t j=jj; j<jend; ++j ) {
            for( size_t i=ii; i<iend; ++i ) {
               v_[i+j*mm_] = (~rhs)(i,j);
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side sparse matrix
inline void CustomMatrix<Type,AF,PF,true>::assign( const SparseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<(~rhs).columns(); ++j )
      for( ConstIterator_<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()+j*mm_] = element->value();
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side sparse matrix
inline void CustomMatrix<Type,AF,PF,true>::assign( const SparseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<(~rhs).rows(); ++i )
      for( ConstIterator_<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i+element->index()*mm_] = element->value();
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side dense matrix
inline DisableIf_<typename CustomMatrix<Type,AF,PF,true>::BLAZE_TEMPLATE VectorizedAddAssign<MT> >
   CustomMatrix<Type,AF,PF,true>::addAssign( const DenseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
   {
      if( IsDiagonal<MT>::value )
      {
         v_[j+j*mm_] += (~rhs)(j,j);
      }
      else
      {
         const size_t ibegin( ( IsLower<MT>::value )
                              ?( IsStrictlyLower<MT>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend  ( ( IsUpper<MT>::value )
                              ?( IsStrictlyUpper<MT>::value ? j : j+1UL )
                              :( m_ ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         size_t i( ibegin );

         for( ; (i+2UL) <= iend; i+=2UL ) {
            v_[i    +j*mm_] += (~rhs)(i    ,j);
            v_[i+1UL+j*mm_] += (~rhs)(i+1UL,j);
         }
         if( i < iend ) {
            v_[i+j*mm_] += (~rhs)(i,j);
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side dense matrix
inline EnableIf_<typename CustomMatrix<Type,AF,PF,true>::BLAZE_TEMPLATE VectorizedAddAssign<MT> >
   CustomMatrix<Type,AF,PF,true>::addAssign( const DenseMatrix<MT,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_DIAGONAL_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const bool remainder( !PF || !IsPadded<MT>::value );

   for( size_t j=0UL; j<n_; ++j )
   {
      const size_t ibegin( ( IsLower<MT>::value )
                           ?( ( IsStrictlyLower<MT>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                           :( 0UL ) );
      const size_t iend  ( ( IsUpper<MT>::value )
                           ?( IsStrictlyUpper<MT>::value ? j : j+1UL )
                           :( m_ ) );
      BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

      const size_t ipos( ( remainder )?( iend & size_t(-SIMDSIZE) ):( iend ) );
      BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

      size_t i( ibegin );
      ConstIterator_<MT> it( (~rhs).begin(j) + ibegin );

      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
         store( i             , j, load(i             ,j) + it.load() ); it += SIMDSIZE;
         store( i+SIMDSIZE    , j, load(i+SIMDSIZE    ,j) + it.load() ); it += SIMDSIZE;
         store( i+SIMDSIZE*2UL, j, load(i+SIMDSIZE*2UL,j) + it.load() ); it += SIMDSIZE;
         store( i+SIMDSIZE*3UL, j, load(i+SIMDSIZE*3UL,j) + it.load() ); it += SIMDSIZE;
      }
      for( ; i<ipos; i+=SIMDSIZE, it+=SIMDSIZE ) {
         store( i, j, load(i,j) + it.load() );
      }
      for( ; remainder && i<iend; ++i, ++it ) {
         v_[i+j*mm_] += *it;
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side dense matrix
inline void CustomMatrix<Type,AF,PF,true>::addAssign( const DenseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t jj=0UL; jj<n_; jj+=block ) {
      const size_t jend( min( n_, jj+block ) );
      for( size_t ii=0UL; ii<m_; ii+=block )
      {
         if( IsLower<MT>::value && ii < jj ) continue;
         if( IsUpper<MT>::value && ii > jj ) break;

         for( size_t j=jj; j<jend; ++j )
         {
            const size_t ibegin( ( IsLower<MT>::value )
                                 ?( max( ( IsStrictlyLower<MT>::value ? j+1UL : j ), ii ) )
                                 :( ii ) );
            const size_t iend  ( ( IsUpper<MT>::value )
                                 ?( min( ( IsStrictlyUpper<MT>::value ? j : j+1UL ), m_, ii+block ) )
                                 :( min( m_, ii+block ) ) );
            BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

            for( size_t i=ibegin; i<iend; ++i ) {
               v_[i+j*mm_] += (~rhs)(i,j);
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side sparse matrix
inline void CustomMatrix<Type,AF,PF,true>::addAssign( const SparseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<(~rhs).columns(); ++j )
      for( ConstIterator_<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()+j*mm_] += element->value();
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side sparse matrix
inline void CustomMatrix<Type,AF,PF,true>::addAssign( const SparseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<(~rhs).rows(); ++i )
      for( ConstIterator_<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i+element->index()*mm_] += element->value();
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side dense matrix
inline DisableIf_<typename CustomMatrix<Type,AF,PF,true>::BLAZE_TEMPLATE VectorizedSubAssign<MT> >
   CustomMatrix<Type,AF,PF,true>::subAssign( const DenseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
   {
      if( IsDiagonal<MT>::value )
      {
         v_[j+j*mm_] -= (~rhs)(j,j);
      }
      else
      {
         const size_t ibegin( ( IsLower<MT>::value )
                              ?( IsStrictlyLower<MT>::value ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend  ( ( IsUpper<MT>::value )
                              ?( IsStrictlyUpper<MT>::value ? j : j+1UL )
                              :( m_ ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         size_t i( ibegin );

         for( ; (i+2UL) <= iend; i+=2UL ) {
            v_[i  +j*mm_] -= (~rhs)(i  ,j);
            v_[i+1+j*mm_] -= (~rhs)(i+1,j);
         }
         if( i < iend ) {
            v_[i+j*mm_] -= (~rhs)(i,j);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the subtraction assignment of a column-major
//        dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side dense matrix
inline EnableIf_<typename CustomMatrix<Type,AF,PF,true>::BLAZE_TEMPLATE VectorizedSubAssign<MT> >
   CustomMatrix<Type,AF,PF,true>::subAssign( const DenseMatrix<MT,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_DIAGONAL_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const bool remainder( !PF || !IsPadded<MT>::value );

   for( size_t j=0UL; j<n_; ++j )
   {
      const size_t ibegin( ( IsLower<MT>::value )
                           ?( ( IsStrictlyLower<MT>::value ? j+1UL : j ) & size_t(-SIMDSIZE) )
                           :( 0UL ) );
      const size_t iend  ( ( IsUpper<MT>::value )
                           ?( IsStrictlyUpper<MT>::value ? j : j+1UL )
                           :( m_ ) );
      BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

      const size_t ipos( ( remainder )?( iend & size_t(-SIMDSIZE) ):( iend ) );
      BLAZE_INTERNAL_ASSERT( !remainder || ( iend - ( iend % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

      size_t i( ibegin );
      ConstIterator_<MT> it( (~rhs).begin(j) + ibegin );

      for( ; (i+SIMDSIZE*3UL) < ipos; i+=SIMDSIZE*4UL ) {
         store( i             , j, load(i             ,j) - it.load() ); it += SIMDSIZE;
         store( i+SIMDSIZE    , j, load(i+SIMDSIZE    ,j) - it.load() ); it += SIMDSIZE;
         store( i+SIMDSIZE*2UL, j, load(i+SIMDSIZE*2UL,j) - it.load() ); it += SIMDSIZE;
         store( i+SIMDSIZE*3UL, j, load(i+SIMDSIZE*3UL,j) - it.load() ); it += SIMDSIZE;
      }
      for( ; i<ipos; i+=SIMDSIZE, it+=SIMDSIZE ) {
         store( i, j, load(i,j) - it.load() );
      }
      for( ; remainder && i<iend; ++i, ++it ) {
         v_[i+j*mm_] -= *it;
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side dense matrix
inline void CustomMatrix<Type,AF,PF,true>::subAssign( const DenseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t block( BLOCK_SIZE );

   for( size_t jj=0UL; jj<n_; jj+=block ) {
      const size_t jend( min( n_, jj+block ) );
      for( size_t ii=0UL; ii<m_; ii+=block )
      {
         if( IsLower<MT>::value && ii < jj ) continue;
         if( IsUpper<MT>::value && ii > jj ) break;

         for( size_t j=jj; j<jend; ++j )
         {
            const size_t ibegin( ( IsLower<MT>::value )
                                 ?( max( ( IsStrictlyLower<MT>::value ? j+1UL : j ), ii ) )
                                 :( ii ) );
            const size_t iend  ( ( IsUpper<MT>::value )
                                 ?( min( ( IsStrictlyUpper<MT>::value ? j : j+1UL ), m_, ii+block ) )
                                 :( min( m_, ii+block ) ) );
            BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

            for( size_t i=ibegin; i<iend; ++i ) {
               v_[i+j*mm_] -= (~rhs)(i,j);
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side sparse matrix
inline void CustomMatrix<Type,AF,PF,true>::subAssign( const SparseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<(~rhs).columns(); ++j )
      for( ConstIterator_<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()+j*mm_] -= element->value();
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
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF >      // Padding flag
template< typename MT >  // Type of the right-hand side sparse matrix
inline void CustomMatrix<Type,AF,PF,true>::subAssign( const SparseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<(~rhs).rows(); ++i )
      for( ConstIterator_<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i+element->index()*mm_] -= element->value();
}
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  CUSTOMMATRIX OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name CustomMatrix operators */
//@{
template< typename Type, bool AF, bool PF, bool SO >
inline void reset( CustomMatrix<Type,AF,PF,SO>& m );

template< typename Type, bool AF, bool PF, bool SO >
inline void reset( CustomMatrix<Type,AF,PF,SO>& m, size_t i );

template< typename Type, bool AF, bool PF, bool SO >
inline void clear( CustomMatrix<Type,AF,PF,SO>& m );

template< typename Type, bool AF, bool PF, bool SO >
inline bool isDefault( const CustomMatrix<Type,AF,PF,SO>& m );

template< typename Type, bool AF, bool PF, bool SO >
inline bool isIntact( const CustomMatrix<Type,AF,PF,SO>& m );

template< typename Type, bool AF, bool PF, bool SO >
inline void swap( CustomMatrix<Type,AF,PF,SO>& a, CustomMatrix<Type,AF,PF,SO>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given custom matrix.
// \ingroup custom_matrix
//
// \param m The matrix to be resetted.
// \return void
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline void reset( CustomMatrix<Type,AF,PF,SO>& m )
{
   m.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset the specified row/column of the given custom matrix.
// \ingroup custom_matrix
//
// \param m The matrix to be resetted.
// \param i The index of the row/column to be resetted.
// \return void
//
// This function resets the values in the specified row/column of the given custom matrix to
// their default value. In case the given matrix is a \a rowMajor matrix the function resets the
// values in row \a i, if it is a \a columnMajor matrix the function resets the values in column
// \a i. Note that the capacity of the row/column remains unchanged.
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline void reset( CustomMatrix<Type,AF,PF,SO>& m, size_t i )
{
   m.reset( i );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given custom matrix.
// \ingroup custom_matrix
//
// \param m The matrix to be cleared.
// \return void
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline void clear( CustomMatrix<Type,AF,PF,SO>& m )
{
   m.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given custom matrix is in default state.
// \ingroup custom_matrix
//
// \param m The matrix to be tested for its default state.
// \return \a true in case the given matrix's rows and columns are zero, \a false otherwise.
//
// This function checks whether the custom matrix is in default (constructed) state, i.e. if
// it's number of rows and columns is 0. In case it is in default state, the function returns
// \a true, else it will return \a false. The following example demonstrates the use of the
// \a isDefault() function:

   \code
   using blaze::aligned;
   using blaze::padded;

   blaze::CustomMatrix<int,aligned,padded> A( ... );
   // ... Resizing and initialization
   if( isDefault( A ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline bool isDefault( const CustomMatrix<Type,AF,PF,SO>& m )
{
   return ( m.rows() == 0UL && m.columns() == 0UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given custom matrix are intact.
// \ingroup custom_matrix
//
// \param m The custom matrix to be tested.
// \return \a true in case the given matrix's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the custom matrix are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   using blaze::aligned;
   using blaze::padded;

   blaze::CustomMatrix<int,aligned,padded> A( ... );
   // ... Resizing and initialization
   if( isIntact( A ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline bool isIntact( const CustomMatrix<Type,AF,PF,SO>& m )
{
   return ( m.rows() * m.columns() <= m.capacity() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two custom matrices.
// \ingroup custom_matrix
//
// \param a The first matrix to be swapped.
// \param b The second matrix to be swapped.
// \return void
*/
template< typename Type  // Data type of the matrix
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , bool SO >      // Storage order
inline void swap( CustomMatrix<Type,AF,PF,SO>& a, CustomMatrix<Type,AF,PF,SO>& b ) noexcept
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
template< typename T, bool AF, bool PF, bool SO >
struct HasConstDataAccess< CustomMatrix<T,AF,PF,SO> > : public TrueType
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
template< typename T, bool AF, bool PF, bool SO >
struct HasMutableDataAccess< CustomMatrix<T,AF,PF,SO> > : public TrueType
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
template< typename T, bool AF, bool PF, bool SO >
struct IsCustom< CustomMatrix<T,AF,PF,SO> > : public TrueType
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
template< typename T, bool PF, bool SO >
struct IsAligned< CustomMatrix<T,aligned,PF,SO> > : public TrueType
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
template< typename T, bool AF, bool SO >
struct IsPadded< CustomMatrix<T,AF,padded,SO> > : public TrueType
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
template< typename T1, bool AF, bool PF, bool SO, typename T2, size_t M, size_t N >
struct AddTrait< CustomMatrix<T1,AF,PF,SO>, StaticMatrix<T2,M,N,SO> >
{
   using Type = StaticMatrix< AddTrait_<T1,T2>, M, N, SO >;
};

template< typename T1, bool AF, bool PF, bool SO1, typename T2, size_t M, size_t N, bool SO2 >
struct AddTrait< CustomMatrix<T1,AF,PF,SO1>, StaticMatrix<T2,M,N,SO2> >
{
   using Type = StaticMatrix< AddTrait_<T1,T2>, M, N, false >;
};

template< typename T1, size_t M, size_t N, bool SO, typename T2, bool AF, bool PF >
struct AddTrait< StaticMatrix<T1,M,N,SO>, CustomMatrix<T2,AF,PF,SO> >
{
   using Type = StaticMatrix< AddTrait_<T1,T2>, M, N, SO >;
};

template< typename T1, size_t M, size_t N, bool SO1, typename T2, bool AF, bool PF, bool SO2 >
struct AddTrait< StaticMatrix<T1,M,N,SO1>, CustomMatrix<T2,AF,PF,SO2> >
{
   using Type = StaticMatrix< AddTrait_<T1,T2>, M, N, false >;
};

template< typename T1, bool AF, bool PF, bool SO, typename T2, size_t M, size_t N >
struct AddTrait< CustomMatrix<T1,AF,PF,SO>, HybridMatrix<T2,M,N,SO> >
{
   using Type = HybridMatrix< AddTrait_<T1,T2>, M, N, SO >;
};

template< typename T1, bool AF, bool PF, bool SO1, typename T2, size_t M, size_t N, bool SO2 >
struct AddTrait< CustomMatrix<T1,AF,PF,SO1>, HybridMatrix<T2,M,N,SO2> >
{
   using Type = HybridMatrix< AddTrait_<T1,T2>, M, N, false >;
};

template< typename T1, size_t M, size_t N, bool SO, typename T2, bool AF, bool PF >
struct AddTrait< HybridMatrix<T1,M,N,SO>, CustomMatrix<T2,AF,PF,SO> >
{
   using Type = HybridMatrix< AddTrait_<T1,T2>, M, N, SO >;
};

template< typename T1, size_t M, size_t N, bool SO1, typename T2, bool AF, bool PF, bool SO2 >
struct AddTrait< HybridMatrix<T1,M,N,SO1>, CustomMatrix<T2,AF,PF,SO2> >
{
   using Type = HybridMatrix< AddTrait_<T1,T2>, M, N, false >;
};

template< typename T1, bool AF, bool PF, bool SO, typename T2 >
struct AddTrait< CustomMatrix<T1,AF,PF,SO>, DynamicMatrix<T2,SO> >
{
   using Type = DynamicMatrix< AddTrait_<T1,T2>, SO >;
};

template< typename T1, bool AF, bool PF, bool SO1, typename T2, bool SO2 >
struct AddTrait< CustomMatrix<T1,AF,PF,SO1>, DynamicMatrix<T2,SO2> >
{
   using Type = DynamicMatrix< AddTrait_<T1,T2>, false >;
};

template< typename T1, bool SO, typename T2, bool AF, bool PF >
struct AddTrait< DynamicMatrix<T1,SO>, CustomMatrix<T2,AF,PF,SO> >
{
   using Type = DynamicMatrix< AddTrait_<T1,T2>, SO >;
};

template< typename T1, bool SO1, typename T2, bool AF, bool PF, bool SO2 >
struct AddTrait< DynamicMatrix<T1,SO1>, CustomMatrix<T2,AF,PF,SO2> >
{
   using Type = DynamicMatrix< AddTrait_<T1,T2>, false >;
};

template< typename T1, bool AF1, bool PF1, bool SO, typename T2, bool AF2, bool PF2 >
struct AddTrait< CustomMatrix<T1,AF1,PF1,SO>, CustomMatrix<T2,AF2,PF2,SO> >
{
   using Type = DynamicMatrix< AddTrait_<T1,T2>, SO >;
};

template< typename T1, bool AF1, bool PF1, bool SO1, typename T2, bool AF2, bool PF2, bool SO2 >
struct AddTrait< CustomMatrix<T1,AF1,PF1,SO1>, CustomMatrix<T2,AF2,PF2,SO2> >
{
   using Type = DynamicMatrix< AddTrait_<T1,T2>, false >;
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
template< typename T1, bool AF, bool PF, bool SO, typename T2, size_t M, size_t N >
struct SubTrait< CustomMatrix<T1,AF,PF,SO>, StaticMatrix<T2,M,N,SO> >
{
   using Type = StaticMatrix< SubTrait_<T1,T2>, M, N, SO >;
};

template< typename T1, bool AF, bool PF, bool SO1, typename T2, size_t M, size_t N, bool SO2 >
struct SubTrait< CustomMatrix<T1,AF,PF,SO1>, StaticMatrix<T2,M,N,SO2> >
{
   using Type = StaticMatrix< SubTrait_<T1,T2>, M, N, false >;
};

template< typename T1, size_t M, size_t N, bool SO, typename T2, bool AF, bool PF >
struct SubTrait< StaticMatrix<T1,M,N,SO>, CustomMatrix<T2,AF,PF,SO> >
{
   using Type = StaticMatrix< SubTrait_<T1,T2>, M, N, SO >;
};

template< typename T1, size_t M, size_t N, bool SO1, typename T2, bool AF, bool PF, bool SO2 >
struct SubTrait< StaticMatrix<T1,M,N,SO1>, CustomMatrix<T2,AF,PF,SO2> >
{
   using Type = StaticMatrix< SubTrait_<T1,T2>, M, N, false >;
};

template< typename T1, bool AF, bool PF, bool SO, typename T2, size_t M, size_t N >
struct SubTrait< CustomMatrix<T1,AF,PF,SO>, HybridMatrix<T2,M,N,SO> >
{
   using Type = HybridMatrix< SubTrait_<T1,T2>, M, N, SO >;
};

template< typename T1, bool AF, bool PF, bool SO1, typename T2, size_t M, size_t N, bool SO2 >
struct SubTrait< CustomMatrix<T1,AF,PF,SO1>, HybridMatrix<T2,M,N,SO2> >
{
   using Type = HybridMatrix< SubTrait_<T1,T2>, M, N, false >;
};

template< typename T1, size_t M, size_t N, bool SO, typename T2, bool AF, bool PF >
struct SubTrait< HybridMatrix<T1,M,N,SO>, CustomMatrix<T2,AF,PF,SO> >
{
   using Type = HybridMatrix< SubTrait_<T1,T2>, M, N, SO >;
};

template< typename T1, size_t M, size_t N, bool SO1, typename T2, bool AF, bool PF, bool SO2 >
struct SubTrait< HybridMatrix<T1,M,N,SO1>, CustomMatrix<T2,AF,PF,SO2> >
{
   using Type = HybridMatrix< SubTrait_<T1,T2>, M, N, false >;
};

template< typename T1, bool AF, bool PF, bool SO, typename T2 >
struct SubTrait< CustomMatrix<T1,AF,PF,SO>, DynamicMatrix<T2,SO> >
{
   using Type = DynamicMatrix< SubTrait_<T1,T2>, SO >;
};

template< typename T1, bool AF, bool PF, bool SO1, typename T2, bool SO2 >
struct SubTrait< CustomMatrix<T1,AF,PF,SO1>, DynamicMatrix<T2,SO2> >
{
   using Type = DynamicMatrix< SubTrait_<T1,T2>, false >;
};

template< typename T1, bool SO, typename T2, bool AF, bool PF >
struct SubTrait< DynamicMatrix<T1,SO>, CustomMatrix<T2,AF,PF,SO> >
{
   using Type = DynamicMatrix< SubTrait_<T1,T2>, SO >;
};

template< typename T1, bool SO1, typename T2, bool AF, bool PF, bool SO2 >
struct SubTrait< DynamicMatrix<T1,SO1>, CustomMatrix<T2,AF,PF,SO2> >
{
   using Type = DynamicMatrix< SubTrait_<T1,T2>, false >;
};

template< typename T1, bool AF1, bool PF1, bool SO, typename T2, bool AF2, bool PF2 >
struct SubTrait< CustomMatrix<T1,AF1,PF1,SO>, CustomMatrix<T2,AF2,PF2,SO> >
{
   using Type = DynamicMatrix< SubTrait_<T1,T2>, SO >;
};

template< typename T1, bool AF1, bool PF1, bool SO1, typename T2, bool AF2, bool PF2, bool SO2 >
struct SubTrait< CustomMatrix<T1,AF1,PF1,SO1>, CustomMatrix<T2,AF2,PF2,SO2> >
{
   using Type = DynamicMatrix< SubTrait_<T1,T2>, false >;
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
template< typename T1, bool AF, bool PF, bool SO, typename T2 >
struct MultTrait< CustomMatrix<T1,AF,PF,SO>, T2, EnableIf_<IsNumeric<T2> > >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, SO >;
};

template< typename T1, typename T2, bool AF, bool PF, bool SO >
struct MultTrait< T1, CustomMatrix<T2,AF,PF,SO>, EnableIf_<IsNumeric<T1> > >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, SO >;
};

template< typename T1, bool AF, bool PF, bool SO, typename T2, size_t N >
struct MultTrait< CustomMatrix<T1,AF,PF,SO>, StaticVector<T2,N,false> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, false >;
};

template< typename T1, size_t N, typename T2, bool AF, bool PF, bool SO >
struct MultTrait< StaticVector<T1,N,true>, CustomMatrix<T2,AF,PF,SO> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, true >;
};

template< typename T1, bool AF, bool PF, bool SO, typename T2, size_t N >
struct MultTrait< CustomMatrix<T1,AF,PF,SO>, HybridVector<T2,N,false> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, false >;
};

template< typename T1, size_t N, typename T2, bool AF, bool PF, bool SO >
struct MultTrait< HybridVector<T1,N,true>, CustomMatrix<T2,AF,PF,SO> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, true >;
};

template< typename T1, bool AF, bool PF, bool SO, typename T2 >
struct MultTrait< CustomMatrix<T1,AF,PF,SO>, DynamicVector<T2,false> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, false >;
};

template< typename T1, typename T2, bool AF, bool PF, bool SO >
struct MultTrait< DynamicVector<T1,true>, CustomMatrix<T2,AF,PF,SO> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, true >;
};

template< typename T1, bool AF1, bool PF1, bool SO, typename T2, bool AF2, bool PF2 >
struct MultTrait< CustomMatrix<T1,AF1,PF1,SO>, CustomVector<T2,AF2,PF2,false> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, false >;
};

template< typename T1, bool AF1, bool PF1, typename T2, bool AF2, bool PF2, bool SO >
struct MultTrait< CustomVector<T1,AF1,PF1,true>, CustomMatrix<T2,AF2,PF2,SO> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, true >;
};

template< typename T1, bool AF, bool PF, bool SO, typename T2 >
struct MultTrait< CustomMatrix<T1,AF,PF,SO>, CompressedVector<T2,false> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, false >;
};

template< typename T1, typename T2, bool AF, bool PF, bool SO >
struct MultTrait< CompressedVector<T1,true>, CustomMatrix<T2,AF,PF,SO> >
{
   using Type = DynamicVector< MultTrait_<T1,T2>, true >;
};

template< typename T1, bool AF, bool PF, bool SO1, typename T2, size_t M, size_t N, bool SO2 >
struct MultTrait< CustomMatrix<T1,AF,PF,SO1>, StaticMatrix<T2,M,N,SO2> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, SO1 >;
};

template< typename T1, size_t M, size_t N, bool SO1, typename T2, bool AF, bool PF, bool SO2 >
struct MultTrait< StaticMatrix<T1,M,N,SO1>, CustomMatrix<T2,AF,PF,SO2> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, SO1 >;
};

template< typename T1, bool AF, bool PF, bool SO1, typename T2, size_t M, size_t N, bool SO2 >
struct MultTrait< CustomMatrix<T1,AF,PF,SO1>, HybridMatrix<T2,M,N,SO2> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, SO1 >;
};

template< typename T1, size_t M, size_t N, bool SO1, typename T2, bool AF, bool PF, bool SO2 >
struct MultTrait< HybridMatrix<T1,M,N,SO1>, CustomMatrix<T2,AF,PF,SO2> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, SO1 >;
};

template< typename T1, bool AF, bool PF, bool SO1, typename T2, bool SO2 >
struct MultTrait< CustomMatrix<T1,AF,PF,SO1>, DynamicMatrix<T2,SO2> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, SO1 >;
};

template< typename T1, bool SO1, typename T2, bool AF, bool PF, bool SO2 >
struct MultTrait< DynamicMatrix<T1,SO1>, CustomMatrix<T2,AF,PF,SO2> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, SO1 >;
};

template< typename T1, bool AF1, bool PF1, bool SO1, typename T2, bool AF2, bool PF2, bool SO2 >
struct MultTrait< CustomMatrix<T1,AF1,PF1,SO1>, CustomMatrix<T2,AF2,PF2,SO2> >
{
   using Type = DynamicMatrix< MultTrait_<T1,T2>, SO1 >;
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
template< typename T1, bool AF, bool PF, bool SO, typename T2 >
struct DivTrait< CustomMatrix<T1,AF,PF,SO>, T2, EnableIf_<IsNumeric<T2> > >
{
   using Type = DynamicMatrix< DivTrait_<T1,T2>, SO >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBMATRIXTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, bool AF, bool PF, bool SO >
struct SubmatrixTrait< CustomMatrix<T1,AF,PF,SO> >
{
   using Type = DynamicMatrix<T1,SO>;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ROWTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, bool AF, bool PF, bool SO >
struct RowTrait< CustomMatrix<T1,AF,PF,SO> >
{
   using Type = DynamicVector<T1,true>;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COLUMNTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, bool AF, bool PF, bool SO >
struct ColumnTrait< CustomMatrix<T1,AF,PF,SO> >
{
   using Type = DynamicVector<T1,false>;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
