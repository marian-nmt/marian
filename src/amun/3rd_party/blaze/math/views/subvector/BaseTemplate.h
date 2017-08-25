//=================================================================================================
/*!
//  \file blaze/math/views/subvector/BaseTemplate.h
//  \brief Header file for the implementation of the Subvector base template
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

#ifndef _BLAZE_MATH_VIEWS_SUBVECTOR_BASETEMPLATE_H_
#define _BLAZE_MATH_VIEWS_SUBVECTOR_BASETEMPLATE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsRowVector.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup subvector Subvector
// \ingroup views
*/
/*!\brief View on a specific subvector of a dense or sparse vector.
// \ingroup subvector
//
// The Subvector class template represents a view on a specific subvector of a dense or sparse
// vector primitive. The type of the vector is specified via the first template parameter:

   \code
   template< typename VT, bool AF, bool TF, bool DF >
   class Subvector;
   \endcode

//  - VT: specifies the type of the vector primitive. Subvector can be used with every vector
//        primitive or view, but does not work with any vector expression type.
//  - AF: the alignment flag specifies whether the subvector is aligned (\a blaze::aligned) or
//        unaligned (\a blaze::unaligned). The default value is \a blaze::unaligned.
//  - TF: specifies whether the vector is a row vector (\a blaze::rowVector) or a column
//        vector (\a blaze::columnVector). This template parameter doesn't have to be explicitly
//        defined, but is automatically derived from the first template parameter.
//  - DF: specifies whether the given vector type is a dense or sparse vector type. This template
//        parameter doesn't have to be defined explicitly, it is automatically derived from the
//        first template parameter. Defining the parameter explicitly may result in a compilation
//        error!
//
//
// \n \section subvector_setup Setup of Subvectors
//
// A view on a dense or sparse subvector can be created very conveniently via the \c subvector()
// function:

   \code
   using DenseVectorType = blaze::DynamicVector<double,blaze::rowVector>;

   DenseVectorType x;
   // ... Resizing and initialization

   // Create a dense subvector from index 8 with a size of 16 (i.e. in the range [8..23])
   blaze::Subvector<DenseVectorType> sv = subvector( x, 8UL, 16UL );
   \endcode

   \code
   using SparseVectorType = blaze::CompressedVector<double,blaze::rowVector>;

   SparseVectorType x;
   // ... Resizing and initialization

   // Create a sparse subvector from index 5 with a size of 7 (i.e. in the range [5..11])
   blaze::Subvector<SparseVectorType> sv = subvector( x, 5UL, 7UL );
   \endcode

// This view can be treated as any other dense or sparse vector, i.e. it can be assigned to, it
// can be copied from, and it can be used in arithmetic operations. The view can also be used on
// both sides of an assignment: The subvector can either be used as an alias to grant write access
// to a specific subvector of a vector primitive on the left-hand side of an assignment or to grant
// read-access to a specific subvector of a vector primitive or expression on the right-hand side
// of an assignment. The following example demonstrates this in detail:

   \code
   using DenseVectorType  = blaze::DynamicVector<double,blaze::rowVector>;
   using SparseVectorType = blaze::CompressedVector<double,blaze::rowVector>;
   using DenseMatrixType  = blaze::DynamicMatrix<double,blaze::rowMajor>;

   DenseVectorType  x;
   SparseVectorType y;
   DenseMatrixType  A;
   // ... Resizing and initialization

   // Create a subvector from index 0 with a size of 10 (i.e. in the range [0..9])
   blaze::Subvector<DenseVectorType> sv = subvector( x, 0UL, 10UL );

   // Setting the first ten elements of x to the 2nd row of matrix A
   sv = row( A, 2UL );

   // Setting the second ten elements of x to y
   subvector( x, 10UL, 10UL ) = y;

   // Setting the 3rd row of A to a subvector of x
   row( A, 3UL ) = subvector( x, 3UL, 10UL );

   // Setting x to a subvector of the result of the addition between y and the 1st row of A
   x = subvector( y + row( A, 1UL ), 2UL, 5UL )
   \endcode

// \n \section subvector_element_access Element access
//
// A subvector can be used like any other dense or sparse vector. For instance, the elements of
// the subvector can be directly accessed with the subscript operator.

   \code
   using VectorType = blaze::DynamicVector<double,blaze::rowVector>;
   VectorType v;
   // ... Resizing and initialization

   // Creating an 8-dimensional subvector, starting from index 4
   blaze::Subvector<VectorType> sv = subvector( v, 4UL, 8UL );

   // Setting the 1st element of the subvector, which corresponds to
   // the element at index 5 in vector v
   sv[1] = 2.0;
   \endcode

// The numbering of the subvector elements is

                             \f[\left(\begin{array}{*{5}{c}}
                             0 & 1 & 2 & \cdots & N-1 \\
                             \end{array}\right),\f]

// where N is the specified size of the subvector. Alternatively, the elements of a subvector can
// be traversed via iterators. Just as with vectors, in case of non-const subvectors, \c begin()
// and \c end() return an Iterator, which allows a manipulation of the non-zero values, in case
// of constant subvectors a ConstIterator is returned:

   \code
   using VectorType    = blaze::DynamicVector<int,blaze::rowVector>;
   using SubvectorType = blaze::Subvector<VectorType>;

   VectorType v( 256UL );
   // ... Resizing and initialization

   // Creating a reference to a specific subvector of vector v
   SubvectorType sv = subvector( v, 16UL, 64UL );

   for( SubvectorType::Iterator it=sv.begin(); it!=sv.end(); ++it ) {
      *it = ...;  // OK: Write access to the dense subvector value.
      ... = *it;  // OK: Read access to the dense subvector value.
   }

   for( SubvectorType::ConstIterator it=sv.begin(); it!=sv.end(); ++it ) {
      *it = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = *it;  // OK: Read access to the dense subvector value.
   }
   \endcode

   \code
   using VectorType    = blaze::CompressedVector<int,blaze::rowVector>;
   using SubvectorType = blaze::Subvector<VectorType>;

   VectorType v( 256UL );
   // ... Resizing and initialization

   // Creating a reference to a specific subvector of vector v
   SubvectorType sv = subvector( v, 16UL, 64UL );

   for( SubvectorType::Iterator it=sv.begin(); it!=sv.end(); ++it ) {
      it->value() = ...;  // OK: Write access to the value of the non-zero element.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }

   for( SubvectorType::ConstIterator it=sv.begin(); it!=sv.end(); ++it ) {
      it->value() = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }
   \endcode

// \n \section subvector_element_insertion Element Insertion
//
// Inserting/accessing elements in a sparse subvector can be done by several alternative functions.
// The following example demonstrates all options:

   \code
   using VectorType = blaze::CompressedVector<double,blaze::rowVector>;
   VectorType v( 256UL );  // Non-initialized vector of size 256

   using SubvectorType = blaze::Subvector<VectorType>;
   SubvectorType sv = subvector( v, 10UL, 60UL );  // View on the range [10..69] of v

   // The subscript operator provides access to all possible elements of the sparse subvector,
   // including the zero elements. In case the subscript operator is used to access an element
   // that is currently not stored in the sparse subvector, the element is inserted into the
   // subvector.
   sv[42] = 2.0;

   // The second operation for inserting elements is the set() function. In case the element is
   // not contained in the subvector it is inserted into the subvector, if it is already contained
   // in the subvector its value is modified.
   sv.set( 45UL, -1.2 );

   // An alternative for inserting elements into the subvector is the insert() function. However,
   // it inserts the element only in case the element is not already contained in the subvector.
   sv.insert( 50UL, 3.7 );

   // Just as in case of vectors, elements can also be inserted via the append() function. In
   // case of subvectors, append() also requires that the appended element's index is strictly
   // larger than the currently largest non-zero index of the subvector and that the subvector's
   // capacity is large enough to hold the new element. Note however that due to the nature of
   // a subvector, which may be an alias to the middle of a sparse vector, the append() function
   // does not work as efficiently for a subvector as it does for a vector.
   sv.reserve( 10UL );
   sv.append( 51UL, -2.1 );
   \endcode

// \n \section subvector_common_operations Common Operations
//
// The current number of subvector elements can be obtained via the \c size() function, the
// current capacity via the \c capacity() function, and the number of non-zero elements via
// the \c nonZeros() function. However, since subvector are views on a specific subvector of
// a vector, several operations are not possible on views, such as resizing and swapping:

   \code
   using VectorType    = blaze::DynamicVector<int,blaze::rowVector>;
   using SubvectorType = blaze::Subvector<VectorType>;

   VectorType v( 42UL );
   // ... Resizing and initialization

   // Creating a view on the range [5..15] of vector v
   SubvectorType sv = subvector( v, 5UL, 10UL );

   sv.size();          // Returns the number of elements in the subvector
   sv.capacity();      // Returns the capacity of the subvector
   sv.nonZeros();      // Returns the number of non-zero elements contained in the subvector

   sv.resize( 84UL );  // Compilation error: Cannot resize a subvector of a vector

   SubvectorType sv2 = subvector( v, 15UL, 10UL );
   swap( sv, sv2 );   // Compilation error: Swap operation not allowed
   \endcode

// \n \section subvector_arithmetic_operations Arithmetic Operations
//
// The following example gives an impression of the use of Subvector within arithmetic operations.
// All operations (addition, subtraction, multiplication, scaling, ...) can be performed on all
// possible combinations of dense and sparse vectors with fitting element types:

   \code
   using DenseVectorType  = blaze::DynamicVector<double,blaze::rowVector>;
   using SparseVectorType = blaze::CompressedVector<double,blaze::rowVector>;
   DenseVectorType d1, d2, d3;
   SparseVectorType s1, s2;

   // ... Resizing and initialization

   using DenseMatrixType = blaze::DynamicMatrix<double,blaze::rowMajor>;
   DenseMatrixType A;

   using SubvectorType = blaze::Subvector<DenseVectorType>;
   SubvectorType sv( subvector( d1, 0UL, 10UL ) );  // View on the range [0..9] of vector d1

   sv = d2;                           // Dense vector initialization of the range [0..9]
   subvector( d1, 10UL, 10UL ) = s1;  // Sparse vector initialization of the range [10..19]

   d3 = sv + d2;                           // Dense vector/dense vector addition
   s2 = s1 + subvector( d1, 10UL, 10UL );  // Sparse vector/dense vector addition
   d2 = sv * subvector( d1, 20UL, 10UL );  // Component-wise vector multiplication

   subvector( d1, 3UL, 4UL ) *= 2.0;      // In-place scaling of the range [3..6]
   d2 = subvector( d1, 7UL, 3UL ) * 2.0;  // Scaling of the range [7..9]
   d2 = 2.0 * subvector( d1, 7UL, 3UL );  // Scaling of the range [7..9]

   subvector( d1, 0UL , 10UL ) += d2;  // Addition assignment
   subvector( d1, 10UL, 10UL ) -= s2;  // Subtraction assignment
   subvector( d1, 20UL, 10UL ) *= sv;  // Multiplication assignment

   double scalar = subvector( d1, 5UL, 10UL ) * trans( s1 );  // Scalar/dot/inner product between two vectors

   A = trans( s1 ) * subvector( d1, 4UL, 16UL );  // Outer product between two vectors
   \endcode

// \n \section subvector_aligned_subvector Aligned Subvectors
//
// Usually subvectors can be defined anywhere within a vector. They may start at any position and
// may have an arbitrary size (only restricted by the size of the underlying vector). However, in
// contrast to vectors themselves, which are always properly aligned in memory and therefore can
// provide maximum performance, this means that subvectors in general have to be considered to be
// unaligned. This can be made explicit by the \a blaze::unaligned flag:

   \code
   using blaze::unaligned;

   using DenseVectorType = blaze::DynamicVector<double,blaze::rowVector>;

   DenseVectorType x;
   // ... Resizing and initialization

   // Identical creations of an unaligned subvector in the range [8..23]
   blaze::Subvector<DenseVectorType>           sv1 = subvector           ( x, 8UL, 16UL );
   blaze::Subvector<DenseVectorType>           sv2 = subvector<unaligned>( x, 8UL, 16UL );
   blaze::Subvector<DenseVectorType,unaligned> sv3 = subvector           ( x, 8UL, 16UL );
   blaze::Subvector<DenseVectorType,unaligned> sv4 = subvector<unaligned>( x, 8UL, 16UL );
   \endcode

// All of these calls to the \c subvector() function are identical. Whether the alignment flag is
// explicitly specified or not, it always returns an unaligned subvector. Whereas this may provide
// full flexibility in the creation of subvectors, this might result in performance restrictions
// (even in case the specified subvector could be aligned). However, it is also possible to create
// aligned subvectors. Aligned subvectors are identical to unaligned subvectors in all aspects,
// except that they may pose additional alignment restrictions and therefore have less flexibility
// during creation, but don't suffer from performance penalties and provide the same performance
// as the underlying vector. Aligned subvectors are created by explicitly specifying the
// \a blaze::aligned flag:

   \code
   using blaze::aligned;

   // Creating an aligned subvector in the range [8..23]
   blaze::Subvector<DenseVectorType,aligned> sv = subvector<aligned>( x, 8UL, 16UL );
   \endcode

// The alignment restrictions refer to system dependent address restrictions for the used element
// type and the available vectorization mode (SSE, AVX, ...). The following source code gives some
// examples for a double precision dense vector, assuming that AVX is available, which packs 4
// \c double values into a SIMD vector:

   \code
   using blaze::columnVector;

   using VectorType    = blaze::DynamicVector<double,columnVector>;
   using SubvectorType = blaze::Subvector<VectorType,aligned>;

   VectorType d( 17UL );
   // ... Resizing and initialization

   // OK: Starts at the beginning and the size is a multiple of 4
   SubvectorType dsv1 = subvector<aligned>( d, 0UL, 12UL );

   // OK: Start index and the size are both a multiple of 4
   SubvectorType dsv2 = subvector<aligned>( d, 4UL, 8UL );

   // OK: The start index is a multiple of 4 and the subvector includes the last element
   SubvectorType dsv3 = subvector<aligned>( d, 8UL, 9UL );

   // Error: Start index is not a multiple of 4
   SubvectorType dsv4 = subvector<aligned>( d, 5UL, 8UL );

   // Error: Size is not a multiple of 4 and the subvector does not include the last element
   SubvectorType dsv5 = subvector<aligned>( d, 8UL, 5UL );
   \endcode

// Note that the discussed alignment restrictions are only valid for aligned dense subvectors.
// In contrast, aligned sparse subvectors at this time don't pose any additional restrictions.
// Therefore aligned and unaligned sparse subvectors are truly fully identical. Still, in case
// the blaze::aligned flag is specified during setup, an aligned subvector is created:

   \code
   using blaze::aligned;

   using SparseVectorType = blaze::CompressedVector<double,blaze::rowVector>;

   SparseVectorType x;
   // ... Resizing and initialization

   // Creating an aligned subvector in the range [8..23]
   blaze::SparseSubvector<SparseVectorType,aligned> sv = subvector<aligned>( x, 8UL, 16UL );
   \endcode

// \n \section subvector_on_subvector Subvectors on Subvectors
//
// It is also possible to create a subvector view on another subvector. In this context it is
// important to remember that the type returned by the \c subvector() function is the same type
// as the type of the given subvector, since the view on a subvector is just another view on the
// underlying vector:

   \code
   using VectorType    = blaze::DynamicVector<double,blaze::rowVector>;
   using SubvectorType = blaze::Subvector<VectorType>;

   VectorType d1;

   // ... Resizing and initialization

   // Creating a subvector view on the dense vector d1
   SubvectorType sv1 = subvector( d1, 5UL, 10UL );

   // Creating a subvector view on the dense subvector sv1
   SubvectorType sv2 = subvector( sv1, 1UL, 5UL );
   \endcode
*/
template< typename VT                           // Type of the vector
        , bool AF = unaligned                   // Alignment flag
        , bool TF = IsRowVector<VT>::value      // Transpose flag
        , bool DF = IsDenseVector<VT>::value >  // Density flag
class Subvector
{};
//*************************************************************************************************

} // namespace blaze

#endif
