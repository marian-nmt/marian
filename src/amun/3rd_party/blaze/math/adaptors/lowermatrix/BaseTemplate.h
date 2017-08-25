//=================================================================================================
/*!
//  \file blaze/math/adaptors/lowermatrix/BaseTemplate.h
//  \brief Header file for the implementation of the base template of the LowerMatrix
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

#ifndef _BLAZE_MATH_ADAPTORS_LOWERMATRIX_BASETEMPLATE_H_
#define _BLAZE_MATH_ADAPTORS_LOWERMATRIX_BASETEMPLATE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/typetraits/IsColumnMajorMatrix.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup lower_matrix LowerMatrix
// \ingroup adaptors
*/
/*!\brief Matrix adapter for lower triangular \f$ N \times N \f$ matrices.
// \ingroup lower_matrix
//
// \section lowermatrix_general General
//
// The LowerMatrix class template is an adapter for existing dense and sparse matrix types. It
// inherits the properties and the interface of the given matrix type \a MT and extends it by
// enforcing the additional invariant that all matrix elements above the diagonal are 0 (lower
// triangular matrix). The type of the adapted matrix can be specified via the first template
// parameter:

   \code
   template< typename MT, bool SO, bool DF >
   class LowerMatrix;
   \endcode

//  - MT: specifies the type of the matrix to be adapted. LowerMatrix can be used with any
//        non-cv-qualified, non-reference, non-pointer, non-expression dense or sparse matrix
//        type. Note that the given matrix type must be either resizable (as for instance
//        HybridMatrix or DynamicMatrix) or must be square at compile time (as for instance
//        StaticMatrix).
//  - SO: specifies the storage order (blaze::rowMajor, blaze::columnMajor) of the matrix. This
//        template parameter doesn't have to be explicitly defined, but is automatically derived
//        from the first template parameter.
//  - DF: specifies whether the given matrix type is a dense or sparse matrix type. This template
//        parameter doesn't have to be defined explicitly, it is automatically derived from the
//        first template parameter. Defining the parameter explicitly may result in a compilation
//        error!
//
// The following examples give an impression of several possible lower matrices:

   \code
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;
   using blaze::columnMajor;

   // Definition of a 3x3 row-major dense lower matrix with static memory
   blaze::LowerMatrix< blaze::StaticMatrix<int,3UL,3UL,rowMajor> > A;

   // Definition of a resizable column-major dense lower matrix based on HybridMatrix
   blaze::LowerMatrix< blaze::HybridMatrix<float,4UL,4UL,columnMajor> B;

   // Definition of a resizable row-major dense lower matrix based on DynamicMatrix
   blaze::LowerMatrix< blaze::DynamicMatrix<double,rowMajor> > C;

   // Definition of a fixed-size row-major dense lower matrix based on CustomMatrix
   blaze::LowerMatrix< blaze::CustomMatrix<double,unaligned,unpadded,rowMajor> > D;

   // Definition of a compressed row-major single precision lower matrix
   blaze::LowerMatrix< blaze::CompressedMatrix<float,rowMajor> > E;
   \endcode

// The storage order of a lower matrix is depending on the storage order of the adapted matrix
// type \a MT. In case the adapted matrix is stored in a row-wise fashion (i.e. is specified
// as blaze::rowMajor), the lower matrix will also be a row-major matrix. Otherwise, if the
// adapted matrix is column-major (i.e. is specified as blaze::columnMajor), the lower matrix
// will also be a column-major matrix.
//
//
// \n \section lowermatrix_special_properties Special Properties of Lower Triangular Matrices
//
// A lower matrix is used exactly like a matrix of the underlying, adapted matrix type \a MT. It
// also provides (nearly) the same interface as the underlying matrix type. However, there are
// some important exceptions resulting from the lower matrix constraint:
//
//  -# <b>\ref lowermatrix_square</b>
//  -# <b>\ref lowermatrix_lower</b>
//  -# <b>\ref lowermatrix_initialization</b>
//  -# <b>\ref lowermatrix_storage</b>
//
// \n \subsection lowermatrix_square Lower Matrices Must Always be Square!
//
// In case a resizable matrix is used (as for instance blaze::HybridMatrix, blaze::DynamicMatrix,
// or blaze::CompressedMatrix), this means that the according constructors, the \c resize() and
// the \c extend() functions only expect a single parameter, which specifies both the number of
// rows and columns, instead of two (one for the number of rows and one for the number of columns):

   \code
   using blaze::DynamicMatrix;
   using blaze::LowerMatrix;
   using blaze::rowMajor;

   // Default constructed, default initialized, row-major 3x3 lower dynamic matrix
   LowerMatrix< DynamicMatrix<double,rowMajor> > A( 3 );

   // Resizing the matrix to 5x5
   A.resize( 5 );

   // Extending the number of rows and columns by 2, resulting in a 7x7 matrix
   A.extend( 2 );
   \endcode

// In case a matrix with a fixed size is used (as for instance blaze::StaticMatrix), the number
// of rows and number of columns must be specified equally:

   \code
   using blaze::StaticMatrix;
   using blaze::LowerMatrix;
   using blaze::columnMajor;

   // Correct setup of a fixed size column-major 3x3 lower static matrix
   LowerMatrix< StaticMatrix<int,3UL,3UL,columnMajor> > A;

   // Compilation error: the provided matrix type is not a square matrix type
   LowerMatrix< StaticMatrix<int,3UL,4UL,columnMajor> > B;
   \endcode

// \n \subsection lowermatrix_lower The Lower Matrix Property is Always Enforced!
//
// This means that it is only allowed to modify elements in the lower part or the diagonal of
// the matrix, but not the elements in the upper part of the matrix. Also, it is only possible
// to assign matrices that are lower matrices themselves:

   \code
   using blaze::CompressedMatrix;
   using blaze::DynamicMatrix;
   using blaze::StaticMatrix;
   using blaze::LowerMatrix;
   using blaze::rowMajor;

   typedef LowerMatrix< CompressedMatrix<double,rowMajor> >  CompressedLower;

   // Default constructed, row-major 3x3 lower compressed matrix
   CompressedLower A( 3 );

   // Initializing elements via the function call operator
   A(0,0) = 1.0;  // Initialization of the diagonal element (0,0)
   A(2,0) = 2.0;  // Initialization of the lower element (2,0)
   A(1,2) = 9.0;  // Throws an exception; invalid modification of upper element

   // Inserting two more elements via the insert() function
   A.insert( 1, 0, 3.0 );  // Inserting the lower element (1,0)
   A.insert( 2, 1, 4.0 );  // Inserting the lower element (2,1)
   A.insert( 0, 2, 9.0 );  // Throws an exception; invalid insertion of upper element

   // Appending an element via the append() function
   A.reserve( 1, 3 );      // Reserving enough capacity in row 1
   A.append( 1, 1, 5.0 );  // Appending the diagonal element (1,1)
   A.append( 1, 2, 9.0 );  // Throws an exception; appending an element in the upper part

   // Access via a non-const iterator
   CompressedLower::Iterator it = A.begin(1);
   *it = 6.0;  // Modifies the lower element (1,0)
   ++it;
   *it = 9.0;  // Modifies the diagonal element (1,1)

   // Erasing elements via the erase() function
   A.erase( 0, 0 );  // Erasing the diagonal element (0,0)
   A.erase( 2, 0 );  // Erasing the lower element (2,0)

   // Construction from a lower dense matrix
   StaticMatrix<double,3UL,3UL> B(  3.0,  0.0,  0.0,
                                    8.0,  0.0,  0.0,
                                   -2.0, -1.0,  4.0 );

   LowerMatrix< DynamicMatrix<double,rowMajor> > C( B );  // OK

   // Assignment of a non-lower dense matrix
   StaticMatrix<double,3UL,3UL> D(  3.0,  0.0, -2.0,
                                    8.0,  0.0,  0.0,
                                   -2.0, -1.0,  4.0 );

   C = D;  // Throws an exception; lower matrix invariant would be violated!
   \endcode

// The lower matrix property is also enforced for lower custom matrices: In case the given array
// of elements does not represent a lower matrix, a \a std::invalid_argument exception is thrown:

   \code
   using blaze::CustomMatrix;
   using blaze::LowerMatrix;
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;

   typedef LowerMatrix< CustomMatrix<double,unaligned,unpadded,rowMajor> >  CustomLower;

   // Creating a 3x3 lower custom matrix from a properly initialized array
   double array[9] = { 1.0, 0.0, 0.0,
                       2.0, 3.0, 0.0,
                       4.0, 5.0, 6.0 };
   CustomLower A( array, 3UL );  // OK

   // Attempt to create a second 3x3 lower custom matrix from an uninitialized array
   CustomLower B( new double[9UL], 3UL, blaze::ArrayDelete() );  // Throws an exception
   \endcode

// Finally, the lower matrix property is enforced for views (rows, columns, submatrices, ...) on
// the lower matrix. The following example demonstrates that modifying the elements of an entire
// row and submatrix of a lower matrix only affects the lower and diagonal matrix elements:

   \code
   using blaze::DynamicMatrix;
   using blaze::LowerMatrix;

   // Setup of the lower matrix
   //
   //       ( 0 0 0 0 )
   //   A = ( 1 2 0 0 )
   //       ( 0 3 0 0 )
   //       ( 4 0 5 0 )
   //
   LowerMatrix< DynamicMatrix<int> > A( 4 );
   A(1,0) = 1;
   A(1,1) = 2;
   A(2,1) = 3;
   A(3,0) = 4;
   A(3,2) = 5;

   // Setting the lower and diagonal elements in the 2nd row to 9 results in the matrix
   //
   //       ( 0 0 0 0 )
   //   A = ( 1 2 0 0 )
   //       ( 9 9 9 0 )
   //       ( 4 0 5 0 )
   //
   row( A, 2 ) = 9;

   // Setting the lower and diagonal elements in the 1st and 2nd column to 7 results in
   //
   //       ( 0 0 0 0 )
   //   A = ( 1 7 0 0 )
   //       ( 9 7 7 0 )
   //       ( 4 7 7 0 )
   //
   submatrix( A, 0, 1, 4, 2 ) = 7;
   \endcode

// The next example demonstrates the (compound) assignment to rows/columns and submatrices of
// lower matrices. Since only lower and diagonal elements may be modified the matrix to be
// assigned must be structured such that the lower matrix invariant of the lower matrix is
// preserved. Otherwise a \a std::invalid_argument exception is thrown:

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::LowerMatrix;
   using blaze::rowVector;

   // Setup of two default 4x4 lower matrices
   LowerMatrix< DynamicMatrix<int> > A1( 4 ), A2( 4 );

   // Setup of a 4-dimensional vector
   //
   //   v = ( 1 2 3 0 )
   //
   DynamicVector<int,rowVector> v( 4, 0 );
   v[0] = 1;
   v[1] = 2;
   v[2] = 3;

   // OK: Assigning v to the 2nd row of A1 preserves the lower matrix invariant
   //
   //        ( 0 0 0 0 )
   //   A1 = ( 0 0 0 0 )
   //        ( 1 2 3 0 )
   //        ( 0 0 0 0 )
   //
   row( A1, 2 ) = v;  // OK

   // Error: Assigning v to the 1st row of A1 violates the lower matrix invariant! The element
   //   marked with X cannot be assigned and triggers an exception.
   //
   //        ( 0 0 0 0 )
   //   A1 = ( 1 2 X 0 )
   //        ( 1 2 3 0 )
   //        ( 0 0 0 0 )
   //
   row( A1, 1 ) = v;  // Assignment throws an exception!

   // Setup of the 3x2 dynamic matrix
   //
   //       ( 0 0 )
   //   B = ( 7 0 )
   //       ( 8 9 )
   //
   DynamicMatrix<int> B( 3UL, 2UL, 0 );
   B(1,0) = 7;
   B(2,0) = 8;
   B(2,1) = 9;

   // OK: Assigning B to a submatrix of A2 such that the lower matrix invariant can be preserved
   //
   //        ( 0 0 0 0 )
   //   A2 = ( 0 7 0 0 )
   //        ( 0 8 9 0 )
   //        ( 0 0 0 0 )
   //
   submatrix( A2, 0UL, 1UL, 3UL, 2UL ) = B;  // OK

   // Error: Assigning B to a submatrix of A2 such that the lower matrix invariant cannot be
   //   preserved! The elements marked with X cannot be assigned without violating the invariant!
   //
   //        ( 0 0 0 0 )
   //   A2 = ( 0 7 X 0 )
   //        ( 0 8 8 X )
   //        ( 0 0 0 0 )
   //
   submatrix( A2, 0UL, 2UL, 3UL, 2UL ) = B;  // Assignment throws an exception!
   \endcode

// \n \subsection lowermatrix_initialization The Upper Elements of a Dense Lower Matrix are Always Default Initialized!
//
// Although this results in a small loss of efficiency during the creation of a dense lower matrix
// this initialization is important since otherwise the lower matrix property of dense lower
// matrices would not be guaranteed:

   \code
   using blaze::DynamicMatrix;
   using blaze::LowerMatrix;

   // Uninitialized, 5x5 row-major dynamic matrix
   DynamicMatrix<int,rowMajor> A( 5, 5 );

   // 5x5 row-major lower dynamic matrix with default initialized upper matrix
   LowerMatrix< DynamicMatrix<int,rowMajor> > B( 5 );
   \endcode

// \n \subsection lowermatrix_storage Dense Lower Matrices Also Store the Upper Elements!
//
// It is important to note that dense lower matrices store all elements, including the elements
// in the upper part of the matrix, and therefore don't provide any kind of memory reduction!
// There are two main reasons for this: First, storing also the upper elements guarantees maximum
// performance for many algorithms that perform vectorized operations on the lower matrix, which
// is especially true for small dense matrices. Second, conceptually the LowerMatrix adaptor
// merely restricts the interface to the matrix type \a MT and does not change the data layout
// or the underlying matrix type.
//
//
// \n \section lowermatrix_arithmetic_operations Arithmetic Operations
//
// A LowerMatrix matrix can participate in numerical operations in any way any other dense or
// sparse matrix can participate. It can also be combined with any other dense or sparse vector
// or matrix. The following code example gives an impression of the use of LowerMatrix within
// arithmetic operations:

   \code
   using blaze::LowerMatrix;
   using blaze::DynamicMatrix;
   using blaze::HybridMatrix;
   using blaze::StaticMatrix;
   using blaze::CompressedMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   DynamicMatrix<double,rowMajor> A( 3, 3 );
   CompressedMatrix<double,rowMajor> B( 3, 3 );

   LowerMatrix< DynamicMatrix<double,rowMajor> > C( 3 );
   LowerMatrix< CompressedMatrix<double,rowMajor> > D( 3 );

   LowerMatrix< HybridMatrix<float,3UL,3UL,rowMajor> > E;
   LowerMatrix< StaticMatrix<float,3UL,3UL,columnMajor> > F;

   E = A + B;     // Matrix addition and assignment to a row-major lower matrix
   F = C - D;     // Matrix subtraction and assignment to a column-major lower matrix
   F = A * D;     // Matrix multiplication between a dense and a sparse matrix

   C *= 2.0;      // In-place scaling of matrix C
   E  = 2.0 * B;  // Scaling of matrix B
   F  = C * 2.0;  // Scaling of matrix C

   E += A - B;    // Addition assignment
   F -= C + D;    // Subtraction assignment
   F *= A * D;    // Multiplication assignment
   \endcode

// \n \section lowermatrix_block_structured Block-Structured Lower Matrices
//
// It is also possible to use block-structured lower matrices:

   \code
   using blaze::CompressedMatrix;
   using blaze::StaticMatrix;
   using blaze::LowerMatrix;

   // Definition of a 5x5 block-structured lower matrix based on CompressedMatrix
   LowerMatrix< CompressedMatrix< StaticMatrix<int,3UL,3UL> > > A( 5 );
   \endcode

// Also in this case the lower matrix invariant is enforced, i.e. it is not possible to manipulate
// elements in the upper part of the matrix:

   \code
   const StaticMatrix<int,3UL,3UL> B( { { 1, -4,  5 },
                                        { 6,  8, -3 },
                                        { 2, -1,  2 } } )

   A.insert( 4, 2, B );  // Inserting the elements (4,2)
   A(2,4)(1,1) = -5;     // Invalid manipulation of upper matrix element; Results in an exception
   \endcode

// \n \section lowermatrix_performance Performance Considerations
//
// The \b Blaze library tries to exploit the properties of lower matrices whenever and wherever
// possible. Thus using a lower triangular matrix instead of a general matrix can result in a
// considerable performance improvement. However, there are also situations when using a lower
// triangular matrix introduces some overhead. The following examples demonstrate several common
// situations where lower matrices can positively or negatively impact performance.
//
// \n \subsection lowermatrix_matrix_matrix_multiplication Positive Impact: Matrix/Matrix Multiplication
//
// When multiplying two matrices, at least one of which is lower triangular, \b Blaze can exploit
// the fact that the upper part of the matrix contains only default elements and restrict the
// algorithm to the lower and diagonal elements. The following example demonstrates this by means
// of a dense matrix/dense matrix multiplication:

   \code
   using blaze::DynamicMatrix;
   using blaze::LowerMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   LowerMatrix< DynamicMatrix<double,rowMajor> > A;
   LowerMatrix< DynamicMatrix<double,columnMajor> > B;
   DynamicMatrix<double,columnMajor> C;

   // ... Resizing and initialization

   C = A * B;
   \endcode

// In comparison to a general matrix multiplication, the performance advantage is significant,
// especially for large matrices. Therefore is it highly recommended to use the LowerMatrix
// adaptor when a matrix is known to be lower triangular. Note however that the performance
// advantage is most pronounced for dense matrices and much less so for sparse matrices.
//
// \n \subsection lowermatrix_matrix_vector_multiplication Positive Impact: Matrix/Vector Multiplication
//
// A similar performance improvement can be gained when using a lower matrix in a matrix/vector
// multiplication:

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::LowerMatrix;
   using blaze::rowMajor;
   using blaze::columnVector;

   LowerMatrix< DynamicMatrix<double,rowMajor> > A;
   DynamicVector<double,columnVector> x, y;

   // ... Resizing and initialization

   y = A * x;
   \endcode

// In this example, \b Blaze also exploits the structure of the matrix and approx. halves the
// runtime of the multiplication. Also in case of matrix/vector multiplications the performance
// improvement is most pronounced for dense matrices and much less so for sparse matrices.
//
// \n \subsection lowermatrix_assignment Negative Impact: Assignment of a General Matrix
//
// In contrast to using a lower triangular matrix on the right-hand side of an assignment (i.e.
// for read access), which introduces absolutely no performance penalty, using a lower matrix on
// the left-hand side of an assignment (i.e. for write access) may introduce additional overhead
// when it is assigned a general matrix, which is not lower triangular at compile time:

   \code
   using blaze::DynamicMatrix;
   using blaze::LowerMatrix;

   LowerMatrix< DynamicMatrix<double> > A, C;
   DynamicMatrix<double> B;

   B = A;  // Only read-access to the lower matrix; no performance penalty
   C = A;  // Assignment of a lower matrix to another lower matrix; no runtime overhead
   C = B;  // Assignment of a general matrix to a lower matrix; some runtime overhead
   \endcode

// When assigning a general, potentially not lower matrix to a lower matrix it is necessary to
// check whether the matrix is lower at runtime in order to guarantee the lower triangular property
// of the lower matrix. In case it turns out to be lower triangular, it is assigned as efficiently
// as possible, if it is not, an exception is thrown. In order to prevent this runtime overhead it
// is therefore generally advisable to assign lower matrices to other lower matrices.\n
// In this context it is especially noteworthy that the addition, subtraction, and multiplication
// of two lower triangular matrices always results in another lower matrix:

   \code
   LowerMatrix< DynamicMatrix<double> > A, B, C;

   C = A + B;  // Results in a lower matrix; no runtime overhead
   C = A - B;  // Results in a lower matrix; no runtime overhead
   C = A * B;  // Results in a lower matrix; no runtime overhead
   \endcode
*/
template< typename MT                               // Type of the adapted matrix
        , bool SO = IsColumnMajorMatrix<MT>::value  // Storage order of the adapted matrix
        , bool DF = IsDenseMatrix<MT>::value >      // Density flag
class LowerMatrix
{};
//*************************************************************************************************

} // namespace blaze

#endif
