//=================================================================================================
/*!
//  \file blaze/math/adaptors/diagonalmatrix/BaseTemplate.h
//  \brief Header file for the implementation of the base template of the DiagonalMatrix
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

#ifndef _BLAZE_MATH_ADAPTORS_DIAGONALMATRIX_BASETEMPLATE_H_
#define _BLAZE_MATH_ADAPTORS_DIAGONALMATRIX_BASETEMPLATE_H_


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
/*!\defgroup diagonal_matrix DiagonalMatrix
// \ingroup adaptors
*/
/*!\brief Matrix adapter for diagonal \f$ N \times N \f$ matrices.
// \ingroup diagonal_matrix
//
// \section diagonalmatrix_general General
//
// The DiagonalMatrix class template is an adapter for existing dense and sparse matrix types.
// It inherits the properties and the interface of the given matrix type \a MT and extends it by
// enforcing the additional invariant that all matrix elements above and below the diagonal are
// 0 (diagonal matrix). The type of the adapted matrix can be specified via the first template
// parameter:

   \code
   template< typename MT, bool SO, bool DF >
   class DiagonalMatrix;
   \endcode

//  - MT: specifies the type of the matrix to be adapted. DiagonalMatrix can be used with any
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
// The following examples give an impression of several possible diagonal matrices:

   \code
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;
   using blaze::columnMajor;

   // Definition of a 3x3 row-major dense diagonal matrix with static memory
   blaze::DiagonalMatrix< blaze::StaticMatrix<int,3UL,3UL,rowMajor> > A;

   // Definition of a resizable column-major dense diagonal matrix based on HybridMatrix
   blaze::DiagonalMatrix< blaze::HybridMatrix<float,4UL,4UL,columnMajor> B;

   // Definition of a resizable row-major dense diagonal matrix based on DynamicMatrix
   blaze::DiagonalMatrix< blaze::DynamicMatrix<double,rowMajor> > C;

   // Definition of a fixed-size row-major dense diagonal matrix based on CustomMatrix
   blaze::DiagonalMatrix< blaze::CustomMatrix<double,unaligned,unpadded,rowMajor> > D;

   // Definition of a compressed row-major single precision diagonal matrix
   blaze::DiagonalMatrix< blaze::CompressedMatrix<float,rowMajor> > E;
   \endcode

// The storage order of a diagonal matrix is depending on the storage order of the adapted matrix
// type \a MT. In case the adapted matrix is stored in a row-wise fashion (i.e. is specified
// as blaze::rowMajor), the diagonal matrix will also be a row-major matrix. Otherwise, if the
// adapted matrix is column-major (i.e. is specified as blaze::columnMajor), the diagonal matrix
// will also be a column-major matrix.
//
//
// \n \section diagonalmatrix_special_properties Special Properties of Diagonal Matrices
//
// A diagonal matrix is used exactly like a matrix of the underlying, adapted matrix type \a MT.
// It also provides (nearly) the same interface as the underlying matrix type. However, there
// are some important exceptions resulting from the diagonal matrix constraint:
//
//  -# <b>\ref diagonalmatrix_square</b>
//  -# <b>\ref diagonalmatrix_diagonal</b>
//  -# <b>\ref diagonalmatrix_initialization</b>
//  -# <b>\ref diagonalmatrix_storage</b>
//
// \n \subsection diagonalmatrix_square Diagonal Matrices Must Always be Square!
//
// In case a resizable matrix is used (as for instance blaze::HybridMatrix, blaze::DynamicMatrix,
// or blaze::CompressedMatrix), this means that the according constructors, the \c resize() and
// the \c extend() functions only expect a single parameter, which specifies both the number of
// rows and columns, instead of two (one for the number of rows and one for the number of columns):

   \code
   using blaze::DynamicMatrix;
   using blaze::DiagonalMatrix;
   using blaze::rowMajor;

   // Default constructed, default initialized, row-major 3x3 diagonal dynamic matrix
   DiagonalMatrix< DynamicMatrix<double,rowMajor> > A( 3 );

   // Resizing the matrix to 5x5
   A.resize( 5 );

   // Extending the number of rows and columns by 2, resulting in a 7x7 matrix
   A.extend( 2 );
   \endcode

// In case a matrix with a fixed size is used (as for instance blaze::StaticMatrix), the number
// of rows and number of columns must be specified equally:

   \code
   using blaze::StaticMatrix;
   using blaze::DiagonalMatrix;
   using blaze::columnMajor;

   // Correct setup of a fixed size column-major 3x3 diagonal static matrix
   DiagonalMatrix< StaticMatrix<int,3UL,3UL,columnMajor> > A;

   // Compilation error: the provided matrix type is not a square matrix type
   DiagonalMatrix< StaticMatrix<int,3UL,4UL,columnMajor> > B;
   \endcode

// \n \subsection diagonalmatrix_diagonal The Diagonal Matrix Property is Always Enforced!
//
// This means that it is only allowed to modify elements on the the diagonal of the matrix, but
// not the elements in the lower or upper part of the matrix. Also, it is only possible to assign
// matrices that are diagonal matrices themselves:

   \code
   using blaze::CompressedMatrix;
   using blaze::DynamicMatrix;
   using blaze::StaticMatrix;
   using blaze::DiagonalMatrix;
   using blaze::rowMajor;

   typedef DiagonalMatrix< CompressedMatrix<double,rowMajor> >  CompressedDiagonal;

   // Default constructed, row-major 3x3 diagonal compressed matrix
   CompressedDiagonal A( 3 );

   // Initializing elements via the function call operator
   A(0,0) = 1.0;  // Initialization of the diagonal element (0,0)
   A(1,0) = 9.0;  // Throws an exception; invalid modification of lower element

   // Inserting more elements via the insert() function
   A.insert( 1, 1, 3.0 );  // Inserting the diagonal element (1,1)
   A.insert( 0, 2, 9.0 );  // Throws an exception; invalid insertion of upper element

   // Appending an element via the append() function
   A.reserve( 2, 1 );      // Reserving enough capacity in row 2
   A.append( 2, 2, 5.0 );  // Appending the diagonal element (2,2)
   A.append( 1, 2, 9.0 );  // Throws an exception; appending an element in the upper part

   // Access via a non-const iterator
   CompressedDiagonal::Iterator it = A.begin(1);
   *it = 6.0;  // Modifies the element (1,1)

   // Erasing elements via the erase() function
   A.erase( 0, 0 );  // Erasing the diagonal element (0,0)
   A.erase( 2, 2 );  // Erasing the diagonal element (2,2)

   // Construction from a diagonal dense matrix
   StaticMatrix<double,3UL,3UL> B(  3.0,  0.0,  0.0,
                                    0.0, -2.0,  0.0,
                                    0.0,  0.0,  4.0 );

   DiagonalMatrix< DynamicMatrix<double,rowMajor> > C( B );  // OK

   // Assignment of a non-diagonal dense matrix
   StaticMatrix<double,3UL,3UL> D(  3.0,  0.0,  9.0,
                                    0.0, -2.0,  0.0,
                                    0.0,  0.0,  4.0 );

   C = D;  // Throws an exception; diagonal matrix invariant would be violated!
   \endcode

// The diagonal matrix property is also enforced for diagonal custom matrices: In case the given
// array of elements does not represent a diagonal matrix, a \a std::invalid_argument exception is
// thrown:

   \code
   using blaze::CustomMatrix;
   using blaze::DiagonalMatrix;
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;

   typedef DiagonalMatrix< CustomMatrix<double,unaligned,unpadded,rowMajor> >  CustomDiagonal;

   // Creating a 3x3 diagonal custom matrix from a properly initialized array
   double array[9] = { 1.0, 0.0, 0.0,
                       0.0, 2.0, 0.0,
                       0.0, 0.0, 3.0 };
   CustomDiagonal A( array, 3UL );  // OK

   // Attempt to create a second 3x3 diagonal custom matrix from an uninitialized array
   CustomDiagonal B( new double[9UL], 3UL, blaze::ArrayDelete() );  // Throws an exception
   \endcode

// Finally, the diagonal matrix property is enforced for views (rows, columns, submatrices, ...)
// on the diagonal matrix. The following example demonstrates that modifying the elements of an
// entire row and submatrix of a diagonal matrix only affects the diagonal matrix elements:

   \code
   using blaze::DynamicMatrix;
   using blaze::DiagonalMatrix;

   // Setup of the diagonal matrix
   //
   //       ( 0 0 0 0 )
   //   A = ( 0 1 0 0 )
   //       ( 0 0 2 0 )
   //       ( 0 0 0 3 )
   //
   DiagonalMatrix< DynamicMatrix<int> > A( 4 );
   A(1,1) = 1;
   A(2,2) = 2;
   A(3,3) = 3;

   // Setting the diagonal element in the 2nd row to 9 results in the matrix
   //
   //       ( 0 0 0 0 )
   //   A = ( 0 1 0 0 )
   //       ( 0 0 9 0 )
   //       ( 0 0 0 3 )
   //
   row( A, 2 ) = 9;

   // Setting the diagonal element in the 1st and 2nd column to 7 results in
   //
   //       ( 0 0 0 0 )
   //   A = ( 0 7 0 0 )
   //       ( 0 0 7 0 )
   //       ( 0 0 0 3 )
   //
   submatrix( A, 0, 1, 4, 2 ) = 7;
   \endcode

// The next example demonstrates the (compound) assignment to rows/columns and submatrices of
// diagonal matrices. Since only diagonal elements may be modified the matrix to be assigned
// must be structured such that the diagonal matrix invariant of the diagonal matrix is
// preserved. Otherwise a \a std::invalid_argument exception is thrown:

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::DiagonalMatrix;
   using blaze::rowVector;

   // Setup of two default 4x4 diagonal matrices
   DiagonalMatrix< DynamicMatrix<int> > A1( 4 ), A2( 4 );

   // Setup of a 4-dimensional vector
   //
   //   v = ( 0 0 3 0 )
   //
   DynamicVector<int,rowVector> v( 4, 0 );
   v[2] = 3;

   // OK: Assigning v to the 2nd row of A1 preserves the diagonal matrix invariant
   //
   //        ( 0 0 0 0 )
   //   A1 = ( 0 0 0 0 )
   //        ( 0 0 3 0 )
   //        ( 0 0 0 0 )
   //
   row( A1, 2 ) = v;  // OK

   // Error: Assigning v to the 1st row of A1 violates the diagonal matrix invariant! The element
   //   marked with X cannot be assigned and triggers an exception.
   //
   //        ( 0 0 0 0 )
   //   A1 = ( 0 0 X 0 )
   //        ( 0 0 3 0 )
   //        ( 0 0 0 0 )
   //
   row( A1, 1 ) = v;  // Assignment throws an exception!

   // Setup of the 3x2 dynamic matrix
   //
   //       ( 0 0 )
   //   B = ( 2 0 )
   //       ( 0 3 )
   //
   DynamicMatrix<int> B( 3UL, 2UL, 0 );
   B(1,0) = 2;
   B(2,1) = 3;

   // OK: Assigning B to a submatrix of A2 such that the diagonal matrix invariant can be preserved
   //
   //        ( 0 0 0 0 )
   //   A2 = ( 0 2 0 0 )
   //        ( 0 0 3 0 )
   //        ( 0 0 0 0 )
   //
   submatrix( A2, 0UL, 1UL, 3UL, 2UL ) = B;  // OK

   // Error: Assigning B to a submatrix of A2 such that the diagonal matrix invariant cannot be
   //   preserved! The elements marked with X cannot be assigned without violating the invariant!
   //
   //        ( 0 0 0 0 )
   //   A2 = ( 0 2 X 0 )
   //        ( 0 0 3 X )
   //        ( 0 0 0 0 )
   //
   submatrix( A2, 0UL, 2UL, 3UL, 2UL ) = B;  // Assignment throws an exception!
   \endcode

// \n \subsection diagonalmatrix_initialization The Lower and Upper Elements of a Dense Diagonal Matrix are Always Default Initialized!
//
// Although this results in a small loss of efficiency during the creation of a dense diagonal
// matrix this initialization is important since otherwise the diagonal matrix property of dense
// diagonal matrices would not be guaranteed:

   \code
   using blaze::DynamicMatrix;
   using blaze::DiagonalMatrix;

   // Uninitialized, 5x5 row-major diagonal matrix
   DynamicMatrix<int,rowMajor> A( 5, 5 );

   // 5x5 row-major diagonal dynamic matrix with default initialized lower and upper matrix
   DiagonalMatrix< DynamicMatrix<int,rowMajor> > B( 5 );
   \endcode

// \n \subsection diagonalmatrix_storage Dense Diagonal Matrices Also Store the Non-diagonal Elements!
//
// It is very important to note that dense diagonal matrices store all elements, including the
// non-diagonal elements, and therefore don't provide any kind of memory reduction! There are
// two main reasons for this: First, storing also the non-diagonal elements guarantees maximum
// performance for many algorithms that perform vectorized operations on the diagonal matrix,
// which is especially true for small dense matrices. Second, conceptually the DiagonalMatrix
// adaptor merely restricts the interface to the matrix type \a MT and does not change the data
// layout or the underlying matrix type. Thus, in order to achieve the perfect combination of
// performance and memory consumption it is recommended to use dense matrices for small diagonal
// matrices and sparse matrices for large diagonal matrices:

   \code
   // Recommendation 1: use dense matrices for small diagonal matrices
   typedef blaze::DiagonalMatrix< blaze::StaticMatrix<float,3UL,3UL> >  SmallDiagonalMatrix;

   // Recommendation 2: use sparse matrices for large diagonal matrices
   typedef blaze::DiagonalMatrix< blaze::CompressedMatrix<float> >  LargeDiagonalMatrix;
   \endcode

// \n \section diagonalmatrix_arithmetic_operations Arithmetic Operations
//
// A DiagonalMatrix matrix can participate in numerical operations in any way any other dense or
// sparse matrix can participate. It can also be combined with any other dense or sparse vector
// or matrix. The following code example gives an impression of the use of DiagonalMatrix within
// arithmetic operations:

   \code
   using blaze::DiagonalMatrix;
   using blaze::DynamicMatrix;
   using blaze::HybridMatrix;
   using blaze::StaticMatrix;
   using blaze::CompressedMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   DynamicMatrix<double,rowMajor> A( 3, 3 );
   CompressedMatrix<double,rowMajor> B( 3, 3 );

   DiagonalMatrix< DynamicMatrix<double,rowMajor> > C( 3 );
   DiagonalMatrix< CompressedMatrix<double,rowMajor> > D( 3 );

   DiagonalMatrix< HybridMatrix<float,3UL,3UL,rowMajor> > E;
   DiagonalMatrix< StaticMatrix<float,3UL,3UL,columnMajor> > F;

   E = A + B;     // Matrix addition and assignment to a row-major diagonal matrix
   F = C - D;     // Matrix subtraction and assignment to a column-major diagonal matrix
   F = A * D;     // Matrix multiplication between a dense and a sparse matrix

   C *= 2.0;      // In-place scaling of matrix C
   E  = 2.0 * B;  // Scaling of matrix B
   F  = C * 2.0;  // Scaling of matrix C

   E += A - B;    // Addition assignment
   F -= C + D;    // Subtraction assignment
   F *= A * D;    // Multiplication assignment
   \endcode

// \n \section diagonalmatrix_block_structured Block-Structured Diagonal Matrices
//
// It is also possible to use block-structured diagonal matrices:

   \code
   using blaze::CompressedMatrix;
   using blaze::StaticMatrix;
   using blaze::DiagonalMatrix;

   // Definition of a 5x5 block-structured diagonal matrix based on CompressedMatrix
   DiagonalMatrix< CompressedMatrix< StaticMatrix<int,3UL,3UL> > > A( 5 );
   \endcode

// Also in this case the diagonal matrix invariant is enforced, i.e. it is not possible to
// manipulate elements in the lower and upper part of the matrix:

   \code
   const StaticMatrix<int,3UL,3UL> B( { { 1, -4,  5 },
                                        { 6,  8, -3 },
                                        { 2, -1,  2 } } )

   A.insert( 2, 2, B );  // Inserting the diagonal elements (2,2)
   A(2,4)(1,1) = -5;     // Invalid manipulation of upper matrix element; Results in an exception
   \endcode

// \n \section diagonalmatrix_performance Performance Considerations
//
// The \b Blaze library tries to exploit the properties of diagonal matrices whenever and wherever
// possible. In fact, diagonal matrices come with several special kernels and additionally profit
// from all optimizations for symmetric and triangular matrices. Thus using a diagonal matrix
// instead of a general matrix can result in a considerable performance improvement. However,
// there are also situations when using a diagonal triangular matrix introduces some overhead. The
// following examples demonstrate several common situations where diagonal matrices can positively
// or negatively impact performance.
//
// \n \subsection diagonalmatrix_matrix_matrix_multiplication Positive Impact: Matrix/Matrix Multiplication
//
// When multiplying two matrices, at least one of which is diagonal, \b Blaze can exploit the fact
// that the lower and upper part of the matrix contains only default elements and restrict the
// algorithm to the diagonal elements. The following example demonstrates this by means of a dense
// matrix/dense matrix multiplication:

   \code
   using blaze::DynamicMatrix;
   using blaze::DiagonalMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   DiagonalMatrix< DynamicMatrix<double,rowMajor> > A;
   DynamicMatrix<double,columnMajor> B, C;

   // ... Resizing and initialization

   C = A * B;
   \endcode

// In comparison to a general matrix multiplication, the performance advantage is significant,
// especially for large matrices. In this particular case, the multiplication performs similarly
// to a matrix addition since the complexity is reduced from \f$ O(N^3) \f$ to \f$ O(N^2) \f$.
// Therefore is it highly recommended to use the DiagonalMatrix adaptor when a matrix is known
// to be diagonal. Note however that the performance advantage is most pronounced for dense
// matrices and much less so for sparse matrices.
//
// \n \subsection diagonalmatrix_matrix_vector_multiplication Positive Impact: Matrix/Vector Multiplication
//
// A similar performance improvement can be gained when using a diagonal matrix in a matrix/vector
// multiplication:

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::DiagonalMatrix;
   using blaze::rowMajor;
   using blaze::columnVector;

   DiagonalMatrix< DynamicMatrix<double,rowMajor> > A;
   DynamicVector<double,columnVector> x, y;

   // ... Resizing and initialization

   y = A * x;
   \endcode

// In this example, \b Blaze also exploits the structure of the matrix and performs similarly to
// a vector addition. Also in case of matrix/vector multiplications the performance improvement
// is most pronounced for dense matrices and much less so for sparse matrices.
//
// \n \subsection diagonalmatrix_assignment Negative Impact: Assignment of a General Matrix
//
// In contrast to using a diagonal matrix on the right-hand side of an assignment (i.e. for read
// access), which introduces absolutely no performance penalty, using a diagonal matrix on the
// left-hand side of an assignment (i.e. for write access) may introduce additional overhead when
// it is assigned a general matrix, which is not diagonal at compile time:

   \code
   using blaze::DynamicMatrix;
   using blaze::DiagonalMatrix;

   DiagonalMatrix< DynamicMatrix<double> > A, C;
   DynamicMatrix<double> B;

   B = A;  // Only read-access to the diagonal matrix; no performance penalty
   C = A;  // Assignment of a diagonal matrix to another diagonal matrix; no runtime overhead
   C = B;  // Assignment of a general matrix to a diagonal matrix; some runtime overhead
   \endcode

// When assigning a general, potentially not diagonal matrix to a diagonal matrix it is necessary
// to check whether the matrix is diagonal at runtime in order to guarantee the diagonal property
// of the diagonal matrix. In case it turns out to be diagonal, it is assigned as efficiently as
// possible, if it is not, an exception is thrown. In order to prevent this runtime overhead it
// is therefore generally advisable to assign diagonal matrices to other diagonal matrices.\n
// In this context it is especially noteworthy that the addition, subtraction, and multiplication
// of two diagonal matrices always results in another diagonal matrix:

   \code
   using blaze::DynamicMatrix;
   using blaze::DiagonalMatrix;

   DiagonalMatrix< DynamicMatrix<double> > A, B, C;

   C = A + B;  // Results in a diagonal matrix; no runtime overhead
   C = A - B;  // Results in a diagonal matrix; no runtime overhead
   C = A * B;  // Results in a diagonal matrix; no runtime overhead
   \endcode
*/
template< typename MT                               // Type of the adapted matrix
        , bool SO = IsColumnMajorMatrix<MT>::value  // Storage order of the adapted matrix
        , bool DF = IsDenseMatrix<MT>::value >      // Density flag
class DiagonalMatrix
{};
//*************************************************************************************************

} // namespace blaze

#endif
