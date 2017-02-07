//=================================================================================================
/*!
//  \file blaze/math/adaptors/unilowermatrix/BaseTemplate.h
//  \brief Header file for the implementation of the base template of the UniLowerMatrix
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

#ifndef _BLAZE_MATH_ADAPTORS_UNILOWERMATRIX_BASETEMPLATE_H_
#define _BLAZE_MATH_ADAPTORS_UNILOWERMATRIX_BASETEMPLATE_H_


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
/*!\defgroup unilower_matrix UniLowerMatrix
// \ingroup adaptors
*/
/*!\brief Matrix adapter for lower unitriangular \f$ N \times N \f$ matrices.
// \ingroup unilower_matrix
//
// \section unilowermatrix_general General
//
// The UniLowerMatrix class template is an adapter for existing dense and sparse matrix types.
// It inherits the properties and the interface of the given matrix type \a MT and extends it
// by enforcing the additional invariant that all diagonal matrix elements are 1 and all matrix
// elements above the diagonal are 0 (lower unitriangular matrix). The type of the adapted matrix
// can be specified via the first template parameter:

   \code
   template< typename MT, bool SO, bool DF >
   class UniLowerMatrix;
   \endcode

//  - MT: specifies the type of the matrix to be adapted. UniLowerMatrix can be used with any
//        non-cv-qualified, non-reference, non-pointer, non-expression dense or sparse matrix
//        type. Also, the given matrix type must have numeric element types (i.e. all integral
//        types except \a bool, floating point and complex types). Note that the given matrix
//        type must be either resizable (as for instance HybridMatrix or DynamicMatrix) or
//        must be square at compile time (as for instance StaticMatrix).
//  - SO: specifies the storage order (blaze::rowMajor, blaze::columnMajor) of the matrix. This
//        template parameter doesn't have to be explicitly defined, but is automatically derived
//        from the first template parameter.
//  - DF: specifies whether the given matrix type is a dense or sparse matrix type. This template
//        parameter doesn't have to be defined explicitly, it is automatically derived from the
//        first template parameter. Defining the parameter explicitly may result in a compilation
//        error!
//
// The following examples give an impression of several possible lower unitriangular matrices:

   \code
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;
   using blaze::columnMajor;

   // Definition of a 3x3 row-major dense unilower matrix with static memory
   blaze::UniLowerMatrix< blaze::StaticMatrix<int,3UL,3UL,rowMajor> > A;

   // Definition of a resizable column-major dense unilower matrix based on HybridMatrix
   blaze::UniLowerMatrix< blaze::HybridMatrix<float,4UL,4UL,columnMajor> B;

   // Definition of a resizable row-major dense unilower matrix based on DynamicMatrix
   blaze::UniLowerMatrix< blaze::DynamicMatrix<double,rowMajor> > C;

   // Definition of a fixed-size row-major dense unilower matrix based on CustomMatrix
   blaze::UniLowerMatrix< blaze::CustomMatrix<double,unaligned,unpadded,rowMajor> > D;

   // Definition of a compressed row-major single precision unilower matrix
   blaze::UniLowerMatrix< blaze::CompressedMatrix<float,rowMajor> > E;
   \endcode

// The storage order of a lower unitriangular matrix is depending on the storage order of the
// adapted matrix type \a MT. In case the adapted matrix is stored in a row-wise fashion (i.e.
// is specified as blaze::rowMajor), the unilower matrix will also be a row-major matrix.
// Otherwise if the adapted matrix is column-major (i.e. is specified as blaze::columnMajor),
// the unilower matrix will also be a column-major matrix.
//
//
// \n \section unilowermatrix_special_properties Special Properties of Lower Unitriangular Matrices
//
// A lower unitriangular matrix is used exactly like a matrix of the underlying, adapted matrix
// type \a MT. It also provides (nearly) the same interface as the underlying matrix type. However,
// there are some important exceptions resulting from the lower unitriangular matrix constraint:
//
//  -# <b>\ref unilowermatrix_square</b>
//  -# <b>\ref unilowermatrix_unilower</b>
//  -# <b>\ref unilowermatrix_initialization</b>
//  -# <b>\ref unilowermatrix_storage</b>
//  -# <b>\ref unilowermatrix_scaling</b>

//
// \n \subsection unilowermatrix_square Lower Unitriangular Matrices Must Always be Square!
//
// In case a resizable matrix is used (as for instance blaze::HybridMatrix, blaze::DynamicMatrix,
// or blaze::CompressedMatrix), this means that the according constructors, the \c resize() and
// the \c extend() functions only expect a single parameter, which specifies both the number of
// rows and columns, instead of two (one for the number of rows and one for the number of columns):

   \code
   using blaze::DynamicMatrix;
   using blaze::UniLowerMatrix;
   using blaze::rowMajor;

   // Default constructed, default initialized, row-major 3x3 unilower dynamic matrix
   UniLowerMatrix< DynamicMatrix<double,rowMajor> > A( 3 );

   // Resizing the matrix to 5x5
   A.resize( 5 );

   // Extending the number of rows and columns by 2, resulting in a 7x7 matrix
   A.extend( 2 );
   \endcode

// In case a matrix with a fixed size is used (as for instance blaze::StaticMatrix), the number
// of rows and number of columns must be specified equally:

   \code
   using blaze::StaticMatrix;
   using blaze::UniLowerMatrix;
   using blaze::columnMajor;

   // Correct setup of a fixed size column-major 3x3 unilower static matrix
   UniLowerMatrix< StaticMatrix<int,3UL,3UL,columnMajor> > A;

   // Compilation error: the provided matrix type is not a square matrix type
   UniLowerMatrix< StaticMatrix<int,3UL,4UL,columnMajor> > B;
   \endcode
//
// \n \subsection unilowermatrix_unilower The Lower Unitriangular Matrix Property is Always Enforced!
//
// The diagonal elements of a lower unitriangular matrix are fixed to 1. This property has two
// implications. First, that means that the diagonal elements of a newly created unilower matrix
// are pre-initialized to 1:

   \code
   using blaze::CompressedMatrix;
   using blaze::DynamicMatrix;
   using blaze::UniLowerMatrix;
   using blaze::rowMajor;

   // Creating a default initialized dense unilower matrix of size 3x3
   //
   //       ( 1 0 0 )
   //   A = ( 0 1 0 )
   //       ( 0 0 1 )
   UniLowerMatrix< DynamicMatrix<int,rowMajor> > A( 3UL );

   // Creating a default initialized sparse unilower matrix of size 3x3
   //
   //       ( 1 0 0 )
   //   B = ( 0 1 0 )
   //       ( 0 0 1 )
   UniLowerMatrix< CompressedMatrix<int,rowMajor> > B( 3UL );
   \endcode

// Second, this means that it is only allowed to modify elements in the lower part of the matrix,
// but not the diagonal elements and not the elements in the upper part of the matrix. Also, it
// is only possible to assign matrices that are lower unitriangular matrices themselves:

   \code
   using blaze::CompressedMatrix;
   using blaze::DynamicMatrix;
   using blaze::StaticMatrix;
   using blaze::UniLowerMatrix;
   using blaze::columnMajor;

   typedef UniLowerMatrix< CompressedMatrix<double,columnMajor> >  CompressedUniLower;

   // Default constructed, row-major 3x3 unilower compressed matrix
   CompressedUniLower A( 3 );

   // Initializing elements via the function call operator
   A(0,0) = 9.0;  // Throws an exception; invalid modification of diagonal element
   A(2,0) = 2.0;  // Initialization of the lower element (2,0)
   A(1,2) = 9.0;  // Throws an exception; invalid modification of upper element

   // Inserting elements via the insert() function
   A.insert( 1, 0, 3.0 );  // Inserting the lower element (1,0)
   A.insert( 1, 1, 9.0 );  // Throws an exception; invalid insertion of diagonal element
   A.insert( 0, 2, 9.0 );  // Throws an exception; invalid insertion of upper element

   // Appending an element via the append() function
   A.reserve( 1, 3 );      // Reserving enough capacity in column 1
   A.append( 1, 1, 9.0 );  // Throws an exception; appending a diagonal element
   A.append( 2, 1, 4.0 );  // Appending the lower element (2,1)

   // Access via a non-const iterator
   CompressedUniLower::Iterator it = A.begin(1);
   *it = 9.0;  // Throws an exception; invalid modification of the diagonal element (1,1)
   ++it;
   *it = 6.0;  // Modifies the lower element (2,1)

   // Erasing elements via the erase() function
   A.erase( 0, 0 );  // Throws an exception; invalid erasure of the diagonal element (0,0)
   A.erase( 2, 0 );  // Erasing the lower element (2,0)

   // Construction from an unilower dense matrix
   StaticMatrix<double,3UL,3UL> B(  1.0,  0.0,  0.0,
                                    8.0,  1.0,  0.0,
                                   -2.0, -1.0,  1.0 );

   UniLowerMatrix< DynamicMatrix<double,rowMajor> > C( B );  // OK

   // Assignment of a non-unilower dense matrix
   StaticMatrix<double,3UL,3UL> D(  3.0,  0.0, -2.0,
                                    8.0,  0.0,  0.0,
                                   -2.0, -1.0,  4.0 );

   C = D;  // Throws an exception; lower unitriangular matrix invariant would be violated!
   \endcode

// The lower unitriangular matrix property is also enforced for unilower custom matrices: In case
// the given array of elements does not represent an unilower matrix, a \a std::invalid_argument
// exception is thrown:

   \code
   using blaze::CustomMatrix;
   using blaze::UniLowerMatrix;
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;

   typedef UniLowerMatrix< CustomMatrix<double,unaligned,unpadded,rowMajor> >  CustomUniLower;

   // Creating a 3x3 unilower custom matrix from a properly initialized array
   double array[9] = { 1.0, 0.0, 0.0,
                       2.0, 1.0, 0.0,
                       3.0, 4.0, 1.0 };
   CustomUniLower A( array, 3UL );  // OK

   // Attempt to create a second 3x3 unilower custom matrix from an uninitialized array
   CustomUniLower B( new double[9UL], 3UL, blaze::ArrayDelete() );  // Throws an exception
   \endcode

// Finally, the lower unitriangular matrix property is enforced for views (rows, columns,
// submatrices, ...) on the unilower matrix. The following example demonstrates that modifying
// the elements of an entire row and submatrix of an unilower matrix only affects the lower
// matrix elements:

   \code
   using blaze::DynamicMatrix;
   using blaze::UniLowerMatrix;

   // Setup of the unilower matrix
   //
   //       ( 1 0 0 0 )
   //   A = ( 2 1 0 0 )
   //       ( 0 3 1 0 )
   //       ( 4 0 5 1 )
   //
   UniLowerMatrix< DynamicMatrix<int> > A( 4 );
   A(1,0) = 2;
   A(2,1) = 3;
   A(3,0) = 4;
   A(3,2) = 5;

   // Setting the lower elements in the 2nd row to 9 results in the matrix
   //
   //       ( 1 0 0 0 )
   //   A = ( 2 1 0 0 )
   //       ( 9 9 1 0 )
   //       ( 4 0 5 1 )
   //
   row( A, 2 ) = 9;

   // Setting the lower elements in the 1st and 2nd column to 7 results in
   //
   //       ( 1 0 0 0 )
   //   A = ( 1 1 0 0 )
   //       ( 9 7 1 0 )
   //       ( 4 7 7 1 )
   //
   submatrix( A, 0, 1, 4, 2 ) = 7;
   \endcode

// The next example demonstrates the (compound) assignment to rows/columns and submatrices of
// unilower matrices. Since only lower elements may be modified the matrix to be assigned must
// be structured such that the lower unitriangular matrix invariant of the unilower matrix is
// preserved. Otherwise a \a std::invalid_argument exception is thrown:

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::UniLowerMatrix;
   using blaze::rowVector;

   // Setup of two default 4x4 unilower matrices
   UniLowerMatrix< DynamicMatrix<int> > A1( 4 ), A2( 4 );

   // Setup of a 4-dimensional vector
   //
   //   v = ( 1 2 3 0 )
   //
   DynamicVector<int,rowVector> v( 4, 0 );
   v[0] = 3;
   v[1] = 2;
   v[2] = 1;

   // OK: Assigning v to the 2nd row of A1 preserves the unilower matrix invariant
   //
   //        ( 0 0 0 0 )
   //   A1 = ( 0 0 0 0 )
   //        ( 3 2 1 0 )
   //        ( 0 0 0 0 )
   //
   row( A1, 2 ) = v;  // OK

   // Error: Assigning v to the 1st row of A1 violates the unilower matrix invariant! The elements
   //   marked with X cannot be assigned and trigger an exception.
   //
   //        ( 0 0 0 0 )
   //   A1 = ( 3 X X 0 )
   //        ( 3 2 1 0 )
   //        ( 0 0 0 0 )
   //
   row( A1, 1 ) = v;  // Assignment throws an exception!

   // Setup of the 3x2 dynamic matrix
   //
   //       ( 1 0 )
   //   B = ( 7 1 )
   //       ( 8 9 )
   //
   DynamicMatrix<int> B( 3UL, 2UL, 0 );
   B(0,0) = 1;
   B(1,0) = 7;
   B(1,1) = 1;
   B(2,0) = 8;
   B(2,1) = 9;

   // OK: Assigning B to a submatrix of A2 such that the unilower matrix invariant can be preserved
   //
   //        ( 1 0 0 0 )
   //   A2 = ( 0 1 0 0 )
   //        ( 0 7 1 0 )
   //        ( 0 8 9 1 )
   //
   submatrix( A2, 1UL, 1UL, 3UL, 2UL ) = B;  // OK

   // Error: Assigning B to a submatrix of A2 such that the lower matrix invariant cannot be
   //   preserved! The elements marked with X cannot be assigned without violating the invariant!
   //
   //        ( 1 0 0 0 )
   //   A2 = ( 0 1 X 0 )
   //        ( 0 7 X X )
   //        ( 0 8 8 X )
   //
   submatrix( A2, 1UL, 2UL, 3UL, 2UL ) = B;  // Assignment throws an exception!
   \endcode

// \n \subsection unilowermatrix_initialization The Upper Elements of a Dense Lower Unitriangular Matrix are Always Default Initialized!
//
// Although this results in a small loss of efficiency during the creation of a dense unilower
// matrix this initialization is important since otherwise the lower unitriangular matrix property
// of dense unilower matrices would not be guaranteed:

   \code
   using blaze::DynamicMatrix;
   using blaze::UniLowerMatrix;

   // Uninitialized, 5x5 row-major dynamic matrix
   DynamicMatrix<int,rowMajor> A( 5, 5 );

   // 5x5 row-major unilower dynamic matrix with default initialized upper matrix
   UniLowerMatrix< DynamicMatrix<int,rowMajor> > B( 5 );
   \endcode

// \n \subsection unilowermatrix_storage Dense Lower Unitriangular Matrices Also Store the Upper Elements!
//
// It is important to note that dense lower unitriangular matrices store all elements, including
// the elements in the upper part of the matrix, and therefore don't provide any kind of memory
// reduction! There are two main reasons for this: First, storing also the upper elements
// guarantees maximum performance for many algorithms that perform vectorized operations on the
// unilower matrix, which is especially true for small dense matrices. Second, conceptually the
// UniLowerMatrix adaptor merely restricts the interface to the matrix type \a MT and does not
// change the data layout or the underlying matrix type.
//
//
// \n \subsection unilowermatrix_scaling Lower Unitriangular Matrices Cannot Be Scaled!
//
// Since the diagonal elements have a fixed value of 1 it is not possible to self-scale an unilower
// matrix:

   \code
   using blaze::DynamicMatrix;
   using blaze::UniLowerMatrix;

   UniLowerMatrix< DynamicMatrix<int> > A( 4 );

   A *= 2;        // Compilation error; Scale operation is not available on an unilower matrix
   A /= 2;        // Compilation error; Scale operation is not available on an unilower matrix
   A.scale( 2 );  // Compilation error; Scale function is not available on an unilower matrix

   A = A * 2;  // Throws an exception; Invalid assignment of non-unilower matrix
   A = A / 2;  // Throws an exception; Invalid assignment of non-unilower matrix
   \endcode

// \n \section unilowermatrix_arithmetic_operations Arithmetic Operations
//
// An UniLowerMatrix matrix can participate in numerical operations in any way any other dense or
// sparse matrix can participate. It can also be combined with any other dense or sparse vector
// or matrix. The following code example gives an impression of the use of UniLowerMatrix within
// arithmetic operations:

   \code
   using blaze::UniLowerMatrix;
   using blaze::DynamicMatrix;
   using blaze::HybridMatrix;
   using blaze::StaticMatrix;
   using blaze::CompressedMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   DynamicMatrix<double,rowMajor> A( 3, 3 );
   CompressedMatrix<double,rowMajor> B( 3, 3 );

   UniLowerMatrix< DynamicMatrix<double,rowMajor> > C( 3 );
   UniLowerMatrix< CompressedMatrix<double,rowMajor> > D( 3 );

   UniLowerMatrix< HybridMatrix<float,3UL,3UL,rowMajor> > E;
   UniLowerMatrix< StaticMatrix<float,3UL,3UL,columnMajor> > F;

   DynamicMatrix<double,rowMajor> G( 3, 3 );     // Initialized as strictly lower matrix
   CompressedMatrix<double,rowMajor> H( 3, 3 );  // Initialized as strictly lower matrix

   E = A + B;   // Matrix addition and assignment to a row-major unilower matrix
   F = A - C;   // Matrix subtraction and assignment to a column-major unilower matrix
   F = A * D;   // Matrix multiplication between a dense and a sparse matrix

   E += G;      // Addition assignment (note that G is a strictly lower matrix)
   F -= H;      // Subtraction assignment (note that H is a strictly lower matrix)
   F *= A * D;  // Multiplication assignment
   \endcode

// \n \section unilowermatrix_performance Performance Considerations
//
// The \b Blaze library tries to exploit the properties of lower (uni)-triangular matrices whenever
// and wherever possible. Thus using a lower (uni-)triangular matrix instead of a general matrix
// can result in a considerable performance improvement. However, there are also situations when
// using a (uni-)lower matrix introduces some overhead. The following examples demonstrate several
// common situations where (uni-)lower matrices can positively or negatively impact performance.
//
// \n \subsection unilowermatrix_matrix_matrix_multiplication Positive Impact: Matrix/Matrix Multiplication
//
// When multiplying two matrices, at least one of which is lower (uni)-triangular, \b Blaze can
// exploit the fact that the upper part of the matrix contains only default elements and restrict
// the algorithm to the lower and diagonal elements. The following example demonstrates this by
// means of a dense matrix/dense matrix multiplication:

   \code
   using blaze::DynamicMatrix;
   using blaze::UniLowerMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   UniLowerMatrix< DynamicMatrix<double,rowMajor> > A;
   UniLowerMatrix< DynamicMatrix<double,columnMajor> > B;
   DynamicMatrix<double,columnMajor> C;

   // ... Resizing and initialization

   C = A * B;
   \endcode

// In comparison to a general matrix multiplication, the performance advantage is significant,
// especially for large matrices. Therefore is it highly recommended to use the UniLowerMatrix
// adaptor when a matrix is known to be lower unitriangular. Note however that the performance
// advantage is most pronounced for dense matrices and much less so for sparse matrices.
//
// \n \subsection unilowermatrix_matrix_vector_multiplication Positive Impact: Matrix/Vector Multiplication
//
// A similar performance improvement can be gained when using a lower (uni-)triangular matrix in
// a matrix/vector multiplication:

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::UniLowerMatrix;
   using blaze::rowMajor;
   using blaze::columnVector;

   UniLowerMatrix< DynamicMatrix<double,rowMajor> > A;
   DynamicVector<double,columnVector> x, y;

   // ... Resizing and initialization

   y = A * x;
   \endcode

// In this example, \b Blaze also exploits the structure of the matrix and approx. halves the
// runtime of the multiplication. Also in case of matrix/vector multiplications the performance
// improvement is most pronounced for dense matrices and much less so for sparse matrices.
//
// \n \subsection unilowermatrix_assignment Negative Impact: Assignment of a General Matrix
//
// In contrast to using a lower (uni-)triangular matrix on the right-hand side of an assignment
// (i.e. for read access), which introduces absolutely no performance penalty, using a (uni-)lower
// matrix on the left-hand side of an assignment (i.e. for write access) may introduce additional
// overhead when it is assigned a general matrix, which is not lower (uni-)triangular at compile
// time:

   \code
   using blaze::DynamicMatrix;
   using blaze::UniLowerMatrix;

   UniLowerMatrix< DynamicMatrix<double> > A, C;
   DynamicMatrix<double> B;

   B = A;  // Only read-access to the unilower matrix; no performance penalty
   C = A;  // Assignment of an unilower matrix to another unilower matrix; no runtime overhead
   C = B;  // Assignment of a general matrix to an unilower matrix; some runtime overhead
   \endcode

// When assigning a general, potentially not unilower matrix to another unilower matrix it is
// necessary to check whether the matrix is unilower at runtime in order to guarantee the lower
// unitriangular property of the unilower matrix. In case it turns out to be lower unitriangular,
// it is assigned as efficiently as possible, if it is not, an exception is thrown. In order to
// prevent this runtime overhead it is therefore generally advisable to assign unilower matrices
// to other unilower matrices.\n
// In this context it is especially noteworthy that the multiplication of two lower unitriangular
// matrices always results in another unilower matrix:

   \code
   UniLowerMatrix< DynamicMatrix<double> > A, B, C;

   C = A * B;  // Results in a lower matrix; no runtime overhead
   \endcode
*/
template< typename MT                               // Type of the adapted matrix
        , bool SO = IsColumnMajorMatrix<MT>::value  // Storage order of the adapted matrix
        , bool DF = IsDenseMatrix<MT>::value >      // Density flag
class UniLowerMatrix
{};
//*************************************************************************************************

} // namespace blaze

#endif
