//=================================================================================================
/*!
//  \file blaze/math/adaptors/hermitianmatrix/BaseTemplate.h
//  \brief Header file for the implementation of the base template of the HeritianMatrix
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

#ifndef _BLAZE_MATH_ADAPTORS_HERMITIANMATRIX_BASETEMPLATE_H_
#define _BLAZE_MATH_ADAPTORS_HERMITIANMATRIX_BASETEMPLATE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/typetraits/IsColumnMajorMatrix.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/util/typetraits/IsNumeric.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup hermitian_matrix HermitianMatrix
// \ingroup adaptors
*/
/*!\brief Matrix adapter for Hermitian \f$ N \times N \f$ matrices.
// \ingroup hermitian_matrix
//
// \section hermitianmatrix_general General
//
// The HermitianMatrix class template is an adapter for existing dense and sparse matrix types.
// It inherits the properties and the interface of the given matrix type \a MT and extends it
// by enforcing the additional invariant of Hermitian symmetry (i.e. the matrix is always equal
// to its conjugate transpose \f$ A = \overline{A^T} \f$). The type of the adapted matrix can
// be specified via the first template parameter:

   \code
   template< typename MT, bool SO, bool DF >
   class HermitianMatrix;
   \endcode

//  - MT: specifies the type of the matrix to be adapted. HermitianMatrix can be used with any
//        non-cv-qualified, non-reference, non-pointer, non-expression dense or sparse matrix
//        type. Also, the given matrix type must have numeric element types (i.e. all integral
//        types except \a bool, floating point and complex types). Note that the given matrix
//        type must be either resizable (as for instance HybridMatrix or DynamicMatrix) or must
//			 be square at compile time (as for instance StaticMatrix).
//  - SO: specifies the storage order (blaze::rowMajor, blaze::columnMajor) of the matrix. This
//        template parameter doesn't have to be explicitly defined, but is automatically derived
//        from the first template parameter.
//  - DF: specifies whether the given matrix type is a dense or sparse matrix type. This template
//        parameter doesn't have to be defined explicitly, it is automatically derived from the
//        first template parameter. Defining the parameter explicitly may result in a compilation
//        error!
//
// The following examples give an impression of several possible Hermitian matrices:

   \code
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;
   using blaze::columnMajor;

   // Definition of a 3x3 row-major dense Hermitian matrix with static memory
   blaze::HermitianMatrix< blaze::StaticMatrix<int,3UL,3UL,rowMajor> > A;

   // Definition of a resizable column-major dense Hermitian matrix based on HybridMatrix
   blaze::HermitianMatrix< blaze::HybridMatrix<float,4UL,4UL,columnMajor> B;

   // Definition of a resizable row-major dense Hermitian matrix based on DynamicMatrix
   blaze::HermitianMatrix< blaze::DynamicMatrix<std::complex<double>,rowMajor> > C;

   // Definition of a fixed-size row-major dense diagonal matrix based on CustomMatrix
   blaze::HermitianMatrix< blaze::CustomMatrix<double,unaligned,unpadded,rowMajor> > D;

   // Definition of a compressed row-major single precision complex Hermitian matrix
   blaze::HermitianMatrix< blaze::CompressedMatrix<std::complex<float>,rowMajor> > E;
   \endcode

// The storage order of a Hermitian matrix is depending on the storage order of the adapted matrix
// type \a MT. In case the adapted matrix is stored in a row-wise fashion (i.e. is specified as
// blaze::rowMajor), the Hermitian matrix will also be a row-major matrix. Otherwise, if the
// adapted matrix is column-major (i.e. is specified as blaze::columnMajor), the Hermitian matrix
// will also be a column-major matrix.
//
//
// \n \section hermitianmatrix_vs_symmetricmatrix Hermitian Matrices vs. Symmetric Matrices
//
// The blaze::HermitianMatrix adaptor and the blaze::SymmetricMatrix adaptor share several traits.
// However, there are a couple of differences, both from a mathematical point of view as well as
// from an implementation point of view.
//
// From a mathematical point of view, a matrix is called symmetric when it is equal to its
// transpose (\f$ A = A^T \f$) and it is called Hermitian when it is equal to its conjugate
// transpose (\f$ A = \overline{A^T} \f$). For matrices of real values, however, these two
// conditions coincide, which means that symmetric matrices of real values are also Hermitian
// and Hermitian matrices of real values are also symmetric.
//
// From an implementation point of view, \b Blaze restricts Hermitian matrices to numeric data
// types (i.e. all integral types except \a bool, floating point and complex types), whereas
// symmetric matrices can also be block structured (i.e. can have vector or matrix elements).
// For built-in element types, the HermitianMatrix adaptor behaves exactly like the according
// SymmetricMatrix implementation. For complex element types, however, the Hermitian property
// is enforced (see also \ref hermitianmatrix_hermitian).

	\code
	using blaze::DynamicMatrix;
	using blaze::DynamicVector;
	using blaze::HermitianMatrix;
	using blaze::SymmetricMatrix;

	// The following two matrices provide an identical experience (including performance)
	HermitianMatrix< DynamicMatrix<double> > A;  // Both Hermitian and symmetric
	SymmetricMatrix< DynamicMatrix<double> > B;  // Both Hermitian and symmetric

	// The following two matrices will behave differently
	HermitianMatrix< DynamicMatrix< complex<double> > > C;  // Only Hermitian
	SymmetricMatrix< DynamicMatrix< complex<double> > > D;  // Only symmetric

	// Block-structured Hermitian matrices are not allowed
	HermitianMatrix< DynamicMatrix< DynamicVector<double> > > E;  // Compilation error!
	SymmetricMatrix< DynamicMatrix< DynamicVector<double> > > F;  // Block-structured symmetric matrix
	\endcode

// \n \section hermitianmatrix_special_properties Special Properties of Hermitian Matrices
//
// A Hermitian matrix is used exactly like a matrix of the underlying, adapted matrix type \a MT.
// It also provides (nearly) the same interface as the underlying matrix type. However, there are
// some important exceptions resulting from the Hermitian symmetry constraint:
//
//  -# <b>\ref hermitianmatrix_square</b>
//  -# <b>\ref hermitianmatrix_hermitian</b>
//  -# <b>\ref hermitianmatrix_initialization</b>
//
// \n \subsection hermitianmatrix_square Hermitian Matrices Must Always be Square!
//
// In case a resizable matrix is used (as for instance blaze::HybridMatrix, blaze::DynamicMatrix,
// or blaze::CompressedMatrix), this means that the according constructors, the \c resize() and
// the \c extend() functions only expect a single parameter, which specifies both the number of
// rows and columns, instead of two (one for the number of rows and one for the number of columns):

   \code
   using blaze::DynamicMatrix;
   using blaze::HermitianMatrix;
   using blaze::rowMajor;

   // Default constructed, default initialized, row-major 3x3 Hermitian dynamic matrix
   HermitianMatrix< DynamicMatrix<std::complex<double>,rowMajor> > A( 3 );

   // Resizing the matrix to 5x5
   A.resize( 5 );

   // Extending the number of rows and columns by 2, resulting in a 7x7 matrix
   A.extend( 2 );
   \endcode

// In case a matrix with a fixed size is used (as for instance blaze::StaticMatrix), the number
// of rows and number of columns must be specified equally:

   \code
   using blaze::StaticMatrix;
   using blaze::HermitianMatrix;
   using blaze::columnMajor;

   // Correct setup of a fixed size column-major 3x3 Hermitian static matrix
   HermitianMatrix< StaticMatrix<std::complex<float>,3UL,3UL,columnMajor> > A;

   // Compilation error: the provided matrix type is not a square matrix type
   HermitianMatrix< StaticMatrix<std::complex<float>,3UL,4UL,columnMajor> > B;
   \endcode

// \n \subsection hermitianmatrix_hermitian The Hermitian Property is Always Enforced!
//
// This means that the following properties of a Hermitian matrix are always guaranteed:
//
//  - The diagonal elements are real numbers, i.e. the imaginary part is zero
//  - Element \f$ a_{ij} \f$ is always the complex conjugate of element \f$ a_{ji} \f$
//
// Thus modifying the element \f$ a_{ij} \f$ of a Hermitian matrix also modifies its
// counterpart element \f$ a_{ji} \f$. Also, it is only possible to assign matrices that
// are Hermitian themselves:

   \code
   using blaze::CompressedMatrix;
   using blaze::DynamicMatrix;
   using blaze::StaticMatrix;
   using blaze::HermitianMatrix;
   using blaze::rowMajor;

	typedef std::complex<double>  cplx;

   // Default constructed, row-major 3x3 Hermitian compressed matrix
   HermitianMatrix< CompressedMatrix<cplx,rowMajor> > A( 3 );

   // Initializing the matrix via the function call operator
	//
	//  ( (1, 0) (0,0) (2,1) )
	//  ( (0, 0) (0,0) (0,0) )
	//  ( (2,-1) (0,0) (0,0) )
   //
   A(0,0) = cplx( 1.0, 0.0 );  // Initialization of the diagonal element (0,0)
   A(0,2) = cplx( 2.0, 1.0 );  // Initialization of the elements (0,2) and (2,0)

   // Inserting three more elements via the insert() function
	//
	//  ( (1,-3) (0,0) (2, 1) )
	//  ( (0, 0) (2,0) (4,-2) )
	//  ( (2,-1) (4,2) (0, 0) )
   //
   A.insert( 1, 1, cplx( 2.0,  0.0 ) );  // Inserting the diagonal element (1,1)
   A.insert( 1, 2, cplx( 4.0, -2.0 ) );  // Inserting the elements (1,2) and (2,1)

   // Access via a non-const iterator
	//
	//  ( (1,-3) (8,1) (2, 1) )
	//  ( (8,-1) (2,0) (4,-2) )
	//  ( (2,-1) (4,2) (0, 0) )
   //
   *A.begin(1UL) = cplx( 8.0, -1.0 );  // Modifies both elements (1,0) and (0,1)

   // Erasing elements via the erase() function
	//
	//  ( (0, 0) (8,1) (0, 0) )
	//  ( (8,-1) (2,0) (4,-2) )
	//  ( (0, 0) (4,2) (0, 0) )
   //
   A.erase( 0, 0 );  // Erasing the diagonal element (0,0)
   A.erase( 0, 2 );  // Erasing the elements (0,2) and (2,0)

   // Construction from a Hermitian dense matrix
   StaticMatrix<cplx,3UL,3UL> B( { { cplx(  3.0,  0.0 ), cplx(  8.0, 2.0 ), cplx( -2.0,  2.0 ) },
                                   { cplx(  8.0,  1.0 ), cplx(  0.0, 0.0 ), cplx( -1.0, -1.0 ) },
                                   { cplx( -2.0, -2.0 ), cplx( -1.0, 1.0 ), cplx(  4.0,  0.0 ) } } );

   HermitianMatrix< DynamicMatrix<double,rowMajor> > C( B );  // OK

   // Assignment of a non-Hermitian dense matrix
	StaticMatrix<cplx,3UL,3UL> D( { { cplx(  3.0, 0.0 ), cplx(  7.0, 2.0 ), cplx( 3.0, 2.0 ) },
                                   { cplx(  8.0, 1.0 ), cplx(  0.0, 0.0 ), cplx( 6.0, 4.0 ) },
                                   { cplx( -2.0, 2.0 ), cplx( -1.0, 1.0 ), cplx( 4.0, 0.0 ) } } );

   C = D;  // Throws an exception; Hermitian invariant would be violated!
   \endcode

// The same restriction also applies to the \c append() function for sparse matrices: Appending
// the element \f$ a_{ij} \f$ additionally inserts the element \f$ a_{ji} \f$ into the matrix.
// Despite the additional insertion, the \c append() function still provides the most efficient
// way to set up a Hermitian sparse matrix. In order to achieve the maximum efficiency, the
// capacity of the individual rows/columns of the matrix should to be specifically prepared with
// \c reserve() calls:

   \code
   using blaze::CompressedMatrix;
   using blaze::HermitianMatrix;
   using blaze::rowMajor;

	typedef std::complex<double>  cplx;

   // Setup of the Hermitian matrix
   //
   //       ( (0, 0) (1,2) (3,-4) )
   //   A = ( (1,-2) (2,0) (0, 0) )
   //       ( (3, 4) (0,0) (0, 0) )
   //
   HermitianMatrix< CompressedMatrix<cplx,rowMajor> > A( 3 );

   A.reserve( 5 );         // Reserving enough space for 5 non-zero elements
   A.reserve( 0, 2 );      // Reserving two non-zero elements in the first row
   A.reserve( 1, 2 );      // Reserving two non-zero elements in the second row
   A.reserve( 2, 1 );      // Reserving a single non-zero element in the third row

   A.append( 0, 1, cplx( 1.0, 2.0 ) );  // Appending an element at position (0,1) and (1,0)
   A.append( 1, 1, cplx( 2.0, 0.0 ) );  // Appending an element at position (1,1)
   A.append( 2, 0, cplx( 3.0, 4.0 ) );  // Appending an element at position (2,0) and (0,2)
   \endcode

// The Hermitian property is also enforced for Hermitian custom matrices: In case the given array
// of elements does not represent a Hermitian matrix, a \a std::invalid_argument exception is
// thrown:

   \code
   using blaze::CustomMatrix;
   using blaze::HermitianMatrix;
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;

   typedef HermitianMatrix< CustomMatrix<double,unaligned,unpadded,rowMajor> >  CustomHermitian;

   // Creating a 3x3 Hermitian custom matrix from a properly initialized array
   double array[9] = { 1.0, 2.0, 4.0,
                       2.0, 3.0, 5.0,
                       4.0, 5.0, 6.0 };
   CustomHermitian A( array, 3UL );  // OK

   // Attempt to create a second 3x3 Hermitian custom matrix from an uninitialized array
   CustomHermitian B( new double[9UL], 3UL, blaze::ArrayDelete() );  // Throws an exception
   \endcode

// Finally, the Hermitian property is enforced for views (rows, columns, submatrices, ...) on the
// Hermitian matrix. The following example demonstrates that modifying the elements of an entire
// row of the Hermitian matrix also affects the counterpart elements in the according column of
// the matrix:

   \code
   using blaze::DynamicMatrix;
   using blaze::HermtianMatrix;

	typedef std::complex<double>  cplx;

   // Setup of the Hermitian matrix
   //
   //       ( (0, 0) (1,-1) (0,0) (2, 1) )
   //   A = ( (1, 1) (3, 0) (4,2) (0, 0) )
   //       ( (0, 0) (4,-2) (0,0) (5,-3) )
   //       ( (2,-1) (0, 0) (5,3) (0, 0) )
   //
   HermitianMatrix< DynamicMatrix<int> > A( 4 );
   A(0,1) = cplx( 1.0, -1.0 );
   A(0,3) = cplx( 2.0,  1.0 );
   A(1,1) = cplx( 3.0,  0.0 );
   A(1,2) = cplx( 4.0,  2.0 );
   A(2,3) = cplx( 5.0,  3.0 );

   // Setting all elements in the 1st row to 0 results in the matrix
   //
   //       ( (0, 0) (0,0) (0,0) (2, 1) )
   //   A = ( (0, 0) (0,0) (0,0) (0, 0) )
   //       ( (0, 0) (0,0) (0,0) (5,-3) )
   //       ( (2,-1) (0,0) (5,3) (0, 0) )
   //
   row( A, 1 ) = cplx( 0.0, 0.0 );
   \endcode

// The next example demonstrates the (compound) assignment to submatrices of Hermitian matrices.
// Since the modification of element \f$ a_{ij} \f$ of a Hermitian matrix also modifies the
// element \f$ a_{ji} \f$, the matrix to be assigned must be structured such that the Hermitian
// symmetry of the matrix is preserved. Otherwise a \a std::invalid_argument exception is thrown:

   \code
   using blaze::DynamicMatrix;
   using blaze::HermitianMatrix;

	std::complex<double>  cplx;

   // Setup of two default 4x4 Hermitian matrices
   HermitianMatrix< DynamicMatrix<cplx> > A1( 4 ), A2( 4 );

   // Setup of the 3x2 dynamic matrix
   //
   //       ( (1,-1) (2, 5) )
   //   B = ( (3, 0) (4,-6) )
   //       ( (5, 0) (6, 0) )
   //
   DynamicMatrix<int> B( 3UL, 2UL );
   B(0,0) = cplx( 1.0, -1.0 );
   B(0,1) = cplx( 2.0,  5.0 );
   B(1,0) = cplx( 3.0,  0.0 );
   B(1,1) = cplx( 4.0, -6.0 );
   B(2,1) = cplx( 5.0,  0.0 );
   B(2,2) = cplx( 6.0,  7.0 );

   // OK: Assigning B to a submatrix of A1 such that the Hermitian property is preserved
   //
   //        ( (0, 0) (0, 0) (1,-1) (2, 5) )
   //   A1 = ( (0, 0) (0, 0) (3, 0) (4,-6) )
   //        ( (1, 1) (3, 0) (5, 0) (6, 0) )
   //        ( (2,-5) (4, 6) (6, 0) (0, 0) )
   //
   submatrix( A1, 0UL, 2UL, 3UL, 2UL ) = B;  // OK

   // Error: Assigning B to a submatrix of A2 such that the Hermitian property isn't preserved!
   //   The elements marked with X cannot be assigned unambiguously!
   //
   //        ( (0, 0) (1,-1) (2,5) (0,0) )
   //   A2 = ( (1, 1) (3, 0) (X,X) (0,0) )
   //        ( (2,-5) (X, X) (6,0) (0,0) )
   //        ( (0, 0) (0, 0) (0,0) (0,0) )
   //
   submatrix( A2, 0UL, 1UL, 3UL, 2UL ) = B;  // Assignment throws an exception!
   \endcode

// \n \subsection hermitianmatrix_initialization The Elements of a Dense Hermitian Matrix are Always Default Initialized!
//
// Although this results in a small loss of efficiency (especially in case all default values are
// overridden afterwards), this property is important since otherwise the Hermitian property of
// dense Hermitian matrices could not be guaranteed:

   \code
   using blaze::DynamicMatrix;
   using blaze::HermitianMatrix;

   // Uninitialized, 5x5 row-major dynamic matrix
   DynamicMatrix<int,rowMajor> A( 5, 5 );

   // Default initialized, 5x5 row-major Hermitian dynamic matrix
   HermitianMatrix< DynamicMatrix<int,rowMajor> > B( 5 );
   \endcode

// \n \section hermitianmatrix_arithmetic_operations Arithmetic Operations
//
// A HermitianMatrix can be used within all numerical operations in any way any other dense or
// sparse matrix can be used. It can also be combined with any other dense or sparse vector or
// matrix. The following code example gives an impression of the use of HermitianMatrix within
// arithmetic operations:

   \code
   using blaze::HermitianMatrix;
   using blaze::DynamicMatrix;
   using blaze::HybridMatrix;
   using blaze::StaticMatrix;
   using blaze::CompressedMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

	typedef complex<float>  cplx;

   DynamicMatrix<cplx,rowMajor> A( 3, 3 );
   CompressedMatrix<cplx,rowMajor> B( 3, 3 );

   HermitianMatrix< DynamicMatrix<cplx,rowMajor> > C( 3 );
   HermitianMatrix< CompressedMatrix<cplx,rowMajor> > D( 3 );

   HermitianMatrix< HybridMatrix<cplx,3UL,3UL,rowMajor> > E;
   HermitianMatrix< StaticMatrix<cplx,3UL,3UL,columnMajor> > F;

   E = A + B;     // Matrix addition and assignment to a row-major Hermitian matrix
   F = C - D;     // Matrix subtraction and assignment to a column-major Hermitian matrix
   F = A * D;     // Matrix multiplication between a dense and a sparse matrix

   C *= 2.0;      // In-place scaling of matrix C
   E  = 2.0 * B;  // Scaling of matrix B
   F  = C * 2.0;  // Scaling of matrix C

   E += A - B;    // Addition assignment
   F -= C + D;    // Subtraction assignment
   F *= A * D;    // Multiplication assignment
   \endcode

// \n \section hermitianmatrix_performance Performance Considerations
//
// When the Hermitian property of a matrix is known beforehands using the HermitianMatrix adaptor
// instead of a general matrix can be a considerable performance advantage. This is particularly
// true in case the Hermitian matrix is also symmetric (i.e. has built-in element types). The
// \b Blaze library tries to exploit the properties of Hermitian (symmetric) matrices whenever
// possible. However, there are also situations when using a Hermitian matrix introduces some
// overhead. The following examples demonstrate several situations where Hermitian matrices can
// positively or negatively impact performance.
//
// \n \subsection hermitianmatrix_matrix_matrix_multiplication Positive Impact: Matrix/Matrix Multiplication
//
// When multiplying two matrices, at least one of which is symmetric, \b Blaze can exploit the fact
// that \f$ A = A^T \f$ and choose the fastest and most suited combination of storage orders for the
// multiplication. The following example demonstrates this by means of a dense matrix/sparse matrix
// multiplication:

   \code
   using blaze::DynamicMatrix;
   using blaze::HermitianMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   HermitianMatrix< DynamicMatrix<double,rowMajor> > A;        // Both Hermitian and symmetric
   HermitianMatrix< CompressedMatrix<double,columnMajor> > B;  // Both Hermitian and symmetric
   DynamicMatrix<double,columnMajor> C;

   // ... Resizing and initialization

   C = A * B;
   \endcode

// Intuitively, the chosen combination of a row-major and a column-major matrix is the most suited
// for maximum performance. However, \b Blaze evaluates the multiplication as

   \code
   C = A * trans( B );
   \endcode

// which significantly increases the performance since in contrast to the original formulation the
// optimized form can be vectorized. Therefore, in the context of matrix multiplications, using a
// symmetric matrix is obviously an advantage.
//
// \n \subsection hermitianmatrix_matrix_vector_multiplication Positive Impact: Matrix/Vector Multiplication
//
// A similar optimization is possible in case of matrix/vector multiplications:

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::CompressedVector;
	using blaze::HermitianMatrix;
   using blaze::rowMajor;
   using blaze::columnVector;

   HermitianMatrix< DynamicMatrix<double,rowMajor> > A;  // Hermitian and symmetric
   CompressedVector<double,columnVector> x;
   DynamicVector<double,columnVector> y;

   // ... Resizing and initialization

   y = A * x;
   \endcode

// In this example it is not intuitively apparent that using a row-major matrix is not the best
// possible choice in terms of performance since the computation cannot be vectorized. Choosing
// a column-major matrix instead, however, would enable a vectorized computation. Therefore
// \b Blaze exploits the fact that \c A is symmetric, selects the best suited storage order and
// evaluates the multiplication as

   \code
   y = trans( A ) * x;
   \endcode

// which also significantly increases the performance.
//
// \n \subsection hermitianmatrix_views Positive Impact: Row/Column Views on Column/Row-Major Matrices
//
// Another example is the optimization of a row view on a column-major symmetric matrix:

   \code
   using blaze::DynamicMatrix;
   using blaze::HermitianMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   typedef HermitianMatrix< DynamicMatrix<double,columnMajor> >  DynamicHermitian;

   DynamicHermitian A( 10UL );  // Both Hermitian and symmetric
   Row<DynamicHermitian> row5 = row( A, 5UL );
   \endcode

// Usually, a row view on a column-major matrix results in a considerable performance decrease in
// comparison to a row view on a row-major matrix due to the non-contiguous storage of the matrix
// elements. However, in case of symmetric matrices, \b Blaze instead uses the according column of
// the matrix, which provides the same performance as if the matrix would be row-major. Note that
// this also works for column views on row-major matrices, where \b Blaze can use the according
// row instead of a column in order to provide maximum performance.
//
// \n \subsection hermitianmatrix_assignment Negative Impact: Assignment of a General Matrix
//
// In contrast to using a Hermitian matrix on the right-hand side of an assignment (i.e. for read
// access), which introduces absolutely no performance penalty, using a Hermitian matrix on the
// left-hand side of an assignment (i.e. for write access) may introduce additional overhead when
// it is assigned a general matrix, which is not Hermitian at compile time:

   \code
   using blaze::DynamicMatrix;
   using blaze::HermitianMatrix;

   HermitianMatrix< DynamicMatrix< complex<double> > > A, C;
   DynamicMatrix<double> B;

   B = A;  // Only read-access to the Hermitian matrix; no performance penalty
   C = A;  // Assignment of a Hermitian matrix to another Hermitian matrix; no runtime overhead
   C = B;  // Assignment of a general matrix to a Hermitian matrix; some runtime overhead
   \endcode

// When assigning a general, potentially not Hermitian matrix to a Hermitian matrix it is necessary
// to check whether the matrix is Hermitian at runtime in order to guarantee the Hermitian property
// of the Hermitian matrix. In case it turns out to be Hermitian, it is assigned as efficiently as
// possible, if it is not, an exception is thrown. In order to prevent this runtime overhead it is
// therefore generally advisable to assign Hermitian matrices to other Hermitian matrices.\n
// In this context it is especially noteworthy that in contrast to additions and subtractions the
// multiplication of two Hermitian matrices does not necessarily result in another Hermitian matrix:

   \code
   HermitianMatrix< DynamicMatrix<double> > A, B, C;

   C = A + B;  // Results in a Hermitian matrix; no runtime overhead
   C = A - B;  // Results in a Hermitian matrix; no runtime overhead
   C = A * B;  // Is not guaranteed to result in a Hermitian matrix; some runtime overhead
   \endcode
*/
template< typename MT                               // Type of the adapted matrix
        , bool SO = IsColumnMajorMatrix<MT>::value  // Storage order of the adapted matrix
        , bool DF = IsDenseMatrix<MT>::value >      // Density flag
class HermitianMatrix
{};
//*************************************************************************************************

} // namespace blaze

#endif
