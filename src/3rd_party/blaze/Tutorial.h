//=================================================================================================
/*!
//  \file blaze/Tutorial.h
//  \brief Tutorial of the Blaze library
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

#ifndef _BLAZE_TUTORIAL_H_
#define _BLAZE_TUTORIAL_H_


//=================================================================================================
//
//  BLAZE TUTORIAL
//
//=================================================================================================

//**Mainpage***************************************************************************************
/*!\mainpage
//
// \image html blaze300x150.jpg
//
// This is the API for the \b Blaze high performance C++ math library. It gives a complete
// overview of the individual features and sublibraries of \b Blaze. To get a first impression
// on \b Blaze, the short \ref getting_started tutorial is a good place to start. Afterwards,
// the following long tutorial covers the most important aspects of the \b Blaze math library.
// The tabs at the top of the page allow a direct access to the individual modules, namespaces,
// classes, and files of the \b Blaze library.\n\n
//
// \section table_of_content Table of Contents
//
// <ul>
//    <li> \ref configuration_and_installation </li>
//    <li> \ref getting_started </li>
//    <li> \ref vectors
//       <ul>
//          <li> \ref vector_types </li>
//          <li> \ref vector_operations </li>
//       </ul>
//    </li>
//    <li> \ref matrices
//       <ul>
//          <li> \ref matrix_types </li>
//          <li> \ref matrix_operations </li>
//       </ul>
//    </li>
//    <li> \ref adaptors
//       <ul>
//          <li> \ref adaptors_symmetric_matrices </li>
//          <li> \ref adaptors_hermitian_matrices </li>
//          <li> \ref adaptors_triangular_matrices </li>
//       </ul>
//    </li>
//    <li> \ref views
//       <ul>
//          <li> \ref views_subvectors </li>
//          <li> \ref views_submatrices </li>
//          <li> \ref views_rows </li>
//          <li> \ref views_columns </li>
//       </ul>
//    </li>
//    <li> \ref arithmetic_operations
//       <ul>
//          <li> \ref addition </li>
//          <li> \ref subtraction </li>
//          <li> \ref scalar_multiplication </li>
//          <li> \ref vector_vector_multiplication
//             <ul>
//                <li> \ref componentwise_multiplication </li>
//                <li> \ref inner_product </li>
//                <li> \ref outer_product </li>
//                <li> \ref cross_product </li>
//             </ul>
//          </li>
//          <li> \ref vector_vector_division </li>
//          <li> \ref matrix_vector_multiplication </li>
//          <li> \ref matrix_matrix_multiplication </li>
//       </ul>
//    </li>
//    <li> \ref custom_operations </li>
//    <li> \ref shared_memory_parallelization
//       <ul>
//          <li> \ref openmp_parallelization </li>
//          <li> \ref cpp_threads_parallelization </li>
//          <li> \ref boost_threads_parallelization </li>
//          <li> \ref serial_execution </li>
//       </ul>
//    </li>
//    <li> \ref serialization
//       <ul>
//          <li> \ref vector_serialization </li>
//          <li> \ref matrix_serialization </li>
//       </ul>
//    </li>
//    <li> \ref blas_functions </li>
//    <li> \ref lapack_functions </li>
//    <li> \ref configuration_files </li>
//    <li> \ref custom_data_types </li>
//    <li> \ref error_reporting_customization </li>
//    <li> \ref intra_statement_optimization </li>
// </ul>
*/
//*************************************************************************************************


//**Configuration and Installation*****************************************************************
/*!\page configuration_and_installation Configuration and Installation
//
// Since \b Blaze is a header-only library, setting up the \b Blaze library on a particular system
// is a fairly easy two step process. In the following, this two step process is explained in
// detail, preceded only by a short summary of the requirements.
//
//
// \n \section requirements Requirements
// <hr>
//
// In order for \b Blaze to work properly, the Boost library must be installed on the system. It
// is recommended to use the newest Boost library available, but \b Blaze requires at minimum the
// Boost version 1.54.0. If you don't have Boost installed on your system, you can download it for
// free from 'http://www.boost.org'.
//
// Additionally, for maximum performance \b Blaze expects you to have a BLAS library installed
// (<a href="http://software.intel.com/en-us/articles/intel-mkl/">Intel MKL</a>,
// <a href="http://developer.amd.com/libraries/acml/">ACML</a>,
// <a href="http://math-atlas.sourceforge.net">Atlas</a>,
// <a href="http://www.tacc.utexas.edu/tacc-projects/gotoblas2">Goto</a>, ...). If you don't
// have a BLAS library installed on your system, \b Blaze will still work and will not be reduced
// in functionality, but performance may be limited. Thus it is strongly recommended to install a
// BLAS library.
//
// Furthermore, for computing the determinant of a dense matrix and for the dense matrix inversion
// \b Blaze requires <a href="https://en.wikipedia.org/wiki/LAPACK">LAPACK</a>. When either of
// these features is used it is necessary to link the LAPACK library to the final executable. If
// no LAPACK library is available the use of these features will result in a linker error.
//
//
// \n \section step_1_installation Step 1: Installation
// <hr>
//
// \subsection step_1_installation_unix Linux/MacOSX User
//
// The first step is the installation of the header files. Since \b Blaze only consists of header
// files, the <tt>./blaze</tt> subdirectory can be simply copied to a standard include directory
// (note that this requires root privileges):

   \code
   cp -r ./blaze /usr/local/include
   \endcode

// Alternatively, on Unix-based machines (which includes Linux and Mac OS X) the
// \c CPLUS_INCLUDE_PATH environment variable can be set. The specified directory will be
// searched after any directories specified on the command line with the option \c -I and
// before the standard default directories (such as \c /usr/local/include and \c /usr/include).
// Assuming a user named 'Jon', the environment variable can be set as follows:

   \code
   CPLUS_INCLUDE_PATH=/usr/home/jon/blaze
   export CPLUS_INCLUDE_PATH
   \endcode

// Last but not least, the <tt>./blaze</tt> subdirectory can be explicitly specified on the
// command line. The following example demonstrates this by means of the GNU C++ compiler:

   \code
   g++ -I/usr/home/jon/blaze -o BlazeTest BlazeTest.cpp
   \endcode

// \n \subsection step_1_installation_windows Windows User
//
// Windows doesn't have a standard include directory. Therefore the \b Blaze header files can be
// copied to any other directory or simply left in the default \b Blaze directory. However, the
// chosen include directory has to be explicitly specified as include path. In Visual Studio,
// this is done via the project property pages, configuration properties, C/C++, General settings.
// Here the additional include directories can be specified.
//
//
// \n \section step_2_configuration Step 2: Configuration
// <hr>
//
// The second step is the configuration and customization of the \b Blaze library. Many aspects of
// \b Blaze can be adapted to specific requirements, environments and architectures by customizing
// the header files in the <tt>./blaze/config/</tt> subdirectory. Since the default settings are
// reasonable for most systems this step can also be skipped. However, in order to achieve maximum
// performance a customization of at least the following configuration files is required:
//
//  - <b><tt>./blaze/config/BLAS.h</tt></b>: Via this configuration file \b Blaze can be enabled
//    to use a third-party BLAS library for several basic linear algebra functions (such as for
//    instance dense matrix multiplications). In case no BLAS library is used, all linear algebra
//    functions use the default implementations of the \b Blaze library and therefore BLAS is not a
//    requirement for the compilation process. However, please note that performance may be limited.
//  - <b><tt>./blaze/config/CacheSize.h</tt></b>: This file contains the hardware specific cache
//    settings. \b Blaze uses this information to optimize its cache usage. For maximum performance
//    it is recommended to adapt these setting to a specific target architecture.
//  - <b><tt>./blaze/config/Thresholds.h</tt></b>: This file contains all thresholds for the
//    customization of the \b Blaze compute kernels. In order to tune the kernels for a specific
//    architecture and to maximize performance it can be necessary to adjust the thresholds,
//    especially for a parallel execution (see \ref shared_memory_parallelization).
//
// For an overview of other customization options and more details, please see the section
// \ref configuration_files.
//
// \n Next: \ref getting_started
*/
//*************************************************************************************************


//**Getting Started********************************************************************************
/*!\page getting_started Getting Started
//
// This short tutorial serves the purpose to give a quick overview of the way mathematical
// expressions have to be formulated in \b Blaze. Starting with \ref vector_types, the following
// long tutorial covers the most important aspects of the \b Blaze math library.
//
//
// \n \section getting_started_vector_example A First Example
//
// \b Blaze is written such that using mathematical expressions is as close to mathematical
// textbooks as possible and therefore as intuitive as possible. In nearly all cases the seemingly
// easiest solution is the right solution and most users experience no problems when trying to
// use \b Blaze in the most natural way. The following example gives a first impression of the
// formulation of a vector addition in \b Blaze:

   \code
   #include <iostream>
   #include <blaze/Math.h>

   using blaze::StaticVector;
   using blaze::DynamicVector;

   // Instantiation of a static 3D column vector. The vector is directly initialized as
   //   ( 4 -2  5 )
   StaticVector<int,3UL> a{ 4, -2, 5 };

   // Instantiation of a dynamic 3D column vector. Via the subscript operator the values are set to
   //   ( 2  5 -3 )
   DynamicVector<int> b( 3UL );
   b[0] = 2;
   b[1] = 5;
   b[2] = -3;

   // Adding the vectors a and b
   DynamicVector<int> c = a + b;

   // Printing the result of the vector addition
   std::cout << "c =\n" << c << "\n";
   \endcode

// Note that the entire \b Blaze math library can be included via the \c blaze/Math.h header
// file. Alternatively, the entire \b Blaze library, including both the math and the entire
// utility module, can be included via the \c blaze/Blaze.h header file. Also note that all
// classes and functions of \b Blaze are contained in the blaze namespace.\n\n
//
// Assuming that this program resides in a source file called \c FirstExample.cpp, it can be
// compiled for instance via the GNU C++ compiler:

   \code
   g++ -ansi -O3 -DNDEBUG -mavx -o FirstExample FirstExample.cpp
   \endcode

// Note the definition of the \c NDEBUG preprocessor symbol. In order to achieve maximum
// performance, it is necessary to compile the program in release mode, which deactivates
// all debugging functionality inside \b Blaze. It is also strongly recommended to specify
// the available architecture specific instruction set (as for instance the AVX instruction
// set, which if available can be activated via the \c -mavx flag). This allows \b Blaze
// to optimize computations via vectorization.\n\n
//
// When running the resulting executable \c FirstExample, the output of the last line of
// this small program is

   \code
   c =
   6
   3
   2
   \endcode

// \n \section getting_started_matrix_example An Example Involving Matrices
//
// Similarly easy and intuitive are expressions involving matrices:

   \code
   #include <blaze/Math.h>

   using namespace blaze;

   // Instantiating a dynamic 3D column vector
   DynamicVector<int> x{ 4, -1, 3 };

   // Instantiating a dynamic 2x3 row-major matrix, preinitialized with 0. Via the function call
   // operator three values of the matrix are explicitly set to get the matrix
   //   ( 1  0  4 )
   //   ( 0 -2  0 )
   DynamicMatrix<int> A( 2UL, 3UL, 0 );
   A(0,0) =  1;
   A(0,2) =  4;
   A(1,1) = -2;

   // Performing a matrix/vector multiplication
   DynamicVector<int> y = A * x;

   // Printing the resulting vector
   std::cout << "y =\n" << y << "\n";

   // Instantiating a static column-major matrix. The matrix is directly initialized as
   //   (  3 -1 )
   //   (  0  2 )
   //   ( -1  0 )
   StaticMatrix<int,3UL,2UL,columnMajor> B{ { 3, -1 }, { 0, 2 }, { -1, 0 } };

   // Performing a matrix/matrix multiplication
   DynamicMatrix<int> C = A * B;

   // Printing the resulting matrix
   std::cout << "C =\n" << C << "\n";
   \endcode

// The output of this program is

   \code
   y =
   16
   2

   C =
   ( -1 -1 )
   (  0  4 )
   \endcode

// \n \section getting_started_complex_example A Complex Example
//
// The following example is much more sophisticated. It shows the implementation of the Conjugate
// Gradient (CG) algorithm (http://en.wikipedia.org/wiki/Conjugate_gradient) by means of the
// \b Blaze library:
//
// \image html cg.jpg
//
// In this example it is not important to understand the CG algorithm itself, but to see the
// advantage of the API of the \b Blaze library. In the \b Blaze implementation we will use a
// sparse matrix/dense vector multiplication for a 2D Poisson equation using \f$ N \times N \f$
// unknowns. It becomes apparent that the core of the algorithm is very close to the mathematical
// formulation and therefore has huge advantages in terms of readability and maintainability,
// while the performance of the code is close to the expected theoretical peak performance:

   \code
   const size_t NN( N*N );

   blaze::CompressedMatrix<double,rowMajor> A( NN, NN );
   blaze::DynamicVector<double,columnVector> x( NN, 1.0 ), b( NN, 0.0 ), r( NN ), p( NN ), Ap( NN );
   double alpha, beta, delta;

   // ... Initializing the sparse matrix A

   // Performing the CG algorithm
   r = b - A * x;
   p = r;
   delta = (r,r);

   for( size_t iteration=0UL; iteration<iterations; ++iteration )
   {
      Ap = A * p;
      alpha = delta / (p,Ap);
      x += alpha * p;
      r -= alpha * Ap;
      beta = (r,r);
      if( std::sqrt( beta ) < 1E-8 ) break;
      p = r + ( beta / delta ) * p;
      delta = beta;
   }
   \endcode

// \n Hopefully this short tutorial gives a good first impression of how mathematical expressions
// are formulated with \b Blaze. The following long tutorial, starting with \ref vector_types,
// will cover all aspects of the \b Blaze math library, i.e. it will introduce all vector and
// matrix types, all possible operations on vectors and matrices, and of course all possible
// mathematical expressions.
//
// \n Previous: \ref configuration_and_installation &nbsp; &nbsp; Next: \ref vectors
*/
//*************************************************************************************************


//**Vectors****************************************************************************************
/*!\page vectors Vectors
//
// \tableofcontents
//
//
// \n \section vectors_general General Concepts
// <hr>
//
// The \b Blaze library currently offers four dense vector types (\ref vector_types_static_vector,
// \ref vector_types_dynamic_vector, \ref vector_types_hybrid_vector, and \ref vector_types_custom_vector)
// and one sparse vector type (\ref vector_types_compressed_vector). All vectors can be specified
// as either column vectors or row vectors:

   \code
   using blaze::DynamicVector;
   using blaze::columnVector;
   using blaze::rowVector;

   // Setup of the 3-dimensional dense column vector
   //
   //    ( 1 )
   //    ( 2 )
   //    ( 3 )
   //
   DynamicVector<int,columnVector> a{ 1, 2, 3 };

   // Setup of the 3-dimensional dense row vector
   //
   //    ( 4  5  6 )
   //
   DynamicVector<int,rowVector> b{ 4, 5, 6 };
   \endcode

// Per default, all vectors in \b Blaze are column vectors:

   \code
   // Instantiation of a 3-dimensional column vector
   blaze::DynamicVector<int> c( 3UL );
   \endcode

// \n \section vectors_details Vector Details
// <hr>
//
//  - \ref vector_types
//  - \ref vector_operations
//
//
// \n \section vectors_examples Examples
// <hr>

   \code
   using blaze::StaticVector;
   using blaze::DynamicVector;
   using blaze::CompressedVector;
   using blaze::rowVector;
   using blaze::columnVector;

   StaticVector<int,6UL> a;            // Instantiation of a 6-dimensional static column vector
   CompressedVector<int,rowVector> b;  // Instantiation of a compressed row vector
   DynamicVector<int,columnVector> c;  // Instantiation of a dynamic column vector

   // ... Resizing and initialization

   c = a + trans( b );
   \endcode

// \n Previous: \ref getting_started &nbsp; &nbsp; Next: \ref vector_types
*/
//*************************************************************************************************


//**Vector Types***********************************************************************************
/*!\page vector_types Vector Types
//
// \tableofcontents
//
//
// \n \section vector_types_static_vector StaticVector
// <hr>
//
// The blaze::StaticVector class template is the representation of a fixed size vector with
// statically allocated elements of arbitrary type. It can be included via the header file

   \code
   #include <blaze/math/StaticVector.h>
   \endcode

// The type of the elements, the number of elements, and the transpose flag of the vector can
// be specified via the three template parameters:

   \code
   template< typename Type, size_t N, bool TF >
   class StaticVector;
   \endcode

//  - \c Type: specifies the type of the vector elements. StaticVector can be used with any
//             non-cv-qualified, non-reference, non-pointer element type.
//  - \c N   : specifies the total number of vector elements. It is expected that StaticVector is
//             only used for tiny and small vectors.
//  - \c TF  : specifies whether the vector is a row vector (\c blaze::rowVector) or a column
//             vector (\c blaze::columnVector). The default value is \c blaze::columnVector.
//
// The blaze::StaticVector is perfectly suited for small to medium vectors whose size is known at
// compile time:

   \code
   // Definition of a 3-dimensional integral column vector
   blaze::StaticVector<int,3UL> a;

   // Definition of a 4-dimensional single precision column vector
   blaze::StaticVector<float,4UL,blaze::columnVector> b;

   // Definition of a 6-dimensional double precision row vector
   blaze::StaticVector<double,6UL,blaze::rowVector> c;
   \endcode

// \n \section vector_types_dynamic_vector DynamicVector
// <hr>
//
// The blaze::DynamicVector class template is the representation of an arbitrary sized vector
// with dynamically allocated elements of arbitrary type. It can be included via the header file

   \code
   #include <blaze/math/DynamicVector.h>
   \endcode

// The type of the elements and the transpose flag of the vector can be specified via the two
// template parameters:

   \code
   template< typename Type, bool TF >
   class DynamicVector;
   \endcode

//  - \c Type: specifies the type of the vector elements. DynamicVector can be used with any
//             non-cv-qualified, non-reference, non-pointer element type.
//  - \c TF  : specifies whether the vector is a row vector (\c blaze::rowVector) or a column
//             vector (\c blaze::columnVector). The default value is \c blaze::columnVector.
//
// The blaze::DynamicVector is the default choice for all kinds of dense vectors and the best
// choice for medium to large vectors. Its size can be modified at runtime:

   \code
   // Definition of a 3-dimensional integral column vector
   blaze::DynamicVector<int> a( 3UL );

   // Definition of a 4-dimensional single precision column vector
   blaze::DynamicVector<float,blaze::columnVector> b( 4UL );

   // Definition of a double precision row vector with size 0
   blaze::DynamicVector<double,blaze::rowVector> c;
   \endcode

// \n \section vector_types_hybrid_vector HybridVector
// <hr>
//
// The blaze::HybridVector class template combines the advantages of the blaze::StaticVector and
// the blaze::DynamicVector class templates. It represents a fixed size vector with statically
// allocated elements, but still can be dynamically resized (within the bounds of the available
// memory). It can be included via the header file

   \code
   #include <blaze/math/HybridVector.h>
   \endcode

// The type of the elements, the number of elements, and the transpose flag of the vector can
// be specified via the three template parameters:

   \code
   template< typename Type, size_t N, bool TF >
   class HybridVector;
   \endcode

//  - \c Type: specifies the type of the vector elements. HybridVector can be used with any
//             non-cv-qualified, non-reference, non-pointer element type.
//  - \c N   : specifies the maximum number of vector elements. It is expected that HybridVector
//             is only used for tiny and small vectors.
//  - \c TF  : specifies whether the vector is a row vector (\c blaze::rowVector) or a column
//             vector (\c blaze::columnVector). The default value is \c blaze::columnVector.
//
// The blaze::HybridVector is a suitable choice for small to medium vectors, whose size is not
// known at compile time or not fixed at runtime, but whose maximum size is known at compile
// time:

   \code
   // Definition of a 3-dimensional integral column vector with a maximum size of 6
   blaze::HybridVector<int,6UL> a( 3UL );

   // Definition of a 4-dimensional single precision column vector with a maximum size of 16
   blaze::HybridVector<float,16UL,blaze::columnVector> b( 4UL );

   // Definition of a double precision row vector with size 0 and a maximum size of 6
   blaze::HybridVector<double,6UL,blaze::rowVector> c;
   \endcode

// \n \section vector_types_custom_vector CustomVector
// <hr>
//
// The blaze::CustomVector class template provides the functionality to represent an external
// array of elements of arbitrary type and a fixed size as a native \b Blaze dense vector data
// structure. Thus in contrast to all other dense vector types a custom vector does not perform
// any kind of memory allocation by itself, but it is provided with an existing array of element
// during construction. A custom vector can therefore be considered an alias to the existing
// array. It can be included via the header file

   \code
   #include <blaze/math/CustomVector.h>
   \endcode

// The type of the elements, the properties of the given array of elements and the transpose
// flag of the vector can be specified via the following four template parameters:

   \code
   template< typename Type, bool AF, bool PF, bool TF >
   class CustomVector;
   \endcode

//  - Type: specifies the type of the vector elements. blaze::CustomVector can be used with
//          any non-cv-qualified, non-reference, non-pointer element type.
//  - AF  : specifies whether the represented, external arrays are properly aligned with
//          respect to the available instruction set (SSE, AVX, ...) or not.
//  - PF  : specified whether the represented, external arrays are properly padded with
//          respect to the available instruction set (SSE, AVX, ...) or not.
//  - TF  : specifies whether the vector is a row vector (\c blaze::rowVector) or a column
//          vector (\c blaze::columnVector). The default value is \c blaze::columnVector.
//
// The blaze::CustomVector is the right choice if any external array needs to be represented as
// a \b Blaze dense vector data structure or if a custom memory allocation strategy needs to be
// realized:

   \code
   using blaze::CustomVector;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;

   // Definition of an unmanaged custom column vector for unaligned, unpadded integer arrays
   typedef CustomVector<int,unaligned,unpadded,columnVector>  UnalignedUnpadded;
   std::vector<int> vec( 7UL );
   UnalignedUnpadded a( &vec[0], 7UL );

   // Definition of a managed custom column vector for unaligned but padded 'float' arrays
   typedef CustomVector<float,unaligned,padded,columnVector>  UnalignedPadded;
   UnalignedPadded b( new float[16], 9UL, 16UL, blaze::ArrayDelete() );

   // Definition of a managed custom row vector for aligned, unpadded 'double' arrays
   typedef CustomVector<double,aligned,unpadded,rowVector>  AlignedUnpadded;
   AlignedUnpadded c( blaze::allocate<double>( 7UL ), 7UL, blaze::Deallocate() );

   // Definition of a managed custom row vector for aligned, padded 'complex<double>' arrays
   typedef CustomVector<complex<double>,aligned,padded,columnVector>  AlignedPadded;
   AlignedPadded d( allocate< complex<double> >( 8UL ), 5UL, 8UL, blaze::Deallocate() );
   \endcode

// In comparison with the remaining \b Blaze dense vector types blaze::CustomVector has several
// special characteristics. All of these result from the fact that a custom vector is not
// performing any kind of memory allocation, but instead is given an existing array of elements.
// The following sections discuss all of these characteristics:
//
//  -# <b>\ref vector_types_custom_vector_memory_management</b>
//  -# <b>\ref vector_types_custom_vector_copy_operations</b>
//  -# <b>\ref vector_types_custom_vector_alignment</b>
//  -# <b>\ref vector_types_custom_vector_padding</b>
//
// \n \subsection vector_types_custom_vector_memory_management Memory Management
//
// The blaze::CustomVector class template acts as an adaptor for an existing array of elements. As
// such it provides everything that is required to use the array just like a native \b Blaze dense
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

// \n \subsection vector_types_custom_vector_copy_operations Copy Operations
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

// \n \subsection vector_types_custom_vector_alignment Alignment
//
// In case the custom vector is specified as \c aligned the passed array must be guaranteed to
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

// In case the alignment requirements are violated, a \c std::invalid_argument exception is
// thrown.
//
// \n \subsection vector_types_custom_vector_padding Padding
//
// Adding padding elements to the end of an array can have a significant impact on the performance.
// For instance, assuming that AVX is available, then two aligned, padded, 3-dimensional vectors
// of double precision values can be added via a single SIMD addition operation:

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
// guarantee that the capacity is a multiple of the SIMD vector width. In case of unaligned
// padded vectors \f$ N-1 \f$ additional padding elements are required, where \f$ N \f$ is
// the SIMD vector width. In case the padding is insufficient with respect to the available
// instruction set, a \c std::invalid_argument exception is thrown.
//
// Please also note that \b Blaze will zero initialize the padding elements in order to achieve
// maximum performance!
//
//
// \n \section vector_types_compressed_vector CompressedVector
// <hr>
//
// The blaze::CompressedVector class is the representation of an arbitrarily sized sparse
// vector, which stores only non-zero elements of arbitrary type. It can be included via the
// header file

   \code
   #include <blaze/math/CompressedVector.h>
   \endcode

// The type of the elements and the transpose flag of the vector can be specified via the two
// template parameters:

   \code
   template< typename Type, bool TF >
   class CompressedVector;
   \endcode

//  - \c Type: specifies the type of the vector elements. CompressedVector can be used with any
//             non-cv-qualified, non-reference, non-pointer element type.
//  - \c TF  : specifies whether the vector is a row vector (\c blaze::rowVector) or a column
//             vector (\c blaze::columnVector). The default value is \c blaze::columnVector.
//
// The blaze::CompressedVector is the right choice for all kinds of sparse vectors:

   \code
   // Definition of a 3-dimensional integral column vector
   blaze::CompressedVector<int> a( 3UL );

   // Definition of a 4-dimensional single precision column vector with capacity for 3 non-zero elements
   blaze::CompressedVector<float,blaze::columnVector> b( 4UL, 3UL );

   // Definition of a double precision row vector with size 0
   blaze::CompressedVector<double,blaze::rowVector> c;
   \endcode

// \n Previous: \ref vectors &nbsp; &nbsp; Next: \ref vector_operations
*/
//*************************************************************************************************


//**Vector Operations******************************************************************************
/*!\page vector_operations Vector Operations
//
// \tableofcontents
//
//
// \n \section vector_operations_constructors Constructors
// <hr>
//
// Instantiating and setting up a vector is very easy and intuitive. However, there are a few
// rules to take care of:
//  - In case the last template parameter (the transpose flag) is omitted, the vector is per
//    default a column vector.
//  - The elements of a \c StaticVector or \c HybridVector are default initialized (i.e. built-in
//    data types are initialized to 0, class types are initialized via the default constructor).
//  - Newly allocated elements of a \c DynamicVector or \c CompressedVector remain uninitialized
//    if they are of built-in type and are default constructed if they are of class type.
//
// \n \subsection vector_operations_default_construction Default Construction

   \code
   using blaze::StaticVector;
   using blaze::DynamicVector;
   using blaze::CompressedVector;

   // All vectors can be default constructed. Whereas the size
   // of StaticVectors is fixed via the second template parameter,
   // the initial size of a default constructed DynamicVector or
   // CompressedVector is 0.
   StaticVector<int,2UL> v1;                // Instantiation of a 2D integer column vector.
                                            // All elements are initialized to 0.
   StaticVector<long,3UL,columnVector> v2;  // Instantiation of a 3D long integer column vector.
                                            // Again, all elements are initialized to 0L.
   DynamicVector<float> v3;                 // Instantiation of a dynamic single precision column
                                            // vector of size 0.
   DynamicVector<double,rowVector> v4;      // Instantiation of a dynamic double precision row
                                            // vector of size 0.
   CompressedVector<int> v5;                // Instantiation of a compressed integer column
                                            // vector of size 0.
   CompressedVector<double,rowVector> v6;   // Instantiation of a compressed double precision row
                                            // vector of size 0.
   \endcode

// \n \subsection vector_operations_size_construction Construction with Specific Size
//
// The \c DynamicVector, \c HybridVector and \c CompressedVector classes offer a constructor that
// allows to immediately give the vector the required size. Whereas both dense vectors (i.e.
// \c DynamicVector and \c HybridVector) use this information to allocate memory for all vector
// elements, \c CompressedVector merely acquires the size but remains empty.

   \code
   DynamicVector<int,columnVector> v7( 9UL );      // Instantiation of an integer dynamic column vector
                                                   // of size 9. The elements are NOT initialized!
   HybridVector< complex<float>, 5UL > v8( 2UL );  // Instantiation of a column vector with two single
                                                   // precision complex values. The elements are
                                                   // default constructed.
   CompressedVector<int,rowVector> v9( 10UL );     // Instantiation of a compressed row vector with
                                                   // size 10. Initially, the vector provides no
                                                   // capacity for non-zero elements.
   \endcode

// \n \subsection vector_operations_initialization_constructors Initialization Constructors
//
// All dense vector classes offer a constructor that allows for a direct, homogeneous initialization
// of all vector elements. In contrast, for sparse vectors the predicted number of non-zero elements
// can be specified

   \code
   StaticVector<int,3UL,rowVector> v10( 2 );            // Instantiation of a 3D integer row vector.
                                                        // All elements are initialized to 2.
   DynamicVector<float> v11( 3UL, 7.0F );               // Instantiation of a dynamic single precision
                                                        // column vector of size 3. All elements are
                                                        // set to 7.0F.
   CompressedVector<float,rowVector> v12( 15UL, 3UL );  // Instantiation of a single precision column
                                                        // vector of size 15, which provides enough
                                                        // space for at least 3 non-zero elements.
   \endcode

// \n \subsection vector_operations_array_construction Array Construction
//
// Alternatively, all dense vector classes offer a constructor for an initialization with a dynamic
// or static array. If the vector is initialized from a dynamic array, the constructor expects the
// actual size of the array as first argument, the array as second argument. In case of a static
// array, the fixed size of the array is used:

   \code
   const unique_ptr<double[]> array1( new double[2] );
   // ... Initialization of the dynamic array
   blaze::StaticVector<double,2UL> v13( 2UL, array1.get() );

   int array2[4] = { 4, -5, -6, 7 };
   blaze::StaticVector<int,4UL> v14( array2 );
   \endcode

// \n \subsection vector_operations_initializer_list_construction Initializer List Construction
//
// In addition, all dense vector classes can be directly initialized by means of an initializer
// list:

   \code
   blaze::DynamicVector<float> v15{ 1.0F, 2.0F, 3.0F, 4.0F };
   \endcode

// \n \subsection vector_operations_copy_construction Copy Construction
//
// All dense and sparse vectors can be created as the copy of any other dense or sparse vector
// with the same transpose flag (i.e. blaze::rowVector or blaze::columnVector).

   \code
   StaticVector<int,9UL,columnVector> v16( v7 );  // Instantiation of the dense column vector v16
                                                  // as copy of the dense column vector v7.
   DynamicVector<int,rowVector> v17( v9 );        // Instantiation of the dense row vector v17 as
                                                  // copy of the sparse row vector v9.
   CompressedVector<int,columnVector> v18( v1 );  // Instantiation of the sparse column vector v18
                                                  // as copy of the dense column vector v1.
   CompressedVector<float,rowVector> v19( v12 );  // Instantiation of the sparse row vector v19 as
                                                  // copy of the row vector v12.
   \endcode

// Note that it is not possible to create a \c StaticVector as a copy of a vector with a different
// size:

   \code
   StaticVector<int,5UL,columnVector> v23( v7 );  // Runtime error: Size does not match!
   StaticVector<int,4UL,rowVector> v24( v10 );    // Compile time error: Size does not match!
   \endcode

// \n \section vector_operations_assignment Assignment
// <hr>
//
// There are several types of assignment to dense and sparse vectors:
// \ref vector_operations_homogeneous_assignment, \ref vector_operations_array_assignment,
// \ref vector_operations_copy_assignment, and \ref vector_operations_compound_assignment.
//
// \n \subsection vector_operations_homogeneous_assignment Homogeneous Assignment
//
// Sometimes it may be necessary to assign the same value to all elements of a dense vector.
// For this purpose, the assignment operator can be used:

   \code
   blaze::StaticVector<int,3UL> v1;
   blaze::DynamicVector<double> v2;

   // Setting all integer elements of the StaticVector to 2
   v1 = 2;

   // Setting all double precision elements of the DynamicVector to 5.0
   v2 = 5.0;
   \endcode

// \n \subsection vector_operations_array_assignment Array Assignment
//
// Dense vectors can also be assigned a static array:

   \code
   blaze::StaticVector<float,2UL> v1;
   blaze::DynamicVector<double,rowVector> v2;

   float  array1[2] = { 1.0F, 2.0F };
   double array2[5] = { 2.1, 4.0, -1.7, 8.6, -7.2 };

   v1 = array1;
   v2 = array2;
   \endcode

// \n \subsection vector_operations_initializer_list_assignment Initializer List Assignment
//
// Alternatively, it is possible to directly assign an initializer list to a dense vector:

   \code
   blaze::StaticVector<float,2UL> v1;
   blaze::DynamicVector<double,rowVector> v2;

   v1 = { 1.0F, 2.0F };
   v2 = { 2.1, 4.0, -1.7, 8.6, -7.2 };
   \endcode

// \n \subsection vector_operations_copy_assignment Copy Assignment
//
// For all vector types it is generally possible to assign another vector with the same transpose
// flag (i.e. blaze::columnVector or blaze::rowVector). Note that in case of \c StaticVectors, the
// assigned vector is required to have the same size as the \c StaticVector since the size of a
// \c StaticVector cannot be adapted!

   \code
   blaze::StaticVector<int,3UL,columnVector> v1;
   blaze::DynamicVector<int,columnVector>    v2( 3UL );
   blaze::DynamicVector<float,columnVector>  v3( 5UL );
   blaze::CompressedVector<int,columnVector> v4( 3UL );
   blaze::CompressedVector<float,rowVector>  v5( 3UL );

   // ... Initialization of the vectors

   v1 = v2;  // OK: Assignment of a 3D dense column vector to another 3D dense column vector
   v1 = v4;  // OK: Assignment of a 3D sparse column vector to a 3D dense column vector
   v1 = v3;  // Runtime error: Cannot assign a 5D vector to a 3D static vector
   v1 = v5;  // Compilation error: Cannot assign a row vector to a column vector
   \endcode

// \n \subsection vector_operations_compound_assignment Compound Assignment
//
// Next to plain assignment, it is also possible to use addition assignment, subtraction
// assignment, and multiplication assignment. Note however, that in contrast to plain assignment
// the size and the transpose flag of the vectors has be to equal in order to able to perform a
// compound assignment.

   \code
   blaze::StaticVector<int,5UL,columnVector>   v1;
   blaze::DynamicVector<int,columnVector>      v2( 5UL );
   blaze::CompressedVector<float,columnVector> v3( 7UL );
   blaze::DynamicVector<float,rowVector>       v4( 7UL );
   blaze::CompressedVector<float,rowVector>    v5( 7UL );

   // ... Initialization of the vectors

   v1 += v2;  // OK: Addition assignment between two column vectors of the same size
   v1 += v3;  // Runtime error: No compound assignment between vectors of different size
   v1 -= v4;  // Compilation error: No compound assignment between vectors of different transpose flag
   v4 *= v5;  // OK: Multiplication assignment between two row vectors of the same size
   \endcode

// \n \section vector_operations_element_access Element Access
// <hr>
//
// The easiest and most intuitive way to access a dense or sparse vector is via the subscript
// operator. The indices to access a vector are zero-based:

   \code
   blaze::DynamicVector<int> v1( 5UL );
   v1[0] = 1;
   v1[1] = 3;
   // ...

   blaze::CompressedVector<float> v2( 5UL );
   v2[2] = 7.3F;
   v2[4] = -1.4F;
   \endcode

// Whereas using the subscript operator on a dense vector only accesses the already existing
// element, accessing an element of a sparse vector via the subscript operator potentially
// inserts the element into the vector and may therefore be more expensive. Consider the
// following example:

   \code
   blaze::CompressedVector<int> v1( 10UL );

   for( size_t i=0UL; i<v1.size(); ++i ) {
      ... = v1[i];
   }
   \endcode

// Although the compressed vector is only used for read access within the for loop, using the
// subscript operator temporarily inserts 10 non-zero elements into the vector. Therefore, all
// vectors (sparse as well as dense) offer an alternate way via the \c begin(), \c cbegin(),
// \c end(), and \c cend() functions to traverse the currently contained elements by iterators.
// In case of non-const vectors, \c begin() and \c end() return an \c Iterator, which allows a
// manipulation of the non-zero value, in case of a constant vector or in case \c cbegin() or
// \c cend() are used a \c ConstIterator is returned:

   \code
   using blaze::CompressedVector;

   CompressedVector<int> v1( 10UL );

   // ... Initialization of the vector

   // Traversing the vector by Iterator
   for( CompressedVector<int>::Iterator it=v1.begin(); it!=v1.end(); ++it ) {
      it->value() = ...;  // OK: Write access to the value of the non-zero element.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the non-zero element.
   }

   // Traversing the vector by ConstIterator
   for( CompressedVector<int>::ConstIterator it=v1.cbegin(); it!=v1.cend(); ++it ) {
      it->value() = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the non-zero element.
   }
   \endcode

// Note that \c begin(), \c cbegin(), \c end(), and \c cend() are also available as free functions:

   \code
   for( CompressedVector<int>::Iterator it=begin( v1 ); it!=end( v1 ); ++it ) {
      // ...
   }

   for( CompressedVector<int>::ConstIterator it=cbegin( v1 ); it!=cend( v1 ); ++it ) {
      // ...
   }
   \endcode

// \n \section vector_operations_element_insertion Element Insertion
// <hr>
//
// In contrast to dense vectors, that store all elements independent of their value and that
// offer direct access to all elements, spares vectors only store the non-zero elements contained
// in the vector. Therefore it is necessary to explicitly add elements to the vector. The first
// option to add elements to a sparse vector is the subscript operator:

   \code
   using blaze::CompressedVector;

   CompressedVector<int> v1( 3UL );
   v1[1] = 2;
   \endcode

// In case the element at the given index is not yet contained in the vector, it is automatically
// inserted. Otherwise the old value is replaced by the new value 2. The operator returns a
// reference to the sparse vector element.\n
// An alternative is the \c set() function: In case the element is not yet contained in the vector
// the element is inserted, else the element's value is modified:

   \code
   // Insert or modify the value at index 3
   v1.set( 3, 1 );
   \endcode

// However, insertion of elements can be better controlled via the \c insert() function. In contrast
// to the subscript operator and the \c set() function it emits an exception in case the element is
// already contained in the vector. In order to check for this case, the \c find() function can be
// used:

   \code
   // In case the element at index 4 is not yet contained in the matrix it is inserted
   // with a value of 6.
   if( v1.find( 4 ) == v1.end() )
      v1.insert( 4, 6 );
   \endcode

// Although the \c insert() function is very flexible, due to performance reasons it is not suited
// for the setup of large sparse vectors. A very efficient, yet also very low-level way to fill
// a sparse vector is the \c append() function. It requires the sparse vector to provide enough
// capacity to insert a new element. Additionally, the index of the new element must be larger
// than the index of the previous element. Violating these conditions results in undefined
// behavior!

   \code
   v1.reserve( 10 );     // Reserving space for 10 non-zero elements
   v1.append( 5, -2 );  // Appending the element -2 at index 5
   v1.append( 6,  4 );  // Appending the element 4 at index 6
   // ...
   \endcode

// \n \section vector_operations_member_functions Member Functions
// <hr>
//
// \subsection vector_operations_size .size()
//
// Via the \c size() member function, the current size of a dense or sparse vector can be queried:

   \code
   // Instantiating a dynamic vector with size 10
   blaze::DynamicVector<int> v1( 10UL );
   v1.size();  // Returns 10

   // Instantiating a compressed vector with size 12 and capacity for 3 non-zero elements
   blaze::CompressedVector<double> v2( 12UL, 3UL );
   v2.size();  // Returns 12
   \endcode

// Alternatively, the free function \c size() can be used to query to current size of a vector.
// In contrast to the member function, the free function can also be used to query the size of
// vector expressions:

   \code
   size( v1 );  // Returns 10, i.e. has the same effect as the member function
   size( v2 );  // Returns 12, i.e. has the same effect as the member function

   blaze::DynamicMatrix<int> A( 15UL, 12UL );
   size( A * v2 );  // Returns 15, i.e. the size of the resulting vector
   \endcode

// \n \subsection vector_operations_capacity .capacity()
//
// Via the \c capacity() (member) function the internal capacity of a dense or sparse vector
// can be queried. Note that the capacity of a vector doesn't have to be equal to the size
// of a vector. In case of a dense vector the capacity will always be greater or equal than
// the size of the vector, in case of a sparse vector the capacity may even be less than
// the size.

   \code
   v1.capacity();   // Returns at least 10
   \endcode

// For symmetry reasons, there is also a free function /c capacity() available that can be used
// to query the capacity:

   \code
   capacity( v1 );  // Returns at least 10, i.e. has the same effect as the member function
   \endcode

// Note, however, that it is not possible to query the capacity of a vector expression:

   \code
   capacity( A * v1 );  // Compilation error!
   \endcode

// \n \subsection vector_operations_nonzeros .nonZeros()
//
// For both dense and sparse vectors the number of non-zero elements can be determined via the
// \c nonZeros() member function. Sparse vectors directly return their number of non-zero
// elements, dense vectors traverse their elements and count the number of non-zero elements.

   \code
   v1.nonZeros();  // Returns the number of non-zero elements in the dense vector
   v2.nonZeros();  // Returns the number of non-zero elements in the sparse vector
   \endcode

// There is also a free function \c nonZeros() available to query the current number of non-zero
// elements:

   \code
   nonZeros( v1 );  // Returns the number of non-zero elements in the dense vector
   nonZeros( v2 );  // Returns the number of non-zero elements in the sparse vector
   \endcode

// The free \c nonZeros() function can also be used to query the number of non-zero elements in
// a vector expression. However, the result is not the exact number of non-zero elements, but
// may be a rough estimation:

   \code
   nonZeros( A * v1 );  // Estimates the number of non-zero elements in the vector expression
   \endcode

// \n \subsection vector_operations_resize_reserve .resize() / .reserve()
//
// The size of a \c StaticVector is fixed by the second template parameter and a \c CustomVector
// cannot be resized. In contrast, the size of \c DynamicVectors, \c HybridVectors as well as
// \c CompressedVectors can be changed via the \c resize() function:

   \code
   using blaze::DynamicVector;
   using blaze::CompressedVector;

   DynamicVector<int,columnVector> v1;
   CompressedVector<int,rowVector> v2( 4 );
   v2[1] = -2;
   v2[3] = 11;

   // Adapting the size of the dynamic and compressed vectors. The (optional) second parameter
   // specifies whether the existing elements should be preserved. Per default, the existing
   // elements are not preserved.
   v1.resize( 5UL );         // Resizing vector v1 to 5 elements. Elements of built-in type remain
                             // uninitialized, elements of class type are default constructed.
   v1.resize( 3UL, false );  // Resizing vector v1 to 3 elements. The old elements are lost, the
                             // new elements are NOT initialized!
   v2.resize( 8UL, true );   // Resizing vector v2 to 8 elements. The old elements are preserved.
   v2.resize( 5UL, false );  // Resizing vector v2 to 5 elements. The old elements are lost.
   \endcode

// Note that resizing a vector invalidates all existing views (see e.g. \ref views_subvectors)
// on the vector:

   \code
   typedef blaze::DynamicVector<int,rowVector>  VectorType;
   typedef blaze::Subvector<VectorType>         SubvectorType;

   VectorType v1( 10UL );                         // Creating a dynamic vector of size 10
   SubvectorType sv = subvector( v1, 2UL, 5UL );  // Creating a view on the range [2..6]
   v1.resize( 6UL );                              // Resizing the vector invalidates the view
   \endcode

// When the internal capacity of a vector is no longer sufficient, the allocation of a larger
// junk of memory is triggered. In order to avoid frequent reallocations, the \c reserve()
// function can be used up front to set the internal capacity:

   \code
   blaze::DynamicVector<int> v1;
   v1.reserve( 100 );
   v1.size();      // Returns 0
   v1.capacity();  // Returns at least 100
   \endcode

// Note that the size of the vector remains unchanged, but only the internal capacity is set
// according to the specified value!
//
//
// \n \section vector_operations_free_functions Free Functions
// <hr>
//
// \subsection vector_operations_reset_clear reset() / clear()
//
// In order to reset all elements of a vector, the \c reset() function can be used:

   \code
   // Setup of a single precision column vector, whose elements are initialized with 2.0F.
   blaze::DynamicVector<float> v1( 3UL, 2.0F );

   // Resetting all elements to 0.0F. Only the elements are reset, the size of the vector is unchanged.
   reset( v1 );  // Resetting all elements
   v1.size();    // Returns 3: size and capacity remain unchanged
   \endcode

// In order to return a vector to its default state (i.e. the state of a default constructed
// vector), the \c clear() function can be used:

   \code
   // Setup of a single precision column vector, whose elements are initialized with -1.0F.
   blaze::DynamicVector<float> v1( 5, -1.0F );

   // Resetting the entire vector.
   clear( v1 );  // Resetting the entire vector
   v1.size();    // Returns 0: size is reset, but capacity remains unchanged
   \endcode

// Note that resetting or clearing both dense and sparse vectors does not change the capacity
// of the vectors.
//
//
// \n \subsection vector_operations_isnan isnan()
//
// The \c isnan() function provides the means to check a dense or sparse vector for non-a-number
// elements:

   \code
   blaze::DynamicVector<double> a;
   // ... Resizing and initialization
   if( isnan( a ) ) { ... }
   \endcode

   \code
   blaze::CompressedVector<double> a;
   // ... Resizing and initialization
   if( isnan( a ) ) { ... }
   \endcode

// If at least one element of the vector is not-a-number, the function returns \c true, otherwise
// it returns \c false. Please note that this function only works for vectors with floating point
// elements. The attempt to use it for a vector with a non-floating point element type results in
// a compile time error.
//
//
// \n \subsection vector_operations_isdefault isDefault()
//
// The \c isDefault() function returns whether the given dense or sparse vector is in default state:

   \code
   blaze::HybridVector<int,20UL> a;
   // ... Resizing and initialization
   if( isDefault( a ) ) { ... }
   \endcode

// A vector is in default state if it appears to just have been default constructed. All resizable
// vectors (\c HybridVector, \c DynamicVector, or \c CompressedVector) and \c CustomVector are
// in default state if its size is equal to zero. A non-resizable vector (\c StaticVector, all
// subvectors, rows, and columns) is in default state if all its elements are in default state.
// For instance, in case the vector is instantiated for a built-in integral or floating point data
// type, the function returns \c true in case all vector elements are 0 and \c false in case any
// vector element is not 0.
//
//
// \n \subsection vector_operations_isUniform isUniform()
//
// In order to check if all vector elements are identical, the \c isUniform function can be used:

   \code
   blaze::DynamicVector<int> a;
   // ... Resizing and initialization
   if( isUniform( a ) ) { ... }
   \endcode

// Note that in case of sparse vectors also the zero elements are also taken into account!
//
//
// \n \subsection vector_operations_min_max min() / max()
//
// The \c min() and the \c max() functions return the smallest and largest element of the given
// dense or sparse vector, respectively:

   \code
   blaze::StaticVector<int,4UL,rowVector> a{ -5, 2,  7,  4 };
   blaze::StaticVector<int,4UL,rowVector> b{ -5, 2, -7, -4 };

   min( a );  // Returns -5
   min( b );  // Returns -7

   max( a );  // Returns 7
   max( b );  // Returns 2
   \endcode

// In case the vector currently has a size of 0, both functions return 0. Additionally, in case
// a given sparse vector is not completely filled, the zero elements are taken into account. For
// example: the following compressed vector has only 2 non-zero elements. However, the minimum
// of this vector is 0:

   \code
   blaze::CompressedVector<int> c( 4UL, 2UL );
   c[0] = 1;
   c[2] = 3;

   min( c );  // Returns 0
   \endcode

// Also note that the \c min() and \c max() functions can be used to compute the smallest and
// largest element of a vector expression:

   \code
   min( a + b + c );  // Returns -9, i.e. the smallest value of the resulting vector
   max( a - b - c );  // Returns 11, i.e. the largest value of the resulting vector
   \endcode

// \n \subsection vector_operators_abs abs()
//
// The \c abs() function can be used to compute the absolute values of each element of a vector.
// For instance, the following computation

   \code
   blaze::StaticVector<int,3UL,rowVector> a{ -1, 2, -3 };
   blaze::StaticVector<int,3UL,rowVector> b( abs( a ) );
   \endcode

// results in the vector

                          \f$ b = \left(\begin{array}{*{1}{c}}
                          1 \\
                          2 \\
                          3 \\
                          \end{array}\right)\f$

// \n \subsection vector_operators_floor_ceil floor() / ceil()
//
// The \c floor() and \c ceil() functions can be used to round down/up each element of a vector,
// respectively:

   \code
   blaze::StaticVector<double,3UL,rowVector> a, b;

   b = floor( a );  // Rounding down each element of the vector
   b = ceil( a );   // Rounding up each element of the vector
   \endcode

// \n \subsection vector_operators_conj conj()
//
// The \c conj() function can be applied on a dense or sparse vector to compute the complex
// conjugate of each element of the vector:

   \code
   using blaze::StaticVector;

   typedef std::complex<double>  cplx;

   // Creating the vector
   //    ( (-2,-1) )
   //    ( ( 1, 1) )
   StaticVector<cplx,2UL> a{ cplx(-2.0,-1.0), cplx(1.0,1.0) };

   // Computing the vector of complex conjugates
   //    ( (-2, 1) )
   //    ( ( 1,-1) )
   StaticVector<cplx,2UL> b;
   b = conj( a );
   \endcode

// Additionally, vectors can be conjugated in-place via the \c conjugate() function:

   \code
   blaze::DynamicVector<cplx> c( 5UL );

   conjugate( c );  // In-place conjugate operation.
   c = conj( c );   // Same as above
   \endcode

// \n \subsection vector_operators_real real()
//
// The \c real() function can be used on a dense or sparse vector to extract the real part of
// each element of the vector:

   \code
   using blaze::StaticVector;

   typedef std::complex<double>  cplx;

   // Creating the vector
   //    ( (-2,-1) )
   //    ( ( 1, 1) )
   StaticVector<cplx,2UL> a{ cplx(-2.0,-1.0), cplx(1.0,1.0) };

   // Extracting the real part of each vector element
   //    ( -2 )
   //    (  1 )
   StaticVector<double,2UL> b;
   b = real( a );
   \endcode

// \n \subsection vector_operators_imag imag()
//
// The \c imag() function can be used on a dense or sparse vector to extract the imaginary part
// of each element of the vector:

   \code
   using blaze::StaticVector;

   typedef std::complex<double>  cplx;

   // Creating the vector
   //    ( (-2,-1) )
   //    ( ( 1, 1) )
   StaticVector<cplx,2UL> a{ cplx(-2.0,-1.0), cplx(1.0,1.0) };

   // Extracting the imaginary part of each vector element
   //    ( -1 )
   //    (  1 )
   StaticVector<double,2UL> b;
   b = imag( a );
   \endcode

// \n \subsection vector_operations_sqrt sqrt() / invsqrt()
//
// Via the \c sqrt() and \c invsqrt() functions the (inverse) square root of each element of a
// vector can be computed:

   \code
   blaze::DynamicVector<double> a, b, c;

   b = sqrt( a );     // Computes the square root of each element
   c = invsqrt( a );  // Computes the inverse square root of each element
   \endcode

// Note that in case of sparse vectors only the non-zero elements are taken into account!
//
//
// \n \subsection vector_operations_cbrt cbrt() / invcbrt()
//
// The \c cbrt() and \c invcbrt() functions can be used to compute the the (inverse) cubic root
// of each element of a vector:

   \code
   blaze::HybridVector<double,3UL> a, b, c;

   b = cbrt( a );     // Computes the cubic root of each element
   c = invcbrt( a );  // Computes the inverse cubic root of each element
   \endcode

// Note that in case of sparse vectors only the non-zero elements are taken into account!
//
//
// \n \subsection vector_operations_pow pow()
//
// The \c pow() function can be used to compute the exponential value of each element of a vector:

   \code
   blaze::StaticVector<double,3UL> a, b;

   b = pow( a, 1.2 );  // Computes the exponential value of each element
   \endcode

// \n \subsection vector_operations_exp exp()
//
// \c exp() computes the base e exponential of each element of a vector:

   \code
   blaze::DynamicVector<double> a, b;

   b = exp( a );  // Computes the base e exponential of each element
   \endcode

// Note that in case of sparse vectors only the non-zero elements are taken into account!
//
//
// \n \subsection vector_operations_log log() / log10()
//
// The \c log() and \c log10() functions can be used to compute the natural and common logarithm
// of each element of a vector:

   \code
   blaze::StaticVector<double,3UL> a, b;

   b = log( a );    // Computes the natural logarithm of each element
   b = log10( a );  // Computes the common logarithm of each element
   \endcode

// \n \subsection vector_operations_trigonometric_functions sin() / cos() / tan() / asin() / acos() / atan()
//
// The following trigonometric functions are available for both dense and sparse vectors:

   \code
   blaze::DynamicVector<double> a, b;

   b = sin( a );  // Computes the sine of each element of the vector
   b = cos( a );  // Computes the cosine of each element of the vector
   b = tan( a );  // Computes the tangent of each element of the vector

   b = asin( a );  // Computes the inverse sine of each element of the vector
   b = acos( a );  // Computes the inverse cosine of each element of the vector
   b = atan( a );  // Computes the inverse tangent of each element of the vector
   \endcode

// Note that in case of sparse vectors only the non-zero elements are taken into account!
//
//
// \n \subsection vector_operations_hyperbolic_functions sinh() / cosh() / tanh() / asinh() / acosh() / atanh()
//
// The following hyperbolic functions are available for both dense and sparse vectors:

   \code
   blaze::DynamicVector<double> a, b;

   b = sinh( a );  // Computes the hyperbolic sine of each element of the vector
   b = cosh( a );  // Computes the hyperbolic cosine of each element of the vector
   b = tanh( a );  // Computes the hyperbolic tangent of each element of the vector

   b = asinh( a );  // Computes the inverse hyperbolic sine of each element of the vector
   b = acosh( a );  // Computes the inverse hyperbolic cosine of each element of the vector
   b = atanh( a );  // Computes the inverse hyperbolic tangent of each element of the vector
   \endcode

// Note that in case of sparse vectors only the non-zero elements are taken into account!
//
//
// \n \subsection vector_operations_erf erf() / erfc()
//
// The \c erf() and \c erfc() functions compute the (complementary) error function of each
// element of a vector:

   \code
   blaze::StaticVector<double,3UL,rowVector> a, b;

   b = erf( a );   // Computes the error function of each element
   b = erfc( a );  // Computes the complementary error function of each element
   \endcode

// Note that in case of sparse vectors only the non-zero elements are taken into account!
//
//
// \n \subsection vector_operations_foreach forEach()
//
// Via the \c forEach() function it is possible to execute custom operations on dense and sparse
// vectors. For instance, the following example demonstrates a custom square root computation via
// a lambda:

   \code
   blaze::DynamicVector<double> a, b;

   b = forEach( a, []( double d ) { return std::sqrt( d ); } );
   \endcode

// Although the computation can be parallelized it is not vectorized and thus cannot perform at
// peak performance. However, it is also possible to create vectorized custom operations. See
// \ref custom_operations for a detailed overview of the possibilities of custom operations.
//
//
// \n \subsection vector_operations_length length() / sqrLength()
//
// In order to calculate the length of a vector, both the \c length() and \c sqrLength() function
// can be used:

   \code
   blaze::StaticVector<float,3UL,rowVector> v{ -1.2F, 2.7F, -2.3F };

   const float len    = length   ( v );  // Computes the current length of the vector
   const float sqrlen = sqrLength( v );  // Computes the square length of the vector
   \endcode

// Note that both functions can only be used for vectors with built-in or complex element type!
//
//
// \n \subsection vector_operations_vector_transpose trans()
//
// As already mentioned, vectors can either be column vectors (blaze::columnVector) or row vectors
// (blaze::rowVector). A column vector cannot be assigned to a row vector and vice versa. However,
// vectors can be transposed via the \c trans() function:

   \code
   blaze::DynamicVector<int,columnVector> v1( 4UL );
   blaze::CompressedVector<int,rowVector> v2( 4UL );

   v1 = v2;            // Compilation error: Cannot assign a row vector to a column vector
   v1 = trans( v2 );   // OK: Transposing the row vector to a column vector and assigning it
                       //     to the column vector v1
   v2 = trans( v1 );   // OK: Transposing the column vector v1 and assigning it to the row vector v2
   v1 += trans( v2 );  // OK: Addition assignment of two column vectors
   \endcode

// \n \subsection vector_operations_conjugate_transpose ctrans()
//
// It is also possible to compute the conjugate transpose of a vector. This operation is available
// via the \c ctrans() function:

   \code
   blaze::CompressedVector< complex<float>, rowVector > v1( 4UL );
   blaze::DynamicVector< complex<float>, columnVector > v2( 4UL );

   v1 = ctrans( v2 );  // Compute the conjugate transpose vector
   \endcode

// Note that the \c ctrans() function has the same effect as manually applying the \c conj() and
// \c trans() function in any order:

   \code
   v1 = trans( conj( v2 ) );  // Computing the conjugate transpose vector
   v1 = conj( trans( v2 ) );  // Computing the conjugate transpose vector
   \endcode

// \n \subsection vector_operations_normalize normalize()
//
// The \c normalize() function can be used to scale any non-zero vector to a length of 1. In
// case the vector does not contain a single non-zero element (i.e. is a zero vector), the
// \c normalize() function returns a zero vector.

   \code
   blaze::DynamicVector<float,columnVector>     v1( 10UL );
   blaze::CompressedVector<double,columnVector> v2( 12UL );

   v1 = normalize( v1 );  // Normalizing the dense vector v1
   length( v1 );          // Returns 1 (or 0 in case of a zero vector)
   v1 = normalize( v2 );  // Assigning v1 the normalized vector v2
   length( v1 );          // Returns 1 (or 0 in case of a zero vector)
   \endcode

// Note that the \c normalize() function only works for floating point vectors. The attempt to
// use it for an integral vector results in a compile time error.
//
// \n \subsection vector_operations_swap swap()
//
// Via the \c swap() function it is possible to completely swap the contents of two vectors of
// the same type:

   \code
   blaze::DynamicVector<int,columnVector> v1( 10UL );
   blaze::DynamicVector<int,columnVector> v2( 20UL );

   swap( v1, v2 );  // Swapping the contents of v1 and v2
   \endcode

// \n Previous: \ref vector_types &nbsp; &nbsp; Next: \ref matrices
*/
//*************************************************************************************************


//**Matrices***************************************************************************************
/*!\page matrices Matrices
//
// \tableofcontents
//
//
// \n \section matrices_general General Concepts
// <hr>
//
// The \b Blaze library currently offers four dense matrix types (\ref matrix_types_static_matrix,
// \ref matrix_types_dynamic_matrix, \ref matrix_types_hybrid_matrix, and \ref matrix_types_custom_matrix)
// and one sparse matrix type (\ref matrix_types_compressed_matrix). All matrices can either be
// stored as row-major matrices or column-major matrices:

   \code
   using blaze::DynamicMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   // Setup of the 2x3 row-major dense matrix
   //
   //    ( 1  2  3 )
   //    ( 4  5  6 )
   //
   DynamicMatrix<int,rowMajor> A{ { 1, 2, 3 },
                                  { 4, 5, 6 } };

   // Setup of the 3x2 column-major dense matrix
   //
   //    ( 1  4 )
   //    ( 2  5 )
   //    ( 3  6 )
   //
   DynamicMatrix<int,columnMajor> B{ { 1, 4 },
                                     { 2, 5 },
                                     { 3, 6 } };
   \endcode

// Per default, all matrices in \b Blaze are row-major matrices:

   \code
   // Instantiation of a 3x3 row-major matrix
   blaze::DynamicMatrix<int> C( 3UL, 3UL );
   \endcode

// \n \section matrices_details Matrix Details
// <hr>
//
//  - \ref matrix_types
//  - \ref matrix_operations
//
//
// \n \section matrices_examples Examples
// <hr>

   \code
   using blaze::StaticMatrix;
   using blaze::DynamicMatrix;
   using blaze::CompressedMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   StaticMatrix<double,6UL,20UL> A;      // Instantiation of a 6x20 row-major static matrix
   CompressedMatrix<double,rowMajor> B;  // Instantiation of a row-major compressed matrix
   DynamicMatrix<double,columnMajor> C;  // Instantiation of a column-major dynamic matrix

   // ... Resizing and initialization

   C = A * B;
   \endcode

// \n Previous: \ref vector_operations &nbsp; &nbsp; Next: \ref matrix_types
*/
//*************************************************************************************************


//**Matrix Types***********************************************************************************
/*!\page matrix_types Matrix Types
//
// \tableofcontents
//
//
// \n \section matrix_types_static_matrix StaticMatrix
// <hr>
//
// The blaze::StaticMatrix class template is the representation of a fixed size matrix with
// statically allocated elements of arbitrary type. It can be included via the header file

   \code
   #include <blaze/math/StaticMatrix.h>
   \endcode

// The type of the elements, the number of rows and columns, and the storage order of the matrix
// can be specified via the four template parameters:

   \code
   template< typename Type, size_t M, size_t N, bool SO >
   class StaticMatrix;
   \endcode

//  - \c Type: specifies the type of the matrix elements. StaticMatrix can be used with any
//             non-cv-qualified, non-reference element type.
//  - \c M   : specifies the total number of rows of the matrix.
//  - \c N   : specifies the total number of columns of the matrix. Note that it is expected
//             that StaticMatrix is only used for tiny and small matrices.
//  - \c SO  : specifies the storage order (blaze::rowMajor, blaze::columnMajor) of the matrix.
//             The default value is blaze::rowMajor.
//
// The blaze::StaticMatrix is perfectly suited for small to medium matrices whose dimensions are
// known at compile time:

   \code
   // Definition of a 3x4 integral row-major matrix
   blaze::StaticMatrix<int,3UL,4UL> A;

   // Definition of a 4x6 single precision row-major matrix
   blaze::StaticMatrix<float,4UL,6UL,blaze::rowMajor> B;

   // Definition of a 6x4 double precision column-major matrix
   blaze::StaticMatrix<double,6UL,4UL,blaze::columnMajor> C;
   \endcode

// \n \section matrix_types_dynamic_matrix DynamicMatrix
// <hr>
//
// The blaze::DynamicMatrix class template is the representation of an arbitrary sized matrix
// with \f$ M \cdot N \f$ dynamically allocated elements of arbitrary type. It can be included
// via the header file

   \code
   #include <blaze/math/DynamicMatrix.h>
   \endcode

// The type of the elements and the storage order of the matrix can be specified via the two
// template parameters:

   \code
   template< typename Type, bool SO >
   class DynamicMatrix;
   \endcode

//  - \c Type: specifies the type of the matrix elements. DynamicMatrix can be used with any
//             non-cv-qualified, non-reference element type.
//  - \c SO  : specifies the storage order (blaze::rowMajor, blaze::columnMajor) of the matrix.
//             The default value is blaze::rowMajor.
//
// The blaze::DynamicMatrix is the default choice for all kinds of dense matrices and the best
// choice for medium to large matrices. The number of rows and columns can be modified at runtime:

   \code
   // Definition of a 3x4 integral row-major matrix
   blaze::DynamicMatrix<int> A( 3UL, 4UL );

   // Definition of a 4x6 single precision row-major matrix
   blaze::DynamicMatrix<float,blaze::rowMajor> B( 4UL, 6UL );

   // Definition of a double precision column-major matrix with 0 rows and columns
   blaze::DynamicMatrix<double,blaze::columnMajor> C;
   \endcode

// \n \section matrix_types_hybrid_matrix HybridMatrix
// <hr>
//
// The HybridMatrix class template combines the flexibility of a dynamically sized matrix with
// the efficiency and performance of a fixed size matrix. It is implemented as a crossing between
// the blaze::StaticMatrix and the blaze::DynamicMatrix class templates: Similar to the static
// matrix it uses static stack memory instead of dynamically allocated memory and similar to the
// dynamic matrix it can be resized (within the extend of the static memory). It can be included
// via the header file

   \code
   #include <blaze/math/HybridMatrix.h>
   \endcode

// The type of the elements, the maximum number of rows and columns and the storage order of the
// matrix can be specified via the four template parameters:

   \code
   template< typename Type, size_t M, size_t N, bool SO >
   class HybridMatrix;
   \endcode

//  - Type: specifies the type of the matrix elements. HybridMatrix can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
//  - M   : specifies the maximum number of rows of the matrix.
//  - N   : specifies the maximum number of columns of the matrix. Note that it is expected
//          that HybridMatrix is only used for tiny and small matrices.
//  - SO  : specifies the storage order (blaze::rowMajor, blaze::columnMajor) of the matrix.
//          The default value is blaze::rowMajor.
//
// The blaze::HybridMatrix is a suitable choice for small to medium matrices, whose dimensions
// are not known at compile time or not fixed at runtime, but whose maximum dimensions are known
// at compile time:

   \code
   // Definition of a 3x4 integral row-major matrix with maximum dimensions of 6x8
   blaze::HybridMatrix<int,6UL,8UL> A( 3UL, 4UL );

   // Definition of a 4x6 single precision row-major matrix with maximum dimensions of 12x16
   blaze::HybridMatrix<float,12UL,16UL,blaze::rowMajor> B( 4UL, 6UL );

   // Definition of a 0x0 double precision column-major matrix and maximum dimensions of 6x6
   blaze::HybridMatrix<double,6UL,6UL,blaze::columnMajor> C;
   \endcode

// \n \section matrix_types_custom_matrix CustomMatrix
// <hr>
//
// The blaze::CustomMatrix class template provides the functionality to represent an external
// array of elements of arbitrary type and a fixed size as a native \b Blaze dense matrix data
// structure. Thus in contrast to all other dense matrix types a custom matrix does not perform
// any kind of memory allocation by itself, but it is provided with an existing array of element
// during construction. A custom matrix can therefore be considered an alias to the existing
// array. It can be included via the header file

   \code
   #include <blaze/math/CustomMatrix.h>
   \endcode

// The type of the elements, the properties of the given array of elements and the storage order
// of the matrix can be specified via the following four template parameters:

   \code
   template< typename Type, bool AF, bool PF, bool SO >
   class CustomMatrix;
   \endcode

//  - Type: specifies the type of the matrix elements. blaze::CustomMatrix can be used with
//          any non-cv-qualified, non-reference, non-pointer element type.
//  - AF  : specifies whether the represented, external arrays are properly aligned with
//          respect to the available instruction set (SSE, AVX, ...) or not.
//  - PF  : specified whether the represented, external arrays are properly padded with
//          respect to the available instruction set (SSE, AVX, ...) or not.
//  - SO  : specifies the storage order (blaze::rowMajor, blaze::columnMajor) of the matrix.
//          The default value is blaze::rowMajor.
//
// The blaze::CustomMatrix is the right choice if any external array needs to be represented as
// a \b Blaze dense matrix data structure or if a custom memory allocation strategy needs to be
// realized:

   \code
   using blaze::CustomMatrix;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;

   // Definition of an unmanaged 3x4 custom matrix for unaligned, unpadded integer arrays
   typedef CustomMatrix<int,unaligned,unpadded,rowMajor>  UnalignedUnpadded;
   std::vector<int> vec( 12UL )
   UnalignedUnpadded A( &vec[0], 3UL, 4UL );

   // Definition of a managed 5x6 custom matrix for unaligned but padded 'float' arrays
   typedef CustomMatrix<float,unaligned,padded,columnMajor>  UnalignedPadded;
   UnalignedPadded B( new float[40], 5UL, 6UL, 8UL, blaze::ArrayDelete() );

   // Definition of a managed 12x13 custom matrix for aligned, unpadded 'double' arrays
   typedef CustomMatrix<double,aligned,unpadded,rowMajor>  AlignedUnpadded;
   AlignedUnpadded C( blaze::allocate<double>( 192UL ), 12UL, 13UL, 16UL, blaze::Deallocate );

   // Definition of a 7x14 custom matrix for aligned, padded 'complex<double>' arrays
   typedef CustomMatrix<complex<double>,aligned,padded,columnMajor>  AlignedPadded;
   AlignedPadded D( blaze::allocate<double>( 112UL ), 7UL, 14UL, 16UL, blaze::Deallocate() );
   \endcode

// In comparison with the remaining \b Blaze dense matrix types blaze::CustomMatrix has several
// special characteristics. All of these result from the fact that a custom matrix is not
// performing any kind of memory allocation, but instead is given an existing array of elements.
// The following sections discuss all of these characteristics:
//
//  -# <b>\ref matrix_types_custom_matrix_memory_management</b>
//  -# <b>\ref matrix_types_custom_matrix_copy_operations</b>
//  -# <b>\ref matrix_types_custom_matrix_alignment</b>
//  -# <b>\ref matrix_types_custom_matrix_padding</b>
//
// \n \subsection matrix_types_custom_matrix_memory_management Memory Management
//
// The blaze::CustomMatrix class template acts as an adaptor for an existing array of elements. As
// such it provides everything that is required to use the array just like a native \b Blaze dense
// matrix data structure. However, this flexibility comes with the price that the user of a custom
// matrix is responsible for the resource management.
//
// When constructing a custom matrix there are two choices: Either a user manually manages the
// array of elements outside the custom matrix, or alternatively passes the responsibility for
// the memory management to an instance of CustomMatrix. In the second case the CustomMatrix
// class employs shared ownership between all copies of the custom matrix, which reference the
// same array.
//
// The following examples give an impression of several possible types of custom matrices:

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

// \n \subsection matrix_types_custom_matrix_copy_operations Copy Operations
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

// \n \subsection matrix_types_custom_matrix_alignment Alignment
//
// In case the custom matrix is specified as \c aligned the passed array must adhere to some
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
// violated, a \c std::invalid_argument exception is thrown.
//
// \n \subsection matrix_types_custom_matrix_padding Padding
//
// Adding padding elements to the end of each row/column can have a significant impact on the
// performance. For instance, assuming that AVX is available, then two aligned, padded, 3x3 double
// precision matrices can be added via three SIMD addition operations:

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
// \c std::invalid_argument exception is thrown.
//
//
// \n \section matrix_types_compressed_matrix CompressedMatrix
// <hr>
//
// The blaze::CompressedMatrix class template is the representation of an arbitrary sized sparse
// matrix with \f$ M \cdot N \f$ dynamically allocated elements of arbitrary type. It can be
// included via the header file

   \code
   #include <blaze/math/CompressedMatrix.h>
   \endcode

// The type of the elements and the storage order of the matrix can be specified via the two
// template parameters:

   \code
   template< typename Type, bool SO >
   class CompressedMatrix;
   \endcode

//  - \c Type: specifies the type of the matrix elements. CompressedMatrix can be used with
//             any non-cv-qualified, non-reference, non-pointer element type.
//  - \c SO  : specifies the storage order (blaze::rowMajor, blaze::columnMajor) of the matrix.
//             The default value is blaze::rowMajor.
//
// The blaze::CompressedMatrix is the right choice for all kinds of sparse matrices:

   \code
   // Definition of a 3x4 integral row-major matrix
   blaze::CompressedMatrix<int> A( 3UL, 4UL );

   // Definition of a 4x6 single precision row-major matrix
   blaze::CompressedMatrix<float,blaze::rowMajor> B( 4UL, 6UL );

   // Definition of a double precision column-major matrix with 0 rows and columns
   blaze::CompressedMatrix<double,blaze::columnMajor> C;
   \endcode

// \n Previous: \ref matrices &nbsp; &nbsp; Next: \ref matrix_operations
*/
//*************************************************************************************************


//**Matrix Operations******************************************************************************
/*!\page matrix_operations Matrix Operations
//
// \tableofcontents
//
//
// \n \section matrix_operations_constructors Constructors
// <hr>
//
// Matrices are just as easy and intuitive to create as vectors. Still, there are a few rules
// to be aware of:
//  - In case the last template parameter (the storage order) is omitted, the matrix is per
//    default stored in row-major order.
//  - The elements of a \c StaticMatrix or \c HybridMatrix are default initialized (i.e. built-in
//    data types are initialized to 0, class types are initialized via the default constructor).
//  - Newly allocated elements of a \c DynamicMatrix or \c CompressedMatrix remain uninitialized
//    if they are of built-in type and are default constructed if they are of class type.
//
// \n \subsection matrix_operations_default_construction Default Construction

   \code
   using blaze::StaticMatrix;
   using blaze::DynamicMatrix;
   using blaze::CompressedMatrix;

   // All matrices can be default constructed. Whereas the size of
   // a StaticMatrix is fixed via the second and third template
   // parameter, the initial size of a constructed DynamicMatrix
   // or CompressedMatrix is 0.
   StaticMatrix<int,2UL,2UL> M1;             // Instantiation of a 2x2 integer row-major
                                             // matrix. All elements are initialized to 0.
   DynamicMatrix<float> M2;                  // Instantiation of a single precision dynamic
                                             // row-major matrix with 0 rows and 0 columns.
   DynamicMatrix<double,columnMajor> M3;     // Instantiation of a double precision dynamic
                                             // column-major matrix with 0 rows and 0 columns.
   CompressedMatrix<int> M4;                 // Instantiation of a compressed integer
                                             // row-major matrix of size 0x0.
   CompressedMatrix<double,columnMajor> M5;  // Instantiation of a compressed double precision
                                             // column-major matrix of size 0x0.
   \endcode

// \n \subsection matrix_operations_size_construction Construction with Specific Size
//
// The \c DynamicMatrix, \c HybridMatrix, and \c CompressedMatrix classes offer a constructor
// that allows to immediately give the matrices a specific number of rows and columns:

   \code
   DynamicMatrix<int> M6( 5UL, 4UL );                   // Instantiation of a 5x4 dynamic row-major
                                                        // matrix. The elements are not initialized.
   HybridMatrix<double,5UL,9UL> M7( 3UL, 7UL );         // Instantiation of a 3x7 hybrid row-major
                                                        // matrix. The elements are not initialized.
   CompressedMatrix<float,columnMajor> M8( 8UL, 6UL );  // Instantiation of an empty 8x6 compressed
                                                        // column-major matrix.
   \endcode

// Note that dense matrices (in this case \c DynamicMatrix and \c HybridMatrix) immediately
// allocate enough capacity for all matrix elements. Sparse matrices on the other hand (in this
// example \c CompressedMatrix) merely acquire the size, but don't necessarily allocate memory.
//
//
// \n \subsection matrix_operations_initialization_constructors Initialization Constructors
//
// All dense matrix classes offer a constructor for a direct, homogeneous initialization of all
// matrix elements. In contrast, for sparse matrices the predicted number of non-zero elements
// can be specified.

   \code
   StaticMatrix<int,4UL,3UL,columnMajor> M9( 7 );  // Instantiation of a 4x3 integer column-major
                                                   // matrix. All elements are initialized to 7.
   DynamicMatrix<float> M10( 2UL, 5UL, 2.0F );     // Instantiation of a 2x5 single precision row-major
                                                   // matrix. All elements are initialized to 2.0F.
   CompressedMatrix<int> M11( 3UL, 4UL, 4 );       // Instantiation of a 3x4 integer row-major
                                                   // matrix with capacity for 4 non-zero elements.
   \endcode

// \n \subsection matrix_operations_array_construction Array Construction
//
// Alternatively, all dense matrix classes offer a constructor for an initialization with a
// dynamic or static array. If the matrix is initialized from a dynamic array, the constructor
// expects the dimensions of values provided by the array as first and second argument, the
// array as third argument. In case of a static array, the fixed size of the array is used:

   \code
   const std::unique_ptr<double[]> array1( new double[6] );
   // ... Initialization of the dynamic array
   blaze::StaticMatrix<double,2UL,3UL> M12( 2UL, 3UL, array1.get() );

   int array2[2][2] = { { 4, -5 }, { -6, 7 } };
   blaze::StaticMatrix<int,2UL,2UL,rowMajor> M13( array2 );
   \endcode

// \n \subsection matrix_operations_initializer_list_construction
//
// In addition, all dense matrix classes can be directly initialized by means of an initializer
// list:

   \code
   blaze::DynamicMatrix<float,columnMajor> M14{ {  3.1F,  6.4F },
                                                { -0.9F, -1.2F },
                                                {  4.8F,  0.6F } };
   \endcode

// \n \subsection matrix_operations_copy_construction Copy Construction
//
// All dense and sparse matrices can be created as a copy of another dense or sparse matrix.

   \code
   StaticMatrix<int,5UL,4UL,rowMajor> M15( M6 );    // Instantiation of the dense row-major matrix M15
                                                    // as copy of the dense row-major matrix M6.
   DynamicMatrix<float,columnMajor> M16( M8 );      // Instantiation of the dense column-major matrix M16
                                                    // as copy of the sparse column-major matrix M8.
   CompressedMatrix<double,columnMajor> M17( M7 );  // Instantiation of the compressed column-major matrix
                                                    // M17 as copy of the dense row-major matrix M7.
   CompressedMatrix<float,rowMajor> M18( M8 );      // Instantiation of the compressed row-major matrix
                                                    // M18 as copy of the compressed column-major matrix M8.
   \endcode

// Note that it is not possible to create a \c StaticMatrix as a copy of a matrix with a different
// number of rows and/or columns:

   \code
   StaticMatrix<int,4UL,5UL,rowMajor> M19( M6 );     // Runtime error: Number of rows and columns
                                                     // does not match!
   StaticMatrix<int,4UL,4UL,columnMajor> M20( M9 );  // Compile time error: Number of columns does
                                                     // not match!
   \endcode

// \n \section matrix_operations_assignment Assignment
// <hr>
//
// There are several types of assignment to dense and sparse matrices:
// \ref matrix_operations_homogeneous_assignment, \ref matrix_operations_array_assignment,
// \ref matrix_operations_copy_assignment, and \ref matrix_operations_compound_assignment.
//
//
// \n \subsection matrix_operations_homogeneous_assignment Homogeneous Assignment
//
// It is possible to assign the same value to all elements of a dense matrix. All dense matrix
// classes provide an according assignment operator:

   \code
   blaze::StaticMatrix<int,3UL,2UL> M1;
   blaze::DynamicMatrix<double> M2;

   // Setting all integer elements of the StaticMatrix to 4
   M1 = 4;

   // Setting all double precision elements of the DynamicMatrix to 3.5
   M2 = 3.5
   \endcode

// \n \subsection matrix_operations_array_assignment Array Assignment
//
// Dense matrices can also be assigned a static array:

   \code
   blaze::StaticMatrix<int,2UL,2UL,rowMajor> M1;
   blaze::StaticMatrix<int,2UL,2UL,columnMajor> M2;
   blaze::DynamicMatrix<double> M3;

   int array1[2][2] = { { 1, 2 }, { 3, 4 } };
   double array2[3][2] = { { 3.1, 6.4 }, { -0.9, -1.2 }, { 4.8, 0.6 } };

   M1 = array1;
   M2 = array1;
   M3 = array2;
   \endcode

// Note that the dimensions of the static array have to match the size of a \c StaticMatrix,
// whereas a \c DynamicMatrix is resized according to the array dimensions:

                          \f$ M3 = \left(\begin{array}{*{2}{c}}
                           3.1 &  6.4 \\
                          -0.9 & -1.2 \\
                           4.8 &  0.6 \\
                          \end{array}\right)\f$

// \n \subsection matrix_operations_initializer_list_assignment Initializer List Assignment
//
// Alternatively, it is possible to directly assign an initializer list to a dense matrix:

   \code
   blaze::DynamicMatrix<double> M;
   M = { { 3.1, 6.4 }, { -0.9, -1.2 }, { 4.8, 0.6 } };
   \endcode

// \n \subsection matrix_operations_copy_assignment Copy Assignment
//
// All kinds of matrices can be assigned to each other. The only restriction is that since a
// \c StaticMatrix cannot change its size, the assigned matrix must match both in the number of
// rows and in the number of columns.

   \code
   blaze::StaticMatrix<int,3UL,2UL,rowMajor>  M1;
   blaze::DynamicMatrix<int,rowMajor>         M2( 3UL, 2UL );
   blaze::DynamicMatrix<float,rowMajor>       M3( 5UL, 2UL );
   blaze::CompressedMatrix<int,rowMajor>      M4( 3UL, 2UL );
   blaze::CompressedMatrix<float,columnMajor> M5( 3UL, 2UL );

   // ... Initialization of the matrices

   M1 = M2;  // OK: Assignment of a 3x2 dense row-major matrix to another 3x2 dense row-major matrix
   M1 = M4;  // OK: Assignment of a 3x2 sparse row-major matrix to a 3x2 dense row-major matrix
   M1 = M3;  // Runtime error: Cannot assign a 5x2 matrix to a 3x2 static matrix
   M1 = M5;  // OK: Assignment of a 3x2 sparse column-major matrix to a 3x2 dense row-major matrix
   \endcode

// \n \subsection matrix_operations_compound_assignment Compound Assignment
//
// Compound assignment is also available for matrices: addition assignment, subtraction assignment,
// and multiplication assignment. In contrast to plain assignment, however, the number of rows
// and columns of the two operands have to match according to the arithmetic operation.

   \code
   blaze::StaticMatrix<int,2UL,3UL,rowMajor>    M1;
   blaze::DynamicMatrix<int,rowMajor>           M2( 2UL, 3UL );
   blaze::CompressedMatrix<float,columnMajor>   M3( 2UL, 3UL );
   blaze::CompressedMatrix<float,rowMajor>      M4( 2UL, 4UL );
   blaze::StaticMatrix<float,2UL,4UL,rowMajor>  M5;
   blaze::CompressedMatrix<float,rowMajor>      M6( 3UL, 2UL );

   // ... Initialization of the matrices

   M1 += M2;  // OK: Addition assignment between two row-major matrices of the same dimensions
   M1 -= M3;  // OK: Subtraction assignment between between a row-major and a column-major matrix
   M1 += M4;  // Runtime error: No compound assignment between matrices of different size
   M1 -= M5;  // Compilation error: No compound assignment between matrices of different size
   M2 *= M6;  // OK: Multiplication assignment between two row-major matrices
   \endcode

// Note that the multiplication assignment potentially changes the number of columns of the
// target matrix:

                          \f$\left(\begin{array}{*{3}{c}}
                          2 & 0 & 1 \\
                          0 & 3 & 2 \\
                          \end{array}\right) \times
                          \left(\begin{array}{*{2}{c}}
                          4 & 0 \\
                          1 & 0 \\
                          0 & 3 \\
                          \end{array}\right) =
                          \left(\begin{array}{*{2}{c}}
                          8 & 3 \\
                          3 & 6 \\
                          \end{array}\right)\f$

// Since a \c StaticMatrix cannot change its size, only a square StaticMatrix can be used in a
// multiplication assignment with other square matrices of the same dimensions.
//
//
// \n \section matrix_operations_element_access Element Access
// <hr>
//
// The easiest way to access a specific dense or sparse matrix element is via the function call
// operator. The indices to access a matrix are zero-based:

   \code
   blaze::DynamicMatrix<int> M1( 4UL, 6UL );
   M1(0,0) = 1;
   M1(0,1) = 3;
   // ...

   blaze::CompressedMatrix<double> M2( 5UL, 3UL );
   M2(0,2) =  4.1;
   M2(1,1) = -6.3;
   \endcode

// Since dense matrices allocate enough memory for all contained elements, using the function
// call operator on a dense matrix directly returns a reference to the accessed value. In case
// of a sparse matrix, if the accessed value is currently not contained in the matrix, the
// value is inserted into the matrix prior to returning a reference to the value, which can
// be much more expensive than the direct access to a dense matrix. Consider the following
// example:

   \code
   blaze::CompressedMatrix<int> M1( 4UL, 4UL );

   for( size_t i=0UL; i<M1.rows(); ++i ) {
      for( size_t j=0UL; j<M1.columns(); ++j ) {
         ... = M1(i,j);
      }
   }
   \endcode

// Although the compressed matrix is only used for read access within the for loop, using the
// function call operator temporarily inserts 16 non-zero elements into the matrix. Therefore,
// all matrices (sparse as well as dense) offer an alternate way via the \c begin(), \c cbegin(),
// \c end() and \c cend() functions to traverse all contained elements by iterator. Note that
// it is not possible to traverse all elements of the matrix, but that it is only possible to
// traverse elements in a row/column-wise fashion. In case of a non-const matrix, \c begin() and
// \c end() return an \c Iterator, which allows a manipulation of the non-zero value, in case of
// a constant matrix or in case \c cbegin() or \c cend() are used a \c ConstIterator is returned:

   \code
   using blaze::CompressedMatrix;

   CompressedMatrix<int,rowMajor> M1( 4UL, 6UL );

   // Traversing the matrix by Iterator
   for( size_t i=0UL; i<A.rows(); ++i ) {
      for( CompressedMatrix<int,rowMajor>::Iterator it=A.begin(i); it!=A.end(i); ++it ) {
         it->value() = ...;  // OK: Write access to the value of the non-zero element.
         ... = it->value();  // OK: Read access to the value of the non-zero element.
         it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
         ... = it->index();  // OK: Read access to the index of the non-zero element.
      }
   }

   // Traversing the matrix by ConstIterator
   for( size_t i=0UL; i<A.rows(); ++i ) {
      for( CompressedMatrix<int,rowMajor>::ConstIterator it=A.cbegin(i); it!=A.cend(i); ++it ) {
         it->value() = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
         ... = it->value();  // OK: Read access to the value of the non-zero element.
         it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
         ... = it->index();  // OK: Read access to the index of the non-zero element.
      }
   }
   \endcode

// Note that \c begin(), \c cbegin(), \c end(), and \c cend() are also available as free functions:

   \code
   for( size_t i=0UL; i<A.rows(); ++i ) {
      for( CompressedMatrix<int,rowMajor>::Iterator it=begin( A, i ); it!=end( A, i ); ++it ) {
         // ...
      }
   }

   for( size_t i=0UL; i<A.rows(); ++i ) {
      for( CompressedMatrix<int,rowMajor>::ConstIterator it=cbegin( A, i ); it!=cend( A, i ); ++it ) {
         // ...
      }
   }
   \endcode

// \n \section matrix_operations_element_insertion Element Insertion
// <hr>
//
// Whereas a dense matrix always provides enough capacity to store all matrix elements, a sparse
// matrix only stores the non-zero elements. Therefore it is necessary to explicitly add elements
// to the matrix. The first possibility to add elements to a sparse matrix is the function call
// operator:

   \code
   using blaze::CompressedMatrix;

   CompressedMatrix<int> M1( 3UL, 4UL );
   M1(1,2) = 9;
   \endcode

// In case the element at the given position is not yet contained in the sparse matrix, it is
// automatically inserted. Otherwise the old value is replaced by the new value 2. The operator
// returns a reference to the sparse vector element.\n
// An alternative is the \c set() function: In case the element is not yet contained in the matrix
// the element is inserted, else the element's value is modified:

   \code
   // Insert or modify the value at position (2,0)
   M1.set( 2, 0, 1 );
   \endcode

// However, insertion of elements can be better controlled via the \c insert() function. In
// contrast to the function call operator and the \c set() function it emits an exception in case
// the element is already contained in the matrix. In order to check for this case, the \c find()
// function can be used:

   \code
   // In case the element at position (2,3) is not yet contained in the matrix it is inserted
   // with a value of 4.
   if( M1.find( 2, 3 ) == M1.end( 2 ) )
      M1.insert( 2, 3, 4 );
   \endcode

// Although the \c insert() function is very flexible, due to performance reasons it is not
// suited for the setup of large sparse matrices. A very efficient, yet also very low-level
// way to fill a sparse matrix is the \c append() function. It requires the sparse matrix to
// provide enough capacity to insert a new element in the specified row. Additionally, the
// index of the new element must be larger than the index of the previous element in the same
// row. Violating these conditions results in undefined behavior!

   \code
   M1.reserve( 0, 3 );     // Reserving space for three non-zero elements in row 0
   M1.append( 0, 1,  2 );  // Appending the element 2 in row 0 at column index 1
   M1.append( 0, 2, -4 );  // Appending the element -4 in row 0 at column index 2
   // ...
   \endcode

// The most efficient way to fill a sparse matrix with elements, however, is a combination of
// \c reserve(), \c append(), and the \c finalize() function:

   \code
   blaze::CompressedMatrix<int> M1( 3UL, 5UL );
   M1.reserve( 3 );       // Reserving enough space for 3 non-zero elements
   M1.append( 0, 1, 1 );  // Appending the value 1 in row 0 with column index 1
   M1.finalize( 0 );      // Finalizing row 0
   M1.append( 1, 1, 2 );  // Appending the value 2 in row 1 with column index 1
   M1.finalize( 1 );      // Finalizing row 1
   M1.append( 2, 0, 3 );  // Appending the value 3 in row 2 with column index 0
   M1.finalize( 2 );      // Finalizing row 2
   \endcode

// \n \section matrix_operations_member_functions Member Functions
// <hr>
//
// \subsection matrix_operations_rows .rows()
//
// The current number of rows of a matrix can be acquired via the \c rows() member function:

   \code
   // Instantiating a dynamic matrix with 10 rows and 8 columns
   blaze::DynamicMatrix<int> M1( 10UL, 8UL );
   M1.rows();  // Returns 10

   // Instantiating a compressed matrix with 8 rows and 12 columns
   blaze::CompressedMatrix<double> M2( 8UL, 12UL );
   M2.rows();  // Returns 8
   \endcode

// Alternatively, the free functions \c rows() can be used to query the current number of rows of
// a matrix. In contrast to the member function, the free function can also be used to query the
// number of rows of a matrix expression:

   \code
   rows( M1 );  // Returns 10, i.e. has the same effect as the member function
   rows( M2 );  // Returns 8, i.e. has the same effect as the member function

   rows( M1 * M2 );  // Returns 10, i.e. the number of rows of the resulting matrix
   \endcode

// \n \subsection matrix_operations_columns .columns()
//
// The current number of columns of a matrix can be acquired via the \c columns() member function:

   \code
   // Instantiating a dynamic matrix with 6 rows and 8 columns
   blaze::DynamicMatrix<int> M1( 6UL, 8UL );
   M1.columns();   // Returns 8

   // Instantiating a compressed matrix with 8 rows and 7 columns
   blaze::CompressedMatrix<double> M2( 8UL, 7UL );
   M2.columns();   // Returns 7
   \endcode

// There is also a free function \c columns() available, which can also be used to query the number
// of columns of a matrix expression:

   \code
   columns( M1 );  // Returns 8, i.e. has the same effect as the member function
   columns( M2 );  // Returns 7, i.e. has the same effect as the member function

   columns( M1 * M2 );  // Returns 7, i.e. the number of columns of the resulting matrix
   \endcode

// \n \subsection matrix_operations_capacity .capacity()
//
// The \c capacity() member function returns the internal capacity of a dense or sparse matrix.
// Note that the capacity of a matrix doesn't have to be equal to the size of a matrix. In case of
// a dense matrix the capacity will always be greater or equal than the total number of elements
// of the matrix. In case of a sparse matrix, the capacity will usually be much less than the
// total number of elements.

   \code
   blaze::DynamicMatrix<float> M1( 5UL, 7UL );
   blaze::StaticMatrix<float,7UL,4UL> M2;
   M1.capacity();  // Returns at least 35
   M2.capacity();  // Returns at least 28
   \endcode

// There is also a free function \c capacity() available to query the capacity. However, please
// note that this function cannot be used to query the capacity of a matrix expression:

   \code
   capacity( M1 );  // Returns at least 35, i.e. has the same effect as the member function
   capacity( M2 );  // Returns at least 28, i.e. has the same effect as the member function

   capacity( M1 * M2 );  // Compilation error!
   \endcode

// \n \subsection matrix_operations_nonzeros .nonZeros()
//
// For both dense and sparse matrices the current number of non-zero elements can be queried
// via the \c nonZeros() member function. In case of matrices there are two flavors of the
// \c nonZeros() function: One returns the total number of non-zero elements in the matrix,
// the second returns the number of non-zero elements in a specific row (in case of a row-major
// matrix) or column (in case of a column-major matrix). Sparse matrices directly return their
// number of non-zero elements, dense matrices traverse their elements and count the number of
// non-zero elements.

   \code
   blaze::DynamicMatrix<int,rowMajor> M1( 3UL, 5UL );

   // ... Initializing the dense matrix

   M1.nonZeros();     // Returns the total number of non-zero elements in the dense matrix
   M1.nonZeros( 2 );  // Returns the number of non-zero elements in row 2
   \endcode

   \code
   blaze::CompressedMatrix<double,columnMajor> M2( 4UL, 7UL );

   // ... Initializing the sparse matrix

   M2.nonZeros();     // Returns the total number of non-zero elements in the sparse matrix
   M2.nonZeros( 3 );  // Returns the number of non-zero elements in column 3
   \endcode

// The free \c nonZeros() function can also be used to query the number of non-zero elements in a
// matrix expression. However, the result is not the exact number of non-zero elements, but may be
// a rough estimation:

   \code
   nonZeros( M1 );     // Has the same effect as the member function
   nonZeros( M1, 2 );  // Has the same effect as the member function

   nonZeros( M2 );     // Has the same effect as the member function
   nonZeros( M2, 3 );  // Has the same effect as the member function

   nonZeros( M1 * M2 );  // Estimates the number of non-zero elements in the matrix expression
   \endcode

// \n \subsection matrix_operations_resize_reserve .resize() / .reserve()
//
// The dimensions of a \c StaticMatrix are fixed at compile time by the second and third template
// parameter and a \c CustomMatrix cannot be resized. In contrast, the number or rows and columns
// of \c DynamicMatrix, \c HybridMatrix, and \c CompressedMatrix can be changed at runtime:

   \code
   using blaze::DynamicMatrix;
   using blaze::CompressedMatrix;

   DynamicMatrix<int,rowMajor> M1;
   CompressedMatrix<int,columnMajor> M2( 3UL, 2UL );

   // Adapting the number of rows and columns via the resize() function. The (optional)
   // third parameter specifies whether the existing elements should be preserved.
   M1.resize( 2UL, 2UL );         // Resizing matrix M1 to 2x2 elements. Elements of built-in type
                                  // remain uninitialized, elements of class type are default
                                  // constructed.
   M1.resize( 3UL, 1UL, false );  // Resizing M1 to 3x1 elements. The old elements are lost, the
                                  // new elements are NOT initialized!
   M2.resize( 5UL, 7UL, true );   // Resizing M2 to 5x7 elements. The old elements are preserved.
   M2.resize( 3UL, 2UL, false );  // Resizing M2 to 3x2 elements. The old elements are lost.
   \endcode

// Note that resizing a matrix invalidates all existing views (see e.g. \ref views_submatrices)
// on the matrix:

   \code
   typedef blaze::DynamicMatrix<int,rowMajor>  MatrixType;
   typedef blaze::Row<MatrixType>              RowType;

   MatrixType M1( 10UL, 20UL );    // Creating a 10x20 matrix
   RowType row8 = row( M1, 8UL );  // Creating a view on the 8th row of the matrix
   M1.resize( 6UL, 20UL );         // Resizing the matrix invalidates the view
   \endcode

// When the internal capacity of a matrix is no longer sufficient, the allocation of a larger
// junk of memory is triggered. In order to avoid frequent reallocations, the \c reserve()
// function can be used up front to set the internal capacity:

   \code
   blaze::DynamicMatrix<int> M1;
   M1.reserve( 100 );
   M1.rows();      // Returns 0
   M1.capacity();  // Returns at least 100
   \endcode

// Additionally it is possible to reserve memory in a specific row (for a row-major matrix) or
// column (for a column-major matrix):

   \code
   blaze::CompressedMatrix<int> M1( 4UL, 6UL );
   M1.reserve( 1, 4 );  // Reserving enough space for four non-zero elements in row 1
   \endcode

// \n \section matrix_operations_free_functions Free Functions
// <hr>
//
// \subsection matrix_operations_reset_clear reset() / clear
//
// In order to reset all elements of a dense or sparse matrix, the \c reset() function can be
// used. The number of rows and columns of the matrix are preserved:

   \code
   // Setting up a single precision row-major matrix, whose elements are initialized with 2.0F.
   blaze::DynamicMatrix<float> M1( 4UL, 5UL, 2.0F );

   // Resetting all elements to 0.0F.
   reset( M1 );  // Resetting all elements
   M1.rows();    // Returns 4: size and capacity remain unchanged
   \endcode

// Alternatively, only a single row or column of the matrix can be resetted:

   \code
   blaze::DynamicMatrix<int,blaze::rowMajor>    M1( 7UL, 6UL, 5 );  // Setup of a row-major matrix
   blaze::DynamicMatrix<int,blaze::columnMajor> M2( 4UL, 5UL, 4 );  // Setup of a column-major matrix

   reset( M1, 2UL );  // Resetting the 2nd row of the row-major matrix
   reset( M2, 3UL );  // Resetting the 3rd column of the column-major matrix
   \endcode

// In order to reset a row of a column-major matrix or a column of a row-major matrix, use a
// row or column view (see \ref views_rows and views_colums).
//
// In order to return a matrix to its default state (i.e. the state of a default constructed
// matrix), the \c clear() function can be used:

   \code
   // Setting up a single precision row-major matrix, whose elements are initialized with 2.0F.
   blaze::DynamicMatrix<float> M1( 4UL, 5UL, 2.0F );

   // Resetting all elements to 0.0F.
   clear( M1 );  // Resetting the entire matrix
   M1.rows();    // Returns 0: size is reset, but capacity remains unchanged
   \endcode

// \n \subsection matrix_operations_isnan isnan()
//
// The \c isnan() function provides the means to check a dense or sparse matrix for non-a-number
// elements:

   \code
   blaze::DynamicMatrix<double> A( 3UL, 4UL );
   // ... Initialization
   if( isnan( A ) ) { ... }
   \endcode

   \code
   blaze::CompressedMatrix<double> A( 3UL, 4UL );
   // ... Initialization
   if( isnan( A ) ) { ... }
   \endcode

// If at least one element of the matrix is not-a-number, the function returns \c true, otherwise
// it returns \c false. Please note that this function only works for matrices with floating point
// elements. The attempt to use it for a matrix with a non-floating point element type results in
// a compile time error.
//
//
// \n \subsection matrix_operations_isdefault isDefault()
//
// The \c isDefault() function returns whether the given dense or sparse matrix is in default state:

   \code
   blaze::HybridMatrix<int,5UL,4UL> A;
   // ... Resizing and initialization
   if( isDefault( A ) ) { ... }
   \endcode

// A matrix is in default state if it appears to just have been default constructed. All resizable
// matrices (\c HybridMatrix, \c DynamicMatrix, or \c CompressedMatrix) and \c CustomMatrix are in
// default state if its size is equal to zero. A non-resizable matrix (\c StaticMatrix and all
// submatrices) is in default state if all its elements are in default state. For instance, in case
// the matrix is instantiated for a built-in integral or floating point data type, the function
// returns \c true in case all matrix elements are 0 and \c false in case any matrix element is
// not 0.
//
//
// \n \subsection matrix_operations_isSquare isSquare()
//
// Whether a dense or sparse matrix is a square matrix (i.e. if the number of rows is equal to the
// number of columns) can be checked via the \c isSquare() function:

   \code
   blaze::DynamicMatrix<double> A;
   // ... Resizing and initialization
   if( isSquare( A ) ) { ... }
   \endcode

// \n \subsection matrix_operations_issymmetric isSymmetric()
//
// Via the \c isSymmetric() function it is possible to check whether a dense or sparse matrix
// is symmetric:

   \code
   blaze::DynamicMatrix<float> A;
   // ... Resizing and initialization
   if( isSymmetric( A ) ) { ... }
   \endcode

// Note that non-square matrices are never considered to be symmetric!
//
//
// \n \subsection matrix_operations_isUniform isUniform()
//
// In order to check if all matrix elements are identical, the \c isUniform function can be used:

   \code
   blaze::DynamicMatrix<int> A;
   // ... Resizing and initialization
   if( isUniform( A ) ) { ... }
   \endcode

// Note that in case of a sparse matrix also the zero elements are also taken into account!
//
//
// \n \subsection matrix_operations_islower isLower()
//
// Via the \c isLower() function it is possible to check whether a dense or sparse matrix is
// lower triangular:

   \code
   blaze::DynamicMatrix<float> A;
   // ... Resizing and initialization
   if( isLower( A ) ) { ... }
   \endcode

// Note that non-square matrices are never considered to be lower triangular!
//
//
// \n \subsection matrix_operations_isunilower isUniLower()
//
// Via the \c isUniLower() function it is possible to check whether a dense or sparse matrix is
// lower unitriangular:

   \code
   blaze::DynamicMatrix<float> A;
   // ... Resizing and initialization
   if( isUniLower( A ) ) { ... }
   \endcode

// Note that non-square matrices are never considered to be lower unitriangular!
//
//
// \n \subsection matrix_operations_isstrictlylower isStrictlyLower()
//
// Via the \c isStrictlyLower() function it is possible to check whether a dense or sparse matrix
// is strictly lower triangular:

   \code
   blaze::DynamicMatrix<float> A;
   // ... Resizing and initialization
   if( isStrictlyLower( A ) ) { ... }
   \endcode

// Note that non-square matrices are never considered to be strictly lower triangular!
//
//
// \n \subsection matrix_operations_isUpper isUpper()
//
// Via the \c isUpper() function it is possible to check whether a dense or sparse matrix is
// upper triangular:

   \code
   blaze::DynamicMatrix<float> A;
   // ... Resizing and initialization
   if( isUpper( A ) ) { ... }
   \endcode

// Note that non-square matrices are never considered to be upper triangular!
//
//
// \n \subsection matrix_operations_isuniupper isUniUpper()
//
// Via the \c isUniUpper() function it is possible to check whether a dense or sparse matrix is
// upper unitriangular:

   \code
   blaze::DynamicMatrix<float> A;
   // ... Resizing and initialization
   if( isUniUpper( A ) ) { ... }
   \endcode

// Note that non-square matrices are never considered to be upper unitriangular!
//
//
// \n \subsection matrix_operations_isstrictlyupper isStrictlyUpper()
//
// Via the \c isStrictlyUpper() function it is possible to check whether a dense or sparse matrix
// is strictly upper triangular:

   \code
   blaze::DynamicMatrix<float> A;
   // ... Resizing and initialization
   if( isStrictlyUpper( A ) ) { ... }
   \endcode

// Note that non-square matrices are never considered to be strictly upper triangular!
//
//
// \n \subsection matrix_operations_isdiagonal isDiagonal()
//
// The \c isDiagonal() function checks if the given dense or sparse matrix is a diagonal matrix,
// i.e. if it has only elements on its diagonal and if the non-diagonal elements are default
// elements:

   \code
   blaze::CompressedMatrix<float> A;
   // ... Resizing and initialization
   if( isDiagonal( A ) ) { ... }
   \endcode

// Note that non-square matrices are never considered to be diagonal!
//
//
// \n \subsection matrix_operations_isidentity isIdentity()
//
// The \c isIdentity() function checks if the given dense or sparse matrix is an identity matrix,
// i.e. if all diagonal elements are 1 and all non-diagonal elements are 0:

   \code
   blaze::CompressedMatrix<float> A;
   // ... Resizing and initialization
   if( isIdentity( A ) ) { ... }
   \endcode

// Note that non-square matrices are never considered to be identity matrices!
//
//
// \n \subsection matrix_operations_min_max min() / max()
//
// The \c min() and the \c max() functions return the smallest and largest element of the given
// dense or sparse matrix, respectively:

   \code
   blaze::StaticMatrix<int,2UL,3UL,rowMajor> A{ { -5, 2, 7 },
                                                {  4, 0, 1 } };
   blaze::StaticMatrix<int,2UL,3UL,rowMajor> B{ { -5, 2, -7 },
                                                { -4, 0, -1 } };

   min( A );  // Returns -5
   min( B );  // Returns -7

   max( A );  // Returns 7
   max( B );  // Returns 2
   \endcode

// In case the matrix currently has 0 rows or 0 columns, both functions return 0. Additionally, in
// case a given sparse matrix is not completely filled, the zero elements are taken into account.
// For example: the following compressed matrix has only 2 non-zero elements. However, the minimum
// of this matrix is 0:

   \code
   blaze::CompressedMatrix<int> C( 2UL, 3UL );
   C(0,0) = 1;
   C(0,2) = 3;

   min( C );  // Returns 0
   \endcode

// Also note that the \c min() and \c max() functions can be used to compute the smallest and
// largest element of a matrix expression:

   \code
   min( A + B + C );  // Returns -9, i.e. the smallest value of the resulting matrix
   max( A - B - C );  // Returns 11, i.e. the largest value of the resulting matrix
   \endcode

// \n \subsection matrix_operators_abs abs()
//
// The \c abs() function can be used to compute the absolute values of each element of a matrix.
// For instance, the following computation

   \code
   blaze::StaticMatrix<int,2UL,3UL,rowMajor> A{ { -1,  2, -3 },
                                                {  4, -5,  6 } };
   blaze::StaticMatrix<int,2UL,3UL,rowMajor> B( abs( A ) );
   \endcode

// results in the matrix

                          \f$ B = \left(\begin{array}{*{3}{c}}
                          1 & 2 & 3 \\
                          4 & 5 & 6 \\
                          \end{array}\right)\f$

// \n \subsection matrix_operators_floor_ceil floor() / ceil()
//
// The \c floor() and \c ceil() functions can be used to round down/up each element of a matrix,
// respectively:

   \code
   blaze::StaticMatrix<double,3UL,3UL> A, B;

   B = floor( A );  // Rounding down each element of the matrix
   B = ceil( A );   // Rounding up each element of the matrix
   \endcode

// \n \subsection matrix_operators_conj conj()
//
// The \c conj() function can be applied on a dense or sparse matrix to compute the complex
// conjugate of each element of the matrix:

   \code
   using blaze::StaticMatrix;

   typedef std::complex<double>  cplx;

   // Creating the matrix
   //    ( (1,0)  (-2,-1) )
   //    ( (1,1)  ( 0, 1) )
   StaticMatrix<cplx,2UL,2UL> A{ { cplx( 1.0, 0.0 ), cplx( -2.0, -1.0 ) },
                                 { cplx( 1.0, 1.0 ), cplx(  0.0,  1.0 ) } };

   // Computing the matrix of conjugate values
   //    ( (1, 0)  (-2, 1) )
   //    ( (1,-1)  ( 0,-1) )
   StaticMatrix<cplx,2UL,2UL> B;
   B = conj( A );
   \endcode

// Additionally, matrices can be conjugated in-place via the \c conjugate() function:

   \code
   blaze::DynamicMatrix<cplx> C( 5UL, 2UL );

   conjugate( C );  // In-place conjugate operation.
   C = conj( C );   // Same as above
   \endcode

// \n \subsection matrix_operators_real real()
//
// The \c real() function can be used on a dense or sparse matrix to extract the real part of
// each element of the matrix:

   \code
   using blaze::StaticMatrix;

   typedef std::complex<double>  cplx;

   // Creating the matrix
   //    ( (1,0)  (-2,-1) )
   //    ( (1,1)  ( 0, 1) )
   StaticMatrix<cplx,2UL,2UL> A{ { cplx( 1.0, 0.0 ), cplx( -2.0, -1.0 ) },
                                 { cplx( 1.0, 1.0 ), cplx(  0.0,  1.0 ) } };

   // Extracting the real part of each matrix element
   //    ( 1 -2 )
   //    ( 1  0 )
   StaticMatrix<double,2UL,2UL> B;
   B = real( A );
   \endcode

// \n \subsection matrix_operators_imag imag()
//
// The \c imag() function can be used on a dense or sparse matrix to extract the imaginary part
// of each element of the matrix:

   \code
   using blaze::StaticMatrix;

   typedef std::complex<double>  cplx;

   // Creating the matrix
   //    ( (1,0)  (-2,-1) )
   //    ( (1,1)  ( 0, 1) )
   StaticMatrix<cplx,2UL,2UL> A{ { cplx( 1.0, 0.0 ), cplx( -2.0, -1.0 ) },
                                 { cplx( 1.0, 1.0 ), cplx(  0.0,  1.0 ) } };

   // Extracting the imaginary part of each matrix element
   //    ( 0 -1 )
   //    ( 1  1 )
   StaticMatrix<double,2UL,2UL> B;
   B = imag( A );
   \endcode

// \n \subsection matrix_operators_sqrt sqrt() / invsqrt()
//
// Via the \c sqrt() and \c invsqrt() functions the (inverse) square root of each element of a
// matrix can be computed:

   \code
   blaze::StaticMatrix<double,3UL,3UL> A, B, C;

   B = sqrt( A );     // Computes the square root of each element
   C = invsqrt( A );  // Computes the inverse square root of each element
   \endcode

// Note that in case of sparse matrices only the non-zero elements are taken into account!
//
//
// \n \subsection matrix_operators_cbrt cbrt() / invcbrt()
//
// The \c cbrt() and \c invcbrt() functions can be used to compute the the (inverse) cubic root
// of each element of a matrix:

   \code
   blaze::DynamicMatrix<double> A, B, C;

   B = cbrt( A );     // Computes the cubic root of each element
   C = invcbrt( A );  // Computes the inverse cubic root of each element
   \endcode

// Note that in case of sparse matrices only the non-zero elements are taken into account!
//
//
// \n \subsection matrix_operators_pow pow()
//
// The \c pow() function can be used to compute the exponential value of each element of a matrix:

   \code
   blaze::StaticMatrix<double,3UL,3UL> A, B;

   B = pow( A, 1.2 );  // Computes the exponential value of each element
   \endcode

// \n \subsection matrix_operators_exp exp()
//
// \c exp() computes the base e exponential of each element of a matrix:

   \code
   blaze::HybridMatrix<double,3UL,3UL> A, B;

   B = exp( A );  // Computes the base e exponential of each element
   \endcode

// Note that in case of sparse matrices only the non-zero elements are taken into account!
//
//
// \n \subsection matrix_operators_log log() / log10()
//
// The \c log() and \c log10() functions can be used to compute the natural and common logarithm
// of each element of a matrix:

   \code
   blaze::StaticMatrix<double,3UL,3UL> A, B;

   B = log( A );    // Computes the natural logarithm of each element
   B = log10( A );  // Computes the common logarithm of each element
   \endcode

// \n \subsection matrix_operators_trigonometric_functions sin() / cos() / tan() / asin() / acos() / atan()
//
// The following trigonometric functions are available for both dense and sparse matrices:

   \code
   blaze::DynamicMatrix<double> A, B;

   B = sin( A );  // Computes the sine of each element of the matrix
   B = cos( A );  // Computes the cosine of each element of the matrix
   B = tan( A );  // Computes the tangent of each element of the matrix

   B = asin( A );  // Computes the inverse sine of each element of the matrix
   B = acos( A );  // Computes the inverse cosine of each element of the matrix
   B = atan( A );  // Computes the inverse tangent of each element of the matrix
   \endcode

// Note that in case of sparse matrices only the non-zero elements are taken into account!
//
//
// \n \subsection matrix_operators_hyperbolic_functions sinh() / cosh() / tanh() / asinh() / acosh() / atanh()
//
// The following hyperbolic functions are available for both dense and sparse matrices:

   \code
   blaze::DynamicMatrix<double> A, B;

   B = sinh( A );  // Computes the hyperbolic sine of each element of the matrix
   B = cosh( A );  // Computes the hyperbolic cosine of each element of the matrix
   B = tanh( A );  // Computes the hyperbolic tangent of each element of the matrix

   B = asinh( A );  // Computes the inverse hyperbolic sine of each element of the matrix
   B = acosh( A );  // Computes the inverse hyperbolic cosine of each element of the matrix
   B = atanh( A );  // Computes the inverse hyperbolic tangent of each element of the matrix
   \endcode

// \n \subsection matrix_operators_erf erf() / erfc()
//
// The \c erf() and \c erfc() functions compute the (complementary) error function of each
// element of a matrix:

   \code
   blaze::StaticMatrix<double,3UL,3UL> A, B;

   B = erf( A );   // Computes the error function of each element
   B = erfc( A );  // Computes the complementary error function of each element
   \endcode

// Note that in case of sparse matrices only the non-zero elements are taken into account!
//
//
// \n \subsection matrix_operations_foreach forEach()
//
// Via the \c forEach() function it is possible to execute custom operations on dense and sparse
// matrices. For instance, the following example demonstrates a custom square root computation via
// a lambda:

   \code
   blaze::DynamicMatrix<double> A, B;

   B = forEach( A, []( double d ) { return std::sqrt( d ); } );
   \endcode

// Although the computation can be parallelized it is not vectorized and thus cannot perform at
// peak performance. However, it is also possible to create vectorized custom operations. See
// \ref custom_operations for a detailed overview of the possibilities of custom operations.
//
//
// \n \subsection matrix_operations_matrix_transpose trans()
//
// Matrices can be transposed via the \c trans() function. Row-major matrices are transposed into
// a column-major matrix and vice versa:

   \code
   blaze::DynamicMatrix<int,rowMajor> M1( 5UL, 2UL );
   blaze::CompressedMatrix<int,columnMajor> M2( 3UL, 7UL );

   M1 = M2;            // Assigning a column-major matrix to a row-major matrix
   M1 = trans( M2 );   // Assigning the transpose of M2 (i.e. a row-major matrix) to M1
   M1 += trans( M2 );  // Addition assignment of two row-major matrices
   \endcode

// Additionally, matrices can be transposed in-place via the \c transpose() function:

   \code
   blaze::DynamicMatrix<int,rowMajor> M( 5UL, 2UL );

   transpose( M );  // In-place transpose operation.
   M = trans( M );  // Same as above
   \endcode

// Note however that the transpose operation fails if ...
//
//  - ... the given matrix has a fixed size and is non-square;
//  - ... the given matrix is a triangular matrix;
//  - ... the given submatrix affects the restricted parts of a triangular matrix;
//  - ... the given submatrix would cause non-deterministic results in a symmetric/Hermitian matrix.
//
//
// \n \subsection matrix_operations_conjugate_transpose ctrans()
//
// The conjugate transpose of a dense or sparse matrix (also called adjoint matrix, Hermitian
// conjugate, or transjugate) can be computed via the \c ctrans() function:

   \code
   blaze::DynamicMatrix< complex<float>, rowMajor > M1( 5UL, 2UL );
   blaze::CompressedMatrix< complex<float>, columnMajor > M2( 2UL, 5UL );

   M1 = ctrans( M2 );  // Compute the conjugate transpose matrix
   \endcode

// Note that the \c ctrans() function has the same effect as manually applying the \c conj() and
// \c trans() function in any order:

   \code
   M1 = trans( conj( M2 ) );  // Computing the conjugate transpose matrix
   M1 = conj( trans( M2 ) );  // Computing the conjugate transpose matrix
   \endcode

// The \c ctranspose() function can be used to perform an in-place conjugate transpose operation:

   \code
   blaze::DynamicMatrix<int,rowMajor> M( 5UL, 2UL );

   ctranspose( M );  // In-place conjugate transpose operation.
   M = ctrans( M );  // Same as above
   \endcode

// Note however that the conjugate transpose operation fails if ...
//
//  - ... the given matrix has a fixed size and is non-square;
//  - ... the given matrix is a triangular matrix;
//  - ... the given submatrix affects the restricted parts of a triangular matrix;
//  - ... the given submatrix would cause non-deterministic results in a symmetric/Hermitian matrix.
//
//
// \n \subsection matrix_operations_matrix_determinant det()
//
// The determinant of a square dense matrix can be computed by means of the \c det() function:

   \code
   blaze::DynamicMatrix<double,blaze::rowMajor> A;
   // ... Resizing and initialization
   double d = det( A );  // Compute the determinant of A
   \endcode

// In case the given dense matrix is not a square matrix, a \c std::invalid_argument exception is
// thrown.
//
// \note The \c det() function can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type or with a sparse matrix results in a compile time error!
//
// \note The function is depending on LAPACK kernels. Thus the function can only be used if the
// fitting LAPACK library is available and linked to the executable. Otherwise a linker error
// will be created.
//
//
// \n \subsection matrix_operations_swap swap()
//
// Via the \c \c swap() function it is possible to completely swap the contents of two matrices
// of the same type:

   \code
   blaze::DynamicMatrix<int,blaze::rowMajor> M1( 10UL, 15UL );
   blaze::DynamicMatrix<int,blaze::rowMajor> M2( 20UL, 10UL );

   swap( M1, M2 );  // Swapping the contents of M1 and M2
   \endcode

// \n \section matrix_operations_matrix_inversion Matrix Inversion
// <hr>
//
// The inverse of a square dense matrix can be computed via the \c inv() function:

   \code
   blaze::DynamicMatrix<float,blaze::rowMajor> A, B;
   // ... Resizing and initialization
   B = inv( A );  // Compute the inverse of A
   \endcode

// Alternatively, an in-place inversion of a dense matrix can be performed via the \c invert()
// function:

   \code
   blaze::DynamicMatrix<double,blaze::rowMajor> A;
   // ... Resizing and initialization
   invert( A );  // In-place matrix inversion
   \endcode

// Both the \c inv() and the \c invert() functions will automatically select the most suited matrix
// inversion algorithm depending on the size and type of the given matrix. For small matrices of
// up to 6x6, both functions use manually optimized kernels for maximum performance. For matrices
// larger than 6x6 the inversion is performed by means of the most suited matrix decomposition
// method: In case of a general or triangular matrix the LU decomposition is used, for symmetric
// matrices the LDLT decomposition is applied and for Hermitian matrices the LDLH decomposition is
// performed. However, via the \c invert() function it is possible to explicitly specify the matrix
// inversion algorithm:

   \code
   using blaze::byLU;
   using blaze::byLDLT;
   using blaze::byLDLH;
   using blaze::byLLH;

   // In-place inversion with automatic selection of the inversion algorithm
   invert( A );

   // In-place inversion of a general matrix by means of an LU decomposition
   invert<byLU>( A );

   // In-place inversion of a symmetric indefinite matrix by means of a Bunch-Kaufman decomposition
   invert<byLDLT>( A );

   // In-place inversion of a Hermitian indefinite matrix by means of a Bunch-Kaufman decomposition
   invert<byLDLH>( A );

   // In-place inversion of a positive definite matrix by means of a Cholesky decomposition
   invert<byLLH>( A );
   \endcode

// Whereas the inversion by means of an LU decomposition works for every general square matrix,
// the inversion by LDLT only works for symmetric indefinite matrices, the inversion by LDLH is
// restricted to Hermitian indefinite matrices and the Cholesky decomposition (LLH) only works
// for Hermitian positive definite matrices. Please note that it is in the responsibility of the
// function caller to guarantee that the selected algorithm is suited for the given matrix. In
// case this precondition is violated the result can be wrong and might not represent the inverse
// of the given matrix!
//
// For both the \c inv() and \c invert() function the matrix inversion fails if ...
//
//  - ... the given matrix is not a square matrix;
//  - ... the given matrix is singular and not invertible.
//
// In all failure cases either a compilation error is created if the failure can be predicted at
// compile time or a \c std::invalid_argument exception is thrown.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type or with a sparse matrix results in a compile time error!
//
// \note The functions invert the dense matrix by means of LAPACK kernels. Thus the functions can
// only be used if the fitting LAPACK library is available and linked to the executable. Otherwise
// a linker error will be created.
//
// \note It is not possible to use any kind of view on the expression object returned by the
// \c inv() function. Also, it is not possible to access individual elements via the function call
// operator on the expression object:

   \code
   row( inv( A ), 2UL );  // Compilation error: Views cannot be used on an inv() expression!
   inv( A )(1,2);         // Compilation error: It is not possible to access individual elements!
   \endcode

// \note The inversion functions do not provide any exception safety guarantee, i.e. in case an
// exception is thrown the matrix may already have been modified.
//
//
// \n \section matrix_operations_decomposition Matrix Decomposition
// <hr>
//
// \note All decomposition functions can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type or with a sparse matrix results in a compile time error!
//
// \note The functions decompose a dense matrix by means of LAPACK kernels. Thus the functions can
// only be used if the fitting LAPACK library is available and linked to the executable. Otherwise
// a linker error will be created.
//
// \subsection matrix_operations_decomposition_lu LU Decomposition
//
// The LU decomposition of a dense matrix can be computed via the \c lu() function:

   \code
   blaze::DynamicMatrix<double,blaze::rowMajor> A;
   // ... Resizing and initialization

   blaze::DynamicMatrix<double,blaze::rowMajor> L, U, P;

   lu( A, L, U, P );  // LU decomposition of a row-major matrix

   assert( A == L * U * P );
   \endcode

   \code
   blaze::DynamicMatrix<double,blaze::columnMajor> A;
   // ... Resizing and initialization

   blaze::DynamicMatrix<double,blaze::columnMajor> L, U, P;

   lu( A, L, U, P );  // LU decomposition of a column-major matrix

   assert( A == P * L * U );
   \endcode

// The function works for both \c rowMajor and \c columnMajor matrices. Note, however, that the
// three matrices \c A, \c L and \c U are required to have the same storage order. Also, please
// note that the way the permutation matrix \c P needs to be applied differs between row-major and
// column-major matrices, since the algorithm uses column interchanges for row-major matrices and
// row interchanges for column-major matrices.
//
// Furthermore, \c lu() can be used with adaptors. For instance, the following example demonstrates
// the LU decomposition of a symmetric matrix into a lower and upper triangular matrix:

   \code
   blaze::SymmetricMatrix< blaze::DynamicMatrix<double,blaze::columnMajor> > A;
   // ... Resizing and initialization

   blaze::LowerMatrix< blaze::DynamicMatrix<double,blaze::columnMajor> > L;
   blaze::UpperMatrix< blaze::DynamicMatrix<double,blaze::columnMajor> > U;
   blaze::DynamicMatrix<double,blaze::columnMajor> P;

   lu( A, L, U, P );  // LU decomposition of A
   \endcode

// \n \subsection matrix_operations_decomposition_llh Cholesky Decomposition
//
// The Cholesky (LLH) decomposition of a dense matrix can be computed via the \c llh() function:

   \code
   blaze::DynamicMatrix<double,blaze::rowMajor> A;
   // ... Resizing and initialization

   blaze::DynamicMatrix<double,blaze::rowMajor> L;

   llh( A, L );  // LLH decomposition of a row-major matrix

   assert( A == L * ctrans( L ) );
   \endcode

// The function works for both \c rowMajor and \c columnMajor matrices and the two matrices \c A
// and \c L can have any storage order.
//
// Furthermore, \c llh() can be used with adaptors. For instance, the following example demonstrates
// the LLH decomposition of a symmetric matrix into a lower triangular matrix:

   \code
   blaze::SymmetricMatrix< blaze::DynamicMatrix<double,blaze::columnMajor> > A;
   // ... Resizing and initialization

   blaze::LowerMatrix< blaze::DynamicMatrix<double,blaze::columnMajor> > L;

   llh( A, L );  // Cholesky decomposition of A
   \endcode

// \n \subsection matrix_operations_decomposition_qr QR Decomposition
//
// The QR decomposition of a dense matrix can be computed via the \c qr() function:

   \code
   blaze::DynamicMatrix<double,blaze::rowMajor> A;
   // ... Resizing and initialization

   blaze::DynamicMatrix<double,blaze::columnMajor> Q;
   blaze::DynamicMatrix<double,blaze::rowMajor> R;

   qr( A, Q, R );  // QR decomposition of a row-major matrix

   assert( A == Q * R );
   \endcode

// The function works for both \c rowMajor and \c columnMajor matrices and the three matrices
// \c A, \c Q and \c R can have any storage order.
//
// Furthermore, \c qr() can be used with adaptors. For instance, the following example demonstrates
// the QR decomposition of a symmetric matrix into a general matrix and an upper triangular matrix:

   \code
   blaze::SymmetricMatrix< blaze::DynamicMatrix<double,blaze::columnMajor> > A;
   // ... Resizing and initialization

   blaze::DynamicMatrix<double,blaze::rowMajor> Q;
   blaze::UpperMatrix< blaze::DynamicMatrix<double,blaze::columnMajor> > R;

   qr( A, Q, R );  // QR decomposition of A
   \endcode

// \n \subsection matrix_operations_decomposition_rq RQ Decomposition
//
// Similar to the QR decomposition, the RQ decomposition of a dense matrix can be computed via
// the \c rq() function:

   \code
   blaze::DynamicMatrix<double,blaze::rowMajor> A;
   // ... Resizing and initialization

   blaze::DynamicMatrix<double,blaze::rowMajor> R;
   blaze::DynamicMatrix<double,blaze::columnMajor> Q;

   rq( A, R, Q );  // RQ decomposition of a row-major matrix

   assert( A == R * Q );
   \endcode

// The function works for both \c rowMajor and \c columnMajor matrices and the three matrices
// \c A, \c R and \c Q can have any storage order.
//
// Also the \c rq() function can be used in combination with matrix adaptors. For instance, the
// following example demonstrates the RQ decomposition of an Hermitian matrix into a general
// matrix and an upper triangular matrix:

   \code
   blaze::HermitianMatrix< blaze::DynamicMatrix<complex<double>,blaze::columnMajor> > A;
   // ... Resizing and initialization

   blaze::UpperMatrix< blaze::DynamicMatrix<complex<double>,blaze::columnMajor> > R;
   blaze::DynamicMatrix<complex<double>,blaze::rowMajor> Q;

   rq( A, R, Q );  // RQ decomposition of A
   \endcode

// \n \subsection matrix_operations_decomposition_ql QL Decomposition
//
// The QL decomposition of a dense matrix can be computed via the \c ql() function:

   \code
   blaze::DynamicMatrix<double,blaze::rowMajor> A;
   // ... Resizing and initialization

   blaze::DynamicMatrix<double,blaze::rowMajor> Q;
   blaze::DynamicMatrix<double,blaze::columnMajor> L;

   ql( A, Q, L );  // QL decomposition of a row-major matrix

   assert( A == Q * L );
   \endcode

// The function works for both \c rowMajor and \c columnMajor matrices and the three matrices
// \c A, \c Q and \c L can have any storage order.
//
// Also the \c ql() function can be used in combination with matrix adaptors. For instance, the
// following example demonstrates the QL decomposition of a symmetric matrix into a general
// matrix and a lower triangular matrix:

   \code
   blaze::SymmetricMatrix< blaze::DynamicMatrix<double,blaze::columnMajor> > A;
   // ... Resizing and initialization

   blaze::DynamicMatrix<double,blaze::rowMajor> Q;
   blaze::LowerMatrix< blaze::DynamicMatrix<double,blaze::columnMajor> > L;

   ql( A, Q, L );  // QL decomposition of A
   \endcode

// \n \subsection matrix_operations_decomposition_lq LQ Decomposition
//
// The LQ decomposition of a dense matrix can be computed via the \c lq() function:

   \code
   blaze::DynamicMatrix<double,blaze::rowMajor> A;
   // ... Resizing and initialization

   blaze::DynamicMatrix<double,blaze::rowMajor> L;
   blaze::DynamicMatrix<double,blaze::columnMajor> Q;

   lq( A, L, Q );  // LQ decomposition of a row-major matrix

   assert( A == L * Q );
   \endcode

// The function works for both \c rowMajor and \c columnMajor matrices and the three matrices
// \c A, \c L and \c Q can have any storage order.
//
// Furthermore, \c lq() can be used with adaptors. For instance, the following example demonstrates
// the LQ decomposition of an Hermitian matrix into a lower triangular matrix and a general matrix:

   \code
   blaze::HermitianMatrix< blaze::DynamicMatrix<complex<double>,blaze::columnMajor> > A;
   // ... Resizing and initialization

   blaze::LowerMatrix< blaze::DynamicMatrix<complex<double>,blaze::columnMajor> > L;
   blaze::DynamicMatrix<complex<double>,blaze::rowMajor> Q;

   lq( A, L, Q );  // LQ decomposition of A
   \endcode

// \n Previous: \ref matrix_types &nbsp; &nbsp; Next: \ref adaptors
*/
//*************************************************************************************************


//**Adaptors***************************************************************************************
/*!\page adaptors Adaptors
//
// \tableofcontents
//
//
// \section adaptors_general General Concepts
// <hr>
//
// Adaptors act as wrappers around the general \ref matrix_types. They adapt the interface of the
// matrices such that certain invariants are preserved. Due to this adaptors can provide a compile
// time guarantee of certain properties, which can be exploited for optimized performance.
//
// The \b Blaze library provides a total of 9 different adaptors:
//
// <ul>
//    <li> \ref adaptors_symmetric_matrices </li>
//    <li> \ref adaptors_hermitian_matrices </li>
//    <li> \ref adaptors_triangular_matrices
//       <ul>
//          <li> \ref adaptors_triangular_matrices "Lower Triangular Matrices"
//             <ul>
//                <li> \ref adaptors_triangular_matrices_lowermatrix </li>
//                <li> \ref adaptors_triangular_matrices_unilowermatrix </li>
//                <li> \ref adaptors_triangular_matrices_strictlylowermatrix </li>
//             </ul>
//          </li>
//          <li> \ref adaptors_triangular_matrices "Upper Triangular Matrices"
//             <ul>
//                <li> \ref adaptors_triangular_matrices_uppermatrix </li>
//                <li> \ref adaptors_triangular_matrices_uniuppermatrix </li>
//                <li> \ref adaptors_triangular_matrices_strictlyuppermatrix </li>
//             </ul>
//          </li>
//          <li> \ref adaptors_triangular_matrices "Diagonal Matrices"
//             <ul>
//                <li> \ref adaptors_triangular_matrices_diagonalmatrix </li>
//             </ul>
//          </li>
//       </ul>
//    </li>
// </ul>
//
// In combination with the general matrix types, \b Blaze provides a total of 40 different matrix
// types that make it possible to exactly adapt the type of matrix to every specific problem.
//
//
// \n \section adaptors_examples Examples
// <hr>
//
// The following code examples give an impression on the use of adaptors. The first example shows
// the multiplication between two lower matrices:

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

// When multiplying two matrices, at least one of which is triangular, \b Blaze can exploit the
// fact that either the lower or upper part of the matrix contains only default elements and
// restrict the algorithm to the non-zero elements. Thus the adaptor provides a significant
// performance advantage in comparison to a general matrix multiplication, especially for large
// matrices.
//
// The second example shows the \c SymmetricMatrix adaptor in a row-major dense matrix/sparse
// vector multiplication:

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::CompressedVector;
   using blaze::rowMajor;
   using blaze::columnVector;

   SymmetricMatrix< DynamicMatrix<double,rowMajor> > A;
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

// which significantly increases the performance.
//
// \n Previous: \ref matrix_operations &nbsp; &nbsp; Next: \ref adaptors_symmetric_matrices
*/
//*************************************************************************************************


//**Symmetric Matrices*****************************************************************************
/*!\page adaptors_symmetric_matrices Symmetric Matrices
//
// \tableofcontents
//
//
// \n \section adaptors_symmetric_matrices_general Symmetric Matrices
// <hr>
//
// In contrast to general matrices, which have no restriction in their number of rows and columns
// and whose elements can have any value, symmetric matrices provide the compile time guarantee
// to be square matrices with pair-wise identical values. Mathematically, this means that a
// symmetric matrix is always equal to its transpose (\f$ A = A^T \f$) and that all non-diagonal
// values have an identical counterpart (\f$ a_{ij} == a_{ji} \f$). This symmetry property can
// be exploited to provide higher efficiency and/or lower memory consumption. Within the \b Blaze
// library, symmetric matrices are realized by the \ref adaptors_symmetric_matrices_symmetricmatrix
// class template.
//
//
// \n \section adaptors_symmetric_matrices_symmetricmatrix SymmetricMatrix
// <hr>
//
// The SymmetricMatrix class template is an adapter for existing dense and sparse matrix types.
// It inherits the properties and the interface of the given matrix type \c MT and extends it
// by enforcing the additional invariant of symmetry (i.e. the matrix is always equal to its
// transpose \f$ A = A^T \f$). It can be included via the header file

   \code
   #include <blaze/math/SymmetricMatrix.h>
   \endcode

// The type of the adapted matrix can be specified via template parameter:

   \code
   template< typename MT >
   class SymmetricMatrix;
   \endcode

// \c MT specifies the type of the matrix to be adapted. SymmetricMatrix can be used with any
// non-cv-qualified, non-reference, non-pointer, non-expression dense or sparse matrix type. Note
// that the given matrix type must be either resizable (as for instance blaze::HybridMatrix or
// blaze::DynamicMatrix) or must be square at compile time (as for instance blaze::StaticMatrix).
//
// The following examples give an impression of several possible symmetric matrices:

   \code
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;
   using blaze::columnMajor;

   // Definition of a 3x3 row-major dense symmetric matrix with static memory
   blaze::SymmetricMatrix< blaze::StaticMatrix<int,3UL,3UL,rowMajor> > A;

   // Definition of a resizable column-major dense symmetric matrix based on HybridMatrix
   blaze::SymmetricMatrix< blaze::HybridMatrix<float,4UL,4UL,columnMajor> B;

   // Definition of a resizable row-major dense symmetric matrix based on DynamicMatrix
   blaze::SymmetricMatrix< blaze::DynamicMatrix<double,rowMajor> > C;

   // Definition of a fixed size row-major dense symmetric matrix based on CustomMatrix
   blaze::SymmetricMatrix< blaze::CustomMatrix<double,unaligned,unpadded,rowMajor> > D;

   // Definition of a compressed row-major single precision symmetric matrix
   blaze::SymmetricMatrix< blaze::CompressedMatrix<float,blaze::rowMajor> > E;
   \endcode

// The storage order of a symmetric matrix is depending on the storage order of the adapted matrix
// type \c MT. In case the adapted matrix is stored in a row-wise fashion (i.e. is specified as
// blaze::rowMajor), the symmetric matrix will also be a row-major matrix. Otherwise, if the
// adapted matrix is column-major (i.e. is specified as blaze::columnMajor), the symmetric matrix
// will also be a column-major matrix.
//
//
// \n \section adaptors_symmetric_matrices_special_properties Special Properties of Symmetric Matrices
// <hr>
//
// A symmetric matrix is used exactly like a matrix of the underlying, adapted matrix type \c MT.
// It also provides (nearly) the same interface as the underlying matrix type. However, there are
// some important exceptions resulting from the symmetry constraint:
//
//  -# <b>\ref adaptors_symmetric_matrices_square</b>
//  -# <b>\ref adaptors_symmetric_matrices_symmetry</b>
//  -# <b>\ref adaptors_symmetric_matrices_initialization</b>
//
// \n \subsection adaptors_symmetric_matrices_square Symmetric Matrices Must Always be Square!
//
// In case a resizable matrix is used (as for instance blaze::HybridMatrix, blaze::DynamicMatrix,
// or blaze::CompressedMatrix), this means that the according constructors, the \c resize() and
// the \c extend() functions only expect a single parameter, which specifies both the number of
// rows and columns, instead of two (one for the number of rows and one for the number of columns):

   \code
   using blaze::DynamicMatrix;
   using blaze::SymmetricMatrix;
   using blaze::rowMajor;

   // Default constructed, default initialized, row-major 3x3 symmetric dynamic matrix
   SymmetricMatrix< DynamicMatrix<double,rowMajor> > A( 3 );

   // Resizing the matrix to 5x5
   A.resize( 5 );

   // Extending the number of rows and columns by 2, resulting in a 7x7 matrix
   A.extend( 2 );
   \endcode

// In case a matrix with a fixed size is used (as for instance blaze::StaticMatrix), the number
// of rows and number of columns must be specified equally:

   \code
   using blaze::StaticMatrix;
   using blaze::SymmetricMatrix;
   using blaze::columnMajor;

   // Correct setup of a fixed size column-major 3x3 symmetric static matrix
   SymmetricMatrix< StaticMatrix<int,3UL,3UL,columnMajor> > A;

   // Compilation error: the provided matrix type is not a square matrix type
   SymmetricMatrix< StaticMatrix<int,3UL,4UL,columnMajor> > B;
   \endcode

// \n \subsection adaptors_symmetric_matrices_symmetry The Symmetric Property is Always Enforced!
//
// This means that modifying the element \f$ a_{ij} \f$ of a symmetric matrix also modifies its
// counterpart element \f$ a_{ji} \f$. Also, it is only possible to assign matrices that are
// symmetric themselves:

   \code
   using blaze::CompressedMatrix;
   using blaze::DynamicMatrix;
   using blaze::StaticMatrix;
   using blaze::SymmetricMatrix;
   using blaze::rowMajor;

   // Default constructed, row-major 3x3 symmetric compressed matrix
   SymmetricMatrix< CompressedMatrix<double,rowMajor> > A( 3 );

   // Initializing three elements via the function call operator
   A(0,0) = 1.0;  // Initialization of the diagonal element (0,0)
   A(0,2) = 2.0;  // Initialization of the elements (0,2) and (2,0)

   // Inserting three more elements via the insert() function
   A.insert( 1, 1, 3.0 );  // Inserting the diagonal element (1,1)
   A.insert( 1, 2, 4.0 );  // Inserting the elements (1,2) and (2,1)

   // Access via a non-const iterator
   *A.begin(1UL) = 10.0;  // Modifies both elements (1,0) and (0,1)

   // Erasing elements via the erase() function
   A.erase( 0, 0 );  // Erasing the diagonal element (0,0)
   A.erase( 0, 2 );  // Erasing the elements (0,2) and (2,0)

   // Construction from a symmetric dense matrix
   StaticMatrix<double,3UL,3UL> B{ {  3.0,  8.0, -2.0 },
                                   {  8.0,  0.0, -1.0 },
                                   { -2.0, -1.0,  4.0 } };

   SymmetricMatrix< DynamicMatrix<double,rowMajor> > C( B );  // OK

   // Assignment of a non-symmetric dense matrix
   StaticMatrix<double,3UL,3UL> D{ {  3.0,  7.0, -2.0 },
                                   {  8.0,  0.0, -1.0 },
                                   { -2.0, -1.0,  4.0 } };

   C = D;  // Throws an exception; symmetric invariant would be violated!
   \endcode

// The same restriction also applies to the \c append() function for sparse matrices: Appending
// the element \f$ a_{ij} \f$ additionally inserts the element \f$ a_{ji} \f$ into the matrix.
// Despite the additional insertion, the \c append() function still provides the most efficient
// way to set up a symmetric sparse matrix. In order to achieve the maximum efficiency, the
// capacity of the individual rows/columns of the matrix should to be specifically prepared with
// \c reserve() calls:

   \code
   using blaze::CompressedMatrix;
   using blaze::SymmetricMatrix;
   using blaze::rowMajor;

   // Setup of the symmetric matrix
   //
   //       ( 0 1 3 )
   //   A = ( 1 2 0 )
   //       ( 3 0 0 )
   //
   SymmetricMatrix< CompressedMatrix<double,rowMajor> > A( 3 );

   A.reserve( 5 );         // Reserving enough space for 5 non-zero elements
   A.reserve( 0, 2 );      // Reserving two non-zero elements in the first row
   A.reserve( 1, 2 );      // Reserving two non-zero elements in the second row
   A.reserve( 2, 1 );      // Reserving a single non-zero element in the third row
   A.append( 0, 1, 1.0 );  // Appending the value 1 at position (0,1) and (1,0)
   A.append( 1, 1, 2.0 );  // Appending the value 2 at position (1,1)
   A.append( 2, 0, 3.0 );  // Appending the value 3 at position (2,0) and (0,2)
   \endcode

// The symmetry property is also enforced for symmetric custom matrices: In case the given array
// of elements does not represent a symmetric matrix, a \c std::invalid_argument exception is
// thrown:

   \code
   using blaze::CustomMatrix;
   using blaze::SymmetricMatrix;
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;

   typedef SymmetricMatrix< CustomMatrix<double,unaligned,unpadded,rowMajor> >  CustomSymmetric;

   // Creating a 3x3 symmetric custom matrix from a properly initialized array
   double array[9] = { 1.0, 2.0, 4.0,
                       2.0, 3.0, 5.0,
                       4.0, 5.0, 6.0 };
   CustomSymmetric A( array, 3UL );  // OK

   // Attempt to create a second 3x3 symmetric custom matrix from an uninitialized array
   CustomSymmetric B( new double[9UL], 3UL, blaze::ArrayDelete() );  // Throws an exception
   \endcode

// Finally, the symmetry property is enforced for views (rows, columns, submatrices, ...) on the
// symmetric matrix. The following example demonstrates that modifying the elements of an entire
// row of the symmetric matrix also affects the counterpart elements in the according column of
// the matrix:

   \code
   using blaze::DynamicMatrix;
   using blaze::SymmetricMatrix;

   // Setup of the symmetric matrix
   //
   //       ( 0 1 0 2 )
   //   A = ( 1 3 4 0 )
   //       ( 0 4 0 5 )
   //       ( 2 0 5 0 )
   //
   SymmetricMatrix< DynamicMatrix<int> > A( 4 );
   A(0,1) = 1;
   A(0,3) = 2;
   A(1,1) = 3;
   A(1,2) = 4;
   A(2,3) = 5;

   // Setting all elements in the 1st row to 0 results in the matrix
   //
   //       ( 0 0 0 2 )
   //   A = ( 0 0 0 0 )
   //       ( 0 0 0 5 )
   //       ( 2 0 5 0 )
   //
   row( A, 1 ) = 0;
   \endcode

// The next example demonstrates the (compound) assignment to submatrices of symmetric matrices.
// Since the modification of element \f$ a_{ij} \f$ of a symmetric matrix also modifies the
// element \f$ a_{ji} \f$, the matrix to be assigned must be structured such that the symmetry
// of the symmetric matrix is preserved. Otherwise a \c std::invalid_argument exception is
// thrown:

   \code
   using blaze::DynamicMatrix;
   using blaze::SymmetricMatrix;

   // Setup of two default 4x4 symmetric matrices
   SymmetricMatrix< DynamicMatrix<int> > A1( 4 ), A2( 4 );

   // Setup of the 3x2 dynamic matrix
   //
   //       ( 1 2 )
   //   B = ( 3 4 )
   //       ( 5 6 )
   //
   DynamicMatrix<int> B{ { 1, 2 }, { 3, 4 }, { 5, 6 } };

   // OK: Assigning B to a submatrix of A1 such that the symmetry can be preserved
   //
   //        ( 0 0 1 2 )
   //   A1 = ( 0 0 3 4 )
   //        ( 1 3 5 6 )
   //        ( 2 4 6 0 )
   //
   submatrix( A1, 0UL, 2UL, 3UL, 2UL ) = B;  // OK

   // Error: Assigning B to a submatrix of A2 such that the symmetry cannot be preserved!
   //   The elements marked with X cannot be assigned unambiguously!
   //
   //        ( 0 1 2 0 )
   //   A2 = ( 1 3 X 0 )
   //        ( 2 X 6 0 )
   //        ( 0 0 0 0 )
   //
   submatrix( A2, 0UL, 1UL, 3UL, 2UL ) = B;  // Assignment throws an exception!
   \endcode

// \n \subsection adaptors_symmetric_matrices_initialization The Elements of a Dense Symmetric Matrix are Always Default Initialized!
//
// Although this results in a small loss of efficiency (especially in case all default values are
// overridden afterwards), this property is important since otherwise the symmetric property of
// dense symmetric matrices could not be guaranteed:

   \code
   using blaze::DynamicMatrix;
   using blaze::SymmetricMatrix;

   // Uninitialized, 5x5 row-major dynamic matrix
   DynamicMatrix<int,rowMajor> A( 5, 5 );

   // Default initialized, 5x5 row-major symmetric dynamic matrix
   SymmetricMatrix< DynamicMatrix<int,rowMajor> > B( 5 );
   \endcode

// \n \section adaptors_symmetric_matrices_arithmetic_operations Arithmetic Operations
// <hr>
//
// A SymmetricMatrix matrix can participate in numerical operations in any way any other dense
// or sparse matrix can participate. It can also be combined with any other dense or sparse vector
// or matrix. The following code example gives an impression of the use of SymmetricMatrix within
// arithmetic operations:

   \code
   using blaze::SymmetricMatrix;
   using blaze::DynamicMatrix;
   using blaze::StaticMatrix;
   using blaze::CompressedMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   CompressedMatrix<float> E( 3, 3 );  // Empty row-major sparse single precision 3x3 matrix

   SymmetricMatrix< HybridMatrix<float,3UL,3UL,rowMajor> > F;
   SymmetricMatrix< StaticMatrix<float,3UL,3UL,columnMajor> > G;

   F = A + B;     // Matrix addition and assignment to a row-major symmetric matrix
   G = A - C;     // Matrix subtraction and assignment to a column-major symmetric matrix
   G = A * E;     // Matrix multiplication between a dense and a sparse matrix

   A *= 2.0;      // In-place scaling of matrix A
   F  = 2.0 * B;  // Scaling of matrix B
   G  = E * 2.0;  // Scaling of matrix E

   F += A - B;    // Addition assignment
   G -= A + C;    // Subtraction assignment
   G *= A * E;    // Multiplication assignment
   \endcode

// \n \section adaptors_symmetric_matrices_block_structured Block-Structured Symmetric Matrices
// <hr>
//
// It is also possible to use block-structured symmetric matrices:

   \code
   using blaze::CompressedMatrix;
   using blaze::StaticMatrix;
   using blaze::SymmetricMatrix;

   // Definition of a 3x3 block-structured symmetric matrix based on CompressedMatrix
   SymmetricMatrix< CompressedMatrix< StaticMatrix<int,3UL,3UL> > > A( 3 );
   \endcode

// Also in this case, the SymmetricMatrix class template enforces the invariant of symmetry and
// guarantees that a modifications of element \f$ a_{ij} \f$ of the adapted matrix is also
// applied to element \f$ a_{ji} \f$:

   \code
   // Inserting the elements (2,4) and (4,2)
   A.insert( 2, 4, StaticMatrix<int,3UL,3UL>{ { 1, -4,  5 },
                                              { 6,  8, -3 },
                                              { 2, -1,  2 } } );

   // Manipulating the elements (2,4) and (4,2)
   A(2,4)(1,1) = -5;
   \endcode

// \n \section adaptors_symmetric_matrices_performance Performance Considerations
// <hr>
//
// When the symmetric property of a matrix is known beforehands using the SymmetricMatrix adaptor
// instead of a general matrix can be a considerable performance advantage. The \b Blaze library
// tries to exploit the properties of symmetric matrices whenever possible. However, there are
// also situations when using a symmetric matrix introduces some overhead. The following examples
// demonstrate several situations where symmetric matrices can positively or negatively impact
// performance.
//
// \n \subsection adaptors_symmetric_matrices_matrix_matrix_multiplication Positive Impact: Matrix/Matrix Multiplication
//
// When multiplying two matrices, at least one of which is symmetric, \b Blaze can exploit the fact
// that \f$ A = A^T \f$ and choose the fastest and most suited combination of storage orders for the
// multiplication. The following example demonstrates this by means of a dense matrix/sparse matrix
// multiplication:

   \code
   using blaze::DynamicMatrix;
   using blaze::SymmetricMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   SymmetricMatrix< DynamicMatrix<double,rowMajor> > A;
   SymmetricMatrix< CompressedMatrix<double,columnMajor> > B;
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
// optimized form can be vectorized. Therefore, in the context of matrix multiplications, using the
// SymmetricMatrix adapter is obviously an advantage.
//
// \n \subsection adaptors_symmetric_matrices_matrix_vector_multiplication Positive Impact: Matrix/Vector Multiplication
//
// A similar optimization is possible in case of matrix/vector multiplications:

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
   using blaze::CompressedVector;
   using blaze::rowMajor;
   using blaze::columnVector;

   SymmetricMatrix< DynamicMatrix<double,rowMajor> > A;
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
// \n \subsection adaptors_symmetric_matrices_views Positive Impact: Row/Column Views on Column/Row-Major Matrices
//
// Another example is the optimization of a row view on a column-major symmetric matrix:

   \code
   using blaze::DynamicMatrix;
   using blaze::SymmetricMatrix;
   using blaze::Row;
   using blaze::rowMajor;
   using blaze::columnMajor;

   typedef SymmetricMatrix< DynamicMatrix<double,columnMajor> >  DynamicSymmetric;

   DynamicSymmetric A( 10UL );
   Row<DynamicSymmetric> row5 = row( A, 5UL );
   \endcode

// Usually, a row view on a column-major matrix results in a considerable performance decrease in
// comparison to a row view on a row-major matrix due to the non-contiguous storage of the matrix
// elements. However, in case of symmetric matrices, \b Blaze instead uses the according column of
// the matrix, which provides the same performance as if the matrix would be row-major. Note that
// this also works for column views on row-major matrices, where \b Blaze can use the according
// row instead of a column in order to provide maximum performance.
//
// \n \subsection adaptors_symmetric_matrices_assignment Negative Impact: Assignment of a General Matrix
//
// In contrast to using a symmetric matrix on the right-hand side of an assignment (i.e. for read
// access), which introduces absolutely no performance penalty, using a symmetric matrix on the
// left-hand side of an assignment (i.e. for write access) may introduce additional overhead when
// it is assigned a general matrix, which is not symmetric at compile time:

   \code
   using blaze::DynamicMatrix;
   using blaze::SymmetricMatrix;

   SymmetricMatrix< DynamicMatrix<double> > A, C;
   DynamicMatrix<double> B;

   B = A;  // Only read-access to the symmetric matrix; no performance penalty
   C = A;  // Assignment of a symmetric matrix to another symmetric matrix; no runtime overhead
   C = B;  // Assignment of a general matrix to a symmetric matrix; some runtime overhead
   \endcode

// When assigning a general, potentially not symmetric matrix to a symmetric matrix it is necessary
// to check whether the matrix is symmetric at runtime in order to guarantee the symmetry property
// of the symmetric matrix. In case it turns out to be symmetric, it is assigned as efficiently as
// possible, if it is not, an exception is thrown. In order to prevent this runtime overhead it is
// therefore generally advisable to assign symmetric matrices to other symmetric matrices.\n
// In this context it is especially noteworthy that in contrast to additions and subtractions the
// multiplication of two symmetric matrices does not necessarily result in another symmetric matrix:

   \code
   SymmetricMatrix< DynamicMatrix<double> > A, B, C;

   C = A + B;  // Results in a symmetric matrix; no runtime overhead
   C = A - B;  // Results in a symmetric matrix; no runtime overhead
   C = A * B;  // Is not guaranteed to result in a symmetric matrix; some runtime overhead
   \endcode

// \n Previous: \ref adaptors &nbsp; &nbsp; Next: \ref adaptors_hermitian_matrices
*/
//*************************************************************************************************


//**Hermitian Matrices*****************************************************************************
/*!\page adaptors_hermitian_matrices Hermitian Matrices
//
// \tableofcontents
//
//
// \n \section adaptors_hermitian_matrices_general Hermitian Matrices
// <hr>
//
// In addition to symmetric matrices, \b Blaze also provides an adaptor for Hermitian matrices.
// Hermitian matrices provide the compile time guarantee to be square matrices with pair-wise
// conjugate complex values. Mathematically, this means that an Hermitian matrix is always equal
// to its conjugate transpose (\f$ A = \overline{A^T} \f$) and that all non-diagonal values have
// a complex conjugate counterpart (\f$ a_{ij} == \overline{a_{ji}} \f$). Within the \b Blaze
// library, Hermitian matrices are realized by the \ref adaptors_hermitian_matrices_hermitianmatrix
// class template.
//
//
// \n \section adaptors_hermitian_matrices_hermitianmatrix HermitianMatrix
// <hr>
//
// The HermitianMatrix class template is an adapter for existing dense and sparse matrix types.
// It inherits the properties and the interface of the given matrix type \c MT and extends it by
// enforcing the additional invariant of Hermitian symmetry (i.e. the matrix is always equal to
// its conjugate transpose \f$ A = \overline{A^T} \f$). It can be included via the header file

   \code
   #include <blaze/math/HermitianMatrix.h>
   \endcode

// The type of the adapted matrix can be specified via template parameter:

   \code
   template< typename MT >
   class HermitianMatrix;
   \endcode

// \c MT specifies the type of the matrix to be adapted. HermitianMatrix can be used with any
// non-cv-qualified, non-reference, non-pointer, non-expression dense or sparse matrix type. Also,
// the given matrix type must have numeric element types (i.e. all integral types except \c bool,
// floating point and complex types). Note that the given matrix type must be either resizable (as
// for instance blaze::HybridMatrix or blaze::DynamicMatrix) or must be square at compile time (as
// for instance blaze::StaticMatrix).
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

   // Definition of a fixed size row-major dense Hermitian matrix based on CustomMatrix
   blaze::HermitianMatrix< blaze::CustomMatrix<double,unaligned,unpadded,rowMajor> > D;

   // Definition of a compressed row-major single precision complex Hermitian matrix
   blaze::HermitianMatrix< blaze::CompressedMatrix<std::complex<float>,rowMajor> > E;
   \endcode

// The storage order of a Hermitian matrix is depending on the storage order of the adapted matrix
// type \c MT. In case the adapted matrix is stored in a row-wise fashion (i.e. is specified as
// blaze::rowMajor), the Hermitian matrix will also be a row-major matrix. Otherwise, if the
// adapted matrix is column-major (i.e. is specified as blaze::columnMajor), the Hermitian matrix
// will also be a column-major matrix.
//
//
// \n \section adaptors_hermitian_matrices_vs_symmetric_matrices Hermitian Matrices vs. Symmetric Matrices
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
// types (i.e. all integral types except \c bool, floating point and complex types), whereas
// symmetric matrices can also be block structured (i.e. can have vector or matrix elements).
// For built-in element types, the HermitianMatrix adaptor behaves exactly like the according
// SymmetricMatrix implementation. For complex element types, however, the Hermitian property
// is enforced (see also \ref adaptors_hermitian_matrices_hermitian).

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

// \n \section adaptors_hermitian_matrices_special_properties Special Properties of Hermitian Matrices
// <hr>
//
// A Hermitian matrix is used exactly like a matrix of the underlying, adapted matrix type \c MT.
// It also provides (nearly) the same interface as the underlying matrix type. However, there are
// some important exceptions resulting from the Hermitian symmetry constraint:
//
//  -# <b>\ref adaptors_hermitian_matrices_square</b>
//  -# <b>\ref adaptors_hermitian_matrices_hermitian</b>
//  -# <b>\ref adaptors_hermitian_matrices_initialization</b>
//
// \n \subsection adaptors_hermitian_matrices_square Hermitian Matrices Must Always be Square!
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

// \n \subsection adaptors_hermitian_matrices_hermitian The Hermitian Property is Always Enforced!
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
   StaticMatrix<cplx,3UL,3UL> B{ { cplx(  3.0,  0.0 ), cplx(  8.0, 2.0 ), cplx( -2.0,  2.0 ) },
                                 { cplx(  8.0,  1.0 ), cplx(  0.0, 0.0 ), cplx( -1.0, -1.0 ) },
                                 { cplx( -2.0, -2.0 ), cplx( -1.0, 1.0 ), cplx(  4.0,  0.0 ) } };

   HermitianMatrix< DynamicMatrix<double,rowMajor> > C( B );  // OK

   // Assignment of a non-Hermitian dense matrix
	StaticMatrix<cplx,3UL,3UL> D{ { cplx(  3.0, 0.0 ), cplx(  7.0, 2.0 ), cplx( 3.0, 2.0 ) },
                                 { cplx(  8.0, 1.0 ), cplx(  0.0, 0.0 ), cplx( 6.0, 4.0 ) },
                                 { cplx( -2.0, 2.0 ), cplx( -1.0, 1.0 ), cplx( 4.0, 0.0 ) } };

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
// of elements does not represent a Hermitian matrix, a \c std::invalid_argument exception is
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
// symmetry of the matrix is preserved. Otherwise a \c std::invalid_argument exception is thrown:

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

// \n \subsection adaptors_hermitian_matrices_initialization The Elements of a Dense Hermitian Matrix are Always Default Initialized!
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

// \n \section adaptors_hermitian_matrices_arithmetic_operations Arithmetic Operations
// <hr>
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

// \n \section adaptors_hermitian_matrices_performance Performance Considerations
// <hr>
//
// When the Hermitian property of a matrix is known beforehands using the HermitianMatrix adaptor
// instead of a general matrix can be a considerable performance advantage. This is particularly
// true in case the Hermitian matrix is also symmetric (i.e. has built-in element types). The
// \b Blaze library tries to exploit the properties of Hermitian (symmetric) matrices whenever
// possible. However, there are also situations when using a Hermitian matrix introduces some
// overhead. The following examples demonstrate several situations where Hermitian matrices can
// positively or negatively impact performance.
//
// \n \subsection adaptors_hermitian_matrices_matrix_matrix_multiplication Positive Impact: Matrix/Matrix Multiplication
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
// \n \subsection adaptors_hermitian_matrices_matrix_vector_multiplication Positive Impact: Matrix/Vector Multiplication
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
// \n \subsection adaptors_hermitian_matrices_views Positive Impact: Row/Column Views on Column/Row-Major Matrices
//
// Another example is the optimization of a row view on a column-major symmetric matrix:

   \code
   using blaze::DynamicMatrix;
   using blaze::HermitianMatrix;
   using blaze::Row;
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
// \n \subsection adaptors_hermitian_matrices_assignment Negative Impact: Assignment of a General Matrix
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

// \n Previous: \ref adaptors_symmetric_matrices &nbsp; &nbsp; Next: \ref adaptors_triangular_matrices
*/
//*************************************************************************************************


//**Triangular Matrices****************************************************************************
/*!\page adaptors_triangular_matrices Triangular Matrices
//
// \tableofcontents
//
//
// \n \section adaptors_triangular_matrices_general Triangular Matrices
// <hr>
//
// Triangular matrices come in three flavors: Lower triangular matrices provide the compile time
// guarantee to be square matrices and that the upper part of the matrix contains only default
// elements that cannot be modified. Upper triangular matrices on the other hand provide the
// compile time guarantee to be square and that the lower part of the matrix contains only fixed
// default elements. Finally, diagonal matrices provide the compile time guarantee to be square
// and that both the lower and upper part of the matrix contain only immutable default elements.
// These properties can be exploited to gain higher performance and/or to save memory. Within the
// \b Blaze library, several kinds of lower and upper triangular and diagonal matrices are realized
// by the following class templates:
//
// Lower triangular matrices:
//  - <b>\ref adaptors_triangular_matrices_lowermatrix</b>
//  - <b>\ref adaptors_triangular_matrices_unilowermatrix</b>
//  - <b>\ref adaptors_triangular_matrices_strictlylowermatrix</b>
//
// Upper triangular matrices:
//  - <b>\ref adaptors_triangular_matrices_uppermatrix</b>
//  - <b>\ref adaptors_triangular_matrices_uniuppermatrix</b>
//  - <b>\ref adaptors_triangular_matrices_strictlyuppermatrix</b>
//
// Diagonal matrices
//  - <b>\ref adaptors_triangular_matrices_diagonalmatrix</b>
//
//
// \n \section adaptors_triangular_matrices_lowermatrix LowerMatrix
// <hr>
//
// The blaze::LowerMatrix class template is an adapter for existing dense and sparse matrix types.
// It inherits the properties and the interface of the given matrix type \c MT and extends it by
// enforcing the additional invariant that all matrix elements above the diagonal are 0 (lower
// triangular matrix):

                        \f[\left(\begin{array}{*{5}{c}}
                        l_{0,0} & 0       & 0       & \cdots & 0       \\
                        l_{1,0} & l_{1,1} & 0       & \cdots & 0       \\
                        l_{2,0} & l_{2,1} & l_{2,2} & \cdots & 0       \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots  \\
                        l_{N,0} & l_{N,1} & l_{N,2} & \cdots & l_{N,N} \\
                        \end{array}\right).\f]

// It can be included via the header file

   \code
   #include <blaze/math/LowerMatrix.h>
   \endcode

// The type of the adapted matrix can be specified via the first template parameter:

   \code
   template< typename MT >
   class LowerMatrix;
   \endcode

// \c MT specifies the type of the matrix to be adapted. blaze::LowerMatrix can be used with any
// non-cv-qualified, non-reference, non-pointer, non-expression dense or sparse matrix type. Note
// that the given matrix type must be either resizable (as for instance blaze::HybridMatrix or
// blaze::DynamicMatrix) or must be square at compile time (as for instance blaze::StaticMatrix).
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

   // Definition of a fixed size row-major dense lower matrix based on CustomMatrix
   blaze::LowerMatrix< blaze::CustomMatrix<double,unaligned,unpadded,rowMajor> > D;

   // Definition of a compressed row-major single precision lower matrix
   blaze::LowerMatrix< blaze::CompressedMatrix<float,rowMajor> > E;
   \endcode

// The storage order of a lower matrix is depending on the storage order of the adapted matrix
// type \c MT. In case the adapted matrix is stored in a row-wise fashion (i.e. is specified
// as blaze::rowMajor), the lower matrix will also be a row-major matrix. Otherwise, if the
// adapted matrix is column-major (i.e. is specified as blaze::columnMajor), the lower matrix
// will also be a column-major matrix.
//
//
// \n \section adaptors_triangular_matrices_unilowermatrix UniLowerMatrix
// <hr>
//
// The blaze::UniLowerMatrix class template is an adapter for existing dense and sparse matrix
// types. It inherits the properties and the interface of the given matrix type \c MT and extends
// it by enforcing the additional invariant that all diagonal matrix elements are 1 and all matrix
// elements above the diagonal are 0 (lower unitriangular matrix):

                        \f[\left(\begin{array}{*{5}{c}}
                        1       & 0       & 0       & \cdots & 0      \\
                        l_{1,0} & 1       & 0       & \cdots & 0      \\
                        l_{2,0} & l_{2,1} & 1       & \cdots & 0      \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots \\
                        l_{N,0} & l_{N,1} & l_{N,2} & \cdots & 1      \\
                        \end{array}\right).\f]

// It can be included via the header file

   \code
   #include <blaze/math/UniLowerMatrix.h>
   \endcode

// The type of the adapted matrix can be specified via the first template parameter:

   \code
   template< typename MT >
   class UniLowerMatrix;
   \endcode

// \c MT specifies the type of the matrix to be adapted. blaze::UniLowerMatrix can be used with any
// non-cv-qualified, non-reference, non-pointer, non-expression dense or sparse matrix type. Also,
// the given matrix type must have numeric element types (i.e. all integral types except \c bool,
// floating point and complex types). Note that the given matrix type must be either resizable (as
// for instance blaze::HybridMatrix or blaze::DynamicMatrix) or must be square at compile time (as
// for instance blaze::StaticMatrix).
//
// The following examples give an impression of several possible lower unitriangular matrices:

   \code
   // Definition of a 3x3 row-major dense unilower matrix with static memory
   blaze::UniLowerMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> > A;

   // Definition of a resizable column-major dense unilower matrix based on HybridMatrix
   blaze::UniLowerMatrix< blaze::HybridMatrix<float,4UL,4UL,blaze::columnMajor> B;

   // Definition of a resizable row-major dense unilower matrix based on DynamicMatrix
   blaze::UniLowerMatrix< blaze::DynamicMatrix<double,blaze::rowMajor> > C;

   // Definition of a compressed row-major single precision unilower matrix
   blaze::UniLowerMatrix< blaze::CompressedMatrix<float,blaze::rowMajor> > D;
   \endcode

// The storage order of a lower unitriangular matrix is depending on the storage order of the
// adapted matrix type \c MT. In case the adapted matrix is stored in a row-wise fashion (i.e.
// is specified as blaze::rowMajor), the unilower matrix will also be a row-major matrix.
// Otherwise if the adapted matrix is column-major (i.e. is specified as blaze::columnMajor),
// the unilower matrix will also be a column-major matrix.
//
//
// \n \section adaptors_triangular_matrices_strictlylowermatrix StrictlyLowerMatrix
// <hr>
//
// The blaze::StrictlyLowerMatrix class template is an adapter for existing dense and sparse matrix
// types. It inherits the properties and the interface of the given matrix type \c MT and extends
// it by enforcing the additional invariant that all diagonal matrix elements and all matrix
// elements above the diagonal are 0 (strictly lower triangular matrix):

                        \f[\left(\begin{array}{*{5}{c}}
                        0       & 0       & 0       & \cdots & 0      \\
                        l_{1,0} & 0       & 0       & \cdots & 0      \\
                        l_{2,0} & l_{2,1} & 0       & \cdots & 0      \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots \\
                        l_{N,0} & l_{N,1} & l_{N,2} & \cdots & 0      \\
                        \end{array}\right).\f]

// It can be included via the header file

   \code
   #include <blaze/math/StrictlyLowerMatrix.h>
   \endcode

// The type of the adapted matrix can be specified via the first template parameter:

   \code
   template< typename MT >
   class StrictlyLowerMatrix;
   \endcode

// \c MT specifies the type of the matrix to be adapted. blaze::StrictlyLowerMatrix can be used
// with any non-cv-qualified, non-reference, non-pointer, non-expression dense or sparse matrix
// type. Note that the given matrix type must be either resizable (as for instance
// blaze::HybridMatrix or blaze::DynamicMatrix) or must be square at compile time (as for instance
// blaze::StaticMatrix).
//
// The following examples give an impression of several possible strictly lower triangular matrices:

   \code
   // Definition of a 3x3 row-major dense strictly lower matrix with static memory
   blaze::StrictlyLowerMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> > A;

   // Definition of a resizable column-major dense strictly lower matrix based on HybridMatrix
   blaze::StrictlyLowerMatrix< blaze::HybridMatrix<float,4UL,4UL,blaze::columnMajor> B;

   // Definition of a resizable row-major dense strictly lower matrix based on DynamicMatrix
   blaze::StrictlyLowerMatrix< blaze::DynamicMatrix<double,blaze::rowMajor> > C;

   // Definition of a compressed row-major single precision strictly lower matrix
   blaze::StrictlyLowerMatrix< blaze::CompressedMatrix<float,blaze::rowMajor> > D;
   \endcode

// The storage order of a strictly lower triangular matrix is depending on the storage order of
// the adapted matrix type \c MT. In case the adapted matrix is stored in a row-wise fashion (i.e.
// is specified as blaze::rowMajor), the strictly lower matrix will also be a row-major matrix.
// Otherwise if the adapted matrix is column-major (i.e. is specified as blaze::columnMajor),
// the strictly lower matrix will also be a column-major matrix.
//
//
// \n \section adaptors_triangular_matrices_uppermatrix UpperMatrix
// <hr>
//
// The blaze::UpperMatrix class template is an adapter for existing dense and sparse matrix types.
// It inherits the properties and the interface of the given matrix type \c MT and extends it by
// enforcing the additional invariant that all matrix elements below the diagonal are 0 (upper
// triangular matrix):

                        \f[\left(\begin{array}{*{5}{c}}
                        u_{0,0} & u_{0,1} & u_{0,2} & \cdots & u_{0,N} \\
                        0       & u_{1,1} & u_{1,2} & \cdots & u_{1,N} \\
                        0       & 0       & u_{2,2} & \cdots & u_{2,N} \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots  \\
                        0       & 0       & 0       & \cdots & u_{N,N} \\
                        \end{array}\right).\f]

// It can be included via the header file

   \code
   #include <blaze/math/UpperMatrix.h>
   \endcode

// The type of the adapted matrix can be specified via the first template parameter:

   \code
   template< typename MT >
   class UpperMatrix;
   \endcode

// \c MT specifies the type of the matrix to be adapted. blaze::UpperMatrix can be used with any
// non-cv-qualified, non-reference, non-pointer, non-expression dense or sparse matrix type. Note
// that the given matrix type must be either resizable (as for instance blaze::HybridMatrix or
// blaze::DynamicMatrix) or must be square at compile time (as for instance blaze::StaticMatrix).
//
// The following examples give an impression of several possible upper matrices:

   \code
   // Definition of a 3x3 row-major dense upper matrix with static memory
   blaze::UpperMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> > A;

   // Definition of a resizable column-major dense upper matrix based on HybridMatrix
   blaze::UpperMatrix< blaze::HybridMatrix<float,4UL,4UL,blaze::columnMajor> B;

   // Definition of a resizable row-major dense upper matrix based on DynamicMatrix
   blaze::UpperMatrix< blaze::DynamicMatrix<double,blaze::rowMajor> > C;

   // Definition of a compressed row-major single precision upper matrix
   blaze::UpperMatrix< blaze::CompressedMatrix<float,blaze::rowMajor> > D;
   \endcode

// The storage order of an upper matrix is depending on the storage order of the adapted matrix
// type \c MT. In case the adapted matrix is stored in a row-wise fashion (i.e. is specified
// as blaze::rowMajor), the upper matrix will also be a row-major matrix. Otherwise, if the
// adapted matrix is column-major (i.e. is specified as blaze::columnMajor), the upper matrix
// will also be a column-major matrix.
//
//
// \n \section adaptors_triangular_matrices_uniuppermatrix UniUpperMatrix
// <hr>
//
// The blaze::UniUpperMatrix class template is an adapter for existing dense and sparse matrix
// types. It inherits the properties and the interface of the given matrix type \c MT and extends
// it by enforcing the additional invariant that all diagonal matrix elements are 1 and all matrix
// elements below the diagonal are 0 (upper unitriangular matrix):

                        \f[\left(\begin{array}{*{5}{c}}
                        1       & u_{0,1} & u_{0,2} & \cdots & u_{0,N} \\
                        0       & 1       & u_{1,2} & \cdots & u_{1,N} \\
                        0       & 0       & 1       & \cdots & u_{2,N} \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots  \\
                        0       & 0       & 0       & \cdots & 1       \\
                        \end{array}\right).\f]

// It can be included via the header file

   \code
   #include <blaze/math/UniUpperMatrix.h>
   \endcode

// The type of the adapted matrix can be specified via the first template parameter:

   \code
   template< typename MT >
   class UniUpperMatrix;
   \endcode

// \c MT specifies the type of the matrix to be adapted. blaze::UniUpperMatrix can be used with any
// non-cv-qualified, non-reference, non-pointer, non-expression dense or sparse matrix type. Also,
// the given matrix type must have numeric element types (i.e. all integral types except \c bool,
// floating point and complex types). Note that the given matrix type must be either resizable (as
// for instance blaze::HybridMatrix or blaze::DynamicMatrix) or must be square at compile time (as
// for instance blaze::StaticMatrix).
//
// The following examples give an impression of several possible upper unitriangular matrices:

   \code
   // Definition of a 3x3 row-major dense uniupper matrix with static memory
   blaze::UniUpperMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> > A;

   // Definition of a resizable column-major dense uniupper matrix based on HybridMatrix
   blaze::UniUpperMatrix< blaze::HybridMatrix<float,4UL,4UL,blaze::columnMajor> B;

   // Definition of a resizable row-major dense uniupper matrix based on DynamicMatrix
   blaze::UniUpperMatrix< blaze::DynamicMatrix<double,blaze::rowMajor> > C;

   // Definition of a compressed row-major single precision uniupper matrix
   blaze::UniUpperMatrix< blaze::CompressedMatrix<float,blaze::rowMajor> > D;
   \endcode

// The storage order of an upper unitriangular matrix is depending on the storage order of the
// adapted matrix type \c MT. In case the adapted matrix is stored in a row-wise fashion (i.e.
// is specified as blaze::rowMajor), the uniupper matrix will also be a row-major matrix.
// Otherwise, if the adapted matrix is column-major (i.e. is specified as blaze::columnMajor),
// the uniupper matrix will also be a column-major matrix.
//
//
// \n \section adaptors_triangular_matrices_strictlyuppermatrix StrictlyUpperMatrix
// <hr>
//
// The blaze::StrictlyUpperMatrix class template is an adapter for existing dense and sparse matrix
// types. It inherits the properties and the interface of the given matrix type \c MT and extends
// it by enforcing the additional invariant that all diagonal matrix elements and all matrix
// elements below the diagonal are 0 (strictly upper triangular matrix):

                        \f[\left(\begin{array}{*{5}{c}}
                        0       & u_{0,1} & u_{0,2} & \cdots & u_{0,N} \\
                        0       & 0       & u_{1,2} & \cdots & u_{1,N} \\
                        0       & 0       & 0       & \cdots & u_{2,N} \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots  \\
                        0       & 0       & 0       & \cdots & 0       \\
                        \end{array}\right).\f]

// It can be included via the header file

   \code
   #include <blaze/math/StrictlyUpperMatrix.h>
   \endcode

// The type of the adapted matrix can be specified via the first template parameter:

   \code
   template< typename MT >
   class StrictlyUpperMatrix;
   \endcode

// \c MT specifies the type of the matrix to be adapted. blaze::StrictlyUpperMatrix can be used
// with any non-cv-qualified, non-reference, non-pointer, non-expression dense or sparse matrix
// type. Note that the given matrix type must be either resizable (as for instance
// blaze::HybridMatrix or blaze::DynamicMatrix) or must be square at compile time (as for instance
// blaze::StaticMatrix).
//
// The following examples give an impression of several possible strictly upper triangular matrices:

   \code
   // Definition of a 3x3 row-major dense strictly upper matrix with static memory
   blaze::StrictlyUpperMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> > A;

   // Definition of a resizable column-major dense strictly upper matrix based on HybridMatrix
   blaze::StrictlyUpperMatrix< blaze::HybridMatrix<float,4UL,4UL,blaze::columnMajor> B;

   // Definition of a resizable row-major dense strictly upper matrix based on DynamicMatrix
   blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<double,blaze::rowMajor> > C;

   // Definition of a compressed row-major single precision strictly upper matrix
   blaze::StrictlyUpperMatrix< blaze::CompressedMatrix<float,blaze::rowMajor> > D;
   \endcode

// The storage order of a strictly upper triangular matrix is depending on the storage order of
// the adapted matrix type \c MT. In case the adapted matrix is stored in a row-wise fashion (i.e.
// is specified as blaze::rowMajor), the strictly upper matrix will also be a row-major matrix.
// Otherwise, if the adapted matrix is column-major (i.e. is specified as blaze::columnMajor),
// the strictly upper matrix will also be a column-major matrix.
//
//
// \n \section adaptors_triangular_matrices_diagonalmatrix DiagonalMatrix
// <hr>
//
// The blaze::DiagonalMatrix class template is an adapter for existing dense and sparse matrix
// types. It inherits the properties and the interface of the given matrix type \c MT and extends
// it by enforcing the additional invariant that all matrix elements above and below the diagonal
// are 0 (diagonal matrix):

                        \f[\left(\begin{array}{*{5}{c}}
                        l_{0,0} & 0       & 0       & \cdots & 0       \\
                        0       & l_{1,1} & 0       & \cdots & 0       \\
                        0       & 0       & l_{2,2} & \cdots & 0       \\
                        \vdots  & \vdots  & \vdots  & \ddots & \vdots  \\
                        0       & 0       & 0       & \cdots & l_{N,N} \\
                        \end{array}\right).\f]

// It can be included via the header file

   \code
   #include <blaze/math/DiagonalMatrix.h>
   \endcode

// The type of the adapted matrix can be specified via the first template parameter:

   \code
   template< typename MT >
   class DiagonalMatrix;
   \endcode

// \c MT specifies the type of the matrix to be adapted. blaze::DiagonalMatrix can be used with any
// non-cv-qualified, non-reference, non-pointer, non-expression dense or sparse matrix type. Note
// that the given matrix type must be either resizable (as for instance blaze::HybridMatrix or
// blaze::DynamicMatrix) or must be square at compile time (as for instance blaze::StaticMatrix).
//
// The following examples give an impression of several possible diagonal matrices:

   \code
   // Definition of a 3x3 row-major dense diagonal matrix with static memory
   blaze::DiagonalMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> > A;

   // Definition of a resizable column-major dense diagonal matrix based on HybridMatrix
   blaze::DiagonalMatrix< blaze::HybridMatrix<float,4UL,4UL,blaze::columnMajor> B;

   // Definition of a resizable row-major dense diagonal matrix based on DynamicMatrix
   blaze::DiagonalMatrix< blaze::DynamicMatrix<double,blaze::rowMajor> > C;

   // Definition of a compressed row-major single precision diagonal matrix
   blaze::DiagonalMatrix< blaze::CompressedMatrix<float,blaze::rowMajor> > D;
   \endcode

// The storage order of a diagonal matrix is depending on the storage order of the adapted matrix
// type \c MT. In case the adapted matrix is stored in a row-wise fashion (i.e. is specified
// as blaze::rowMajor), the diagonal matrix will also be a row-major matrix. Otherwise, if the
// adapted matrix is column-major (i.e. is specified as blaze::columnMajor), the diagonal matrix
// will also be a column-major matrix.
//
//
// \n \section adaptors_triangular_matrices_special_properties Special Properties of Triangular Matrices
// <hr>
//
// A triangular matrix is used exactly like a matrix of the underlying, adapted matrix type \c MT.
// It also provides (nearly) the same interface as the underlying matrix type. However, there are
// some important exceptions resulting from the triangular matrix constraint:
//
//  -# <b>\ref adaptors_triangular_matrices_square</b>
//  -# <b>\ref adaptors_triangular_matrices_triangular</b>
//  -# <b>\ref adaptors_triangular_matrices_initialization</b>
//  -# <b>\ref adaptors_triangular_matrices_storage</b>
//  -# <b>\ref adaptors_triangular_matrices_scaling</b>
//
// \n \subsection adaptors_triangular_matrices_square Triangular Matrices Must Always be Square!
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

// \n \subsection adaptors_triangular_matrices_triangular The Triangular Property is Always Enforced!
//
// This means that it is only allowed to modify elements in the lower part or the diagonal of
// a lower triangular matrix and in the upper part or the diagonal of an upper triangular matrix.
// Unitriangular and strictly triangular matrices are even more restrictive and don't allow the
// modification of diagonal elements. Also, triangular matrices can only be assigned matrices that
// don't violate their triangular property. The following example demonstrates this restriction
// by means of the blaze::LowerMatrix adaptor. For examples with other triangular matrix types
// see the according class documentations.

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
   StaticMatrix<double,3UL,3UL> B{ {  3.0,  0.0,  0.0 },
                                   {  8.0,  0.0,  0.0 },
                                   { -2.0, -1.0,  4.0 } };

   LowerMatrix< DynamicMatrix<double,rowMajor> > C( B );  // OK

   // Assignment of a non-lower dense matrix
   StaticMatrix<double,3UL,3UL> D{ {  3.0,  0.0, -2.0 },
                                   {  8.0,  0.0,  0.0 },
                                   { -2.0, -1.0,  4.0 } };

   C = D;  // Throws an exception; lower matrix invariant would be violated!
   \endcode

// The triangular property is also enforced during the construction of triangular custom matrices:
// In case the given array of elements does not represent the according triangular matrix type, a
// \c std::invalid_argument exception is thrown:

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

// Finally, the triangular matrix property is enforced for views (rows, columns, submatrices, ...)
// on the triangular matrix. The following example demonstrates that modifying the elements of an
// entire row and submatrix of a lower matrix only affects the lower and diagonal matrix elements.
// Again, this example uses blaze::LowerMatrix, for examples with other triangular matrix types
// see the according class documentations.

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
// triangular matrices. Since only lower/upper and potentially diagonal elements may be modified
// the matrix to be assigned must be structured such that the triangular matrix invariant of the
// matrix is preserved. Otherwise a \c std::invalid_argument exception is thrown:

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
   DynamicVector<int,rowVector> v{ 1, 2, 3, 0 };

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

// \n \subsection adaptors_triangular_matrices_initialization The Elements of a Dense Triangular Matrix are Always Default Initialized!
//
// Although this results in a small loss of efficiency during the creation of a dense lower or
// upper matrix this initialization is important since otherwise the lower/upper matrix property
// of dense lower matrices would not be guaranteed:

   \code
   using blaze::DynamicMatrix;
   using blaze::LowerMatrix;
   using blaze::UpperMatrix;

   // Uninitialized, 5x5 row-major dynamic matrix
   DynamicMatrix<int,rowMajor> A( 5, 5 );

   // 5x5 row-major lower dynamic matrix with default initialized upper matrix
   LowerMatrix< DynamicMatrix<int,rowMajor> > B( 5 );

   // 7x7 column-major upper dynamic matrix with default initialized lower matrix
   UpperMatrix< DynamicMatrix<int,columnMajor> > C( 7 );

   // 3x3 row-major diagonal dynamic matrix with default initialized lower and upper matrix
   DiagonalMatrix< DynamicMatrix<int,rowMajor> > D( 3 );
   \endcode

// \n \subsection adaptors_triangular_matrices_storage Dense Triangular Matrices Store All Elements!
//
// All dense triangular matrices store all \f$ N \times N \f$ elements, including the immutable
// elements in the lower or upper part, respectively. Therefore dense triangular matrices don't
// provide any kind of memory reduction! There are two main reasons for this: First, storing also
// the zero elements guarantees maximum performance for many algorithms that perform vectorized
// operations on the triangular matrices, which is especially true for small dense matrices.
// Second, conceptually all triangular adaptors merely restrict the interface to the matrix type
// \c MT and do not change the data layout or the underlying matrix type.
//
// This property matters most for diagonal matrices. In order to achieve the perfect combination
// of performance and memory consumption for a diagonal matrix it is recommended to use dense
// matrices for small diagonal matrices and sparse matrices for large diagonal matrices:

   \code
   // Recommendation 1: use dense matrices for small diagonal matrices
   typedef blaze::DiagonalMatrix< blaze::StaticMatrix<float,3UL,3UL> >  SmallDiagonalMatrix;

   // Recommendation 2: use sparse matrices for large diagonal matrices
   typedef blaze::DiagonalMatrix< blaze::CompressedMatrix<float> >  LargeDiagonalMatrix;
   \endcode

// \n \subsection adaptors_triangular_matrices_scaling Unitriangular Matrices Cannot Be Scaled!
//
// Since the diagonal elements of a unitriangular matrix have a fixed value of 1 it is not possible
// to self-scale such a matrix:

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

// \n \section adaptors_triangular_matrices_arithmetic_operations Arithmetic Operations
// <hr>
//
// A lower and upper triangular matrix can participate in numerical operations in any way any other
// dense or sparse matrix can participate. It can also be combined with any other dense or sparse
// vector or matrix. The following code example gives an impression of the use of blaze::LowerMatrix
// and blaze::UpperMatrix within arithmetic operations:

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
   UpperMatrix< CompressedMatrix<double,rowMajor> > D( 3 );

   LowerMatrix< HybridMatrix<float,3UL,3UL,rowMajor> > E;
   UpperMatrix< StaticMatrix<float,3UL,3UL,columnMajor> > F;

   E = A + B;     // Matrix addition and assignment to a row-major lower matrix
   F = C - D;     // Matrix subtraction and assignment to a column-major upper matrix
   F = A * D;     // Matrix multiplication between a dense and a sparse matrix

   C *= 2.0;      // In-place scaling of matrix C
   E  = 2.0 * B;  // Scaling of matrix B
   F  = C * 2.0;  // Scaling of matrix C

   E += A - B;    // Addition assignment
   F -= C + D;    // Subtraction assignment
   F *= A * D;    // Multiplication assignment
   \endcode

// Note that diagonal, unitriangular and strictly triangular matrix types can be used in the same
// way, but may pose some additional restrictions (see the according class documentations).
//
//
// \n \section adaptors_triangular_matrices_block_structured Block-Structured Triangular Matrices
// <hr>
//
// It is also possible to use block-structured triangular matrices:

   \code
   using blaze::CompressedMatrix;
   using blaze::DynamicMatrix;
   using blaze::StaticMatrix;
   using blaze::LowerMatrix;
   using blaze::UpperMatrix;

   // Definition of a 5x5 block-structured lower matrix based on DynamicMatrix
   LowerMatrix< DynamicMatrix< StaticMatrix<int,3UL,3UL> > > A( 5 );

   // Definition of a 7x7 block-structured upper matrix based on CompressedMatrix
   UpperMatrix< CompressedMatrix< StaticMatrix<int,3UL,3UL> > > B( 7 );
   \endcode

// Also in this case the triangular matrix invariant is enforced, i.e. it is not possible to
// manipulate elements in the upper part (lower triangular matrix) or the lower part (upper
// triangular matrix) of the matrix:

   \code
   const StaticMatrix<int,3UL,3UL> C{ { 1, -4,  5 },
                                      { 6,  8, -3 },
                                      { 2, -1,  2 } };

   A(2,4)(1,1) = -5;     // Invalid manipulation of upper matrix element; Results in an exception
   B.insert( 4, 2, C );  // Invalid insertion of the elements (4,2); Results in an exception
   \endcode

// Note that unitriangular matrices are restricted to numeric element types and therefore cannot
// be used for block-structured matrices:

   \code
   using blaze::CompressedMatrix;
   using blaze::DynamicMatrix;
   using blaze::StaticMatrix;
   using blaze::UniLowerMatrix;
   using blaze::UniUpperMatrix;

   // Compilation error: lower unitriangular matrices are restricted to numeric element types
   UniLowerMatrix< DynamicMatrix< StaticMatrix<int,3UL,3UL> > > A( 5 );

   // Compilation error: upper unitriangular matrices are restricted to numeric element types
   UniUpperMatrix< CompressedMatrix< StaticMatrix<int,3UL,3UL> > > B( 7 );
   \endcode

// \n \section adaptors_triangular_matrices_performance Performance Considerations
// <hr>
//
// The \b Blaze library tries to exploit the properties of lower and upper triangular matrices
// whenever and wherever possible. Therefore using triangular matrices instead of a general
// matrices can result in a considerable performance improvement. However, there are also
// situations when using a triangular matrix introduces some overhead. The following examples
// demonstrate several common situations where triangular matrices can positively or negatively
// impact performance.
//
// \n \subsection adaptors_triangular_matrices_matrix_matrix_multiplication Positive Impact: Matrix/Matrix Multiplication
//
// When multiplying two matrices, at least one of which is triangular, \b Blaze can exploit the
// fact that either the lower or upper part of the matrix contains only default elements and
// restrict the algorithm to the non-zero elements. The following example demonstrates this by
// means of a dense matrix/dense matrix multiplication with lower triangular matrices:

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
// especially for large matrices. Therefore is it highly recommended to use the blaze::LowerMatrix
// and blaze::UpperMatrix adaptors when a matrix is known to be lower or upper triangular,
// respectively. Note however that the performance advantage is most pronounced for dense matrices
// and much less so for sparse matrices.
//
// \n \subsection adaptors_triangular_matrices_matrix_vector_multiplication Positive Impact: Matrix/Vector Multiplication
//
// A similar performance improvement can be gained when using a triangular matrix in a matrix/vector
// multiplication:

   \code
   using blaze::DynamicMatrix;
   using blaze::DynamicVector;
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
// \n \subsection adaptors_triangular_matrices_assignment Negative Impact: Assignment of a General Matrix
//
// In contrast to using a triangular matrix on the right-hand side of an assignment (i.e. for
// read access), which introduces absolutely no performance penalty, using a triangular matrix
// on the left-hand side of an assignment (i.e. for write access) may introduce additional
// overhead when it is assigned a general matrix, which is not triangular at compile time:

   \code
   using blaze::DynamicMatrix;
   using blaze::LowerMatrix;

   LowerMatrix< DynamicMatrix<double> > A, C;
   DynamicMatrix<double> B;

   B = A;  // Only read-access to the lower matrix; no performance penalty
   C = A;  // Assignment of a lower matrix to another lower matrix; no runtime overhead
   C = B;  // Assignment of a general matrix to a lower matrix; some runtime overhead
   \endcode

// When assigning a general (potentially not lower triangular) matrix to a lower matrix or a
// general (potentially not upper triangular) matrix to an upper matrix it is necessary to check
// whether the matrix is lower or upper at runtime in order to guarantee the triangular property
// of the matrix. In case it turns out to be lower or upper, respectively, it is assigned as
// efficiently as possible, if it is not, an exception is thrown. In order to prevent this runtime
// overhead it is therefore generally advisable to assign lower or upper triangular matrices to
// other lower or upper triangular matrices.\n
// In this context it is especially noteworthy that the addition, subtraction, and multiplication
// of two triangular matrices of the same structure always results in another triangular matrix:

   \code
   LowerMatrix< DynamicMatrix<double> > A, B, C;

   C = A + B;  // Results in a lower matrix; no runtime overhead
   C = A - B;  // Results in a lower matrix; no runtime overhead
   C = A * B;  // Results in a lower matrix; no runtime overhead
   \endcode

   \code
   UpperMatrix< DynamicMatrix<double> > A, B, C;

   C = A + B;  // Results in a upper matrix; no runtime overhead
   C = A - B;  // Results in a upper matrix; no runtime overhead
   C = A * B;  // Results in a upper matrix; no runtime overhead
   \endcode

// \n Previous: \ref adaptors_hermitian_matrices &nbsp; &nbsp; Next: \ref views
*/
//*************************************************************************************************


//**Views******************************************************************************************
/*!\page views Views
//
// \tableofcontents
//
//
// \section views_general General Concepts
// <hr>
//
// Views represents parts of a vector or matrix, such as a subvector, a submatrix, or a specific
// row or column of a matrix. As such, views act as a reference to a specific part of a vector
// or matrix. This reference is valid and can be used in every way as any other vector or matrix
// can be used as long as the referenced vector or matrix is not resized or entirely destroyed.
// Views also act as alias to the elements of the vector or matrix: Changes made to the elements
// (e.g. modifying values, inserting or erasing elements) via the view are immediately visible in
// the vector or matrix and changes made via the vector or matrix are immediately visible in the
// view.
//
// The \b Blaze library provides the following views on vectors and matrices:
//
// Vector views:
//  - \ref views_subvectors
//
// Matrix views:
//  - \ref views_submatrices
//  - \ref views_rows
//  - \ref views_columns
//
//
// \n \section views_examples Examples

   \code
   using blaze::DynamicMatrix;
   using blaze::StaticVector;

   // Setup of the 3x5 row-major matrix
   //
   //  ( 1  0 -2  3  0 )
   //  ( 0  2  5 -1 -1 )
   //  ( 1  0  0  2  1 )
   //
   DynamicMatrix<int> A{ { 1,  0, -2,  3,  0 },
                         { 0,  2,  5, -1, -1 },
                         { 1,  0,  0,  2,  1 } };

   // Setup of the 2-dimensional row vector
   //
   //  ( 18 19 )
   //
   StaticVector<int,rowVector> vec{ 18, 19 };

   // Assigning to the elements (1,2) and (1,3) via a subvector of a row
   //
   //  ( 1  0 -2  3  0 )
   //  ( 0  2 18 19 -1 )
   //  ( 1  0  0  2  1 )
   //
   subvector( row( A, 1UL ), 2UL, 2UL ) = vec;
   \endcode

// \n Previous: \ref adaptors_triangular_matrices &nbsp; &nbsp; Next: \ref views_subvectors
*/
//*************************************************************************************************


//**Subvectors*************************************************************************************
/*!\page views_subvectors Subvectors
//
// \tableofcontents
//
//
// Subvectors provide views on a specific part of a dense or sparse vector. As such, subvectors
// act as a reference to a specific range within a vector. This reference is valid and can be
// used in every way any other dense or sparse vector can be used as long as the vector containing
// the subvector is not resized or entirely destroyed. The subvector also acts as an alias to the
// vector elements in the specified range: Changes made to the elements (e.g. modifying values,
// inserting or erasing elements) are immediately visible in the vector and changes made via the
// vector are immediately visible in the subvector.
//
//
// \n \section views_subvectors_class The Subvector Class Template
// <hr>
//
// The blaze::Subvector class template represents a view on a specific subvector of a dense or
// sparse vector primitive. It can be included via the header file

   \code
   #include <blaze/math/Subvector.h>
   \endcode

// The type of the vector is specified via two template parameters:

   \code
   template< typename VT, bool AF >
   class Subvector;
   \endcode

//  - \c VT: specifies the type of the vector primitive. Subvector can be used with every vector
//           primitive or view, but does not work with any vector expression type.
//  - \c AF: the alignment flag specifies whether the subvector is aligned (blaze::aligned) or
//           unaligned (blaze::unaligned). The default value is blaze::unaligned.
//
//
// \n \section views_subvectors_setup Setup of Subvectors
// <hr>
//
// A view on a dense or sparse subvector can be created very conveniently via the \c subvector()
// function. This view can be treated as any other vector, i.e. it can be assigned to, it can
// be copied from, and it can be used in arithmetic operations. A subvector created from a row
// vector can be used as any other row vector, a subvector created from a column vector can be
// used as any other column vector. The view can also be used on both sides of an assignment:
// The subvector can either be used as an alias to grant write access to a specific subvector
// of a vector primitive on the left-hand side of an assignment or to grant read-access to a
// specific subvector of a vector primitive or expression on the right-hand side of an assignment.
// The following example demonstrates this in detail:

   \code
   typedef blaze::DynamicVector<double,blaze::rowVector>  DenseVectorType;
   typedef blaze::CompressedVector<int,blaze::rowVector>  SparseVectorType;

   DenseVectorType  d1, d2;
   SparseVectorType s1, s2;
   // ... Resizing and initialization

   // Creating a view on the first ten elements of the dense vector d1
   blaze::Subvector<DenseVectorType> dsv = subvector( d1, 0UL, 10UL );

   // Creating a view on the second ten elements of the sparse vector s1
   blaze::Subvector<SparseVectorType> ssv = subvector( s1, 10UL, 10UL );

   // Creating a view on the addition of d2 and s2
   dsv = subvector( d2 + s2, 5UL, 10UL );

   // Creating a view on the multiplication of d2 and s2
   ssv = subvector( d2 * s2, 2UL, 10UL );
   \endcode

// The \c subvector() function can be used on any dense or sparse vector, including expressions,
// as demonstrated in the example. Note however that a blaze::Subvector can only be instantiated
// with a dense or sparse vector primitive, i.e. with types that can be written, and not with an
// expression type.
//
//
// \n \section views_subvectors_common_operations Common Operations
// <hr>
//
// A subvector view can be used like any other dense or sparse vector. For instance, the current
// number of elements can be obtained via the \c size() function, the current capacity via the
// \c capacity() function, and the number of non-zero elements via the \c nonZeros() function.
// However, since subvectors are references to a specific range of a vector, several operations
// are not possible on views, such as resizing and swapping. The following example shows this by
// means of a dense subvector view:

   \code
   typedef blaze::DynamicVector<int,blaze::rowVector>  VectorType;
   typedef blaze::Subvector<VectorType>                SubvectorType;

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

// \n \section views_subvectors_element_access Element Access
// <hr>
//
// The elements of a subvector can be directly accessed via the subscript operator:

   \code
   typedef blaze::DynamicVector<double,blaze::rowVector>  VectorType;
   VectorType v;
   // ... Resizing and initialization

   // Creating an 8-dimensional subvector, starting from index 4
   blaze::Subvector<VectorType> sv = subvector( v, 4UL, 8UL );

   // Setting the 1st element of the subvector, which corresponds to
   // the element at index 5 in vector v
   sv[1] = 2.0;
   \endcode

   \code
   typedef blaze::CompressedVector<double,blaze::rowVector>  VectorType;
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
   typedef blaze::DynamicVector<int,blaze::rowVector>  VectorType;
   typedef blaze::Subvector<VectorType>                SubvectorType;

   VectorType v( 256UL );
   // ... Resizing and initialization

   // Creating a reference to a specific subvector of the dense vector v
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
   typedef blaze::CompressedVector<int,blaze::rowVector>  VectorType;
   typedef blaze::Subvector<VectorType>                   SubvectorType;

   VectorType v( 256UL );
   // ... Resizing and initialization

   // Creating a reference to a specific subvector of the sparse vector v
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

// \n \section views_subvectors_element_insertion Element Insertion
// <hr>
//
// Inserting/accessing elements in a sparse subvector can be done by several alternative functions.
// The following example demonstrates all options:

   \code
   typedef blaze::CompressedVector<double,blaze::rowVector>  VectorType;
   VectorType v( 256UL );  // Non-initialized vector of size 256

   typedef blaze::Subvector<VectorType>  SubvectorType;
   SubvectorType sv( subvector( v, 10UL, 60UL ) );  // View on the range [10..69] of v

   // The subscript operator provides access to all possible elements of the sparse subvector,
   // including the zero elements. In case the subscript operator is used to access an element
   // that is currently not stored in the sparse subvector, the element is inserted into the
   // subvector.
   sv[42] = 2.0;

   // The second operation for inserting elements is the set() function. In case the element
   // is not contained in the vector it is inserted into the vector, if it is already contained
   // in the vector its value is modified.
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

// \n \section views_subvectors_arithmetic_operations Arithmetic Operations
// <hr>
//
// Both dense and sparse subvectors can be used in all arithmetic operations that any other dense
// or sparse vector can be used in. The following example gives an impression of the use of dense
// subvectors within arithmetic operations. All operations (addition, subtraction, multiplication,
// scaling, ...) can be performed on all possible combinations of dense and sparse subvectors with
// fitting element types:

   \code
   typedef blaze::DynamicVector<double,blaze::rowVector>     DenseVectorType;
   typedef blaze::CompressedVector<double,blaze::rowVector>  SparseVectorType;
   DenseVectorType d1, d2, d3;
   SparseVectorType s1, s2;

   // ... Resizing and initialization

   typedef blaze::DynamicMatrix<double,blaze::rowMajor>  DenseMatrixType;
   DenseMatrixType A;

   typedef blaze::Subvector<DenseVectorType>  SubvectorType;
   SubvectorType dsv( subvector( d1, 0UL, 10UL ) );  // View on the range [0..9] of vector d1

   dsv = d2;                          // Dense vector initialization of the range [0..9]
   subvector( d1, 10UL, 10UL ) = s1;  // Sparse vector initialization of the range [10..19]

   d3 = dsv + d2;                           // Dense vector/dense vector addition
   s2 = s1 + subvector( d1, 10UL, 10UL );   // Sparse vector/dense vector addition
   d2 = dsv * subvector( d1, 20UL, 10UL );  // Component-wise vector multiplication

   subvector( d1, 3UL, 4UL ) *= 2.0;      // In-place scaling of the range [3..6]
   d2 = subvector( d1, 7UL, 3UL ) * 2.0;  // Scaling of the range [7..9]
   d2 = 2.0 * subvector( d1, 7UL, 3UL );  // Scaling of the range [7..9]

   subvector( d1, 0UL , 10UL ) += d2;   // Addition assignment
   subvector( d1, 10UL, 10UL ) -= s2;   // Subtraction assignment
   subvector( d1, 20UL, 10UL ) *= dsv;  // Multiplication assignment

   double scalar = subvector( d1, 5UL, 10UL ) * trans( s1 );  // Scalar/dot/inner product between two vectors

   A = trans( s1 ) * subvector( d1, 4UL, 16UL );  // Outer product between two vectors
   \endcode

// \n \section views_aligned_subvectors Aligned Subvectors
// <hr>
//
// Usually subvectors can be defined anywhere within a vector. They may start at any position and
// may have an arbitrary size (only restricted by the size of the underlying vector). However, in
// contrast to vectors themselves, which are always properly aligned in memory and therefore can
// provide maximum performance, this means that subvectors in general have to be considered to be
// unaligned. This can be made explicit by the blaze::unaligned flag:

   \code
   using blaze::unaligned;

   typedef blaze::DynamicVector<double,blaze::rowVector>  DenseVectorType;

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
// full flexibility in the creation of subvectors, this might result in performance disadvantages
// in comparison to vector primitives (even in case the specified subvector could be aligned).
// Whereas vector primitives are guaranteed to be properly aligned and therefore provide maximum
// performance in all operations, a general view on a vector might not be properly aligned. This
// may cause a performance penalty on some platforms and/or for some operations.
//
// However, it is also possible to create aligned subvectors. Aligned subvectors are identical to
// unaligned subvectors in all aspects, except that they may pose additional alignment restrictions
// and therefore have less flexibility during creation, but don't suffer from performance penalties
// and provide the same performance as the underlying vector. Aligned subvectors are created by
// explicitly specifying the blaze::aligned flag:

   \code
   using blaze::aligned;

   // Creating an aligned dense subvector in the range [8..23]
   blaze::Subvector<DenseVectorType,aligned> sv = subvector<aligned>( x, 8UL, 16UL );
   \endcode

// The alignment restrictions refer to system dependent address restrictions for the used element
// type and the available vectorization mode (SSE, AVX, ...). In order to be properly aligned the
// first element of the subvector must be aligned. The following source code gives some examples
// for a double precision dynamic vector, assuming that AVX is available, which packs 4 \c double
// values into a SIMD vector:

   \code
   using blaze::aligned;
   using blaze::columnVector;

   typedef blaze::DynamicVector<double,columnVector>  VectorType;
   typedef blaze::Subvector<VectorType,aligned>  SubvectorType;

   VectorType d( 17UL );
   // ... Resizing and initialization

   // OK: Starts at the beginning, i.e. the first element is aligned
   SubvectorType dsv1 = subvector<aligned>( d, 0UL, 13UL );

   // OK: Start index is a multiple of 4, i.e. the first element is aligned
   SubvectorType dsv2 = subvector<aligned>( d, 4UL, 7UL );

   // OK: The start index is a multiple of 4 and the subvector includes the last element
   SubvectorType dsv3 = subvector<aligned>( d, 8UL, 9UL );

   // Error: Start index is not a multiple of 4, i.e. the first element is not aligned
   SubvectorType dsv4 = subvector<aligned>( d, 5UL, 8UL );
   \endcode

// Note that the discussed alignment restrictions are only valid for aligned dense subvectors.
// In contrast, aligned sparse subvectors at this time don't pose any additional restrictions.
// Therefore aligned and unaligned sparse subvectors are truly fully identical. Still, in case
// the blaze::aligned flag is specified during setup, an aligned subvector is created:

   \code
   using blaze::aligned;

   typedef blaze::CompressedVector<double,blaze::rowVector>  SparseVectorType;

   SparseVectorType x;
   // ... Resizing and initialization

   // Creating an aligned subvector in the range [8..23]
   blaze::Subvector<SparseVectorType,aligned> sv = subvector<aligned>( x, 8UL, 16UL );
   \endcode

// \n \section views_subvectors_on_subvectors Subvectors on Subvectors
// <hr>
//
// It is also possible to create a subvector view on another subvector. In this context it is
// important to remember that the type returned by the \c subvector() function is the same type
// as the type of the given subvector, not a nested subvector type, since the view on a subvector
// is just another view on the underlying vector:

   \code
   typedef blaze::DynamicVector<double,blaze::rowVector>  VectorType;
   typedef blaze::Subvector<VectorType>                   SubvectorType;

   VectorType d1;

   // ... Resizing and initialization

   // Creating a subvector view on the dense vector d1
   SubvectorType sv1 = subvector( d1, 5UL, 10UL );

   // Creating a subvector view on the dense subvector sv1
   SubvectorType sv2 = subvector( sv1, 1UL, 5UL );
   \endcode

// \n Previous: \ref views &nbsp; &nbsp; Next: \ref views_submatrices
*/
//*************************************************************************************************


//**Submatrices************************************************************************************
/*!\page views_submatrices Submatrices
//
// \tableofcontents
//
//
// Submatrices provide views on a specific part of a dense or sparse matrix just as subvectors
// provide views on specific parts of vectors. As such, submatrices act as a reference to a
// specific block within a matrix. This reference is valid and can be used in evary way any
// other dense or sparse matrix can be used as long as the matrix containing the submatrix is
// not resized or entirely destroyed. The submatrix also acts as an alias to the matrix elements
// in the specified block: Changes made to the elements (e.g. modifying values, inserting or
// erasing elements) are immediately visible in the matrix and changes made via the matrix are
// immediately visible in the submatrix.
//
//
// \n \section views_submatrices_class The Submatrix Class Template
// <hr>
//
// The blaze::Submatrix class template represents a view on a specific submatrix of a dense or
// sparse matrix primitive. It can be included via the header file

   \code
   #include <blaze/math/Submatrix.h>
   \endcode

// The type of the matrix is specified via two template parameters:

   \code
   template< typename MT, bool AF >
   class Submatrix;
   \endcode

//  - \c MT: specifies the type of the matrix primitive. Submatrix can be used with every matrix
//           primitive, but does not work with any matrix expression type.
//  - \c AF: the alignment flag specifies whether the submatrix is aligned (blaze::aligned) or
//           unaligned (blaze::unaligned). The default value is blaze::unaligned.
//
//
// \n \section views_submatrices_setup Setup of Submatrices
// <hr>
//
// A view on a submatrix can be created very conveniently via the \c submatrix() function.
// This view can be treated as any other matrix, i.e. it can be assigned to, it can be copied
// from, and it can be used in arithmetic operations. A submatrix created from a row-major
// matrix will itself be a row-major matrix, a submatrix created from a column-major matrix
// will be a column-major matrix. The view can also be used on both sides of an assignment:
// The submatrix can either be used as an alias to grant write access to a specific submatrix
// of a matrix primitive on the left-hand side of an assignment or to grant read-access to
// a specific submatrix of a matrix primitive or expression on the right-hand side of an
// assignment. The following example demonstrates this in detail:

   \code
   typedef blaze::DynamicMatrix<double,blaze::rowMajor>     DenseMatrixType;
   typedef blaze::CompressedVector<int,blaze::columnMajor>  SparseMatrixType;

   DenseMatrixType  D1, D2;
   SparseMatrixType S1, S2;
   // ... Resizing and initialization

   // Creating a view on the first 8x16 block of the dense matrix D1
   blaze::Submatrix<DenseMatrixType> dsm = submatrix( D1, 0UL, 0UL, 8UL, 16UL );

   // Creating a view on the second 8x16 block of the sparse matrix S1
   blaze::Submatrix<SparseMatrixType> ssm = submatrix( S1, 0UL, 16UL, 8UL, 16UL );

   // Creating a view on the addition of D2 and S2
   dsm = submatrix( D2 + S2, 5UL, 10UL, 8UL, 16UL );

   // Creating a view on the multiplication of D2 and S2
   ssm = submatrix( D2 * S2, 7UL, 13UL, 8UL, 16UL );
   \endcode

// \n \section views_submatrices_common_operations Common Operations
// <hr>
//
// The current size of the matrix, i.e. the number of rows or columns can be obtained via the
// \c rows() and \c columns() functions, the current total capacity via the \c capacity() function,
// and the number of non-zero elements via the \c nonZeros() function. However, since submatrices
// are views on a specific submatrix of a matrix, several operations are not possible on views,
// such as resizing and swapping:

   \code
   typedef blaze::DynamicMatrix<int,blaze::rowMajor>  MatrixType;
   typedef blaze::Submatrix<MatrixType>               SubmatrixType;

   MatrixType A;
   // ... Resizing and initialization

   // Creating a view on the a 8x12 submatrix of matrix A
   SubmatrixType sm = submatrix( A, 0UL, 0UL, 8UL, 12UL );

   sm.rows();      // Returns the number of rows of the submatrix
   sm.columns();   // Returns the number of columns of the submatrix
   sm.capacity();  // Returns the capacity of the submatrix
   sm.nonZeros();  // Returns the number of non-zero elements contained in the submatrix

   sm.resize( 10UL, 8UL );  // Compilation error: Cannot resize a submatrix of a matrix

   SubmatrixType sm2 = submatrix( A, 8UL, 0UL, 12UL, 8UL );
   swap( sm, sm2 );  // Compilation error: Swap operation not allowed
   \endcode

// \n \section views_submatrices_element_access Element Access
// <hr>
//
// The elements of a submatrix can be directly accessed with the function call operator:

   \code
   typedef blaze::DynamicMatrix<double,blaze::rowMajor>  MatrixType;
   MatrixType A;
   // ... Resizing and initialization

   // Creating a 8x8 submatrix, starting from position (4,4)
   blaze::Submatrix<MatrixType> sm = submatrix( A, 4UL, 4UL, 8UL, 8UL );

   // Setting the element (0,0) of the submatrix, which corresponds to
   // the element at position (4,4) in matrix A
   sm(0,0) = 2.0;
   \endcode

   \code
   typedef blaze::CompressedMatrix<double,blaze::rowMajor>  MatrixType;
   MatrixType A;
   // ... Resizing and initialization

   // Creating a 8x8 submatrix, starting from position (4,4)
   blaze::Submatrix<MatrixType> sm = submatrix( A, 4UL, 4UL, 8UL, 8UL );

   // Setting the element (0,0) of the submatrix, which corresponds to
   // the element at position (4,4) in matrix A
   sm(0,0) = 2.0;
   \endcode

// Alternatively, the elements of a submatrix can be traversed via (const) iterators. Just as
// with matrices, in case of non-const submatrices, \c begin() and \c end() return an Iterator,
// which allows a manipulation of the non-zero values, in case of constant submatrices a
// ConstIterator is returned:

   \code
   typedef blaze::DynamicMatrix<int,blaze::rowMajor>  MatrixType;
   typedef blaze::Submatrix<MatrixType>               SubmatrixType;

   MatrixType A( 256UL, 512UL );
   // ... Resizing and initialization

   // Creating a reference to a specific submatrix of the dense matrix A
   SubmatrixType sm = submatrix( A, 16UL, 16UL, 64UL, 128UL );

   // Traversing the elements of the 0th row via iterators to non-const elements
   for( SubmatrixType::Iterator it=sm.begin(0); it!=sm.end(0); ++it ) {
      *it = ...;  // OK: Write access to the dense submatrix value.
      ... = *it;  // OK: Read access to the dense submatrix value.
   }

   // Traversing the elements of the 1st row via iterators to const elements
   for( SubmatrixType::ConstIterator it=sm.begin(1); it!=sm.end(1); ++it ) {
      *it = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = *it;  // OK: Read access to the dense submatrix value.
   }
   \endcode

   \code
   typedef blaze::CompressedMatrix<int,blaze::rowMajor>  MatrixType;
   typedef blaze::Submatrix<MatrixType>                  SubmatrixType;

   MatrixType A( 256UL, 512UL );
   // ... Resizing and initialization

   // Creating a reference to a specific submatrix of the sparse matrix A
   SubmatrixType sm = submatrix( A, 16UL, 16UL, 64UL, 128UL );

   // Traversing the elements of the 0th row via iterators to non-const elements
   for( SubmatrixType::Iterator it=sm.begin(0); it!=sm.end(0); ++it ) {
      it->value() = ...;  // OK: Write access to the value of the non-zero element.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }

   // Traversing the elements of the 1st row via iterators to const elements
   for( SubmatrixType::ConstIterator it=sm.begin(1); it!=sm.end(1); ++it ) {
      it->value() = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }
   \endcode

// \n \section views_submatrices_element_insertion Element Insertion
// <hr>
//
// Inserting/accessing elements in a sparse submatrix can be done by several alternative functions.
// The following example demonstrates all options:

   \code
   typedef blaze::CompressedMatrix<double,blaze::rowMajor>  MatrixType;
   MatrixType A( 256UL, 512UL );  // Non-initialized matrix of size 256x512

   typedef blaze::Submatrix<MatrixType>  SubmatrixType;
   SubmatrixType sm = submatrix( A, 10UL, 10UL, 16UL, 16UL );  // View on a 16x16 submatrix of A

   // The function call operator provides access to all possible elements of the sparse submatrix,
   // including the zero elements. In case the subscript operator is used to access an element
   // that is currently not stored in the sparse submatrix, the element is inserted into the
   // submatrix.
   sm(2,4) = 2.0;

   // The second operation for inserting elements is the set() function. In case the element is
   // not contained in the submatrix it is inserted into the submatrix, if it is already contained
   // in the submatrix its value is modified.
   sm.set( 2UL, 5UL, -1.2 );

   // An alternative for inserting elements into the submatrix is the insert() function. However,
   // it inserts the element only in case the element is not already contained in the submatrix.
   sm.insert( 2UL, 6UL, 3.7 );

   // Just as in case of sparse matrices, elements can also be inserted via the append() function.
   // In case of submatrices, append() also requires that the appended element's index is strictly
   // larger than the currently largest non-zero index in the according row or column of the
   // submatrix and that the according row's or column's capacity is large enough to hold the new
   // element. Note however that due to the nature of a submatrix, which may be an alias to the
   // middle of a sparse matrix, the append() function does not work as efficiently for a
   // submatrix as it does for a matrix.
   sm.reserve( 2UL, 10UL );
   sm.append( 2UL, 10UL, -2.1 );
   \endcode

// \n \section views_submatrices_arithmetic_operations Arithmetic Operations
// <hr>
//
// Both dense and sparse submatrices can be used in all arithmetic operations that any other dense
// or sparse matrix can be used in. The following example gives an impression of the use of dense
// submatrices within arithmetic operations. All operations (addition, subtraction, multiplication,
// scaling, ...) can be performed on all possible combinations of dense and sparse matrices with
// fitting element types:

   \code
   typedef blaze::DynamicMatrix<double,blaze::rowMajor>     DenseMatrixType;
   typedef blaze::CompressedMatrix<double,blaze::rowMajor>  SparseMatrixType;
   DenseMatrixType D1, D2, D3;
   SparseMatrixType S1, S2;

   typedef blaze::CompressedVector<double,blaze::columnVector>  SparseVectorType;
   SparseVectorType a, b;

   // ... Resizing and initialization

   typedef Submatrix<DenseMatrixType>  SubmatrixType;
   SubmatrixType sm = submatrix( D1, 0UL, 0UL, 8UL, 8UL );  // View on the 8x8 submatrix of matrix D1
                                                            // starting from row 0 and column 0

   submatrix( D1, 0UL, 8UL, 8UL, 8UL ) = D2;  // Dense matrix initialization of the 8x8 submatrix
                                              // starting in row 0 and column 8
   sm = S1;                                   // Sparse matrix initialization of the second 8x8 submatrix

   D3 = sm + D2;                                    // Dense matrix/dense matrix addition
   S2 = S1  - submatrix( D1, 8UL, 0UL, 8UL, 8UL );  // Sparse matrix/dense matrix subtraction
   D2 = sm * submatrix( D1, 8UL, 8UL, 8UL, 8UL );   // Dense matrix/dense matrix multiplication

   submatrix( D1, 8UL, 0UL, 8UL, 8UL ) *= 2.0;      // In-place scaling of a submatrix of D1
   D2 = submatrix( D1, 8UL, 8UL, 8UL, 8UL ) * 2.0;  // Scaling of the a submatrix of D1
   D2 = 2.0 * sm;                                   // Scaling of the a submatrix of D1

   submatrix( D1, 0UL, 8UL, 8UL, 8UL ) += D2;  // Addition assignment
   submatrix( D1, 8UL, 0UL, 8UL, 8UL ) -= S1;  // Subtraction assignment
   submatrix( D1, 8UL, 8UL, 8UL, 8UL ) *= sm;  // Multiplication assignment

   a = submatrix( D1, 4UL, 4UL, 8UL, 8UL ) * b;  // Dense matrix/sparse vector multiplication
   \endcode

// \n \section views_aligned_submatrices Aligned Submatrices
// <hr>
//
// Usually submatrices can be defined anywhere within a matrix. They may start at any position and
// may have an arbitrary extension (only restricted by the extension of the underlying matrix).
// However, in contrast to matrices themselves, which are always properly aligned in memory and
// therefore can provide maximum performance, this means that submatrices in general have to be
// considered to be unaligned. This can be made explicit by the blaze::unaligned flag:

   \code
   using blaze::unaligned;

   typedef blaze::DynamicMatrix<double,blaze::rowMajor>  DenseMatrixType;

   DenseMatrixType A;
   // ... Resizing and initialization

   // Identical creations of an unaligned submatrix of size 8x8, starting in row 0 and column 0
   blaze::Submatrix<DenseMatrixType>           sm1 = submatrix           ( A, 0UL, 0UL, 8UL, 8UL );
   blaze::Submatrix<DenseMatrixType>           sm2 = submatrix<unaligned>( A, 0UL, 0UL, 8UL, 8UL );
   blaze::Submatrix<DenseMatrixType,unaligned> sm3 = submatrix           ( A, 0UL, 0UL, 8UL, 8UL );
   blaze::Submatrix<DenseMatrixType,unaligned> sm4 = submatrix<unaligned>( A, 0UL, 0UL, 8UL, 8UL );
   \endcode

// All of these calls to the \c submatrix() function are identical. Whether the alignment flag is
// explicitly specified or not, it always returns an unaligned submatrix. Whereas this may provide
// full flexibility in the creation of submatrices, this might result in performance disadvantages
// in comparison to matrix primitives (even in case the specified submatrix could be aligned).
// Whereas matrix primitives are guaranteed to be properly aligned and therefore provide maximum
// performance in all operations, a general view on a matrix might not be properly aligned. This
// may cause a performance penalty on some platforms and/or for some operations.
//
// However, it is also possible to create aligned submatrices. Aligned submatrices are identical to
// unaligned submatrices in all aspects, except that they may pose additional alignment restrictions
// and therefore have less flexibility during creation, but don't suffer from performance penalties
// and provide the same performance as the underlying matrix. Aligned submatrices are created by
// explicitly specifying the blaze::aligned flag:

   \code
   using blaze::aligned;

   // Creating an aligned submatrix of size 8x8, starting in row 0 and column 0
   blaze::Submatrix<DenseMatrixType,aligned> sv = submatrix<aligned>( A, 0UL, 0UL, 8UL, 8UL );
   \endcode

// The alignment restrictions refer to system dependent address restrictions for the used element
// type and the available vectorization mode (SSE, AVX, ...). In order to be properly aligned the
// first element of each row/column of the submatrix must be aligned. The following source code
// gives some examples for a double precision row-major dynamic matrix, assuming that padding is
// enabled and that AVX is available, which packs 4 \c double values into a SIMD vector:

   \code
   using blaze::aligned;
   using blaze::rowMajor;

   typedef blaze::DynamicMatrix<double,rowMajor>  MatrixType;
   typedef blaze::Submatrix<MatrixType,aligned>   SubmatrixType;

   MatrixType D( 13UL, 17UL );
   // ... Resizing and initialization

   // OK: Starts at position (0,0), i.e. the first element of each row is aligned (due to padding)
   SubmatrixType dsm1 = submatrix<aligned>( D, 0UL, 0UL, 7UL, 11UL );

   // OK: First column is a multiple of 4, i.e. the first element of each row is aligned (due to padding)
   SubmatrixType dsm2 = submatrix<aligned>( D, 3UL, 12UL, 8UL, 16UL );

   // OK: First column is a multiple of 4 and the submatrix includes the last row and column
   SubmatrixType dsm3 = submatrix<aligned>( D, 4UL, 0UL, 9UL, 17UL );

   // Error: First column is not a multiple of 4, i.e. the first element is not aligned
   SubmatrixType dsm4 = submatrix<aligned>( D, 2UL, 3UL, 12UL, 12UL );
   \endcode

// Note that the discussed alignment restrictions are only valid for aligned dense submatrices.
// In contrast, aligned sparse submatrices at this time don't pose any additional restrictions.
// Therefore aligned and unaligned sparse submatrices are truly fully identical. Still, in case
// the blaze::aligned flag is specified during setup, an aligned submatrix is created:

   \code
   using blaze::aligned;

   typedef blaze::CompressedMatrix<double,blaze::rowMajor>  SparseMatrixType;

   SparseMatrixType A;
   // ... Resizing and initialization

   // Creating an aligned submatrix of size 8x8, starting in row 0 and column 0
   blaze::Submatrix<SparseMatrixType,aligned> sv = submatrix<aligned>( A, 0UL, 0UL, 8UL, 8UL );
   \endcode

// \n \section views_submatrices_on_submatrices Submatrices on Submatrices
// <hr>
//
// It is also possible to create a submatrix view on another submatrix. In this context it is
// important to remember that the type returned by the \c submatrix() function is the same type
// as the type of the given submatrix, since the view on a submatrix is just another view on the
// underlying matrix:

   \code
   typedef blaze::DynamicMatrix<double,blaze::rowMajor>  MatrixType;
   typedef blaze::Submatrix<MatrixType>                  SubmatrixType;

   MatrixType D1;

   // ... Resizing and initialization

   // Creating a submatrix view on the dense matrix D1
   SubmatrixType sm1 = submatrix( D1, 4UL, 4UL, 8UL, 16UL );

   // Creating a submatrix view on the dense submatrix sm1
   SubmatrixType sm2 = submatrix( sm1, 1UL, 1UL, 4UL, 8UL );
   \endcode

// \n \section views_submatrices_on_symmetric_matrices Submatrices on Symmetric Matrices
//
// Submatrices can also be created on symmetric matrices (see the \c SymmetricMatrix class template):

   \code
   using blaze::DynamicMatrix;
   using blaze::SymmetricMatrix;
   using blaze::Submatrix;

   typedef SymmetricMatrix< DynamicMatrix<int> >  SymmetricDynamicType;
   typedef Submatrix< SymmetricDynamicType >      SubmatrixType;

   // Setup of a 16x16 symmetric matrix
   SymmetricDynamicType A( 16UL );

   // Creating a dense submatrix of size 8x12, starting in row 2 and column 4
   SubmatrixType sm = submatrix( A, 2UL, 4UL, 8UL, 12UL );
   \endcode

// It is important to note, however, that (compound) assignments to such submatrices have a
// special restriction: The symmetry of the underlying symmetric matrix must not be broken!
// Since the modification of element \f$ a_{ij} \f$ of a symmetric matrix also modifies the
// element \f$ a_{ji} \f$, the matrix to be assigned must be structured such that the symmetry
// of the symmetric matrix is preserved. Otherwise a \c std::invalid_argument exception is
// thrown:

   \code
   using blaze::DynamicMatrix;
   using blaze::SymmetricMatrix;

   // Setup of two default 4x4 symmetric matrices
   SymmetricMatrix< DynamicMatrix<int> > A1( 4 ), A2( 4 );

   // Setup of the 3x2 dynamic matrix
   //
   //       ( 1 2 )
   //   B = ( 3 4 )
   //       ( 5 6 )
   //
   DynamicMatrix<int> B{ { 1, 2 }, { 3, 4 }, { 5, 6 } };

   // OK: Assigning B to a submatrix of A1 such that the symmetry can be preserved
   //
   //        ( 0 0 1 2 )
   //   A1 = ( 0 0 3 4 )
   //        ( 1 3 5 6 )
   //        ( 2 4 6 0 )
   //
   submatrix( A1, 0UL, 2UL, 3UL, 2UL ) = B;  // OK

   // Error: Assigning B to a submatrix of A2 such that the symmetry cannot be preserved!
   //   The elements marked with X cannot be assigned unambiguously!
   //
   //        ( 0 1 2 0 )
   //   A2 = ( 1 3 X 0 )
   //        ( 2 X 6 0 )
   //        ( 0 0 0 0 )
   //
   submatrix( A2, 0UL, 1UL, 3UL, 2UL ) = B;  // Assignment throws an exception!
   \endcode

// \n Previous: \ref views_subvectors &nbsp; &nbsp; Next: \ref views_rows
*/
//*************************************************************************************************


//**Rows*******************************************************************************************
/*!\page views_rows Rows
//
// \tableofcontents
//
//
// Rows provide views on a specific row of a dense or sparse matrix. As such, rows act as a
// reference to a specific row. This reference is valid and can be used in every way any other
// row vector can be used as long as the matrix containing the row is not resized or entirely
// destroyed. The row also acts as an alias to the row elements: Changes made to the elements
// (e.g. modifying values, inserting or erasing elements) are immediately visible in the matrix
// and changes made via the matrix are immediately visible in the row.
//
//
// \n \section views_rows_class The Row Class Template
// <hr>
//
// The blaze::Row class template represents a reference to a specific row of a dense or sparse
// matrix primitive. It can be included via the header file

   \code
   #include <blaze/math/Row.h>
   \endcode

// The type of the matrix is specified via template parameter:

   \code
   template< typename MT >
   class Row;
   \endcode

// \c MT specifies the type of the matrix primitive. Row can be used with every matrix primitive,
// but does not work with any matrix expression type.
//
//
// \n \section views_rows_setup Setup of Rows
// <hr>
//
// A reference to a dense or sparse row can be created very conveniently via the \c row() function.
// This reference can be treated as any other row vector, i.e. it can be assigned to, it can be
// copied from, and it can be used in arithmetic operations. The reference can also be used on
// both sides of an assignment: The row can either be used as an alias to grant write access to a
// specific row of a matrix primitive on the left-hand side of an assignment or to grant read-access
// to a specific row of a matrix primitive or expression on the right-hand side of an assignment.
// The following two examples demonstrate this for dense and sparse matrices:

   \code
   typedef blaze::DynamicVector<double,rowVector>     DenseVectorType;
   typedef blaze::CompressedVector<double,rowVector>  SparseVectorType;
   typedef blaze::DynamicMatrix<double,rowMajor>      DenseMatrixType;
   typedef blaze::CompressedMatrix<double,rowMajor>   SparseMatrixType;

   DenseVectorType  x;
   SparseVectorType y;
   DenseMatrixType  A, B;
   SparseMatrixType C, D;
   // ... Resizing and initialization

   // Setting the 2nd row of matrix A to x
   blaze::Row<DenseMatrixType> row2 = row( A, 2UL );
   row2 = x;

   // Setting the 3rd row of matrix B to y
   row( B, 3UL ) = y;

   // Setting x to the 4th row of the result of the matrix multiplication
   x = row( A * B, 4UL );

   // Setting y to the 2nd row of the result of the sparse matrix multiplication
   y = row( C * D, 2UL );
   \endcode

// The \c row() function can be used on any dense or sparse matrix, including expressions, as
// illustrated by the source code example. However, rows cannot be instantiated for expression
// types, but only for matrix primitives, respectively, i.e. for matrix types that offer write
// access.
//
//
// \n \section views_rows_common_operations Common Operations
// <hr>
//
// A row view can be used like any other row vector. For instance, the current number of elements
// can be obtained via the \c size() function, the current capacity via the \c capacity() function,
// and the number of non-zero elements via the \c nonZeros() function. However, since rows are
// references to specific rows of a matrix, several operations are not possible on views, such
// as resizing and swapping. The following example shows this by means of a dense row view:

   \code
   typedef blaze::DynamicMatrix<int,rowMajor>  MatrixType;
   typedef blaze::Row<MatrixType>              RowType;

   MatrixType A( 42UL, 42UL );
   // ... Resizing and initialization

   // Creating a reference to the 2nd row of matrix A
   RowType row2 = row( A, 2UL );

   row2.size();          // Returns the number of elements in the row
   row2.capacity();      // Returns the capacity of the row
   row2.nonZeros();      // Returns the number of non-zero elements contained in the row

   row2.resize( 84UL );  // Compilation error: Cannot resize a single row of a matrix

   RowType row3 = row( A, 3UL );
   swap( row2, row3 );   // Compilation error: Swap operation not allowed
   \endcode

// \n \section views_rows_element_access Element Access
// <hr>
//
// The elements of the row can be directly accessed with the subscript operator. The numbering
// of the row elements is

                             \f[\left(\begin{array}{*{5}{c}}
                             0 & 1 & 2 & \cdots & N-1 \\
                             \end{array}\right),\f]

// where N is the number of columns of the referenced matrix. Alternatively, the elements of
// a row can be traversed via iterators. Just as with vectors, in case of non-const rows,
// \c begin() and \c end() return an Iterator, which allows a manipulation of the non-zero
// value, in case of a constant row a ConstIterator is returned:

   \code
   typedef blaze::DynamicMatrix<int,rowMajor>  MatrixType;
   typedef blaze::Row<MatrixType>              RowType;

   MatrixType A( 128UL, 256UL );
   // ... Resizing and initialization

   // Creating a reference to the 31st row of matrix A
   RowType row31 = row( A, 31UL );

   for( RowType::Iterator it=row31.begin(); it!=row31.end(); ++it ) {
      *it = ...;  // OK; Write access to the dense row value
      ... = *it;  // OK: Read access to the dense row value.
   }

   for( RowType::ConstIterator it=row31.begin(); it!=row31.end(); ++it ) {
      *it = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = *it;  // OK: Read access to the dense row value.
   }
   \endcode

   \code
   typedef blaze::CompressedMatrix<int,rowMajor>  MatrixType;
   typedef blaze::Row<MatrixType>                 RowType;

   MatrixType A( 128UL, 256UL );
   // ... Resizing and initialization

   // Creating a reference to the 31st row of matrix A
   RowType row31 = row( A, 31UL );

   for( RowType::Iterator it=row31.begin(); it!=row31.end(); ++it ) {
      it->value() = ...;  // OK: Write access to the value of the non-zero element.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }

   for( RowType::Iterator it=row31.begin(); it!=row31.end(); ++it ) {
      it->value() = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }
   \endcode

// \n \section views_rows_element_insertion Element Insertion
// <hr>
//
// Inserting/accessing elements in a sparse row can be done by several alternative functions.
// The following example demonstrates all options:

   \code
   typedef blaze::CompressedMatrix<double,blaze::rowMajor>  MatrixType;
   MatrixType A( 10UL, 100UL );  // Non-initialized 10x100 matrix

   typedef blaze::Row<MatrixType>  RowType;
   RowType row0( row( A, 0UL ) );  // Reference to the 0th row of A

   // The subscript operator provides access to all possible elements of the sparse row,
   // including the zero elements. In case the subscript operator is used to access an element
   // that is currently not stored in the sparse row, the element is inserted into the row.
   row0[42] = 2.0;

   // The second operation for inserting elements is the set() function. In case the element
   // is not contained in the row it is inserted into the row, if it is already contained in
   // the row its value is modified.
   row0.set( 45UL, -1.2 );

   // An alternative for inserting elements into the row is the insert() function. However,
   // it inserts the element only in case the element is not already contained in the row.
   row0.insert( 50UL, 3.7 );

   // A very efficient way to add new elements to a sparse row is the append() function.
   // Note that append() requires that the appended element's index is strictly larger than
   // the currently largest non-zero index of the row and that the row's capacity is large
   // enough to hold the new element.
   row0.reserve( 10UL );
   row0.append( 51UL, -2.1 );
   \endcode

// \n \section views_rows_arithmetic_operations Arithmetic Operations
// <hr>
//
// Both dense and sparse rows can be used in all arithmetic operations that any other dense or
// sparse row vector can be used in. The following example gives an impression of the use of
// dense rows within arithmetic operations. All operations (addition, subtraction, multiplication,
// scaling, ...) can be performed on all possible combinations of dense and sparse rows with
// fitting element types:

   \code
   blaze::DynamicVector<double,blaze::rowVector> a( 2UL, 2.0 ), b;
   blaze::CompressedVector<double,blaze::rowVector> c( 2UL );
   c[1] = 3.0;

   typedef blaze::DynamicMatrix<double,blaze::rowMajor>  DenseMatrix;
   DenseMatrix A( 4UL, 2UL );  // Non-initialized 4x2 matrix

   typedef blaze::Row<DenseMatrix>  RowType;
   RowType row0( row( A, 0UL ) );  // Reference to the 0th row of A

   row0[0] = 0.0;        // Manual initialization of the 0th row of A
   row0[1] = 0.0;
   row( A, 1UL ) = 1.0;  // Homogeneous initialization of the 1st row of A
   row( A, 2UL ) = a;    // Dense vector initialization of the 2nd row of A
   row( A, 3UL ) = c;    // Sparse vector initialization of the 3rd row of A

   b = row0 + a;              // Dense vector/dense vector addition
   b = c + row( A, 1UL );     // Sparse vector/dense vector addition
   b = row0 * row( A, 2UL );  // Component-wise vector multiplication

   row( A, 1UL ) *= 2.0;     // In-place scaling of the 1st row
   b = row( A, 1UL ) * 2.0;  // Scaling of the 1st row
   b = 2.0 * row( A, 1UL );  // Scaling of the 1st row

   row( A, 2UL ) += a;              // Addition assignment
   row( A, 2UL ) -= c;              // Subtraction assignment
   row( A, 2UL ) *= row( A, 0UL );  // Multiplication assignment

   double scalar = row( A, 1UL ) * trans( c );  // Scalar/dot/inner product between two vectors

   A = trans( c ) * row( A, 1UL );  // Outer product between two vectors
   \endcode

// \n \section views_rows_non_fitting_storage_order Views on Matrices with Non-Fitting Storage Order
// <hr>
//
// Especially noteworthy is that row views can be created for both row-major and column-major
// matrices. Whereas the interface of a row-major matrix only allows to traverse a row directly
// and the interface of a column-major matrix only allows to traverse a column, via views it is
// possible to traverse a row of a column-major matrix or a column of a row-major matrix. For
// instance:

   \code
   typedef blaze::CompressedMatrix<int,columnMajor>  MatrixType;
   typedef blaze::Row<MatrixType>                    RowType;

   MatrixType A( 64UL, 32UL );
   // ... Resizing and initialization

   // Creating a reference to the 31st row of a column-major matrix A
   RowType row1 = row( A, 1UL );

   for( RowType::Iterator it=row1.begin(); it!=row1.end(); ++it ) {
      // ...
   }
   \endcode

// However, please note that creating a row view on a matrix stored in a column-major fashion
// can result in a considerable performance decrease in comparison to a view on a matrix with
// a fitting storage orientation. This is due to the non-contiguous storage of the matrix
// elements. Therefore care has to be taken in the choice of the most suitable storage order:

   \code
   // Setup of two column-major matrices
   CompressedMatrix<double,columnMajor> A( 128UL, 128UL );
   CompressedMatrix<double,columnMajor> B( 128UL, 128UL );
   // ... Resizing and initialization

   // The computation of the 15th row of the multiplication between A and B ...
   CompressedVector<double,rowVector> x = row( A * B, 15UL );

   // ... is essentially the same as the following computation, which multiplies
   // the 15th row of the column-major matrix A with B.
   CompressedVector<double,rowVector> x = row( A, 15UL ) * B;
   \endcode

// Although \b Blaze performs the resulting vector/matrix multiplication as efficiently as possible
// using a row-major storage order for matrix A would result in a more efficient evaluation.
//
// \n Previous: \ref views_submatrices &nbsp; &nbsp; Next: \ref views_columns
*/
//*************************************************************************************************


//**Columns****************************************************************************************
/*!\page views_columns Columns
//
// \tableofcontents
//
//
// Just as rows provide a view on a specific row of a matrix, columns provide views on a specific
// column of a dense or sparse matrix. As such, columns act as a reference to a specific column.
// This reference is valid an can be used in every way any other column vector can be used as long
// as the matrix containing the column is not resized or entirely destroyed. Changes made to the
// elements (e.g. modifying values, inserting or erasing elements) are immediately visible in the
// matrix and changes made via the matrix are immediately visible in the column.
//
//
// \n \section views_columns_class The Column Class Template
// <hr>
//
// The blaze::Column class template represents a reference to a specific column of a dense or
// sparse matrix primitive. It can be included via the header file

   \code
   #include <blaze/math/Column.h>
   \endcode

// The type of the matrix is specified via template parameter:

   \code
   template< typename MT >
   class Column;
   \endcode

// \c MT specifies the type of the matrix primitive. Column can be used with every matrix
// primitive, but does not work with any matrix expression type.
//
//
// \n \section views_colums_setup Setup of Columns
// <hr>
//
// Similar to the setup of a row, a reference to a dense or sparse column can be created very
// conveniently via the \c column() function. This reference can be treated as any other column
// vector, i.e. it can be assigned to, copied from, and be used in arithmetic operations. The
// column can either be used as an alias to grant write access to a specific column of a matrix
// primitive on the left-hand side of an assignment or to grant read-access to a specific column
// of a matrix primitive or expression on the right-hand side of an assignment. The following
// two examples demonstrate this for dense and sparse matrices:

   \code
   typedef blaze::DynamicVector<double,columnVector>     DenseVectorType;
   typedef blaze::CompressedVector<double,columnVector>  SparseVectorType;
   typedef blaze::DynamicMatrix<double,columnMajor>      DenseMatrixType;
   typedef blaze::CompressedMatrix<double,columnMajor>   SparseMatrixType;

   DenseVectorType  x;
   SparseVectorType y;
   DenseMatrixType  A, B;
   SparseMatrixType C, D;
   // ... Resizing and initialization

   // Setting the 1st column of matrix A to x
   blaze::Column<DenseMatrixType> col1 = column( A, 1UL );
   col1 = x;

   // Setting the 4th column of matrix B to y
   column( B, 4UL ) = y;

   // Setting x to the 2nd column of the result of the matrix multiplication
   x = column( A * B, 2UL );

   // Setting y to the 2nd column of the result of the sparse matrix multiplication
   y = column( C * D, 2UL );
   \endcode

// The \c column() function can be used on any dense or sparse matrix, including expressions, as
// illustrated by the source code example. However, columns cannot be instantiated for expression
// types, but only for matrix primitives, respectively, i.e. for matrix types that offer write
// access.
//
//
// \n \section views_columns_common_operations Common Operations
// <hr>
//
// A column view can be used like any other column vector. For instance, the current number of
// elements can be obtained via the \c size() function, the current capacity via the \c capacity()
// function, and the number of non-zero elements via the \c nonZeros() function. However, since
// columns are references to specific columns of a matrix, several operations are not possible on
// views, such as resizing and swapping. The following example shows this by means of a dense
// column view:

   \code
   typedef blaze::DynamicMatrix<int,columnMajor>  MatrixType;
   typedef blaze::Column<MatrixType>              ColumnType;

   MatrixType A( 42UL, 42UL );
   // ... Resizing and initialization

   // Creating a reference to the 2nd column of matrix A
   ColumnType col2 = column( A, 2UL );

   col2.size();          // Returns the number of elements in the column
   col2.capacity();      // Returns the capacity of the column
   col2.nonZeros();      // Returns the number of non-zero elements contained in the column

   col2.resize( 84UL );  // Compilation error: Cannot resize a single column of a matrix

   ColumnType col3 = column( A, 3UL );
   swap( col2, col3 );   // Compilation error: Swap operation not allowed
   \endcode

// \n \section views_columns_element_access Element Access
// <hr>
//
// The elements of the column can be directly accessed with the subscript operator. The numbering
// of the column elements is

                             \f[\left(\begin{array}{*{5}{c}}
                             0 & 1 & 2 & \cdots & N-1 \\
                             \end{array}\right),\f]

// where N is the number of rows of the referenced matrix. Alternatively, the elements of
// a column can be traversed via iterators. Just as with vectors, in case of non-const columns,
// \c begin() and \c end() return an Iterator, which allows a manipulation of the non-zero
// value, in case of a constant column a ConstIterator is returned:

   \code
   typedef blaze::DynamicMatrix<int,columnMajor>  MatrixType;
   typedef blaze::Column<MatrixType>              ColumnType;

   MatrixType A( 128UL, 256UL );
   // ... Resizing and initialization

   // Creating a reference to the 31st column of matrix A
   ColumnType col31 = column( A, 31UL );

   for( ColumnType::Iterator it=col31.begin(); it!=col31.end(); ++it ) {
      *it = ...;  // OK; Write access to the dense column value
      ... = *it;  // OK: Read access to the dense column value.
   }

   for( ColumnType::ConstIterator it=col31.begin(); it!=col31.end(); ++it ) {
      *it = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = *it;  // OK: Read access to the dense column value.
   }
   \endcode

   \code
   typedef blaze::CompressedMatrix<int,columnMajor>  MatrixType;
   typedef blaze::Column<MatrixType>                 ColumnType;

   MatrixType A( 128UL, 256UL );
   // ... Resizing and initialization

   // Creating a reference to the 31st column of matrix A
   ColumnType col31 = column( A, 31UL );

   for( ColumnType::Iterator it=col31.begin(); it!=col31.end(); ++it ) {
      it->value() = ...;  // OK: Write access to the value of the non-zero element.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }

   for( ColumnType::Iterator it=col31.begin(); it!=col31.end(); ++it ) {
      it->value() = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }
   \endcode

// \n \section views_columns_element_insertion Element Insertion
// <hr>
//
// Inserting/accessing elements in a sparse column can be done by several alternative functions.
// The following example demonstrates all options:

   \code
   typedef blaze::CompressedMatrix<double,blaze::columnMajor>  MatrixType;
   MatrixType A( 100UL, 10UL );  // Non-initialized 10x100 matrix

   typedef blaze::Column<MatrixType>  ColumnType;
   ColumnType col0( column( A, 0UL ) );  // Reference to the 0th column of A

   // The subscript operator provides access to all possible elements of the sparse column,
   // including the zero elements. In case the subscript operator is used to access an element
   // that is currently not stored in the sparse column, the element is inserted into the column.
   col0[42] = 2.0;

   // The second operation for inserting elements is the set() function. In case the element
   // is not contained in the column it is inserted into the column, if it is already contained
   // in the column its value is modified.
   col0.set( 45UL, -1.2 );

   // An alternative for inserting elements into the column is the insert() function. However,
   // it inserts the element only in case the element is not already contained in the column.
   col0.insert( 50UL, 3.7 );

   // A very efficient way to add new elements to a sparse column is the append() function.
   // Note that append() requires that the appended element's index is strictly larger than
   // the currently largest non-zero index of the column and that the column's capacity is
   // large enough to hold the new element.
   col0.reserve( 10UL );
   col0.append( 51UL, -2.1 );
   \endcode

// \n \section views_columns_arithmetic_operations Arithmetic Operations
// <hr>
//
// Both dense and sparse columns can be used in all arithmetic operations that any other dense or
// sparse column vector can be used in. The following example gives an impression of the use of
// dense columns within arithmetic operations. All operations (addition, subtraction, multiplication,
// scaling, ...) can be performed on all possible combinations of dense and sparse columns with
// fitting element types:

   \code
   blaze::DynamicVector<double,blaze::columnVector> a( 2UL, 2.0 ), b;
   blaze::CompressedVector<double,blaze::columnVector> c( 2UL );
   c[1] = 3.0;

   typedef blaze::DynamicMatrix<double,blaze::columnMajor>  MatrixType;
   MatrixType A( 2UL, 4UL );  // Non-initialized 2x4 matrix

   typedef blaze::Column<DenseMatrix>  ColumnType;
   ColumnType col0( column( A, 0UL ) );  // Reference to the 0th column of A

   col0[0] = 0.0;           // Manual initialization of the 0th column of A
   col0[1] = 0.0;
   column( A, 1UL ) = 1.0;  // Homogeneous initialization of the 1st column of A
   column( A, 2UL ) = a;    // Dense vector initialization of the 2nd column of A
   column( A, 3UL ) = c;    // Sparse vector initialization of the 3rd column of A

   b = col0 + a;                 // Dense vector/dense vector addition
   b = c + column( A, 1UL );     // Sparse vector/dense vector addition
   b = col0 * column( A, 2UL );  // Component-wise vector multiplication

   column( A, 1UL ) *= 2.0;     // In-place scaling of the 1st column
   b = column( A, 1UL ) * 2.0;  // Scaling of the 1st column
   b = 2.0 * column( A, 1UL );  // Scaling of the 1st column

   column( A, 2UL ) += a;                 // Addition assignment
   column( A, 2UL ) -= c;                 // Subtraction assignment
   column( A, 2UL ) *= column( A, 0UL );  // Multiplication assignment

   double scalar = trans( c ) * column( A, 1UL );  // Scalar/dot/inner product between two vectors

   A = column( A, 1UL ) * trans( c );  // Outer product between two vectors
   \endcode

// \n \section views_columns_non_fitting_storage_order Views on Matrices with Non-Fitting Storage Order
// <hr>
//
// Especially noteworthy is that column views can be created for both row-major and column-major
// matrices. Whereas the interface of a row-major matrix only allows to traverse a row directly
// and the interface of a column-major matrix only allows to traverse a column, via views it is
// possible to traverse a row of a column-major matrix or a column of a row-major matrix. For
// instance:

   \code
   typedef blaze::CompressedMatrix<int,rowMajor>  MatrixType;
   typedef blaze::Column<MatrixType>              ColumnType;

   MatrixType A( 64UL, 32UL );
   // ... Resizing and initialization

   // Creating a reference to the 31st column of a row-major matrix A
   ColumnType col1 = column( A, 1UL );

   for( ColumnType::Iterator it=col1.begin(); it!=col1.end(); ++it ) {
      // ...
   }
   \endcode

// However, please note that creating a column view on a matrix stored in a row-major fashion
// can result in a considerable performance decrease in comparison to a view on a matrix with
// a fitting storage orientation. This is due to the non-contiguous storage of the matrix
// elements. Therefore care has to be taken in the choice of the most suitable storage order:

   \code
   // Setup of two row-major matrices
   CompressedMatrix<double,rowMajor> A( 128UL, 128UL );
   CompressedMatrix<double,rowMajor> B( 128UL, 128UL );
   // ... Resizing and initialization

   // The computation of the 15th column of the multiplication between A and B ...
   CompressedVector<double,columnVector> x = column( A * B, 15UL );

   // ... is essentially the same as the following computation, which multiplies
   // the 15th column of the row-major matrix B with A.
   CompressedVector<double,columnVector> x = A * column( B, 15UL );
   \endcode

// Although \b Blaze performs the resulting matrix/vector multiplication as efficiently as possible
// using a column-major storage order for matrix B would result in a more efficient evaluation.
//
// \n Previous: \ref views_rows &nbsp; &nbsp; Next: \ref arithmetic_operations
*/
//*************************************************************************************************


//**Arithmetic Operations**************************************************************************
/*!\page arithmetic_operations Arithmetic Operations
//
// \tableofcontents
//
//
// \b Blaze provides the following arithmetic operations for vectors and matrices:
//
// <ul>
//    <li> \ref addition </li>
//    <li> \ref subtraction </li>
//    <li> \ref scalar_multiplication </li>
//    <li> \ref vector_vector_multiplication
//       <ul>
//          <li> \ref componentwise_multiplication </li>
//          <li> \ref inner_product </li>
//          <li> \ref outer_product </li>
//          <li> \ref cross_product </li>
//       </ul>
//    </li>
//    <li> \ref vector_vector_division </li>
//    <li> \ref matrix_vector_multiplication </li>
//    <li> \ref matrix_matrix_multiplication </li>
// </ul>
//
// \n Previous: \ref views_columns &nbsp; &nbsp; Next: \ref addition
*/
//*************************************************************************************************


//**Addition***************************************************************************************
/*!\page addition Addition
//
// The addition of vectors and matrices is as intuitive as the addition of scalar values. For both
// the vector addition as well as the matrix addition the addition operator can be used. It even
// enables the addition of dense and sparse vectors as well as the addition of dense and sparse
// matrices:

   \code
   blaze::DynamicVector<int>      v1( 5UL ), v3;
   blaze::CompressedVector<float> v2( 5UL );

   // ... Initializing the vectors

   v3 = v1 + v2;  // Addition of a two column vectors of different data type
   \endcode

   \code
   blaze::DynamicMatrix<float,rowMajor>        M1( 7UL, 3UL );
   blaze::CompressedMatrix<size_t,columnMajor> M2( 7UL, 3UL ), M3;

   // ... Initializing the matrices

   M3 = M1 + M2;  // Addition of a row-major and a column-major matrix of different data type
   \endcode

// Note that it is necessary that both operands have exactly the same dimensions. Violating this
// precondition results in an exception. Also note that in case of vectors it is only possible to
// add vectors with the same transpose flag:

   \code
   blaze::DynamicVector<int,columnVector>   v1( 5UL );
   blaze::CompressedVector<float,rowVector> v2( 5UL );

   v1 + v2;           // Compilation error: Cannot add a column vector and a row vector
   v1 + trans( v2 );  // OK: Addition of two column vectors
   \endcode

// In case of matrices, however, it is possible to add row-major and column-major matrices. Note
// however that in favor of performance the addition of two matrices with the same storage order
// is favorable. The same argument holds for the element type: In case two vectors or matrices
// with the same element type are added, the performance can be much higher due to vectorization
// of the operation.

   \code
   blaze::DynamicVector<double>v1( 100UL ), v2( 100UL ), v3;

   // ... Initialization of the vectors

   v3 = v1 + v2;  // Vectorized addition of two double precision vectors
   \endcode

   \code
   blaze::DynamicMatrix<float> M1( 50UL, 70UL ), M2( 50UL, 70UL ), M3;

   // ... Initialization of the matrices

   M3 = M1 + M2;  // Vectorized addition of two row-major, single precision dense matrices
   \endcode

// \n Previous: \ref arithmetic_operations &nbsp; &nbsp; Next: \ref subtraction
*/
//*************************************************************************************************


//**Subtraction************************************************************************************
/*!\page subtraction Subtraction
//
// The subtraction of vectors and matrices works exactly as intuitive as the addition, but with
// the subtraction operator. For both the vector subtraction as well as the matrix subtraction
// the subtraction operator can be used. It also enables the subtraction of dense and sparse
// vectors as well as the subtraction of dense and sparse matrices:

   \code
   blaze::DynamicVector<int>      v1( 5UL ), v3;
   blaze::CompressedVector<float> v2( 5UL );

   // ... Initializing the vectors

   v3 = v1 - v2;  // Subtraction of a two column vectors of different data type


   blaze::DynamicMatrix<float,rowMajor>        M1( 7UL, 3UL );
   blaze::CompressedMatrix<size_t,columnMajor> M2( 7UL, 3UL ), M3;

   // ... Initializing the matrices

   M3 = M1 - M2;  // Subtraction of a row-major and a column-major matrix of different data type
   \endcode

// Note that it is necessary that both operands have exactly the same dimensions. Violating this
// precondition results in an exception. Also note that in case of vectors it is only possible to
// subtract vectors with the same transpose flag:

   \code
   blaze::DynamicVector<int,columnVector>   v1( 5UL );
   blaze::CompressedVector<float,rowVector> v2( 5UL );

   v1 - v2;           // Compilation error: Cannot subtract a row vector from a column vector
   v1 - trans( v2 );  // OK: Subtraction of two column vectors
   \endcode

// In case of matrices, however, it is possible to subtract row-major and column-major matrices.
// Note however that in favor of performance the subtraction of two matrices with the same storage
// order is favorable. The same argument holds for the element type: In case two vectors or matrices
// with the same element type are added, the performance can be much higher due to vectorization
// of the operation.

   \code
   blaze::DynamicVector<double>v1( 100UL ), v2( 100UL ), v3;

   // ... Initialization of the vectors

   v3 = v1 - v2;  // Vectorized subtraction of two double precision vectors


   blaze::DynamicMatrix<float> M1( 50UL, 70UL ), M2( 50UL, 70UL ), M3;

   // ... Initialization of the matrices

   M3 = M1 - M2;  // Vectorized subtraction of two row-major, single precision dense matrices
   \endcode

// \n Previous: \ref addition &nbsp; &nbsp; Next: \ref scalar_multiplication
*/
//*************************************************************************************************


//**Scalar Multiplication**************************************************************************
/*!\page scalar_multiplication Scalar Multiplication
//
// The scalar multiplication is the multiplication of a scalar value with a vector or a matrix.
// In \b Blaze it is possible to use all built-in/fundamental data types except bool as scalar
// values. Additionally, it is possible to use std::complex values with the same built-in data
// types as element type.

   \code
   blaze::StaticVector<int,3UL> v1{ 1, 2, 3 };

   blaze::DynamicVector<double>   v2 = v1 * 1.2;
   blaze::CompressedVector<float> v3 = -0.3F * v1;
   \endcode

   \code
   blaze::StaticMatrix<int,3UL,2UL> M1{ { 1, 2 }, { 3, 4 }, { 5, 6 } };

   blaze::DynamicMatrix<double>   M2 = M1 * 1.2;
   blaze::CompressedMatrix<float> M3 = -0.3F * M1;
   \endcode

// Vectors and matrices cannot be used for as scalar value for scalar multiplications (see the
// following example). However, each vector and matrix provides the \c scale() function, which
// can be used to scale a vector or matrix element-wise with arbitrary scalar data types:

   \code
   blaze::CompressedMatrix< blaze::StaticMatrix<int,3UL,3UL> > M1;
   blaze::StaticMatrix<int,3UL,3UL> scalar;

   M1 * scalar;  // No scalar multiplication, but matrix/matrix multiplication

   M1.scale( scalar );  // Scalar multiplication
   \endcode

// \n Previous: \ref subtraction &nbsp; &nbsp; Next: \ref componentwise_multiplication
*/
//*************************************************************************************************


//**Vector/Vector Multiplication*******************************************************************
/*!\page vector_vector_multiplication Vector/Vector Multiplication
//
// \n \section componentwise_multiplication Componentwise Multiplication
// <hr>
//
// Multiplying two vectors with the same transpose flag (i.e. either blaze::columnVector or
// blaze::rowVector) via the multiplication operator results in a componentwise multiplication
// of the two vectors:

   \code
   using blaze::DynamicVector;
   using blaze::CompressedVector;

   CompressedVector<int,columnVector> v1( 17UL );
   DynamicVector<int,columnVector>    v2( 17UL );

   StaticVector<double,10UL,rowVector> v3;
   DynamicVector<double,rowVector>     v4( 10UL );

   // ... Initialization of the vectors

   CompressedVector<int,columnVector> v5( v1 * v2 );  // Componentwise multiplication of a sparse and
                                                      // a dense column vector. The result is a sparse
                                                      // column vector.
   DynamicVector<double,rowVector>    v6( v3 * v4 );  // Componentwise multiplication of two dense row
                                                      // vectors. The result is a dense row vector.
   \endcode

// \n \section inner_product Inner Product / Scalar Product / Dot Product
// <hr>
//
// The multiplication between a row vector and a column vector results in an inner product between
// the two vectors:

   \code
   blaze::StaticVector<int,3UL,rowVector> v1{  2, 5, -1 };
   blaze::DynamicVector<int,columnVector> v2{ -1, 3, -2 };

   int result = v1 * v2;  // Results in the value 15
   \endcode

// The \c trans() function can be used to transpose a vector as necessary:

   \code
   blaze::StaticVector<int,3UL,rowVector> v1{  2, 5, -1 };
   blaze::StaticVector<int,3UL,rowVector> v2{ -1, 3, -2 };

   int result = v1 * trans( v2 );  // Also results in the value 15
   \endcode

// Alternatively, either the \c dot() function or the comma operator can be used for any combination
// of vectors (row or column vectors) to perform an inner product:

   \code
   blaze::StaticVector<int,3UL,rowVector> v1{  2, 5, -1 };
   blaze::StaticVector<int,3UL,rowVector> v2{ -1, 3, -2 };

   int result = dot( v1, v2 );  // Inner product between two row vectors
   \endcode

   \code
   blaze::StaticVector<int,3UL,columnVector> v1{  2, 5, -1 };
   blaze::StaticVector<int,3UL,columnVector> v2{ -1, 3, -2 };

   int result = (v1,v2);  // Inner product between two column vectors
   \endcode

// When using the comma operator, please note the brackets embracing the inner product expression.
// Due to the low precedence of the comma operator (lower even than the assignment operator) these
// brackets are strictly required for a correct evaluation of the inner product.
//
//
// \n \section outer_product Outer Product
// <hr>
//
// The multiplication between a column vector and a row vector results in the outer product of
// the two vectors:

   \code
   blaze::StaticVector<int,3UL,columnVector> v1{ 2, 5, -1 };
   blaze::DynamicVector<int,rowVector> v2{ -1, 3, -2 };

   StaticMatrix<int,3UL,3UL> M1 = v1 * v2;
   \endcode

// The \c trans() function can be used to transpose a vector as necessary:

   \code
   blaze::StaticVector<int,3UL,rowVector> v1{  2, 5, -1 };
   blaze::StaticVector<int,3UL,rowVector> v2{ -1, 3, -2 };

   int result = trans( v1 ) * v2;
   \endcode

// Alternatively, the \c outer() function can be used for any combination of vectors (row or column
// vectors) to perform an outer product:

   \code
   blaze::StaticVector<int,3UL,rowVector> v1{  2, 5, -1 };
   blaze::StaticVector<int,3UL,rowVector> v2{ -1, 3, -2 };

   StaticMatrix<int,3UL,3UL> M1 = outer( v1, v2 );  // Outer product between two row vectors
   \endcode

// \n \section cross_product Cross Product
// <hr>
//
// Two vectors with the same transpose flag can be multiplied via the cross product. The cross
// product between two vectors \f$ a \f$ and \f$ b \f$ is defined as

   \f[
   \left(\begin{array}{*{1}{c}}
   c_0 \\
   c_1 \\
   c_2 \\
   \end{array}\right)
   =
   \left(\begin{array}{*{1}{c}}
   a_1 b_2 - a_2 b_1 \\
   a_2 b_0 - a_0 b_2 \\
   a_0 b_1 - a_1 b_0 \\
   \end{array}\right).
   \f]

// Due to the absence of a \f$ \times \f$ operator in the C++ language, the cross product is
// realized via the \c cross() function. Alternatively, the modulo operator (i.e. \c operator%)
// can be used in case infix notation is required:

   \code
   blaze::StaticVector<int,3UL,columnVector> v1{  2, 5, -1 };
   blaze::DynamicVector<int,columnVector>    v2{ -1, 3, -2 };

   blaze::StaticVector<int,3UL,columnVector> v3( cross( v1, v2 ) );
   blaze::StaticVector<int,3UL,columnVector> v4( v1 % v2 );
   \endcode

// Please note that the cross product is restricted to three dimensional (dense and sparse)
// column vectors.
//
// \n Previous: \ref scalar_multiplication &nbsp; &nbsp; Next: \ref vector_vector_division
*/
//*************************************************************************************************


//**Vector/Vector Division*************************************************************************
/*!\page vector_vector_division Vector/Vector Division
//
// \n \section componentwise_division Componentwise Division
// <hr>
//
// Dividing a vector by a dense vector with the same transpose flag (i.e. either blaze::columnVector
// or blaze::rowVector) via the division operator results in a componentwise division:

   \code
   using blaze::DynamicVector;
   using blaze::CompressedVector;

   CompressedVector<int,columnVector> v1( 17UL );
   DynamicVector<int,columnVector>    v2( 17UL );

   StaticVector<double,10UL,rowVector> v3;
   DynamicVector<double,rowVector>     v4( 10UL );

   // ... Initialization of the vectors

   CompressedVector<int,columnVector> v5( v1 / v2 );  // Componentwise division of a sparse and a
                                                      // dense column vector. The result is a sparse
                                                      // column vector.
   DynamicVector<double,rowVector>    v6( v3 / v4 );  // Componentwise division of two dense row
                                                      // vectors. The result is a dense row vector.
   \endcode

// Note that all values of the divisor must be non-zero and that no checks are performed to assert
// this precondition!
//
// \n Previous: \ref vector_vector_multiplication &nbsp; &nbsp; Next: \ref matrix_vector_multiplication
*/
//*************************************************************************************************


//**Matrix/Vector Multiplication*******************************************************************
/*!\page matrix_vector_multiplication Matrix/Vector Multiplication
//
// In \b Blaze matrix/vector multiplications can be as intuitively formulated as in mathematical
// textbooks. Just as in textbooks there are two different multiplications between a matrix and
// a vector: a matrix/column vector multiplication and a row vector/matrix multiplication:

   \code
   using blaze::StaticVector;
   using blaze::DynamicVector;
   using blaze::DynamicMatrix;

   DynamicMatrix<int>                  M1( 39UL, 12UL );
   StaticVector<int,12UL,columnVector> v1;

   // ... Initialization of the matrix and the vector

   DynamicVector<int,columnVector> v2 = M1 * v1;           // Matrix/column vector multiplication
   DynamicVector<int,rowVector>    v3 = trans( v1 ) * M1;  // Row vector/matrix multiplication
   \endcode

// Note that the storage order of the matrix poses no restrictions on the operation. Also note,
// that the highest performance for a multiplication between a dense matrix and a dense vector can
// be achieved if both the matrix and the vector have the same scalar element type.
//
// \n Previous: \ref vector_vector_division &nbsp; &nbsp; Next: \ref matrix_matrix_multiplication
*/
//*************************************************************************************************


//**Matrix/Matrix Multiplication*******************************************************************
/*!\page matrix_matrix_multiplication Matrix/Matrix Multiplication
//
// The matrix/matrix multiplication can be formulated exactly as in mathematical textbooks:

   \code
   using blaze::DynamicMatrix;
   using blaze::CompressedMatrix;

   DynamicMatrix<double>   M1( 45UL, 85UL );
   CompressedMatrix<float> M2( 85UL, 37UL );

   // ... Initialization of the matrices

   DynamicMatrix<double> M3 = M1 * M2;
   \endcode

// The storage order of the two matrices poses no restrictions on the operation, all variations
// are possible. Note however that the highest performance for a multiplication between two dense
// matrices can be expected for two matrices with the same scalar element type.
//
// \n Previous: \ref matrix_vector_multiplication &nbsp; &nbsp; Next: \ref custom_operations
*/
//*************************************************************************************************


//**Custom Operations******************************************************************************
/*!\page custom_operations Custom Operations
//
// In addition to the provided operations on vectors and matrices it is possible to define custom
// operations. For this purpose, \b Blaze provides the \c forEach() function, which allows to pass
// the required operation via functor or lambda:

   \code
   blaze::DynamicMatrix<double> A, B;

   B = forEach( A, []( double d ){ return std::sqrt( d ); } );
   \endcode

// This example demonstrates the most convenient way of defining a custom operation by passing a
// lambda to the \c forEach() function. The lambda is executed on each single element of a dense
// vector or matrix or each non-zero element of a sparse vector or matrix.
//
// Alternatively, it is possible to pass a custom functor:

   \code
   struct Sqrt
   {
      double operator()( double a ) const
      {
         return std::sqrt( a );
      }
   };

   B = forEach( A, Sqrt() );
   \endcode

// In order for the functor to work in a call to \c forEach() it must define a function call
// operator, which accepts arguments of the type of the according vector or matrix elements.
//
// Although the operation is automatically parallelized depending on the size of the vector or
// matrix, no automatic vectorization is possible. In order to enable vectorization, a \c load()
// function can be added to the functor, which handles the vectorized computation. Depending on
// the data type this function is passed one of the following \b Blaze SIMD data types:
//
// <ul>
//    <li>SIMD data types for fundamental data types
//       <ul>
//          <li>\c blaze::SIMDint8: Packed SIMD type for 8-bit signed integral data types</li>
//          <li>\c blaze::SIMDuint8: Packed SIMD type for 8-bit unsigned integral data types</li>
//          <li>\c blaze::SIMDint16: Packed SIMD type for 16-bit signed integral data types</li>
//          <li>\c blaze::SIMDuint16: Packed SIMD type for 16-bit unsigned integral data types</li>
//          <li>\c blaze::SIMDint32: Packed SIMD type for 32-bit signed integral data types</li>
//          <li>\c blaze::SIMDuint32: Packed SIMD type for 32-bit unsigned integral data types</li>
//          <li>\c blaze::SIMDint64: Packed SIMD type for 64-bit signed integral data types</li>
//          <li>\c blaze::SIMDuint64: Packed SIMD type for 64-bit unsigned integral data types</li>
//          <li>\c blaze::SIMDfloat: Packed SIMD type for single precision floating point data</li>
//          <li>\c blaze::SIMDdouble: Packed SIMD type for double precision floating point data</li>
//       </ul>
//    </li>
//    <li>SIMD data types for complex data types
//       <ul>
//          <li>\c blaze::cint8: Packed SIMD type for complex 8-bit signed integral data types</li>
//          <li>\c blaze::cuint8: Packed SIMD type for complex 8-bit unsigned integral data types</li>
//          <li>\c blaze::cint16: Packed SIMD type for complex 16-bit signed integral data types</li>
//          <li>\c blaze::cuint16: Packed SIMD type for complex 16-bit unsigned integral data types</li>
//          <li>\c blaze::cint32: Packed SIMD type for complex 32-bit signed integral data types</li>
//          <li>\c blaze::cuint32: Packed SIMD type for complex 32-bit unsigned integral data types</li>
//          <li>\c blaze::cint64: Packed SIMD type for complex 64-bit signed integral data types</li>
//          <li>\c blaze::cuint64: Packed SIMD type for complex 64-bit unsigned integral data types</li>
//          <li>\c blaze::cfloat: Packed SIMD type for complex single precision floating point data</li>
//          <li>\c blaze::cdouble: Packed SIMD type for complex double precision floating point data</li>
//       </ul>
//    </li>
// </ul>
//
// All SIMD types provide the \c value data member for a direct access to the underlying intrinsic
// data element. In the following example, this intrinsic element is passed to the AVX function
// \c _mm256_sqrt_pd():

   \code
   struct Sqrt
   {
      double operator()( double a ) const
      {
         return std::sqrt( a );
      }

      simd_double_t load( simd_double_t a ) const
      {
         return _mm256_sqrt_pd( a.value );
      }
   };
   \endcode

// In this example, whenever vectorization is generally applicable, the \c load() function is
// called instead of the function call operator for as long as the number of remaining elements
// is larger-or-equal to the width of the packed SIMD type. In all other cases (which also
// includes peel-off and remainder loops) the scalar operation is used.
//
// Please note that this example has two drawbacks: First, it will only compile in case the
// intrinsic \c _mm256_sqrt_pd() function is available (i.e. when AVX is active). Second, the
// availability of AVX is not taken into account. The first drawback can be alleviated by making
// the \c load() function a function template. The second drawback can be dealt with by adding a
// \c simdEnabled() function template to the functor:

   \code
   struct Sqrt
   {
      double operator()( double a ) const
      {
         return std::sqrt( a );
      }

      template< typename T >
      T load( T a ) const
      {
         return _mm256_sqrt_pd( a.value );
      }

      template< typename T >
      static constexpr bool simdEnabled() {
#if defined(__AVX__)
         return true;
#else
         return false;
#endif
      }
   };
   \endcode

// The \c simdEnabled() function must be a \c static, \c constexpr function and must return whether
// or not vectorization is available for the given data type \c T. In case the function returns
// \c true, the \c load() function is used for a vectorized evaluation, in case the function
// returns \c false, \c load() is not called.
//
// Note that this is a simplified example that is only working when used for dense vectors and
// matrices with double precision floating point elements. The following code shows the complete
// implementation of the according functor that is used within the \b Blaze library. The \b Blaze
// \c Sqrt functor is working for all data types that are providing a square root operation:

   \code
   namespace blaze {

   struct Sqrt
   {
      template< typename T >
      BLAZE_ALWAYS_INLINE auto operator()( const T& a ) const -> decltype( sqrt( a ) )
      {
         return sqrt( a );
      }

      template< typename T >
      static constexpr bool simdEnabled() { return HasSIMDSqrt<T>::value; }

      template< typename T >
      BLAZE_ALWAYS_INLINE auto load( const T& a ) const -> decltype( sqrt( a ) )
      {
         BLAZE_CONSTRAINT_MUST_BE_SIMD_TYPE( T );
         return sqrt( a );
      }
   };

   } // namespace blaze
   \endcode

// For more information on the available \b Blaze SIMD data types and functions, please see the
// SIMD module in the complete \b Blaze documentation.
//
// \n Previous: \ref matrix_matrix_multiplication &nbsp; &nbsp; Next: \ref shared_memory_parallelization
*/
//*************************************************************************************************


//**Shared Memory Parallelization******************************************************************
/*!\page shared_memory_parallelization Shared Memory Parallelization
//
// One of the main motivations of the \b Blaze 1.x releases was to achieve maximum performance
// on a single CPU core for all possible operations. However, today's CPUs are not single core
// anymore, but provide several (homogeneous or heterogeneous) compute cores. In order to fully
// exploit the performance potential of a multicore CPU, computations have to be parallelized
// across all available cores of a CPU. For this purpose, \b Blaze provides three different
// shared memory parallelization techniques:
//
//  - \ref openmp_parallelization
//  - \ref cpp_threads_parallelization
//  - \ref boost_threads_parallelization
//
// In addition, \b Blaze provides means to enforce the serial execution of specific operations:
//
//  - \ref serial_execution
//
// \n Previous: \ref custom_operations &nbsp; &nbsp; Next: \ref openmp_parallelization
*/
//*************************************************************************************************


//**OpenMP Parallelization*************************************************************************
/*!\page openmp_parallelization OpenMP Parallelization
//
// \tableofcontents
//
//
// \n \section openmp_setup OpenMP Setup
// <hr>
//
// To enable the OpenMP-based parallelization, all that needs to be done is to explicitly specify
// the use of OpenMP on the command line:

   \code
   -fopenmp   // GNU C++ compiler
   -openmp    // Intel C++ compiler
   /openmp    // Visual Studio
   \endcode

// This simple action will cause the \b Blaze library to automatically try to run all operations
// in parallel with the specified number of threads.
//
// As common for OpenMP, the number of threads can be specified either via an environment variable

   \code
   export OMP_NUM_THREADS=4  // Unix systems
   set OMP_NUM_THREADS=4     // Windows systems
   \endcode

// or via an explicit call to the \c omp_set_num_threads() function:

   \code
   omp_set_num_threads( 4 );
   \endcode

// Alternatively, the number of threads can also be specified via the \c setNumThreads() function
// provided by the \b Blaze library:

   \code
   blaze::setNumThreads( 4 );
   \endcode

// Please note that the \b Blaze library does not limit the available number of threads. Therefore
// it is in YOUR responsibility to choose an appropriate number of threads. The best performance,
// though, can be expected if the specified number of threads matches the available number of
// cores.
//
// In order to query the number of threads used for the parallelization of operations, the
// \c getNumThreads() function can be used:

   \code
   const size_t threads = blaze::getNumThreads();
   \endcode

// In the context of OpenMP, the function returns the maximum number of threads OpenMP will use
// within a parallel region and is therefore equivalent to the \c omp_get_max_threads() function.
//
//
// \n \section openmp_configuration OpenMP Configuration
// <hr>
//
// Note that \b Blaze is not unconditionally running an operation in parallel. In case \b Blaze
// deems the parallel execution as counterproductive for the overall performance, the operation
// is executed serially. One of the main reasons for not executing an operation in parallel is
// the size of the operands. For instance, a vector addition is only executed in parallel if the
// size of both vector operands exceeds a certain threshold. Otherwise, the performance could
// seriously decrease due to the overhead caused by the thread setup. However, in order to be
// able to adjust the \b Blaze library to a specific system, it is possible to configure these
// thresholds manually. All shared memory thresholds are contained within the configuration file
// <tt>./blaze/config/Thresholds.h</tt>.
//
// Please note that these thresholds are highly sensitiv to the used system architecture and
// the shared memory parallelization technique (see also \ref cpp_threads_parallelization and
// \ref boost_threads_parallelization). Therefore the default values cannot guarantee maximum
// performance for all possible situations and configurations. They merely provide a reasonable
// standard for the current CPU generation.
//
//
// \n \section openmp_first_touch First Touch Policy
// <hr>
//
// So far the \b Blaze library does not (yet) automatically initialize dynamic memory according
// to the first touch principle. Consider for instance the following vector triad example:

   \code
   using blaze::columnVector;

   const size_t N( 1000000UL );

   blaze::DynamicVector<double,columnVector> a( N ), b( N ), c( N ), d( N );

   // Initialization of the vectors b, c, and d
   for( size_t i=0UL; i<N; ++i ) {
      b[i] = rand<double>();
      c[i] = rand<double>();
      d[i] = rand<double>();
   }

   // Performing a vector triad
   a = b + c * d;
   \endcode

// If this code, which is prototypical for many OpenMP applications that have not been optimized
// for ccNUMA architectures, is run across several locality domains (LD), it will not scale
// beyond the maximum performance achievable on a single LD if the working set does not fit into
// the cache. This is because the initialization loop is executed by a single thread, writing to
// \c b, \c c, and \c d for the first time. Hence, all memory pages belonging to those arrays will
// be mapped into a single LD.
//
// As mentioned above, this problem can be solved by performing vector initialization in parallel:

   \code
   // ...

   // Initialization of the vectors b, c, and d
   #pragma omp parallel for
   for( size_t i=0UL; i<N; ++i ) {
      b[i] = rand<double>();
      c[i] = rand<double>();
      d[i] = rand<double>();
   }

   // ...
   \endcode

// This simple modification makes a huge difference on ccNUMA in memory-bound situations (as for
// instance in all BLAS level 1 operations and partially BLAS level 2 operations). Therefore, in
// order to achieve the maximum possible performance, it is imperative to initialize the memory
// according to the later use of the data structures.
//
//
// \n \section openmp_limitations Limitations of the OpenMP Parallelization
// <hr>
//
// There are a few important limitations to the current \b Blaze OpenMP parallelization. The first
// one involves the explicit use of an OpenMP parallel region (see \ref openmp_parallel), the
// other one the OpenMP \c sections directive (see \ref openmp_sections).
//
//
// \n \subsection openmp_parallel The Parallel Directive
//
// In OpenMP threads are explicitly spawned via the an OpenMP parallel directive:

   \code
   // Serial region, executed by a single thread

   #pragma omp parallel
   {
      // Parallel region, executed by the specified number of threads
   }

   // Serial region, executed by a single thread
   \endcode

// Conceptually, the specified number of threads (see \ref openmp_setup) is created every time a
// parallel directive is encountered. Therefore, from a performance point of view, it seems to be
// beneficial to use a single OpenMP parallel directive for several operations:

   \code
   blaze::DynamicVector<double> x, y1, y2;
   blaze::DynamicMatrix<double> A, B;

   #pragma omp parallel
   {
      y1 = A * x;
      y2 = B * x;
   }
   \endcode

// Unfortunately, this optimization approach is not allowed within the \b Blaze library. More
// explicitly, it is not allowed to put an operation into a parallel region. The reason is that
// the entire code contained within a parallel region is executed by all threads. Although this
// appears to just comprise the contained computations, a computation (or more specifically the
// assignment of an expression to a vector or matrix) can contain additional logic that must not
// be handled by multiple threads (as for instance memory allocations, setup of temporaries, etc.).
// Therefore it is not possible to manually start a parallel region for several operations, but
// \b Blaze will spawn threads automatically, depending on the specifics of the operation at hand
// and the given operands.
//
// \n \subsection openmp_sections The Sections Directive
//
// OpenMP provides several work-sharing construct to distribute work among threads. One of these
// constructs is the \c sections directive:

   \code
   blaze::DynamicVector<double> x, y1, y2;
   blaze::DynamicMatrix<double> A, B;

   // ... Resizing and initialization

   #pragma omp sections
   {
   #pragma omp section

      y1 = A * x;

   #pragma omp section

      y2 = B * x;

   }
   \endcode

// In this example, two threads are used to compute two distinct matrix/vector multiplications
// concurrently. Thereby each of the \c sections is executed by exactly one thread.
//
// Unfortunately \b Blaze does not support concurrent parallel computations and therefore this
// approach does not work with any of the \b Blaze parallelization techniques. All techniques
// (including the C++11 and Boost thread parallelizations; see \ref cpp_threads_parallelization
// and \ref boost_threads_parallelization) are optimized for the parallel computation of an
// operation within a single thread of execution. This means that \b Blaze tries to use all
// available threads to compute the result of a single operation as efficiently as possible.
// Therefore, for this special case, it is advisable to disable all \b Blaze parallelizations
// and to let \b Blaze compute all operations within a \c sections directive in serial. This can
// be done by either completely disabling the \b Blaze parallelization (see \ref serial_execution)
// or by selectively serializing all operations within a \c sections directive via the \c serial()
// function:

   \code
   blaze::DynamicVector<double> x, y1, y2;
   blaze::DynamicMatrix<double> A, B;

   // ... Resizing and initialization

   #pragma omp sections
   {
   #pragma omp section

      y1 = serial( A * x );

   #pragma omp section

      y2 = serial( B * x );

   }
   \endcode

// Please note that the use of the \c BLAZE_SERIAL_SECTION (see also \ref serial_execution) does
// NOT work in this context!
//
// \n Previous: \ref shared_memory_parallelization &nbsp; &nbsp; Next: \ref cpp_threads_parallelization
*/
//*************************************************************************************************


//**C++11 Thread Parallelization*******************************************************************
/*!\page cpp_threads_parallelization C++11 Thread Parallelization
//
// \tableofcontents
//
//
// In addition to the OpenMP-based shared memory parallelization, starting with \b Blaze 2.1,
// \b Blaze also provides a shared memory parallelization based on C++11 threads.
//
//
// \n \section cpp_threads_setup C++11 Thread Setup
// <hr>
//
// In order to enable the C++11 thread-based parallelization, first the according C++11-specific
// compiler flags have to be used and second the \c BLAZE_USE_CPP_THREADS command line argument
// has to be explicitly specified. For instance, in case of the GNU C++ and Clang compilers the
// compiler flags have to be extended by

   \code
   ... -std=c++11 -DBLAZE_USE_CPP_THREADS ...
   \endcode

// This simple action will cause the \b Blaze library to automatically try to run all operations
// in parallel with the specified number of C++11 threads. Note that in case both OpenMP and C++11
// threads are enabled on the command line, the OpenMP-based parallelization has priority and
// is preferred.
//
// The number of threads can be either specified via the environment variable \c BLAZE_NUM_THREADS

   \code
   export BLAZE_NUM_THREADS=4  // Unix systems
   set BLAZE_NUM_THREADS=4     // Windows systems
   \endcode

// or alternatively via the \c setNumThreads() function provided by the \b Blaze library:

   \code
   blaze::setNumThreads( 4 );
   \endcode

// Please note that the \b Blaze library does not limit the available number of threads. Therefore
// it is in YOUR responsibility to choose an appropriate number of threads. The best performance,
// though, can be expected if the specified number of threads matches the available number of
// cores.
//
// In order to query the number of threads used for the parallelization of operations, the
// \c getNumThreads() function can be used:

   \code
   const size_t threads = blaze::getNumThreads();
   \endcode

// In the context of C++11 threads, the function will return the previously specified number of
// threads.
//
//
// \n \section cpp_threads_configuration C++11 Thread Configuration
// <hr>
//
// As in case of the OpenMP-based parallelization \b Blaze is not unconditionally running an
// operation in parallel. In case \b Blaze deems the parallel execution as counterproductive for
// the overall performance, the operation is executed serially. One of the main reasons for not
// executing an operation in parallel is the size of the operands. For instance, a vector addition
// is only executed in parallel if the size of both vector operands exceeds a certain threshold.
// Otherwise, the performance could seriously decrease due to the overhead caused by the thread
// setup. However, in order to be able to adjust the \b Blaze library to a specific system, it
// is possible to configure these thresholds manually. All thresholds are contained within the
// configuration file <tt>./blaze/config/Thresholds.h</tt>.
//
// Please note that these thresholds are highly sensitiv to the used system architecture and
// the shared memory parallelization technique. Therefore the default values cannot guarantee
// maximum performance for all possible situations and configurations. They merely provide a
// reasonable standard for the current CPU generation. Also note that the provided defaults
// have been determined using the OpenMP parallelization and require individual adaption for
// the C++11 thread parallelization.
//
//
// \n \section cpp_threads_known_issues Known Issues
// <hr>
//
// There is a known issue in Visual Studio 2012 and 2013 that may cause C++11 threads to hang
// if their destructor is executed after the \c main() function:
//
//    http://connect.microsoft.com/VisualStudio/feedback/details/747145
//
// Unfortunately, the C++11 parallelization of the \b Blaze library is affected from this bug.
// In order to circumvent this problem, \b Blaze provides the \c shutDownThreads() function,
// which can be used to manually destroy all threads at the end of the \c main() function:

   \code
   int main()
   {
      // ... Using the C++11 thread parallelization of Blaze

      shutDownThreads();
   }
   \endcode

// Please note that this function may only be used at the end of the \c main() function. After
// this function no further computation may be executed! Also note that this function has an
// effect for Visual Studio compilers only and doesn't need to be used with any other compiler.
//
// \n Previous: \ref openmp_parallelization &nbsp; &nbsp; Next: \ref boost_threads_parallelization
*/
//*************************************************************************************************


//**Boost Thread Parallelization*******************************************************************
/*!\page boost_threads_parallelization Boost Thread Parallelization
//
// \tableofcontents
//
//
// The third available shared memory parallelization provided with \b Blaze is based on Boost
// threads.
//
//
// \n \section boost_threads_setup Boost Thread Setup
// <hr>
//
// In order to enable the Boost thread-based parallelization, two steps have to be taken: First,
// the \c BLAZE_USE_BOOST_THREADS command line argument has to be explicitly specified during
// compilation:

   \code
   ... -DBLAZE_USE_BOOST_THREADS ...
   \endcode

// Second, the according Boost libraries have to be linked. These two simple actions will cause
// the \b Blaze library to automatically try to run all operations in parallel with the specified
// number of Boost threads. Note that the OpenMP-based and C++11 thread-based parallelizations
// have priority, i.e. are preferred in case either is enabled in combination with the Boost
// thread parallelization.
//
// The number of threads can be either specified via the environment variable \c BLAZE_NUM_THREADS

   \code
   export BLAZE_NUM_THREADS=4  // Unix systems
   set BLAZE_NUM_THREADS=4     // Windows systems
   \endcode

// or alternatively via the \c setNumThreads() function provided by the \b Blaze library:

   \code
   blaze::setNumThreads( 4 );
   \endcode

// Please note that the \b Blaze library does not limit the available number of threads. Therefore
// it is in YOUR responsibility to choose an appropriate number of threads. The best performance,
// though, can be expected if the specified number of threads matches the available number of
// cores.
//
// In order to query the number of threads used for the parallelization of operations, the
// \c getNumThreads() function can be used:

   \code
   const size_t threads = blaze::getNumThreads();
   \endcode

// In the context of Boost threads, the function will return the previously specified number of
// threads.
//
//
// \n \section boost_threads_configuration Boost Thread Configuration
// <hr>
//
// As in case of the other shared memory parallelizations \b Blaze is not unconditionally running
// an operation in parallel (see \ref openmp_parallelization or \ref cpp_threads_parallelization).
// All thresholds related to the Boost thread parallelization are also contained within the
// configuration file <tt>./blaze/config/Thresholds.h</tt>.
//
// Please note that these thresholds are highly sensitiv to the used system architecture and
// the shared memory parallelization technique. Therefore the default values cannot guarantee
// maximum performance for all possible situations and configurations. They merely provide a
// reasonable standard for the current CPU generation. Also note that the provided defaults
// have been determined using the OpenMP parallelization and require individual adaption for
// the Boost thread parallelization.
//
// \n Previous: \ref cpp_threads_parallelization &nbsp; &nbsp; Next: \ref serial_execution
*/
//*************************************************************************************************


//**Serial Execution*******************************************************************************
/*!\page serial_execution Serial Execution
//
// Sometimes it may be necessary to enforce the serial execution of specific operations. For this
// purpose, the \b Blaze library offers three possible options: the serialization of a single
// expression via the \c serial() function, the serialization of a block of expressions via the
// \c BLAZE_SERIAL_SECTION, and the general deactivation of the parallel execution.
//
//
// \n \section serial_execution_serial_expression Option 1: Serialization of a Single Expression
// <hr>
//
// The first option is the serialization of a specific operation via the \c serial() function:

   \code
   blaze::DynamicMatrix<double> A, B, C;
   // ... Resizing and initialization
   C = serial( A + B );
   \endcode

// \c serial() enforces the serial evaluation of the enclosed expression. It can be used on any
// kind of dense or sparse vector or matrix expression.
//
//
// \n \section serial_execution_serial_section Option 2: Serialization of Multiple Expressions
// <hr>
//
// The second option is the temporary and local enforcement of a serial execution via the
// \c BLAZE_SERIAL_SECTION:

   \code
   using blaze::rowMajor;
   using blaze::columnVector;

   blaze::DynamicMatrix<double,rowMajor> A;
   blaze::DynamicVector<double,columnVector> b, c, d, x, y, z;

   // ... Resizing and initialization

   // Parallel execution
   // If possible and beneficial for performance the following operation is executed in parallel.
   x = A * b;

   // Serial execution
   // All operations executed within the serial section are guaranteed to be executed in
   // serial (even if a parallel execution would be possible and/or beneficial).
   BLAZE_SERIAL_SECTION
   {
      y = A * c;
      z = A * d;
   }

   // Parallel execution continued
   // ...
   \endcode

// Within the scope of the \c BLAZE_SERIAL_SECTION, all operations are guaranteed to run in serial.
// Outside the scope of the serial section, all operations are run in parallel (if beneficial for
// the performance).
//
// Note that the \c BLAZE_SERIAL_SECTION must only be used within a single thread of execution.
// The use of the serial section within several concurrent threads will result undefined behavior!
//
//
// \n \section serial_execution_deactivate_parallelism Option 3: Deactivation of Parallel Execution
// <hr>
//
// The third option is the general deactivation of the parallel execution (even in case OpenMP is
// enabled on the command line). This can be achieved via the \c BLAZE_USE_SHARED_MEMORY_PARALLELIZATION
// switch in the <tt>./blaze/config/SMP.h</tt> configuration file:

   \code
   #define BLAZE_USE_SHARED_MEMORY_PARALLELIZATION 1
   \endcode

// In case the \c BLAZE_USE_SHARED_MEMORY_PARALLELIZATION switch is set to 0, the shared memory
// parallelization is deactivated altogether.
//
// \n Previous: \ref boost_threads_parallelization &nbsp; &nbsp; Next: \ref serialization
*/
//*************************************************************************************************


//**Serialization**********************************************************************************
/*!\page serialization Serialization
//
// Sometimes it is necessary to store vector and/or matrices on disk, for instance for storing
// results or for sharing specific setups with other people. The \b Blaze math serialization
// module provides the according functionality to create platform independent, portable, binary
// representations of vectors and matrices that can be used to store the \b Blaze data structures
// without loss of precision and to reliably transfer them from one machine to another.
//
// The following two pages explain how to serialize vectors and matrices:
//
//  - \ref vector_serialization
//  - \ref matrix_serialization
//
// \n Previous: \ref serial_execution &nbsp; &nbsp; Next: \ref vector_serialization
*/
//*************************************************************************************************


//**Vector Serialization***************************************************************************
/*!\page vector_serialization Vector Serialization
//
// The following example demonstrates the (de-)serialization of dense and sparse vectors:

   \code
   using blaze::columnVector;
   using blaze::rowVector;

   // Serialization of both vectors
   {
      blaze::StaticVector<double,5UL,rowVector> d;
      blaze::CompressedVector<int,columnVector> s;

      // ... Resizing and initialization

      // Creating an archive that writes into a the file "vectors.blaze"
      blaze::Archive<std::ofstream> archive( "vectors.blaze" );

      // Serialization of both vectors into the same archive. Note that d lies before s!
      archive << d << s;
   }

   // Reconstitution of both vectors
   {
      blaze::DynamicVector<double,rowVector> d1;
      blaze::DynamicVector<int,rowVector> d2;

      // Creating an archive that reads from the file "vectors.blaze"
      blaze::Archive<std::ifstream> archive( "vectors.blaze" );

      // Reconstituting the former d vector into d1. Note that it is possible to reconstitute
      // the vector into a differrent kind of vector (StaticVector -> DynamicVector), but that
      // the type of elements has to be the same.
      archive >> d1;

      // Reconstituting the former s vector into d2. Note that is is even possible to reconstitute
      // a sparse vector as a dense vector (also the reverse is possible) and that a column vector
      // can be reconstituted as row vector (and vice versa). Note however that also in this case
      // the type of elements is the same!
      archive >> d2
   }
   \endcode

// The (de-)serialization of vectors is not restricted to vectors of built-in data type, but can
// also be used for vectors with vector or matrix element type:

   \code
   // Serialization
   {
      blaze::CompressedVector< blaze::DynamicVector< blaze::complex<double> > > vec;

      // ... Resizing and initialization

      // Creating an archive that writes into a the file "vector.blaze"
      blaze::Archive<std::ofstream> archive( "vector.blaze" );

      // Serialization of the vector into the archive
      archive << vec;
   }

   // Deserialization
   {
      blaze::CompressedVector< blaze::DynamicVector< blaze::complex<double> > > vec;

      // Creating an archive that reads from the file "vector.blaze"
      blaze::Archive<std::ifstream> archive( "vector.blaze" );

      // Reconstitution of the vector from the archive
      archive >> vec;
   }
   \endcode

// As the examples demonstrates, the vector serialization offers an enormous flexibility. However,
// several actions result in errors:
//
//  - vectors cannot be reconstituted as matrices (and vice versa)
//  - the element type of the serialized and reconstituted vector must match, which means
//    that on the source and destination platform the general type (signed/unsigned integral
//    or floating point) and the size of the type must be exactly the same
//  - when reconstituting a \c StaticVector, its size must match the size of the serialized vector
//
// In case an error is encountered during (de-)serialization, a \c std::runtime_exception is
// thrown.
//
// \n Previous: \ref serialization &nbsp; &nbsp; Next: \ref matrix_serialization
*/
//*************************************************************************************************


//**Matrix Serialization***************************************************************************
/*!\page matrix_serialization Matrix Serialization
//
// The serialization of matrices works in the same manner as the serialization of vectors. The
// following example demonstrates the (de-)serialization of dense and sparse matrices:

   \code
   using blaze::rowMajor;
   using blaze::columnMajor;

   // Serialization of both matrices
   {
      blaze::StaticMatrix<double,3UL,5UL,rowMajor> D;
      blaze::CompressedMatrix<int,columnMajor> S;

      // ... Resizing and initialization

      // Creating an archive that writes into a the file "matrices.blaze"
      blaze::Archive<std::ofstream> archive( "matrices.blaze" );

      // Serialization of both matrices into the same archive. Note that D lies before S!
      archive << D << S;
   }

   // Reconstitution of both matrices
   {
      blaze::DynamicMatrix<double,rowMajor> D1;
      blaze::DynamicMatrix<int,rowMajor> D2;

      // Creating an archive that reads from the file "matrices.blaze"
      blaze::Archive<std::ifstream> archive( "matrices.blaze" );

      // Reconstituting the former D matrix into D1. Note that it is possible to reconstitute
      // the matrix into a differrent kind of matrix (StaticMatrix -> DynamicMatrix), but that
      // the type of elements has to be the same.
      archive >> D1;

      // Reconstituting the former S matrix into D2. Note that is is even possible to reconstitute
      // a sparse matrix as a dense matrix (also the reverse is possible) and that a column-major
      // matrix can be reconstituted as row-major matrix (and vice versa). Note however that also
      // in this case the type of elements is the same!
      archive >> D2
   }
   \endcode

// Note that also in case of matrices it is possible to (de-)serialize matrices with vector or
// matrix elements:

   \code
   // Serialization
   {
      blaze::CompressedMatrix< blaze::DynamicMatrix< blaze::complex<double> > > mat;

      // ... Resizing and initialization

      // Creating an archive that writes into a the file "matrix.blaze"
      blaze::Archive<std::ofstream> archive( "matrix.blaze" );

      // Serialization of the matrix into the archive
      archive << mat;
   }

   // Deserialization
   {
      blaze::CompressedMatrix< blaze::DynamicMatrix< blaze::complex<double> > > mat;

      // Creating an archive that reads from the file "matrix.blaze"
      blaze::Archive<std::ifstream> archive( "matrix.blaze" );

      // Reconstitution of the matrix from the archive
      archive >> mat;
   }
   \endcode

// Note that just as the vector serialization, the matrix serialization is restricted by a
// few important rules:
//
//  - matrices cannot be reconstituted as vectors (and vice versa)
//  - the element type of the serialized and reconstituted matrix must match, which means
//    that on the source and destination platform the general type (signed/unsigned integral
//    or floating point) and the size of the type must be exactly the same
//  - when reconstituting a \c StaticMatrix, the number of rows and columns must match those
//    of the serialized matrix
//
// In case an error is encountered during (de-)serialization, a \c std::runtime_exception is
// thrown.
//
// \n Previous: \ref vector_serialization &nbsp; &nbsp; Next: \ref blas_functions \n
*/
//*************************************************************************************************


//**BLAS Functions*********************************************************************************
/*!\page blas_functions BLAS Functions
//
// \tableofcontents
//
//
// For vector/vector, matrix/vector and matrix/matrix multiplications with large dense matrices
// \b Blaze relies on the efficiency of BLAS libraries. For this purpose, \b Blaze implements
// several convenient C++ wrapper functions for several BLAS functions. The following sections
// give a complete overview of all available BLAS level 1, 2 and 3 functions.
//
//
// \n \section blas_level_1 BLAS Level 1
// <hr>
//
// \subsection blas_level_1_dot Dot Product (dot)
//
// The following wrapper functions provide a generic interface for the BLAS functions for the
// dot product of two dense vectors (\c sdot(), \c ddot(), \c cdotu_sub(), and \c zdotu_sub()):

   \code
   namespace blaze {

   float dot( const int n, const float* x, const int incX, const float* y, const int incY );

   double dot( const int n, const double* x, const int incX, const double* y, const int incY );

   complex<float> dot( const int n, const complex<float>* x, const int incX,
                       const complex<float>* y, const int incY );

   complex<double> dot( const int n, const complex<double>* x, const int incX,
                        const complex<double>* y, const int incY );

   template< typename VT1, bool TF1, typename VT2, bool TF2 >
   ElementType_<VT1> dot( const DenseVector<VT1,TF1>& x, const DenseVector<VT2,TF2>& y );

   } // namespace blaze
   \endcode

// \n \section blas_level_2 BLAS Level 2
// <hr>
//
// \subsection blas_level_2_gemv General Matrix/Vector Multiplication (gemv)
//
// The following wrapper functions provide a generic interface for the BLAS functions for the
// general matrix/vector multiplication (\c sgemv(), \c dgemv(), \c cgemv(), and \c zgemv()):

   \code
   namespace blaze {

   void gemv( CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, int m, int n, float alpha,
              const float* A, int lda, const float* x, int incX,
              float beta, float* y, int incY );

   void gemv( CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, int m, int n, double alpha,
              const double* A, int lda, const double* x, int incX,
              double beta, double* y, int incY );

   void gemv( CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, int m, int n, complex<float> alpha,
              const complex<float>* A, int lda, const complex<float>* x, int incX,
              complex<float> beta, complex<float>* y, int incY );

   void gemv( CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, int m, int n, complex<double> alpha,
              const complex<double>* A, int lda, const complex<double>* x, int incX,
              complex<double> beta, complex<double>* y, int incY );

   template< typename VT1, typename MT1, bool SO, typename VT2, typename ST >
   void gemv( DenseVector<VT1,false>& y, const DenseMatrix<MT1,SO>& A,
              const DenseVector<VT2,false>& x, ST alpha, ST beta );

   template< typename VT1, typename VT2, typename MT1, bool SO, typename ST >
   void gemv( DenseVector<VT1,true>& y, const DenseVector<VT2,true>& x,
              const DenseMatrix<MT1,SO>& A, ST alpha, ST beta );

   } // namespace blaze
   \endcode

// \n \subsection blas_level_2_trmv Triangular Matrix/Vector Multiplication (trmv)
//
// The following wrapper functions provide a generic interface for the BLAS functions for the
// matrix/vector multiplication with a triangular matrix (\c strmv(), \c dtrmv(), \c ctrmv(),
// and \c ztrmv()):

   \code
   namespace blaze {

   void trmv( CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
              int n, const float* A, int lda, float* x, int incX );

   void trmv( CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
              int n, const double* A, int lda, double* x, int incX );

   void trmv( CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
              int n, const complex<float>* A, int lda, complex<float>* x, int incX );

   void trmv( CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
              int n, const complex<double>* A, int lda, complex<double>* x, int incX );

   template< typename VT, typename MT, bool SO >
   void trmv( DenseVector<VT,false>& x, const DenseMatrix<MT,SO>& A, CBLAS_UPLO uplo );

   template< typename VT, typename MT, bool SO >
   void trmv( DenseVector<VT,true>& x, const DenseMatrix<MT,SO>& A, CBLAS_UPLO uplo );

   } // namespace blaze
   \endcode

// \n \section blas_level_3 BLAS Level 3
// <hr>
//
// \subsection blas_level_3_gemm General Matrix/Matrix Multiplication (gemm)
//
// The following wrapper functions provide a generic interface for the BLAS functions for the
// general matrix/matrix multiplication (\c sgemm(), \c dgemm(), \c cgemm(), and \c zgemm()):

   \code
   namespace blaze {

   void gemm( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
              int m, int n, int k, float alpha, const float* A, int lda,
              const float* B, int ldb, float beta, float* C, int ldc );

   void gemm( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
              int m, int n, int k, double alpha, const double* A, int lda,
              const double* B, int ldb, double beta, float* C, int ldc );

   void gemm( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
              int m, int n, int k, complex<float> alpha, const complex<float>* A, int lda,
              const complex<float>* B, int ldb, complex<float> beta, float* C, int ldc );

   void gemm( CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
              int m, int n, int k, complex<double> alpha, const complex<double>* A, int lda,
              const complex<double>* B, int ldb, complex<double> beta, float* C, int ldc );

   template< typename MT1, bool SO1, typename MT2, bool SO2, typename MT3, bool SO3, typename ST >
   void gemm( DenseMatrix<MT1,SO1>& C, const DenseMatrix<MT2,SO2>& A,
              const DenseMatrix<MT3,SO3>& B, ST alpha, ST beta );

   } // namespace blaze
   \endcode

// \n \subsection blas_level_3_trmm Triangular Matrix/Matrix Multiplication (trmm)
//
// The following wrapper functions provide a generic interface for the BLAS functions for the
// matrix/matrix multiplication with a triangular matrix (\c strmm(), \c dtrmm(), \c ctrmm(), and
// \c ztrmm()):

   \code
   namespace blaze {

   void trmm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
              CBLAS_DIAG diag, int m, int n, float alpha, const float* A,
              int lda, float* B, int ldb );

   void trmm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
              CBLAS_DIAG diag, int m, int n, double alpha, const double* A,
              int lda, double* B, int ldb );

   void trmm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
              CBLAS_DIAG diag, int m, int n, complex<float> alpha, const complex<float>* A,
              int lda, complex<float>* B, int ldb );

   void trmm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
              CBLAS_DIAG diag, int m, int n, complex<double> alpha, const complex<double>* A,
              int lda, complex<double>* B, int ldb );

   template< typename MT1, bool SO1, typename MT2, bool SO2, typename ST >
   void trmm( DenseMatrix<MT1,SO1>& B, const DenseMatrix<MT2,SO2>& A,
              CBLAS_SIDE side, CBLAS_UPLO uplo, ST alpha );

   } // namespace blaze
   \endcode

// \n \subsection blas_level_3_trsm Triangular System Solver (trsm)
//
// The following wrapper functions provide a generic interface for the BLAS functions for solving
// a triangular system of equations (\c strsm(), \c dtrsm(), \c ctrsm(), and \c ztrsm()):

   \code
   namespace blaze {

   void trsm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
              CBLAS_DIAG diag, int m, int n, float alpha, const float* A,
              int lda, float* B, int ldb );

   void trsm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
              CBLAS_DIAG diag, int m, int n, double alpha, const double* A,
              int lda, double* B, int ldb );

   void trsm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
              CBLAS_DIAG diag, int m, int n, complex<float> alpha, const complex<float>* A,
              int lda, complex<float>* B, int ldb );

   void trsm( CBLAS_ORDER order, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
              CBLAS_DIAG diag, int m, int n, complex<double> alpha, const complex<double>* A,
              int lda, complex<double>* B, int ldb );

   template< typename MT, bool SO, typename VT, bool TF, typename ST >
   void trsm( const DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b,
              CBLAS_SIDE side, CBLAS_UPLO uplo, ST alpha );

   template< typename MT1, bool SO1, typename MT2, bool SO2, typename ST >
   void trsm( const DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& B,
              CBLAS_SIDE side, CBLAS_UPLO uplo, ST alpha );

   } // namespace blaze
   \endcode

// \n Previous: \ref matrix_serialization &nbsp; &nbsp; Next: \ref lapack_functions \n
*/
//*************************************************************************************************


//**LAPACK Functions*******************************************************************************
/*!\page lapack_functions LAPACK Functions
//
// \tableofcontents
//
//
// The \b Blaze library makes extensive use of the LAPACK functionality for various compute tasks
// (including the decomposition, inversion and the computation of the determinant of dense matrices).
// For this purpose, \b Blaze implements several convenient C++ wrapper functions for all required
// LAPACK functions. The following sections give a complete overview of all available LAPACK wrapper
// functions. For more details on the individual LAPACK functions see the \b Blaze function
// documentation or the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note All functions only work for general, non-adapted matrices with \c float, \c double,
// \c complex<float>, or \c complex<double> element type. The attempt to call the function with
// adaptors or matrices of any other element type results in a compile time error!
//
// \note All functions can only be used if the fitting LAPACK library is available and linked to
// the final executable. Otherwise a call to this function will result in a linker error.
//
// \note For performance reasons all functions do only provide the basic exception safety guarantee,
// i.e. in case an exception is thrown the given matrix may already have been modified.
//
//
// \n \section lapack_decomposition Matrix Decomposition
// <hr>
//
// The following functions decompose/factorize the given dense matrix. Based on this decomposition
// the matrix can be inverted or used to solve a linear system of equations.
//
//
// \n \subsection lapack_lu_decomposition LU Decomposition
//
// The following functions provide an interface for the LAPACK functions \c sgetrf(), \c dgetrf(),
// \c cgetrf(), and \c zgetrf(), which compute the LU decomposition for the given general matrix:

   \code
   namespace blaze {

   void getrf( int m, int n, float* A, int lda, int* ipiv, int* info );

   void getrf( int m, int n, double* A, int lda, int* ipiv, int* info );

   void getrf( int m, int n, complex<float>* A, int lda, int* ipiv, int* info );

   void getrf( int m, int n, complex<double>* A, int lda, int* ipiv, int* info );

   template< typename MT, bool SO >
   void getrf( DenseMatrix<MT,SO>& A, int* ipiv );

   } // namespace blaze
   \endcode

// The decomposition has the form

                          \f[ A = P \cdot L \cdot U, \f]\n

// where \c P is a permutation matrix, \c L is a lower unitriangular matrix, and \c U is an upper
// triangular matrix. The resulting decomposition is stored within \a A: In case of a column-major
// matrix, \c L is stored in the lower part of \a A and \c U is stored in the upper part. The unit
// diagonal elements of \c L are not stored. In case \a A is a row-major matrix the result is
// transposed.
//
// \note The LU decomposition will never fail, even for singular matrices. However, in case of a
// singular matrix the resulting decomposition cannot be used for a matrix inversion or solving
// a linear system of equations.
//
//
// \n \subsection lapack_ldlt_decomposition LDLT Decomposition
//
// The following functions provide an interface for the LAPACK functions \c ssytrf(), \c dsytrf(),
// \c csytrf(), and \c zsytrf(), which compute the LDLT (Bunch-Kaufman) decomposition for the given
// symmetric indefinite matrix:

   \code
   namespace blaze {

   void sytrf( char uplo, int n, float* A, int lda, int* ipiv, float* work, int lwork, int* info );

   void sytrf( char uplo, int n, double* A, int lda, int* ipiv, double* work, int lwork, int* info );

   void sytrf( char uplo, int n, complex<float>* A, int lda, int* ipiv, complex<float>* work, int lwork, int* info );

   void sytrf( char uplo, int n, complex<double>* A, int lda, int* ipiv, complex<double>* work, int lwork, int* info );

   template< typename MT, bool SO >
   void sytrf( DenseMatrix<MT,SO>& A, char uplo, int* ipiv );

   } // namespace blaze
   \endcode

// The decomposition has the form

                      \f[ A = U D U^{T} \texttt{ (if uplo = 'U'), or }
                          A = L D L^{T} \texttt{ (if uplo = 'L'), } \f]

// where \c U (or \c L) is a product of permutation and unit upper (lower) triangular matrices,
// and \c D is symmetric and block diagonal with 1-by-1 and 2-by-2 diagonal blocks. The resulting
// decomposition is stored within \a A: In case \a uplo is set to \c 'L' the result is stored in
// the lower part of the matrix and the upper part remains untouched, in case \a uplo is set to
// \c 'U' the result is stored in the upper part and the lower part remains untouched.
//
// \note The Bunch-Kaufman decomposition will never fail, even for singular matrices. However, in
// case of a singular matrix the resulting decomposition cannot be used for a matrix inversion or
// solving a linear system of equations.
//
//
// \n \subsection lapack_ldlh_decomposition LDLH Decomposition
//
// The following functions provide an interface for the LAPACK functions \c chetrf() and \c zsytrf(),
// which compute the LDLH (Bunch-Kaufman) decomposition for the given Hermitian indefinite matrix:

   \code
   namespace blaze {

   void hetrf( char uplo, int n, complex<float>* A, int lda, int* ipiv, complex<float>* work, int lwork, int* info );

   void hetrf( char uplo, int n, complex<double>* A, int lda, int* ipiv, complex<double>* work, int lwork, int* info );

   template< typename MT, bool SO >
   void hetrf( DenseMatrix<MT,SO>& A, char uplo, int* ipiv );

   } // namespace blaze
   \endcode

// The decomposition has the form

                      \f[ A = U D U^{H} \texttt{ (if uplo = 'U'), or }
                          A = L D L^{H} \texttt{ (if uplo = 'L'), } \f]

// where \c U (or \c L) is a product of permutation and unit upper (lower) triangular matrices,
// and \c D is Hermitian and block diagonal with 1-by-1 and 2-by-2 diagonal blocks. The resulting
// decomposition is stored within \a A: In case \a uplo is set to \c 'L' the result is stored in
// the lower part of the matrix and the upper part remains untouched, in case \a uplo is set to
// \c 'U' the result is stored in the upper part and the lower part remains untouched.
//
// \note The Bunch-Kaufman decomposition will never fail, even for singular matrices. However, in
// case of a singular matrix the resulting decomposition cannot be used for a matrix inversion or
// solving a linear system of equations.
//
//
// \n \subsection lapack_llh_decomposition Cholesky Decomposition
//
// The following functions provide an interface for the LAPACK functions \c spotrf(), \c dpotrf(),
// \c cpotrf(), and \c zpotrf(), which compute the Cholesky (LLH) decomposition for the given
// positive definite matrix:

   \code
   namespace blaze {

   void potrf( char uplo, int n, float* A, int lda, int* info );

   void potrf( char uplo, int n, double* A, int lda, int* info );

   void potrf( char uplo, int n, complex<float>* A, int lda, int* info );

   void potrf( char uplo, int n, complex<double>* A, int lda, int* info );

   template< typename MT, bool SO >
   void potrf( DenseMatrix<MT,SO>& A, char uplo );

   } // namespace blaze
   \endcode

// The decomposition has the form

                      \f[ A = U^{T} U \texttt{ (if uplo = 'U'), or }
                          A = L L^{T} \texttt{ (if uplo = 'L'), } \f]

// where \c U is an upper triangular matrix and \c L is a lower triangular matrix. The Cholesky
// decomposition fails if the given matrix \a A is not a positive definite matrix. In this case
// a \a std::std::invalid_argument exception is thrown.
//
//
// \n \subsection lapack_qr_decomposition QR Decomposition
//
// The following functions provide an interface for the LAPACK functions \c sgeqrf(), \c dgeqrf(),
// \c cgeqrf(), and \c zgeqrf(), which compute the QR decomposition of the given general matrix:

   \code
   namespace blaze {

   void geqrf( int m, int n, float* A, int lda, float* tau, float* work, int lwork, int* info );

   void geqrf( int m, int n, double* A, int lda, double* tau, double* work, int lwork, int* info );

   void geqrf( int m, int n, complex<float>* A, int lda, complex<float>* tau, complex<float>* work, int lwork, int* info );

   void geqrf( int m, int n, complex<double>* A, int lda, complex<double>* tau, complex<double>* work, int lwork, int* info );

   template< typename MT, bool SO >
   void geqrf( DenseMatrix<MT,SO>& A, typename MT::ElementType* tau );

   } // namespace blaze
   \endcode

// The decomposition has the form

                              \f[ A = Q \cdot R, \f]

// where the \c Q is represented as a product of elementary reflectors

               \f[ Q = H(1) H(2) . . . H(k) \texttt{, with k = min(m,n).} \f]

// Each H(i) has the form

                      \f[ H(i) = I - tau \cdot v \cdot v^T, \f]

// where \c tau is a real scalar, and \c v is a real vector with <tt>v(0:i-1) = 0</tt> and
// <tt>v(i) = 1</tt>. <tt>v(i+1:m)</tt> is stored on exit in <tt>A(i+1:m,i)</tt>, and \c tau
// in \c tau(i). Thus on exit the elements on and above the diagonal of the matrix contain the
// min(\a m,\a n)-by-\a n upper trapezoidal matrix \c R (\c R is upper triangular if \a m >= \a n);
// the elements below the diagonal, with the array \c tau, represent the orthogonal matrix \c Q as
// a product of min(\a m,\a n) elementary reflectors.
//
// The following functions provide an interface for the LAPACK functions \c sorgqr(), \c dorgqr(),
// \c cungqr(), and \c zunqqr(), which reconstruct the \c Q matrix from a QR decomposition:

   \code
   namespace blaze {

   void orgqr( int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* info );

   void orgqr( int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* info );

   void ungqr( int m, int n, int k, complex<float>* A, int lda, const complex<float>* tau, complex<float>* work, int lwork, int* info );

   void ungqr( int m, int n, int k, complex<double>* A, int lda, const complex<double>* tau, complex<double>* work, int lwork, int* info );

   template< typename MT, bool SO >
   void orgqr( DenseMatrix<MT,SO>& A, const typename MT::ElementType* tau );

   template< typename MT, bool SO >
   void ungqr( DenseMatrix<MT,SO>& A, const typename MT::ElementType* tau );

   } // namespace blaze
   \endcode

// The following functions provide an interface for the LAPACK functions \c sormqr(), \c dormqr(),
// \c cunmqr(), and \c zunmqr(), which can be used to multiply a matrix with the \c Q matrix from
// a QR decomposition:

   \code
   namespace blaze {

   void ormqr( char side, char trans, int m, int n, int k, const float* A, int lda, const float* tau, float* C, int ldc, float* work, int lwork, int* info );

   void ormqr( char side, char trans, int m, int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, double* work, int lwork, int* info );

   void unmqr( char side, char trans, int m, int n, int k, const complex<float>* A, int lda, const complex<float>* tau, complex<float>* C, int ldc, complex<float>* work, int lwork, int* info );

   void unmqr( char side, char trans, int m, int n, int k, const complex<double>* A, int lda, const complex<double>* tau, complex<double>* C, int ldc, complex<double>* work, int lwork, int* info );

   template< typename MT1, bool SO1, typename MT2, bool SO2 >
   void ormqr( DenseMatrix<MT1,SO1>& C, const DenseMatrix<MT2,SO2>& A, char side, char trans, const ElementType_<MT2>* tau );

   template< typename MT1, bool SO, typename MT2 >
   void unmqr( DenseMatrix<MT1,SO>& C, DenseMatrix<MT2,SO>& A, char side, char trans, ElementType_<MT2>* tau );

   } // namespace blaze
   \endcode

// \n \subsection lapack_rq_decomposition RQ Decomposition
//
// The following functions provide an interface for the LAPACK functions \c sgerqf(), \c dgerqf(),
// \c cgerqf(), and \c zgerqf(), which compute the RQ decomposition of the given general matrix:

   \code
   namespace blaze {

   void gerqf( int m, int n, float* A, int lda, float* tau, float* work, int lwork, int* info );

   void gerqf( int m, int n, double* A, int lda, double* tau, double* work, int lwork, int* info );

   void gerqf( int m, int n, complex<float>* A, int lda, complex<float>* tau, complex<float>* work, int lwork, int* info );

   void gerqf( int m, int n, complex<double>* A, int lda, complex<double>* tau, complex<double>* work, int lwork, int* info );

   template< typename MT, bool SO >
   void gerqf( DenseMatrix<MT,SO>& A, typename MT::ElementType* tau );

   } // namespace blaze
   \endcode

// The decomposition has the form

                              \f[ A = R \cdot Q, \f]

// where the \c Q is represented as a product of elementary reflectors

               \f[ Q = H(1) H(2) . . . H(k) \texttt{, with k = min(m,n).} \f]

// Each H(i) has the form

                      \f[ H(i) = I - tau \cdot v \cdot v^T, \f]

// where \c tau is a real scalar, and \c v is a real vector with <tt>v(n-k+i+1:n) = 0</tt> and
// <tt>v(n-k+i) = 1</tt>. <tt>v(1:n-k+i-1)</tt> is stored on exit in <tt>A(m-k+i,1:n-k+i-1)</tt>,
// and \c tau in \c tau(i). Thus in case \a m <= \a n, the upper triangle of the subarray
// <tt>A(1:m,n-m+1:n)</tt> contains the \a m-by-\a m upper triangular matrix \c R and in case
// \a m >= \a n, the elements on and above the (\a m-\a n)-th subdiagonal contain the \a m-by-\a n
// upper trapezoidal matrix \c R; the remaining elements in combination with the array \c tau
// represent the orthogonal matrix \c Q as a product of min(\a m,\a n) elementary reflectors.
//
// The following functions provide an interface for the LAPACK functions \c sorgrq(), \c dorgrq(),
// \c cungrq(), and \c zunqrq(), which reconstruct the \c Q matrix from a RQ decomposition:

   \code
   namespace blaze {

   void orgrq( int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* info );

   void orgrq( int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* info );

   void ungrq( int m, int n, int k, complex<float>* A, int lda, const complex<float>* tau, complex<float>* work, int lwork, int* info );

   void ungrq( int m, int n, int k, complex<double>* A, int lda, const complex<double>* tau, complex<double>* work, int lwork, int* info );

   template< typename MT, bool SO >
   void orgrq( DenseMatrix<MT,SO>& A, const typename MT::ElementType* tau );

   template< typename MT, bool SO >
   void ungrq( DenseMatrix<MT,SO>& A, const typename MT::ElementType* tau );

   } // namespace blaze
   \endcode

// The following functions provide an interface for the LAPACK functions \c sormrq(), \c dormrq(),
// \c cunmrq(), and \c zunmrq(), which can be used to multiply a matrix with the \c Q matrix from
// a RQ decomposition:

   \code
   namespace blaze {

   void ormrq( char side, char trans, int m, int n, int k, const float* A, int lda, const float* tau, float* C, int ldc, float* work, int lwork, int* info );

   void ormrq( char side, char trans, int m, int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, double* work, int lwork, int* info );

   void unmrq( char side, char trans, int m, int n, int k, const complex<float>* A, int lda, const complex<float>* tau, complex<float>* C, int ldc, complex<float>* work, int lwork, int* info );

   void unmrq( char side, char trans, int m, int n, int k, const complex<double>* A, int lda, const complex<double>* tau, complex<double>* C, int ldc, complex<double>* work, int lwork, int* info );

   template< typename MT1, bool SO1, typename MT2, bool SO2 >
   void ormrq( DenseMatrix<MT1,SO1>& C, const DenseMatrix<MT2,SO2>& A, char side, char trans, const ElementType_<MT2>* tau );

   template< typename MT1, bool SO, typename MT2 >
   void unmrq( DenseMatrix<MT1,SO>& C, DenseMatrix<MT2,SO>& A, char side, char trans, ElementType_<MT2>* tau );

   } // namespace blaze
   \endcode

// \n \subsection lapack_ql_decomposition QL Decomposition
//
// The following functions provide an interface for the LAPACK functions \c sgeqlf(), \c dgeqlf(),
// \c cgeqlf(), and \c zgeqlf(), which compute the QL decomposition of the given general matrix:

   \code
   namespace blaze {

   void geqlf( int m, int n, float* A, int lda, float* tau, float* work, int lwork, int* info );

   void geqlf( int m, int n, double* A, int lda, double* tau, double* work, int lwork, int* info );

   void geqlf( int m, int n, complex<float>* A, int lda, complex<float>* tau, complex<float>* work, int lwork, int* info );

   void geqlf( int m, int n, complex<double>* A, int lda, complex<double>* tau, complex<double>* work, int lwork, int* info );

   template< typename MT, bool SO >
   void geqlf( DenseMatrix<MT,SO>& A, typename MT::ElementType* tau );

   } // namespace blaze
   \endcode

// The decomposition has the form

                              \f[ A = Q \cdot L, \f]

// where the \c Q is represented as a product of elementary reflectors

               \f[ Q = H(k) . . . H(2) H(1) \texttt{, with k = min(m,n).} \f]

// Each H(i) has the form

                      \f[ H(i) = I - tau \cdot v \cdot v^T, \f]

// where \c tau is a real scalar, and \c v is a real vector with <tt>v(m-k+i+1:m) = 0</tt> and
// <tt>v(m-k+i) = 1</tt>. <tt>v(1:m-k+i-1)</tt> is stored on exit in <tt>A(1:m-k+i-1,n-k+i)</tt>,
// and \c tau in \c tau(i). Thus in case \a m >= \a n, the lower triangle of the subarray
// A(m-n+1:m,1:n) contains the \a n-by-\a n lower triangular matrix \c L and in case \a m <= \a n,
// the elements on and below the (\a n-\a m)-th subdiagonal contain the \a m-by-\a n lower
// trapezoidal matrix \c L; the remaining elements in combination with the array \c tau represent
// the orthogonal matrix \c Q as a product of min(\a m,\a n) elementary reflectors.
//
// The following functions provide an interface for the LAPACK functions \c sorgql(), \c dorgql(),
// \c cungql(), and \c zunqql(), which reconstruct the \c Q matrix from an QL decomposition:

   \code
   namespace blaze {

   void orgql( int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* info );

   void orgql( int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* info );

   void ungql( int m, int n, int k, complex<float>* A, int lda, const complex<float>* tau, complex<float>* work, int lwork, int* info );

   void ungql( int m, int n, int k, complex<double>* A, int lda, const complex<double>* tau, complex<double>* work, int lwork, int* info );

   template< typename MT, bool SO >
   void orgql( DenseMatrix<MT,SO>& A, const typename MT::ElementType* tau );

   template< typename MT, bool SO >
   void ungql( DenseMatrix<MT,SO>& A, const typename MT::ElementType* tau );

   } // namespace blaze
   \endcode

// The following functions provide an interface for the LAPACK functions \c sormql(), \c dormql(),
// \c cunmql(), and \c zunmql(), which can be used to multiply a matrix with the \c Q matrix from
// a QL decomposition:

   \code
   namespace blaze {

   void ormql( char side, char trans, int m, int n, int k, const float* A, int lda, const float* tau, float* C, int ldc, float* work, int lwork, int* info );

   void ormql( char side, char trans, int m, int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, double* work, int lwork, int* info );

   void unmql( char side, char trans, int m, int n, int k, const complex<float>* A, int lda, const complex<float>* tau, complex<float>* C, int ldc, complex<float>* work, int lwork, int* info );

   void unmql( char side, char trans, int m, int n, int k, const complex<double>* A, int lda, const complex<double>* tau, complex<double>* C, int ldc, complex<double>* work, int lwork, int* info );

   template< typename MT1, bool SO1, typename MT2, bool SO2 >
   void ormql( DenseMatrix<MT1,SO1>& C, const DenseMatrix<MT2,SO2>& A, char side, char trans, const ElementType_<MT2>* tau );

   template< typename MT1, bool SO, typename MT2 >
   void unmql( DenseMatrix<MT1,SO>& C, DenseMatrix<MT2,SO>& A, char side, char trans, ElementType_<MT2>* tau );

   } // namespace blaze
   \endcode

// \n \subsection lapack_lq_decomposition LQ Decomposition
//
// The following functions provide an interface for the LAPACK functions \c sgelqf(), \c dgelqf(),
// \c cgelqf(), and \c zgelqf(), which compute the LQ decomposition of the given general matrix:

   \code
   namespace blaze {

   void gelqf( int m, int n, float* A, int lda, float* tau, float* work, int lwork, int* info );

   void gelqf( int m, int n, double* A, int lda, double* tau, double* work, int lwork, int* info );

   void gelqf( int m, int n, complex<float>* A, int lda, complex<float>* tau, complex<float>* work, int lwork, int* info );

   void gelqf( int m, int n, complex<double>* A, int lda, complex<double>* tau, complex<double>* work, int lwork, int* info );

   template< typename MT, bool SO >
   void gelqf( DenseMatrix<MT,SO>& A, typename MT::ElementType* tau );

   } // namespace blaze
   \endcode

// The decomposition has the form

                              \f[ A = L \cdot Q, \f]

// where the \c Q is represented as a product of elementary reflectors

               \f[ Q = H(k) . . . H(2) H(1) \texttt{, with k = min(m,n).} \f]

// Each H(i) has the form

                      \f[ H(i) = I - tau \cdot v \cdot v^T, \f]

// where \c tau is a real scalar, and \c v is a real vector with <tt>v(0:i-1) = 0</tt> and
// <tt>v(i) = 1</tt>. <tt>v(i+1:n)</tt> is stored on exit in <tt>A(i,i+1:n)</tt>, and \c tau
// in \c tau(i). Thus on exit the elements on and below the diagonal of the matrix contain the
// \a m-by-min(\a m,\a n) lower trapezoidal matrix \c L (\c L is lower triangular if \a m <= \a n);
// the elements above the diagonal, with the array \c tau, represent the orthogonal matrix \c Q
// as a product of min(\a m,\a n) elementary reflectors.
//
// The following functions provide an interface for the LAPACK functions \c sorglq(), \c dorglq(),
// \c cunglq(), and \c zunqlq(), which reconstruct the \c Q matrix from an LQ decomposition:

   \code
   namespace blaze {

   void orglq( int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* info );

   void orglq( int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* info );

   void unglq( int m, int n, int k, complex<float>* A, int lda, const complex<float>* tau, complex<float>* work, int lwork, int* info );

   void unglq( int m, int n, int k, complex<double>* A, int lda, const complex<double>* tau, complex<double>* work, int lwork, int* info );

   template< typename MT, bool SO >
   void orglq( DenseMatrix<MT,SO>& A, const typename MT::ElementType* tau );

   template< typename MT, bool SO >
   void unglq( DenseMatrix<MT,SO>& A, const typename MT::ElementType* tau );

   } // namespace blaze
   \endcode

// The following functions provide an interface for the LAPACK functions \c sormlq(), \c dormlq(),
// \c cunmlq(), and \c zunmlq(), which can be used to multiply a matrix with the \c Q matrix from
// a LQ decomposition:

   \code
   namespace blaze {

   void ormlq( char side, char trans, int m, int n, int k, const float* A, int lda, const float* tau, float* C, int ldc, float* work, int lwork, int* info );

   void ormlq( char side, char trans, int m, int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, double* work, int lwork, int* info );

   void unmlq( char side, char trans, int m, int n, int k, const complex<float>* A, int lda, const complex<float>* tau, complex<float>* C, int ldc, complex<float>* work, int lwork, int* info );

   void unmlq( char side, char trans, int m, int n, int k, const complex<double>* A, int lda, const complex<double>* tau, complex<double>* C, int ldc, complex<double>* work, int lwork, int* info );

   template< typename MT1, bool SO1, typename MT2, bool SO2 >
   void ormlq( DenseMatrix<MT1,SO1>& C, const DenseMatrix<MT2,SO2>& A, char side, char trans, const ElementType_<MT2>* tau );

   template< typename MT1, bool SO, typename MT2 >
   void unmlq( DenseMatrix<MT1,SO>& C, DenseMatrix<MT2,SO>& A, char side, char trans, ElementType_<MT2>* tau );

   } // namespace blaze
   \endcode

// \n \section lapack_inversion Matrix Inversion
// <hr>
//
// Given a matrix that has already been decomposed, the following functions can be used to invert
// the matrix in-place.
//
//
// \n \subsection lapack_lu_inversion LU-based Inversion
//
// The following functions provide an interface for the LAPACK functions \c sgetri(), \c dgetri(),
// \c cgetri(), and \c zgetri(), which invert a general matrix that has already been decomposed by
// an \ref lapack_lu_decomposition :

   \code
   namespace blaze {

   void getri( int n, float* A, int lda, const int* ipiv, float* work, int lwork, int* info );

   void getri( int n, double* A, int lda, const int* ipiv, double* work, int lwork, int* info );

   void getri( int n, complex<float>* A, int lda, const int* ipiv, complex<float>* work, int lwork, int* info );

   void getri( int n, complex<double>* A, int lda, const int* ipiv, complex<double>* work, int lwork, int* info );

   template< typename MT, bool SO >
   void getri( DenseMatrix<MT,SO>& A, const int* ipiv );

   } // namespace blaze
   \endcode

// The functions fail if ...
//
//  - ... the given matrix is not a square matrix;
//  - ... the given matrix is singular and not invertible.
//
// The first four functions report failure via the \c info argument, the fifth function throws a
// \a std::invalid_argument exception in case of an error.
//
//
// \n \subsection lapack_ldlt_inversion LDLT-based Inversion
//
// The following functions provide an interface for the LAPACK functions \c ssytri(), \c dsytri(),
// \c csytri(), and \c zsytri(), which invert a symmetric indefinite matrix that has already been
// decomposed by an \ref lapack_ldlt_decomposition :

   \code
   namespace blaze {

   void sytri( char uplo, int n, float* A, int lda, const int* ipiv, float* work, int* info );

   void sytri( char uplo, int n, double* A, int lda, const int* ipiv, double* work, int* info );

   void sytri( char uplo, int n, complex<float>* A, int lda, const int* ipiv, complex<float>* work, int* info );

   void sytri( char uplo, int n, complex<double>* A, int lda, const int* ipiv, complex<double>* work, int* info );

   template< typename MT, bool SO >
   void sytri( DenseMatrix<MT,SO>& A, char uplo, const int* ipiv );

   } // namespace blaze
   \endcode

// The functions fail if ...
//
//  - ... the given matrix is not a square matrix;
//  - ... the given matrix is singular and not invertible.
//
// The first four functions report failure via the \c info argument, the fifth function throws a
// \a std::invalid_argument exception in case of an error.
//
//
// \n \subsection lapack_ldlh_inversion LDLH-based Inversion
//
// The following functions provide an interface for the LAPACK functions \c chetri() and
// \c zhetri(), which invert an Hermitian indefinite matrix that has already been decomposed by
// an \ref lapack_ldlh_decomposition :

   \code
   namespace blaze {

   void hetri( char uplo, int n, complex<float>* A, int lda, const int* ipiv, complex<float>* work, int* info );

   void hetri( char uplo, int n, complex<double>* A, int lda, const int* ipiv, complex<double>* work, int* info );

   template< typename MT, bool SO >
   void hetri( DenseMatrix<MT,SO>& A, char uplo, const int* ipiv );

   } // namespace blaze
   \endcode

// The functions fail if ...
//
//  - ... the given matrix is not a square matrix;
//  - ... the given matrix is singular and not invertible.
//
// The first four functions report failure via the \c info argument, the fifth function throws a
// \a std::invalid_argument exception in case of an error.
//
//
// \n \subsection lapack_llh_inversion Cholesky-based Inversion
//
// The following functions provide an interface for the LAPACK functions \c spotri(), \c dpotri(),
// \c cpotri(), and \c zpotri(), which invert a positive definite matrix that has already been
// decomposed by an \ref lapack_llh_decomposition :

   \code
   namespace blaze {

   void potri( char uplo, int n, float* A, int lda, int* info );

   void potri( char uplo, int n, double* A, int lda, int* info );

   void potri( char uplo, int n, complex<float>* A, int lda, int* info );

   void potri( char uplo, int n, complex<double>* A, int lda, int* info );

   template< typename MT, bool SO >
   void potri( DenseMatrix<MT,SO>& A, char uplo );

   } // namespace blaze
   \endcode

// The functions fail if ...
//
//  - ... the given matrix is not a square matrix;
//  - ... the given \a uplo argument is neither 'L' nor 'U';
//  - ... the given matrix is singular and not invertible.
//
// The first four functions report failure via the \c info argument, the fifth function throws a
// \a std::invalid_argument exception in case of an error.
//
//
// \n \subsection lapack_triangular_inversion Inversion of Triangular Matrices
//
// The following functions provide an interface for the LAPACK functions \c strtri(), \c dtrtri(),
// \c ctrtri(), and \c ztrtri(), which invert the given triangular matrix in-place:

   \code
   namespace blaze {

   void trtri( char uplo, char diag, int n, float* A, int lda, int* info );

   void trtri( char uplo, char diag, int n, double* A, int lda, int* info );

   void trtri( char uplo, char diag, int n, complex<float>* A, int lda, int* info );

   void trtri( char uplo, char diag, int n, complex<double>* A, int lda, int* info );

   template< typename MT, bool SO >
   void trtri( DenseMatrix<MT,SO>& A, char uplo, char diag );

   } // namespace blaze
   \endcode

// The functions fail if ...
//
//  - ... the given matrix is not a square matrix;
//  - ... the given \a uplo argument is neither 'L' nor 'U';
//  - ... the given \a diag argument is neither 'U' nor 'N';
//  - ... the given matrix is singular and not invertible.
//
// The first four functions report failure via the \c info argument, the fifth function throws a
// \a std::invalid_argument exception in case of an error.
//
//
// \n \section lapack_substitution Substitution
// <hr>
//
// Given a matrix that has already been decomposed the following functions can be used to perform
// the forward/backward substitution step to compute the solution to a system of linear equations.
// Note that depending on the storage order of the system matrix and the given right-hand side the
// functions solve different equation systems:
//
// Single right-hand side:
//  - \f$ A  *x=b \f$ if \a A is column-major
//  - \f$ A^T*x=b \f$ if \a A is row-major
//
// Multiple right-hand sides:
//  - \f$ A  *X  =B   \f$ if both \a A and \a B are column-major
//  - \f$ A^T*X  =B   \f$ if \a A is row-major and \a B is column-major
//  - \f$ A  *X^T=B^T \f$ if \a A is column-major and \a B is row-major
//  - \f$ A^T*X^T=B^T \f$ if both \a A and \a B are row-major
//
// In this context the general system matrix \a A is a n-by-n matrix that has already been
// factorized by the according decomposition function, \a x and \a b are n-dimensional vectors
// and \a X and \a B are either row-major m-by-n matrices or column-major n-by-m matrices.
//
//
// \n \subsection lapack_lu_substitution LU-based Substitution
//
// The following functions provide an interface for the LAPACK functions \c sgetrs(), \c dgetrs(),
// \c cgetrs(), and \c zgetrs(), which perform the substitution step for a general matrix that has
// already been decomposed by an \ref lapack_lu_decomposition :

   \code
   namespace blaze {

   void getrs( char trans, int n, int nrhs, const float* A, int lda, const int* ipiv, float* B, int ldb, int* info );

   void getrs( char trans, int n, int nrhs, const double* A, int lda, const int* ipiv, double* B, int ldb, int* info );

   void getrs( char trans, int n, const complex<float>* A, int lda, const int* ipiv, complex<float>* B, int ldb, int* info );

   void getrs( char trans, int n, const complex<double>* A, int lda, const int* ipiv, complex<double>* B, int ldb, int* info );

   template< typename MT, bool SO, typename VT, bool TF >
   void getrs( const DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b, char trans, const int* ipiv );

   template< typename MT1, bool SO1, typename MT2, bool SO2 >
   void getrs( const DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& B, char trans, const int* ipiv );

   } // namespace blaze
   \endcode

// If the function exits successfully, the vector \a b or the matrix \a B contain the solution(s)
// of the linear system of equations. The function fails if ...
//
//  - ... the given system matrix is not a square matrix;
//  - ... the given \a trans argument is neither 'N' nor 'T' nor 'C';
//  - ... the sizes of the two given matrices do not match.
//
// The first four functions report failure via the \c info argument, the last two functions throw
// a \a std::invalid_argument exception in case of an error.
//
//
// \n \subsection lapack_ldlt_substitution LDLT-based Substitution
//
// The following functions provide an interface for the LAPACK functions \c ssytrs(), \c dsytrs(),
// \c csytrs(), and \c zsytrs(), which perform the substitution step for a symmetric indefinite
// matrix that has already been decomposed by an \ref lapack_ldlt_decomposition :

   \code
   namespace blaze {

   void sytrs( char uplo, int n, int nrhs, const float* A, int lda, const int* ipiv, float* B, int ldb, int* info );

   void sytrs( char uplo, int n, int nrhs, const double* A, int lda, const int* ipiv, double* B, int ldb, int* info );

   void sytrs( char uplo, int n, int nrhs, const complex<float>* A, int lda, const int* ipiv, complex<float>* B, int ldb, int* info );

   void sytrs( char uplo, int n, int nrhs, const complex<double>* A, int lda, const int* ipiv, complex<double>* B, int ldb, int* info );

   template< typename MT, bool SO, typename VT, bool TF >
   void sytrs( const DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b, char uplo, const int* ipiv );

   template< typename MT1, bool SO1, typename MT2, bool SO2 >
   void sytrs( const DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& B, char uplo, const int* ipiv );

   } // namespace blaze
   \endcode

// If the function exits successfully, the vector \a b or the matrix \a B contain the solution(s)
// of the linear system of equations. The function fails if ...
//
//  - ... the given system matrix is not a square matrix;
//  - ... the given \a uplo argument is neither 'L' nor 'U';
//  - ... the sizes of the two given matrices do not match.
//
// The first four functions report failure via the \c info argument, the last two functions throw
// a \a std::invalid_argument exception in case of an error.
//
//
// \n \subsection lapack_ldlh_substitution LDLH-based Substitution
//
// The following functions provide an interface for the LAPACK functions \c chetrs(), and \c zhetrs(),
// which perform the substitution step for an Hermitian indefinite matrix that has already been
// decomposed by an \ref lapack_ldlh_decomposition :

   \code
   namespace blaze {

   void hetrs( char uplo, int n, int nrhs, const complex<float>* A, int lda, const int* ipiv, complex<float>* B, int ldb, int* info );

   void hetrs( char uplo, int n, int nrhs, const complex<double>* A, int lda, const int* ipiv, complex<double>* B, int ldb, int* info );

   template< typename MT, bool SO, typename VT, bool TF >
   void hetrs( const DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b, char uplo, const int* ipiv );

   template< typename MT1, bool SO1, typename MT2, bool SO2 >
   void hetrs( const DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& B, char uplo, const int* ipiv );

   } // namespace blaze
   \endcode

// If the function exits successfully, the vector \a b or the matrix \a B contain the solution(s)
// of the linear system of equations. The function fails if ...
//
//  - ... the given system matrix is not a square matrix;
//  - ... the given \a uplo argument is neither 'L' nor 'U';
//  - ... the sizes of the two given matrices do not match.
//
// The first two functions report failure via the \c info argument, the last two functions throw
// a \a std::invalid_argument exception in case of an error.
//
//
// \n \subsection lapack_llh_substitution Cholesky-based Substitution
//
// The following functions provide an interface for the LAPACK functions \c spotrs(), \c dpotrs(),
// \c cpotrs(), and \c zpotrs(), which perform the substitution step for a positive definite matrix
// that has already been decomposed by an \ref lapack_llh_decomposition :

   \code
   namespace blaze {

   void potrs( char uplo, int n, int nrhs, const float* A, int lda, float* B, int ldb, int* info );

   void potrs( char uplo, int n, int nrhs, const double* A, int lda, double* B, int ldb, int* info );

   void potrs( char uplo, int n, int nrhs, const complex<float>* A, int lda, complex<float>* B, int ldb, int* info );

   void potrs( char uplo, int n, int nrhs, const complex<double>* A, int lda, complex<double>* B, int ldb, int* info );

   template< typename MT, bool SO, typename VT, bool TF >
   void potrs( const DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b, char uplo );

   template< typename MT1, bool SO1, typename MT2, bool SO2 >
   void potrs( const DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& B, char uplo );

   } // namespace blaze
   \endcode

// If the function exits successfully, the vector \a b or the matrix \a B contain the solution(s)
// of the linear system of equations. The function fails if ...
//
//  - ... the given system matrix is not a square matrix;
//  - ... the given \a uplo argument is neither 'L' nor 'U';
//  - ... the sizes of the two given matrices do not match.
//
// The first two functions report failure via the \c info argument, the last two functions throw
// a \a std::invalid_argument exception in case of an error.
//
//
// \n \subsection lapack_triangular_substitution Substitution for Triangular Matrices
//
// The following functions provide an interface for the LAPACK functions \c strtrs(), \c dtrtrs(),
// \c ctrtrs(), and \c ztrtrs(), which perform the substitution step for a triangular matrix:

   \code
   namespace blaze {

   void trtrs( char uplo, char trans, char diag, int n, int nrhs, const float* A, int lda, float* B, int ldb, int* info );

   void trtrs( char uplo, char trans, char diag, int n, int nrhs, const double* A, int lda, double* B, int ldb, int* info );

   void trtrs( char uplo, char trans, char diag, int n, int nrhs, const complex<float>* A, int lda, complex<float>* B, int ldb, int* info );

   void trtrs( char uplo, char trans, char diag, int n, int nrhs, const complex<double>* A, int lda, complex<double>* B, int ldb, int* info );

   template< typename MT, bool SO, typename VT, bool TF >
   void trtrs( const DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b, char uplo, char trans, char diag );

   template< typename MT1, bool SO1, typename MT2, bool SO2 >
   void trtrs( const DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& B, char uplo, char trans, char diag );

   } // namespace blaze
   \endcode

// If the function exits successfully, the vector \a b or the matrix \a B contain the solution(s)
// of the linear system of equations. The function fails if ...
//
//  - ... the given system matrix is not a square matrix;
//  - ... the given \a uplo argument is neither 'L' nor 'U';
//  - ... the given \a trans argument is neither 'N' nor 'T' nor 'C';
//  - ... the given \a diag argument is neither 'U' nor 'N';
//  - ... the sizes of the two given matrices do not match.
//
// The first four functions report failure via the \c info argument, the last two functions throw
// a \a std::invalid_argument exception in case of an error.
//
//
// \n \section lapack_linear_system_solver Linear System Solver
// <hr>
//
// The following functions represent compound functions that perform both the decomposition step
// as well as the substitution step to compute the solution to a system of linear equations. Note
// that depending on the storage order of the system matrix and the given right-hand side the
// functions solve different equation systems:
//
// Single right-hand side:
//  - \f$ A  *x=b \f$ if \a A is column-major
//  - \f$ A^T*x=b \f$ if \a A is row-major
//
// Multiple right-hand sides:
//  - \f$ A  *X  =B   \f$ if both \a A and \a B are column-major
//  - \f$ A^T*X  =B   \f$ if \a A is row-major and \a B is column-major
//  - \f$ A  *X^T=B^T \f$ if \a A is column-major and \a B is row-major
//  - \f$ A^T*X^T=B^T \f$ if both \a A and \a B are row-major
//
// In this context the general system matrix \a A is a n-by-n matrix that has already been
// factorized by the according decomposition function, \a x and \a b are n-dimensional vectors
// and \a X and \a B are either row-major m-by-n matrices or column-major n-by-m matrices.
//
//
// \subsection lapack_lu_linear_system_solver LU-based Linear System Solver
//
// The following functions provide an interface for the LAPACK functions \c sgesv(), \c dgesv(),
// \c cgesv(), and \c zgesv(), which combine an \ref lapack_lu_decomposition and the according
// \ref lapack_lu_substitution :

   \code
   namespace blaze {

   void gesv( int n, int nrhs, float* A, int lda, int* ipiv, float* B, int ldb, int* info );

   void gesv( int n, int nrhs, double* A, int lda, int* ipiv, double* B, int ldb, int* info );

   void gesv( int n, int nrhs, complex<float>* A, int lda, int* ipiv, complex<float>* B, int ldb, int* info );

   void gesv( int n, int nrhs, complex<double>* A, int lda, int* ipiv, complex<double>* B, int ldb, int* info );

   template< typename MT, bool SO, typename VT, bool TF >
   void gesv( DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b, int* ipiv );

   template< typename MT1, bool SO1, typename MT2, bool SO2 >
   void gesv( DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& B, int* ipiv );

   } // namespace blaze
   \endcode

// If the function exits successfully, the vector \a b or the matrix \a B contain the
// solution(s) of the linear system of equations and \a A has been decomposed by means of an
// \ref lapack_lu_decomposition.
//
// The functions fail if ...
//
//  - ... the given system matrix is not a square matrix;
//  - ... the given system matrix is singular and not invertible.
//
// The first four functions report failure via the \c info argument, the fifth function throws a
// \a std::invalid_argument exception in case of an error.
//
//
// \n \subsection lapack_ldlt_linear_system_solver LDLT-based Linear System Solver
//
// The following functions provide an interface for the LAPACK functions \c ssysv(), \c dsysv(),
// \c csysv(), and \c zsysv(), which combine an \ref lapack_ldlt_decomposition and the according
// \ref lapack_ldlt_substitution :

   \code
   namespace blaze {

   void sysv( char uplo, int n, int nrhs, float* A, int lda, int* ipiv, float* B, int ldb, float* work, int lwork, int* info );

   void sysv( char uplo, int n, int nrhs, double* A, int lda, int* ipiv, double* B, int ldb, double* work, int lwork, int* info );

   void sysv( char uplo, int n, int nrhs, complex<float>* A, int lda, int* ipiv, complex<float>* B, int ldb, complex<float>* work, int lwork, int* info );

   void sysv( char uplo, int n, int nrhs, complex<double>* A, int lda, int* ipiv, complex<double>* B, int ldb, complex<double>* work, int lwork, int* info );

   template< typename MT, bool SO, typename VT, bool TF >
   void sysv( DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b, char uplo, int* ipiv );

   template< typename MT1, bool SO1, typename MT2, bool SO2 >
   void sysv( DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& B, char uplo, int* ipiv );

   } // namespace blaze
   \endcode

// If the function exits successfully, the vector \a b or the matrix \a B contain the
// solution(s) of the linear system of equations and \a A has been decomposed by means of an
// \ref lapack_ldlt_decomposition.
//
// The functions fail if ...
//
//  - ... the given system matrix is not a square matrix;
//  - ... the given \a uplo argument is neither 'L' nor 'U';
//  - ... the sizes of the two given matrices do not match;
//  - ... the given system matrix is singular and not invertible.
//
// The first four functions report failure via the \c info argument, the fifth function throws a
// \a std::invalid_argument exception in case of an error.
//
//
// \n \subsection lapack_ldlh_linear_system_solver LDLH-based Linear System Solver
//
// The following functions provide an interface for the LAPACK functions \c shesv(), \c dhesv(),
// \c chesv(), and \c zhesv(), which combine an \ref lapack_ldlh_decomposition and the according
// \ref lapack_ldlh_substitution :

   \code
   namespace blaze {

   void hesv( char uplo, int n, int nrhs, complex<float>* A, int lda, int* ipiv, complex<float>* B, int ldb, complex<float>* work, int lwork, int* info );

   void hesv( char uplo, int n, int nrhs, complex<double>* A, int lda, int* ipiv, complex<double>* B, int ldb, complex<double>* work, int lwork, int* info );

   template< typename MT, bool SO, typename VT, bool TF >
   void hesv( DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b, char uplo, int* ipiv );

   template< typename MT1, bool SO1, typename MT2, bool SO2 >
   void hesv( DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& B, char uplo, int* ipiv );

   } // namespace blaze
   \endcode

// If the function exits successfully, the vector \a b or the matrix \a B contain the
// solution(s) of the linear system of equations and \a A has been decomposed by means of an
// \ref lapack_ldlh_decomposition.
//
// The functions fail if ...
//
//  - ... the given system matrix is not a square matrix;
//  - ... the given \a uplo argument is neither 'L' nor 'U';
//  - ... the sizes of the two given matrices do not match;
//  - ... the given system matrix is singular and not invertible.
//
// The first two functions report failure via the \c info argument, the fifth function throws a
// \a std::invalid_argument exception in case of an error.
//
//
// \n \subsection lapack_llh_linear_system_solver Cholesky-based Linear System Solver
//
// The following functions provide an interface for the LAPACK functions \c sposv(), \c dposv(),
// \c cposv(), and \c zposv(), which combine an \ref lapack_llh_decomposition and the according
// \ref lapack_llh_substitution :

   \code
   namespace blaze {

   void posv( char uplo, int n, int nrhs, float* A, int lda, float* B, int ldb, int* info );

   void posv( char uplo, int n, int nrhs, double* A, int lda, double* B, int ldb, int* info );

   void posv( char uplo, int n, int nrhs, complex<float>* A, int lda, complex<float>* B, int ldb, int* info );

   void posv( char uplo, int n, int nrhs, complex<double>* A, int lda, complex<double>* B, int ldb, int* info );

   template< typename MT, bool SO, typename VT, bool TF >
   void posv( DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b, char uplo );

   template< typename MT1, bool SO1, typename MT2, bool SO2 >
   void posv( DenseMatrix<MT1,SO1>& A, DenseMatrix<MT2,SO2>& B, char uplo );

   } // namespace blaze
   \endcode

// If the function exits successfully, the vector \a b or the matrix \a B contain the
// solution(s) of the linear system of equations and \a A has been decomposed by means of an
// \ref lapack_llh_decomposition.
//
// The functions fail if ...
//
//  - ... the given system matrix is not a square matrix;
//  - ... the given \a uplo argument is neither 'L' nor 'U';
//  - ... the sizes of the two given matrices do not match;
//  - ... the given system matrix is singular and not invertible.
//
// The first four functions report failure via the \c info argument, the fifth function throws a
// \a std::invalid_argument exception in case of an error.
//
//
// \n \subsection lapack_triangular_linear_system_solver Linear System Solver for Triangular Matrices
//
// The following functions provide an interface for the LAPACK functions \c strsv(), \c dtrsv(),
// \c ctrsv(), and \c ztrsv():

   \code
   namespace blaze {

   void trsv( char uplo, char trans, char diag, int n, const float* A, int lda, float* x, int incX );

   void trsv( char uplo, char trans, char diag, int n, const double* A, int lda, double* x, int incX );

   void trsv( char uplo, char trans, char diag, int n, const complex<float>* A, int lda, complex<float>* x, int incX );

   void trsv( char uplo, char trans, char diag, int n, const complex<double>* A, int lda, complex<double>* x, int incX );

   template< typename MT, bool SO, typename VT, bool TF >
   void trsv( const DenseMatrix<MT,SO>& A, DenseVector<VT,TF>& b, char uplo, char trans, char diag );

   } // namespace blaze
   \endcode

// If the function exits successfully, the vector \a b or the matrix \a B contain the
// solution(s) of the linear system of equations.
//
// The functions fail if ...
//
//  - ... the given system matrix is not a square matrix;
//  - ... the given \a uplo argument is neither 'L' nor 'U';
//  - ... the given \a trans argument is neither 'N' nor 'T' nor 'C';
//  - ... the given \a diag argument is neither 'U' nor 'N'.
//
// The last function throws a \a std::invalid_argument exception in case of an error. Note that
// none of the functions does perform any test for singularity or near-singularity. Such tests
// must be performed prior to calling this function!
//
//
// \n Previous: \ref blas_functions &nbsp; &nbsp; Next: \ref configuration_files \n
*/
//*************************************************************************************************


//**Configuration Files****************************************************************************
/*!\page configuration_files Configuration Files
//
// \tableofcontents
//
//
// Sometimes it might necessary to adapt \b Blaze to specific requirements. For this purpose
// \b Blaze provides several configuration files in the <tt>./blaze/config/</tt> subdirectory,
// which provide ample opportunity to customize internal settings, behavior, and thresholds.
// This chapter explains the most important of these configuration files.
//
//
// \n \section transpose_flag Default Vector Storage
// <hr>
//
// The \b Blaze default is that all vectors are created as column vectors (if not specified
// explicitly):

   \code
   blaze::StaticVector<double,3UL> x;  // Creates a 3-dimensional static column vector
   \endcode

// The header file <tt>./blaze/config/TransposeFlag.h</tt> allows the configuration of the default
// vector storage (i.e. the default transpose flag of the vectors). Via the \c defaultTransposeFlag
// value the default transpose flag for all vector of the \b Blaze library can be specified:

   \code
   constexpr bool defaultTransposeFlag = columnVector;
   \endcode

// Valid settings for the \c defaultTransposeFlag are blaze::rowVector and blaze::columnVector.
//
//
// \n \section storage_order Default Matrix Storage
// <hr>
//
// Matrices are by default created as row-major matrices:

   \code
   blaze::StaticMatrix<double,3UL,3UL>  A;  // Creates a 3x3 row-major matrix
   \endcode

// The header file <tt>./blaze/config/StorageOrder.h</tt> allows the configuration of the default
// matrix storage order. Via the \c defaultStorageOrder value the default storage order for all
// matrices of the \b Blaze library can be specified.

   \code
   constexpr bool defaultStorageOrder = rowMajor;
   \endcode

// Valid settings for the \c defaultStorageOrder are blaze::rowMajor and blaze::columnMajor.
//
//
// \n \section blas_mode BLAS Mode
// <hr>
//
// In order to achieve maximum performance for multiplications with dense matrices, \b Blaze can
// be configured to use a BLAS library. Via the following compilation switch in the configuration
// file <tt>./blaze/config/BLAS.h</tt> BLAS can be enabled:

   \code
   #define BLAZE_BLAS_MODE 1
   \endcode

// In case the selected BLAS library provides parallel execution, the \c BLAZE_BLAS_IS_PARALLEL
// switch should be activated to prevent \b Blaze from parallelizing on its own:

   \code
   #define BLAZE_BLAS_IS_PARALLEL 1
   \endcode

// In case no BLAS library is available, \b Blaze will still work and will not be reduced in
// functionality, but performance may be limited.
//
//
// \n \section cache_size Cache Size
// <hr>
//
// The optimization of several \b Blaze compute kernels depends on the cache size of the target
// architecture. By default, \b Blaze assumes a cache size of 3 MiByte. However, for optimal
// speed the exact cache size of the system should be provided via the \c cacheSize value in the
// <tt>./blaze/config/CacheSize.h</tt> configuration file:

   \code
   constexpr size_t cacheSize = 3145728UL;
   \endcode

// \n \section vectorization Vectorization
// <hr>
//
// In order to achieve maximum performance and to exploit the compute power of a target platform
// the \b Blaze library attempts to vectorize all linear algebra operations by SSE, AVX, and/or
// MIC intrinsics, depending on which instruction set is available. However, it is possible to
// disable the vectorization entirely by the compile time switch in the configuration file
// <tt>./blaze/config/Vectorization.h</tt>:

   \code
   #define BLAZE_USE_VECTORIZATION 1
   \endcode

// In case the switch is set to 1, vectorization is enabled and the \b Blaze library is allowed
// to use intrinsics to speed up computations. In case the switch is set to 0, vectorization is
// disabled entirely and the \b Blaze library chooses default, non-vectorized functionality for
// the operations. Note that deactivating the vectorization may pose a severe performance
// limitation for a large number of operations!
//
//
// \n \section thresholds Thresholds
// <hr>
//
// \b Blaze provides several thresholds that can be adapted to the characteristics of the target
// platform. For instance, the \c DMATDVECMULT_THRESHOLD specifies the threshold between the
// application of the custom \b Blaze kernels for small dense matrix/dense vector multiplications
// and the BLAS kernels for large multiplications. All thresholds, including the thresholds for
// the OpenMP-based parallelization, are contained within the configuration file
// <tt>./blaze/config/Thresholds.h</tt>.
//
//
// \n \section padding Padding
// <hr>
//
// By default the \b Blaze library uses padding for all dense vectors and matrices in order to
// achieve maximum performance in all operations. Due to padding, the proper alignment of data
// elements can be guaranteed and the need for remainder loops is minimized. However, on the
// downside padding introduces an additional memory overhead, which can be large depending on
// the used data type.
//
// The configuration file <tt>./blaze/config/Optimizations.h</tt> provides a compile time switch
// that can be used to (de-)activate padding:

   \code
   constexpr bool usePadding = true;
   \endcode

// If \c usePadding is set to \c true padding is enabled for all dense vectors and matrices, if
// it is set to \c false padding is disabled. Note however that disabling padding can considerably
// reduce the performance of all dense vector and matrix operations!
//
//
// \n \section streaming Streaming (Non-Temporal Stores)
// <hr>
//
// For vectors and matrices that don't fit into the cache anymore non-temporal stores can provide
// a significant performance advantage of about 20%. However, this advantage is only in effect in
// case the memory bandwidth of the target architecture is maxed out. If the target architecture's
// memory bandwidth cannot be exhausted the use of non-temporal stores can decrease performance
// instead of increasing it.
//
// The configuration file <tt>./blaze/config/Optimizations.h</tt> provides a compile time switch
// that can be used to (de-)activate streaming:

   \code
   constexpr bool useStreaming = true;
   \endcode

// If \c useStreaming is set to \c true streaming is enabled, if it is set to \c false streaming
// is disabled. It is recommended to consult the target architecture's white papers to decide
// whether streaming is beneficial or hurtful for performance.
//
//
// \n Previous: \ref lapack_functions &nbsp; &nbsp; Next: \ref custom_data_types \n
*/
//*************************************************************************************************


//**Custom Data Types******************************************************************************
/*!\page custom_data_types Custom Data Types
//
//
// The \b Blaze library tries hard to make the use of custom data types as convenient, easy and
// intuitive as possible. However, unfortunately it is not possible to meet the requirements of
// all possible data types. Thus it might be necessary to provide \b Blaze with some additional
// information about the data type. The following sections give an overview of the necessary steps
// to enable the use of the hypothetical custom data type \c custom::double_t for vector and
// matrix operations. For example:

   \code
   blaze::DynamicVector<custom::double_t> a, b, c;
   // ... Resizing and initialization
   c = a + b;
   \endcode

// The \b Blaze library assumes that the \c custom::double_t data type provides \c operator+()
// for additions, \c operator-() for subtractions, \c operator*() for multiplications and
// \c operator/() for divisions. If any of these functions is missing it is necessary to implement
// the operator to perform the according operation. For this example we assume that the custom
// data type provides the four following functions instead of operators:

   \code
   namespace custom {

   double_t add ( const double_t& a, const double_t b );
   double_t sub ( const double_t& a, const double_t b );
   double_t mult( const double_t& a, const double_t b );
   double_t div ( const double_t& a, const double_t b );

   } // namespace custom
   \endcode

// The following implementations will satisfy the requirements of the \b Blaze library:

   \code
   inline custom::double_t operator+( const custom::double_t& a, const custom::double_t& b )
   {
      return add( a, b );
   }

   inline custom::double_t operator-( const custom::double_t& a, const custom::double_t& b )
   {
      return sub( a, b );
   }

   inline custom::double_t operator*( const custom::double_t& a, const custom::double_t& b )
   {
      return mult( a, b );
   }

   inline custom::double_t operator/( const custom::double_t& a, const custom::double_t& b )
   {
      return div( a, b );
   }
   \endcode

// \b Blaze will use all the information provided with these functions (for instance the return
// type) to properly handle the operations. In the rare case that the return type cannot be
// automatically determined from the operator it might be additionally necessary to provide a
// specialization of the following four \b Blaze class templates:

   \code
   namespace blaze {

   template<>
   struct AddTrait<custom::double_t,custom::double_t> {
      typedef custom::double_t  Type;
   };

   template<>
   struct SubTrait<custom::double_t,custom::double_t> {
      typedef custom::double_t  Type;
   };

   template<>
   struct MultTrait<custom::double_t,custom::double_t> {
      typedef custom::double_t  Type;
   };

   template<>
   struct DivTrait<custom::double_t,custom::double_t> {
      typedef custom::double_t  Type;
   };

   } // namespace blaze
   \endcode

// The same steps are necessary if several custom data types need to be combined (as for instance
// \c custom::double_t and \c custom::float_t). Note that in this case both permutations need to
// be taken into account:

   \code
   custom::double_t operator+( const custom::double_t& a, const custom::float_t& b );
   custom::double_t operator+( const custom::float_t& a, const custom::double_t& b );
   // ...
   \endcode

// Please note that only built-in data types apply for vectorization and thus custom data types
// cannot achieve maximum performance!
//
//
// \n Previous: \ref configuration_files &nbsp; &nbsp; Next: \ref error_reporting_customization \n
*/
//*************************************************************************************************


//**Customization of the Error Reporting Mechanism*************************************************
/*!\page error_reporting_customization Customization of the Error Reporting Mechanism
//
// \tableofcontents
//
//
// \n \section error_reporting_background Background
// <hr>
//
// The default way of \b Blaze to report errors of any kind is to throw a standard exception.
// However, although in general this approach works well, in certain environments and under
// special circumstances exceptions may not be the mechanism of choice and a different error
// reporting mechanism may be desirable. For this reason, \b Blaze provides several macros,
// which enable the customization of the error reporting mechanism. Via these macros it is
// possible to replace the standard exceptions by some other exception type or a completely
// different approach to report errors.
//
//
// \n \section error_reporting_general_customization Customization of the Reporting Mechanism
// <hr>
//
// In some cases it might be necessary to adapt the entire error reporting mechanism and to
// replace it by some other means to signal failure. The primary macro for this purpose is the
// \c BLAZE_THROW macro:

   \code
   #define BLAZE_THROW( EXCEPTION ) \
      throw EXCEPTION
   \endcode

// This macro represents the default mechanism of the \b Blaze library to report errors of any
// kind. In order to customize the error reporing mechanism all that needs to be done is to
// define the macro prior to including any \b Blaze header file. This will cause the \b Blaze
// specific mechanism to be overridden. The following example demonstrates this by replacing
// exceptions by a call to a \c log() function and a direct call to abort:

   \code
   #define BLAZE_THROW( EXCEPTION ) \
      log( "..." ); \
      abort()

   #include <blaze/Blaze.h>
   \endcode

// Doing this will trigger a call to \c log() and an abort instead of throwing an exception
// whenever an error (such as an invalid argument) is detected.
//
// \note It is possible to execute several statements instead of executing a single statement to
// throw an exception. Also note that it is recommended to define the macro such that a subsequent
// semicolon is required!
//
// \warning This macro is provided with the intention to assist in adapting \b Blaze to special
// conditions and environments. However, the customization of the error reporting mechanism via
// this macro can have a significant effect on the library. Thus be advised to use the macro
// with due care!
//
//
// \n \section error_reporting_exception_customization Customization of the Type of Exceptions
// <hr>
//
// In addition to the customization of the entire error reporting mechanism it is also possible
// to customize the type of exceptions being thrown. This can be achieved by customizing any
// number of the following macros:

   \code
   #define BLAZE_THROW_BAD_ALLOC \
      BLAZE_THROW( std::bad_alloc() )

   #define BLAZE_THROW_LOGIC_ERROR( MESSAGE ) \
      BLAZE_THROW( std::logic_error( MESSAGE ) )

   #define BLAZE_THROW_INVALID_ARGUMENT( MESSAGE ) \
      BLAZE_THROW( std::invalid_argument( MESSAGE ) )

   #define BLAZE_THROW_LENGTH_ERROR( MESSAGE ) \
      BLAZE_THROW( std::length_error( MESSAGE ) )

   #define BLAZE_THROW_OUT_OF_RANGE( MESSAGE ) \
      BLAZE_THROW( std::out_of_range( MESSAGE ) )

   #define BLAZE_THROW_RUNTIME_ERROR( MESSAGE ) \
      BLAZE_THROW( std::runtime_error( MESSAGE ) )
   \endcode

// In order to customize the type of exception the according macro has to be defined prior to
// including any \b Blaze header file. This will override the \b Blaze default behavior. The
// following example demonstrates this by replacing \c std::invalid_argument by a custom
// exception type:

   \code
   class InvalidArgument
   {
    public:
      InvalidArgument();
      explicit InvalidArgument( const std::string& message );
      // ...
   };

   #define BLAZE_THROW_INVALID_ARGUMENT( MESSAGE ) \
      BLAZE_THROW( InvalidArgument( MESSAGE ) )

   #include <blaze/Blaze.h>
   \endcode

// By manually defining the macro, an \c InvalidArgument exception is thrown instead of a
// \c std::invalid_argument exception. Note that it is recommended to define the macro such
// that a subsequent semicolon is required!
//
// \warning These macros are provided with the intention to assist in adapting \b Blaze to
// special conditions and environments. However, the customization of the type of an exception
// via this macro may have an effect on the library. Thus be advised to use the macro with due
// care!
//
//
// \n \section error_reporting_special_errors Customization of Special Errors
// <hr>
//
// Last but not least it is possible to customize the error reporting for special kinds of errors.
// This can be achieved by customizing any number of the following macros:

   \code
   #define BLAZE_THROW_DIVISION_BY_ZERO( MESSAGE ) \
      BLAZE_THROW_RUNTIME_ERROR( MESSAGE )

   #define BLAZE_THROW_LAPACK_ERROR( MESSAGE ) \
      BLAZE_THROW_RUNTIME_ERROR( MESSAGE )
   \endcode

// As explained in the previous sections, in order to customize the handling of special errors
// the according macro has to be defined prior to including any \b Blaze header file. This will
// override the \b Blaze default behavior.
//
//
// \n Previous: \ref custom_data_types &nbsp; &nbsp; Next: \ref intra_statement_optimization \n
*/
//*************************************************************************************************


//**Intra-Statement Optimization*******************************************************************
/*!\page intra_statement_optimization Intra-Statement Optimization
//
// One of the prime features of the \b Blaze library is the automatic intra-statement optimization.
// In order to optimize the overall performance of every single statement \b Blaze attempts to
// rearrange the operands based on their types. For instance, the following addition of dense and
// sparse vectors

   \code
   blaze::DynamicVector<double> d1, d2, d3;
   blaze::CompressedVector<double> s1;

   // ... Resizing and initialization

   d3 = d1 + s1 + d2;
   \endcode

// is automatically rearranged and evaluated as

   \code
   // ...
   d3 = d1 + d2 + s1;  // <- Note that s1 and d2 have been rearranged
   \endcode

// This order of operands is highly favorable for the overall performance since the addition of
// the two dense vectors \c d1 and \c d2 can be handled much more efficiently in a vectorized
// fashion.
//
// This intra-statement optimization can have a tremendous effect on the performance of a statement.
// Consider for instance the following computation:

   \code
   blaze::DynamicMatrix<double> A, B;
   blaze::DynamicVector<double> x, y;

   // ... Resizing and initialization

   y = A * B * x;
   \endcode

// Since multiplications are evaluated from left to right, this statement would result in a
// matrix/matrix multiplication, followed by a matrix/vector multiplication. However, if the
// right subexpression is evaluated first, the performance can be dramatically improved since the
// matrix/matrix multiplication can be avoided in favor of a second matrix/vector multiplication.
// The \b Blaze library exploits this by automatically restructuring the expression such that the
// right multiplication is evaluated first:

   \code
   // ...
   y = A * ( B * x );
   \endcode

// Note however that although this intra-statement optimization may result in a measurable or
// even significant performance improvement, this behavior may be undesirable for several reasons,
// for instance because of numerical stability. Therefore, in case the order of evaluation matters,
// the best solution is to be explicit and to separate a statement into several statements:

   \code
   blaze::DynamicVector<double> d1, d2, d3;
   blaze::CompressedVector<double> s1;

   // ... Resizing and initialization

   d3  = d1 + s1;  // Compute the dense vector/sparse vector addition first ...
   d3 += d2;       // ... and afterwards add the second dense vector
   \endcode

   \code
   // ...
   blaze::DynamicMatrix<double> A, B, C;
   blaze::DynamicVector<double> x, y;

   // ... Resizing and initialization

   C = A * B;  // Compute the left-hand side matrix-matrix multiplication first ...
   y = C * x;  // ... before the right-hand side matrix-vector multiplication
   \endcode

// Alternatively, it is also possible to use the \c eval() function to fix the order of evaluation:

   \code
   blaze::DynamicVector<double> d1, d2, d3;
   blaze::CompressedVector<double> s1;

   // ... Resizing and initialization

   d3 = d1 + eval( s1 + d2 );
   \endcode

   \code
   blaze::DynamicMatrix<double> A, B;
   blaze::DynamicVector<double> x, y;

   // ... Resizing and initialization

   y = eval( A * B ) * x;
   \endcode

// \n Previous: \ref error_reporting_customization
*/
//*************************************************************************************************

#endif
