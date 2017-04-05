//=================================================================================================
/*!
//  \file blaze/math/adaptors/UniLowerMatrix.h
//  \brief Header file for the implementation of a lower unitriangular matrix adaptor
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

#ifndef _BLAZE_MATH_ADAPTORS_UNILOWERMATRIX_H_
#define _BLAZE_MATH_ADAPTORS_UNILOWERMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/adaptors/lowermatrix/BaseTemplate.h>
#include <blaze/math/adaptors/unilowermatrix/BaseTemplate.h>
#include <blaze/math/adaptors/unilowermatrix/Dense.h>
#include <blaze/math/adaptors/unilowermatrix/Sparse.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Hermitian.h>
#include <blaze/math/constraints/Lower.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/UniTriangular.h>
#include <blaze/math/constraints/Upper.h>
#include <blaze/math/dense/StaticMatrix.h>
#include <blaze/math/Forward.h>
#include <blaze/math/Functions.h>
#include <blaze/math/lapack/trtri.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsOne.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/ColumnTrait.h>
#include <blaze/math/traits/DerestrictTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/ForEachTrait.h>
#include <blaze/math/traits/MathTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/RowTrait.h>
#include <blaze/math/traits/SubmatrixTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/typetraits/Columns.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/IsAdaptor.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsRestricted.h>
#include <blaze/math/typetraits/IsSquare.h>
#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/math/typetraits/RemoveAdaptor.h>
#include <blaze/math/typetraits/Rows.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  UNILOWERMATRIX OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name UniLowerMatrix operators */
//@{
template< typename MT, bool SO, bool DF >
inline void reset( UniLowerMatrix<MT,SO,DF>& m );

template< typename MT, bool SO, bool DF >
inline void reset( UniLowerMatrix<MT,SO,DF>& m, size_t i );

template< typename MT, bool SO, bool DF >
inline void clear( UniLowerMatrix<MT,SO,DF>& m );

template< typename MT, bool SO, bool DF >
inline bool isDefault( const UniLowerMatrix<MT,SO,DF>& m );

template< typename MT, bool SO, bool DF >
inline bool isIntact( const UniLowerMatrix<MT,SO,DF>& m );

template< typename MT, bool SO, bool DF >
inline void swap( UniLowerMatrix<MT,SO,DF>& a, UniLowerMatrix<MT,SO,DF>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given unilower matrix.
// \ingroup unilower_matrix
//
// \param m The unilower matrix to be resetted.
// \return void
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline void reset( UniLowerMatrix<MT,SO,DF>& m )
{
   m.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the specified row/column of the given unilower matrix.
// \ingroup unilower_matrix
//
// \param m The unilower matrix to be resetted.
// \param i The index of the row/column to be resetted.
// \return void
//
// This function resets the values in the specified row/column of the given unilower matrix to
// their default value. In case the given matrix is a \a rowMajor matrix the function resets the
// values in row \a i, if it is a \a columnMajor matrix the function resets the values in column
// \a i. Note that the capacity of the row/column remains unchanged.
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline void reset( UniLowerMatrix<MT,SO,DF>& m, size_t i )
{
   m.reset( i );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given unilower matrix.
// \ingroup unilower_matrix
//
// \param m The unilower matrix to be cleared.
// \return void
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline void clear( UniLowerMatrix<MT,SO,DF>& m )
{
   m.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given resizable unilower matrix is in default state.
// \ingroup unilower_matrix
//
// \param m The unilower matrix to be tested for its default state.
// \return \a true in case the given matrix is in default state, \a false otherwise.
//
// This function checks whether the resizable lower unitriangular matrix is in default state.
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline bool isDefault_backend( const UniLowerMatrix<MT,SO,DF>& m, TrueType )
{
   return ( m.rows() == 0UL );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given fixed-size unilower matrix is in default state.
// \ingroup unilower_matrix
//
// \param m The unilower matrix to be tested for its default state.
// \return \a true in case the given matrix is in default state, \a false otherwise.
//
// This function checks whether the fixed-size lower unitriangular matrix is in default state.
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline bool isDefault_backend( const UniLowerMatrix<MT,SO,DF>& m, FalseType )
{
   return isIdentity( m );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given unilower matrix is in default state.
// \ingroup unilower_matrix
//
// \param m The unilower matrix to be tested for its default state.
// \return \a true in case the given matrix is in default state, \a false otherwise.
//
// This function checks whether the lower unitriangular matrix is in default state. The following
// example demonstrates the use of the \a isDefault function:

   \code
   using blaze::DynamicMatrix;
   using blaze::UniLowerMatrix;

   UniLowerMatrix< DynamicMatrix<int> > A;
   // ... Resizing and initialization
   if( isDefault( A ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline bool isDefault( const UniLowerMatrix<MT,SO,DF>& m )
{
   return isDefault_backend( m, typename IsResizable<MT>::Type() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given unilower matrix are intact.
// \ingroup unilower_matrix
//
// \param m The unilower matrix to be tested.
// \return \a true in case the given matrix's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the unilower matrix are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   using blaze::DynamicMatrix;
   using blaze::UniLowerMatrix;

   UniLowerMatrix< DynamicMatrix<int> > A;
   // ... Resizing and initialization
   if( isIntact( A ) ) { ... }
   \endcode
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline bool isIntact( const UniLowerMatrix<MT,SO,DF>& m )
{
   return m.isIntact();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two matrices.
// \ingroup unilower_matrix
//
// \param a The first matrix to be swapped.
// \param b The second matrix to be swapped.
// \return void
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline void swap( UniLowerMatrix<MT,SO,DF>& a, UniLowerMatrix<MT,SO,DF>& b ) noexcept
{
   a.swap( b );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given unilower dense \f$ 2 \times 2 \f$ matrix.
// \ingroup unilower_matrix
//
// \param m The unilower dense matrix to be inverted.
// \return void
//
// This function inverts the given unilower dense \f$ 2 \times 2 \f$ matrix via the rule of Sarrus.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invert2x2( UniLowerMatrix<MT,SO,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   BLAZE_INTERNAL_ASSERT( m.rows()    == 2UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( m.columns() == 2UL, "Invalid number of columns detected" );

   DerestrictTrait_<MT> A( derestrict( m ) );

   A(1,0) = -A(1,0);

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given unilower dense \f$ 3 \times 3 \f$ matrix.
// \ingroup unilower_matrix
//
// \param m The unilower dense matrix to be inverted.
// \return void
//
// This function inverts the given unilower dense \f$ 3 \times 3 \f$ matrix via the rule of Sarrus.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invert3x3( UniLowerMatrix<MT,SO,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   BLAZE_INTERNAL_ASSERT( m.rows()    == 3UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( m.columns() == 3UL, "Invalid number of columns detected" );

   typedef ElementType_<MT>  ET;

   const StaticMatrix<ET,3UL,3UL,SO> A( m );
   DerestrictTrait_<MT> B( derestrict( m ) );

   B(1,0) = - A(1,0);
   B(2,0) =   A(1,0)*A(2,1) - A(2,0);
   B(2,1) = - A(2,1);

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given unilower dense \f$ 4 \times 4 \f$ matrix.
// \ingroup unilower_matrix
//
// \param m The unilower dense matrix to be inverted.
// \return void
//
// This function inverts the given unilower dense \f$ 4 \times 4 \f$ matrix via the rule of Sarrus.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invert4x4( UniLowerMatrix<MT,SO,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   BLAZE_INTERNAL_ASSERT( m.rows()    == 4UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( m.columns() == 4UL, "Invalid number of columns detected" );

   typedef ElementType_<MT>  ET;

   const StaticMatrix<ET,4UL,4UL,SO> A( m );
   DerestrictTrait_<MT> B( derestrict( m ) );

   const ET tmp( A(2,1)*A(3,2) - A(3,1) );

   B(1,0) = - A(1,0);
   B(2,0) =   A(1,0)*A(2,1) - A(2,0);
   B(3,0) =   A(2,0)*A(3,2) - A(3,0) - A(1,0)*tmp;
   B(2,1) = - A(2,1);
   B(3,1) =   tmp;
   B(3,2) = - A(3,2);

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given unilower dense \f$ 5 \times 5 \f$ matrix.
// \ingroup unilower_matrix
//
// \param m The unilower dense matrix to be inverted.
// \return void
//
// This function inverts the given unilower dense \f$ 5 \times 5 \f$ matrix via the rule of Sarrus.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invert5x5( UniLowerMatrix<MT,SO,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   BLAZE_INTERNAL_ASSERT( m.rows()    == 5UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( m.columns() == 5UL, "Invalid number of columns detected" );

   typedef ElementType_<MT>  ET;

   const StaticMatrix<ET,5UL,5UL,SO> A( m );
   DerestrictTrait_<MT> B( derestrict( m ) );

   const ET tmp1( A(3,2)*A(4,3) - A(4,2) );
   const ET tmp2( A(2,1)*A(3,2) - A(3,1) );
   const ET tmp3( A(2,1)*tmp1 - A(3,1)*A(4,3) + A(4,1) );

   B(1,0) = - A(1,0);
   B(2,0) =   A(1,0)*A(2,1) - A(2,0);
   B(3,0) = - A(1,0)*tmp2 + A(2,0)*A(3,2) - A(3,0);
   B(4,0) =   A(1,0)*tmp3 - A(2,0)*tmp1 + A(3,0)*A(4,3) - A(4,0);
   B(2,1) = - A(2,1);
   B(3,1) =   tmp2;
   B(4,1) = - tmp3;
   B(3,2) = - A(3,2);
   B(4,2) =   A(4,3)*A(3,2) - A(4,2);
   B(4,3) = - A(4,3);

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given unilower dense \f$ 6 \times 6 \f$ matrix.
// \ingroup unilower_matrix
//
// \param m The unilower dense matrix to be inverted.
// \return void
//
// This function inverts the given unilower dense \f$ 6 \times 6 \f$ matrix via the rule of Sarrus.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invert6x6( UniLowerMatrix<MT,SO,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   BLAZE_INTERNAL_ASSERT( m.rows()    == 6UL, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( m.columns() == 6UL, "Invalid number of columns detected" );

   typedef ElementType_<MT>  ET;

   const StaticMatrix<ET,6UL,6UL,SO> A( m );
   DerestrictTrait_<MT> B( derestrict( m ) );

   const ET tmp1( A(4,3)*A(5,4) - A(5,3) );
   const ET tmp2( A(3,2)*A(4,3) - A(4,2) );
   const ET tmp3( A(3,2)*tmp1 - A(4,2)*A(5,4) + A(5,2) );
   const ET tmp4( A(2,1)*A(3,2) - A(3,1) );
   const ET tmp5( A(2,1)*tmp2 - A(3,1)*A(4,3) + A(4,1) );
   const ET tmp6( A(2,1)*tmp3 - A(3,1)*tmp1 + A(4,1)*A(5,4) - A(5,1) );

   B(1,0) = - A(1,0);
   B(2,0) =   A(1,0)*A(2,1) - A(2,0);
   B(3,0) = - A(1,0)*tmp4 + A(2,0)*A(3,2) - A(3,0);
   B(4,0) =   A(1,0)*tmp5 - A(2,0)*tmp2 + A(3,0)*A(4,3) - A(4,0);
   B(5,0) = - A(1,0)*tmp6 + A(2,0)*tmp3 - A(3,0)*tmp1 + A(4,0)*A(5,4) - A(5,0);
   B(2,1) = - A(2,1);
   B(3,1) =   tmp4;
   B(4,1) = - tmp5;
   B(5,1) =   tmp6;
   B(3,2) = - A(3,2);
   B(4,2) =   tmp2;
   B(5,2) = - tmp3;
   B(4,3) =   A(5,3)*A(4,5) - A(4,3);
   B(5,3) =   A(5,4)*A(4,3) - A(5,3);
   B(5,4) =   A(5,3)*A(3,4) - A(5,4);

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place inversion of the given unilower dense matrix.
// \ingroup unilower_matrix
//
// \param m The unilower dense matrix to be inverted.
// \return void
//
// This function inverts the given unilower dense matrix by means of the most suited matrix
// inversion algorithm.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a linker error will be created.
//
// \note This function does only provide the basic exception safety guarantee, i.e. in case of an
// exception \a m may already have been modified.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invertByDefault( UniLowerMatrix<MT,SO,true>& m )
{
   invertByLU( m );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place LU-based inversion of the given unilower dense matrix.
// \ingroup unilower_matrix
//
// \param m The unilower dense matrix to be inverted.
// \return void
//
// This function inverts the given unilower dense matrix by means of an LU decomposition.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a linker error will be created.
//
// \note This function does only provide the basic exception safety guarantee, i.e. in case of an
// exception \a m may already have been modified.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invertByLU( UniLowerMatrix<MT,SO,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   DerestrictTrait_<MT> A( derestrict( ~m ) );

   trtri( A, 'L', 'U' );

   BLAZE_INTERNAL_ASSERT( isIntact( m ), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place Bunch-Kaufman-based inversion of the given unilower dense matrix.
// \ingroup unilower_matrix
//
// \param m The unilower dense matrix to be inverted.
// \return void
//
// This function inverts the given unilower dense matrix by means of a Bunch-Kaufman decomposition.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a linker error will be created.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invertByLDLT( UniLowerMatrix<MT,SO,true>& m )
{
   invertByLLH( m );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place Bunch-Kaufman-based inversion of the given unilower dense matrix.
// \ingroup unilower_matrix
//
// \param m The unilower dense matrix to be inverted.
// \return void
//
// This function inverts the given unilower dense matrix by means of a Bunch-Kaufman decomposition.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
//
// \note This function can only be used if the fitting LAPACK library is available and linked to
// the executable. Otherwise a linker error will be created.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invertByLDLH( UniLowerMatrix<MT,SO,true>& m )
{
   invertByLLH( m );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place Cholesky-based inversion of the given unilower dense matrix.
// \ingroup unilower_matrix
//
// \param m The unilower dense matrix to be inverted.
// \return void
//
// This function inverts the given unilower dense matrix by means of a Cholesky decomposition.
//
// \note The matrix inversion can only be used for dense matrices with \c float, \c double,
// \c complex<float> or \c complex<double> element type. The attempt to call the function with
// matrices of any other element type results in a compile time error!
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void invertByLLH( UniLowerMatrix<MT,SO,true>& m )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT> );

   BLAZE_INTERNAL_ASSERT( isIdentity( ~m ), "Violation of preconditions detected" );

   UNUSED_PARAMETER( m );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief LU decomposition of the given unilower dense matrix.
// \ingroup unilower_matrix
//
// \param A The unilower matrix to be decomposed.
// \param L The resulting lower triangular matrix.
// \param U The resulting upper triangular matrix.
// \param P The resulting permutation matrix.
// \return void
//
// This function performs the dense matrix (P)LU decomposition of a unilower n-by-n matrix. The
// resulting decomposition is written to the three distinct matrices \c L, \c U, and \c P, which
// are resized to the correct dimensions (if possible and necessary).
//
// \note The LU decomposition will never fail, even for singular matrices. However, in case of a
// singular matrix the resulting decomposition cannot be used for a matrix inversion or solving
// a linear system of equations.
*/
template< typename MT1, bool SO1, typename MT2, typename MT3, typename MT4, bool SO2 >
inline void lu( const UniLowerMatrix<MT1,SO1,true>& A, DenseMatrix<MT2,SO1>& L,
                DenseMatrix<MT3,SO1>& U, Matrix<MT4,SO2>& P )
{
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_<MT1> );

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_HERMITIAN_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_UPPER_MATRIX_TYPE( MT2 );

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT3 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_HERMITIAN_MATRIX_TYPE( MT3 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT3 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_LOWER_MATRIX_TYPE( MT3 );

   typedef ElementType_<MT3>  ET3;
   typedef ElementType_<MT4>  ET4;

   const size_t n( (~A).rows() );

   DerestrictTrait_<MT3> U2( derestrict( ~U ) );

   (~L) = A;

   resize( ~U, n, n );
   reset( U2 );

   resize( ~P, n, n );
   reset( ~P );

   for( size_t i=0UL; i<n; ++i ) {
      U2(i,i)   = ET3(1);
      (~P)(i,i) = ET4(1);
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a dense vector to an unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side dense vector to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the adapted matrix
        , bool SO        // Storage order of the adapted matrix
        , bool DF        // Density flag
        , typename VT >  // Type of the right-hand side dense vector
inline bool tryAssign( const UniLowerMatrix<MT,SO,DF>& lhs,
                       const DenseVector<VT,false>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.rows() - row, "Invalid number of rows" );

   UNUSED_PARAMETER( lhs );

   if( column < row )
      return true;

   const bool containsDiagonal( column < row + (~rhs).size() );
   const size_t iend( min( column - row, (~rhs).size() ) );

   for( size_t i=0UL; i<iend; ++i ) {
      if( !isDefault( (~rhs)[i] ) )
         return false;
   }

   if( containsDiagonal && !isOne( (~rhs)[iend] ) )
      return false;

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a dense vector to an unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side dense vector to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the adapted matrix
        , bool SO        // Storage order of the adapted matrix
        , bool DF        // Density flag
        , typename VT >  // Type of the right-hand side dense vector
inline bool tryAssign( const UniLowerMatrix<MT,SO,DF>& lhs,
                       const DenseVector<VT,true>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   if( row >= column + (~rhs).size() )
      return true;

   const bool containsDiagonal( row >= column );
   const size_t ibegin( ( !containsDiagonal )?( 0UL ):( row - column + 1UL ) );

   if( containsDiagonal && !isOne( (~rhs)[row-column] ) )
      return false;

   for( size_t i=ibegin; i<(~rhs).size(); ++i ) {
      if( !isDefault( (~rhs)[i] ) )
         return false;
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a sparse vector to an unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side sparse vector to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the adapted matrix
        , bool SO        // Storage order of the adapted matrix
        , bool DF        // Density flag
        , typename VT >  // Type of the right-hand side sparse vector
inline bool tryAssign( const UniLowerMatrix<MT,SO,DF>& lhs,
                       const SparseVector<VT,false>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.rows() - row, "Invalid number of rows" );

   UNUSED_PARAMETER( lhs );

   typedef typename VT::ConstIterator  RhsIterator;

   if( column < row )
      return true;

   const bool containsDiagonal( column < row + (~rhs).size() );
   const size_t index( column - row );
   const RhsIterator last( (~rhs).lowerBound( index ) );

   if( containsDiagonal ) {
      if( last == (~rhs).end() || last->index() != index || !isOne( last->value() ) )
         return false;
   }

   for( RhsIterator element=(~rhs).begin(); element!=last; ++element ) {
      if( !isDefault( element->value() ) )
         return false;
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a sparse vector to an unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side sparse vector to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the adapted matrix
        , bool SO        // Storage order of the adapted matrix
        , bool DF        // Density flag
        , typename VT >  // Type of the right-hand side sparse vector
inline bool tryAssign( const UniLowerMatrix<MT,SO,DF>& lhs,
                       const SparseVector<VT,true>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   typedef typename VT::ConstIterator  RhsIterator;

   if( row >= column + (~rhs).size() )
      return true;

   const bool containsDiagonal( row >= column );
   const size_t index( ( containsDiagonal )?( row - column ):( 0UL ) );
   const RhsIterator last( (~rhs).end() );
   RhsIterator element( (~rhs).lowerBound( index ) );

   if( containsDiagonal ) {
      if( element == last || element->index() != index || !isOne( element->value() ) )
         return false;
      ++element;
   }

   for( ; element!=last; ++element ) {
      if( !isDefault( element->value() ) )
         return false;
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a dense matrix to an unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side dense matrix to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1    // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF         // Density flag
        , typename MT2 >  // Type of the right-hand side dense matrix
inline bool tryAssign( const UniLowerMatrix<MT1,SO,DF>& lhs,
                       const DenseMatrix<MT2,false>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT2 );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= lhs.rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   const size_t M( (~rhs).rows()    );
   const size_t N( (~rhs).columns() );

   if( row + 1UL >= column + N )
      return true;

   const size_t iend( min( column + N - row, M ) );

   for( size_t i=0UL; i<iend; ++i )
   {
      const bool containsDiagonal( row + i >= column );

      if( containsDiagonal && !isOne( (~rhs)(i,row+i-column) ) )
         return false;

      const size_t jbegin( ( containsDiagonal )?( row + i - column + 1UL ):( 0UL ) );

      for( size_t j=jbegin; j<N; ++j ) {
         if( !isDefault( (~rhs)(i,j) ) )
            return false;
      }
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a dense matrix to an unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side dense matrix to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1    // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF         // Density flag
        , typename MT2 >  // Type of the right-hand side dense matrix
inline bool tryAssign( const UniLowerMatrix<MT1,SO,DF>& lhs,
                       const DenseMatrix<MT2,true>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT2 );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= lhs.rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   const size_t M( (~rhs).rows()    );
   const size_t N( (~rhs).columns() );

   if( row + 1UL >= column + N )
      return true;

   const size_t jbegin( ( row < column )?( 0UL ):( row - column ) );

   for( size_t j=jbegin; j<N; ++j )
   {
      const size_t iend( min( column + j - row, M ) );

      for( size_t i=0UL; i<iend; ++i ) {
         if( !isDefault( (~rhs)(i,j) ) )
            return false;
      }

	  const bool containsDiagonal( column + j < row + M );

      if( containsDiagonal && !isOne( (~rhs)(iend,j) ) )
         return false;
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a sparse matrix to an unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side sparse matrix to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1    // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF         // Density flag
        , typename MT2 >  // Type of the right-hand side sparse matrix
inline bool tryAssign( const UniLowerMatrix<MT1,SO,DF>& lhs,
                       const SparseMatrix<MT2,false>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT2 );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= lhs.rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   typedef typename MT2::ConstIterator  RhsIterator;

   const size_t M( (~rhs).rows()    );
   const size_t N( (~rhs).columns() );

   if( row + 1UL >= column + N )
      return true;

   const size_t iend( min( column + N - row, M ) );

   for( size_t i=0UL; i<iend; ++i )
   {
      const bool containsDiagonal( row + i >= column );
      const size_t index( ( containsDiagonal )?( row + i - column ):( 0UL ) );

      const RhsIterator last( (~rhs).end(i) );
      RhsIterator element( (~rhs).lowerBound( i, index ) );

      if( containsDiagonal ) {
         if( element == last || ( element->index() != index ) || !isOne( element->value() ) )
            return false;
         ++element;
      }

      for( ; element!=last; ++element ) {
         if( !isDefault( element->value() ) )
            return false;
      }
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the assignment of a sparse matrix to an unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side sparse matrix to be assigned.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1    // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF         // Density flag
        , typename MT2 >  // Type of the right-hand side sparse matrix
inline bool tryAssign( const UniLowerMatrix<MT1,SO,DF>& lhs,
                       const SparseMatrix<MT2,true>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT2 );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= lhs.rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   typedef typename MT2::ConstIterator  RhsIterator;

   const size_t M( (~rhs).rows()    );
   const size_t N( (~rhs).columns() );

   if( row + 1UL >= column + N )
      return true;

   const size_t jbegin( ( row < column )?( 0UL ):( row - column ) );

   for( size_t j=jbegin; j<N; ++j )
   {
      const bool containsDiagonal( column + j < row + M );

      const size_t index( column + j - row );
      const RhsIterator last( (~rhs).lowerBound( min( index, M ), j ) );

      if( containsDiagonal ) {
         if( last == (~rhs).end(j) || ( last->index() != index ) || !isOne( last->value() ) )
            return false;
      }

      for( RhsIterator element=(~rhs).begin(j); element!=last; ++element ) {
         if( !isDefault( element->value() ) )
            return false;
      }
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a dense vector to an
//        unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side dense vector to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the adapted matrix
        , bool SO        // Storage order of the adapted matrix
        , bool DF        // Density flag
        , typename VT >  // Type of the right-hand side dense vector
inline bool tryAddAssign( const UniLowerMatrix<MT,SO,DF>& lhs,
                          const DenseVector<VT,false>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.rows() - row, "Invalid number of rows" );

   UNUSED_PARAMETER( lhs );

   if( column < row )
      return true;

   const size_t iend( min( column - row + 1UL, (~rhs).size() ) );

   for( size_t i=0UL; i<iend; ++i ) {
      if( !isDefault( (~rhs)[i] ) )
         return false;
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a dense vector to an
//        unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side dense vector to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the adapted matrix
        , bool SO        // Storage order of the adapted matrix
        , bool DF        // Density flag
        , typename VT >  // Type of the right-hand side dense vector
inline bool tryAddAssign( const UniLowerMatrix<MT,SO,DF>& lhs,
                          const DenseVector<VT,true>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   const size_t ibegin( ( row <= column )?( 0UL ):( row - column ) );

   for( size_t i=ibegin; i<(~rhs).size(); ++i ) {
      if( !isDefault( (~rhs)[i] ) )
         return false;
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a sparse vector to an
//        unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side sparse vector to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the adapted matrix
        , bool SO        // Storage order of the adapted matrix
        , bool DF        // Density flag
        , typename VT >  // Type of the right-hand side sparse vector
inline bool tryAddAssign( const UniLowerMatrix<MT,SO,DF>& lhs,
                          const SparseVector<VT,false>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.rows() - row, "Invalid number of rows" );

   UNUSED_PARAMETER( lhs );

   typedef typename VT::ConstIterator  RhsIterator;

   if( column < row )
      return true;

   const RhsIterator last( (~rhs).lowerBound( column - row + 1UL ) );

   for( RhsIterator element=(~rhs).begin(); element!=last; ++element ) {
      if( !isDefault( element->value() ) )
         return false;
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a sparse vector to an
//        unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side sparse vector to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the adapted matrix
        , bool SO        // Storage order of the adapted matrix
        , bool DF        // Density flag
        , typename VT >  // Type of the right-hand side sparse vector
inline bool tryAddAssign( const UniLowerMatrix<MT,SO,DF>& lhs,
                          const SparseVector<VT,true>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   typedef typename VT::ConstIterator  RhsIterator;

   const RhsIterator last( (~rhs).end() );
   RhsIterator element( (~rhs).lowerBound( ( row <= column )?( 0UL ):( row - column ) ) );

   for( ; element!=last; ++element ) {
      if( !isDefault( element->value() ) )
         return false;
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a dense matrix to an
//        unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side dense matrix to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1    // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF         // Density flag
        , typename MT2 >  // Type of the right-hand side dense matrix
inline bool tryAddAssign( const UniLowerMatrix<MT1,SO,DF>& lhs,
                          const DenseMatrix<MT2,false>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT2 );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= lhs.rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   const size_t M( (~rhs).rows()    );
   const size_t N( (~rhs).columns() );

   if( row + 1UL >= column + N )
      return true;

   const size_t iend( min( column + N - row, M ) );

   for( size_t i=0UL; i<iend; ++i )
   {
      const bool containsDiagonal( row + i >= column );
      const size_t jbegin( ( containsDiagonal )?( row + i - column ):( 0UL ) );

      for( size_t j=jbegin; j<N; ++j ) {
         if( !isDefault( (~rhs)(i,j) ) )
            return false;
      }
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a dense matrix to an
//        unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side dense matrix to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1    // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF         // Density flag
        , typename MT2 >  // Type of the right-hand side dense matrix
inline bool tryAddAssign( const UniLowerMatrix<MT1,SO,DF>& lhs,
                          const DenseMatrix<MT2,true>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT2 );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= lhs.rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   const size_t M( (~rhs).rows()    );
   const size_t N( (~rhs).columns() );

   if( row + 1UL >= column + N )
      return true;

   const size_t jbegin( ( row <= column )?( 0UL ):( row - column ) );

   for( size_t j=jbegin; j<N; ++j )
   {
      const size_t iend( min( column + j - row + 1UL, M ) );

      for( size_t i=0UL; i<iend; ++i ) {
         if( !isDefault( (~rhs)(i,j) ) )
            return false;
      }
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a sparse matrix to an
//        unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side sparse matrix to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1    // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF         // Density flag
        , typename MT2 >  // Type of the right-hand side sparse matrix
inline bool tryAddAssign( const UniLowerMatrix<MT1,SO,DF>& lhs,
                          const SparseMatrix<MT2,false>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT2 );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= lhs.rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   typedef typename MT2::ConstIterator  RhsIterator;

   const size_t M( (~rhs).rows()    );
   const size_t N( (~rhs).columns() );

   if( row + 1UL >= column + N )
      return true;

   const size_t iend( min( column + N - row, M ) );

   for( size_t i=0UL; i<iend; ++i )
   {
      const bool containsDiagonal( row + i >= column );
      const size_t index( ( containsDiagonal )?( row + i - column ):( 0UL ) );

      const RhsIterator last( (~rhs).end(i) );
      RhsIterator element( (~rhs).lowerBound( i, index ) );

      for( ; element!=last; ++element ) {
         if( !isDefault( element->value() ) )
            return false;
      }
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the addition assignment of a sparse matrix to an
//        unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side sparse matrix to be added.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1    // Type of the adapted matrix
        , bool SO         // Storage order of the adapted matrix
        , bool DF         // Density flag
        , typename MT2 >  // Type of the right-hand side sparse matrix
inline bool tryAddAssign( const UniLowerMatrix<MT1,SO,DF>& lhs,
                          const SparseMatrix<MT2,true>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT2 );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() <= lhs.rows() - row, "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( (~rhs).columns() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   typedef typename MT2::ConstIterator  RhsIterator;

   const size_t M( (~rhs).rows()    );
   const size_t N( (~rhs).columns() );

   if( row + 1UL >= column + N )
      return true;

   const size_t jbegin( ( row < column )?( 0UL ):( row - column ) );

   for( size_t j=jbegin; j<N; ++j )
   {
      const size_t index( column + j - row + 1UL );
      const RhsIterator last( (~rhs).lowerBound( min( index, M ), j ) );

      for( RhsIterator element=(~rhs).begin(j); element!=last; ++element ) {
         if( !isDefault( element->value() ) )
            return false;
      }
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a vector to an unilower
//        matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side vector to be subtracted.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF      // Density flag
        , typename VT  // Type of the right-hand side vector
        , bool TF >    // Transpose flag of the right-hand side vector
inline bool trySubAssign( const UniLowerMatrix<MT,SO,DF>& lhs,
                          const Vector<VT,TF>& rhs, size_t row, size_t column )
{
   return tryAddAssign( lhs, ~rhs, row, column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the subtraction assignment of a matrix to an unilower
//        matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side matrix to be subtracted.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the adapted matrix
        , bool SO1      // Storage order of the adapted matrix
        , bool DF       // Density flag
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline bool trySubAssign( const UniLowerMatrix<MT1,SO1,DF>& lhs,
                          const Matrix<MT2,SO2>& rhs, size_t row, size_t column )
{
   return tryAddAssign( lhs, ~rhs, row, column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to an
//        unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side vector to be multiplied.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the adapted matrix
        , bool SO        // Storage order of the adapted matrix
        , bool DF        // Density flag
        , typename VT >  // Type of the right-hand side vector
inline bool tryMultAssign( const UniLowerMatrix<MT,SO,DF>& lhs,
                           const Vector<VT,false>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.rows() - row, "Invalid number of rows" );

   UNUSED_PARAMETER( lhs );

   return ( column < row || (~rhs).size() <= column - row || isOne( (~rhs)[column-row] ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the multiplication assignment of a vector to an
//        unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side vector to be multiplied.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT    // Type of the adapted matrix
        , bool SO        // Storage order of the adapted matrix
        , bool DF        // Density flag
        , typename VT >  // Type of the right-hand side vector
inline bool tryMultAssign( const UniLowerMatrix<MT,SO,DF>& lhs,
                           const Vector<VT,true>& rhs, size_t row, size_t column )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( VT );

   BLAZE_INTERNAL_ASSERT( row <= lhs.rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( column <= lhs.columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( (~rhs).size() <= lhs.columns() - column, "Invalid number of columns" );

   UNUSED_PARAMETER( lhs );

   return ( row < column || (~rhs).size() <= row - column || isOne( (~rhs)[row-column] ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Predict invariant violations by the division assignment of a vector to an unilower matrix.
// \ingroup unilower_matrix
//
// \param lhs The target left-hand side unilower matrix.
// \param rhs The right-hand side vector divisor.
// \param row The row index of the first element to be modified.
// \param column The column index of the first element to be modified.
// \return \a true in case the assignment would be successful, \a false if not.
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF      // Density flag
        , typename VT  // Type of the right-hand side vector
        , bool TF >    // Transpose flag of the right-hand side vector
inline bool tryDivAssign( const UniLowerMatrix<MT,SO,DF>& lhs,
                          const Vector<VT,TF>& rhs, size_t row, size_t column )
{
   return tryMultAssign( lhs, ~rhs, row, column );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns a reference to the instance without the access restrictions to the upper part.
// \ingroup math_shims
//
// \param m The unilower matrix to be derestricted.
// \return Reference to the matrix without access restrictions.
//
// This function returns a reference to the given unilower matrix instance that has no access
// restrictions to the upper part of the matrix.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in the violation of invariants, erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the adapted matrix
        , bool SO      // Storage order of the adapted matrix
        , bool DF >    // Density flag
inline MT& derestrict( UniLowerMatrix<MT,SO,DF>& m )
{
   return m.matrix_;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ROWS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF >
struct Rows< UniLowerMatrix<MT,SO,DF> > : public Rows<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COLUMNS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF >
struct Columns< UniLowerMatrix<MT,SO,DF> > : public Columns<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSQUARE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF >
struct IsSquare< UniLowerMatrix<MT,SO,DF> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISUNILOWER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF >
struct IsUniLower< UniLowerMatrix<MT,SO,DF> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISADAPTOR SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF >
struct IsAdaptor< UniLowerMatrix<MT,SO,DF> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISRESTRICTED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF >
struct IsRestricted< UniLowerMatrix<MT,SO,DF> > : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HASCONSTDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO >
struct HasConstDataAccess< UniLowerMatrix<MT,SO,true> > : public TrueType
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
template< typename MT, bool SO, bool DF >
struct IsAligned< UniLowerMatrix<MT,SO,DF> > : public BoolConstant< IsAligned<MT>::value >
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
template< typename MT, bool SO, bool DF >
struct IsPadded< UniLowerMatrix<MT,SO,DF> > : public BoolConstant< IsPadded<MT>::value >
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
template< typename MT, bool SO, bool DF >
struct IsResizable< UniLowerMatrix<MT,SO,DF> > : public BoolConstant< IsResizable<MT>::value >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  REMOVEADAPTOR SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF >
struct RemoveAdaptor< UniLowerMatrix<MT,SO,DF> >
{
   using Type = MT;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DERESTRICTTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF >
struct DerestrictTrait< UniLowerMatrix<MT,SO,DF> >
{
   using Type = MT&;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ADDTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO1, bool DF, typename T, size_t M, size_t N, bool SO2 >
struct AddTrait< UniLowerMatrix<MT,SO1,DF>, StaticMatrix<T,M,N,SO2> >
{
   using Type = AddTrait_< MT, StaticMatrix<T,M,N,SO2> >;
};

template< typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2, bool DF >
struct AddTrait< StaticMatrix<T,M,N,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = AddTrait_< StaticMatrix<T,M,N,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, typename T, size_t M, size_t N, bool SO2 >
struct AddTrait< UniLowerMatrix<MT,SO1,DF>, HybridMatrix<T,M,N,SO2> >
{
   using Type = AddTrait_< MT, HybridMatrix<T,M,N,SO2> >;
};

template< typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2, bool DF >
struct AddTrait< HybridMatrix<T,M,N,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = AddTrait_< HybridMatrix<T,M,N,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, typename T, bool SO2 >
struct AddTrait< UniLowerMatrix<MT,SO1,DF>, DynamicMatrix<T,SO2> >
{
   using Type = AddTrait_< MT, DynamicMatrix<T,SO2> >;
};

template< typename T, bool SO1, typename MT, bool SO2, bool DF >
struct AddTrait< DynamicMatrix<T,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = AddTrait_< DynamicMatrix<T,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, typename T, bool AF, bool PF, bool SO2 >
struct AddTrait< UniLowerMatrix<MT,SO1,DF>, CustomMatrix<T,AF,PF,SO2> >
{
   using Type = AddTrait_< MT, CustomMatrix<T,AF,PF,SO2> >;
};

template< typename T, bool AF, bool PF, bool SO1, typename MT, bool SO2, bool DF >
struct AddTrait< CustomMatrix<T,AF,PF,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = AddTrait_< CustomMatrix<T,AF,PF,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, typename T, bool SO2 >
struct AddTrait< UniLowerMatrix<MT,SO1,DF>, CompressedMatrix<T,SO2> >
{
   using Type = AddTrait_< MT, CompressedMatrix<T,SO2> >;
};

template< typename T, bool SO1, typename MT, bool SO2, bool DF >
struct AddTrait< CompressedMatrix<T,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = AddTrait_< CompressedMatrix<T,SO1>, MT >;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2, bool NF >
struct AddTrait< UniLowerMatrix<MT1,SO1,DF1>, SymmetricMatrix<MT2,SO2,DF2,NF> >
{
   using Type = AddTrait_<MT1,MT2>;
};

template< typename MT1, bool SO1, bool DF1, bool NF, typename MT2, bool SO2, bool DF2 >
struct AddTrait< SymmetricMatrix<MT1,SO1,DF1,NF>, UniLowerMatrix<MT2,SO2,DF2> >
{
   using Type = AddTrait_<MT1,MT2>;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct AddTrait< UniLowerMatrix<MT1,SO1,DF1>, HermitianMatrix<MT2,SO2,DF2> >
{
   using Type = AddTrait_<MT1,MT2>;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct AddTrait< HermitianMatrix<MT1,SO1,DF1>, UniLowerMatrix<MT2,SO2,DF2> >
{
   using Type = AddTrait_<MT1,MT2>;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct AddTrait< UniLowerMatrix<MT1,SO1,DF1>, LowerMatrix<MT2,SO2,DF2> >
{
   using Type = LowerMatrix< AddTrait_<MT1,MT2> >;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct AddTrait< LowerMatrix<MT1,SO1,DF1>, UniLowerMatrix<MT2,SO2,DF2> >
{
   using Type = LowerMatrix< AddTrait_<MT1,MT2> >;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct AddTrait< UniLowerMatrix<MT1,SO1,DF1>, UniLowerMatrix<MT2,SO2,DF2> >
{
   using Type = LowerMatrix< AddTrait_<MT1,MT2> >;
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
template< typename MT, bool SO1, bool DF, typename T, size_t M, size_t N, bool SO2 >
struct SubTrait< UniLowerMatrix<MT,SO1,DF>, StaticMatrix<T,M,N,SO2> >
{
   using Type = SubTrait_< MT, StaticMatrix<T,M,N,SO2> >;
};

template< typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2, bool DF >
struct SubTrait< StaticMatrix<T,M,N,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = SubTrait_< StaticMatrix<T,M,N,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, typename T, size_t M, size_t N, bool SO2 >
struct SubTrait< UniLowerMatrix<MT,SO1,DF>, HybridMatrix<T,M,N,SO2> >
{
   using Type = SubTrait_< MT, HybridMatrix<T,M,N,SO2> >;
};

template< typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2, bool DF >
struct SubTrait< HybridMatrix<T,M,N,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = SubTrait_< HybridMatrix<T,M,N,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, typename T, bool SO2 >
struct SubTrait< UniLowerMatrix<MT,SO1,DF>, DynamicMatrix<T,SO2> >
{
   using Type = SubTrait_< MT, DynamicMatrix<T,SO2> >;
};

template< typename T, bool SO1, typename MT, bool SO2, bool DF >
struct SubTrait< DynamicMatrix<T,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = SubTrait_< DynamicMatrix<T,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, typename T, bool AF, bool PF, bool SO2 >
struct SubTrait< UniLowerMatrix<MT,SO1,DF>, CustomMatrix<T,AF,PF,SO2> >
{
   using Type = SubTrait_< MT, CustomMatrix<T,AF,PF,SO2> >;
};

template< typename T, bool AF, bool PF, bool SO1, typename MT, bool SO2, bool DF >
struct SubTrait< CustomMatrix<T,AF,PF,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = SubTrait_< CustomMatrix<T,AF,PF,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, typename T, bool SO2 >
struct SubTrait< UniLowerMatrix<MT,SO1,DF>, CompressedMatrix<T,SO2> >
{
   using Type = SubTrait_< MT, CompressedMatrix<T,SO2> >;
};

template< typename T, bool SO1, typename MT, bool SO2, bool DF >
struct SubTrait< CompressedMatrix<T,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = SubTrait_< CompressedMatrix<T,SO1>, MT >;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2, bool NF >
struct SubTrait< UniLowerMatrix<MT1,SO1,DF1>, SymmetricMatrix<MT2,SO2,DF2,NF> >
{
   using Type = SubTrait_<MT1,MT2>;
};

template< typename MT1, bool SO1, bool DF1, bool NF, typename MT2, bool SO2, bool DF2 >
struct SubTrait< SymmetricMatrix<MT1,SO1,DF1,NF>, UniLowerMatrix<MT2,SO2,DF2> >
{
   using Type = SubTrait_<MT1,MT2>;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct SubTrait< UniLowerMatrix<MT1,SO1,DF1>, HermitianMatrix<MT2,SO2,DF2> >
{
   using Type = SubTrait_<MT1,MT2>;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct SubTrait< HermitianMatrix<MT1,SO1,DF1>, UniLowerMatrix<MT2,SO2,DF2> >
{
   using Type = SubTrait_<MT1,MT2>;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct SubTrait< UniLowerMatrix<MT1,SO1,DF1>, LowerMatrix<MT2,SO2,DF2> >
{
   using Type = LowerMatrix< SubTrait_<MT1,MT2> >;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct SubTrait< LowerMatrix<MT1,SO1,DF1>, UniLowerMatrix<MT2,SO2,DF2> >
{
   using Type = LowerMatrix< SubTrait_<MT1,MT2> >;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct SubTrait< UniLowerMatrix<MT1,SO1,DF1>, UniLowerMatrix<MT2,SO2,DF2> >
{
   using Type = LowerMatrix< SubTrait_<MT1,MT2> >;
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
template< typename MT, bool SO, bool DF, typename T >
struct MultTrait< UniLowerMatrix<MT,SO,DF>, T, EnableIf_< IsNumeric<T> > >
{
   using Type = LowerMatrix< MultTrait_<MT,T> >;
};

template< typename T, typename MT, bool SO, bool DF >
struct MultTrait< T, UniLowerMatrix<MT,SO,DF>, EnableIf_< IsNumeric<T> > >
{
   using Type = LowerMatrix< MultTrait_<T,MT> >;
};

template< typename MT, bool SO, bool DF, typename T, size_t N >
struct MultTrait< UniLowerMatrix<MT,SO,DF>, StaticVector<T,N,false> >
{
   using Type = MultTrait_< MT, StaticVector<T,N,false> >;
};

template< typename T, size_t N, typename MT, bool SO, bool DF >
struct MultTrait< StaticVector<T,N,true>, UniLowerMatrix<MT,SO,DF> >
{
   using Type = MultTrait_< StaticVector<T,N,true>, MT >;
};

template< typename MT, bool SO, bool DF, typename T, size_t N >
struct MultTrait< UniLowerMatrix<MT,SO,DF>, HybridVector<T,N,false> >
{
   using Type = MultTrait_< MT, HybridVector<T,N,false> >;
};

template< typename T, size_t N, typename MT, bool SO, bool DF >
struct MultTrait< HybridVector<T,N,true>, UniLowerMatrix<MT,SO,DF> >
{
   using Type = MultTrait_< HybridVector<T,N,true>, MT >;
};

template< typename MT, bool SO, bool DF, typename T >
struct MultTrait< UniLowerMatrix<MT,SO,DF>, DynamicVector<T,false> >
{
   using Type = MultTrait_< MT, DynamicVector<T,false> >;
};

template< typename T, typename MT, bool SO, bool DF >
struct MultTrait< DynamicVector<T,true>, UniLowerMatrix<MT,SO,DF> >
{
   using Type = MultTrait_< DynamicVector<T,true>, MT >;
};

template< typename MT, bool SO, bool DF, typename T, bool AF, bool PF >
struct MultTrait< UniLowerMatrix<MT,SO,DF>, CustomVector<T,AF,PF,false> >
{
   using Type = MultTrait_< MT, CustomVector<T,AF,PF,false> >;
};

template< typename T, bool AF, bool PF, typename MT, bool SO, bool DF >
struct MultTrait< CustomVector<T,AF,PF,true>, UniLowerMatrix<MT,SO,DF> >
{
   using Type = MultTrait_< CustomVector<T,AF,PF,true>, MT >;
};

template< typename MT, bool SO, bool DF, typename T >
struct MultTrait< UniLowerMatrix<MT,SO,DF>, CompressedVector<T,false> >
{
   using Type = MultTrait_< MT, CompressedVector<T,false> >;
};

template< typename T, typename MT, bool SO, bool DF >
struct MultTrait< CompressedVector<T,true>, UniLowerMatrix<MT,SO,DF> >
{
   using Type = MultTrait_< CompressedVector<T,true>, MT >;
};

template< typename MT, bool SO1, bool DF, typename T, size_t M, size_t N, bool SO2 >
struct MultTrait< UniLowerMatrix<MT,SO1,DF>, StaticMatrix<T,M,N,SO2> >
{
   using Type = MultTrait_< MT, StaticMatrix<T,M,N,SO2> >;
};

template< typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2, bool DF >
struct MultTrait< StaticMatrix<T,M,N,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = MultTrait_< StaticMatrix<T,M,N,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, typename T, size_t M, size_t N, bool SO2 >
struct MultTrait< UniLowerMatrix<MT,SO1,DF>, HybridMatrix<T,M,N,SO2> >
{
   using Type = MultTrait_< MT, HybridMatrix<T,M,N,SO2> >;
};

template< typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2, bool DF >
struct MultTrait< HybridMatrix<T,M,N,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = MultTrait_< HybridMatrix<T,M,N,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, typename T, bool SO2 >
struct MultTrait< UniLowerMatrix<MT,SO1,DF>, DynamicMatrix<T,SO2> >
{
   using Type = MultTrait_< MT, DynamicMatrix<T,SO2> >;
};

template< typename T, bool SO1, typename MT, bool SO2, bool DF >
struct MultTrait< DynamicMatrix<T,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = MultTrait_< DynamicMatrix<T,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, typename T, bool AF, bool PF, bool SO2 >
struct MultTrait< UniLowerMatrix<MT,SO1,DF>, CustomMatrix<T,AF,PF,SO2> >
{
   using Type = MultTrait_< MT, CustomMatrix<T,AF,PF,SO2> >;
};

template< typename T, bool AF, bool PF, bool SO1, typename MT, bool SO2, bool DF >
struct MultTrait< CustomMatrix<T,AF,PF,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = MultTrait_< CustomMatrix<T,AF,PF,SO1>, MT >;
};

template< typename MT, bool SO1, bool DF, typename T, bool SO2 >
struct MultTrait< UniLowerMatrix<MT,SO1,DF>, CompressedMatrix<T,SO2> >
{
   using Type = MultTrait_< MT, CompressedMatrix<T,SO2> >;
};

template< typename T, bool SO1, typename MT, bool SO2, bool DF >
struct MultTrait< CompressedMatrix<T,SO1>, UniLowerMatrix<MT,SO2,DF> >
{
   using Type = MultTrait_< CompressedMatrix<T,SO1>, MT >;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2, bool NF >
struct MultTrait< UniLowerMatrix<MT1,SO1,DF1>, SymmetricMatrix<MT2,SO2,DF2,NF> >
{
   using Type = MultTrait_<MT1,MT2>;
};

template< typename MT1, bool SO1, bool DF1, bool NF, typename MT2, bool SO2, bool DF2 >
struct MultTrait< SymmetricMatrix<MT1,SO1,DF1,NF>, UniLowerMatrix<MT2,SO2,DF2> >
{
   using Type = MultTrait_<MT1,MT2>;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct MultTrait< UniLowerMatrix<MT1,SO1,DF1>, HermitianMatrix<MT2,SO2,DF2> >
{
   using Type = MultTrait_<MT1,MT2>;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct MultTrait< HermitianMatrix<MT1,SO1,DF1>, UniLowerMatrix<MT2,SO2,DF2> >
{
   using Type = MultTrait_<MT1,MT2>;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct MultTrait< UniLowerMatrix<MT1,SO1,DF1>, LowerMatrix<MT2,SO2,DF2> >
{
   using Type = LowerMatrix< MultTrait_<MT1,MT2> >;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct MultTrait< LowerMatrix<MT1,SO1,DF1>, UniLowerMatrix<MT2,SO2,DF2> >
{
   using Type = LowerMatrix< MultTrait_<MT1,MT2> >;
};

template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct MultTrait< UniLowerMatrix<MT1,SO1,DF1>, UniLowerMatrix<MT2,SO2,DF2> >
{
   using Type = UniLowerMatrix< MultTrait_<MT1,MT2> >;
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
template< typename MT, bool SO, bool DF, typename T >
struct DivTrait< UniLowerMatrix<MT,SO,DF>, T, EnableIf_< IsNumeric<T> > >
{
   using Type = LowerMatrix< DivTrait_<MT,T> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  FOREACHTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, bool SO, bool DF, typename ET >
struct ForEachTrait< UniLowerMatrix<MT,SO,DF>, Pow<ET> >
{
   using Type = UniLowerMatrix< ForEachTrait_< MT, Pow<ET> > >;
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
template< typename MT1, bool SO1, bool DF1, typename MT2, bool SO2, bool DF2 >
struct MathTrait< UniLowerMatrix<MT1,SO1,DF1>, UniLowerMatrix<MT2,SO2,DF2> >
{
   using HighType = UniLowerMatrix< typename MathTrait<MT1,MT2>::HighType >;
   using LowType  = UniLowerMatrix< typename MathTrait<MT1,MT2>::LowType  >;
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
template< typename MT, bool SO, bool DF >
struct SubmatrixTrait< UniLowerMatrix<MT,SO,DF> >
{
   using Type = SubmatrixTrait_<MT>;
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
template< typename MT, bool SO, bool DF >
struct RowTrait< UniLowerMatrix<MT,SO,DF> >
{
   using Type = RowTrait_<MT>;
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
template< typename MT, bool SO, bool DF >
struct ColumnTrait< UniLowerMatrix<MT,SO,DF> >
{
   using Type = ColumnTrait_<MT>;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
