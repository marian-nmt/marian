//=================================================================================================
/*!
//  \file blaze/math/dense/HybridMatrix.h
//  \brief Header file for the implementation of a fixed-size matrix
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

#ifndef _BLAZE_MATH_DENSE_HYBRIDMATRIX_H_
#define _BLAZE_MATH_DENSE_HYBRIDMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <utility>
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
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/Conjugate.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/ColumnTrait.h>
#include <blaze/math/traits/CTransExprTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/InvExprTrait.h>
#include <blaze/math/traits/MathTrait.h>
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
#include <blaze/math/typetraits/IsDiagonal.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsColumnMajorMatrix.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/IsSparseMatrix.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/StorageOrder.h>
#include <blaze/util/AlignedArray.h>
#include <blaze/util/AlignmentCheck.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/FloatingPoint.h>
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
/*!\defgroup hybrid_matrix HybridMatrix
// \ingroup dense_matrix
*/
/*!\brief Efficient implementation of a dynamically sized matrix with static memory.
// \ingroup hybrid_matrix
//
// The HybridMatrix class template combines the flexibility of a dynamically sized matrix with
// the efficiency and performance of a fixed-size matrix. It is implemented as a crossing between
// the blaze::StaticMatrix and the blaze::DynamicMatrix class templates: Similar to the static
// matrix it uses static stack memory instead of dynamically allocated memory and similar to the
// dynamic matrix it can be resized (within the extend of the static memory). The type of the
// elements, the maximum number of rows and columns and the storage order of the matrix can be
// specified via the four template parameters:

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
// Depending on the storage order, the matrix elements are either stored in a row-wise fashion
// or in a column-wise fashion. Given the 2x3 matrix

                          \f[\left(\begin{array}{*{3}{c}}
                          1 & 2 & 3 \\
                          4 & 5 & 6 \\
                          \end{array}\right)\f]\n

// in case of row-major order the elements are stored in the order

                          \f[\left(\begin{array}{*{6}{c}}
                          1 & 2 & 3 & 4 & 5 & 6. \\
                          \end{array}\right)\f]

// In case of column-major order the elements are stored in the order

                          \f[\left(\begin{array}{*{6}{c}}
                          1 & 4 & 2 & 5 & 3 & 6. \\
                          \end{array}\right)\f]

// The use of HybridMatrix is very natural and intuitive. All operations (addition, subtraction,
// multiplication, scaling, ...) can be performed on all possible combination of row-major and
// column-major dense and sparse matrices with fitting element types. The following example gives
// an impression of the use of HybridMatrix:

   \code
   using blaze::HybridMatrix;
   using blaze::CompressedMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   HybridMatrix<double,rowMajor> A( 2, 3 );   // Default constructed, non-initialized, row-major 2x3 matrix
   A(0,0) = 1.0; A(0,1) = 2.0; A(0,2) = 3.0;  // Initialization of the first row
   A(1,0) = 4.0; A(1,1) = 5.0; A(1,2) = 6.0;  // Initialization of the second row

   HybridMatrix<float,columnMajor> B( 2, 3 );   // Default constructed column-major single precision 2x3 matrix
   B(0,0) = 1.0; B(0,1) = 3.0; B(0,2) = 5.0;    // Initialization of the first row
   B(1,0) = 2.0; B(1,1) = 4.0; B(1,2) = 6.0;    // Initialization of the second row

   CompressedMatrix<float> C( 2, 3 );        // Empty row-major sparse single precision matrix
   HybridMatrix<float>     D( 3, 2, 4.0F );  // Directly, homogeneously initialized single precision 3x2 matrix

   HybridMatrix<double,rowMajor>    E( A );  // Creation of a new row-major matrix as a copy of A
   HybridMatrix<double,columnMajor> F;       // Creation of a default column-major matrix

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
        , size_t M                         // Number of rows
        , size_t N                         // Number of columns
        , bool SO = defaultStorageOrder >  // Storage order
class HybridMatrix : public DenseMatrix< HybridMatrix<Type,M,N,SO>, SO >
{
 public:
   //**Type definitions****************************************************************************
   typedef HybridMatrix<Type,M,N,SO>   This;           //!< Type of this HybridMatrix instance.
   typedef DenseMatrix<This,SO>        BaseType;       //!< Base type of this HybridMatrix instance.
   typedef This                        ResultType;     //!< Result type for expression template evaluations.
   typedef HybridMatrix<Type,M,N,!SO>  OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef HybridMatrix<Type,N,M,!SO>  TransposeType;  //!< Transpose type for expression template evaluations.
   typedef Type                        ElementType;    //!< Type of the matrix elements.
   typedef SIMDTrait_<ElementType>     SIMDType;       //!< SIMD type of the matrix elements.
   typedef const Type&                 ReturnType;     //!< Return type for expression template evaluations.
   typedef const This&                 CompositeType;  //!< Data type for composite expression templates.

   typedef Type&        Reference;       //!< Reference to a non-constant matrix value.
   typedef const Type&  ConstReference;  //!< Reference to a constant matrix value.
   typedef Type*        Pointer;         //!< Pointer to a non-constant matrix value.
   typedef const Type*  ConstPointer;    //!< Pointer to a constant matrix value.

   typedef DenseIterator<Type,usePadding>        Iterator;       //!< Iterator over non-constant elements.
   typedef DenseIterator<const Type,usePadding>  ConstIterator;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a HybridMatrix with different data/element type.
   */
   template< typename ET >  // Data type of the other matrix
   struct Rebind {
      typedef HybridMatrix<ET,M,N,SO>  Other;  //!< The type of the other HybridMatrix.
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
   enum : bool { smpAssignable = false };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline HybridMatrix();
   explicit inline HybridMatrix( size_t m, size_t n );
   explicit inline HybridMatrix( size_t m, size_t n, const Type& init );
   explicit inline HybridMatrix( initializer_list< initializer_list<Type> > list );

   template< typename Other >
   explicit inline HybridMatrix( size_t m, size_t n, const Other* array );

   template< typename Other, size_t M2, size_t N2 >
   explicit inline HybridMatrix( const Other (&array)[M2][N2] );

                                     inline HybridMatrix( const HybridMatrix& m );
   template< typename MT, bool SO2 > inline HybridMatrix( const Matrix<MT,SO2>& m );
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
   inline HybridMatrix& operator=( const Type& set );
   inline HybridMatrix& operator=( initializer_list< initializer_list<Type> > list );

   template< typename Other, size_t M2, size_t N2 >
   inline HybridMatrix& operator=( const Other (&array)[M2][N2] );

                                     inline HybridMatrix& operator= ( const HybridMatrix& rhs );
   template< typename MT, bool SO2 > inline HybridMatrix& operator= ( const Matrix<MT,SO2>& rhs );
   template< typename MT, bool SO2 > inline HybridMatrix& operator+=( const Matrix<MT,SO2>& rhs );
   template< typename MT, bool SO2 > inline HybridMatrix& operator-=( const Matrix<MT,SO2>& rhs );
   template< typename MT, bool SO2 > inline HybridMatrix& operator*=( const Matrix<MT,SO2>& rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, HybridMatrix >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, HybridMatrix >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t           rows() const noexcept;
                              inline size_t           columns() const noexcept;
                              inline constexpr size_t spacing() const noexcept;
                              inline constexpr size_t capacity() const noexcept;
                              inline           size_t capacity( size_t i ) const noexcept;
                              inline size_t           nonZeros() const;
                              inline size_t           nonZeros( size_t i ) const;
                              inline void             reset();
                              inline void             reset( size_t i );
                              inline void             clear();
                                     void             resize ( size_t m, size_t n, bool preserve=true );
                              inline void             extend ( size_t m, size_t n, bool preserve=true );
                              inline HybridMatrix&    transpose();
                              inline HybridMatrix&    ctranspose();
   template< typename Other > inline HybridMatrix&    scale( const Other& scalar );
                              inline void             swap( HybridMatrix& m ) noexcept;
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
   template< typename MT >
   struct VectorizedAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<MT> >::value &&
                            IsRowMajorMatrix<MT>::value };
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
                            IsRowMajorMatrix<MT>::value &&
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
                            IsRowMajorMatrix<MT>::value &&
                            !IsDiagonal<MT>::value };
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   //! The number of elements packed within a single SIMD element.
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

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t i, size_t j ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t i, size_t j, const SIMDType& value ) noexcept;

   template< typename MT, bool SO2 >
   inline DisableIf_<VectorizedAssign<MT> > assign( const DenseMatrix<MT,SO2>& rhs );

   template< typename MT, bool SO2 >
   inline EnableIf_<VectorizedAssign<MT> > assign( const DenseMatrix<MT,SO2>& rhs );

   template< typename MT > inline void assign( const SparseMatrix<MT,SO>&  rhs );
   template< typename MT > inline void assign( const SparseMatrix<MT,!SO>& rhs );

   template< typename MT, bool SO2 >
   inline DisableIf_<VectorizedAddAssign<MT> > addAssign( const DenseMatrix<MT,SO2>& rhs );

   template< typename MT, bool SO2 >
   inline EnableIf_<VectorizedAddAssign<MT> > addAssign( const DenseMatrix<MT,SO2>& rhs );

   template< typename MT > inline void addAssign( const SparseMatrix<MT,SO>&  rhs );
   template< typename MT > inline void addAssign( const SparseMatrix<MT,!SO>& rhs );

   template< typename MT, bool SO2 >
   inline DisableIf_<VectorizedSubAssign<MT> > subAssign( const DenseMatrix<MT,SO2>& rhs );

   template< typename MT, bool SO2 >
   inline EnableIf_<VectorizedSubAssign<MT> > subAssign( const DenseMatrix<MT,SO2>& rhs );

   template< typename MT > inline void subAssign( const SparseMatrix<MT,SO>&  rhs );
   template< typename MT > inline void subAssign( const SparseMatrix<MT,!SO>& rhs );
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
   AlignedArray<Type,M*NN> v_;  //!< The statically allocated matrix elements.
                                /*!< Access to the matrix elements is gained via the function call
                                     operator. In case of row-major order the memory layout of the
                                     elements is
                                     \f[\left(\begin{array}{*{5}{c}}
                                     0            & 1             & 2             & \cdots & N-1         \\
                                     N            & N+1           & N+2           & \cdots & 2 \cdot N-1 \\
                                     \vdots       & \vdots        & \vdots        & \ddots & \vdots      \\
                                     M \cdot N-N  & M \cdot N-N+1 & M \cdot N-N+2 & \cdots & M \cdot N-1 \\
                                     \end{array}\right)\f]. */
   size_t m_;                   //!< The current number of rows of the matrix.
   size_t n_;                   //!< The current number of columns of the matrix.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE  ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST         ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE      ( Type );
   BLAZE_STATIC_ASSERT( !usePadding || NN % SIMDSIZE == 0UL );
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
/*!\brief The default constructor for HybridMatrix.
//
// The size of a default constructed HybridMatrix is initially set to 0.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline HybridMatrix<Type,M,N,SO>::HybridMatrix()
   : v_()       // The statically allocated matrix elements
   , m_( 0UL )  // The current number of rows of the matrix
   , n_( 0UL )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   if( IsNumeric<Type>::value ) {
      for( size_t i=0UL; i<M*NN; ++i )
         v_[i] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a matrix of size \f$ m \times n \f$. No element initialization is performed!
//
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \exception std::invalid_argument Invalid number of rows for hybrid matrix.
// \exception std::invalid_argument Invalid number of columns for hybrid matrix.
//
// This constructor creates a hybrid matrix of size \f$ m \times n \f$, but leaves the elements
// uninitialized. In case \a m is larger than the maximum allowed number of rows (i.e. \a m > M)
// or \a n is larger than the maximum allowed number of columns a \a std::invalid_argument
// exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline HybridMatrix<Type,M,N,SO>::HybridMatrix( size_t m, size_t n )
   : v_()     // The statically allocated matrix elements
   , m_( m )  // The current number of rows of the matrix
   , n_( n )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   if( m > M ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of rows for hybrid matrix" );
   }

   if( n > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of columns for hybrid matrix" );
   }

   if( IsNumeric<Type>::value ) {
      for( size_t i=0UL; i<M*NN; ++i )
         v_[i] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a homogenous initialization of all \f$ m \times n \f$ matrix elements.
//
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \param init The initial value of the matrix elements.
// \exception std::invalid_argument Invalid number of rows for hybrid matrix.
// \exception std::invalid_argument Invalid number of columns for hybrid matrix.
//
// This constructor creates a hybrid matrix of size \f$ m \times n \f$ and initializes all
// matrix elements with the specified value. In case \a m is larger than the maximum allowed
// number of rows (i.e. \a m > M) or \a n is larger than the maximum allowed number of columns
// a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline HybridMatrix<Type,M,N,SO>::HybridMatrix( size_t m, size_t n, const Type& init )
   : v_()     // The statically allocated matrix elements
   , m_( m )  // The current number of rows of the matrix
   , n_( n )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   if( m > M ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of rows for hybrid matrix" );
   }

   if( n > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of columns for hybrid matrix" );
   }

   for( size_t i=0UL; i<m; ++i ) {
      for( size_t j=0UL; j<n; ++j )
         v_[i*NN+j] = init;

      if( IsNumeric<Type>::value ) {
         for( size_t j=n; j<NN; ++j )
            v_[i*NN+j] = Type();
      }
   }

   if( IsNumeric<Type>::value ) {
      for( size_t i=m; i<M; ++i )
         for( size_t j=0UL; j<NN; ++j )
            v_[i*NN+j] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List initialization of all matrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid number of rows for hybrid matrix.
// \exception std::invalid_argument Invalid number of columns for hybrid matrix.
//
// This constructor provides the option to explicitly initialize the elements of the matrix by
// means of an initializer list:

   \code
   using blaze::rowMajor;

   blaze::HybridMatrix<int,3,3,rowMajor> A{ { 1, 2, 3 },
                                            { 4, 5 },
                                            { 7, 8, 9 } };
   \endcode

// The matrix is sized according to the size of the initializer list and all its elements are
// initialized by the values of the given initializer list. Missing values are initialized as
// default (as e.g. the value 6 in the example). Note that in case the size of the top-level
// initializer list exceeds the number of rows or the size of any nested list exceeds the
// number of columns, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline HybridMatrix<Type,M,N,SO>::HybridMatrix( initializer_list< initializer_list<Type> > list )
   : v_()                            // The statically allocated matrix elements
   , m_( list.size() )               // The current number of rows of the matrix
   , n_( determineColumns( list ) )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   if( m_ > M ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of rows for hybrid matrix" );
   }

   if( n_ > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of columns for hybrid matrix" );
   }

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      std::fill( std::copy( rowList.begin(), rowList.end(), v_+i*NN ), v_+(i+1UL)*NN, Type() );
      ++i;
   }

   if( IsNumeric<Type>::value ) {
      for( ; i<M; ++i )
         for( size_t j=0UL; j<NN; ++j )
            v_[i*NN+j] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array initialization of all matrix elements.
//
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \param array Dynamic array for the initialization.
// \exception std::invalid_argument Invalid number of rows for hybrid matrix.
// \exception std::invalid_argument Invalid number of columns for hybrid matrix.
//
// This constructor offers the option to directly initialize the elements of the matrix with
// a dynamic array:

   \code
   using blaze::rowMajor;

   int* array = new int[20];
   // ... Initialization of the dynamic array
   blaze::HybridMatrix<int,4,5,rowMajor> v( 4UL, 5UL, array );
   delete[] array;
   \endcode

// The matrix is sized according to the given size of the array and initialized with the values
// from the given array. In case \a m is larger than the maximum allowed number of rows (i.e.
// \a m > M) or \a n is larger than the maximum allowed number of columns a \a std::invalid_argument
// exception is thrown. Note that it is expected that the given \a array has at least \a m by
// \a n elements. Providing an array with less elements results in undefined behavior!
*/
template< typename Type     // Data type of the matrix
        , size_t M          // Number of rows
        , size_t N          // Number of columns
        , bool SO >         // Storage order
template< typename Other >  // Data type of the initialization array
inline HybridMatrix<Type,M,N,SO>::HybridMatrix( size_t m, size_t n, const Other* array )
   : v_()     // The statically allocated matrix elements
   , m_( m )  // The current number of rows of the matrix
   , n_( n )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   if( m > M ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of rows for hybrid matrix" );
   }

   if( n > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of columns for hybrid matrix" );
   }

   for( size_t i=0UL; i<m; ++i ) {
      for( size_t j=0UL; j<n; ++j )
         v_[i*NN+j] = array[i*n+j];

      if( IsNumeric<Type>::value ) {
         for( size_t j=n; j<NN; ++j )
            v_[i*NN+j] = Type();
      }
   }

   if( IsNumeric<Type>::value ) {
      for( size_t i=m; i<M; ++i )
         for( size_t j=0UL; j<NN; ++j )
            v_[i*NN+j] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array initialization of all matrix elements.
//
// \param array \f$ M \times N \f$ dimensional array for the initialization.
//
// This constructor offers the option to directly initialize the elements of the matrix with
// a static array:

   \code
   using blaze::rowMajor;

   const int init[3][3] = { { 1, 2, 3 },
                            { 4, 5 },
                            { 7, 8, 9 } };
   blaze::HybridMatrix<int,3,3,rowMajor> A( init );
   \endcode

// The matrix is sized according to the size of the array and initialized with the values from
// the given array. Missing values are initialized with default values (as e.g. the value 6 in
// the example).
*/
template< typename Type   // Data type of the matrix
        , size_t M        // Number of rows
        , size_t N        // Number of columns
        , bool SO >       // Storage order
template< typename Other  // Data type of the initialization array
        , size_t M2       // Number of rows of the initialization array
        , size_t N2 >     // Number of columns of the initialization array
inline HybridMatrix<Type,M,N,SO>::HybridMatrix( const Other (&array)[M2][N2] )
   : v_()      // The statically allocated matrix elements
   , m_( M2 )  // The current number of rows of the matrix
   , n_( N2 )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( M2 <= M );
   BLAZE_STATIC_ASSERT( N2 <= N );
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   for( size_t i=0UL; i<M2; ++i ) {
      for( size_t j=0UL; j<N2; ++j )
         v_[i*NN+j] = array[i][j];

      if( IsNumeric<Type>::value ) {
         for( size_t j=N2; j<NN; ++j )
            v_[i*NN+j] = Type();
      }
   }

   if( IsNumeric<Type>::value ) {
      for( size_t i=M2; i<M; ++i )
         for( size_t j=0UL; j<NN; ++j )
            v_[i*NN+j] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for HybridMatrix.
//
// \param m Matrix to be copied.
//
// The copy constructor is explicitly defined due to the required dynamic memory management
// and in order to enable/facilitate NRV optimization.
*/
template< typename Type     // Data type of the matrix
        , size_t M          // Number of rows
        , size_t N          // Number of columns
        , bool SO >         // Storage order
inline HybridMatrix<Type,M,N,SO>::HybridMatrix( const HybridMatrix& m )
   : v_()        // The statically allocated matrix elements
   , m_( m.m_ )  // The current number of rows of the matrix
   , n_( m.n_ )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   for( size_t i=0UL; i<m_; ++i ) {
      for( size_t j=0UL; j<n_; ++j )
         v_[i*NN+j] = m.v_[i*NN+j];

      if( IsNumeric<Type>::value ) {
         for( size_t j=n_; j<NN; ++j )
            v_[i*NN+j] = Type();
      }
   }

   if( IsNumeric<Type>::value ) {
      for( size_t i=m_; i<M; ++i )
         for( size_t j=0UL; j<NN; ++j )
            v_[i*NN+j] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different matrices.
//
// \param m Matrix to be copied.
// \exception std::invalid_argument Invalid setup of hybrid matrix.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT    // Type of the foreign matrix
        , bool SO2 >     // Storage order of the foreign matrix
inline HybridMatrix<Type,M,N,SO>::HybridMatrix( const Matrix<MT,SO2>& m )
   : v_()                  // The statically allocated matrix elements
   , m_( (~m).rows() )     // The current number of rows of the matrix
   , n_( (~m).columns() )  // The current number of columns of the matrix
{
   using blaze::assign;

   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || NN == N );

   if( (~m).rows() > M || (~m).columns() > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of hybrid matrix" );
   }

   for( size_t i=0UL; i<m_; ++i ) {
      for( size_t j=( IsSparseMatrix<MT>::value ? 0UL : n_ );
                  j<( IsNumeric<Type>::value    ? NN  : n_ );
                  ++j ) {
         v_[i*NN+j] = Type();
      }
   }

   if( IsNumeric<Type>::value ) {
      for( size_t i=m_; i<M; ++i )
         for( size_t j=0UL; j<NN; ++j )
            v_[i*NN+j] = Type();
   }

   assign( *this, ~m );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
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
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::Reference
   HybridMatrix<Type,M,N,SO>::operator()( size_t i, size_t j ) noexcept
{
   BLAZE_USER_ASSERT( i<M, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<N, "Invalid column access index" );
   return v_[i*NN+j];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief 2D-access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return Reference-to-const to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::ConstReference
   HybridMatrix<Type,M,N,SO>::operator()( size_t i, size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( i<M, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<N, "Invalid column access index" );
   return v_[i*NN+j];
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::Reference
   HybridMatrix<Type,M,N,SO>::at( size_t i, size_t j )
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::ConstReference
   HybridMatrix<Type,M,N,SO>::at( size_t i, size_t j ) const
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
// This function returns a pointer to the internal storage of the hybrid matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The hybrid matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::Pointer
   HybridMatrix<Type,M,N,SO>::data() noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the matrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the hybrid matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The hybrid matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::ConstPointer
   HybridMatrix<Type,M,N,SO>::data() const noexcept
{
   return v_;
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::Pointer
   HybridMatrix<Type,M,N,SO>::data( size_t i ) noexcept
{
   BLAZE_USER_ASSERT( i < M, "Invalid dense matrix row access index" );
   return v_ + i*NN;
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::ConstPointer
   HybridMatrix<Type,M,N,SO>::data( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < M, "Invalid dense matrix row access index" );
   return v_ + i*NN;
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::Iterator
   HybridMatrix<Type,M,N,SO>::begin( size_t i ) noexcept
{
   BLAZE_USER_ASSERT( i < M, "Invalid dense matrix row access index" );
   return Iterator( v_ + i*NN );
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::ConstIterator
   HybridMatrix<Type,M,N,SO>::begin( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < M, "Invalid dense matrix row access index" );
   return ConstIterator( v_ + i*NN );
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::ConstIterator
   HybridMatrix<Type,M,N,SO>::cbegin( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < M, "Invalid dense matrix row access index" );
   return ConstIterator( v_ + i*NN );
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::Iterator
   HybridMatrix<Type,M,N,SO>::end( size_t i ) noexcept
{
   BLAZE_USER_ASSERT( i < M, "Invalid dense matrix row access index" );
   return Iterator( v_ + i*NN + N );
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::ConstIterator
   HybridMatrix<Type,M,N,SO>::end( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < M, "Invalid dense matrix row access index" );
   return ConstIterator( v_ + i*NN + N );
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline typename HybridMatrix<Type,M,N,SO>::ConstIterator
   HybridMatrix<Type,M,N,SO>::cend( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < M, "Invalid dense matrix row access index" );
   return ConstIterator( v_ + i*NN + N );
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
// \param set Scalar value to be assigned to all matrix elements.
// \return Reference to the assigned matrix.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline HybridMatrix<Type,M,N,SO>& HybridMatrix<Type,M,N,SO>::operator=( const Type& set )
{
   BLAZE_INTERNAL_ASSERT( m_ <= M, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( n_ <= N, "Invalid number of columns detected" );

   for( size_t i=0UL; i<m_; ++i )
      for( size_t j=0UL; j<n_; ++j )
         v_[i*NN+j] = set;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List assignment to all matrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid number of rows for hybrid matrix.
// \exception std::invalid_argument Invalid number of columns for hybrid matrix.
//
// This assignment operator offers the option to directly assign to all elements of the matrix
// by means of an initializer list:

   \code
   using blaze::rowMajor;

   blaze::DynamicMatrix<int,rowMajor> A;
   A = { { 1, 2, 3 },
         { 4, 5 },
         { 7, 8, 9 } };
   \endcode

// The matrix is resized according to the given initializer list and all its elements are
// assigned the values from the given initializer list. Missing values are initialized as
// default (as e.g. the value 6 in the example). Note that in case the size of the top-level
// initializer list exceeds the number of rows or the size of any nested list exceeds the
// number of columns, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline HybridMatrix<Type,M,N,SO>&
   HybridMatrix<Type,M,N,SO>::operator=( initializer_list< initializer_list<Type> > list )
{
   const size_t m( list.size() );
   const size_t n( determineColumns( list ) );

   if( m > M ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of rows for hybrid matrix" );
   }

   if( n > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of columns for hybrid matrix" );
   }

   resize( m, n, false );

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      std::fill( std::copy( rowList.begin(), rowList.end(), v_+i*NN ), v_+(i+1UL)*NN, Type() );
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
//
// This assignment operator offers the option to directly set all elements of the matrix:

   \code
   using blaze::rowMajor;

   const int init[3][3] = { { 1, 2, 3 },
                            { 4, 5 },
                            { 7, 8, 9 } };
   blaze::HybridMatrix<int,3UL,3UL,rowMajor> A;
   A = init;
   \endcode

// The matrix is assigned the values from the given array. Missing values are initialized with
// default values (as e.g. the value 6 in the example).
*/
template< typename Type   // Data type of the matrix
        , size_t M        // Number of rows
        , size_t N        // Number of columns
        , bool SO >       // Storage order
template< typename Other  // Data type of the initialization array
        , size_t M2       // Number of rows of the initialization array
        , size_t N2 >     // Number of columns of the initialization array
inline HybridMatrix<Type,M,N,SO>& HybridMatrix<Type,M,N,SO>::operator=( const Other (&array)[M2][N2] )
{
   BLAZE_STATIC_ASSERT( M2 <= M );
   BLAZE_STATIC_ASSERT( N2 <= N );

   resize( M2, N2 );

   for( size_t i=0UL; i<M2; ++i )
      for( size_t j=0UL; j<N2; ++j )
         v_[i*NN+j] = array[i][j];

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Copy assignment operator for HybridMatrix.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
//
// Explicit definition of a copy assignment operator for performance reasons.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline HybridMatrix<Type,M,N,SO>& HybridMatrix<Type,M,N,SO>::operator=( const HybridMatrix& rhs )
{
   using blaze::assign;

   BLAZE_INTERNAL_ASSERT( m_ <= M, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( n_ <= N, "Invalid number of columns detected" );

   resize( rhs.rows(), rhs.columns() );
   assign( *this, ~rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different matrices.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
// \exception std::invalid_argument Invalid assignment to hybrid matrix.
//
// This constructor initializes the matrix as a copy of the given matrix. In case the
// number of rows of the given matrix is not M or the number of columns is not N, a
// \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline HybridMatrix<Type,M,N,SO>& HybridMatrix<Type,M,N,SO>::operator=( const Matrix<MT,SO2>& rhs )
{
   using blaze::assign;

   typedef TransExprTrait_<This>   TT;
   typedef CTransExprTrait_<This>  CT;
   typedef InvExprTrait_<This>     IT;

   if( (~rhs).rows() > M || (~rhs).columns() > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to hybrid matrix" );
   }

   if( IsSame<MT,TT>::value && (~rhs).isAliased( this ) ) {
      transpose();
   }
   else if( IsSame<MT,CT>::value && (~rhs).isAliased( this ) ) {
      ctranspose();
   }
   else if( !IsSame<MT,IT>::value && (~rhs).canAlias( this ) ) {
      HybridMatrix tmp( ~rhs );
      resize( tmp.rows(), tmp.columns() );
      assign( *this, tmp );
   }
   else {
      resize( (~rhs).rows(), (~rhs).columns() );
      if( IsSparseMatrix<MT>::value )
         reset();
      assign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline HybridMatrix<Type,M,N,SO>& HybridMatrix<Type,M,N,SO>::operator+=( const Matrix<MT,SO2>& rhs )
{
   using blaze::addAssign;

   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_<MT> tmp( ~rhs );
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline HybridMatrix<Type,M,N,SO>& HybridMatrix<Type,M,N,SO>::operator-=( const Matrix<MT,SO2>& rhs )
{
   using blaze::subAssign;

   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_<MT> tmp( ~rhs );
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline HybridMatrix<Type,M,N,SO>& HybridMatrix<Type,M,N,SO>::operator*=( const Matrix<MT,SO2>& rhs )
{
   if( n_ != (~rhs).rows() || (~rhs).columns() > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const HybridMatrix tmp( *this * (~rhs) );
   this->operator=( tmp );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

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
        , size_t M          // Number of rows
        , size_t N          // Number of columns
        , bool SO >         // Storage order
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, HybridMatrix<Type,M,N,SO> >&
   HybridMatrix<Type,M,N,SO>::operator*=( Other rhs )
{
   using blaze::assign;

   assign( *this, (*this) * rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment operator for the division of a matrix by a scalar value
//        (\f$ A/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the matrix.
//
// \note A division by zero is only checked by an user assert.
*/
template< typename Type     // Data type of the matrix
        , size_t M          // Number of rows
        , size_t N          // Number of columns
        , bool SO >         // Storage order
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, HybridMatrix<Type,M,N,SO> >&
   HybridMatrix<Type,M,N,SO>::operator/=( Other rhs )
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
/*!\brief Returns the current number of rows of the matrix.
//
// \return The number of rows of the matrix.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline size_t HybridMatrix<Type,M,N,SO>::rows() const noexcept
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline size_t HybridMatrix<Type,M,N,SO>::columns() const noexcept
{
   return n_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the spacing between the beginning of two rows.
//
// \return The spacing between the beginning of two rows.
//
// This function returns the spacing between the beginning of two rows, i.e. the total number
// of elements of a row.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline constexpr size_t HybridMatrix<Type,M,N,SO>::spacing() const noexcept
{
   return NN;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the matrix.
//
// \return The capacity of the matrix.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline constexpr size_t HybridMatrix<Type,M,N,SO>::capacity() const noexcept
{
   return M*NN;
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline size_t HybridMatrix<Type,M,N,SO>::capacity( size_t i ) const noexcept
{
   UNUSED_PARAMETER( i );

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   return NN;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of non-zero elements in the matrix
//
// \return The number of non-zero elements in the matrix.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline size_t HybridMatrix<Type,M,N,SO>::nonZeros() const
{
   size_t nonzeros( 0UL );

   for( size_t i=0UL; i<m_; ++i )
      for( size_t j=0UL; j<n_; ++j )
         if( !isDefault( v_[i*NN+j] ) )
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline size_t HybridMatrix<Type,M,N,SO>::nonZeros( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   const size_t jend( i*NN + n_ );
   size_t nonzeros( 0UL );

   for( size_t j=i*NN; j<jend; ++j )
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void HybridMatrix<Type,M,N,SO>::reset()
{
   using blaze::clear;

   for( size_t i=0UL; i<m_; ++i )
      for( size_t j=0UL; j<n_; ++j )
         clear( v_[i*NN+j] );
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void HybridMatrix<Type,M,N,SO>::reset( size_t i )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   for( size_t j=0UL; j<n_; ++j )
      clear( v_[i*NN+j] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the hybrid matrix.
//
// \return void
//
// After the clear() function, the size of the matrix is 0.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void HybridMatrix<Type,M,N,SO>::clear()
{
   resize( 0UL, 0UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the matrix.
//
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function resizes the matrix using the given size to \f$ m \times n \f$. In case the given
// number of rows \a m is larger than the maximum number of rows (i.e. if m > M) or in case the
// given number of columns \a n is larger than the maximum number of column (i.e. if n > N) a
// \a std::invalid_argument exception is thrown. Note that this function may invalidate all
// existing views (submatrices, rows, columns, ...) on the matrix if it is used to shrink the
// matrix. Additionally, during this operation all matrix elements are potentially changed. In
// order to preserve the old matrix values, the \a preserve flag can be set to \a true.
//
// Note that in case the number of rows or columns is increased new matrix elements are not
// initialized! The following example illustrates the resize operation of a \f$ 2 \times 4 \f$
// matrix to a \f$ 4 \times 2 \f$ matrix. The new, uninitialized elements are marked with \a x:

                              \f[
                              \left(\begin{array}{*{4}{c}}
                              1 & 2 & 3 & 4 \\
                              5 & 6 & 7 & 8 \\
                              \end{array}\right)

                              \Longrightarrow

                              \left(\begin{array}{*{2}{c}}
                              1 & 2 \\
                              5 & 6 \\
                              x & x \\
                              x & x \\
                              \end{array}\right)
                              \f]
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
void HybridMatrix<Type,M,N,SO>::resize( size_t m, size_t n, bool preserve )
{
   UNUSED_PARAMETER( preserve );

   if( m > M ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of rows for hybrid matrix" );
   }

   if( n > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of columns for hybrid matrix" );
   }

   if( IsVectorizable<Type>::value && n < n_ ) {
      for( size_t i=0UL; i<m; ++i )
         for( size_t j=n; j<n_; ++j )
            v_[i*NN+j] = Type();
   }

   if( IsVectorizable<Type>::value && m < m_ ) {
      for( size_t i=m; i<m_; ++i )
         for( size_t j=0UL; j<n_; ++j )
            v_[i*NN+j] = Type();
   }

   m_ = m;
   n_ = n;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Extending the size of the matrix.
//
// \param m Number of additional rows.
// \param n Number of additional columns.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function increases the matrix size by \a m rows and \a n columns. In case the resulting
// number of rows or columns is larger than the maximum number of rows or columns (i.e. if m > M
// or n > N) a \a std::invalid_argument exception is thrown. During this operation, all matrix
// elements are potentially changed. In order to preserve the old matrix values, the \a preserve
// flag can be set to \a true.\n
// Note that new matrix elements are not initialized!
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void HybridMatrix<Type,M,N,SO>::extend( size_t m, size_t n, bool preserve )
{
   UNUSED_PARAMETER( preserve );
   resize( m_+m, n_+n );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the matrix.
//
// \return Reference to the transposed matrix.
// \exception std::logic_error Impossible transpose operation.
//
// This function transposes the hybrid matrix in-place. Note that this function can only be used
// on hybrid matrices whose current dimensions allow an in-place transpose operation. In case the
// current number of rows is larger than the maximum number of columns or if the current number
// of columns is larger than the maximum number of rows, an \a std::logic_error is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline HybridMatrix<Type,M,N,SO>& HybridMatrix<Type,M,N,SO>::transpose()
{
   using std::swap;

   if( m_ > N || n_ > M ) {
      BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
   }

   const size_t maxsize( max( m_, n_ ) );
   for( size_t i=1UL; i<maxsize; ++i ) {
      for( size_t j=0UL; j<i; ++j ) {
         swap( v_[i*NN+j], v_[j*NN+i] );
      }
   }

   if( IsVectorizable<Type>::value && m_ < n_ ) {
      for( size_t i=0UL; i<m_; ++i ) {
         for( size_t j=m_; j<n_; ++j ) {
            v_[i*NN+j] = Type();
         }
      }
   }

   if( IsVectorizable<Type>::value && m_ > n_ ) {
      for( size_t i=n_; i<m_; ++i ) {
         for( size_t j=0UL; j<n_; ++j ) {
            v_[i*NN+j] = Type();
         }
      }
   }

   swap( m_, n_ );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the matrix.
//
// \return Reference to the transposed matrix.
// \exception std::logic_error Impossible transpose operation.
//
// This function transposes the hybrid matrix in-place. Note that this function can only be used
// on hybrid matrices whose current dimensions allow an in-place transpose operation. In case the
// current number of rows is larger than the maximum number of columns or if the current number
// of columns is larger than the maximum number of rows, an \a std::logic_error is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline HybridMatrix<Type,M,N,SO>& HybridMatrix<Type,M,N,SO>::ctranspose()
{
   using std::swap;

   if( m_ > N || n_ > M ) {
      BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
   }

   const size_t maxsize( max( m_, n_ ) );
   for( size_t i=0UL; i<maxsize; ++i ) {
      for( size_t j=0UL; j<i; ++j ) {
         cswap( v_[i*NN+j], v_[j*NN+i] );
      }
      conjugate( v_[i*NN+i] );
   }

   if( IsVectorizable<Type>::value && m_ < n_ ) {
      for( size_t i=0UL; i<m_; ++i ) {
         for( size_t j=m_; j<n_; ++j ) {
            v_[i*NN+j] = Type();
         }
      }
   }

   if( IsVectorizable<Type>::value && m_ > n_ ) {
      for( size_t i=n_; i<m_; ++i ) {
         for( size_t j=0UL; j<n_; ++j ) {
            v_[i*NN+j] = Type();
         }
      }
   }

   swap( m_, n_ );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the matrix by the scalar value \a scalar (\f$ A*=s \f$).
//
// \param scalar The scalar value for the matrix scaling.
// \return Reference to the matrix.
*/
template< typename Type     // Data type of the matrix
        , size_t M          // Number of rows
        , size_t N          // Number of columns
        , bool SO >         // Storage order
template< typename Other >  // Data type of the scalar value
inline HybridMatrix<Type,M,N,SO>& HybridMatrix<Type,M,N,SO>::scale( const Other& scalar )
{
   for( size_t i=0UL; i<m_; ++i )
      for( size_t j=0UL; j<n_; ++j )
         v_[i*NN+j] *= scalar;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two hybrid matrices.
//
// \param m The matrix to be swapped.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void HybridMatrix<Type,M,N,SO>::swap( HybridMatrix& m ) noexcept
{
   using std::swap;

   const size_t maxrows( max( m_, m.m_ ) );
   const size_t maxcols( max( n_, m.n_ ) );

   for( size_t i=0UL; i<maxrows; ++i ) {
      for( size_t j=0UL; j<maxcols; ++j ) {
         swap( v_[i*NN+j], m(i,j) );
      }
   }

   swap( m_, m.m_ );
   swap( n_, m.n_ );
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
// dynamic memory based on the alignment restrictions of the HybridMatrix class template.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void* HybridMatrix<Type,M,N,SO>::operator new( std::size_t size )
{
   UNUSED_PARAMETER( size );

   BLAZE_INTERNAL_ASSERT( size == sizeof( HybridMatrix ), "Invalid number of bytes detected" );

   return allocate<HybridMatrix>( 1UL );
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
// dynamic memory based on the alignment restrictions of the HybridMatrix class template.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void* HybridMatrix<Type,M,N,SO>::operator new[]( std::size_t size )
{
   BLAZE_INTERNAL_ASSERT( size >= sizeof( HybridMatrix )       , "Invalid number of bytes detected" );
   BLAZE_INTERNAL_ASSERT( size %  sizeof( HybridMatrix ) == 0UL, "Invalid number of bytes detected" );

   return allocate<HybridMatrix>( size/sizeof(HybridMatrix) );
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
// dynamic memory based on the alignment restrictions of the HybridMatrix class template.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void* HybridMatrix<Type,M,N,SO>::operator new( std::size_t size, const std::nothrow_t& )
{
   UNUSED_PARAMETER( size );

   BLAZE_INTERNAL_ASSERT( size == sizeof( HybridMatrix ), "Invalid number of bytes detected" );

   return allocate<HybridMatrix>( 1UL );
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
// dynamic memory based on the alignment restrictions of the HybridMatrix class template.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void* HybridMatrix<Type,M,N,SO>::operator new[]( std::size_t size, const std::nothrow_t& )
{
   BLAZE_INTERNAL_ASSERT( size >= sizeof( HybridMatrix )       , "Invalid number of bytes detected" );
   BLAZE_INTERNAL_ASSERT( size %  sizeof( HybridMatrix ) == 0UL, "Invalid number of bytes detected" );

   return allocate<HybridMatrix>( size/sizeof(HybridMatrix) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of operator delete.
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void HybridMatrix<Type,M,N,SO>::operator delete( void* ptr )
{
   deallocate( static_cast<HybridMatrix*>( ptr ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of operator delete[].
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void HybridMatrix<Type,M,N,SO>::operator delete[]( void* ptr )
{
   deallocate( static_cast<HybridMatrix*>( ptr ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of no-throw operator delete.
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void HybridMatrix<Type,M,N,SO>::operator delete( void* ptr, const std::nothrow_t& )
{
   deallocate( static_cast<HybridMatrix*>( ptr ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of no-throw operator delete[].
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void HybridMatrix<Type,M,N,SO>::operator delete[]( void* ptr, const std::nothrow_t& )
{
   deallocate( static_cast<HybridMatrix*>( ptr ) );
}
//*************************************************************************************************




//=================================================================================================
//
//  DEBUGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the invariants of the hybrid matrix are intact.
//
// \return \a true in case the hybrid matrix's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the hybrid matrix are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline bool HybridMatrix<Type,M,N,SO>::isIntact() const noexcept
{
   if( m_ > M || n_ > N )
      return false;

   if( IsNumeric<Type>::value )
   {
      for( size_t i=0UL; i<m_; ++i ) {
         for( size_t j=n_; j<NN; ++j ) {
            if( v_[i*NN+j] != Type() )
               return false;
         }
      }

      for( size_t i=m_; i<M; ++i ) {
         for( size_t j=0UL; j<NN; ++j ) {
            if( v_[i*NN+j] != Type() )
               return false;
         }
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
        , size_t M          // Number of rows
        , size_t N          // Number of columns
        , bool SO >         // Storage order
template< typename Other >  // Data type of the foreign expression
inline bool HybridMatrix<Type,M,N,SO>::canAlias( const Other* alias ) const noexcept
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
        , size_t M          // Number of rows
        , size_t N          // Number of columns
        , bool SO >         // Storage order
template< typename Other >  // Data type of the foreign expression
inline bool HybridMatrix<Type,M,N,SO>::isAliased( const Other* alias ) const noexcept
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline bool HybridMatrix<Type,M,N,SO>::isAligned() const noexcept
{
   return ( usePadding || columns() % SIMDSIZE == 0UL );
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE typename HybridMatrix<Type,M,N,SO>::SIMDType
   HybridMatrix<Type,M,N,SO>::load( size_t i, size_t j ) const noexcept
{
   if( usePadding )
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE typename HybridMatrix<Type,M,N,SO>::SIMDType
   HybridMatrix<Type,M,N,SO>::loada( size_t i, size_t j ) const noexcept
{
   using blaze::loada;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= NN, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( &v_[i*NN+j] ), "Invalid alignment detected" );

   return loada( &v_[i*NN+j] );
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE typename HybridMatrix<Type,M,N,SO>::SIMDType
   HybridMatrix<Type,M,N,SO>::loadu( size_t i, size_t j ) const noexcept
{
   using blaze::loadu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= NN, "Invalid column access index" );

   return loadu( &v_[i*NN+j] );
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE void
   HybridMatrix<Type,M,N,SO>::store( size_t i, size_t j, const SIMDType& value ) noexcept
{
   if( usePadding )
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE void
   HybridMatrix<Type,M,N,SO>::storea( size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::storea;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= NN, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( &v_[i*NN+j] ), "Invalid alignment detected" );

   storea( &v_[i*NN+j], value );
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE void
   HybridMatrix<Type,M,N,SO>::storeu( size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::storeu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= NN, "Invalid column access index" );

   storeu( &v_[i*NN+j], value );
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
BLAZE_ALWAYS_INLINE void
   HybridMatrix<Type,M,N,SO>::stream( size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::stream;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= NN, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( &v_[i*NN+j] ), "Invalid alignment detected" );

   stream( &v_[i*NN+j], value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a dense matrix.
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side dense matrix
        , bool SO2 >     // Storage order of the right-hand side dense matrix
inline DisableIf_<typename HybridMatrix<Type,M,N,SO>::BLAZE_TEMPLATE VectorizedAssign<MT> >
   HybridMatrix<Type,M,N,SO>::assign( const DenseMatrix<MT,SO2>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t i=0UL; i<m_; ++i ) {
      for( size_t j=0UL; j<n_; ++j ) {
         v_[i*NN+j] = (~rhs)(i,j);
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SIMD optimized implementation of the assignment of a dense matrix.
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side dense matrix
        , bool SO2 >     // Storage order of the right-hand side dense matrix
inline EnableIf_<typename HybridMatrix<Type,M,N,SO>::BLAZE_TEMPLATE VectorizedAssign<MT> >
   HybridMatrix<Type,M,N,SO>::assign( const DenseMatrix<MT,SO2>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   const bool remainder( !usePadding || !IsPadded<MT>::value );

   const size_t jpos( ( remainder )?( n_ & size_t(-SIMDSIZE) ):( n_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( n_ - ( n_ % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<m_; ++i )
   {
      size_t j( 0UL );

      for( ; j<jpos; j+=SIMDSIZE ) {
         store( i, j, (~rhs).load(i,j) );
      }
      for( ; remainder && j<n_; ++j ) {
         v_[i*NN+j] = (~rhs)(i,j);
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline void HybridMatrix<Type,M,N,SO>::assign( const SparseMatrix<MT,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i*NN+element->index()] = element->value();
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline void HybridMatrix<Type,M,N,SO>::assign( const SparseMatrix<MT,!SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()*NN+j] = element->value();
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side dense matrix
        , bool SO2 >     // Storage order of the right-hand side dense matrix
inline DisableIf_<typename HybridMatrix<Type,M,N,SO>::BLAZE_TEMPLATE VectorizedAddAssign<MT> >
   HybridMatrix<Type,M,N,SO>::addAssign( const DenseMatrix<MT,SO2>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t i=0UL; i<m_; ++i )
   {
      if( IsDiagonal<MT>::value )
      {
         v_[i*NN+i] += (~rhs)(i,i);
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

         for( size_t j=jbegin; j<jend; ++j ) {
            v_[i*NN+j] += (~rhs)(i,j);
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SIMD optimized implementation of the addition assignment of a dense matrix.
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side dense matrix
        , bool SO2 >     // Storage order of the right-hand side dense matrix
inline EnableIf_<typename HybridMatrix<Type,M,N,SO>::BLAZE_TEMPLATE VectorizedAddAssign<MT> >
   HybridMatrix<Type,M,N,SO>::addAssign( const DenseMatrix<MT,SO2>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_DIAGONAL_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   const bool remainder( !usePadding || !IsPadded<MT>::value );

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

      for( ; j<jpos; j+=SIMDSIZE ) {
         store( i, j, load(i,j) + (~rhs).load(i,j) );
      }
      for( ; remainder && j<jend; ++j ) {
         v_[i*NN+j] += (~rhs)(i,j);
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline void HybridMatrix<Type,M,N,SO>::addAssign( const SparseMatrix<MT,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i*NN+element->index()] += element->value();
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline void HybridMatrix<Type,M,N,SO>::addAssign( const SparseMatrix<MT,!SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()*NN+j] += element->value();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a dense matrix.
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side dense matrix
        , bool SO2 >     // Storage order of the right-hand side dense matrix
inline DisableIf_<typename HybridMatrix<Type,M,N,SO>::BLAZE_TEMPLATE VectorizedSubAssign<MT> >
   HybridMatrix<Type,M,N,SO>::subAssign( const DenseMatrix<MT,SO2>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t i=0UL; i<m_; ++i )
   {
      if( IsDiagonal<MT>::value )
      {
         v_[i*NN+i] -= (~rhs)(i,i);
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

         for( size_t j=jbegin; j<jend; ++j ) {
            v_[i*NN+j] -= (~rhs)(i,j);
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SIMD optimized implementation of the subtraction assignment of a dense matrix.
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side dense matrix
        , bool SO2 >     // Storage order of the right-hand side dense matrix
inline EnableIf_<typename HybridMatrix<Type,M,N,SO>::BLAZE_TEMPLATE VectorizedSubAssign<MT> >
   HybridMatrix<Type,M,N,SO>::subAssign( const DenseMatrix<MT,SO2>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_DIAGONAL_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   const bool remainder( !usePadding || !IsPadded<MT>::value );

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

      for( ; j<jpos; j+=SIMDSIZE ) {
         store( i, j, load(i,j) - (~rhs).load(i,j) );
      }
      for( ; remainder && j<jend; ++j ) {
         v_[i*NN+j] -= (~rhs)(i,j);
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline void HybridMatrix<Type,M,N,SO>::subAssign( const SparseMatrix<MT,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i*NN+element->index()] -= element->value();
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
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline void HybridMatrix<Type,M,N,SO>::subAssign( const SparseMatrix<MT,!SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()*NN+j] -= element->value();
}
//*************************************************************************************************








//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR COLUMN-MAJOR MATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of HybridMatrix for column-major matrices.
// \ingroup hybrid_matrix
//
// This specialization of HybridMatrix adapts the class template to the requirements of
// column-major matrices.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
class HybridMatrix<Type,M,N,true> : public DenseMatrix< HybridMatrix<Type,M,N,true>, true >
{
 public:
   //**Type definitions****************************************************************************
   typedef HybridMatrix<Type,M,N,true>   This;           //!< Type of this HybridMatrix instance.
   typedef DenseMatrix<This,true>        BaseType;       //!< Base type of this HybridMatrix instance.
   typedef This                          ResultType;     //!< Result type for expression template evaluations.
   typedef HybridMatrix<Type,M,N,false>  OppositeType;   //!< Result type with opposite storage order for expression template evaluations.
   typedef HybridMatrix<Type,N,M,false>  TransposeType;  //!< Transpose type for expression template evaluations.
   typedef Type                          ElementType;    //!< Type of the matrix elements.
   typedef SIMDTrait_<ElementType>       SIMDType;       //!< SIMD type of the matrix elements.
   typedef const Type&                   ReturnType;     //!< Return type for expression template evaluations.
   typedef const This&                   CompositeType;  //!< Data type for composite expression templates.

   typedef Type&        Reference;       //!< Reference to a non-constant matrix value.
   typedef const Type&  ConstReference;  //!< Reference to a constant matrix value.
   typedef Type*        Pointer;         //!< Pointer to a non-constant matrix value.
   typedef const Type*  ConstPointer;    //!< Pointer to a constant matrix value.

   typedef DenseIterator<Type,usePadding>        Iterator;       //!< Iterator over non-constant elements.
   typedef DenseIterator<const Type,usePadding>  ConstIterator;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a HybridMatrix with different data/element type.
   */
   template< typename ET >  // Data type of the other matrix
   struct Rebind {
      typedef HybridMatrix<ET,M,N,true>  Other;  //!< The type of the other HybridMatrix.
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
   enum : bool { smpAssignable = false };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline HybridMatrix();
   explicit inline HybridMatrix( size_t m, size_t n );
   explicit inline HybridMatrix( size_t m, size_t n, const Type& init );
   explicit inline HybridMatrix( initializer_list< initializer_list<Type> > list );

   template< typename Other >
   explicit inline HybridMatrix( size_t m, size_t n, const Other* array );

   template< typename Other, size_t M2, size_t N2 >
   explicit inline HybridMatrix( const Other (&array)[M2][N2] );

                                    inline HybridMatrix( const HybridMatrix& m );
   template< typename MT, bool SO > inline HybridMatrix( const Matrix<MT,SO>& m );
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
   inline HybridMatrix& operator=( const Type& set );
   inline HybridMatrix& operator=( initializer_list< initializer_list<Type> > list );

   template< typename Other, size_t M2, size_t N2 >
   inline HybridMatrix& operator=( const Other (&array)[M2][N2] );

                                    inline HybridMatrix& operator= ( const HybridMatrix& rhs );
   template< typename MT, bool SO > inline HybridMatrix& operator= ( const Matrix<MT,SO>& rhs );
   template< typename MT, bool SO > inline HybridMatrix& operator+=( const Matrix<MT,SO>& rhs );
   template< typename MT, bool SO > inline HybridMatrix& operator-=( const Matrix<MT,SO>& rhs );
   template< typename MT, bool SO > inline HybridMatrix& operator*=( const Matrix<MT,SO>& rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, HybridMatrix >& operator*=( Other rhs );

   template< typename Other >
   inline EnableIf_<IsNumeric<Other>, HybridMatrix >& operator/=( Other rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
                              inline size_t           rows() const noexcept;
                              inline size_t           columns() const noexcept;
                              inline constexpr size_t spacing() const noexcept;
                              inline constexpr size_t capacity() const noexcept;
                              inline           size_t capacity( size_t j ) const noexcept;
                              inline size_t           nonZeros() const;
                              inline size_t           nonZeros( size_t j ) const;
                              inline void             reset();
                              inline void             reset( size_t i );
                              inline void             clear();
                                     void             resize ( size_t m, size_t n, bool preserve=true );
                              inline void             extend ( size_t m, size_t n, bool preserve=true );
                              inline HybridMatrix&    transpose();
                              inline HybridMatrix&    ctranspose();
   template< typename Other > inline HybridMatrix&    scale( const Other& scalar );
                              inline void             swap( HybridMatrix& m ) noexcept;
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
   //! Helper structure for the explicit application of the SFINAE principle.
   template< typename MT >
   struct VectorizedAssign {
      enum : bool { value = useOptimizedKernels &&
                            simdEnabled && MT::simdEnabled &&
                            AreSIMDCombinable< Type, ElementType_<MT> >::value &&
                            IsColumnMajorMatrix<MT>::value };
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
                            IsColumnMajorMatrix<MT>::value &&
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
                            IsColumnMajorMatrix<MT>::value &&
                            !IsDiagonal<MT>::value };
   };
   //**********************************************************************************************

   //**********************************************************************************************
   //! The number of elements packed within a single SIMD element.
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

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t i, size_t j ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t i, size_t j, const SIMDType& value ) noexcept;

   template< typename MT, bool SO >
   inline DisableIf_<VectorizedAssign<MT> > assign( const DenseMatrix<MT,SO>& rhs );

   template< typename MT, bool SO >
   inline EnableIf_<VectorizedAssign<MT> > assign( const DenseMatrix<MT,SO>& rhs );

   template< typename MT > inline void assign( const SparseMatrix<MT,true>&  rhs );
   template< typename MT > inline void assign( const SparseMatrix<MT,false>& rhs );

   template< typename MT, bool SO >
   inline DisableIf_<VectorizedAddAssign<MT> > addAssign( const DenseMatrix<MT,SO>& rhs );

   template< typename MT, bool SO >
   inline EnableIf_<VectorizedAddAssign<MT> > addAssign( const DenseMatrix<MT,SO>& rhs );

   template< typename MT > inline void addAssign( const SparseMatrix<MT,true>&  rhs );
   template< typename MT > inline void addAssign( const SparseMatrix<MT,false>& rhs );

   template< typename MT, bool SO >
   inline DisableIf_<VectorizedSubAssign<MT> > subAssign( const DenseMatrix<MT,SO>& rhs );

   template< typename MT, bool SO >
   inline EnableIf_<VectorizedSubAssign<MT> > subAssign( const DenseMatrix<MT,SO>& rhs );

   template< typename MT > inline void subAssign( const SparseMatrix<MT,true>&  rhs );
   template< typename MT > inline void subAssign( const SparseMatrix<MT,false>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   //! Alignment adjustment.
   enum : size_t { MM = ( usePadding )?( NextMultiple< SizeT<M>, SizeT<SIMDSIZE> >::value ):( M ) };
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   AlignedArray<Type,MM*N> v_;  //!< The statically allocated matrix elements.
                                /*!< Access to the matrix elements is gained via the
                                     function call operator. */
   size_t m_;                   //!< The current number of rows of the matrix.
   size_t n_;                   //!< The current number of columns of the matrix.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE  ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST         ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE      ( Type );
   BLAZE_STATIC_ASSERT( !usePadding || MM % SIMDSIZE == 0UL );
   BLAZE_STATIC_ASSERT( MM >= M );
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
/*!\brief The default constructor for HybridMatrix.
//
// All matrix elements are initialized to the default value (i.e. 0 for integral data types).
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline HybridMatrix<Type,M,N,true>::HybridMatrix()
   : v_()       // The statically allocated matrix elements
   , m_( 0UL )  // The current number of rows of the matrix
   , n_( 0UL )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || MM == M );

   if( IsNumeric<Type>::value ) {
      for( size_t i=0UL; i<MM*N; ++i )
         v_[i] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a matrix of size \f$ m \times n \f$. No element initialization is performed!
//
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \exception std::invalid_argument Invalid number of rows for hybrid matrix.
// \exception std::invalid_argument Invalid number of columns for hybrid matrix.
//
// This constructor creates a hybrid matrix of size \f$ m \times n \f$, but leaves the elements
// uninitialized. In case \a m is larger than the maximum allowed number of rows (i.e. \a m > M)
// or \a n is larger than the maximum allowed number of columns a \a std::invalid_argument
// exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline HybridMatrix<Type,M,N,true>::HybridMatrix( size_t m, size_t n )
   : v_()     // The statically allocated matrix elements
   , m_( m )  // The current number of rows of the matrix
   , n_( n )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || MM == M );

   if( m > M ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of rows for hybrid matrix" );
   }

   if( n > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of columns for hybrid matrix" );
   }

   if( IsNumeric<Type>::value ) {
      for( size_t i=0UL; i<MM*N; ++i )
         v_[i] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a homogenous initialization of all \f$ m \times n \f$ matrix elements.
//
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \param init The initial value of the matrix elements.
// \exception std::invalid_argument Invalid number of rows for hybrid matrix.
// \exception std::invalid_argument Invalid number of columns for hybrid matrix.
//
// This constructor creates a hybrid matrix of size \f$ m \times n \f$ and initializes all
// matrix elements with the specified value. In case \a m is larger than the maximum allowed
// number of rows (i.e. \a m > M) or \a n is larger than the maximum allowed number of columns
// a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline HybridMatrix<Type,M,N,true>::HybridMatrix( size_t m, size_t n, const Type& init )
   : v_()     // The statically allocated matrix elements
   , m_( m )  // The current number of rows of the matrix
   , n_( n )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || MM == M );

   if( m > M ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of rows for hybrid matrix" );
   }

   if( n > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of columns for hybrid matrix" );
   }

   for( size_t j=0UL; j<n; ++j ) {
      for( size_t i=0UL; i<m; ++i )
         v_[i+j*MM] = init;

      if( IsNumeric<Type>::value ) {
         for( size_t i=m; i<MM; ++i )
            v_[i+j*MM] = Type();
      }
   }

   if( IsNumeric<Type>::value ) {
      for( size_t j=n; j<N; ++j )
         for( size_t i=0UL; i<MM; ++i )
            v_[i+j*MM] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List initialization of all matrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid number of rows for hybrid matrix.
// \exception std::invalid_argument Invalid number of columns for hybrid matrix.
//
// This constructor provides the option to explicitly initialize the elements of the matrix by
// means of an initializer list:

   \code
   using blaze::columnMajor;

   blaze::HybridMatrix<int,3,3,columnMajor> A{ { 1, 2, 3 },
                                               { 4, 5 },
                                               { 7, 8, 9 } };
   \endcode

// The matrix is sized according to the size of the initializer list and all its elements are
// initialized by the values of the given initializer list. Missing values are initialized as
// default (as e.g. the value 6 in the example). Note that in case the size of the top-level
// initializer list exceeds the number of rows or the size of any nested list exceeds the
// number of columns, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline HybridMatrix<Type,M,N,true>::HybridMatrix( initializer_list< initializer_list<Type> > list )
   : v_()                            // The statically allocated matrix elements
   , m_( list.size() )               // The current number of rows of the matrix
   , n_( determineColumns( list ) )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || MM == M );

   if( m_ > M ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of rows for hybrid matrix" );
   }

   if( n_ > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of columns for hybrid matrix" );
   }

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      size_t j( 0UL );
      for( const auto& element : rowList ) {
         v_[i+j*MM] = element;
         ++j;
      }
      if( IsNumeric<Type>::value ) {
         for( ; j<N; ++j ) {
            v_[i+j*MM] = Type();
         }
      }
      ++i;
   }

   BLAZE_INTERNAL_ASSERT( i == m_, "Invalid number of elements detected" );

   if( IsNumeric<Type>::value ) {
      for( ; i<MM; ++i ) {
         for( size_t j=0UL; j<N; ++j ) {
            v_[i+j*MM] = Type();
         }
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Array initialization of all matrix elements.
//
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \param array Dynamic array for the initialization.
// \exception std::invalid_argument Invalid number of rows for hybrid matrix.
// \exception std::invalid_argument Invalid number of columns for hybrid matrix.
//
// This constructor offers the option to directly initialize the elements of the matrix with
// a dynamic array:

   \code
   using blaze::columnMajor;

   int* array = new int[20];
   // ... Initialization of the dynamic array
   blaze::HybridMatrix<int,4,5,columnMajor> v( 4UL, 5UL, array );
   delete[] array;
   \endcode

// The matrix is sized according to the given size of the array and initialized with the values
// from the given array. In case \a m is larger than the maximum allowed number of rows (i.e.
// \a m > M) or \a n is larger than the maximum allowed number of columns a \a std::invalid_argument
// exception is thrown. Note that it is expected that the given \a array has at least \a m by
// \a n elements. Providing an array with less elements results in undefined behavior!
*/
template< typename Type     // Data type of the matrix
        , size_t M          // Number of rows
        , size_t N >        // Number of columns
template< typename Other >  // Data type of the initialization array
inline HybridMatrix<Type,M,N,true>::HybridMatrix( size_t m, size_t n, const Other* array )
   : v_()     // The statically allocated matrix elements
   , m_( m )  // The current number of rows of the matrix
   , n_( n )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || MM == M );

   if( m > M ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of rows for hybrid matrix" );
   }

   if( n > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of columns for hybrid matrix" );
   }

   for( size_t j=0UL; j<n; ++j ) {
      for( size_t i=0UL; i<m; ++i )
         v_[i+j*MM] = array[i+j*m];

      if( IsNumeric<Type>::value ) {
         for( size_t i=m; i<MM; ++i )
            v_[i+j*MM] = Type();
      }
   }

   if( IsNumeric<Type>::value ) {
      for( size_t j=n; j<N; ++j )
         for( size_t i=0UL; i<MM; ++i )
            v_[i+j*MM] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Array initialization of all matrix elements.
//
// \param array \f$ M \times N \f$ dimensional array for the initialization.
//
// This constructor offers the option to directly initialize the elements of the matrix with
// a static array:

   \code
   using blaze::columnMajor;

   const int init[3][3] = { { 1, 2, 3 },
                            { 4, 5 },
                            { 7, 8, 9 } };
   blaze::HybridMatrix<int,3,3,columnMajor> A( init );
   \endcode

// The matrix is sized according to the size of the array and initialized with the values from
// the given array. Missing values are initialized with default values (as e.g. the value 6 in
// the example).
*/
template< typename Type   // Data type of the matrix
        , size_t M        // Number of rows
        , size_t N >      // Number of columns
template< typename Other  // Data type of the initialization array
        , size_t M2       // Number of rows of the initialization array
        , size_t N2 >     // Number of columns of the initialization array
inline HybridMatrix<Type,M,N,true>::HybridMatrix( const Other (&array)[M2][N2] )
   : v_()      // The statically allocated matrix elements
   , m_( M2 )  // The current number of rows of the matrix
   , n_( N2 )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( M2 <= M );
   BLAZE_STATIC_ASSERT( N2 <= N );
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || MM == M );

   for( size_t j=0UL; j<N2; ++j ) {
      for( size_t i=0UL; i<M2; ++i )
         v_[i+j*MM] = array[i][j];

      if( IsNumeric<Type>::value ) {
         for( size_t i=M2; i<MM; ++i )
            v_[i+j*MM] = Type();
      }
   }

   if( IsNumeric<Type>::value ) {
      for( size_t j=N2; j<N; ++j )
         for( size_t i=0UL; i<MM; ++i )
            v_[i+j*MM] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The copy constructor for HybridMatrix.
//
// \param m Matrix to be copied.
//
// The copy constructor is explicitly defined due to the required dynamic memory management
// and in order to enable/facilitate NRV optimization.
*/
template< typename Type     // Data type of the matrix
        , size_t M          // Number of rows
        , size_t N >        // Number of columns
inline HybridMatrix<Type,M,N,true>::HybridMatrix( const HybridMatrix& m )
   : v_()        // The statically allocated matrix elements
   , m_( m.m_ )  // The current number of rows of the matrix
   , n_( m.n_ )  // The current number of columns of the matrix
{
   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || MM == M );

   for( size_t j=0UL; j<n_; ++j ) {
      for( size_t i=0UL; i<m_; ++i )
         v_[i+j*MM] = m.v_[i+j*MM];

      if( IsNumeric<Type>::value ) {
         for( size_t i=m_; i<MM; ++i )
            v_[i+j*MM] = Type();
      }
   }

   if( IsNumeric<Type>::value ) {
      for( size_t j=n_; j<N; ++j )
         for( size_t i=0UL; i<MM; ++i )
            v_[i+j*MM] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Conversion constructor from different matrices.
//
// \param m Matrix to be copied.
// \exception std::invalid_argument Invalid setup of hybrid matrix.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT    // Type of the foreign matrix
        , bool SO2 >     // Storage order of the foreign matrix
inline HybridMatrix<Type,M,N,true>::HybridMatrix( const Matrix<MT,SO2>& m )
   : v_()                  // The statically allocated matrix elements
   , m_( (~m).rows() )     // The current number of rows of the matrix
   , n_( (~m).columns() )  // The current number of columns of the matrix
{
   using blaze::assign;

   BLAZE_STATIC_ASSERT( IsVectorizable<Type>::value || MM == M );

   if( (~m).rows() > M || (~m).columns() > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of hybrid matrix" );
   }

   for( size_t j=0UL; j<n_; ++j ) {
      for( size_t i=( IsSparseMatrix<MT>::value ? 0UL : m_ );
                  i<( IsNumeric<Type>::value    ? MM  : m_ );
                  ++i ) {
         v_[i+j*MM] = Type();
      }
   }

   if( IsNumeric<Type>::value ) {
      for( size_t j=n_; j<N; ++j )
         for( size_t i=0UL; i<MM; ++i )
            v_[i+j*MM] = Type();
   }

   assign( *this, ~m );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
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
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::Reference
   HybridMatrix<Type,M,N,true>::operator()( size_t i, size_t j ) noexcept
{
   BLAZE_USER_ASSERT( i<M, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<N, "Invalid column access index" );
   return v_[i+j*MM];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return Reference-to-const to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::ConstReference
   HybridMatrix<Type,M,N,true>::operator()( size_t i, size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( i<M, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<N, "Invalid column access index" );
   return v_[i+j*MM];
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::Reference
   HybridMatrix<Type,M,N,true>::at( size_t i, size_t j )
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::ConstReference
   HybridMatrix<Type,M,N,true>::at( size_t i, size_t j ) const
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
// This function returns a pointer to the internal storage of the hybrid matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The hybrid matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a column are given by the \c columns() member functions, the total number
// of elements including padding is given by the \c spacing() member function.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::Pointer
   HybridMatrix<Type,M,N,true>::data() noexcept
{
   return v_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the matrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the hybrid matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The hybrid matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a column are given by the \c columns() member functions, the total number
// of elements including padding is given by the \c spacing() member function.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::ConstPointer
   HybridMatrix<Type,M,N,true>::data() const noexcept
{
   return v_;
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::Pointer
   HybridMatrix<Type,M,N,true>::data( size_t j ) noexcept
{
   BLAZE_USER_ASSERT( j < N, "Invalid dense matrix column access index" );
   return v_ + j*MM;
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
// This function returns a pointer to the internal storage for the elements in column \a j
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::ConstPointer
   HybridMatrix<Type,M,N,true>::data( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < N, "Invalid dense matrix column access index" );
   return v_ + j*MM;
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::Iterator
   HybridMatrix<Type,M,N,true>::begin( size_t j ) noexcept
{
   BLAZE_USER_ASSERT( j < N, "Invalid dense matrix column access index" );
   return Iterator( v_ + j*MM );
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::ConstIterator
   HybridMatrix<Type,M,N,true>::begin( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < N, "Invalid dense matrix column access index" );
   return ConstIterator( v_ + j*MM );
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::ConstIterator
   HybridMatrix<Type,M,N,true>::cbegin( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < N, "Invalid dense matrix column access index" );
   return ConstIterator( v_ + j*MM );
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::Iterator
   HybridMatrix<Type,M,N,true>::end( size_t j ) noexcept
{
   BLAZE_USER_ASSERT( j < N, "Invalid dense matrix column access index" );
   return Iterator( v_ + j*MM + M );
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::ConstIterator
   HybridMatrix<Type,M,N,true>::end( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < N, "Invalid dense matrix column access index" );
   return ConstIterator( v_ + j*MM + M );
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename HybridMatrix<Type,M,N,true>::ConstIterator
   HybridMatrix<Type,M,N,true>::cend( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < N, "Invalid dense matrix column access index" );
   return ConstIterator( v_ + j*MM + M );
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
// \param set Scalar value to be assigned to all matrix elements.
// \return Reference to the assigned matrix.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline HybridMatrix<Type,M,N,true>&
   HybridMatrix<Type,M,N,true>::operator=( const Type& set )
{
   BLAZE_INTERNAL_ASSERT( m_ <= M, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( n_ <= N, "Invalid number of columns detected" );

   for( size_t j=0UL; j<n_; ++j )
      for( size_t i=0UL; i<m_; ++i )
         v_[i+j*MM] = set;

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all matrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid number of rows for hybrid matrix.
// \exception std::invalid_argument Invalid number of columns for hybrid matrix.
//
// This assignment operator offers the option to directly assign to all elements of the matrix
// by means of an initializer list:

   \code
   using blaze::columnMajor;

   blaze::HybridMatrix<int,3,3,columnMajor> A;
   A = { { 1, 2, 3 },
         { 4, 5 },
         { 7, 8, 9 } };
   \endcode

// The matrix is resized according to the given initializer list and all its elements are
// assigned the values from the given initializer list. Missing values are initialized as
// default (as e.g. the value 6 in the example). Note that in case the size of the top-level
// initializer list exceeds the number of rows or the size of any nested list exceeds the
// number of columns, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline HybridMatrix<Type,M,N,true>&
   HybridMatrix<Type,M,N,true>::operator=( initializer_list< initializer_list<Type> > list )
{
   const size_t m( list.size() );
   const size_t n( determineColumns( list ) );

   if( m > M ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of rows for hybrid matrix" );
   }

   if( n > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of columns for hybrid matrix" );
   }

   resize( m, n, false );

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      size_t j( 0UL );
      for( const auto& element : rowList ) {
         v_[i+j*MM] = element;
         ++j;
      }
      for( ; j<n_; ++j ) {
         v_[i+j*MM] = Type();
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
//
// This assignment operator offers the option to directly set all elements of the matrix:

   \code
   using blaze::columnMajor;

   const int init[3][3] = { { 1, 2, 3 },
                            { 4, 5 },
                            { 7, 8, 9 } };
   blaze::HybridMatrix<int,3UL,3UL,columnMajor> A;
   A = init;
   \endcode

// The matrix is assigned the values from the given array. Missing values are initialized with
// default values (as e.g. the value 6 in the example).
*/
template< typename Type   // Data type of the matrix
        , size_t M        // Number of rows
        , size_t N >      // Number of columns
template< typename Other  // Data type of the initialization array
        , size_t M2       // Number of rows of the initialization array
        , size_t N2 >     // Number of columns of the initialization array
inline HybridMatrix<Type,M,N,true>&
   HybridMatrix<Type,M,N,true>::operator=( const Other (&array)[M2][N2] )
{
   BLAZE_STATIC_ASSERT( M2 <= M );
   BLAZE_STATIC_ASSERT( N2 <= N );

   resize( M2, N2 );

   for( size_t j=0UL; j<N2; ++j )
      for( size_t i=0UL; i<M2; ++i )
         v_[i+j*MM] = array[i][j];

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for HybridMatrix.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
//
// Explicit definition of a copy assignment operator for performance reasons.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline HybridMatrix<Type,M,N,true>&
   HybridMatrix<Type,M,N,true>::operator=( const HybridMatrix& rhs )
{
   using blaze::assign;

   BLAZE_INTERNAL_ASSERT( m_ <= M, "Invalid number of rows detected"    );
   BLAZE_INTERNAL_ASSERT( n_ <= N, "Invalid number of columns detected" );

   resize( rhs.rows(), rhs.columns() );
   assign( *this, ~rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

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
// \exception std::invalid_argument Invalid assignment to hybrid matrix.
//
// This constructor initializes the matrix as a copy of the given matrix. In case the
// number of rows of the given matrix is not M or the number of columns is not N, a
// \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT    // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline HybridMatrix<Type,M,N,true>& HybridMatrix<Type,M,N,true>::operator=( const Matrix<MT,SO>& rhs )
{
   using blaze::assign;

   typedef TransExprTrait_<This>   TT;
   typedef CTransExprTrait_<This>  CT;
   typedef InvExprTrait_<This>     IT;

   if( (~rhs).rows() > M || (~rhs).columns() > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to hybrid matrix" );
   }

   if( IsSame<MT,TT>::value && (~rhs).isAliased( this ) ) {
      transpose();
   }
   else if( IsSame<MT,CT>::value && (~rhs).isAliased( this ) ) {
      ctranspose();
   }
   else if( !IsSame<MT,IT>::value && (~rhs).canAlias( this ) ) {
      HybridMatrix tmp( ~rhs );
      resize( tmp.rows(), tmp.columns() );
      assign( *this, tmp );
   }
   else {
      resize( (~rhs).rows(), (~rhs).columns() );
      if( IsSparseMatrix<MT>::value )
         reset();
      assign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT    // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline HybridMatrix<Type,M,N,true>& HybridMatrix<Type,M,N,true>::operator+=( const Matrix<MT,SO>& rhs )
{
   using blaze::addAssign;

   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_<MT> tmp( ~rhs );
      addAssign( *this, tmp );
   }
   else {
      addAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT    // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline HybridMatrix<Type,M,N,true>& HybridMatrix<Type,M,N,true>::operator-=( const Matrix<MT,SO>& rhs )
{
   using blaze::subAssign;

   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_<MT> tmp( ~rhs );
      subAssign( *this, tmp );
   }
   else {
      subAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT    // Type of the right-hand side matrix
        , bool SO >      // Storage order of the right-hand side matrix
inline HybridMatrix<Type,M,N,true>& HybridMatrix<Type,M,N,true>::operator*=( const Matrix<MT,SO>& rhs )
{
   if( n_ != (~rhs).rows() || (~rhs).columns() > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const HybridMatrix tmp( *this * (~rhs) );
   this->operator=( tmp );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

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
        , size_t M          // Number of rows
        , size_t N >        // Number of columns
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, HybridMatrix<Type,M,N,true> >&
   HybridMatrix<Type,M,N,true>::operator*=( Other rhs )
{
   using blaze::assign;

   assign( *this, (*this) * rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

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
//
// \note A division by zero is only checked by an user assert.
*/
template< typename Type     // Data type of the matrix
        , size_t M          // Number of rows
        , size_t N >        // Number of columns
template< typename Other >  // Data type of the right-hand side scalar
inline EnableIf_<IsNumeric<Other>, HybridMatrix<Type,M,N,true> >&
   HybridMatrix<Type,M,N,true>::operator/=( Other rhs )
{
   using blaze::assign;

   BLAZE_USER_ASSERT( rhs != Other(0), "Division by zero detected" );

   assign( *this, (*this) / rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline size_t HybridMatrix<Type,M,N,true>::rows() const noexcept
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline size_t HybridMatrix<Type,M,N,true>::columns() const noexcept
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr size_t HybridMatrix<Type,M,N,true>::spacing() const noexcept
{
   return MM;
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr size_t HybridMatrix<Type,M,N,true>::capacity() const noexcept
{
   return MM*N;
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline size_t HybridMatrix<Type,M,N,true>::capacity( size_t j ) const noexcept
{
   UNUSED_PARAMETER( j );

   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return MM;
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline size_t HybridMatrix<Type,M,N,true>::nonZeros() const
{
   size_t nonzeros( 0UL );

   for( size_t j=0UL; j<n_; ++j )
      for( size_t i=0UL; i<m_; ++i )
         if( !isDefault( v_[i+j*MM] ) )
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline size_t HybridMatrix<Type,M,N,true>::nonZeros( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   const size_t iend( j*MM + m_ );
   size_t nonzeros( 0UL );

   for( size_t i=j*MM; i<iend; ++i )
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void HybridMatrix<Type,M,N,true>::reset()
{
   using blaze::clear;

   for( size_t j=0UL; j<n_; ++j )
      for( size_t i=0UL; i<m_; ++i )
         clear( v_[i+j*MM] );
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void HybridMatrix<Type,M,N,true>::reset( size_t j )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );
   for( size_t i=0UL; i<m_; ++i )
      clear( v_[i+j*MM] );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the hybrid matrix.
//
// \return void
//
// After the clear() function, the size of the matrix is 0.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void HybridMatrix<Type,M,N,true>::clear()
{
   resize( 0UL, 0UL );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Changing the size of the matrix.
//
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function resizes the matrix using the given size to \f$ m \times n \f$. In case the given
// number of rows \a m is larger than the maximum number of rows (i.e. if m > M) or in case the
// given number of columns \a n is larger than the maximum number of column (i.e. if n > N) a
// \a std::invalid_argument exception is thrown. Note that this function may invalidate all
// existing views (submatrices, rows, columns, ...) on the matrix if it is used to shrink the
// matrix. Additionally, during this operation all matrix elements are potentially changed. In
// order to preserve the old matrix values, the \a preserve flag can be set to \a true.
//
// Note that in case the number of rows or columns is increased new matrix elements are not
// initialized! The following example illustrates the resize operation of a \f$ 2 \times 4 \f$
// matrix to a \f$ 4 \times 2 \f$ matrix. The new, uninitialized elements are marked with \a x:

                              \f[
                              \left(\begin{array}{*{4}{c}}
                              1 & 2 & 3 & 4 \\
                              5 & 6 & 7 & 8 \\
                              \end{array}\right)

                              \Longrightarrow

                              \left(\begin{array}{*{2}{c}}
                              1 & 2 \\
                              5 & 6 \\
                              x & x \\
                              x & x \\
                              \end{array}\right)
                              \f]
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
void HybridMatrix<Type,M,N,true>::resize( size_t m, size_t n, bool preserve )
{
   UNUSED_PARAMETER( preserve );

   if( m > M ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of rows for hybrid matrix" );
   }

   if( n > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid number of columns for hybrid matrix" );
   }

   if( IsVectorizable<Type>::value && m < m_ ) {
      for( size_t j=0UL; j<n; ++j )
         for( size_t i=m; i<m_; ++i )
            v_[i+j*MM] = Type();
   }

   if( IsVectorizable<Type>::value && n < n_ ) {
      for( size_t j=n; j<n_; ++j )
         for( size_t i=0UL; i<m_; ++i )
            v_[i+j*MM] = Type();
   }

   m_ = m;
   n_ = n;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Extending the size of the matrix.
//
// \param m Number of additional rows.
// \param n Number of additional columns.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function increases the matrix size by \a m rows and \a n columns. In case the resulting
// number of rows or columns is larger than the maximum number of rows or columns (i.e. if m > M
// or n > N) a \a std::invalid_argument exception is thrown. During this operation, all matrix
// elements are potentially changed. In order to preserve the old matrix values, the \a preserve
// flag can be set to \a true.\n
// Note that new matrix elements are not initialized!
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void HybridMatrix<Type,M,N,true>::extend( size_t m, size_t n, bool preserve )
{
   UNUSED_PARAMETER( preserve );
   resize( m_+m, n_+n );
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
// This function transposes the hybrid matrix in-place. Note that this function can only be used
// on hybrid matrices whose current dimensions allow an in-place transpose operation. In case the
// current number of rows is larger than the maximum number of columns or if the current number
// of columns is larger than the maximum number of rows, an \a std::logic_error is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline HybridMatrix<Type,M,N,true>& HybridMatrix<Type,M,N,true>::transpose()
{
   using std::swap;

   if( m_ > N || n_ > M ) {
      BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
   }

   const size_t maxsize( max( m_, n_ ) );
   for( size_t j=1UL; j<maxsize; ++j ) {
      for( size_t i=0UL; i<j; ++i ) {
         swap( v_[i+j*MM], v_[j+i*MM] );
      }
   }

   if( IsVectorizable<Type>::value && n_ < m_ ) {
      for( size_t j=0UL; j<n_; ++j ) {
         for( size_t i=n_; i<m_; ++i ) {
            v_[i+j*MM] = Type();
         }
      }
   }

   if( IsVectorizable<Type>::value && n_ > m_ ) {
      for( size_t j=m_; j<n_; ++j ) {
         for( size_t i=0UL; i<m_; ++i ) {
            v_[i+j*MM] = Type();
         }
      }
   }

   swap( m_, n_ );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

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
// This function transposes the hybrid matrix in-place. Note that this function can only be used
// on hybrid matrices whose current dimensions allow an in-place transpose operation. In case the
// current number of rows is larger than the maximum number of columns or if the current number
// of columns is larger than the maximum number of rows, an \a std::logic_error is thrown.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline HybridMatrix<Type,M,N,true>& HybridMatrix<Type,M,N,true>::ctranspose()
{
   using std::swap;

   if( m_ > N || n_ > M ) {
      BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
   }

   const size_t maxsize( max( m_, n_ ) );
   for( size_t j=0UL; j<maxsize; ++j ) {
      for( size_t i=0UL; i<j; ++i ) {
         cswap( v_[i+j*MM], v_[j+i*MM] );
      }
      conjugate( v_[j+j*MM] );
   }

   if( IsVectorizable<Type>::value && n_ < m_ ) {
      for( size_t j=0UL; j<n_; ++j ) {
         for( size_t i=n_; i<m_; ++i ) {
            v_[i+j*MM] = Type();
         }
      }
   }

   if( IsVectorizable<Type>::value && n_ > m_ ) {
      for( size_t j=m_; j<n_; ++j ) {
         for( size_t i=0UL; i<m_; ++i ) {
            v_[i+j*MM] = Type();
         }
      }
   }

   swap( m_, n_ );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the matrix by the scalar value \a scalar (\f$ A*=s \f$).
//
// \param scalar The scalar value for the matrix scaling.
// \return Reference to the matrix.
*/
template< typename Type     // Data type of the matrix
        , size_t M          // Number of rows
        , size_t N >        // Number of columns
template< typename Other >  // Data type of the scalar value
inline HybridMatrix<Type,M,N,true>&
   HybridMatrix<Type,M,N,true>::scale( const Other& scalar )
{
   for( size_t j=0UL; j<n_; ++j )
      for( size_t i=0UL; i<m_; ++i )
         v_[i+j*MM] *= scalar;

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Swapping the contents of two hybrid matrices.
//
// \param m The matrix to be swapped.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void HybridMatrix<Type,M,N,true>::swap( HybridMatrix& m ) noexcept
{
   using std::swap;

   const size_t maxrows( max( m_, m.m_ ) );
   const size_t maxcols( max( n_, m.n_ ) );

   for( size_t j=0UL; j<maxcols; ++j ) {
      for( size_t i=0UL; i<maxrows; ++i ) {
         swap( v_[i+j*MM], m(i,j) );
      }
   }

   swap( m_, m.m_ );
   swap( n_, m.n_ );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MEMORY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Class specific implementation of operator new.
//
// \param size The total number of bytes to be allocated.
// \return Pointer to the newly allocated memory.
// \exception std::bad_alloc Allocation failed.
//
// This class-specific implementation of operator new provides the functionality to allocate
// dynamic memory based on the alignment restrictions of the HybridMatrix class template.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void* HybridMatrix<Type,M,N,true>::operator new( std::size_t size )
{
   UNUSED_PARAMETER( size );

   BLAZE_INTERNAL_ASSERT( size == sizeof( HybridMatrix ), "Invalid number of bytes detected" );

   return allocate<HybridMatrix>( 1UL );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Class specific implementation of operator new[].
//
// \param size The total number of bytes to be allocated.
// \return Pointer to the newly allocated memory.
// \exception std::bad_alloc Allocation failed.
//
// This class-specific implementation of operator new provides the functionality to allocate
// dynamic memory based on the alignment restrictions of the HybridMatrix class template.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void* HybridMatrix<Type,M,N,true>::operator new[]( std::size_t size )
{
   BLAZE_INTERNAL_ASSERT( size >= sizeof( HybridMatrix )       , "Invalid number of bytes detected" );
   BLAZE_INTERNAL_ASSERT( size %  sizeof( HybridMatrix ) == 0UL, "Invalid number of bytes detected" );

   return allocate<HybridMatrix>( size/sizeof(HybridMatrix) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Class specific implementation of the no-throw operator new.
//
// \param size The total number of bytes to be allocated.
// \return Pointer to the newly allocated memory.
// \exception std::bad_alloc Allocation failed.
//
// This class-specific implementation of operator new provides the functionality to allocate
// dynamic memory based on the alignment restrictions of the HybridMatrix class template.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void* HybridMatrix<Type,M,N,true>::operator new( std::size_t size, const std::nothrow_t& )
{
   UNUSED_PARAMETER( size );

   BLAZE_INTERNAL_ASSERT( size == sizeof( HybridMatrix ), "Invalid number of bytes detected" );

   return allocate<HybridMatrix>( 1UL );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Class specific implementation of the no-throw operator new[].
//
// \param size The total number of bytes to be allocated.
// \return Pointer to the newly allocated memory.
// \exception std::bad_alloc Allocation failed.
//
// This class-specific implementation of operator new provides the functionality to allocate
// dynamic memory based on the alignment restrictions of the HybridMatrix class template.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void* HybridMatrix<Type,M,N,true>::operator new[]( std::size_t size, const std::nothrow_t& )
{
   BLAZE_INTERNAL_ASSERT( size >= sizeof( HybridMatrix )       , "Invalid number of bytes detected" );
   BLAZE_INTERNAL_ASSERT( size %  sizeof( HybridMatrix ) == 0UL, "Invalid number of bytes detected" );

   return allocate<HybridMatrix>( size/sizeof(HybridMatrix) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Class specific implementation of operator delete.
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void HybridMatrix<Type,M,N,true>::operator delete( void* ptr )
{
   deallocate( static_cast<HybridMatrix*>( ptr ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Class specific implementation of operator delete[].
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void HybridMatrix<Type,M,N,true>::operator delete[]( void* ptr )
{
   deallocate( static_cast<HybridMatrix*>( ptr ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Class specific implementation of no-throw operator delete.
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void HybridMatrix<Type,M,N,true>::operator delete( void* ptr, const std::nothrow_t& )
{
   deallocate( static_cast<HybridMatrix*>( ptr ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Class specific implementation of no-throw operator delete[].
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void HybridMatrix<Type,M,N,true>::operator delete[]( void* ptr, const std::nothrow_t& )
{
   deallocate( static_cast<HybridMatrix*>( ptr ) );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DEBUGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the hybrid matrix are intact.
//
// \return \a true in case the hybrid matrix's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the hybrid matrix are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline bool HybridMatrix<Type,M,N,true>::isIntact() const noexcept
{
   if( m_ > M || n_ > N )
      return false;

   if( IsNumeric<Type>::value )
   {
      for( size_t j=0UL; j<n_; ++j ) {
         for( size_t i=m_; i<MM; ++i ) {
            if( v_[i+j*MM] != Type() )
               return false;
         }
      }

      for( size_t j=n_; j<N; ++j ) {
         for( size_t i=0UL; i<MM; ++i ) {
            if( v_[i+j*MM] != Type() )
               return false;
         }
      }
   }

   return true;
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
        , size_t M          // Number of rows
        , size_t N >        // Number of columns
template< typename Other >  // Data type of the foreign expression
inline bool HybridMatrix<Type,M,N,true>::canAlias( const Other* alias ) const noexcept
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
        , size_t M          // Number of rows
        , size_t N >        // Number of columns
template< typename Other >  // Data type of the foreign expression
inline bool HybridMatrix<Type,M,N,true>::isAliased( const Other* alias ) const noexcept
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline bool HybridMatrix<Type,M,N,true>::isAligned() const noexcept
{
   return ( usePadding || rows() % SIMDSIZE == 0UL );
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE typename HybridMatrix<Type,M,N,true>::SIMDType
   HybridMatrix<Type,M,N,true>::load( size_t i, size_t j ) const noexcept
{
   if( usePadding )
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE typename HybridMatrix<Type,M,N,true>::SIMDType
   HybridMatrix<Type,M,N,true>::loada( size_t i, size_t j ) const noexcept
{
   using blaze::loada;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= MM, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( &v_[i+j*MM] ), "Invalid alignment detected" );

   return loada( &v_[i+j*MM] );
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE typename HybridMatrix<Type,M,N,true>::SIMDType
   HybridMatrix<Type,M,N,true>::loadu( size_t i, size_t j ) const noexcept
{
   using blaze::loadu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= MM, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );

   return loadu( &v_[i+j*MM] );
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE void
   HybridMatrix<Type,M,N,true>::store( size_t i, size_t j, const SIMDType& value ) noexcept
{
   if( usePadding )
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE void
   HybridMatrix<Type,M,N,true>::storea( size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::storea;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= MM, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( &v_[i+j*MM] ), "Invalid alignment detected" );

   storea( &v_[i+j*MM], value );
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE void
   HybridMatrix<Type,M,N,true>::storeu( size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::storeu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= MM, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );

   storeu( &v_[i+j*MM], value );
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE void
   HybridMatrix<Type,M,N,true>::stream( size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::stream;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( i + SIMDSIZE <= MM, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || i % SIMDSIZE == 0UL, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( &v_[i+j*MM] ), "Invalid alignment detected" );

   stream( &v_[i+j*MM], value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a dense matrix.
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT    // Type of the right-hand side dense matrix
        , bool SO >      // Storage order of the right-hand side dense matrix
inline DisableIf_<typename HybridMatrix<Type,M,N,true>::BLAZE_TEMPLATE VectorizedAssign<MT> >
   HybridMatrix<Type,M,N,true>::assign( const DenseMatrix<MT,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t j=0UL; j<n_; ++j ) {
      for( size_t i=0UL; i<m_; ++i ) {
         v_[i+j*MM] = (~rhs)(i,j);
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the assignment of a dense matrix.
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT    // Type of the right-hand side dense matrix
        , bool SO >      // Storage order of the right-hand side dense matrix
inline EnableIf_<typename HybridMatrix<Type,M,N,true>::BLAZE_TEMPLATE VectorizedAssign<MT> >
   HybridMatrix<Type,M,N,true>::assign( const DenseMatrix<MT,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   const bool remainder( !usePadding || !IsPadded<MT>::value );

   const size_t ipos( ( remainder )?( m_ & size_t(-SIMDSIZE) ):( m_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( m_ - ( m_ % (SIMDSIZE) ) ) == ipos, "Invalid end calculation" );

   for( size_t j=0UL; j<n_; ++j )
   {
      size_t i( 0UL );

      for( ; i<ipos; i+=SIMDSIZE ) {
         store( i, j, (~rhs).load(i,j) );
      }
      for( ; remainder && i<m_; ++i ) {
         v_[i+j*MM] = (~rhs)(i,j);
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side sparse matrix
inline void HybridMatrix<Type,M,N,true>::assign( const SparseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()+j*MM] = element->value();
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side sparse matrix
inline void HybridMatrix<Type,M,N,true>::assign( const SparseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i+element->index()*MM] = element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a dense matrix.
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT    // Type of the right-hand side dense matrix
        , bool SO >      // Storage order of the right-hand side dense matrix
inline DisableIf_<typename HybridMatrix<Type,M,N,true>::BLAZE_TEMPLATE VectorizedAddAssign<MT> >
   HybridMatrix<Type,M,N,true>::addAssign( const DenseMatrix<MT,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t j=0UL; j<n_; ++j )
   {
      if( IsDiagonal<MT>::value )
      {
         v_[j+j*MM] += (~rhs)(j,j);
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

         for( size_t i=ibegin; i<iend; ++i ) {
            v_[i+j*MM] += (~rhs)(i,j);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the addition assignment of a dense matrix.
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT    // Type of the right-hand side dense matrix
        , bool SO >      // Storage order of the right-hand side dense matrix
inline EnableIf_<typename HybridMatrix<Type,M,N,true>::BLAZE_TEMPLATE VectorizedAddAssign<MT> >
   HybridMatrix<Type,M,N,true>::addAssign( const DenseMatrix<MT,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_DIAGONAL_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   const bool remainder( !usePadding || !IsPadded<MT>::value );

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

      for( ; i<ipos; i+=SIMDSIZE ) {
         store( i, j, load(i,j) + (~rhs).load(i,j) );
      }
      for( ; remainder && i<iend; ++i ) {
         v_[i+j*MM] += (~rhs)(i,j);
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side sparse matrix
inline void HybridMatrix<Type,M,N,true>::addAssign( const SparseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t j=0UL; j<n_; ++j )
      for( ConstIterator_<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()+j*MM] += element->value();
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side sparse matrix
inline void HybridMatrix<Type,M,N,true>::addAssign( const SparseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t i=0UL; i<m_; ++i )
      for( ConstIterator_<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i+element->index()*MM] += element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a dense matrix.
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT    // Type of the right-hand side dense matrix
        , bool SO >      // Storage order of the right-hand side dense matrix
inline DisableIf_<typename HybridMatrix<Type,M,N,true>::BLAZE_TEMPLATE VectorizedSubAssign<MT> >
   HybridMatrix<Type,M,N,true>::subAssign( const DenseMatrix<MT,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   for( size_t j=0UL; j<n_; ++j )
   {
      if( IsDiagonal<MT>::value )
      {
         v_[j+j*MM] -= (~rhs)(j,j);
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

         for( size_t i=ibegin; i<iend; ++i ) {
            v_[i+j*MM] -= (~rhs)(i,j);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the subtraction assignment of a dense matrix.
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT    // Type of the right-hand side dense matrix
        , bool SO >      // Storage order of the right-hand side dense matrix
inline EnableIf_<typename HybridMatrix<Type,M,N,true>::BLAZE_TEMPLATE VectorizedSubAssign<MT> >
   HybridMatrix<Type,M,N,true>::subAssign( const DenseMatrix<MT,SO>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_DIAGONAL_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   const bool remainder( !usePadding || !IsPadded<MT>::value );

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

      for( ; i<ipos; i+=SIMDSIZE ) {
         store( i, j, load(i,j) - (~rhs).load(i,j) );
      }
      for( ; remainder && i<iend; ++i ) {
         v_[i+j*MM] -= (~rhs)(i,j);
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side sparse matrix
inline void HybridMatrix<Type,M,N,true>::subAssign( const SparseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   typedef ConstIterator_<MT>  RhsConstIterator;

   for( size_t j=0UL; j<n_; ++j )
      for( RhsConstIterator element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()+j*MM] -= element->value();
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
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side sparse matrix
inline void HybridMatrix<Type,M,N,true>::subAssign( const SparseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == m_ && (~rhs).columns() == n_, "Invalid matrix size" );

   typedef ConstIterator_<MT>  RhsConstIterator;

   for( size_t i=0UL; i<m_; ++i )
      for( RhsConstIterator element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i+element->index()*MM] -= element->value();
}
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  UNDEFINED CLASS TEMPLATE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of HybridMatrix for zero columns.
// \ingroup hybrid_matrix
//
// This specialization of the HybridMatrix class template is left undefined and therefore
// prevents the instantiation for zero columns.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , bool SO >      // Storage order
class HybridMatrix<Type,M,0UL,SO>;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of HybridMatrix for zero rows.
// \ingroup hybrid_matrix
//
// This specialization of the HybridMatrix class template is left undefined and therefore
// prevents the instantiation for zero rows.
*/
template< typename Type  // Data type of the matrix
        , size_t N       // Number of columns
        , bool SO >      // Storage order
class HybridMatrix<Type,0UL,N,SO>;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of HybridMatrix for 0 rows and 0 columns.
// \ingroup hybrid_matrix
//
// This specialization of the HybridMatrix class template is left undefined and therefore
// prevents the instantiation for 0 rows and 0 columns.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
class HybridMatrix<Type,0UL,0UL,SO>;
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  STATICMATRIX OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name HybridMatrix operators */
//@{
template< typename Type, size_t M, size_t N, bool SO >
inline void reset( HybridMatrix<Type,M,N,SO>& m );

template< typename Type, size_t M, size_t N, bool SO >
inline void reset( HybridMatrix<Type,M,N,SO>& m, size_t i );

template< typename Type, size_t M, size_t N, bool SO >
inline void clear( HybridMatrix<Type,M,N,SO>& m );

template< typename Type, size_t M, size_t N, bool SO >
inline bool isDefault( const HybridMatrix<Type,M,N,SO>& m );

template< typename Type, size_t M, size_t N, bool SO >
inline bool isIntact( const HybridMatrix<Type,M,N,SO>& m ) noexcept;

template< typename Type, size_t M, size_t N, bool SO >
inline void swap( HybridMatrix<Type,M,N,SO>& a, HybridMatrix<Type,M,N,SO>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given hybrid matrix.
// \ingroup hybrid_matrix
//
// \param m The matrix to be resetted.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void reset( HybridMatrix<Type,M,N,SO>& m )
{
   m.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset the specified row/column of the given hybrid matrix.
// \ingroup hybrid_matrix
//
// \param m The matrix to be resetted.
// \param i The index of the row/column to be resetted.
// \return void
//
// This function resets the values in the specified row/column of the given hybrid matrix to
// their default value. In case the given matrix is a \a rowMajor matrix the function resets the
// values in row \a i, if it is a \a columnMajor matrix the function resets the values in column
// \a i. Note that the capacity of the row/column remains unchanged.
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void reset( HybridMatrix<Type,M,N,SO>& m, size_t i )
{
   m.reset( i );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given hybrid matrix.
// \ingroup hybrid_matrix
//
// \param m The matrix to be cleared.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void clear( HybridMatrix<Type,M,N,SO>& m )
{
   m.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given hybrid matrix is in default state.
// \ingroup hybrid_matrix
//
// \param m The hybrid matrix to be tested for its default state.
// \return \a true in case the given matrix's rows and columns are zero, \a false otherwise.
//
// This function checks whether the hybrid matrix is in default (constructed) state, i.e. if
// it's number of rows and columns is 0. In case it is in default state, the function returns
// \a true, else it will return \a false. The following example demonstrates the use of the
// \a isDefault() function:

   \code
   blaze::HybridMatrix<double,3,5> A;
   // ... Resizing and initialization
   if( isDefault( A ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline bool isDefault( const HybridMatrix<Type,M,N,SO>& m )
{
   return ( m.rows() == 0UL && m.columns() == 0UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given hybrid matrix are intact.
// \ingroup hybrid_matrix
//
// \param m The hybrid matrix to be tested.
// \return \a true in case the given matrix's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the hybrid matrix are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   blaze::HybridMatrix<double,3,5> A;
   // ... Resizing and initialization
   if( isIntact( A ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline bool isIntact( const HybridMatrix<Type,M,N,SO>& m ) noexcept
{
   return m.isIntact();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two hybrid matrices.
// \ingroup hybrid_matrix
//
// \param a The first matrix to be swapped.
// \param b The second matrix to be swapped.
// \return void
*/
template< typename Type  // Data type of the matrix
        , size_t M       // Number of rows
        , size_t N       // Number of columns
        , bool SO >      // Storage order
inline void swap( HybridMatrix<Type,M,N,SO>& a, HybridMatrix<Type,M,N,SO>& b ) noexcept
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
template< typename T, size_t M, size_t N, bool SO >
struct HasConstDataAccess< HybridMatrix<T,M,N,SO> > : public TrueType
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
template< typename T, size_t M, size_t N, bool SO >
struct HasMutableDataAccess< HybridMatrix<T,M,N,SO> > : public TrueType
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
template< typename T, size_t M, size_t N, bool SO >
struct IsAligned< HybridMatrix<T,M,N,SO> > : public BoolConstant<usePadding>
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
template< typename T, size_t M, size_t N, bool SO >
struct IsPadded< HybridMatrix<T,M,N,SO> > : public BoolConstant<usePadding>
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
template< typename T, size_t M, size_t N, bool SO >
struct IsResizable< HybridMatrix<T,M,N,SO> > : public TrueType
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
template< typename T1, size_t M1, size_t N1, bool SO, typename T2, size_t M2, size_t N2 >
struct AddTrait< HybridMatrix<T1,M1,N1,SO>, StaticMatrix<T2,M2,N2,SO> >
{
   using Type = StaticMatrix< AddTrait_<T1,T2>, M2, N2, SO >;
};

template< typename T1, size_t M1, size_t N1, bool SO1, typename T2, size_t M2, size_t N2, bool SO2 >
struct AddTrait< HybridMatrix<T1,M1,N1,SO1>, StaticMatrix<T2,M2,N2,SO2> >
{
   using Type = StaticMatrix< AddTrait_<T1,T2>, M2, N2, false >;
};

template< typename T1, size_t M1, size_t N1, bool SO, typename T2, size_t M2, size_t N2 >
struct AddTrait< StaticMatrix<T1,M1,N1,SO>, HybridMatrix<T2,M2,N2,SO> >
{
   using Type = StaticMatrix< AddTrait_<T1,T2>, M2, N2, SO >;
};

template< typename T1, size_t M1, size_t N1, bool SO1, typename T2, size_t M2, size_t N2, bool SO2 >
struct AddTrait< StaticMatrix<T1,M1,N1,SO1>, HybridMatrix<T2,M2,N2,SO2> >
{
   using Type = StaticMatrix< AddTrait_<T1,T2>, M2, N2, false >;
};

template< typename T1, size_t M1, size_t N1, bool SO, typename T2, size_t M2, size_t N2 >
struct AddTrait< HybridMatrix<T1,M1,N1,SO>, HybridMatrix<T2,M2,N2,SO> >
{
   using Type = HybridMatrix< AddTrait_<T1,T2>, ( M1 < M2 )?( M1 ):( M2 ), ( N1 < N2 )?( N1 ):( N2 ), SO >;
};

template< typename T1, size_t M1, size_t N1, bool SO1, typename T2, size_t M2, size_t N2, bool SO2 >
struct AddTrait< HybridMatrix<T1,M1,N1,SO1>, HybridMatrix<T2,M2,N2,SO2> >
{
   using Type = HybridMatrix< AddTrait_<T1,T2>, ( M1 < M2 )?( M1 ):( M2 ), ( N1 < N2 )?( N1 ):( N2 ), false >;
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
template< typename T1, size_t M1, size_t N1, bool SO, typename T2, size_t M2, size_t N2 >
struct SubTrait< HybridMatrix<T1,M1,N1,SO>, StaticMatrix<T2,M2,N2,SO> >
{
   using Type = StaticMatrix< SubTrait_<T1,T2>, M2, N2, SO >;
};

template< typename T1, size_t M1, size_t N1, bool SO1, typename T2, size_t M2, size_t N2, bool SO2 >
struct SubTrait< HybridMatrix<T1,M1,N1,SO1>, StaticMatrix<T2,M2,N2,SO2> >
{
   using Type = StaticMatrix< SubTrait_<T1,T2>, M2, N2, false >;
};

template< typename T1, size_t M1, size_t N1, bool SO, typename T2, size_t M2, size_t N2 >
struct SubTrait< StaticMatrix<T1,M1,N1,SO>, HybridMatrix<T2,M2,N2,SO> >
{
   using Type = StaticMatrix< SubTrait_<T1,T2>, M2, N2, SO >;
};

template< typename T1, size_t M1, size_t N1, bool SO1, typename T2, size_t M2, size_t N2, bool SO2 >
struct SubTrait< StaticMatrix<T1,M1,N1,SO1>, HybridMatrix<T2,M2,N2,SO2> >
{
   using Type = StaticMatrix< SubTrait_<T1,T2>, M2, N2, false >;
};

template< typename T1, size_t M1, size_t N1, bool SO, typename T2, size_t M2, size_t N2 >
struct SubTrait< HybridMatrix<T1,M1,N1,SO>, HybridMatrix<T2,M2,N2,SO> >
{
   using Type = HybridMatrix< SubTrait_<T1,T2>, ( M1 < M2 )?( M1 ):( M2 ), ( N1 < N2 )?( N1 ):( N2 ), SO >;
};

template< typename T1, size_t M1, size_t N1, bool SO1, typename T2, size_t M2, size_t N2, bool SO2 >
struct SubTrait< HybridMatrix<T1,M1,N1,SO1>, HybridMatrix<T2,M2,N2,SO2> >
{
   using Type = HybridMatrix< SubTrait_<T1,T2>, ( M1 < M2 )?( M1 ):( M2 ), ( N1 < N2 )?( N1 ):( N2 ), false >;
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
template< typename T1, size_t M, size_t N, bool SO, typename T2 >
struct MultTrait< HybridMatrix<T1,M,N,SO>, T2, EnableIf_<IsNumeric<T2> > >
{
   using Type = HybridMatrix< MultTrait_<T1,T2>, M, N, SO >;
};

template< typename T1, typename T2, size_t M, size_t N, bool SO >
struct MultTrait< T1, HybridMatrix<T2,M,N,SO>, EnableIf_<IsNumeric<T1> > >
{
   using Type = HybridMatrix< MultTrait_<T1,T2>, M, N, SO >;
};

template< typename T1, size_t M, size_t N, bool SO, typename T2, size_t K >
struct MultTrait< HybridMatrix<T1,M,N,SO>, StaticVector<T2,K,false> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, M, false >;
};

template< typename T1, size_t K, typename T2, size_t M, size_t N, bool SO >
struct MultTrait< StaticVector<T1,K,true>, HybridMatrix<T2,M,N,SO> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, N, true >;
};

template< typename T1, size_t M, size_t N, bool SO, typename T2, size_t K >
struct MultTrait< HybridMatrix<T1,M,N,SO>, HybridVector<T2,K,false> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, M, false >;
};

template< typename T1, size_t K, typename T2, size_t M, size_t N, bool SO >
struct MultTrait< HybridVector<T1,K,true>, HybridMatrix<T2,M,N,SO> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, N, true >;
};

template< typename T1, size_t M, size_t N, bool SO, typename T2 >
struct MultTrait< HybridMatrix<T1,M,N,SO>, DynamicVector<T2,false> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, M, false >;
};

template< typename T1, typename T2, size_t M, size_t N, bool SO >
struct MultTrait< DynamicVector<T1,true>, HybridMatrix<T2,M,N,SO> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, N, true >;
};

template< typename T1, size_t M, size_t N, bool SO, typename T2, bool AF, bool PF >
struct MultTrait< HybridMatrix<T1,M,N,SO>, CustomVector<T2,AF,PF,false> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, M, false >;
};

template< typename T1, bool AF, bool PF, typename T2, size_t M, size_t N, bool SO >
struct MultTrait< CustomVector<T1,AF,PF,true>, HybridMatrix<T2,M,N,SO> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, N, true >;
};

template< typename T1, size_t M, size_t N, bool SO, typename T2 >
struct MultTrait< HybridMatrix<T1,M,N,SO>, CompressedVector<T2,false> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, M, false >;
};

template< typename T1, typename T2, size_t M, size_t N, bool SO >
struct MultTrait< CompressedVector<T1,true>, HybridMatrix<T2,M,N,SO> >
{
   using Type = HybridVector< MultTrait_<T1,T2>, N, true >;
};

template< typename T1, size_t M1, size_t N1, bool SO1, typename T2, size_t M2, size_t N2, bool SO2 >
struct MultTrait< HybridMatrix<T1,M1,N1,SO1>, StaticMatrix<T2,M2,N2,SO2> >
{
   using Type = HybridMatrix< MultTrait_<T1,T2>, M1, N2, SO1 >;
};

template< typename T1, size_t M1, size_t N1, bool SO1, typename T2, size_t M2, size_t N2, bool SO2 >
struct MultTrait< StaticMatrix<T1,M1,N1,SO1>, HybridMatrix<T2,M2,N2,SO2> >
{
   using Type = HybridMatrix< MultTrait_<T1,T2>, M1, N2, SO1 >;
};

template< typename T1, size_t M1, size_t N1, bool SO1, typename T2, size_t M2, size_t N2, bool SO2 >
struct MultTrait< HybridMatrix<T1,M1,N1,SO1>, HybridMatrix<T2,M2,N2,SO2> >
{
   using Type = HybridMatrix< MultTrait_<T1,T2>, M1, N2, SO1 >;
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
template< typename T1, size_t M, size_t N, bool SO, typename T2 >
struct DivTrait< HybridMatrix<T1,M,N,SO>, T2, EnableIf_<IsNumeric<T2> > >
{
   using Type = HybridMatrix< DivTrait_<T1,T2>, M, N, SO >;
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
template< typename T1, size_t M, size_t N, bool SO, typename T2 >
struct MathTrait< HybridMatrix<T1,M,N,SO>, HybridMatrix<T2,M,N,SO> >
{
   using HighType = HybridMatrix< typename MathTrait<T1,T2>::HighType, M, N, SO >;
   using LowType  = HybridMatrix< typename MathTrait<T1,T2>::LowType , M, N, SO >;
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
template< typename T1, size_t M, size_t N, bool SO >
struct SubmatrixTrait< HybridMatrix<T1,M,N,SO> >
{
   using Type = HybridMatrix<T1,M,N,SO>;
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
template< typename T1, size_t M, size_t N, bool SO >
struct RowTrait< HybridMatrix<T1,M,N,SO> >
{
   using Type = HybridVector<T1,N,true>;
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
template< typename T1, size_t M, size_t N, bool SO >
struct ColumnTrait< HybridMatrix<T1,M,N,SO> >
{
   using Type = HybridVector<T1,M,false>;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
