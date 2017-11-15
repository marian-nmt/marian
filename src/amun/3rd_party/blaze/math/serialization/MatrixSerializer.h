//=================================================================================================
/*!
//  \file blaze/math/serialization/MatrixSerializer.h
//  \brief Serialization of dense and sparse matrices
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

#ifndef _BLAZE_MATH_SERIALIZATION_MATRIXSERIALIZER_H_
#define _BLAZE_MATH_SERIALIZATION_MATRIXSERIALIZER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Matrix.h>
#include <blaze/math/dense/DynamicMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/expressions/Matrix.h>
#include <blaze/math/serialization/TypeValueMapping.h>
#include <blaze/math/sparse/CompressedMatrix.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/util/Assert.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsNumeric.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Serializer for dense and sparse matrices.
// \ingroup math_serialization
//
// The MatrixSerializer implements the necessary logic to serialize dense and sparse matrices,
// i.e. to convert them into a portable, binary representation. The following example demonstrates
// the (de-)serialization process of matrices:

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
      blaze::Archive<std::ofstream> archive( "matrices.blaze" );

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

// Note that it is even possible to (de-)serialize matrices with vector or matrix elements:

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
      blaze::Archive<std::ofstream> archive( "matrix.blaze" );

      // Reconstitution of the matrix from the archive
      archive >> mat;
   }
   \endcode

// As the examples demonstrates, the matrix serialization offers an enormous flexibility. However,
// several actions result in errors:
//
//  - matrices cannot be reconstituted as vectors (and vice versa)
//  - the element type of the serialized and reconstituted matrix must match, which means
//    that on the source and destination platform the general type (signed/unsigned integral
//    or floating point) and the size of the type must be exactly the same
//  - when reconstituting a StaticMatrix, the number of rows and columns must match those of
//    the serialized matrix
//
// In case an error is encountered during (de-)serialization, a \a std::runtime_exception is
// thrown.
*/
class MatrixSerializer
{
 private:
   //**Private class MatrixValueMappingHelper******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Auxiliary helper class for the MatrixValueMapping class template.
   //
   // The MatrixValueMapping class template is an auxiliary class for the MatrixSerializer. It
   // maps a matrix type into an integral representation. For the mapping, the following bit
   // mapping is used:

      \code
      0x01 - Vector/Matrix flag
      0x02 - Dense/Sparse flag
      0x04 - Row-/Column-major flag
      \endcode
   */
   template< bool IsDenseMatrix, bool IsRowMajorMatrix >
   struct MatrixValueMappingHelper;
   /*! \endcond */
   //**********************************************************************************************

   //**Private class MatrixValueMapping************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Serialization of the type of a matrix.
   //
   // This class template converts the given matrix type into an integral representation suited
   // for serialization. Depending on the given matrix type, the \a value member enumeration is
   // set to the according integral representation.
   */
   template< typename T >
   struct MatrixValueMapping
   {
      enum { value = MatrixValueMappingHelper< IsDenseMatrix<T>::value, IsRowMajorMatrix<T>::value >::value };
      BLAZE_CONSTRAINT_MUST_BE_MATRIX_TYPE( T );
   };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Constructor*********************************************************************************
   /*!\name Constructor */
   //@{
   explicit inline MatrixSerializer();
   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   // No explicitly declared copy assignment operator.
   //**********************************************************************************************

   //**Serialization functions*********************************************************************
   /*!\name Serialization functions */
   //@{
   template< typename Archive, typename MT, bool SO >
   void serialize( Archive& archive, const Matrix<MT,SO>& mat );
   //@}
   //**********************************************************************************************

   //**Deserialization functions*******************************************************************
   /*!\name Deserialization functions */
   //@{
   template< typename Archive, typename MT, bool SO >
   void deserialize( Archive& archive, Matrix<MT,SO>& mat );
   //@}
   //**********************************************************************************************

 private:
   //**Serialization functions*********************************************************************
   /*!\name Serialization functions */
   //@{
   template< typename Archive, typename MT >
   void serializeHeader( Archive& archive, const MT& mat );

   template< typename Archive, typename MT, bool SO >
   void serializeMatrix( Archive& archive, const DenseMatrix<MT,SO>& mat );

   template< typename Archive, typename MT, bool SO >
   void serializeMatrix( Archive& archive, const SparseMatrix<MT,SO>& mat );
   //@}
   //**********************************************************************************************

   //**Deserialization functions*******************************************************************
   /*!\name Deserialization functions */
   //@{
   template< typename Archive, typename MT >
   void deserializeHeader( Archive& archive, const MT& mat );

   template< typename MT, bool SO >
   DisableIf_< IsResizable<MT> > prepareMatrix( DenseMatrix<MT,SO>& mat );

   template< typename MT, bool SO >
   DisableIf_< IsResizable<MT> > prepareMatrix( SparseMatrix<MT,SO>& mat );

   template< typename MT >
   EnableIf_< IsResizable<MT> > prepareMatrix( MT& mat );

   template< typename Archive, typename MT >
   void deserializeMatrix( Archive& archive, MT& mat );

   template< typename Archive, typename MT >
   EnableIfTrue_< MT::simdEnabled >
      deserializeDenseRowMatrix( Archive& archive, DenseMatrix<MT,rowMajor>& mat );

   template< typename Archive, typename MT, bool SO >
   void deserializeDenseRowMatrix( Archive& archive, DenseMatrix<MT,SO>& mat );

   template< typename Archive, typename MT, bool SO >
   DisableIf_< IsNumeric< ElementType_<MT> > >
      deserializeDenseRowMatrix( Archive& archive, SparseMatrix<MT,SO>& mat );

   template< typename Archive, typename MT, bool SO >
   EnableIf_< IsNumeric< ElementType_<MT> > >
      deserializeDenseRowMatrix( Archive& archive, SparseMatrix<MT,SO>& mat );

   template< typename Archive, typename MT >
   EnableIfTrue_< MT::simdEnabled>
      deserializeDenseColumnMatrix( Archive& archive, DenseMatrix<MT,columnMajor>& mat );

   template< typename Archive, typename MT, bool SO >
   void deserializeDenseColumnMatrix( Archive& archive, DenseMatrix<MT,SO>& mat );

   template< typename Archive, typename MT, bool SO >
   DisableIf_< IsNumeric< ElementType_<MT> > >
      deserializeDenseColumnMatrix( Archive& archive, SparseMatrix<MT,SO>& mat );

   template< typename Archive, typename MT, bool SO >
   EnableIf_< IsNumeric< ElementType_<MT> > >
      deserializeDenseColumnMatrix( Archive& archive, SparseMatrix<MT,SO>& mat );

   template< typename Archive, typename MT, bool SO >
   void deserializeSparseRowMatrix( Archive& archive, DenseMatrix<MT,SO>& mat );

   template< typename Archive, typename MT >
   void deserializeSparseRowMatrix( Archive& archive, SparseMatrix<MT,rowMajor>& mat );

   template< typename Archive, typename MT >
   void deserializeSparseRowMatrix( Archive& archive, SparseMatrix<MT,columnMajor>& mat );

   template< typename Archive, typename MT, bool SO >
   void deserializeSparseColumnMatrix( Archive& archive, DenseMatrix<MT,SO>& mat );

   template< typename Archive, typename MT >
   void deserializeSparseColumnMatrix( Archive& archive, SparseMatrix<MT,rowMajor>& mat );

   template< typename Archive, typename MT >
   void deserializeSparseColumnMatrix( Archive& archive, SparseMatrix<MT,columnMajor>& mat );
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   uint8_t  version_;      //!< The version of the archive.
   uint8_t  type_;         //!< The type of the matrix.
   uint8_t  elementType_;  //!< The type of an element.
   uint8_t  elementSize_;  //!< The size in bytes of a single element of the matrix.
   uint64_t rows_;         //!< The number of rows of the matrix.
   uint64_t columns_;      //!< The number of columns of the matrix.
   uint64_t number_;       //!< The total number of elements contained in the matrix.
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The default constructor of the MatrixSerializer class.
*/
MatrixSerializer::MatrixSerializer()
   : version_    ( 0U  )  // The version of the archive
   , type_       ( 0U  )  // The type of the matrix
   , elementType_( 0U  )  // The type of an element
   , elementSize_( 0U  )  // The size in bytes of a single element of the matrix
   , rows_       ( 0UL )  // The number of rows of the matrix
   , columns_    ( 0UL )  // The number of columns of the matrix
   , number_     ( 0UL )  // The total number of elements contained in the matrix
{}
//*************************************************************************************************




//=================================================================================================
//
//  SERIALIZATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Serializes the given matrix and writes it to the archive.
//
// \param archive The archive to be written.
// \param mat The matrix to be serialized.
// \return void
// \exception std::runtime_error Error during serialization.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
void MatrixSerializer::serialize( Archive& archive, const Matrix<MT,SO>& mat )
{
   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Faulty archive detected" );
   }

   serializeHeader( archive, ~mat );
   serializeMatrix( archive, ~mat );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Serializes all meta information about the given matrix.
//
// \param archive The archive to be written.
// \param mat The matrix to be serialized.
// \return void
// \exception std::runtime_error File header could not be serialized.
*/
template< typename Archive  // Type of the archive
        , typename MT >     // Type of the matrix
void MatrixSerializer::serializeHeader( Archive& archive, const MT& mat )
{
   typedef ElementType_<MT>  ET;

   archive << uint8_t ( 1U );
   archive << uint8_t ( MatrixValueMapping<MT>::value );
   archive << uint8_t ( TypeValueMapping<ET>::value );
   archive << uint8_t ( sizeof( ET ) );
   archive << uint64_t( mat.rows() );
   archive << uint64_t( mat.columns() );
   archive << uint64_t( ( IsDenseMatrix<MT>::value ) ? ( mat.rows()*mat.columns() ) : ( mat.nonZeros() ) );

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "File header could not be serialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Serializes the elements of a dense matrix.
//
// \param archive The archive to be written.
// \param mat The matrix to be serialized.
// \return void
// \exception std::runtime_error Dense matrix could not be serialized.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
void MatrixSerializer::serializeMatrix( Archive& archive, const DenseMatrix<MT,SO>& mat )
{
   if( IsRowMajorMatrix<MT>::value ) {
      for( size_t i=0UL; i<(~mat).rows(); ++i ) {
         for( size_t j=0UL; j<(~mat).columns(); ++j ) {
            archive << (~mat)(i,j);
         }
      }
   }
   else {
      for( size_t j=0UL; j<(~mat).columns(); ++j ) {
         for( size_t i=0UL; i<(~mat).rows(); ++i ) {
            archive << (~mat)(i,j);
         }
      }
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Dense matrix could not be serialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Serializes the elements of a sparse matrix.
//
// \param archive The archive to be written.
// \param mat The matrix to be serialized.
// \return void
// \exception std::runtime_error Sparse matrix could not be serialized.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
void MatrixSerializer::serializeMatrix( Archive& archive, const SparseMatrix<MT,SO>& mat )
{
   typedef ConstIterator_<MT>  ConstIterator;

   if( IsRowMajorMatrix<MT>::value ) {
      for( size_t i=0UL; i<(~mat).rows(); ++i ) {
         archive << uint64_t( (~mat).nonZeros( i ) );
         for( ConstIterator element=(~mat).begin(i); element!=(~mat).end(i); ++element ) {
            archive << element->index() << element->value();
         }
      }
   }
   else {
      for( size_t j=0UL; j<(~mat).columns(); ++j ) {
         archive << uint64_t( (~mat).nonZeros( j ) );
         for( ConstIterator element=(~mat).begin(j); element!=(~mat).end(j); ++element ) {
            archive << element->index() << element->value();
         }
      }
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Sparse matrix could not be serialized" );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  DESERIALIZATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Deserializes a matrix from the given archive.
//
// \param archive The archive to be read from.
// \param mat The matrix to be deserialized.
// \return void
// \exception std::runtime_error Error during deserialization.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
void MatrixSerializer::deserialize( Archive& archive, Matrix<MT,SO>& mat )
{
   if( !archive ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Faulty archive detected" );
   }

   deserializeHeader( archive, ~mat );
   prepareMatrix( ~mat );
   deserializeMatrix( archive, ~mat );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes all meta information about the given matrix.
//
// \param archive The archive to be read from.
// \param mat The matrix to be deserialized.
// \return void
// \exception std::runtime_error Error during deserialization.
*/
template< typename Archive  // Type of the archive
        , typename MT >     // Type of the matrix
void MatrixSerializer::deserializeHeader( Archive& archive, const MT& mat )
{
   typedef ElementType_<MT>  ET;

   if( !( archive >> version_ >> type_ >> elementType_ >> elementSize_ >> rows_ >> columns_ >> number_ ) ) {
      BLAZE_THROW_RUNTIME_ERROR( "Corrupt archive detected" );
   }
   else if( version_ != 1UL ) {
      BLAZE_THROW_RUNTIME_ERROR( "Invalid version detected" );
   }
   else if( ( type_ & 1U ) != 1U || ( type_ & (~7U) ) != 0U ) {
      BLAZE_THROW_RUNTIME_ERROR( "Invalid matrix type detected" );
   }
   else if( elementType_ != TypeValueMapping<ET>::value ) {
      BLAZE_THROW_RUNTIME_ERROR( "Invalid element type detected" );
   }
   else if( elementSize_ != sizeof( ET ) ) {
      BLAZE_THROW_RUNTIME_ERROR( "Invalid element size detected" );
   }
   else if( !IsResizable<MT>::value && ( rows_ != mat.rows() || columns_ != mat.columns() ) ) {
      BLAZE_THROW_RUNTIME_ERROR( "Invalid matrix size detected" );
   }
   else if( number_ > rows_*columns_ ) {
      BLAZE_THROW_RUNTIME_ERROR( "Invalid number of elements detected" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Prepares the given non-resizable dense matrix for the deserialization process.
//
// \param mat The dense matrix to be prepared.
// \return void
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order
DisableIf_< IsResizable<MT> > MatrixSerializer::prepareMatrix( DenseMatrix<MT,SO>& mat )
{
   reset( ~mat );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Prepares the given non-resizable sparse matrix for the deserialization process.
//
// \param mat The sparse matrix to be prepared.
// \return void
*/
template< typename MT  // Type of the sparse matrix
        , bool SO >    // Storage order
DisableIf_< IsResizable<MT> > MatrixSerializer::prepareMatrix( SparseMatrix<MT,SO>& mat )
{
   (~mat).reserve( number_ );
   reset( ~mat );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Prepares the given resizable matrix for the deserialization process.
//
// \param mat The matrix to be prepared.
// \return void
*/
template< typename MT >  // Type of the matrix
EnableIf_< IsResizable<MT> > MatrixSerializer::prepareMatrix( MT& mat )
{
   mat.resize ( rows_, columns_, false );
   mat.reserve( number_ );
   reset( mat );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The matrix to be reconstituted.
// \return void
// \exception std::runtime_error Error during deserialization.
//
// This function deserializes the contents of the matrix from the archive and reconstitutes the
// given matrix.
*/
template< typename Archive  // Type of the archive
        , typename MT >     // Type of the matrix
void MatrixSerializer::deserializeMatrix( Archive& archive, MT& mat )
{
   if( type_ == 1U ) {
      deserializeDenseRowMatrix( archive, ~mat );
   }
   else if( type_ == 5UL ) {
      deserializeDenseColumnMatrix( archive, ~mat );
   }
   else if( type_ == 3UL ) {
      deserializeSparseRowMatrix( archive, ~mat );
   }
   else if( type_ == 7UL ) {
      deserializeSparseColumnMatrix( archive, ~mat );
   }
   else {
      BLAZE_INTERNAL_ASSERT( false, "Undefined type flag" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a row-major dense matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The dense matrix to be reconstituted.
// \return void
// \exception std::runtime_error Dense matrix could not be deserialized.
//
// This function deserializes a row-major dense matrix from the archive and reconstitutes
// the given row-major dense matrix. In case any error is detected during the deserialization
// process, a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT >     // Type of the matrix
EnableIfTrue_< MT::simdEnabled >
   MatrixSerializer::deserializeDenseRowMatrix( Archive& archive, DenseMatrix<MT,rowMajor>& mat )
{
   if( columns_ == 0UL ) return;

   for( size_t i=0UL; i<rows_; ++i ) {
      archive.read( &(~mat)(i,0), columns_ );
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Dense matrix could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a row-major dense matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The dense matrix to be reconstituted.
// \return void
// \exception std::runtime_error Dense matrix could not be deserialized.
//
// This function deserializes a row-major dense matrix from the archive and reconstitutes
// the given dense matrix. In case any error is detected during the deserialization process,
// a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
void MatrixSerializer::deserializeDenseRowMatrix( Archive& archive, DenseMatrix<MT,SO>& mat )
{
   typedef ElementType_<MT>  ET;

   ET value = ET();

   for( size_t i=0UL; i<rows_; ++i ) {
      size_t j( 0UL );
      while( ( j != columns_ ) && ( archive >> value ) ) {
         (~mat)(i,j) = value;
         ++j;
      }
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Dense matrix could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a row-major dense matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The dense matrix to be reconstituted.
// \return void
// \exception std::runtime_error Sparse matrix could not be deserialized.
//
// This function deserializes a row-major dense matrix from the archive and reconstitutes
// the given sparse matrix. In case any error is detected during the deserialization
// process, a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
EnableIf_< IsNumeric< ElementType_<MT> > >
   MatrixSerializer::deserializeDenseRowMatrix( Archive& archive, SparseMatrix<MT,SO>& mat )
{
   DynamicMatrix< ElementType_<MT>, rowMajor > tmp( rows_, columns_ );
   deserializeDenseRowMatrix( archive, tmp );
   (~mat) = tmp;

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Sparse matrix could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a row-major dense matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The dense matrix to be reconstituted.
// \return void
// \exception std::runtime_error Sparse matrix could not be deserialized.
//
// This function deserializes a row-major dense matrix from the archive and reconstitutes
// the given sparse matrix. In case any error is detected during the deserialization
// process, a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
DisableIf_< IsNumeric< ElementType_<MT> > >
   MatrixSerializer::deserializeDenseRowMatrix( Archive& archive, SparseMatrix<MT,SO>& mat )
{
   typedef ElementType_<MT>  ET;

   ET value = ET();

   const size_t dim1( ( SO == rowMajor )?( rows_ ):( columns_ ) );
   const size_t dim2( ( SO != rowMajor )?( rows_ ):( columns_ ) );

   for( size_t i=0UL; i<dim1; ++i ) {
      (~mat).reserve( i, dim2 );
   }

   for( size_t i=0UL; i<rows_; ++i ) {
      size_t j( 0UL );
      while( ( j != columns_ ) && ( archive >> value ) ) {
         (~mat).append( i, j, value, false );
         ++j;
      }
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Sparse matrix could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a column-major dense matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The dense matrix to be reconstituted.
// \return void
// \exception std::runtime_error Dense matrix could not be deserialized.
//
// This function deserializes a column-major dense matrix from the archive and reconstitutes
// the given column-major dense matrix. In case any error is detected during the deserialization
// process, a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT >     // Type of the matrix
EnableIfTrue_< MT::simdEnabled >
   MatrixSerializer::deserializeDenseColumnMatrix( Archive& archive, DenseMatrix<MT,columnMajor>& mat )
{
   if( rows_ == 0UL ) return;

   for( size_t j=0UL; j<columns_; ++j ) {
      archive.read( &(~mat)(0,j), rows_ );
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Dense matrix could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a column-major dense matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The dense matrix to be reconstituted.
// \return void
// \exception std::runtime_error Dense matrix could not be deserialized.
//
// This function deserializes a column-major dense matrix from the archive and reconstitutes
// the given dense matrix. In case any error is detected during the deserialization process,
// a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
void MatrixSerializer::deserializeDenseColumnMatrix( Archive& archive, DenseMatrix<MT,SO>& mat )
{
   typedef ElementType_<MT>  ET;

   ET value = ET();

   for( size_t j=0UL; j<columns_; ++j ) {
      size_t i( 0UL );
      while( ( i != rows_ ) && ( archive >> value ) ) {
         (~mat)(i,j) = value;
         ++i;
      }
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Dense matrix could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a column-major dense matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The sparse matrix to be reconstituted.
// \return void
// \exception std::runtime_error Sparse matrix could not be deserialized.
//
// This function deserializes a column-major dense matrix from the archive and reconstitutes
// the given sparse matrix. In case any error is detected during the deserialization process,
// a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
EnableIf_< IsNumeric< ElementType_<MT> > >
   MatrixSerializer::deserializeDenseColumnMatrix( Archive& archive, SparseMatrix<MT,SO>& mat )
{
   DynamicMatrix< ElementType_<MT>, columnMajor > tmp( rows_, columns_ );
   deserializeDenseColumnMatrix( archive, tmp );
   (~mat) = tmp;

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Sparse matrix could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a column-major dense matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The sparse matrix to be reconstituted.
// \return void
// \exception std::runtime_error Sparse matrix could not be deserialized.
//
// This function deserializes a column-major dense matrix from the archive and reconstitutes
// the given sparse matrix. In case any error is detected during the deserialization process,
// a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
DisableIf_< IsNumeric< ElementType_<MT> > >
   MatrixSerializer::deserializeDenseColumnMatrix( Archive& archive, SparseMatrix<MT,SO>& mat )
{
   typedef ElementType_<MT>  ET;

   ET value = ET();

   const size_t dim1( ( SO == rowMajor )?( rows_ ):( columns_ ) );
   const size_t dim2( ( SO != rowMajor )?( rows_ ):( columns_ ) );

   for( size_t i=0UL; i<dim1; ++i ) {
      (~mat).reserve( i, dim2 );
   }

   for( size_t j=0UL; j<columns_; ++j ) {
      size_t i( 0UL );
      while( ( i != rows_ ) && ( archive >> value ) ) {
         (~mat).append( i, j, value, false );
         ++i;
      }
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Sparse matrix could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a row-major sparse matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The matrix to be reconstituted.
// \return void
// \exception std::runtime_error Dense matrix could not be deserialized.
//
// This function deserializes a row-major sparse matrix from the archive and reconstitutes
// the given dense matrix. In case any error is detected during the deserialization process,
// a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
void MatrixSerializer::deserializeSparseRowMatrix( Archive& archive, DenseMatrix<MT,SO>& mat )
{
   typedef ElementType_<MT>  ET;

   uint64_t number( 0UL );
   size_t   index ( 0UL );
   ET       value = ET();

   for( size_t i=0UL; i<rows_; ++i ) {
      archive >> number;
      size_t j( 0UL );
      while( ( j != number ) && ( archive >> index >> value ) ) {
         (~mat)(i,index) = value;
         ++j;
      }
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Dense matrix could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a row-major sparse matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The matrix to be reconstituted.
// \return void
// \exception std::runtime_error Sparse matrix could not be deserialized.
//
// This function deserializes a row-major sparse matrix from the archive and reconstitutes
// the given row-major sparse matrix. In case any error is detected during the deserialization
// process, a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT >     // Type of the matrix
void MatrixSerializer::deserializeSparseRowMatrix( Archive& archive, SparseMatrix<MT,rowMajor>& mat )
{
   typedef ElementType_<MT>  ET;

   uint64_t number( 0UL );
   size_t   index ( 0UL );
   ET       value = ET();

   for( size_t i=0UL; i<rows_; ++i )
   {
      archive >> number;

      size_t j( 0UL );
      while( ( j != number ) && ( archive >> index >> value ) ) {
         (~mat).append( i, index, value, false );
         ++j;
      }

      (~mat).finalize( i );
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Sparse matrix could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a row-major sparse matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The matrix to be reconstituted.
// \return void
// \exception std::runtime_error Sparse matrix could not be deserialized.
//
// This function deserializes a row-major sparse matrix from the archive and reconstitutes
// the given column-major sparse matrix. In case any error is detected during the deserialization
// process, a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT >     // Type of the matrix
void MatrixSerializer::deserializeSparseRowMatrix( Archive& archive, SparseMatrix<MT,columnMajor>& mat )
{
   CompressedMatrix< ElementType_<MT>, rowMajor > tmp( rows_, columns_, number_ );
   deserializeSparseRowMatrix( archive, tmp );
   (~mat) = tmp;

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Sparse matrix could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a column-major sparse matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The matrix to be reconstituted.
// \return void
// \exception std::runtime_error Dense matrix could not be deserialized.
//
// This function deserializes a column-major sparse matrix from the archive and reconstitutes
// the given dense matrix. In case any error is detected during the deserialization process,
// a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
void MatrixSerializer::deserializeSparseColumnMatrix( Archive& archive, DenseMatrix<MT,SO>& mat )
{
   typedef ElementType_<MT>  ET;

   uint64_t number( 0UL );
   size_t   index ( 0UL );
   ET       value = ET();

   for( size_t j=0UL; j<columns_; ++j ) {
      archive >> number;
      size_t i( 0UL );
      while( ( i != number ) && ( archive >> index >> value ) ) {
         (~mat)(index,j) = value;
         ++i;
      }
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Dense matrix could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a column-major sparse matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The matrix to be reconstituted.
// \return void
// \exception std::runtime_error Sparse matrix could not be deserialized.
//
// This function deserializes a column-major sparse matrix from the archive and reconstitutes
// the given row-major sparse matrix. In case any error is detected during the deserialization
// process, a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT >     // Type of the matrix
void MatrixSerializer::deserializeSparseColumnMatrix( Archive& archive, SparseMatrix<MT,rowMajor>& mat )
{
   CompressedMatrix< ElementType_<MT>, columnMajor > tmp( rows_, columns_, number_ );
   deserializeSparseColumnMatrix( archive, tmp );
   (~mat) = tmp;

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Sparse matrix could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a column-major sparse matrix from the archive.
//
// \param archive The archive to be read from.
// \param mat The matrix to be reconstituted.
// \return void
// \exception std::runtime_error Sparse matrix could not be deserialized.
//
// This function deserializes a column-major sparse matrix from the archive and reconstitutes
// the given column-major sparse matrix. In case any error is detected during the deserialization
// process, a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT >     // Type of the matrix
void MatrixSerializer::deserializeSparseColumnMatrix( Archive& archive, SparseMatrix<MT,columnMajor>& mat )
{
   typedef ElementType_<MT>  ET;

   uint64_t number( 0UL );
   size_t   index ( 0UL );
   ET       value = ET();

   for( size_t j=0UL; j<columns_; ++j )
   {
      archive >> number;

      size_t i( 0UL );
      while( ( i != number ) && ( archive >> index >> value ) ) {
         (~mat).append( index, j, value, false );
         ++i;
      }

      (~mat).finalize( j );
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Sparse matrix could not be deserialized" );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  MATRIXVALUEMAPPINGHELPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the MatrixValueMappingHelper class template for row-major dense matrices.
*/
template<>
struct MatrixSerializer::MatrixValueMappingHelper<true,true>
{
   enum { value = 1 };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the MatrixValueMappingHelper class template for column-major dense matrices.
*/
template<>
struct MatrixSerializer::MatrixValueMappingHelper<true,false>
{
   enum { value = 5 };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the MatrixValueMappingHelper class template for row-major sparse matrices.
*/
template<>
struct MatrixSerializer::MatrixValueMappingHelper<false,true>
{
   enum { value = 3 };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the MatrixValueMappingHelper class template for column-major sparse matrices.
*/
template<>
struct MatrixSerializer::MatrixValueMappingHelper<false,false>
{
   enum { value = 7 };
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Serializes the given matrix and writes it to the archive.
//
// \param archive The archive to be written.
// \param mat The matrix to be serialized.
// \return void
// \exception std::runtime_error Matrix could not be serialized.
//
// The serialize() function converts the given matrix into a portable, binary representation.
// The following example demonstrates the (de-)serialization process of matrices:

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

      // ... Resizing and initialization

      // Creating an archive that reads from the file "matrices.blaze"
      blaze::Archive<std::ofstream> archive( "matrices.blaze" );

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

// As the example demonstrates, the matrix serialization offers an enormous flexibility. However,
// several actions result in errors:
//
//  - matrices cannot be reconstituted as vectors (and vice versa)
//  - the element type of the serialized and reconstituted matrix must match, which means
//    that on the source and destination platform the general type (signed/unsigned integral
//    or floating point) and the size of the type must be exactly the same
//  - when reconstituting a StaticMatrix, the number of rows and columns must match those of
//    the serialized matrix
//
// In case an error is encountered during (de-)serialization, a \a std::runtime_exception is
// thrown.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
void serialize( Archive& archive, const Matrix<MT,SO>& mat )
{
   MatrixSerializer().serialize( archive, ~mat );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a matrix from the given archive.
//
// \param archive The archive to be read from.
// \param mat The matrix to be deserialized.
// \return void
// \exception std::runtime_error Matrix could not be deserialized.
//
// The deserialize() function converts the portable, binary representation contained in the
// given archive into the given matrix type. For a detailed example that demonstrates the
// (de-)serialization process of matrices, see the serialize() function.
*/
template< typename Archive  // Type of the archive
        , typename MT       // Type of the matrix
        , bool SO >         // Storage order
void deserialize( Archive& archive, Matrix<MT,SO>& mat )
{
   MatrixSerializer().deserialize( archive, ~mat );
}
//*************************************************************************************************

} // namespace blaze

#endif
