//=================================================================================================
/*!
//  \file blaze/math/serialization/VectorSerializer.h
//  \brief Serialization of dense and sparse vectors
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

#ifndef _BLAZE_MATH_SERIALIZATION_VECTORSERIALIZER_H_
#define _BLAZE_MATH_SERIALIZATION_VECTORSERIALIZER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Vector.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/SparseVector.h>
#include <blaze/math/expressions/Vector.h>
#include <blaze/math/serialization/TypeValueMapping.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsResizable.h>
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
/*!\brief Serializer for dense and sparse vectors.
// \ingroup math_serialization
//
// The VectorSerializer implements the necessary logic to serialize dense and sparse vectors, i.e.
// to convert them into a portable, binary representation. The following example demonstrates the
// (de-)serialization process of vectors:

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
      blaze::Archive<std::ofstream> archive( "vectors.blaze" );

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

// Note that it is even possible to (de-)serialize vectors with vector or matrix elements:

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
      blaze::Archive<std::ofstream> archive( "vector.blaze" );

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
//  - when reconstituting a StaticVector, its size must match the size of the serialized vector
//
// In case an error is encountered during (de-)serialization, a \a std::runtime_exception is
// thrown.
*/
class VectorSerializer
{
 private:
   //**Private class VectorValueMappingHelper******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Auxiliary helper class for the VectorValueMapping class template.
   //
   // The VectorValueMapping class template is an auxiliary class for the VectorSerializer. It
   // maps a vector type into an integral representation. For the mapping, the following bit
   // mapping is used:

      \code
      0x01 - Vector/Matrix flag
      0x02 - Dense/Sparse flag
      0x04 - Row-/Column-major flag
      \endcode
   */
   template< bool IsDenseVector >
   struct VectorValueMappingHelper;
   /*! \endcond */
   //**********************************************************************************************

   //**Private class VectorValueMapping************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Serialization of the type of a vector.
   //
   // This class template converts the given vector type into an integral representation suited
   // for serialization. Depending on the given vector type, the \a value member enumeration is
   // set to the according integral representation.
   */
   template< typename T >
   struct VectorValueMapping
   {
      enum { value = VectorValueMappingHelper< IsDenseVector<T>::value >::value };
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_TYPE( T );
   };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Constructor*********************************************************************************
   /*!\name Constructor */
   //@{
   explicit inline VectorSerializer();
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
   template< typename Archive, typename VT, bool TF >
   void serialize( Archive& archive, const Vector<VT,TF>& vec );
   //@}
   //**********************************************************************************************

   //**Deserialization functions*********************************************************************
   /*!\name Deserialization functions */
   //@{
   template< typename Archive, typename VT, bool TF >
   void deserialize( Archive& archive, Vector<VT,TF>& vec );
   //@}
   //**********************************************************************************************

 private:
   //**Serialization functions*********************************************************************
   /*!\name Serialization functions */
   //@{
   template< typename Archive, typename VT >
   void serializeHeader( Archive& archive, const VT& vec );

   template< typename Archive, typename VT, bool TF >
   void serializeVector( Archive& archive, const DenseVector<VT,TF>& vec );

   template< typename Archive, typename VT, bool TF >
   void serializeVector( Archive& archive, const SparseVector<VT,TF>& vec );
   //@}
   //**********************************************************************************************

   //**Deserialization functions*******************************************************************
   /*!\name Deserialization functions */
   //@{
   template< typename Archive, typename VT >
   void deserializeHeader( Archive& archive, const VT& vec );

   template< typename VT, bool TF >
   DisableIf_< IsResizable<VT> > prepareVector( DenseVector<VT,TF>& vec );

   template< typename VT, bool TF >
   DisableIf_< IsResizable<VT> > prepareVector( SparseVector<VT,TF>& vec );

   template< typename VT >
   EnableIf_< IsResizable<VT> > prepareVector( VT& vec );

   template< typename Archive, typename VT >
   void deserializeVector( Archive& archive, VT& vec );

   template< typename Archive, typename VT, bool TF >
   typename DisableIfTrue< VT::simdEnabled >::Type
      deserializeDenseVector( Archive& archive, DenseVector<VT,TF>& vec );

   template< typename Archive, typename VT, bool TF >
   EnableIfTrue_< VT::simdEnabled >
      deserializeDenseVector( Archive& archive, DenseVector<VT,TF>& vec );

   template< typename Archive, typename VT, bool TF >
   void deserializeDenseVector( Archive& archive, SparseVector<VT,TF>& vec );

   template< typename Archive, typename VT, bool TF >
   void deserializeSparseVector( Archive& archive, DenseVector<VT,TF>& vec );

   template< typename Archive, typename VT, bool TF >
   void deserializeSparseVector( Archive& archive, SparseVector<VT,TF>& vec );
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   uint8_t  version_;      //!< The version of the archive.
   uint8_t  type_;         //!< The type of the vector.
   uint8_t  elementType_;  //!< The type of an element.
   uint8_t  elementSize_;  //!< The size in bytes of a single element of the vector.
   uint64_t size_;         //!< The size of the vector.
   uint64_t number_;       //!< The total number of elements contained in the vector.
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
/*!\brief The default constructor of the VectorSerializer class.
*/
VectorSerializer::VectorSerializer()
   : version_    ( 0U  )  // The version of the archive
   , type_       ( 0U  )  // The type of the vector
   , elementType_( 0U  )  // The type of an element
   , elementSize_( 0U  )  // The size in bytes of a single element of the vector
   , size_       ( 0UL )  // The size of the vector
   , number_     ( 0UL )  // The total number of elements contained in the vector
{}
//*************************************************************************************************




//=================================================================================================
//
//  SERIALIZATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Serializes the given vector and writes it to the archive.
//
// \param archive The archive to be written.
// \param vec The vector to be serialized.
// \return void
// \exception std::runtime_error Error during serialization.
//
// This function serializes the given vector and writes it to the given archive. In case any
// error is detected during the serialization, a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename VT       // Type of the vector
        , bool TF >         // Transpose flag
void VectorSerializer::serialize( Archive& archive, const Vector<VT,TF>& vec )
{
   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Faulty archive detected" );
   }

   serializeHeader( archive, ~vec );
   serializeVector( archive, ~vec );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Serializes all meta information about the given vector.
//
// \param archive The archive to be written.
// \param vec The vector to be serialized.
// \return void
// \exception std::runtime_error File header could not be serialized.
*/
template< typename Archive  // Type of the archive
        , typename VT >     // Type of the vector
void VectorSerializer::serializeHeader( Archive& archive, const VT& vec )
{
   typedef ElementType_<VT>  ET;

   archive << uint8_t ( 1U );
   archive << uint8_t ( VectorValueMapping<VT>::value );
   archive << uint8_t ( TypeValueMapping<ET>::value );
   archive << uint8_t ( sizeof( ET ) );
   archive << uint64_t( vec.size() );
   archive << uint64_t( IsDenseVector<VT>::value ? vec.size() : vec.nonZeros() );

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "File header could not be serialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Serializes the elements of a dense vector.
//
// \param archive The archive to be written.
// \param vec The vector to be serialized.
// \return void
// \exception std::runtime_error Dense vector could not be serialized.
*/
template< typename Archive  // Type of the archive
        , typename VT       // Type of the vector
        , bool TF >         // Transpose flag
void VectorSerializer::serializeVector( Archive& archive, const DenseVector<VT,TF>& vec )
{
   size_t i( 0UL );
   while( ( i < (~vec).size() ) && ( archive << (~vec)[i] ) ) {
      ++i;
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Dense vector could not be serialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Serializes the elements of a sparse vector.
//
// \param archive The archive to be written.
// \param vec The vector to be serialized.
// \return void
// \exception std::runtime_error Sparse vector could not be serialized.
*/
template< typename Archive  // Type of the archive
        , typename VT       // Type of the vector
        , bool TF >         // Transpose flag
void VectorSerializer::serializeVector( Archive& archive, const SparseVector<VT,TF>& vec )
{
   typedef ConstIterator_<VT>  ConstIterator;

   ConstIterator element( (~vec).begin() );
   while( ( element != (~vec).end() ) &&
          ( archive << element->index() << element->value() ) ) {
      ++element;
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Sparse vector could not be serialized" );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  DESERIALIZATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Deserializes a vector from the given archive.
//
// \param archive The archive to be read from.
// \param vec The vector to be deserialized.
// \return void
// \exception std::runtime_error Error during deserialization.
*/
template< typename Archive  // Type of the archive
        , typename VT       // Type of the vector
        , bool TF >         // Transpose flag
void VectorSerializer::deserialize( Archive& archive, Vector<VT,TF>& vec )
{
   if( !archive ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Faulty archive detected" );
   }

   deserializeHeader( archive, ~vec );
   prepareVector( ~vec );
   deserializeVector( archive, ~vec );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes all meta information about the given vector.
//
// \param archive The archive to be read from.
// \param vec The vector to be deserialized.
// \return void
// \exception std::runtime_error Error during deserialization.
//
// This function deserializes all meta information about the given vector contained in the
// header of the given archive. In case any error is detected during the deserialization
// process (for instance an invalid type of vector, element type, element size, or vector
// size) a \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename VT >     // Type of the vector
void VectorSerializer::deserializeHeader( Archive& archive, const VT& vec )
{
   typedef ElementType_<VT>  ET;

   if( !( archive >> version_ >> type_ >> elementType_ >> elementSize_ >> size_ >> number_ ) ) {
      BLAZE_THROW_RUNTIME_ERROR( "Corrupt archive detected" );
   }
   else if( version_ != 1UL ) {
      BLAZE_THROW_RUNTIME_ERROR( "Invalid version detected" );
   }
   else if( ( type_ & 1U ) != 0U || ( type_ & (~3U) ) != 0U ) {
      BLAZE_THROW_RUNTIME_ERROR( "Invalid vector type detected" );
   }
   else if( elementType_ != TypeValueMapping<ET>::value ) {
      BLAZE_THROW_RUNTIME_ERROR( "Invalid element type detected" );
   }
   else if( elementSize_ != sizeof( ET ) ) {
      BLAZE_THROW_RUNTIME_ERROR( "Invalid element size detected" );
   }
   else if( !IsResizable<VT>::value && size_ != vec.size() ) {
      BLAZE_THROW_RUNTIME_ERROR( "Invalid vector size detected" );
   }
   else if( number_ > size_ ) {
      BLAZE_THROW_RUNTIME_ERROR( "Invalid number of elements detected" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Prepares the given non-resizable dense vector for the deserialization process.
//
// \param vec The dense vector to be prepared.
// \return void
*/
template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
DisableIf_< IsResizable<VT> > VectorSerializer::prepareVector( DenseVector<VT,TF>& vec )
{
   reset( ~vec );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Prepares the given non-resizable sparse vector for the deserialization process.
//
// \param vec The sparse vector to be prepared.
// \return void
*/
template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
DisableIf_< IsResizable<VT> > VectorSerializer::prepareVector( SparseVector<VT,TF>& vec )
{
   (~vec).reserve( number_ );
   reset( ~vec );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Prepares the given resizable vector for the deserialization process.
//
// \param vec The vector to be prepared.
// \return void
*/
template< typename VT >  // Type of the vector
EnableIf_< IsResizable<VT> > VectorSerializer::prepareVector( VT& vec )
{
   vec.resize ( size_, false );
   vec.reserve( number_ );
   reset( vec );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a vector from the archive.
//
// \param archive The archive to be read from.
// \param vec The vector to be reconstituted.
// \return void
// \exception std::runtime_error Error during deserialization.
//
// This function deserializes the contents of the vector from the archive and reconstitutes the
// given vector.
*/
template< typename Archive  // Type of the archive
        , typename VT >     // Type of the vector
void VectorSerializer::deserializeVector( Archive& archive, VT& vec )
{
   if( type_ == 0U ) {
      deserializeDenseVector( archive, vec );
   }
   else if( type_ == 2U ) {
      deserializeSparseVector( archive, vec );
   }
   else {
      BLAZE_INTERNAL_ASSERT( false, "Undefined type flag" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a dense vector from the archive.
//
// \param archive The archive to be read from.
// \param vec The dense vector to be reconstituted.
// \return void
// \exception std::runtime_error Dense vector could not be deserialized.
//
// This function deserializes a dense vector from the archive and reconstitutes the given
// dense vector. In case any error is detected during the deserialization process, a
// \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename VT       // Type of the vector
        , bool TF >         // Transpose flag
typename DisableIfTrue< VT::simdEnabled >::Type
   VectorSerializer::deserializeDenseVector( Archive& archive, DenseVector<VT,TF>& vec )
{
   typedef ElementType_<VT>  ET;

   size_t i( 0UL );
   ET value = ET();

   while( ( i != size_ ) && ( archive >> value ) ) {
      (~vec)[i] = value;
      ++i;
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Dense vector could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a dense vector from the archive.
//
// \param archive The archive to be read from.
// \param vec The dense vector to be reconstituted.
// \return void
// \exception std::runtime_error Dense vector could not be deserialized.
//
// This function deserializes a dense vector from the archive and reconstitutes the given
// dense vector. In case any error is detected during the deserialization process, a
// \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename VT       // Type of the vector
        , bool TF >         // Transpose flag
EnableIfTrue_< VT::simdEnabled >
   VectorSerializer::deserializeDenseVector( Archive& archive, DenseVector<VT,TF>& vec )
{
   if( size_ == 0UL ) return;
   archive.read( &(~vec)[0], size_ );

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Dense vector could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a dense vector from the archive.
//
// \param archive The archive to be read from.
// \param vec The sparse vector to be reconstituted.
// \return void
// \exception std::runtime_error Sparse vector could not be deserialized.
//
// This function deserializes a dense vector from the archive and reconstitutes the given
// sparse vector. In case any error is detected during the deserialization process, a
// \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename VT       // Type of the vector
        , bool TF >         // Transpose flag
void VectorSerializer::deserializeDenseVector( Archive& archive, SparseVector<VT,TF>& vec )
{
   typedef ElementType_<VT>  ET;

   size_t i( 0UL );
   ET value = ET();

   while( ( i != size_ ) && ( archive >> value ) ) {
      (~vec)[i] = value;
      ++i;
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Sparse vector could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a sparse vector from the archive.
//
// \param archive The archive to be read from.
// \param vec The dense vector to be reconstituted.
// \return void
// \exception std::runtime_error Dense vector could not be deserialized.
//
// This function deserializes a sparse vector from the archive and reconstitutes the given
// dense vector. In case any error is detected during the deserialization process, a
// \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename VT       // Type of the vector
        , bool TF >         // Transpose flag
void VectorSerializer::deserializeSparseVector( Archive& archive, DenseVector<VT,TF>& vec )
{
   typedef ElementType_<VT>  ET;

   size_t i( 0UL );
   size_t index( 0UL );
   ET     value = ET();

   while( ( i != number_ ) && ( archive >> index >> value ) ) {
      (~vec)[index] = value;
      ++i;
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Dense vector could not be deserialized" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a sparse vector from the archive.
//
// \param archive The archive to be read from.
// \param vec The sparse vector to be reconstituted.
// \return void
// \exception std::runtime_error Sparse vector could not be deserialized.
//
// This function deserializes a sparse vector from the archive and reconstitutes the given
// sparse vector. In case any error is detected during the deserialization process, a
// \a std::runtime_error is thrown.
*/
template< typename Archive  // Type of the archive
        , typename VT       // Type of the vector
        , bool TF >         // Transpose flag
void VectorSerializer::deserializeSparseVector( Archive& archive, SparseVector<VT,TF>& vec )
{
   typedef ElementType_<VT>  ET;

   size_t i( 0UL );
   size_t index( 0UL );
   ET     value = ET();

   while( ( i != number_ ) && ( archive >> index >> value ) ) {
      (~vec).append( index, value, false );
      ++i;
   }

   if( !archive ) {
      BLAZE_THROW_RUNTIME_ERROR( "Sparse vector could not be deserialized" );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  VECTORVALUEMAPPINGHELPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the VectorValueMappingHelper class template for dense vectors.
*/
template<>
struct VectorSerializer::VectorValueMappingHelper<true>
{
   enum { value = 0 };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the VectorValueMappingHelper class template for sparse vectors.
*/
template<>
struct VectorSerializer::VectorValueMappingHelper<false>
{
   enum { value = 2 };
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Serializes the given vector and writes it to the archive.
//
// \param archive The archive to be written.
// \param vec The vector to be serialized.
// \return void
// \exception std::runtime_error Error during serialization.
//
// The serialize() function converts the given vector into a portable, binary representation.
// The following example demonstrates the (de-)serialization process of vectors:

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

      // ... Resizing and initialization

      // Creating an archive that reads from the file "vectors.blaze"
      blaze::Archive<std::ofstream> archive( "vectors.blaze" );

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

// As the example demonstrates, the vector serialization offers an enormous flexibility. However,
// several actions result in errors:
//
//  - vectors cannot be reconstituted as matrices (and vice versa)
//  - the element type of the serialized and reconstituted vector must match, which means
//    that on the source and destination platform the general type (signed/unsigned integral
//    or floating point) and the size of the type must be exactly the same
//  - when reconstituting a StaticVector, its size must match the size of the serialized vector
//
// In case an error is encountered during (de-)serialization, a \a std::runtime_exception is
// thrown.
*/
template< typename Archive  // Type of the archive
        , typename VT       // Type of the vector
        , bool TF >         // Transpose flag
void serialize( Archive& archive, const Vector<VT,TF>& vec )
{
   VectorSerializer().serialize( archive, ~vec );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deserializes a vector from the given archive.
//
// \param archive The archive to be read from.
// \param vec The vector to be deserialized.
// \return void
// \exception std::runtime_error Vector could not be deserialized.
//
// The deserialize() function converts the portable, binary representation contained in
// the given archive into the given vector type. For a detailed example that demonstrates
// the (de-)serialization process of vectors, see the serialize() function.
*/
template< typename Archive  // Type of the archive
        , typename VT       // Type of the vector
        , bool TF >         // Transpose flag
void deserialize( Archive& archive, Vector<VT,TF>& vec )
{
   VectorSerializer().deserialize( archive, ~vec );
}
//*************************************************************************************************

} // namespace blaze

#endif
