//=================================================================================================
/*!
//  \file blaze/math/serialization/TypeValueMapping.h
//  \brief Header file for the TypeValueMapping class template
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

#ifndef _BLAZE_MATH_SERIALIZATION_TypeValueMapping_H_
#define _BLAZE_MATH_SERIALIZATION_TypeValueMapping_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/typetraits/IsComplex.h>
#include <blaze/util/typetraits/IsFloatingPoint.h>
#include <blaze/util/typetraits/IsIntegral.h>
#include <blaze/util/typetraits/IsSigned.h>
#include <blaze/util/typetraits/IsUnsigned.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the TypeValueMapping class template.
// \ingroup math_serialization
*/
template< bool IsSignedIntegral, bool IsUnsignedIntegral, bool IsFloatingPoint, bool IsComplex >
struct TypeValueMappingHelper;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the TypeValueMappingHelper for compound data types.
// \ingroup math_serialization
*/
template<>
struct TypeValueMappingHelper<false,false,false,false>
{
 public:
   //**********************************************************************************************
   enum { value = 0 };
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the TypeValueMappingHelper for signed integral data types.
// \ingroup math_serialization
*/
template<>
struct TypeValueMappingHelper<true,false,false,false>
{
 public:
   //**********************************************************************************************
   enum { value = 1 };
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the TypeValueMappingHelper for unsigned integral data types.
// \ingroup math_serialization
*/
template<>
struct TypeValueMappingHelper<false,true,false,false>
{
 public:
   //**********************************************************************************************
   enum { value = 2 };
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the TypeValueMappingHelper for floating-point data types.
// \ingroup math_serialization
*/
template<>
struct TypeValueMappingHelper<false,false,true,false>
{
 public:
   //**********************************************************************************************
   enum { value = 3 };
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the TypeValueMappingHelper for complex data types.
// \ingroup math_serialization
*/
template<>
struct TypeValueMappingHelper<false,false,false,true>
{
 public:
   //**********************************************************************************************
   enum { value = 4 };
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion from a data type to a serial representation.
// \ingroup math_serialization
//
// This class template converts the given data type into an integral representation suited for
// serialization. Depending on the given data type, the \a value member enumeration is set to
// the according serial representation.
*/
template< typename T >
struct TypeValueMapping
{
 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   enum { value = TypeValueMappingHelper< IsIntegral<T>::value && IsSigned<T>::value
                                        , IsIntegral<T>::value && IsUnsigned<T>::value
                                        , IsFloatingPoint<T>::value
                                        , IsComplex<T>::value
                                        >::value };
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blaze

#endif
