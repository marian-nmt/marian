//=================================================================================================
/*!
//  \file blaze/math/simd/SIMDTrait.h
//  \brief Header file for the SIMD trait
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

#ifndef _BLAZE_MATH_SIMD_SIMDTRAIT_H_
#define _BLAZE_MATH_SIMD_SIMDTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/simd/BasicTypes.h>
#include <blaze/system/Vectorization.h>
#include <blaze/util/Complex.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/HasSize.h>
#include <blaze/util/typetraits/IsIntegral.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsSigned.h>
#include <blaze/util/typetraits/RemoveCV.h>


namespace blaze {

//=================================================================================================
//
//  CLASS SIMDTRAITBASE
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Base template for the SIMDTraitBase class.
// \ingroup simd
*/
template< typename T
        , typename = void >
struct SIMDTraitBase
{
   using Type = T;
   enum : size_t { size = 1UL };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SIMDTraitBase class template for 1-byte integral data types.
// \ingroup simd
*/
template< typename T >
struct SIMDTraitBase< T, EnableIf_< And< IsNumeric<T>, IsIntegral<T>, Has1Byte<T> > > >
{
   using Type = If_< IsSigned<T>, SIMDint8, SIMDuint8 >;
   enum : size_t { size = Type::size };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SIMDTraitBase class template for 1-byte integral complex data types.
// \ingroup simd
*/
template< typename T >
struct SIMDTraitBase< complex<T>, EnableIf_< And< IsNumeric<T>, IsIntegral<T>, Has1Byte<T> > > >
{
   using Type = If_< IsSigned<T>, SIMDcint8, SIMDcuint8 >;
   enum : size_t { size = Type::size };

   BLAZE_STATIC_ASSERT( sizeof( complex<T> ) == 2UL*sizeof( T ) );
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SIMDTraitBase class template for 2-byte integral data types.
// \ingroup simd
*/
template< typename T >
struct SIMDTraitBase< T, EnableIf_< And< IsNumeric<T>, IsIntegral<T>, Has2Bytes<T> > > >
{
   using Type = If_< IsSigned<T>, SIMDint16, SIMDuint16 >;
   enum : size_t { size = Type::size };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SIMDTraitBase class template for 2-byte integral complex data types.
// \ingroup simd
*/
template< typename T >
struct SIMDTraitBase< complex<T>, EnableIf_< And< IsNumeric<T>, IsIntegral<T>, Has2Bytes<T> > > >
{
   using Type = If_< IsSigned<T>, SIMDcint16, SIMDcuint16 >;
   enum : size_t { size = Type::size };

   BLAZE_STATIC_ASSERT( sizeof( complex<T> ) == 2UL*sizeof( T ) );
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SIMDTraitBase class template for 4-byte integral data types.
// \ingroup simd
*/
template< typename T >
struct SIMDTraitBase< T, EnableIf_< And< IsNumeric<T>, IsIntegral<T>, Has4Bytes<T> > > >
{
   using Type = If_< IsSigned<T>, SIMDint32, SIMDuint32 >;
   enum : size_t { size = Type::size };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SIMDTraitBase class template for 4-byte integral complex data types.
// \ingroup simd
*/
template< typename T >
struct SIMDTraitBase< complex<T>, EnableIf_< And< IsNumeric<T>, IsIntegral<T>, Has4Bytes<T> > > >
{
   using Type = If_< IsSigned<T>, SIMDcint32, SIMDcuint32 >;
   enum : size_t { size = Type::size };

   BLAZE_STATIC_ASSERT( sizeof( complex<T> ) == 2UL*sizeof( T ) );
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SIMDTraitBase class template for 8-byte integral data types.
// \ingroup simd
*/
template< typename T >
struct SIMDTraitBase< T, EnableIf_< And< IsNumeric<T>, IsIntegral<T>, Has8Bytes<T> > > >
{
   using Type = If_< IsSigned<T>, SIMDint64, SIMDuint64 >;
   enum : size_t { size = Type::size };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SIMDTraitBase class template for 8-byte integral complex data types.
// \ingroup simd
*/
template< typename T >
struct SIMDTraitBase< complex<T>, EnableIf_< And< IsNumeric<T>, IsIntegral<T>, Has8Bytes<T> > > >
{
   using Type = If_< IsSigned<T>, SIMDcint64, SIMDcuint64 >;
   enum : size_t { size = Type::size };

   BLAZE_STATIC_ASSERT( sizeof( complex<T> ) == 2UL*sizeof( T ) );
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SIMDTraitBase class template for 'float'.
// \ingroup simd
*/
template<>
struct SIMDTraitBase<float>
{
   using Type = SIMDfloat;
   enum : size_t { size = Type::size };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SIMDTraitBase class template for 'complex<float>'.
// \ingroup simd
*/
template<>
struct SIMDTraitBase< complex<float> >
{
   using Type = SIMDcfloat;
   enum : size_t { size = Type::size };

   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SIMDTraitBase class template for 'double'.
// \ingroup simd
*/
template<>
struct SIMDTraitBase<double>
{
   using Type = SIMDdouble;
   enum : size_t { size = Type::size };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the SIMDTraitBase class template for 'complex<double>'.
// \ingroup simd
*/
template<>
struct SIMDTraitBase< complex<double> >
{
   using Type = SIMDcdouble;
   enum : size_t { size = Type::size };

   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CLASS SIMDTRAIT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief SIMD characteristics of data types.
// \ingroup simd
//
// The SIMDTrait class template provides the SIMD characteristics of a specific data type:
//
//  - The nested data type \a Type corresponds to the according packed, SIMD data type. In case
//    the data type doesn't have a SIMD representation, \a Type corresonds to the given data
//    type itself.
//  - The \a size member constant corresponds to the number of values of the given data type that
//    are packed together in one SIMD vector type. In case the data type cannot be vectorized,
//    \a size is set to 1.
*/
template< typename T >
class SIMDTrait : public SIMDTraitBase< RemoveCV_<T> >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the SIMDTrait class template.
// \ingroup simd
//
// The SIMDTrait_ alias declaration provides a convenient shortcut to access the nested \a Type
// of the SIMDTrait class template. For instance, given the type \a T the following two type
// definitions are identical:

   \code
   using Type1 = typename SIMDTrait<T>::Type;
   using Type2 = SIMDTrait_<T>;
   \endcode
*/
template< typename T >
using SIMDTrait_ = typename SIMDTrait<T>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
