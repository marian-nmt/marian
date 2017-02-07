//=================================================================================================
/*!
//  \file blaze/math/traits/MathTrait.h
//  \brief Header file for the mathematical trait
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

#ifndef _BLAZE_MATH_TRAITS_MATHTRAIT_H_
#define _BLAZE_MATH_TRAITS_MATHTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cstddef>
#include <blaze/util/Complex.h>
#include <blaze/util/InvalidType.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/typetraits/Decay.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/typetraits/IsReference.h>
#include <blaze/util/typetraits/IsVolatile.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Base template for the MathTrait class.
// \ingroup math_traits
//
// \section mathtrait_general General
//
// The MathTrait class template determines the more significant, dominating data type and the
// less significant, submissive data type of the two given data types \a T1 and \a T2. The more
// significant data type is represented by the nested type \a HighType, the less significant
// data type by the nested type \a LowType. For instance, in case both \a T1 and \a T2 are
// built-in data types, \a HighType is set to the larger or signed data type and \a LowType
// is set to the smaller or unsigned data type. In case no dominating data type can be selected,
// \a Type is set to \a INVALID_TYPE. Note that \a const and \a volatile qualifiers and reference
// modifiers are generally ignored.
//
// Per default, the MathTrait template provides specializations for the following built-in data
// types:
//
// <ul>
//    <li>Integral types</li>
//    <ul>
//       <li>unsigned char, signed char, char, wchar_t</li>
//       <li>unsigned short, short</li>
//       <li>unsigned int, int</li>
//       <li>unsigned long, long</li>
//       <li>std::size_t, std::ptrdiff_t (for certain 64-bit compilers)</li>
//    </ul>
//    <li>Floating point types</li>
//    <ul>
//       <li>float</li>
//       <li>double</li>
//       <li>long double</li>
//    </ul>
// </ul>
//
// Additionally, the Blaze library provides specializations for the following user-defined
// arithmetic types, wherever a more/less significant data type can be selected:
//
// <ul>
//    <li>std::complex</li>
//    <li>blaze::StaticVector</li>
//    <li>blaze::HybridVector</li>
//    <li>blaze::DynamicVector</li>
//    <li>blaze::CompressedVector</li>
//    <li>blaze::StaticMatrix</li>
//    <li>blaze::HybridMatrix</li>
//    <li>blaze::DynamicMatrix</li>
//    <li>blaze::CompressedMatrix</li>
//    <li>blaze::SymmetricMatrix</li>
//    <li>blaze::LowerMatrix</li>
//    <li>blaze::UniLowerMatrix</li>
//    <li>blaze::StrictlyLowerMatrix</li>
//    <li>blaze::UpperMatrix</li>
//    <li>blaze::UniUpperMatrix</li>
//    <li>blaze::StrictlyUpperMatrix</li>
//    <li>blaze::DiagonalMatrix</li>
// </ul>
//
//
// \n \section mathtrait_specializations Creating custom specializations
//
// It is possible to specialize the MathTrait template for additional user-defined data types.
// The following example shows the according specialization for two dynamic column vectors:

   \code
   template< typename T1, typename T2 >
   struct MathTrait< DynamicVector<T1,false>, DynamicVector<T2,false> >
   {
      typedef DynamicVector< typename MathTrait<T1,T2>::Type, false >  Type;
   };
   \endcode
*/
template< typename T1, typename T2 >
struct MathTrait
{
 private:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   struct Failure {
      using HighType = INVALID_TYPE;
      using LowType  = INVALID_TYPE;
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Helper = MathTrait< Decay_<T1>, Decay_<T2> >;
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using HighType = typename If_< Or< IsConst<T1>, IsVolatile<T1>, IsReference<T1>
                                    , IsConst<T2>, IsVolatile<T2>, IsReference<T2> >
                                , Helper
                                , Failure >::HighType;

   using LowType = typename If_< Or< IsConst<T1>, IsVolatile<T1>, IsReference<T1>
                                   , IsConst<T2>, IsVolatile<T2>, IsReference<T2> >
                               , Helper
                               , Failure >::LowType;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  MATHTRAIT SPECIALIZATION FOR IDENTICAL TYPES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization for two identical types.
// \ingroup math_traits
//
// This specialization of the MathTrait class template handles the special case that the two
// given types are identical. In this case, the nested types \a HighType and \a LowType are
// set to the given type \a T (ignoring \a const and \a volatile qualifiers and reference
// modifiers).
*/
template< typename T >
struct MathTrait<T,T>
{
   //**********************************************************************************************
   using HighType = Decay_<T>;
   using LowType  = HighType;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MATHTRAIT SPECIALIZATION MACRO
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the creation of MathTrait specializations for the built-in data types.
// \ingroup math_traits
//
// This macro is used for the setup of the MathTrait specializations for the built-in data types.
*/
#define BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION(T1,T2,HIGH,LOW) \
   template<> \
   struct MathTrait< T1, T2 > \
   { \
      using HighType = HIGH; \
      using LowType  = LOW;  \
   }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the creation of MathTrait specializations for the complex data type.
// \ingroup math_traits
//
// This macro is used for the setup of the MathTrait specializations for the complex data type.
*/
#define BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( T1 ) \
   template< typename T2 > \
   struct MathTrait< T1, complex<T2> > \
   { \
      using HighType = complex<T2>; \
      using LowType  = T1;  \
   }; \
   template< typename T2 > \
   struct MathTrait< complex<T2>, T1 > \
   { \
      using HighType = complex<T2>; \
      using LowType  = T1;  \
   }
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  UNSIGNED CHAR SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , unsigned char , unsigned char , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , char          , char          , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , signed char   , signed char   , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , wchar_t       , wchar_t       , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , unsigned short, unsigned short, unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , short         , short         , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , unsigned int  , unsigned int  , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , int           , int           , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , unsigned long , unsigned long , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , long          , long          , unsigned char  );
#if defined(_WIN64)
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , std::size_t   , std::size_t   , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , std::ptrdiff_t, std::ptrdiff_t, unsigned char  );
#endif
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , float         , float         , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , double        , double        , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned char , long double   , long double   , unsigned char  );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CHAR SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , unsigned char , char          , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , char          , char          , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , signed char   , signed char   , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , wchar_t       , wchar_t       , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , unsigned short, unsigned short, char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , short         , short         , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , unsigned int  , unsigned int  , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , int           , int           , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , unsigned long , unsigned long , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , long          , long          , char           );
#if defined(_WIN64)
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , std::size_t   , std::size_t   , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , std::ptrdiff_t, std::ptrdiff_t, char           );
#endif
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , float         , float         , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , double        , double        , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( char          , long double   , long double   , char           );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SIGNED CHAR SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , unsigned char , signed char   , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , char          , signed char   , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , signed char   , signed char   , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , wchar_t       , wchar_t       , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , unsigned short, unsigned short, signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , short         , short         , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , unsigned int  , unsigned int  , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , int           , int           , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , unsigned long , unsigned long , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , long          , long          , signed char    );
#if defined(_WIN64)
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , std::size_t   , std::size_t   , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , std::ptrdiff_t, std::ptrdiff_t, signed char    );
#endif
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , float         , float         , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , double        , double        , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( signed char   , long double   , long double   , signed char    );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  WCHAR_T SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , unsigned char , wchar_t       , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , char          , wchar_t       , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , signed char   , wchar_t       , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , wchar_t       , wchar_t       , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , unsigned short, unsigned short, wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , short         , short         , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , unsigned int  , unsigned int  , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , int           , int           , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , unsigned long , unsigned long , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , long          , long          , wchar_t        );
#if defined(_WIN64)
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , std::size_t   , std::size_t   , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , std::ptrdiff_t, std::ptrdiff_t, wchar_t        );
#endif
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , float         , float         , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , double        , double        , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( wchar_t       , long double   , long double   , wchar_t        );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  UNSIGNED SHORT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, unsigned char , unsigned short, unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, char          , unsigned short, char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, signed char   , unsigned short, signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, wchar_t       , unsigned short, wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, unsigned short, unsigned short, unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, short         , short         , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, unsigned int  , unsigned int  , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, int           , int           , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, unsigned long , unsigned long , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, long          , long          , unsigned short );
#if defined(_WIN64)
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, std::size_t   , std::size_t   , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, std::ptrdiff_t, std::ptrdiff_t, unsigned short );
#endif
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, float         , float         , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, double        , double        , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned short, long double   , long double   , unsigned short );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SHORT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , unsigned char , short         , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , char          , short         , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , signed char   , short         , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , wchar_t       , short         , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , unsigned short, short         , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , short         , short         , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , unsigned int  , unsigned int  , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , int           , int           , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , unsigned long , unsigned long , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , long          , long          , short          );
#if defined(_WIN64)
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , std::size_t   , std::size_t   , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , std::ptrdiff_t, std::ptrdiff_t, short          );
#endif
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , float         , float         , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , double        , double        , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( short         , long double   , long double   , short          );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  UNSIGNED INT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , unsigned char , unsigned int  , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , char          , unsigned int  , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , signed char   , unsigned int  , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , wchar_t       , unsigned int  , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , unsigned short, unsigned int  , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , short         , unsigned int  , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , unsigned int  , unsigned int  , unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , int           , int           , unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , unsigned long , unsigned long , unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , long          , long          , unsigned int   );
#if defined(_WIN64)
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , std::size_t   , std::size_t   , unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , std::ptrdiff_t, std::ptrdiff_t, unsigned int   );
#endif
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , float         , float         , unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , double        , double        , unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned int  , long double   , long double   , unsigned int   );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  INT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , unsigned char , int           , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , char          , int           , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , signed char   , int           , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , wchar_t       , int           , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , unsigned short, int           , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , short         , int           , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , unsigned int  , int           , unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , int           , int           , int            );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , unsigned long , unsigned long , int            );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , long          , long          , int            );
#if defined(_WIN64)
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , std::size_t   , std::size_t   , int            );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , std::ptrdiff_t, std::ptrdiff_t, int            );
#endif
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , float         , float         , int            );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , double        , double        , int            );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( int           , long double   , long double   , int            );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  UNSIGNED LONG SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , unsigned char , unsigned long , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , char          , unsigned long , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , signed char   , unsigned long , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , wchar_t       , unsigned long , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , unsigned short, unsigned long , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , short         , unsigned long , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , unsigned int  , unsigned long , unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , int           , unsigned long , int            );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , unsigned long , unsigned long , unsigned long  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , long          , long          , unsigned long  );
#if defined(_WIN64)
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , std::size_t   , std::size_t   , unsigned long  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , std::ptrdiff_t, std::ptrdiff_t, unsigned long  );
#endif
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , float         , float         , unsigned long  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , double        , double        , unsigned long  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( unsigned long , long double   , long double   , unsigned long  );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LONG SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , unsigned char , long          , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , char          , long          , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , signed char   , long          , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , wchar_t       , long          , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , unsigned short, long          , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , short         , long          , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , unsigned int  , long          , unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , int           , long          , int            );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , unsigned long , long          , unsigned long  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , long          , long          , long           );
#if defined(_WIN64)
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , std::size_t   , std::size_t   , long           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , std::ptrdiff_t, std::ptrdiff_t, long           );
#endif
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , float         , float         , long           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , double        , double        , long           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long          , long double   , long double   , long           );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SIZE_T SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
#if defined(_WIN64)
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , unsigned char , std::size_t   , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , char          , std::size_t   , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , signed char   , std::size_t   , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , wchar_t       , std::size_t   , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , unsigned short, std::size_t   , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , short         , std::size_t   , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , unsigned int  , std::size_t   , unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , int           , std::size_t   , int            );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , unsigned long , std::size_t   , unsigned long  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , long          , std::size_t   , long           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , std::size_t   , std::size_t   , std::size_t    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , std::ptrdiff_t, std::ptrdiff_t, std::size_t    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , float         , float         , std::size_t    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , double        , double        , std::size_t    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::size_t   , long double   , long double   , std::size_t    );
/*! \endcond */
#endif
//*************************************************************************************************




//=================================================================================================
//
//  PTRDIFF_T SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
#if defined(_WIN64)
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, unsigned char , std::ptrdiff_t, unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, char          , std::ptrdiff_t, char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, signed char   , std::ptrdiff_t, signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, wchar_t       , std::ptrdiff_t, wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, unsigned short, std::ptrdiff_t, unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, short         , std::ptrdiff_t, short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, unsigned int  , std::ptrdiff_t, unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, int           , std::ptrdiff_t, int            );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, unsigned long , std::ptrdiff_t, unsigned long  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, long          , std::ptrdiff_t, long           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, std::size_t   , std::ptrdiff_t, std::size_t    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, float         , float         , std::ptrdiff_t );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, double        , double        , std::ptrdiff_t );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t, long double   , long double   , std::ptrdiff_t );
/*! \endcond */
#endif
//*************************************************************************************************




//=================================================================================================
//
//  FLOAT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , unsigned char , float         , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , char          , float         , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , signed char   , float         , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , wchar_t       , float         , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , unsigned short, float         , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , short         , float         , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , unsigned int  , float         , unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , int           , float         , int            );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , unsigned long , float         , unsigned long  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , long          , float         , long           );
#if defined(_WIN64)
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , std::size_t   , float         , std::size_t    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , std::ptrdiff_t, float         , std::ptrdiff_t );
#endif
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , float         , float         , float          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , double        , double        , float          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( float         , long double   , long double   , float          );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DOUBLE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , unsigned char , double        , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , char          , double        , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , signed char   , double        , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , wchar_t       , double        , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , unsigned short, double        , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , short         , double        , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , unsigned int  , double        , unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , int           , double        , int            );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , unsigned long , double        , unsigned long  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , long          , double        , long           );
#if defined(_WIN64)
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , std::size_t   , double        , std::size_t    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , std::ptrdiff_t, double        , std::ptrdiff_t );
#endif
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , float         , double        , float          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , double        , double        , double         );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( double        , long double   , long double   , double         );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LONG DOUBLE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//                                             Type 1          Type 2          High type       Low type
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , unsigned char , long double   , unsigned char  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , char          , long double   , char           );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , signed char   , long double   , signed char    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , wchar_t       , long double   , wchar_t        );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , unsigned short, long double   , unsigned short );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , short         , long double   , short          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , unsigned int  , long double   , unsigned int   );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , int           , long double   , int            );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , unsigned long , long double   , unsigned long  );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , long          , long double   , long           );
#if defined(_WIN64)
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , std::size_t   , long double   , std::size_t    );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , std::ptrdiff_t, long double   , std::ptrdiff_t );
#endif
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , float         , long double   , float          );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , double        , long double   , double         );
BLAZE_CREATE_BUILTIN_MATHTRAIT_SPECIALIZATION( long double   , long double   , long double   , long double    );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COMPLEX SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( unsigned char  );
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( char           );
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( signed char    );
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( wchar_t        );
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( unsigned short );
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( short          );
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( unsigned int   );
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( int            );
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( unsigned long  );
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( long           );
#if defined(_WIN64)
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( std::size_t    );
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( std::ptrdiff_t );
#endif
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( float          );
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( double         );
BLAZE_CREATE_COMPLEX_MATHTRAIT_SPECIALIZATION( long double    );
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2 >
struct MathTrait< complex<T1>, complex<T2> >
{
   using HighType = complex<typename MathTrait<T1,T2>::HighType>;
   using LowType  = complex<typename MathTrait<T1,T2>::LowType>;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
