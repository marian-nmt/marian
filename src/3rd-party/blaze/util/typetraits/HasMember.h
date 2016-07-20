//=================================================================================================
/*!
//  \file blaze/util/typetraits/HasMember.h
//  \brief Header file for the HasMember type traits
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

#ifndef _BLAZE_UTIL_TYPETRAITS_HASMEMBER_H_
#define _BLAZE_UTIL_TYPETRAITS_HASMEMBER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/IntegralConstant.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/typetraits/IsBuiltin.h>




//=================================================================================================
//
//  MACRO DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Macro for the creation of a type trait for compile time checks for member data and functions.
// \ingroup math_type_traits
//
// This macro creates the definition of a type trait \a TYPE_TRAIT_NAME that can determine whether
// the specified element \a MEMBER_NAME is a data or function member of a given type. The following
// example demonstrates the use of the macro and the resulting type trait:

   \code
   class MyType {
    public:
      void publicCompute();

    private:
      void privateCompute();

      int value_;
   };

   BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( HasPublicCompute , publicCompute  );
   BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( HasPrivateCompute, privateCompute );
   BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( HasValue         , value_         );

   BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( HasEvaluate , evalute   );
   BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( HasDetermine, determine );
   BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( HasData     , data_     );

   HasPublicCompute<MyType>::value  // Evaluates to 'true'
   HasPrivateCompute<MyType>::Type  // Results in TrueType
   HasValue<MyType>                 // Is derived from TrueType
   HasEvaluate<MyType>::value       // Evaluates to 'false'
   HasDetermine<MyType>::Type       // Results in FalseType
   HasData<MyType>                  // Is derived from FalseType
   \endcode

// The macro results in the definition of a new class with the specified name \a TYPE_TRAIT_NAME
// within the current namespace. This may cause name collisions with any other entity called
// \a TYPE_TRAIT_NAME in the same namespace. Therefore it is advisable to create the type trait
// as locally as possible to minimize the probability of name collisions. Note however that the
// macro cannot be used within function scope since a template declaration cannot appear at
// block scope.
*/
#define BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( TYPE_TRAIT_NAME, MEMBER_NAME )  \
                                                                                             \
template < typename TYPE1230 >                                                               \
class TYPE_TRAIT_NAME##HELPER                                                                \
{                                                                                            \
 private:                                                                                    \
   using Yes = char[1];                                                                      \
   using No  = char[2];                                                                      \
                                                                                             \
   struct Base {};                                                                           \
                                                                                             \
   template< typename U, U > struct Check;                                                   \
                                                                                             \
   struct Fallback { int MEMBER_NAME; };                                                     \
                                                                                             \
   struct Derived                                                                            \
      : blaze::If< blaze::IsBuiltin<TYPE1230>, Base, TYPE1230 >::Type                        \
      , Fallback                                                                             \
   {};                                                                                       \
                                                                                             \
   template < typename U >                                                                   \
   static No& test( Check<int Fallback::*, &U::MEMBER_NAME>* );                              \
                                                                                             \
   template < typename U >                                                                   \
   static Yes& test( ... );                                                                  \
                                                                                             \
 public:                                                                                     \
   enum : bool { value = ( sizeof( test<Derived>( nullptr ) ) == sizeof( Yes ) ) };          \
};                                                                                           \
                                                                                             \
template< typename TYPE1230 >                                                                \
struct TYPE_TRAIT_NAME                                                                       \
   : public blaze::BoolConstant< TYPE_TRAIT_NAME##HELPER<TYPE1230>::value >                  \
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Macro for the creation of a type trait for compile time checks for member types.
// \ingroup math_type_traits
//
// This macro creates the definition of a type trait \a TYPE_TRAIT_NAME that can determine whether
// the specified element \a MEMBER_NAME is a type member of a given type. The following example
// demonstrates the use of the macro and the resulting type trait:

   \code
   class MyType {
    public:
      typedef int  PublicType;

    protected:
      typedef float  ProtectedType;

    private:
      typedef double  PrivateType;
   };

   BLAZE_CREATE_HAS_TYPE_MEMBER_TYPE_TRAIT( HasPublicType   , PublicType    );
   BLAZE_CREATE_HAS_TYPE_MEMBER_TYPE_TRAIT( HasProtectedType, ProtectedType );
   BLAZE_CREATE_HAS_TYPE_MEMBER_TYPE_TRAIT( HasPrivateType  , PrivateType   );

   BLAZE_CREATE_HAS_TYPE_MEMBER_TYPE_TRAIT( HasValueType  , ValueType    );
   BLAZE_CREATE_HAS_TYPE_MEMBER_TYPE_TRAIT( HasElementType, ElementTypeType );
   BLAZE_CREATE_HAS_TYPE_MEMBER_TYPE_TRAIT( HasDataType   , DataType   );

   HasPublicType<MyType>::value    // Evaluates to 'true'
   HasProtectedType<MyType>::Type  // Results in TrueType
   HasPrivateType<MyType>          // Is derived from TrueType
   HasValueType<MyType>::value     // Evaluates to 'false'
   HasElementType<MyType>::Type    // Results in FalseType
   HasDataType<MyType>             // Is derived from FalseType
   \endcode

// The macro results in the definition of a new class with the specified name \a TYPE_TRAIT_NAME
// within the current namespace. This may cause name collisions with any other entity called
// \a TYPE_TRAIT_NAME in the same namespace. Therefore it is advisable to create the type trait
// as locally as possible to minimize the probability of name collisions. Note however that the
// macro cannot be used within function scope since a template declaration cannot appear at
// block scope.
//
// Please note that due to an error in the Intel compilers prior to version 14.0 the type trait
// generated from this macro does NOT work properly, i.e. will not correctly determine whether
// the specified element is a type member of the given type!
*/
#define BLAZE_CREATE_HAS_TYPE_MEMBER_TYPE_TRAIT( TYPE_TRAIT_NAME, MEMBER_NAME )            \
                                                                                           \
template < typename TYPE1231 >                                                             \
struct TYPE_TRAIT_NAME##HELPER                                                             \
{                                                                                          \
 private:                                                                                  \
   using Yes = char[1];                                                                    \
   using No  = char[2];                                                                    \
                                                                                           \
   struct Base {};                                                                         \
                                                                                           \
   struct Fallback { struct MEMBER_NAME { }; };                                            \
                                                                                           \
   struct Derived                                                                          \
      : blaze::If< blaze::IsBuiltin<TYPE1231>, Base, TYPE1231 >::Type                      \
      , Fallback                                                                           \
   {};                                                                                     \
                                                                                           \
   template < class U >                                                                    \
   static No& test( typename U::MEMBER_NAME* );                                            \
                                                                                           \
   template < typename U >                                                                 \
   static Yes& test( U* );                                                                 \
                                                                                           \
 public:                                                                                   \
   enum : bool { value = ( sizeof( test<Derived>( nullptr ) ) == sizeof( Yes ) ) };        \
};                                                                                         \
                                                                                           \
template< typename TYPE1231 >                                                              \
struct TYPE_TRAIT_NAME                                                                     \
   : public blaze::BoolConstant< TYPE_TRAIT_NAME##HELPER<TYPE1231>::value >                \
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Macro for the creation of a type trait for compile time checks for members.
// \ingroup math_type_traits
//
// This macro creates the definition of a type trait \a TYPE_TRAIT_NAME that can determine whether
// the specified element \a MEMBER_NAME is a data, function, or type member of a given type. The
// following example demonstrates the use of the macro and the resulting type trait:

   \code
   class MyType {
    public:
      void publicCompute();

    protected:
      typedef float  ProtectedType;

    private:
      int value_;
   };

   BLAZE_CREATE_HAS_MEMBER_TYPE_TRAIT( HasCompute      , publicCompute );
   BLAZE_CREATE_HAS_MEMBER_TYPE_TRAIT( HasProtectedType, ProtectedType );
   BLAZE_CREATE_HAS_MEMBER_TYPE_TRAIT( HasValue        , value_        );

   BLAZE_CREATE_HAS_MEMBER_TYPE_TRAIT( HasPublicType, PublicType );
   BLAZE_CREATE_HAS_MEMBER_TYPE_TRAIT( HasDetermine , determine  );
   BLAZE_CREATE_HAS_MEMBER_TYPE_TRAIT( HasData      , data_      );

   HasCompute<MyType>::value       // Evaluates to 'true'
   HasProtectedType<MyType>::Type  // Results in TrueType
   HasValue<MyType>                // Is derived from TrueType
   HasPublicType<MyType>::value    // Evaluates to 'false'
   HasDetermine<MyType>::Type      // Results in FalseType
   HasData<MyType>                 // Is derived from FalseType
   \endcode

// The macro results in the definition of a new class with the specified name \a TYPE_TRAIT_NAME
// within the current namespace. This may cause name collisions with any other entity called
// \a TYPE_TRAIT_NAME in the same namespace. Therefore it is advisable to create the type trait
// as locally as possible to minimize the probability of name collisions. Note however that the
// macro cannot be used within function scope since a template declaration cannot appear at
// block scope.
//
// Please note that due to an error in the Intel compilers prior to version 14.0 the type trait
// generated from this macro does NOT work properly, i.e. will not correctly determine whether
// the specified element is a type member of the given type!
*/
#define BLAZE_CREATE_HAS_MEMBER_TYPE_TRAIT( TYPE_TRAIT_NAME, MEMBER_NAME )                  \
                                                                                            \
template< typename Type1232 >                                                               \
struct TYPE_TRAIT_NAME##HELPER                                                              \
{                                                                                           \
 private:                                                                                   \
   BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( LOCAL_TYPE_TRAIT_1, MEMBER_NAME );  \
   BLAZE_CREATE_HAS_TYPE_MEMBER_TYPE_TRAIT( LOCAL_TYPE_TRAIT_2, MEMBER_NAME );              \
                                                                                            \
 public:                                                                                    \
   static constexpr bool value = ( LOCAL_TYPE_TRAIT_1<Type1232>::value ||                   \
                                   LOCAL_TYPE_TRAIT_2<Type1232>::value );                   \
};                                                                                          \
                                                                                            \
template< typename Type1232 >                                                               \
struct TYPE_TRAIT_NAME                                                                      \
   : public blaze::BoolConstant< TYPE_TRAIT_NAME##HELPER<Type1232>::value >                 \
{};
//*************************************************************************************************

#endif
