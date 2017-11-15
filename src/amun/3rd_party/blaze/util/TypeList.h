//=================================================================================================
/*!
//  \file blaze/util/TypeList.h
//  \brief Header file for a type list implementation
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

#ifndef _BLAZE_UTIL_TYPELIST_H_
#define _BLAZE_UTIL_TYPELIST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/NullType.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS TYPELIST
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup typelist Type lists
// \ingroup util
//
// Type lists provide the functionality to create lists of data types. In constrast to lists
// of data values (as for instance the std::list class template), type lists are created at
// compile time, not at run time. The type list implementation of the Blaze library closely
// resembles the original implementation of Andrei Alexandrescu (taken from his book Modern
// C++, ISBN: 0201704315). The following example demonstrates, how type lists are created
// and manipulated:

   \code
   // Creating a type list consisting of two fundamental floating point data types
   typedef BLAZE_TYPELIST_2( float, double )  Tmp;

   // Appending a type to the type list
   typedef blaze::Append< Tmp, long double >::Result  Floats;  // Type list contains all floating point data types

   // Calculating the length of the type list (at compile time!)
   const int length = Length< Floats >::value;  // Value evaluates to 3

   // Accessing a specific type of the type list via indexing
   typedef blaze::TypeAt< Floats, 0 >::Result  Index0;

   // Searching the type list for a specific type
   const int index1 = blaze::Contains< Floats, double >::value;   // Value evaluates to 1
   const int index2 = blaze::Contains< Floats, int    >::value;   // Value evaluates to 0

   // Estimating the index of a specific type in the type list
   const int index3 = blaze::IndexOf< Floats, double >::value;    // Value evaluates to 1
   const int index4 = blaze::IndexOf< Floats, int    >::value;    // Value evaluates to -1

   // Erasing the first occurrence of float from the type list
   typedef blaze::Erase< Floats, float >::Result  NoFloat;

   // Removing all duplicates from the type list
   typedef blaze::Unique< Floats >::Result  NoDuplicates;
   \endcode
*/
/*!\brief Implementation of a type list.
// \ingroup typelist
//
// The TypeList class is an implementation of a type list according to the example of Andrei
// Alexandrescu. The type list merely consists of the two data types \a Head and \a Tail. In
// order to create type lists of more data types, the TypeList class is used recursively:

   \code
   // Type list containing the three fundamental floating point data types
   TypeList< float, TypeList< double, TypeList< long double, NullType > > >
   \endcode

// The NullType data type is used to terminate a type list.\n
// In order to create a type list, one of the predefined setup macros should be used:

   \code
   // Creating a type list consisting of the three fundamental data types
   typedef BLAZE_TYPELIST_3( float, double, long double )  Floats;
   \endcode
*/
template< typename H    // Head of the type list
        , typename T >  // Tail of the type list
struct TypeList
{
   //**Type definitions****************************************************************************
   typedef H  Head;  //!< Type of the head of the type list.
   typedef T  Tail;  //!< Type of the tail of the type list.
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  TYPE LIST GENERATION MACROS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Type list generation macro.
// \ingroup typelist
//
// This macro creates a type list only consisting of the type \a T1. The terminating type for
// the type list is the NullType. The following example demonstrates the use of this macro:

   \code
   // Definition of a new type list consisting of a single data type
   typedef BLAZE_TYPELIST_1( int )  MyTypes;

   // Calculating the length of the type list
   const int length = Length<MyTypes>::value;
   \endcode
*/
#define BLAZE_TYPELIST_1( T1 ) \
   TypeList< T1, NullType >
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Type list generation macro.
// \ingroup typelist
//
// This macro creates a type list consisting of the two types \a T1 and \a T2. The terminating
// type for the type list is the NullType. The following example demonstrates the use of this
// macro:

   \code
   // Definition of a new type list consisting of two data types
   typedef BLAZE_TYPELIST_2( int, unsigned int )  MyTypes;

   // Calculating the length of the type list
   const int length = Length<MyTypes>::value;
   \endcode
*/
#define BLAZE_TYPELIST_2( T1, T2 ) \
   TypeList< T1, BLAZE_TYPELIST_1( T2 ) >
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Type list generation macro.
// \ingroup typelist
//
// This macro creates a type list consisting of the three types \a T1, \a T2 and \a T3. The
// terminating type for the type list is the NullType. The following example demonstrates
// the use of this macro:

   \code
   // Definition of a new type list consisting of three data types
   typedef BLAZE_TYPELIST_3( float, double, long double )  MyTypes;

   // Calculating the length of the type list
   const int length = Length<MyTypes>::value;
   \endcode
*/
#define BLAZE_TYPELIST_3( T1, T2, T3 ) \
   TypeList< T1, BLAZE_TYPELIST_2( T2, T3 ) >
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Type list generation macro.
// \ingroup typelist
//
// This macro creates a type list consisting of the four types \a T1, \a T2, \a T3 and \a T4.
// The terminating type for the type list is the NullType. The following example demonstrates
// the use of this macro:

   \code
   // Definition of a new type list consisting of four data types
   typedef BLAZE_TYPELIST_4( unsigned char, signed char, char, wchar_t )  MyTypes;

   // Calculating the length of the type list
   const int length = Length<MyTypes>::value;
   \endcode
*/
#define BLAZE_TYPELIST_4( T1, T2, T3, T4 ) \
   TypeList< T1, BLAZE_TYPELIST_3( T2, T3, T4 ) >
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Type list generation macro.
// \ingroup typelist
//
// This macro creates a type list consisting of the five types \a T1, \a T2, \a T3, \a T4
// and \a T5. The terminating type for the type list is the NullType. The following example
// demonstrates the use of this macro:

   \code
   // Definition of a new type list consisting of five data types
   typedef BLAZE_TYPELIST_5( char, short, int, long, float )  MyTypes;

   // Calculating the length of the type list
   const int length = Length<MyTypes>::value;
   \endcode
*/
#define BLAZE_TYPELIST_5( T1, T2, T3, T4, T5 ) \
   TypeList< T1, BLAZE_TYPELIST_4( T2, T3, T4, T5 ) >
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Type list generation macro.
// \ingroup typelist
//
// This macro creates a type list consisting of the six types \a T1, \a T2, \a T3, \a T4, \a T5
// and \a T6. The terminating type for the type list is the NullType. The following example
// demonstrates the use of this macro:

   \code
   // Definition of a new type list consisting of six data types
   typedef BLAZE_TYPELIST_6( char, short, int, long, float, double )  MyTypes;

   // Calculating the length of the type list
   const int length = Length<MyTypes>::value;
   \endcode
*/
#define BLAZE_TYPELIST_6( T1, T2, T3, T4, T5, T6 ) \
   TypeList< T1, BLAZE_TYPELIST_5( T2, T3, T4, T5, T6 ) >
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Type list generation macro.
// \ingroup typelist
//
// This macro creates a type list consisting of the seven types \a T1, \a T2, \a T3, \a T4,
// \a T5, \a T6 and \a T7. The terminating type for the type list is the NullType. The
// following example demonstrates the use of this macro:

   \code
   // Definition of a new type list consisting of seven data types
   typedef BLAZE_TYPELIST_7( char, short, int, long, float, double, long double )  MyTypes;

   // Calculating the length of the type list
   const int length = Length<MyTypes>::value;
   \endcode
*/
#define BLAZE_TYPELIST_7( T1, T2, T3, T4, T5, T6, T7 ) \
   TypeList< T1, BLAZE_TYPELIST_6( T2, T3, T4, T5, T6, T7 ) >
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Type list generation macro.
// \ingroup typelist
//
// This macro creates a type list consisting of the eight types \a T1, \a T2, \a T3, \a T4,
// \a T5, \a T6, \a T7 and \a T8. The terminating type for the type list is the NullType.
// The following example demonstrates the use of this macro:

   \code
   // Definition of a new type list consisting of eight data types
   typedef BLAZE_TYPELIST_8( char, wchar_t, short, int, long, float, double, long double )  MyTypes;

   // Calculating the length of the type list
   const int length = Length<MyTypes>::value;
   \endcode
*/
#define BLAZE_TYPELIST_8( T1, T2, T3, T4, T5, T6, T7, T8 ) \
   TypeList< T1, BLAZE_TYPELIST_7( T2, T3, T4, T5, T6, T7, T8 ) >
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Type list generation macro.
// \ingroup typelist
//
// This macro creates a type list consisting of the nine types \a T1, \a T2, \a T3, \a T4,
// \a T5, \a T6, \a T7, \a T8 and \a T9. The terminating type for the type list is the NullType.
// The following example demonstrates the use of this macro:

   \code
   // Definition of a new type list consisting of nine data types
   typedef BLAZE_TYPELIST_9( char, signed char, wchar_t, short, int, long, float, double, long double )  MyTypes;

   // Calculating the length of the type list
   const int length = Length<MyTypes>::value;
   \endcode
*/
#define BLAZE_TYPELIST_9( T1, T2, T3, T4, T5, T6, T7, T8, T9 ) \
   TypeList< T1, BLAZE_TYPELIST_8( T2, T3, T4, T5, T6, T7, T8, T9 ) >
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Type list generation macro.
// \ingroup typelist
//
// This macro creates a type list consisting of the ten types \a T1, \a T2, \a T3, \a T4,
// \a T5, \a T6, \a T7, \a T8, \a T9 and \a T10. The terminating type for the type list is
// the NullType. The following example demonstrates the use of this macro:

   \code
   // Definition of a new type list consisting of ten data types
   typedef BLAZE_TYPELIST_10( unsigned char, signed char, char, wchar_t, unsigned short,
                              short, unsigned int, int, unsigned long, long )  MyTypes;

   // Calculating the length of the type list
   const int length = Length<MyTypes>::value;
   \endcode
*/
#define BLAZE_TYPELIST_10( T1, T2, T3, T4, T5, T6, T7, T8, T9, T10 ) \
   TypeList< T1, BLAZE_TYPELIST_9( T2, T3, T4, T5, T6, T7, T8, T9, T10 ) >
//*************************************************************************************************




//=================================================================================================
//
//  LENGTH OF A TYPE LIST
//
//=================================================================================================

//*************************************************************************************************
/*!\class blaze::Length
// \brief Calculating the length of a type list.
// \ingroup typelist
//
// The Length class can be used to obtain the length of a type list (i.e. the number
// of contained types). In order to obtain the length of a type list, the Length class
// has to be instantiated for a particular type list. The length of the type list can
// be obtained using the member enumeration \a value. The following example gives an
// impression of the use of the Length class:

   \code
   typedef BLAZE_TYPELIST_3( float, double, long double )  Floats;  // Defining a new type list
   const int length = blaze::Length< Floats >::value;               // The length of the type list
   \endcode
*/
template< typename TList >  // Type of the type list
struct Length;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Length class for empty type lists.
// \ingroup typelist
*/
template<>
struct Length< NullType >
{
   //**Member enumeration**************************************************************************
   enum { value = 0 };
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Length class for general type lists.
// \ingroup typelist
*/
template< typename Head    // Type of the head of the type list
        , typename Tail >  // Type of the tail of the type list
struct Length< TypeList<Head,Tail> >
{
   //**Member enumeration**************************************************************************
   enum { value = 1 + Length<Tail>::value };
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  INDEXED ACCESS
//
//=================================================================================================

//*************************************************************************************************
/*!\class blaze::TypeAt
// \brief Indexing a type list.
// \ingroup typelist
//
// The TypeAt class can be used to access a type list at a specified position to query the
// according type. In order to index a type list, the TypeAt class has to be instantiated
// for a particular type list and an index value. The indexed type is available via the
// member type definition \a Result. The following example gives an impression of the use
// of the TypeAt class:

   \code
   typedef BLAZE_TYPELIST_3( float, double, long double )  Floats;  // Defining a new type list
   typedef blaze::TypeAt< Floats, 0 >::Result              Index0;  // Indexing of the type list at index 0
   \endcode

// \note The access index is zero based!
*/
template< typename TList  // Type of the type list
        , size_t Index >  // Type list access index
struct TypeAt;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the TypeAt class for an index of 0.
// \ingroup typelist
*/
template< typename Head    // Type of the head of the type list
        , typename Tail >  // Type of the tail of the type list
struct TypeAt< TypeList<Head,Tail>, 0 >
{
   //**Member enumeration**************************************************************************
   typedef Head  Result;  //!< Data type at index 0.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the TypeAt class for the terminating NullType.
// \ingroup typelist
*/
template< size_t Index >  // Type list access index
struct TypeAt< NullType, Index >
{
   //**Member enumeration**************************************************************************
   typedef NullType  Result;  //!< The resulting data type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the TypeAt class for a general index.
// \ingroup typelist
*/
template< typename Head   // Type of the head of the type list
        , typename Tail   // Type of the tail of the type list
        , size_t Index >  // Type list access index
struct TypeAt< TypeList<Head,Tail>, Index >
{
   //**Member enumeration**************************************************************************
   typedef typename TypeAt< Tail, Index-1 >::Result  Result;  //!< Data type at indexed position.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TYPE LIST SEARCH
//
//=================================================================================================

//*************************************************************************************************
/*!\class blaze::Contains
// \brief Searching a type list.
// \ingroup typelist
//
// The Contains class can be used to search the type list for a particular type \a Type. In
// contrast to the IndexOf class, the Contains class does not evaluate the index of the type
// but only checks whether or not the type is contained in the type list. Additionally, in
// contrast to the ContainsRelated class, the Contains class strictly searches for the given
// type \a Type and not for a related data type. In case the type is contained in the type
// list, the \a value member enumeration is set to 1, else it is set to 0. In order to check
// whether a type is part of a type list, the Contains class has to be instantiated for a
// particular type list and another type. The following example gives an impression of the
// use of the Contains class:

   \code
   typedef BLAZE_TYPELIST_3( float, double, long double )  Floats;  // Defining a new type list
   const int index1 = blaze::Contains< Floats, double >::value;     // Value evaluates to 1
   const int index2 = blaze::Contains< Floats, int    >::value;     // Value evaluates to 0
   \endcode
*/
template< typename TList   // Type of the type list
        , typename Type >  // The search type
struct Contains;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Contains class for the terminating NullType.
// \ingroup typelist
*/
template< typename Type >  // The search type
struct Contains< NullType, Type >
{
   //**Member enumeration**************************************************************************
   enum { value = 0 };  //!< \a Type is not contained in the type list.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Contains class for a successful search.
// \ingroup typelist
*/
template< typename Tail    // Type of the tail of the type list
        , typename Type >  // The search type
struct Contains< TypeList<Type,Tail>, Type >
{
   //**Member enumeration**************************************************************************
   enum { value = 1 };  //!< \a Type is the head of the type list.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Contains class for a general type list.
// \ingroup typelist
*/
template< typename Head    // Type of the head of the type list
        , typename Tail    // Type of the tail of the type list
        , typename Type >  // The search type
struct Contains< TypeList<Head,Tail>, Type >
{
   //**Member enumeration**************************************************************************
   enum { value = Contains<Tail,Type>::value };  //!< Search result for type \a Type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\class blaze::ContainsRelated
// \brief Searching a type list.
// \ingroup typelist
//
// The ContainsRelated class can be used to search the type list for a type related to \a Type.
// In contrast to the Contains class, the ContainsRelated class only searches for a type the
// given data type \a Type can be converted to. In case a related type is found in the type
// list, the \a value member enumeration is set to 1, else it is set to 0. In order to check
// whether a related type is contained in the type list, the ContainsRelated class has to be
// instantiated for a particular type list and another type. The following example gives an
// impression of the use of the ContainsRelated class:

   \code
   class A {};
   class B : public A {};
   class C {};
   class D {};

   // Defining a new type list
   typedef BLAZE_TYPELIST_2( A, C )  Types;

   // Searching for the type A in the type list
   const int a = blaze::ContainsRelated< Types, A >::value;  // Evaluates to 1, type A is found

   // Searching for the derived type B in the type list
   const int b = blaze::ContainsRelated< Types, B >::value;  // Evaluates to 1, base type A is found

   // Searching for the type C in the type list
   const int c = blaze::ContainsRelated< Types, D >::value;  // Evaluates to 0, no related type found
   \endcode
*/
template< typename TList   // Type of the type list
        , typename Type >  // The search type
struct ContainsRelated;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the ContainsRelated class for the terminating NullType.
// \ingroup typelist
*/
template< typename Type >  // The search type
struct ContainsRelated< NullType, Type >
{
   //**Member enumeration**************************************************************************
   enum { value = 0 };  //!< No related type of \a Type is contained in the type list.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the ContainsRelated class for a general type list.
// \ingroup typelist
*/
template< typename Head    // Type of the head of the type list
        , typename Tail    // Type of the tail of the type list
        , typename Type >  // The search type
struct ContainsRelated< TypeList<Head,Tail>, Type >
{
 private:
   //**********************************************************************************************
   class No  {};
   class Yes { No no[2]; };
   //**********************************************************************************************

   //**********************************************************************************************
   static Yes  test( Head );
   static No   test( ... );
   static Type createType();
   //**********************************************************************************************

   //**Member enumeration**************************************************************************
   enum { tmp = sizeof( test( createType() ) ) == sizeof( Yes ) ? 1 : 0 };  //!< Relationship evaluation.
   //**********************************************************************************************

 public:
   //**Member enumeration**************************************************************************
   enum { value = tmp == 1 ? 1 : ( ContainsRelated<Tail,Type>::value ) };  //!< Search result for type \a Type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\class blaze::IndexOf
// \brief Searching a type list.
// \ingroup typelist
//
// The IndexOf class can be used to search the type list for a particular type \a Type. In
// contrast to the Contains and the ContainsRelated classes, the IndexOf class evaluates the
// index of the given type in the type list. In case the type is contained in the type list,
// the \a value member represents the index of the queried type. Otherwise the \a value member
// is set to -1. In order to search for a type, the IndexOf class has to be instantiated for
// a particular type list and a search type. The following example gives an impression of the
// use of the IndexOf class:

   \code
   typedef BLAZE_TYPELIST_3( float, double, long double )  Floats;  // Defining a new type list
   const int index1 = blaze::IndexOf< Floats, double >::value;      // Value evaluates to 1
   const int index2 = blaze::IndexOf< Floats, int    >::value;      // Value evaluates to -1
   \endcode
*/
template< typename TList   // Type of the type list
        , typename Type >  // The search type
struct IndexOf;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the IndexOf class for the terminating NullType.
// \ingroup typelist
*/
template< typename Type >  // The search type
struct IndexOf< NullType, Type >
{
   //**Member enumeration**************************************************************************
   enum { value = -1 };  //!< \a Type is not contained in the type list.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the IndexOf class for a successful search.
// \ingroup typelist
*/
template< typename Tail    // Type of the tail of the type list
        , typename Type >  // The search type
struct IndexOf< TypeList<Type,Tail>, Type >
{
   //**Member enumeration**************************************************************************
   enum { value = 0 };  //!< \a Type is the head of the type list.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the IndexOf class for a general type list.
// \ingroup typelist
*/
template< typename Head    // Type of the head of the type list
        , typename Tail    // Type of the tail of the type list
        , typename Type >  // The search type
struct IndexOf< TypeList<Head,Tail>, Type >
{
 private:
   //**Member enumeration**************************************************************************
   enum { tmp = IndexOf<Tail,Type>::value };  //!< Index of \a Type in the tail of the type list.
   //**********************************************************************************************

 public:
   //**Member enumeration**************************************************************************
   enum { value = tmp == -1 ? -1 : 1 + tmp };  //!< Index of \a Type in the entire type list.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  APPENDING TO TYPE LISTS
//
//=================================================================================================

//*************************************************************************************************
/*!\class blaze::Append
// \brief Appending a type to a type list.
// \ingroup typelist
//
// The Append class can be used to append the data type \a Type to a type list \a TList. In
// order to append a data type, the Append class has to be instantiated for a particular type
// list and another type. The following example gives an impression of the use of the Append
// class:

   \code
   typedef BLAZE_TYPELIST_2( float, double )       Tmp;     // Defining a temporary type list
   typedef blaze::Append<Tmp,long double>::Result  Floats;  // Type list contains all floating point data types
   \endcode
*/
template< typename TList   // Type of the type list
        , typename Type >  // The type to be appended to the type list
struct Append;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Append class for appending the NullType.
// \ingroup typelist
*/
template<>
struct Append< NullType, NullType >
{
   //**Type definitions****************************************************************************
   typedef NullType  Result;  //!< The resulting data type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Append class for appending a general type to the NullType.
// \ingroup typelist
*/
template< typename Type >  // The type to be appended to the type list
struct Append< NullType, Type >
{
   //**Type definitions****************************************************************************
   typedef BLAZE_TYPELIST_1( Type )  Result;  //!< The resulting data type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Append class for appending a type list to the NullType.
// \ingroup typelist
*/
template< typename Head    // Type of the head of the type list
        , typename Tail >  // Type of the tail of the type list
struct Append< NullType, TypeList<Head,Tail> >
{
   //**Type definitions****************************************************************************
   typedef TypeList<Head,Tail>  Result;  //!< The resulting data type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Append class for appending a general type to a type list.
// \ingroup typelist
*/
template< typename Head    // Type of the head of the type list
        , typename Tail    // Type of the tail of the type list
        , typename Type >  // The type to be appended to the type list
struct Append< TypeList<Head,Tail>, Type >
{
   //**Type definitions****************************************************************************
   typedef TypeList< Head, typename Append<Tail,Type>::Result >  Result;  //!< The resulting data type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ERASING FROM TYPE LISTS
//
//=================================================================================================

//*************************************************************************************************
/*!\class blaze::Erase
// \brief Erasing the first occurrence of a type from a type list.
// \ingroup typelist
//
// The Erase class can be used to erase the first occurrence of data type \a Type from a type
// list \a TList. In order to erase the first occurrence of a data type, the Erase class has to
// be instantiated for a particular type list and another type. The following example gives an
// impression of the use of the Erase class:

   \code
   // Defining a temporary type list containing the type int twice
   typedef BLAZE_TYPELIST_4( float, int, double, int )  Tmp;

   // Erasing the first occurrence of int from the type list
   typedef blaze::Erase<Tmp,int>::Result  SingleInt;
   \endcode
*/
template< typename TList   // Type of the type list
        , typename Type >  // The type to be erased from the type list
struct Erase;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Erase class for the terminating NullType.
// \ingroup typelist
*/
template< typename Type >  // The type to be erased from the type list
struct Erase< NullType, Type >
{
   //**Type definitions****************************************************************************
   typedef NullType  Result;  //!< The resulting data type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Erase class for erasing the first occurrence of T.
// \ingroup typelist
*/
template< typename Type    // The type to be erased from the type list
        , typename Tail >  // Type of the tail of the type list
struct Erase< TypeList<Type,Tail>, Type >
{
   //**Type definitions****************************************************************************
   typedef Tail  Result;  //!< The resulting data type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Erase class for a general type list.
// \ingroup typelist
*/
template< typename Head    // Type of the head of the type list
        , typename Tail    // Type of the tail of the type list
        , typename Type >  // The type to be erased from the type list
struct Erase< TypeList<Head,Tail>, Type >
{
   //**Type definitions****************************************************************************
   typedef TypeList<Head,typename Erase<Tail,Type>::Result>  Result;  //!< The resulting data type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\class blaze::EraseAll
// \brief Erasing all occurrences of a type from a type list.
// \ingroup typelist
//
// The EraseAll class can be used to erase all occurrences of data type \a Type from a type list
// \a TList. In order to erase all occurrences of a data type, the EraseAll class has to be
// instantiated for a particular type list and another type. The following example gives an
// impression of the use of the EraseAll class:

   \code
   // Defining a temporary type list containing the type int twice
   typedef BLAZE_TYPELIST_4( float, int, double, int )  Tmp;

   // Erasing the all occurrences of int from the type list
   typedef blaze::EraseAll<Tmp,int>::Result  NoInt;
   \endcode
*/
template< typename TList   // Type of the type list
        , typename Type >  // The type to be erased from the type list
struct EraseAll;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the EraseAll class for the terminating NullType.
// \ingroup typelist
*/
template< typename Type >  // The type to be erased from the type list
struct EraseAll< NullType, Type >
{
   //**Type definitions****************************************************************************
   typedef NullType  Result;  //!< The resulting data type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the EraseAll class for erasing an occurrence of T.
// \ingroup typelist
*/
template< typename Type    // The type to be erased from the type list
        , typename Tail >  // Type of the tail of the type list
struct EraseAll< TypeList<Type,Tail>, Type >
{
   //**Type definitions****************************************************************************
   typedef typename EraseAll<Tail,Type>::Result  Result;  //!< The resulting data type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the EraseAll class for a general type list.
// \ingroup typelist
*/
template< typename Head    // Type of the head of the type list
        , typename Tail    // Type of the tail of the type list
        , typename Type >  // The type to be erased from the type list
struct EraseAll< TypeList<Head,Tail>, Type >
{
   //**Type definitions****************************************************************************
   typedef TypeList<Head,typename EraseAll<Tail,Type>::Result>  Result;  //!< The resulting data type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  REMOVING DUPLICATES FROM TYPE LISTS
//
//=================================================================================================

//*************************************************************************************************
/*!\class blaze::Unique
// \brief Erasing all duplicates from a type list.
// \ingroup typelist
//
// The Unique class can be used to erase all duplicates from a type list \a TList. In order to
// erase all duplicates, the Unique class has to be instantiated for a particular type list.
// The following example gives an impression of the use of the Unique class:

   \code
   // Defining a temporary type list containing the types int and float twice
   typedef BLAZE_TYPELIST_5( float, int, double, int, float )  Tmp;

   // Removing all duplicates from the type list
   typedef blaze::Unique<Tmp>::Result  NoDuplicates;
   \endcode
*/
template< typename TList >  // Type of the type list
struct Unique;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Unique class for the terminating NullType.
// \ingroup typelist
*/
template<>
struct Unique< NullType >
{
   //**Type definitions****************************************************************************
   typedef NullType  Result;  //!< The resulting data type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Spezialization of the Unique class for a general type list.
// \ingroup typelist
*/
template< typename Head    // Type of the head of the type list
        , typename Tail >  // Type of the tail of the type list
struct Unique< TypeList<Head,Tail> >
{
 private:
   //**Type definitions****************************************************************************
   typedef typename Unique<Tail>::Result     TL1;
   typedef typename Erase<TL1,Head>::Result  TL2;
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef TypeList<Head,TL2>  Result;  //!< The resulting data type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
