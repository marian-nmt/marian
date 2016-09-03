//=================================================================================================
/*!
//  \file blaze/util/singleton/Singleton.h
//  \brief Header file for the Singleton class
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
// //=================================================================================================

#ifndef _BLAZE_UTIL_SINGLETON_SINGLETON_H_
#define _BLAZE_UTIL_SINGLETON_SINGLETON_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <memory>
#include <mutex>
#include <blaze/util/constraints/DerivedFrom.h>
#include <blaze/util/NonCopyable.h>
#include <blaze/util/NullType.h>
#include <blaze/util/Suffix.h>
#include <blaze/util/TypeList.h>


namespace blaze {

//=================================================================================================
//
//  ::blaze NAMESPACE FORWARD DECLARATIONS
//
//=================================================================================================

template< typename > class Dependency;
template< typename T, typename TL, bool C > struct HasCyclicDependency;




//=================================================================================================
//
//  CLASS HASCYCLICDEPENDENCYHELPER
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the HasCyclicDependency class template.
// \ingroup singleton
//
// Helper template class for the HasCyclicDependency template class to resolve all lifetime
// dependencies represented by means of a dependency type list.
*/
template< typename TL                      // Type list of checked lifetime dependencies
        , typename D                       // Type list of lifetime dependencies to check
        , size_t   N = Length<D>::value >  // Length of the dependency type list
struct HasCyclicDependencyHelper;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasCyclicDependencyHelper class template.
// \ingroup singleton
//
// This specialization of the HasCyclicDependencyHelper class is selected in case the given
// dependency type list is empty. In this case no cyclic lifetime dependency could be detected.
*/
template< typename TL   // Type list of checked lifetime dependencies
        , size_t   N >  // Length of the dependency type list
struct HasCyclicDependencyHelper<TL,NullType,N>
{
   enum : bool { value = 0 };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasCyclicDependencyHelper class template.
// \ingroup singleton
//
// This specialization of the HasCyclicDependencyHelper class is selected in case the length
// of the given type list is 1.
*/
template< typename TL   // Type list of checked lifetime dependencies
        , typename D >  // Type list of lifetime dependencies to check
struct HasCyclicDependencyHelper<TL,D,1>
{
   typedef typename TypeAt<D,0>::Result  D1;

   enum : bool { value = HasCyclicDependency<D1,TL,Contains<TL,D1>::value>::value };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasCyclicDependencyHelper class template.
// \ingroup singleton
//
// This specialization of the HasCyclicDependencyHelper class is selected in case the length
// of the given type list is 2.
*/
template< typename TL   // Type list of checked lifetime dependencies
        , typename D >  // Type list of lifetime dependencies to check
struct HasCyclicDependencyHelper<TL,D,2>
{
   typedef typename TypeAt<D,0>::Result  D1;
   typedef typename TypeAt<D,1>::Result  D2;

   enum : bool { value = HasCyclicDependency<D1,TL,Contains<TL,D1>::value>::value ||
                         HasCyclicDependency<D2,TL,Contains<TL,D2>::value>::value };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasCyclicDependencyHelper class template.
// \ingroup singleton
//
// This specialization of the HasCyclicDependencyHelper class is selected in case the length
// of the given type list is 3.
*/
template< typename TL   // Type list of checked lifetime dependencies
        , typename D >  // Type list of lifetime dependencies to check
struct HasCyclicDependencyHelper<TL,D,3>
{
   typedef typename TypeAt<D,0>::Result  D1;
   typedef typename TypeAt<D,1>::Result  D2;
   typedef typename TypeAt<D,2>::Result  D3;

   enum : bool { value = HasCyclicDependency<D1,TL,Contains<TL,D1>::value>::value ||
                         HasCyclicDependency<D2,TL,Contains<TL,D2>::value>::value ||
                         HasCyclicDependency<D3,TL,Contains<TL,D3>::value>::value };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasCyclicDependencyHelper class template.
// \ingroup singleton
//
// This specialization of the HasCyclicDependencyHelper class is selected in case the length
// of the given type list is 4.
*/
template< typename TL   // Type list of checked lifetime dependencies
        , typename D >  // Type list of lifetime dependencies to check
struct HasCyclicDependencyHelper<TL,D,4>
{
   typedef typename TypeAt<D,0>::Result  D1;
   typedef typename TypeAt<D,1>::Result  D2;
   typedef typename TypeAt<D,2>::Result  D3;
   typedef typename TypeAt<D,3>::Result  D4;

   enum : bool { value = HasCyclicDependency<D1,TL,Contains<TL,D1>::value>::value ||
                         HasCyclicDependency<D2,TL,Contains<TL,D2>::value>::value ||
                         HasCyclicDependency<D3,TL,Contains<TL,D3>::value>::value ||
                         HasCyclicDependency<D4,TL,Contains<TL,D4>::value>::value };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasCyclicDependencyHelper class template.
// \ingroup singleton
//
// This specialization of the HasCyclicDependencyHelper class is selected in case the length
// of the given type list is 5.
*/
template< typename TL   // Type list of checked lifetime dependencies
        , typename D >  // Type list of lifetime dependencies to check
struct HasCyclicDependencyHelper<TL,D,5>
{
   typedef typename TypeAt<D,0>::Result  D1;
   typedef typename TypeAt<D,1>::Result  D2;
   typedef typename TypeAt<D,2>::Result  D3;
   typedef typename TypeAt<D,3>::Result  D4;
   typedef typename TypeAt<D,4>::Result  D5;

   enum : bool { value = HasCyclicDependency<D1,TL,Contains<TL,D1>::value>::value ||
                         HasCyclicDependency<D2,TL,Contains<TL,D2>::value>::value ||
                         HasCyclicDependency<D3,TL,Contains<TL,D3>::value>::value ||
                         HasCyclicDependency<D4,TL,Contains<TL,D4>::value>::value ||
                         HasCyclicDependency<D5,TL,Contains<TL,D5>::value>::value };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasCyclicDependencyHelper class template.
// \ingroup singleton
//
// This specialization of the HasCyclicDependencyHelper class is selected in case the length
// of the given type list is 6.
*/
template< typename TL   // Type list of checked lifetime dependencies
        , typename D >  // Type list of lifetime dependencies to check
struct HasCyclicDependencyHelper<TL,D,6>
{
   typedef typename TypeAt<D,0>::Result  D1;
   typedef typename TypeAt<D,1>::Result  D2;
   typedef typename TypeAt<D,2>::Result  D3;
   typedef typename TypeAt<D,3>::Result  D4;
   typedef typename TypeAt<D,4>::Result  D5;
   typedef typename TypeAt<D,5>::Result  D6;

   enum : bool { value = HasCyclicDependency<D1,TL,Contains<TL,D1>::value>::value ||
                         HasCyclicDependency<D2,TL,Contains<TL,D2>::value>::value ||
                         HasCyclicDependency<D3,TL,Contains<TL,D3>::value>::value ||
                         HasCyclicDependency<D4,TL,Contains<TL,D4>::value>::value ||
                         HasCyclicDependency<D5,TL,Contains<TL,D5>::value>::value ||
                         HasCyclicDependency<D6,TL,Contains<TL,D6>::value>::value };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasCyclicDependencyHelper class template.
// \ingroup singleton
//
// This specialization of the HasCyclicDependencyHelper class is selected in case the length
// of the given type list is 7.
*/
template< typename TL   // Type list of checked lifetime dependencies
        , typename D >  // Type list of lifetime dependencies to check
struct HasCyclicDependencyHelper<TL,D,7>
{
   typedef typename TypeAt<D,0>::Result  D1;
   typedef typename TypeAt<D,1>::Result  D2;
   typedef typename TypeAt<D,2>::Result  D3;
   typedef typename TypeAt<D,3>::Result  D4;
   typedef typename TypeAt<D,4>::Result  D5;
   typedef typename TypeAt<D,5>::Result  D6;
   typedef typename TypeAt<D,6>::Result  D7;

   enum : bool { value = HasCyclicDependency<D1,TL,Contains<TL,D1>::value>::value ||
                         HasCyclicDependency<D2,TL,Contains<TL,D2>::value>::value ||
                         HasCyclicDependency<D3,TL,Contains<TL,D3>::value>::value ||
                         HasCyclicDependency<D4,TL,Contains<TL,D4>::value>::value ||
                         HasCyclicDependency<D5,TL,Contains<TL,D5>::value>::value ||
                         HasCyclicDependency<D6,TL,Contains<TL,D6>::value>::value ||
                         HasCyclicDependency<D7,TL,Contains<TL,D7>::value>::value };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasCyclicDependencyHelper class template.
// \ingroup singleton
//
// This specialization of the HasCyclicDependencyHelper class is selected in case the length
// of the given type list is 8.
*/
template< typename TL   // Type list of checked lifetime dependencies
        , typename D >  // Type list of lifetime dependencies to check
struct HasCyclicDependencyHelper<TL,D,8>
{
   typedef typename TypeAt<D,0>::Result  D1;
   typedef typename TypeAt<D,1>::Result  D2;
   typedef typename TypeAt<D,2>::Result  D3;
   typedef typename TypeAt<D,3>::Result  D4;
   typedef typename TypeAt<D,4>::Result  D5;
   typedef typename TypeAt<D,5>::Result  D6;
   typedef typename TypeAt<D,6>::Result  D7;
   typedef typename TypeAt<D,7>::Result  D8;

   enum : bool { value = HasCyclicDependency<D1,TL,Contains<TL,D1>::value>::value ||
                         HasCyclicDependency<D2,TL,Contains<TL,D2>::value>::value ||
                         HasCyclicDependency<D3,TL,Contains<TL,D3>::value>::value ||
                         HasCyclicDependency<D4,TL,Contains<TL,D4>::value>::value ||
                         HasCyclicDependency<D5,TL,Contains<TL,D5>::value>::value ||
                         HasCyclicDependency<D6,TL,Contains<TL,D6>::value>::value ||
                         HasCyclicDependency<D7,TL,Contains<TL,D7>::value>::value ||
                         HasCyclicDependency<D8,TL,Contains<TL,D8>::value>::value };
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CLASS HASCYCLICDEPENDENCY
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Class template for the detection of cyclic lifetime dependencies.
// \ingroup singleton
//
// This class template checks the given type \a T for cyclic lifetime dependencies. In case a
// cyclic lifetime dependency is detected, the \a value member enumeration is set to 1. Otherwise
// it is set to 0.
*/
template< typename T                      // The type to be checked for cyclic lifetime dependencies
        , typename TL                     // Type list of checked lifetime dependencies
        , bool C=Contains<TL,T>::value >  // Flag to indicate whether T is contained in TL
struct HasCyclicDependency
{
   typedef typename Append<TL,T>::Result  ETL;
   enum : bool { value = HasCyclicDependencyHelper<ETL,typename T::Dependencies>::value };
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasCyclicDependency class template.
// \ingroup singleton
//
// This specialization of the HasCyclicDependency class is selected in case the given type \a T
// is contained in the given lifetime dependency type list \a TL. In this case a cyclic lifetime
// dependency was detected and the \a value member enumeration is set to 1.
*/
template< typename T     // The type to be checked for cyclic lifetime dependencies
        , typename TL >  // Type list of checked lifetime dependencies
struct HasCyclicDependency<T,TL,true>
{
   enum : bool { value = 1 };
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DETECT_CYCLIC_LIFETIME_DEPENDENCY CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup singleton
//
// In case the given data type \a T is not an integral data type, a compilation error is created.
*/
#define BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY(T) \
   static_assert( ( !blaze::HasCyclicDependency<T,blaze::NullType>::value ), "Cyclic dependency detected" )
//*************************************************************************************************




//=================================================================================================
//
//  BEFRIEND_SINGLETON MACRO
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Friendship declaration for the Singleton class template.
// \ingroup singleton
//
// This macro has to be used in order to declare the Singleton functionality as friend of the
// class deriving from Singleton.
*/
#define BLAZE_BEFRIEND_SINGLETON \
   template< typename, typename, typename, typename, typename, typename, typename, typename, typename > friend class blaze::Singleton; \
   template< typename, typename, bool > friend struct blaze::HasCyclicDependency; \
   template< typename > friend class blaze::Dependency;
//*************************************************************************************************




//=================================================================================================
//
//  CLASS SINGLETON
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup singleton Singleton
// \ingroup util
//
// \section motivation Motivation
//
// The singleton design pattern is one of the most popular and most important design patterns
// available. It can be used to ensures that a specific class has only exactly one instance,
// and provides a global access point to this instance [1,2]. Additionally, via the singleton
// pattern it is posssible to manage the lifetime of objects, and especially the lifetime
// dependencies between several objects.\n
//
// In the Blaze library the singleton pattern is realized by the Singleton class template.
// Classes that are supposed to be implemented in terms of the singleton pattern only have
// to derive from this class in order to gain all necessary characteristics of a singleton:
//
//  - non-copyability via the NonCopyable base class
//  - a single point of access via the thread-safe [3,4] instance() member function
//  - explicit specification of lifetime dependencies; this feature provides a controlled
//    order of destruction of all singleton objects depending on a non-cyclic dependency
//    tree [4,5]
//  - compile time detection of cyclic lifetime dependencies
//
// The only precondition on classes deriving from the Singleton class is the availability of
// a default constructor. In case it is not possible to instantiate the class via a default
// constructor, i.e., in case the class has only constructors that require at least a single
// argument, the Blaze Singleton implementation cannot be used!
//
//
// \section usage Usage of the Singleton
//
// In order to make a specific class a singleton, two modifications have to be applied to this
// class:
//  -# The class has to derive (publicly or non-publicly) from the Singleton class. In case the
//     class derives publicly the instance() member function, which the class inherits from the
//     Singleton class, is publicly accessible and provides a point of access to the singleton
//     instance. In case the class derives non-publicly, the instance() function is not publicly
//     accessible and therefore the class has to provide another point of access to the singleton
//     instance.\n
//     The first template parameter has to be the class itself. The following template parameters
//     define lifetime dependencies of this class, i.e., specify on which singleton instances the
//     class depends. It is possible to specify up to 8 lifetime dependencies. The example below
//     demonstrates this for the MySingleton class, which is solely depending on the Logger class,
//     which represents the core of the Blaze logging functionality.
//  -# The class needs to befriend the Singleton via the blaze::BLAZE_BEFRIEND_SINGLETON macro.
//     This macro provides a convenient way to express this friendship relation and works both in
//     case the class derives publicly or non-publicly from the Singleton class. This friendship
//     is necessary since in order to guarantee the uniqueness of the singleton instance the
//     constructor of the deriving class must be declared in a non-public section of the class
//     definition. However, in order for the Singleton class to provide the instance() function,
//     the constructor must be accessible. This is achieved by the blaze::BLAZE_BEFRIEND_SINGLETON
//     macro. The following example demonstrates this by means of the MySingleton class:

   \code
   class MySingleton : private Singleton<MySingleton,Logger>
   {
    private:
      MySingleton();

      ...
      BLAZE_BEFRIEND_SINGLETON;
      ...
   };
   \endcode

// \section references References
//
// [1] E. Gamma, R. Helm, R.E. Johnson, J.M. Vlissides: Design Patterns, Addison-Wesley
//     Professional Computing Series, 2008, ISBN: 978-0-201-63361-0\n
// [2] S. Meyers: Effective C++, Third Edition, Addison-Wesley Professional Computing Series,
//     2008, ISBN: 978-0-321-33487-9\n
// [3] J. Ringle: Singleton Creation the Thread-safe Way, Dr. Dobb's (www.drdobbs.com), 1999\n
// [4] A. Alexandrescu: Modern C++ Design, Generic Programming and Design Patterns Applied,
//     Addison-Wesley, 2001, ISBN: 978-0201704310\n
// [5] E. Gabrilovich: Controlling the Destruction Order of Singleton Objects, Dr. Dobbs
//     (www.drdobbs.com), 1999\n
*/
/*!\brief Base class for all lifetime managed singletons.
// \ingroup singleton
//
// The Singleton class represents the base class for all lifetime managed singletons of the
// Blaze library. Classes, which are supposed to be implemented in terms of the singleton
// pattern, only have to derive from this class in order to gain all basic characteristics
// of a singleton:
//
//  - non-copyability via the NonCopyable base class
//  - a single point of access via the thread-safe instance() member function
//  - explicit specification of lifetime dependencies; this feature provides a controlled
//    order of destruction of all singleton objects depending on a non-cyclic dependency
//    tree
//  - compile time detection of cyclic lifetime dependencies
//
// The only prerequisite for classes deriving from the Singleton class template is the existance
// of a default constructor. In case no default constructor is available, the Blaze singleton
// functionality cannot be used!\n
// When using the Singleton base class, lifetime dependencies between classes can be expressed
// very conveniently. The following example demonstrates this by means of the MySingleton class,
// which defines a lifetime dependency on the Logger class, which represents the core of the
// \b Blaze logging functionality:

   \code
   // Definition of the MySingleton class
   class MySingleton : private Singleton<MySingleton,Logger>
   {
    private:
      MySingleton();

      ...
      BLAZE_BEFRIEND_SINGLETON;
      ...
   };
   \endcode

// In order to make a specific class a singleton, two modifications have to be applied to this
// class:
//  -# The class has to derive (publicly or non-publicly) from the Singleton class. In case the
//     class derives publicly the instance() member function, which the class inherits from the
//     Singleton class, is publicly accessible and provides a point of access to the singleton
//     instance. In case the class derives non-publicly, the instance() function is not publicly
//     accessible and therefore the class has to provide another point of access to the singleton
//     instance.\n
//     The first template parameter has to be the class itself. The following template parameters
//     define lifetime dependencies of this class, i.e., specify on which singleton instances the
//     class depends. It is possible to specify up to 8 lifetime dependencies. The example above
//     demonstrates this for the MySingleton class, which is solely depending on the Logger class,
//     which represents the core of the Blaze logging functionality.
//  -# The class needs to befriend the Singleton via the blaze::BLAZE_BEFRIEND_SINGLETON macro.
//     This macro provides a convenient way to express this friendship relation and works both in
//     case the class derives publicly or non-publicly from the Singleton class. This friendship
//     is necessary since in order to guarantee the uniqueness of the singleton instance the
//     constructor of the deriving class must be declared in a non-public section of the class
//     definition. However, in order for the Singleton class to provide the instance() function,
//     the constructor must be accessible. This is achieved by the blaze::BLAZE_BEFRIEND_SINGLETON
//     macro.
*/
template< typename T                // Type of the singleton (CRTP pattern)
        , typename D1 = NullType    // Type of the first lifetime dependency
        , typename D2 = NullType    // Type of the second lifetime dependency
        , typename D3 = NullType    // Type of the third lifetime dependency
        , typename D4 = NullType    // Type of the fourth lifetime dependency
        , typename D5 = NullType    // Type of the fifth lifetime dependency
        , typename D6 = NullType    // Type of the sixth lifetime dependency
        , typename D7 = NullType    // Type of the seventh lifetime dependency
        , typename D8 = NullType >  // Type of the eighth lifetime dependency
class Singleton : private NonCopyable
{
 public:
   //**Type definitions****************************************************************************
   //! Type of this Singleton instance.
   typedef Singleton<T,D1,D2,D3,D4,D5,D6,D7,D8>  SingletonType;

   //! Type list of all lifetime dependencies.
   typedef BLAZE_TYPELIST_8( D1, D2, D3, D4, D5, D6, D7, D8 )  Dependencies;
   //**********************************************************************************************

 protected:
   //**Constructor*********************************************************************************
   /*!\brief Constructor for the Singleton class.
   //
   // In case a cyclic lifetime dependency is detected, a compilation error is created.
   */
   explicit Singleton()
      : dependency1_( D1::instance() )  // Handle to the first lifetime dependency
      , dependency2_( D2::instance() )  // Handle to the second lifetime dependency
      , dependency3_( D3::instance() )  // Handle to the third lifetime dependency
      , dependency4_( D4::instance() )  // Handle to the fourth lifetime dependency
      , dependency5_( D5::instance() )  // Handle to the fifth lifetime dependency
      , dependency6_( D6::instance() )  // Handle to the sixth lifetime dependency
      , dependency7_( D7::instance() )  // Handle to the seventh lifetime dependency
      , dependency8_( D8::instance() )  // Handle to the eighth lifetime dependency
   {
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( T, SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D1, typename D1::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D2, typename D2::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D3, typename D3::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D4, typename D4::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D5, typename D5::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D6, typename D6::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D7, typename D7::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D8, typename D8::SingletonType );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D1 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D2 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D3 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D4 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D5 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D6 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D7 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D8 );
   }
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\brief Destructor for the Singleton class.
   */
   ~Singleton()
   {}
   //**********************************************************************************************

 public:
   //**Instance function***************************************************************************
   /*!\name Instance function */
   //@{
   static std::shared_ptr<T> instance()
   {
      static std::shared_ptr<T> object( new T() );
      return object;
   }
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::shared_ptr<D1> dependency1_;  //!< Handle to the first lifetime dependency.
   std::shared_ptr<D2> dependency2_;  //!< Handle to the second lifetime dependency.
   std::shared_ptr<D3> dependency3_;  //!< Handle to the third lifetime dependency.
   std::shared_ptr<D4> dependency4_;  //!< Handle to the fourth lifetime dependency.
   std::shared_ptr<D5> dependency5_;  //!< Handle to the fifth lifetime dependency.
   std::shared_ptr<D6> dependency6_;  //!< Handle to the sixth lifetime dependency.
   std::shared_ptr<D7> dependency7_;  //!< Handle to the seventh lifetime dependency.
   std::shared_ptr<D8> dependency8_;  //!< Handle to the eighth lifetime dependency.
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  SINGLETON SPECIALIZATION (7 LIFETIME DEPENDENCIES)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Singleton class for 7 lifetime dependencies.
// \ingroup singleton
//
// This specialization of the Singleton class template is used in case 7 lifetime dependencies
// are specified.
*/
template< typename T     // Type of the singleton (CRTP pattern)
        , typename D1    // Type of the first lifetime dependency
        , typename D2    // Type of the second lifetime dependency
        , typename D3    // Type of the third lifetime dependency
        , typename D4    // Type of the fourth lifetime dependency
        , typename D5    // Type of the fifth lifetime dependency
        , typename D6    // Type of the sixth lifetime dependency
        , typename D7 >  // Type of the eighth lifetime dependency
class Singleton<T,D1,D2,D3,D4,D5,D6,D7,NullType> : private NonCopyable
{
 public:
   //**Type definitions****************************************************************************
   //! Type of this Singleton instance.
   typedef Singleton<T,D1,D2,D3,D4,D5,D6,D7,NullType>  SingletonType;

   //! Type list of all lifetime dependencies.
   typedef BLAZE_TYPELIST_7( D1, D2, D3, D4, D5, D6, D7 )  Dependencies;
   //**********************************************************************************************

 protected:
   //**Constructor*********************************************************************************
   /*!\brief Constructor for the Singleton class.
   //
   // In case a cyclic lifetime dependency is detected, a compilation error is created.
   */
   explicit Singleton()
      : dependency1_( D1::instance() )  // Handle to the first lifetime dependency
      , dependency2_( D2::instance() )  // Handle to the second lifetime dependency
      , dependency3_( D3::instance() )  // Handle to the third lifetime dependency
      , dependency4_( D4::instance() )  // Handle to the fourth lifetime dependency
      , dependency5_( D5::instance() )  // Handle to the fifth lifetime dependency
      , dependency6_( D6::instance() )  // Handle to the sixth lifetime dependency
      , dependency7_( D7::instance() )  // Handle to the seventh lifetime dependency
   {
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( T, SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D1, typename D1::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D2, typename D2::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D3, typename D3::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D4, typename D4::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D5, typename D5::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D6, typename D6::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D7, typename D7::SingletonType );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D1 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D2 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D3 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D4 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D5 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D6 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D7 );
   }
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\brief Destructor for the Singleton class.
   */
   ~Singleton()
   {}
   //**********************************************************************************************

 public:
   //**Instance function***************************************************************************
   /*!\name Instance function */
   //@{
   static std::shared_ptr<T> instance()
   {
      static std::shared_ptr<T> object( new T() );
      return object;
   }
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::shared_ptr<D1> dependency1_;  //!< Handle to the first lifetime dependency.
   std::shared_ptr<D2> dependency2_;  //!< Handle to the second lifetime dependency.
   std::shared_ptr<D3> dependency3_;  //!< Handle to the third lifetime dependency.
   std::shared_ptr<D4> dependency4_;  //!< Handle to the fourth lifetime dependency.
   std::shared_ptr<D5> dependency5_;  //!< Handle to the fifth lifetime dependency.
   std::shared_ptr<D6> dependency6_;  //!< Handle to the sixth lifetime dependency.
   std::shared_ptr<D7> dependency7_;  //!< Handle to the seventh lifetime dependency.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SINGLETON SPECIALIZATION (6 LIFETIME DEPENDENCIES)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Singleton class for 6 lifetime dependencies.
// \ingroup singleton
//
// This specialization of the Singleton class template is used in case 6 lifetime dependencies
// are specified.
*/
template< typename T     // Type of the singleton (CRTP pattern)
        , typename D1    // Type of the first lifetime dependency
        , typename D2    // Type of the second lifetime dependency
        , typename D3    // Type of the third lifetime dependency
        , typename D4    // Type of the fourth lifetime dependency
        , typename D5    // Type of the fifth lifetime dependency
        , typename D6 >  // Type of the eighth lifetime dependency
class Singleton<T,D1,D2,D3,D4,D5,D6,NullType,NullType> : private NonCopyable
{
 public:
   //**Type definitions****************************************************************************
   //! Type of this Singleton instance.
   typedef Singleton<T,D1,D2,D3,D4,D5,D6,NullType,NullType>  SingletonType;

   //! Type list of all lifetime dependencies.
   typedef BLAZE_TYPELIST_6( D1, D2, D3, D4, D5, D6 )  Dependencies;
   //**********************************************************************************************

 protected:
   //**Constructor*********************************************************************************
   /*!\brief Constructor for the Singleton class.
   //
   // In case a cyclic lifetime dependency is detected, a compilation error is created.
   */
   explicit Singleton()
      : dependency1_( D1::instance() )  // Handle to the first lifetime dependency
      , dependency2_( D2::instance() )  // Handle to the second lifetime dependency
      , dependency3_( D3::instance() )  // Handle to the third lifetime dependency
      , dependency4_( D4::instance() )  // Handle to the fourth lifetime dependency
      , dependency5_( D5::instance() )  // Handle to the fifth lifetime dependency
      , dependency6_( D6::instance() )  // Handle to the sixth lifetime dependency
   {
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( T, SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D1, typename D1::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D2, typename D2::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D3, typename D3::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D4, typename D4::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D5, typename D5::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D6, typename D6::SingletonType );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D1 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D2 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D3 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D4 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D5 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D6 );
   }
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\brief Destructor for the Singleton class.
   */
   ~Singleton()
   {}
   //**********************************************************************************************

 public:
   //**Instance function***************************************************************************
   /*!\name Instance function */
   //@{
   static std::shared_ptr<T> instance()
   {
      static std::shared_ptr<T> object( new T() );
      return object;
   }
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::shared_ptr<D1> dependency1_;  //!< Handle to the first lifetime dependency.
   std::shared_ptr<D2> dependency2_;  //!< Handle to the second lifetime dependency.
   std::shared_ptr<D3> dependency3_;  //!< Handle to the third lifetime dependency.
   std::shared_ptr<D4> dependency4_;  //!< Handle to the fourth lifetime dependency.
   std::shared_ptr<D5> dependency5_;  //!< Handle to the fifth lifetime dependency.
   std::shared_ptr<D6> dependency6_;  //!< Handle to the sixth lifetime dependency.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SINGLETON SPECIALIZATION (5 LIFETIME DEPENDENCIES)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Singleton class for 5 lifetime dependencies.
// \ingroup singleton
//
// This specialization of the Singleton class template is used in case 5 lifetime dependencies
// are specified.
*/
template< typename T     // Type of the singleton (CRTP pattern)
        , typename D1    // Type of the first lifetime dependency
        , typename D2    // Type of the second lifetime dependency
        , typename D3    // Type of the third lifetime dependency
        , typename D4    // Type of the fourth lifetime dependency
        , typename D5 >  // Type of the fifth lifetime dependency
class Singleton<T,D1,D2,D3,D4,D5,NullType,NullType,NullType> : private NonCopyable
{
 public:
   //**Type definitions****************************************************************************
   //! Type of this Singleton instance.
   typedef Singleton<T,D1,D2,D3,D4,D5,NullType,NullType,NullType>  SingletonType;

   //! Type list of all lifetime dependencies.
   typedef BLAZE_TYPELIST_5( D1, D2, D3, D4, D5 )  Dependencies;
   //**********************************************************************************************

 protected:
   //**Constructor*********************************************************************************
   /*!\brief Constructor for the Singleton class.
   //
   // In case a cyclic lifetime dependency is detected, a compilation error is created.
   */
   explicit Singleton()
      : dependency1_( D1::instance() )  // Handle to the first lifetime dependency
      , dependency2_( D2::instance() )  // Handle to the second lifetime dependency
      , dependency3_( D3::instance() )  // Handle to the third lifetime dependency
      , dependency4_( D4::instance() )  // Handle to the fourth lifetime dependency
      , dependency5_( D5::instance() )  // Handle to the fifth lifetime dependency
   {
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( T, SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D1, typename D1::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D2, typename D2::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D3, typename D3::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D4, typename D4::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D5, typename D5::SingletonType );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D1 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D2 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D3 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D4 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D5 );
   }
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\brief Destructor for the Singleton class.
   */
   ~Singleton()
   {}
   //**********************************************************************************************

 public:
   //**Instance function***************************************************************************
   /*!\name Instance function */
   //@{
   static std::shared_ptr<T> instance()
   {
      static std::shared_ptr<T> object( new T() );
      return object;
   }
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::shared_ptr<D1> dependency1_;  //!< Handle to the first lifetime dependency.
   std::shared_ptr<D2> dependency2_;  //!< Handle to the second lifetime dependency.
   std::shared_ptr<D3> dependency3_;  //!< Handle to the third lifetime dependency.
   std::shared_ptr<D4> dependency4_;  //!< Handle to the fourth lifetime dependency.
   std::shared_ptr<D5> dependency5_;  //!< Handle to the fifth lifetime dependency.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SINGLETON SPECIALIZATION (4 LIFETIME DEPENDENCIES)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Singleton class for 4 lifetime dependencies.
// \ingroup singleton
//
// This specialization of the Singleton class template is used in case 4 lifetime dependencies
// are specified.
*/
template< typename T     // Type of the singleton (CRTP pattern)
        , typename D1    // Type of the first lifetime dependency
        , typename D2    // Type of the second lifetime dependency
        , typename D3    // Type of the third lifetime dependency
        , typename D4 >  // Type of the fourth lifetime dependency
class Singleton<T,D1,D2,D3,D4,NullType,NullType,NullType,NullType> : private NonCopyable
{
 public:
   //**Type definitions****************************************************************************
   //! Type of this Singleton instance.
   typedef Singleton<T,D1,D2,D3,D4,NullType,NullType,NullType,NullType>  SingletonType;

   //! Type list of all lifetime dependencies.
   typedef BLAZE_TYPELIST_4( D1, D2, D3, D4 )  Dependencies;
   //**********************************************************************************************

 protected:
   //**Constructor*********************************************************************************
   /*!\brief Constructor for the Singleton class.
   //
   // In case a cyclic lifetime dependency is detected, a compilation error is created.
   */
   explicit Singleton()
      : dependency1_( D1::instance() )  // Handle to the first lifetime dependency
      , dependency2_( D2::instance() )  // Handle to the second lifetime dependency
      , dependency3_( D3::instance() )  // Handle to the third lifetime dependency
      , dependency4_( D4::instance() )  // Handle to the fourth lifetime dependency
   {
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( T, SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D1, typename D1::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D2, typename D2::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D3, typename D3::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D4, typename D4::SingletonType );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D1 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D2 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D3 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D4 );
   }
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\brief Destructor for the Singleton class.
   */
   ~Singleton()
   {}
   //**********************************************************************************************

 public:
   //**Instance function***************************************************************************
   /*!\name Instance function */
   //@{
   static std::shared_ptr<T> instance()
   {
      static std::shared_ptr<T> object( new T() );
      return object;
   }
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::shared_ptr<D1> dependency1_;  //!< Handle to the first lifetime dependency.
   std::shared_ptr<D2> dependency2_;  //!< Handle to the second lifetime dependency.
   std::shared_ptr<D3> dependency3_;  //!< Handle to the third lifetime dependency.
   std::shared_ptr<D4> dependency4_;  //!< Handle to the fourth lifetime dependency.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SINGLETON SPECIALIZATION (3 LIFETIME DEPENDENCIES)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Singleton class for 3 lifetime dependencies.
// \ingroup singleton
//
// This specialization of the Singleton class template is used in case 3 lifetime dependencies
// are specified.
*/
template< typename T     // Type of the singleton (CRTP pattern)
        , typename D1    // Type of the first lifetime dependency
        , typename D2    // Type of the second lifetime dependency
        , typename D3 >  // Type of the third lifetime dependency
class Singleton<T,D1,D2,D3,NullType,NullType,NullType,NullType,NullType> : private NonCopyable
{
 public:
   //**Type definitions****************************************************************************
   //! Type of this Singleton instance.
   typedef Singleton<T,D1,D2,D3,NullType,NullType,NullType,NullType,NullType>  SingletonType;

   //! Type list of all lifetime dependencies.
   typedef BLAZE_TYPELIST_3( D1, D2, D3 )  Dependencies;
   //**********************************************************************************************

 protected:
   //**Constructor*********************************************************************************
   /*!\brief Constructor for the Singleton class.
   //
   // In case a cyclic lifetime dependency is detected, a compilation error is created.
   */
   explicit Singleton()
      : dependency1_( D1::instance() )  // Handle to the first lifetime dependency
      , dependency2_( D2::instance() )  // Handle to the second lifetime dependency
      , dependency3_( D3::instance() )  // Handle to the third lifetime dependency
   {
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( T, SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D1, typename D1::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D2, typename D2::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D3, typename D3::SingletonType );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D1 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D2 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D3 );
   }
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\brief Destructor for the Singleton class.
   */
   ~Singleton()
   {}
   //**********************************************************************************************

 public:
   //**Instance function***************************************************************************
   /*!\name Instance function */
   //@{
   static std::shared_ptr<T> instance()
   {
      static std::shared_ptr<T> object( new T() );
      return object;
   }
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::shared_ptr<D1> dependency1_;  //!< Handle to the first lifetime dependency.
   std::shared_ptr<D2> dependency2_;  //!< Handle to the second lifetime dependency.
   std::shared_ptr<D3> dependency3_;  //!< Handle to the third lifetime dependency.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SINGLETON SPECIALIZATION (2 LIFETIME DEPENDENCIES)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Singleton class for 2 lifetime dependencies.
// \ingroup singleton
//
// This specialization of the Singleton class template is used in case 2 lifetime dependencies
// are specified.
*/
template< typename T     // Type of the singleton (CRTP pattern)
        , typename D1    // Type of the first lifetime dependency
        , typename D2 >  // Type of the second lifetime dependency
class Singleton<T,D1,D2,NullType,NullType,NullType,NullType,NullType,NullType> : private NonCopyable
{
 public:
   //**Type definitions****************************************************************************
   //! Type of this Singleton instance.
   typedef Singleton<T,D1,D2,NullType,NullType,NullType,NullType,NullType,NullType>  SingletonType;

   //! Type list of all lifetime dependencies.
   typedef BLAZE_TYPELIST_2( D1, D2 )  Dependencies;
   //**********************************************************************************************

 protected:
   //**Constructor*********************************************************************************
   /*!\brief Constructor for the Singleton class.
   //
   // In case a cyclic lifetime dependency is detected, a compilation error is created.
   */
   explicit Singleton()
      : dependency1_( D1::instance() )  // Handle to the first lifetime dependency
      , dependency2_( D2::instance() )  // Handle to the second lifetime dependency
   {
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( T, SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D1, typename D1::SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D2, typename D2::SingletonType );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D1 );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D2 );
   }
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\brief Destructor for the Singleton class.
   */
   ~Singleton()
   {}
   //**********************************************************************************************

 public:
   //**Instance function***************************************************************************
   /*!\name Instance function */
   //@{
   static std::shared_ptr<T> instance()
   {
      static std::shared_ptr<T> object( new T() );
      return object;
   }
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::shared_ptr<D1> dependency1_;  //!< Handle to the first lifetime dependency.
   std::shared_ptr<D2> dependency2_;  //!< Handle to the second lifetime dependency.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SINGLETON SPECIALIZATION (1 LIFETIME DEPENDENCY)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Singleton class for a single lifetime dependency.
// \ingroup singleton
//
// This specialization of the Singleton class template is used in case a single lifetime
// dependency is specified.
*/
template< typename T     // Type of the singleton (CRTP pattern)
        , typename D1 >  // Type of the lifetime dependency
class Singleton<T,D1,NullType,NullType,NullType,NullType,NullType,NullType,NullType> : private NonCopyable
{
 public:
   //**Type definitions****************************************************************************
   //! Type of this Singleton instance.
   typedef Singleton<T,D1,NullType,NullType,NullType,NullType,NullType,NullType,NullType>  SingletonType;

   //! Type list of all lifetime dependencies.
   typedef BLAZE_TYPELIST_1( D1 )  Dependencies;
   //**********************************************************************************************

 protected:
   //**Constructor*********************************************************************************
   /*!\brief Constructor for the Singleton class.
   //
   // In case a cyclic lifetime dependency is detected, a compilation error is created.
   */
   explicit Singleton()
      : dependency1_( D1::instance() )  // Handle to the lifetime dependency
   {
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( T, SingletonType );
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( D1, typename D1::SingletonType );
      BLAZE_DETECT_CYCLIC_LIFETIME_DEPENDENCY( D1 );
   }
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\brief Destructor for the Singleton class.
   */
   ~Singleton()
   {}
   //**********************************************************************************************

 public:
   //**Instance function***************************************************************************
   /*!\name Instance function */
   //@{
   static std::shared_ptr<T> instance()
   {
      static std::shared_ptr<T> object( new T() );
      return object;
   }
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::shared_ptr<D1> dependency1_;  //!< Handle to the lifetime dependency.
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SINGLETON SPECIALIZATION (0 LIFETIME DEPENDENCIES)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Singleton class for no lifetime dependencies.
// \ingroup singleton
//
// This specialization of the Singleton class template is used in case no lifetime dependencies
// are specified.
*/
template< typename T >  // Type of the singleton (CRTP pattern)
class Singleton<T,NullType,NullType,NullType,NullType,NullType,NullType,NullType,NullType> : private NonCopyable
{
 public:
   //**Type definitions****************************************************************************
   //! Type of this Singleton instance.
   typedef Singleton<T,NullType,NullType,NullType,NullType,NullType,NullType,NullType,NullType>  SingletonType;

   //! Type list of all lifetime dependencies.
   typedef NullType  Dependencies;
   //**********************************************************************************************

 protected:
   //**Constructor*********************************************************************************
   /*!\brief Constructor for the Singleton class.
   //
   // In case a cyclic lifetime dependency is detected, a compilation error is created.
   */
   explicit Singleton()
   {
      BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( T, SingletonType );
   }
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\brief Destructor for the Singleton class.
   */
   ~Singleton()
   {}
   //**********************************************************************************************

 public:
   //**Instance function***************************************************************************
   /*!\name Instance function */
   //@{
   static std::shared_ptr<T> instance()
   {
      static std::shared_ptr<T> object( new T() );
      return object;
   }
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
