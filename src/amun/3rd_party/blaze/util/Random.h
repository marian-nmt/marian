//=================================================================================================
/*!
//  \file blaze/util/Random.h
//  \brief Implementation of a random number generator.
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

#ifndef _BLAZE_UTIL_RANDOM_H_
#define _BLAZE_UTIL_RANDOM_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <ctime>
#include <limits>
#include <random>
#include <blaze/system/Random.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Complex.h>
#include <blaze/util/NonCreatable.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  DOXYGEN DOCUMENTATION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup random Random number generation
// \ingroup util
//
// The random number module provides the functionality to generate pseudo-random numbers within
// the Blaze library. In order to create series of random numbers, the following functions are
// are provided:
//
// - blaze::rand<T>();
// - blaze::getSeed();
// - blaze::setSeed( uint32_t seed );
//
// The templated rand() function is capable of generating random numbers for built-in integer
// and floating point data types as well as complex values. The rand() function can be given up
// to five parameters that depending on the type of the random number may have different meaning.
// The following example demonstrates the random number generation:

   \code
   using namespace blaze;

   // The setSeed function sets the seed for the random number generator. This function can
   // be used to set a specific seed for the random number generation, e.g. in order to
   // reproduce an exact series of random numbers. If the setSeed function is not used, the
   // random number generation uses a random seed.
   setSeed( 12345 );

   // In order to acquire the currently used seed, the getSeed() function can be used.
   uint32_t seed = getSeed();

   // Generating random numbers of built-in type. In case no range is provided, random numbers
   // of integral type are generated in the range [0..max], where max is the largest possible
   // value of the specified type and random floating point numbers are generated in the range
   // [0..1). In the example, the random double precision floating point value is created in
   // the range [2..4].
   int    i = rand<int>();
   double d = rand<double>( 2.0, 4.0 );

   // Generating random complex numbers. In case no range is provided, both the real and the
   // imaginary part are created depending on the element type of the complex number. In case
   // one range is specified, both the real and the imaginary part are created within the
   // specified range. In the example, both parts are created within the range [1..4]. If two
   // ranges are provided, the real part is created in the first range and the imaginary part
   // is created in the second range. The last example demonstrates this by restricting the
   // real part to the range [2..3] and the imaginary part to the range [1..5].
   complex<float> c1 = rand< complex<float> >();
   complex<float> c2 = rand< complex<float> >( 1.0F, 4.0F );
   complex<float> c3 = rand< complex<float> >( 2.0F, 3.0F, 1.0F, 5.0F );
   \endcode

// \note In order to reproduce certain series of random numbers, the seed of the random number
// generator has to be set explicitly via the setSeed() function. Otherwise a random seed is used
// for the random number generation.
*/
//*************************************************************************************************




//=================================================================================================
//
//  ::blaze NAMESPACE FORWARD DECLARATIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Random number functions */
//@{
template< typename T >
inline T rand();

template< typename T, typename... Args >
inline T rand( Args&&... args );

template< typename T >
inline void randomize( T& value );

template< typename T, typename... Args >
inline void randomize( T& value, Args&&... args );

inline uint32_t defaultSeed();
inline uint32_t getSeed();
inline void     setSeed( uint32_t seed );
//@}
//*************************************************************************************************




//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Random number generator.
// \ingroup random
//
// The Random class encapsulates the initialization of the given random number generator with
// a pseudo-random seed obtained by the std::time() function. Currently, the mersenne-twister
// mt19937 as provided by the C++ standard library is used per default. For more information
// see the for instance the following documentation of the random number functionality of the
// C++11 standard library:
//
//   http://en.cppreference.com/w/cpp/numeric/random
*/
template< typename Type >  // Type of the random number generator
class Random : private NonCreatable
{
 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   static uint32_t seed_;  //!< The current seed for the variate generator.
   static Type     rng_;   //!< The mersenne twister variate generator.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   template< typename T > friend class Rand;
                          friend uint32_t getSeed();
                          friend void     setSeed( uint32_t seed );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  DEFINITION AND INITIALIZATION OF THE STATIC MEMBER VARIABLES
//
//=================================================================================================

template< typename Type > uint32_t Random<Type>::seed_( defaultSeed() );
template< typename Type > Type     Random<Type>::rng_ ( defaultSeed() );




//=================================================================================================
//
//  CLASS RAND
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Default implementation of the Rand class for integral data types.
// \ingroup random
//
// This default implementation of the Rand class creates random, integral numbers in the range
// \f$ [0..max] \f$, where \a max is the maximal value of the given data type \a T.
*/
template< typename T >  // Type of the random number
class Rand
{
 public:
   //**Generate functions**************************************************************************
   /*!\name Generate functions */
   //@{
   inline T generate() const;
   inline T generate( T min, T max ) const;
   //@}
   //**********************************************************************************************

   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( T& value ) const;
   inline void randomize( T& value, T min, T max ) const;
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Generation of a random value in the range \f$ [0..max] \f$.
//
// \return The generated random value.
//
// This \a generate function creates a random number in the range \f$ [0..max] \f$, where \a max
// is the maximal value of the given data type \a T.
*/
template< typename T >  // Type of the random number
inline T Rand<T>::generate() const
{
   std::uniform_int_distribution<T> dist( 0, std::numeric_limits<T>::max() );
   return dist( Random<RNG>::rng_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Generation of a random value in the range \f$ [min..max] \f$.
//
// \param min The smallest possible random value.
// \param max The largest possible random value.
// \return The generated random value.
//
// This \a generate function creates a random number in the range \f$ [min..max] \f$, where
// \a min must be smaller or equal to \a max. Note that this requirement is only checked in
// debug mode. In release mode, no check is performed to enforce the validity of the values.
// Therefore the returned value is undefined if \a min is larger than \a max.
*/
template< typename T >  // Type of the random number
inline T Rand<T>::generate( T min, T max ) const
{
   BLAZE_INTERNAL_ASSERT( min <= max, "Invalid min/max value pair" );
   std::uniform_int_distribution<T> dist( min, max );
   return dist( Random<RNG>::rng_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Randomization of the given variable with a new value in the range \f$ [0..max] \f$.
//
// \param value The variable to be randomized.
// \return void
//
// This function randomizes the given variable to a new value in the range \f$ [0..max] \f$,
// where \a max is the maximal value of the given data type \a T.
*/
template< typename T >  // Type of the random number
inline void Rand<T>::randomize( T& value ) const
{
   value = generate();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Randomization of the given variable with a new value in the range \f$ [min..max] \f$.
//
// \param value The variable to be randomized.
// \param min The smallest possible random value.
// \param max The largest possible random value.
// \return void
//
// This function randomizes the given variable to a new value in the range \f$ [min..max] \f$, where
// \a min must be smaller or equal to \a max. Note that this requirement is only checked in debug
// mode. In release mode, no check is performed to enforce the validity of the values. Therefore
// the returned value is undefined if \a min is larger than \a max.
*/
template< typename T >  // Type of the random number
inline void Rand<T>::randomize( T& value, T min, T max ) const
{
   value = generate( min, max );
}
//*************************************************************************************************




//=================================================================================================
//
//  RAND SPECIALIZATION (FLOAT)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for single precision floating point values.
// \ingroup random
//
// This specialization of the Rand class creates random, single precision values in the range
// \f$ [0..1) \f$.
*/
template<>
class Rand<float>
{
 public:
   //**Generate functions**************************************************************************
   /*!\name Generate functions */
   //@{
   inline float generate() const;
   inline float generate( float min, float max ) const;
   //@}
   //**********************************************************************************************

   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( float& value ) const;
   inline void randomize( float& value, float min, float max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random single precision value in the range \f$ [0..1) \f$.
//
// \return The generated random single precision value.
//
// This function creates a random single precision value in the range \f$ [0..1) \f$.
*/
inline float Rand<float>::generate() const
{
   std::uniform_real_distribution<float> dist( 0.0, 1.0 );
   return dist( Random<RNG>::rng_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random single precision value in the range \f$ [min..max] \f$.
//
// \param min The smallest possible random value.
// \param max The largest possible random value.
// \return The generated random single precision value.
//
// This function creates a random single precision number in the range \f$ [min..max] \f$, where
// \a min must be smaller or equal to \a max. Note that this requirement is only checked in debug
// mode. In release mode, no check is performed to enforce the validity of the values. Therefore
// the returned value is undefined if \a min is larger than \a max.
*/
inline float Rand<float>::generate( float min, float max ) const
{
   BLAZE_INTERNAL_ASSERT( min <= max, "Invalid min/max values" );
   std::uniform_real_distribution<float> dist( min, max );
   return dist( Random<RNG>::rng_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of the given single precision variable to a value in in the range \f$ [0..1) \f$.
//
// \param value The variable to be randomized.
// \return void
//
// This function randomizes the given single precision variable to a value in the range \f$ [0..1) \f$.
*/
inline void Rand<float>::randomize( float& value ) const
{
   value = generate();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of the given single precision variable to a value in the range \f$ [min..max] \f$.
//
// \param value The variable to be randomized.
// \param min The smallest possible random value.
// \param max The largest possible random value.
// \return void
//
// This function randomizes the given single precision variable to a value in the range
// \f$ [min..max] \f$, where \a min must be smaller or equal to \a max. Note that this requirement
// is only checked in debug mode. In release mode, no check is performed to enforce the validity
// of the values. Therefore the returned value is undefined if \a min is larger than \a max.
*/
inline void Rand<float>::randomize( float& value, float min, float max ) const
{
   value = generate( min, max );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RAND SPECIALIZATION (DOUBLE)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for double precision floating point values.
// \ingroup random
//
// This specialization of the Rand class creates random, double precision values in the range
// \f$ [0..1) \f$.
*/
template<>
class Rand<double>
{
 public:
   //**Generate functions**************************************************************************
   /*!\name Generate functions */
   //@{
   inline double generate() const;
   inline double generate( double min, double max ) const;
   //@}
   //**********************************************************************************************

   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( double& value ) const;
   inline void randomize( double& value, double min, double max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random double precision value in the range \f$ [0..1) \f$.
//
// \return The generated random double precision value.
//
// This function creates a random double precision value in the range \f$ [0..1) \f$.
*/
inline double Rand<double>::generate() const
{
   std::uniform_real_distribution<double> dist( 0.0, 1.0 );
   return dist( Random<RNG>::rng_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random double precision value in the range \f$ [min..max] \f$.
//
// \param min The smallest possible random value.
// \param max The largest possible random value.
// \return The generated random double precision value.
//
// This function creates a random double precision number in the range \f$ [min..max] \f$, where
// \a min must be smaller or equal to \a max. Note that this requirement is only checked in debug
// mode. In release mode, no check is performed to enforce the validity of the values. Therefore
// the returned value is undefined if \a min is larger than \a max.
*/
inline double Rand<double>::generate( double min, double max ) const
{
   BLAZE_INTERNAL_ASSERT( min <= max, "Invalid min/max values" );
   std::uniform_real_distribution<double> dist( min, max );
   return dist( Random<RNG>::rng_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of the given double precision variable to a value in in the range \f$ [0..1) \f$.
//
// \param value The variable to be randomized.
// \return void
//
// This function randomizes the given double precision variable to a value in the range \f$ [0..1) \f$.
*/
inline void Rand<double>::randomize( double& value ) const
{
   value = generate();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of the given double precision variable to a value in the range \f$ [min..max] \f$.
//
// \param value The variable to be randomized.
// \param min The smallest possible random value.
// \param max The largest possible random value.
// \return void
//
// This function randomizes the given double precision variable to a value in the range
// \f$ [min..max] \f$, where \a min must be smaller or equal to \a max. Note that this requirement
// is only checked in debug mode. In release mode, no check is performed to enforce the validity
// of the values. Therefore the returned value is undefined if \a min is larger than \a max.
*/
inline void Rand<double>::randomize( double& value, double min, double max ) const
{
   value = generate( min, max );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RAND SPECIALIZATION (LONG DOUBLE)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for extended precision floating point values.
// \ingroup random
//
// This specialization of the Rand class creates random, extended precision values in the range
// \f$ [0..1) \f$.
*/
template<>
class Rand<long double>
{
 public:
   //**Generate functions**************************************************************************
   /*!\name Generate functions */
   //@{
   inline long double generate() const;
   inline long double generate( long double min, long double max ) const;
   //@}
   //**********************************************************************************************

   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( long double& value ) const;
   inline void randomize( long double& value, long double min, long double max ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random extended precision value in the range \f$ [0..1) \f$.
//
// \return The generated random extended precision value.
//
// This function creates a random extended precision value in the range \f$ [0..1) \f$.
*/
inline long double Rand<long double>::generate() const
{
   std::uniform_real_distribution<long double> dist( 0.0, 1.0 );
   return dist( Random<RNG>::rng_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random extended precision value in the range \f$ [min..max] \f$.
//
// \param min The smallest possible random value.
// \param max The largest possible random value.
// \return The generated random extended precision value.
//
// This function creates a random extended precision number in the range \f$ [min..max] \f$, where
// \a min must be smaller or equal to \a max. Note that this requirement is only checked in debug
// mode. In release mode, no check is performed to enforce the validity of the values. Therefore
// the returned value is undefined if \a min is larger than \a max.
*/
inline long double Rand<long double>::generate( long double min, long double max ) const
{
   BLAZE_INTERNAL_ASSERT( min <= max, "Invalid min/max values" );
   std::uniform_real_distribution<long double> dist( min, max );
   return dist( Random<RNG>::rng_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of the given extended precision variable to a value in in the range \f$ [0..1) \f$.
//
// \param value The variable to be randomized.
// \return void
//
// This function randomizes the given extended precision variable to a value in the range \f$ [0..1) \f$.
*/
inline void Rand<long double>::randomize( long double& value ) const
{
   value = generate();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of the given extended precision variable to a value in the range \f$ [min..max] \f$.
//
// \param value The variable to be randomized.
// \param min The smallest possible random value.
// \param max The largest possible random value.
// \return void
//
// This function randomizes the given extended precision variable to a value in the range
// \f$ [min..max] \f$, where \a min must be smaller or equal to \a max. Note that this requirement
// is only checked in debug mode. In release mode, no check is performed to enforce the validity
// of the values. Therefore the returned value is undefined if \a min is larger than \a max.
*/
inline void Rand<long double>::randomize( long double& value, long double min, long double max ) const
{
   value = generate( min, max );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RAND SPECIALIZATION (COMPLEX)
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for complex values.
// \ingroup random
//
// This specialization of the Rand class creates random, complex values.
*/
template< typename T >  // Type of the values
class Rand< complex<T> >
{
 public:
   //**Generate functions**************************************************************************
   /*!\name Generate functions */
   //@{
   inline const complex<T> generate() const;
   inline const complex<T> generate( const T& min, const T& max ) const;
   inline const complex<T> generate( const T& realmin, const T& realmax,
                                     const T& imagmin, const T& imagmax ) const;
   //@}
   //**********************************************************************************************

   //**Randomize functions*************************************************************************
   /*!\name Randomize functions */
   //@{
   inline void randomize( complex<T>& value ) const;
   inline void randomize( complex<T>& value, const T& min, const T& max ) const;
   inline void randomize( complex<T>& value, const T& realmin, const T& realmax,
                                             const T& imagmin, const T& imagmax ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random complex number.
//
// \return The generated random complex number.
//
// This function generates a random complex number, where both the real and the imaginary part
// are initialized with random values in the full range of the data type \a T.
*/
template< typename T >  // Type of the values
inline const complex<T> Rand< complex<T> >::generate() const
{
   Rand<T> tmp;
   return complex<T>( tmp.generate(), tmp.generate() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random complex number in the range \f$ [min..max] \f$.
//
// \param min The smallest possible random value.
// \param max The largest possible random value.
// \return The generated random complex number.
//
// This function generates a random complex number, where both the real and the imaginary part
// are initialized with random values in the range \f$ [min..max] \f$. Note that \a min must be
// smaller or equal to \a max. This requirement is only checked in debug mode. In release mode,
// no check is performed to enforce the validity of the values. Therefore the returned value is
// undefined if \a min is larger than \a max.
*/
template< typename T >  // Type of the values
inline const complex<T> Rand< complex<T> >::generate( const T& min, const T& max ) const
{
   Rand<T> tmp;
   return complex<T>( tmp.generate( min, max ), tmp.generate( min, max ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Generation of a random complex number.
//
// \param realmin The smallest possible random value for the real part.
// \param realmax The largest possible random value for the real part
// \param realmin The smallest possible random value for the imaginary part.
// \param realmax The largest possible random value for the imaginary part.
// \return The generated random complex number.
//
// This function creates a random, complex number, where the real part is in the range
// \f$ [realmin..realmax] \f$ and the imaginary part is in the range \f$ [imagmin..imagmax] \f$.
// \a realmin must be smaller or equal to \a realmax and \a imagmin must be smaller or equal to
// \a imagmax. These requirements are only checked in debug mode. In release mode, no check is
// performed to enforce the validity of the values. Therefore the returned value is undefined
// if \a realmin is larger than \a realmax or \a imagmin is larger than \a imagmax.
*/
template< typename T >  // Type of the values
inline const complex<T> Rand< complex<T> >::generate( const T& realmin, const T& realmax,
                                                      const T& imagmin, const T& imagmax ) const
{
   Rand<T> tmp;
   return complex<T>( tmp.generate( realmin, realmax ), tmp.generate( imagmin, imagmax ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a complex number.
//
// \param value The variable to be randomized.
// \return void
//
// This function randomizes the given complex number. Both the real and the imaginary part are
// initialized with random values in the full range of the data type \a T.
*/
template< typename T >  // Type of the values
inline void Rand< complex<T> >::randomize( complex<T>& value ) const
{
   value = generate();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a complex number to values in the range \f$ [min..max] \f$.
//
// \param value The variable to be randomized.
// \param min The smallest possible random value.
// \param max The largest possible random value.
// \return void
//
// This function randomizes the given complex number. Both the real and the imaginary part are
// initialized with random values in the range \f$ [min..max] \f$. Note that \a min must be
// smaller or equal to \a max. This requirement is only checked in debug mode. In release mode,
// no check is performed to enforce the validity of the values. Therefore the returned value is
// undefined if \a min is larger than \a max.
*/
template< typename T >  // Type of the values
inline void Rand< complex<T> >::randomize( complex<T>& value, const T& min, const T& max ) const
{
   value = generate( min, max );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Randomization of a complex number.
//
// \param value The variable to be randomized.
// \param realmin The smallest possible random value for the real part.
// \param realmax The largest possible random value for the real part
// \param realmin The smallest possible random value for the imaginary part.
// \param realmax The largest possible random value for the imaginary part.
// \return void
//
// This function randomizes the given complex number. The real part is set to a random value in
// the range \f$ [realmin..realmax] \f$ and the imaginary part is set to a value in the range
// \f$ [imagmin..imagmax] \f$. \a realmin must be smaller or equal to \a realmax and \a imagmin
// must be smaller or equal to \a imagmax. These requirements are only checked in debug mode.
// In release mode, no check is performed to enforce the validity of the values. Therefore the
// returned value is undefined if \a realmin is larger than \a realmax or \a imagmin is larger
// than \a imagmax.
*/
template< typename T >  // Type of the values
inline void Rand< complex<T> >::randomize( complex<T>& value, const T& realmin, const T& realmax,
                                                              const T& imagmin, const T& imagmax ) const
{
   value = generate( realmin, realmax, imagmin, imagmax );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RANDOM NUMBER FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Random number function.
// \ingroup random
//
// \return The generated random number.
//
// The rand() function returns a default random number depending on the given data type. In case
// of integral data types, the function returns a random number in the range \f$ [0..max] \f$,
// where \a max is the maximal value of the data type \a T. In case of floating point data types,
// the function returns a random number in the range \f$ [0..1) \f$. In case of complex data
// types, the function returns a random complex value where both the real and the imaginary part
// have been set according to the element type of the complex value (\f$ [0..max] \f$ for integral
// elements, \f$ [0..1) \f$ for floating point elements).
*/
template< typename T >  // Type of the random number
inline T rand()
{
   Rand<T> tmp;
   return tmp.generate();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Random number function.
// \ingroup random
//
// \param args The arguments for the random number generation.
// \return The generated random number.
//
// This rand() function creates a random number based on the given arguments \a args.
*/
template< typename T          // Type of the random number
        , typename... Args >  // Types of the optional arguments
inline T rand( Args&&... args )
{
   Rand<T> tmp;
   return tmp.generate( std::forward<Args>( args )... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Randomization of a given variable.
// \ingroup random
//
// \param value The variable to be randomized.
// \return void
//
// The randomize() function randomizes the given variable depending on its data type. In case
// of integral data types, the function returns a random number in the range \f$ [0..max] \f$,
// where \a max is the maximal value of the data type \a T. In case of floating point data types,
// the function returns a random number in the range \f$ [0..1) \f$. In case of complex data
// types, the function returns a random complex value where both the real and the imaginary part
// have been set according to the element type of the complex value (\f$ [0..max] \f$ for integral
// elements, \f$ [0..1) \f$ for floating point elements).
*/
template< typename T >  // Type of the random number
inline void randomize( T& value )
{
   Rand<T> tmp;
   tmp.randomize( value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Randomization of a given variable.
// \ingroup random
//
// \param value The value to be randomized.
// \param args The arguments for the random number generation.
// \return void
//
// This randomize() function randomizes the given variable based on the given arguments \a args.
*/
template< typename T          // Type of the random number
        , typename... Args >  // Types of the optional arguments
inline void randomize( T& value, Args&&... args )
{
   Rand<T> tmp;
   tmp.randomize( value, std::forward<Args>( args )... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the default random seed.
// \ingroup random
//
// \return The default random seed.
*/
inline uint32_t defaultSeed()
{
   static const uint32_t seed = static_cast<uint32_t>( std::time(0) );
   return seed;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current seed of the random number generator.
// \ingroup random
//
// \return The current seed of the random number generator.
*/
inline uint32_t getSeed()
{
   return Random<RNG>::seed_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the seed of the random number generator.
// \ingroup random
//
// \param seed The new seed for the random number generator.
// \return void
//
// This function can be used to set the seed for the random number generation in order to
// create a reproducible series of random numbers.
*/
inline void setSeed( uint32_t seed )
{
   Random<RNG>::seed_ = seed;
   Random<RNG>::rng_.seed( seed );
}
//*************************************************************************************************

} // namespace blaze

#endif
