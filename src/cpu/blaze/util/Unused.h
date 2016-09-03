//=================================================================================================
/*!
//  \file blaze/util/Unused.h
//  \brief Header file for the UNUSED_PARAMETER function template
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

#ifndef _BLAZE_UTIL_UNUSED_H_
#define _BLAZE_UTIL_UNUSED_H_


namespace blaze {

//=================================================================================================
//
//  UNUSED_PARAMETER FUNCTION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Suppression of unused parameter warnings.
// \ingroup util
//
// \return void
//
// The UNUSED_PARAMETER function provides the functionality to suppress warnings about up to
// six unused parameters. Usually this problem occurs in case a parameter is given a name but
// is not used within the function:

   \code
   void f( int x )
   {}  // x is not used within f. This may result in an unused parameter warning.
   \endcode

// A possible solution is to keep the parameter unnamed:

   \code
   void f( int )
   {}  // No warning about unused parameter is issued
   \endcode

// However, there are situations where is approach is not possible, as for instance in case the
// variable must be documented via Doxygen. For these cases, the UNUSED_PARAMETER class can be
// used to suppress the warnings:

   \code
   void f( int x )
   {
      UNUSED_PARAMETER( x );  // Suppresses the unused parameter warnings
   }
   \endcode
*/
template< typename T1 >
inline void UNUSED_PARAMETER( const T1& )
{}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the UNUSED_PARAMETER function for two parameters.
// \ingroup util
//
// \return void
*/
template< typename T1, typename T2 >
inline void UNUSED_PARAMETER( const T1&, const T2& )
{}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the UNUSED_PARAMETER function for three parameters.
// \ingroup util
//
// \return void
*/
template< typename T1, typename T2, typename T3 >
inline void UNUSED_PARAMETER( const T1&, const T2&, const T3& )
{}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the UNUSED_PARAMETER function for four parameters.
// \ingroup util
//
// \return void
*/
template< typename T1, typename T2, typename T3, typename T4 >
inline void UNUSED_PARAMETER( const T1&, const T2&, const T3&, const T4& )
{}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the UNUSED_PARAMETER function for five parameters.
// \ingroup util
//
// \return void
*/
template< typename T1, typename T2, typename T3, typename T4, typename T5 >
inline void UNUSED_PARAMETER( const T1&, const T2&, const T3&, const T4&, const T5& )
{}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the UNUSED_PARAMETER function for six parameters.
// \ingroup util
//
// \return void
*/
template< typename T1, typename T2, typename T3, typename T4, typename T5, typename T6 >
inline void UNUSED_PARAMETER( const T1&, const T2&, const T3&, const T4&, const T5&, const T6& )
{}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
