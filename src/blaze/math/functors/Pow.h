//=================================================================================================
/*!
//  \file blaze/math/functors/Pow.h
//  \brief Header file for the Pow functor
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

#ifndef _BLAZE_MATH_FUNCTORS_POW_H_
#define _BLAZE_MATH_FUNCTORS_POW_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <utility>
#include <blaze/math/constraints/SIMDPack.h>
#include <blaze/math/shims/Pow.h>
#include <blaze/math/simd/Pow.h>
#include <blaze/math/simd/SIMDTrait.h>
#include <blaze/math/typetraits/HasSIMDPow.h>
#include <blaze/system/Inline.h>
#include <blaze/util/constraints/Numeric.h>
#include <blaze/util/typetraits/IsSame.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Generic wrapper for the pow() function.
// \ingroup functors
*/
template< typename ET >  // Type of the exponent
struct Pow
{
 public:
   //**Type definitions****************************************************************************
   typedef SIMDTrait_<ET>  SIMDET;  //!< The SIMD exponent type.
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Constructor of the Pow functor.
   //
   // \param exp The exponent.
   */
   explicit inline Pow( ET exp )
      : exp_    ( exp )          // The scalar exponent
      , simdExp_( set( exp_ ) )  // The SIMD exponent
   {}
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns the result of the pow() function for the given object/value.
   //
   // \param a The given object/value.
   // \return The result of the pow() function for the given object/value.
   */
   template< typename T >
   BLAZE_ALWAYS_INLINE auto operator()( const T& a ) const
      -> decltype( pow( a, std::declval<ET>() ) )
   {
      return pow( a, exp_ );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether SIMD is enabled for the specified data type \a T.
   //
   // \return \a true in case SIMD is enabled for the data type \a T, \a false if not.
   */
   template< typename T >
   static constexpr bool simdEnabled() { return IsSame<T,ET>::value && HasSIMDPow<T>::value; }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns the result of the pow() function for the given SIMD vector.
   //
   // \param a The given SIMD vector.
   // \return The result of the pow() function for the given SIMD vector.
   */
   template< typename T >
   BLAZE_ALWAYS_INLINE auto load( const T& a ) const
      -> decltype( pow( a, std::declval<SIMDET>() ) )
   {
      BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK( T );
      return pow( a, simdExp_ );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   ET     exp_;      //!< The scalar exponent.
   SIMDET simdExp_;  //!< The SIMD exponent.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( ET );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blaze

#endif
