//=================================================================================================
/*!
//  \file blaze/math/simd/SIMDPack.h
//  \brief Header file for the SIMDPack base class
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

#ifndef _BLAZE_MATH_SIMD_SIMDPACK_H_
#define _BLAZE_MATH_SIMD_SIMDPACK_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/system/Inline.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Base class for all SIMD data types.
// \ingroup simd
//
// The SIMDPack class template is a base class for all SIMD data types within the Blaze library.
// It provides an abstraction from the actual type of the SIMD pack, but enables a conversion
// back to this type via the 'Curiously Recurring Template Pattern' (CRTP).
*/
template< typename T >  // Type of the SIMD pack
struct SIMDPack
{
   //**Non-const conversion operator***************************************************************
   /*!\brief Conversion operator for non-constant vectors.
   //
   // \return Reference of the actual type of the vector.
   */
   BLAZE_ALWAYS_INLINE T& operator~() noexcept {
      return *static_cast<T*>( this );
   }
   //**********************************************************************************************

   //**Const conversion operators******************************************************************
   /*!\brief Conversion operator for constant vectors.
   //
   // \return Const reference of the actual type of the vector.
   */
   BLAZE_ALWAYS_INLINE const T& operator~() const noexcept {
      return *static_cast<const T*>( this );
   }
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blaze

#endif
