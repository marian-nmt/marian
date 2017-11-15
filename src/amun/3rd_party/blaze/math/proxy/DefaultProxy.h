//=================================================================================================
/*!
//  \file blaze/math/proxy/DefaultProxy.h
//  \brief Header file for the DefaultProxy class
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

#ifndef _BLAZE_MATH_PROXY_DEFAULTPROXY_H_
#define _BLAZE_MATH_PROXY_DEFAULTPROXY_H_


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
/*!\brief Default proxy backend for built-in and alternate user-specific class types.
// \ingroup math
//
// The DefaultProxy class serves as a backend for the Proxy class. It is used in case the data
// type represented by the proxy is a built-in or alternate user-specific class type. This proxy
// does not augment the Proxy interface by any additional interface.
*/
template< typename PT    // Type of the proxy
        , typename RT >  // Type of the represented element
class DefaultProxy
{
 public:
   //**Conversion operators************************************************************************
   /*!\name Conversion operators */
   //@{
   BLAZE_ALWAYS_INLINE PT&       operator~();
   BLAZE_ALWAYS_INLINE const PT& operator~() const;
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion operator for non-constant proxies.
//
// \return Reference to the actual type of the proxy.
//
// This function provides a type-safe downcast to the actual type of the proxy.
*/
template< typename PT    // Type of the proxy
        , typename CT >  // Type of the complex number
BLAZE_ALWAYS_INLINE PT& DefaultProxy<PT,CT>::operator~()
{
   return *static_cast<PT*>( this );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator for constant proxies.
//
// \return Reference to the actual type of the proxy.
//
// This function provides a type-safe downcast to the actual type of the proxy.
*/
template< typename PT    // Type of the proxy
        , typename CT >  // Type of the complex number
BLAZE_ALWAYS_INLINE const PT& DefaultProxy<PT,CT>::operator~() const
{
   return *static_cast<const PT*>( this );
}
//*************************************************************************************************

} // namespace blaze

#endif
