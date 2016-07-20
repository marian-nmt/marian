//=================================================================================================
/*!
//  \file blaze/math/PaddingFlag.h
//  \brief Header file for the padding flag values
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

#ifndef _BLAZE_MATH_PADDINGFLAG_H_
#define _BLAZE_MATH_PADDINGFLAG_H_


namespace blaze {

//=================================================================================================
//
//  PADDING FLAG VALUES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Padding flag for unpadded vectors and matrices.
// \ingroup math
//
// Via this flag it is possible to specify custom vectors and matrices as unpadded. The following
// example demonstrates the setup of an unaligned, unpadded custom row vector of size 7:

   \code
   using blaze::CustomVector;
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::columnVector;

   std::vector<int> vec( 7UL );
   CustomVector<int,unaligned,unpadded,columnVector> a( &vec[0], 7UL );
   \endcode
*/
const bool unpadded = false;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Padding flag for padded vectors and matrices.
// \ingroup math
//
// Via this flag it is possible to specify custom vectors and matrices as aligned. The following
// example demonstrates the setup of an aligned, padded custom row vector of size 7:

   \code
   using blaze::CustomVector;
   using blaze::ArrayDelete;
   using blaze::aligned;
   using blaze::padded;
   using blaze::columnVector;

   std::vector<int> vec( 16UL );
   CustomVector<int,unaligned,padded,columnVector> a( &vec[0], 7UL, 16UL );
   \endcode
*/
const bool padded = true;
//*************************************************************************************************

} // namespace blaze

#endif
