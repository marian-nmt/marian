//=================================================================================================
/*!
//  \file blaze/math/AlignmentFlag.h
//  \brief Header file for the alignment flag values
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

#ifndef _BLAZE_MATH_ALIGNMENTFLAG_H_
#define _BLAZE_MATH_ALIGNMENTFLAG_H_


namespace blaze {

//=================================================================================================
//
//  ALIGNMENT FLAG VALUES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Alignment flag for unaligned vectors and matrices.
// \ingroup math
//
// Via this flag it is possible to specify subvectors, submatrices, custom vectors and matrices
// as unaligned. The following example demonstrates the setup of an unaligned subvector:

   \code
   using blaze::columnVector;
   using blaze::unaligned;

   typedef blaze::DynamicVector<int,columnVector>  VectorType;

   VectorType v( 100UL );
   Subvector<VectorType,unaligned> sv = subvector<unaligned>( v, 10UL, 20UL );
   \endcode
*/
const bool unaligned = false;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alignment flag for aligned vectors and matrices.
// \ingroup math
//
// Via this flag it is possible to specify subvectors, submatrices, custom vectors and matrices
// as aligned. The following example demonstrates the setup of an aligned subvector:

   \code
   using blaze::columnVector;
   using blaze::aligned;

   typedef blaze::DynamicVector<int,columnVector>  VectorType;

   VectorType v( 100UL );
   Subvector<VectorType,aligned> sv = subvector<aligned>( v, 8UL, 32UL );
   \endcode
*/
const bool aligned = true;
//*************************************************************************************************

} // namespace blaze

#endif
