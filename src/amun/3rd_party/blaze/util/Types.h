//=================================================================================================
/*!
//  \file blaze/util/Types.h
//  \brief Header file for basic type definitions
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

#ifndef _BLAZE_UTIL_TYPES_H_
#define _BLAZE_UTIL_TYPES_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cstddef>
#include <cstdint>


namespace blaze {

//=================================================================================================
//
//  TYPE DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\class blaze::size_t
// \brief Size type of the Blaze library.
// \ingroup util
*/
using std::size_t;
//*************************************************************************************************


//*************************************************************************************************
/*!\class blaze::ptrdiff_t
// \brief Pointer difference type of the Blaze library.
// \ingroup util
*/
using std::ptrdiff_t;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Byte data type of the Blaze library.
// \ingroup util
//
// The \a byte data type is guaranteed to be an integral data type of size 1.
*/
using byte_t = unsigned char;
//*************************************************************************************************


//*************************************************************************************************
/*!\class blaze::int8_t
// \brief 8-bit signed integer type of the Blaze library.
// \ingroup util
*/
using std::int8_t;
//*************************************************************************************************


//*************************************************************************************************
/*!\class blaze::uint8_t
// \brief 8-bit unsigned integer type of the Blaze library.
// \ingroup util
*/
using std::uint8_t;
//*************************************************************************************************


//*************************************************************************************************
/*!\class blaze::int16_t
// \brief 16-bit signed integer type of the Blaze library.
// \ingroup util
*/
using std::int16_t;
//*************************************************************************************************


//*************************************************************************************************
/*!\class blaze::uint16_t
// \brief 16-bit unsigned integer type of the Blaze library.
// \ingroup util
*/
using std::uint16_t;
//*************************************************************************************************


//*************************************************************************************************
/*!\class blaze::int32_t
// \brief 32-bit signed integer type of the Blaze library.
// \ingroup util
*/
using std::int32_t;
//*************************************************************************************************


//*************************************************************************************************
/*!\class blaze::uint32_t
// \brief 32-bit unsigned integer type of the Blaze library.
// \ingroup util
*/
using std::uint32_t;
//*************************************************************************************************


//*************************************************************************************************
/*!\class blaze::int64_t
// \brief 64-bit signed integer type of the Blaze library.
// \ingroup util
*/
using std::int64_t;
//*************************************************************************************************


//*************************************************************************************************
/*!\class blaze::uint64_t
// \brief 64-bit unsigned integer type of the Blaze library.
// \ingroup util
*/
using std::uint64_t;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The largest available signed integer data type.
// \ingroup util
*/
using large_t = int64_t;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The largest available unsigned integer data type.
// \ingroup util
*/
using ularge_t = uint64_t;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Unsigned integer data type for integral IDs.
// \ingroup util
*/
using id_t = ularge_t;
//*************************************************************************************************

} // namespace blaze

#endif
