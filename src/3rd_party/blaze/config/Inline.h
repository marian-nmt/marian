//=================================================================================================
/*!
//  \file blaze/config/Inline.h
//  \brief Configuration of the inline policy of the Blaze library
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


//*************************************************************************************************
/*!\brief Compilation switch for a strengthened inline keyword.
// \ingroup config
//
// The regular C++ \c inline keyword merely represents a hint to the compiler to inline a function.
// Due to that, when using the \c inline keyword for performance critical functions, one is at the
// mercy of the compiler to properly inline the functions. In order to improve the likelihood of
// a function being properly inlined the BLAZE_STRONG_INLINE keyword can be used. In contrast to
// the regular \c inline keyword, BLAZE_STRONG_INLINE uses platform-specific keywords and modifiers
// to improve the likelihood of a function being properly inlined. Please note, however, that even
// in case the platform-specific inline is used, there is no guarantee that a function is inlined
// (see for instance the http://msdn.microsoft.com/en-us/library/z8y1yy88.aspx).
//
// This compilation switch enables/disables the BLAZE_STRONG_INLINE keyword. When disabled, the
// keyword uses the regular \c inline keyword as fallback. Possible setting for the switch are:
//  - Deactivated: \b 0
//  - Activated  : \b 1
*/
#define BLAZE_USE_STRONG_INLINE 1
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for an enforced inline keyword.
// \ingroup config
//
// Although the BLAZE_STRONG_INLINE keyword improves the likelihood of a function being inlined it
// does not provide a 100% guarantee. Depending on the availability of an according keyword and/or
// modifier on a specific platform, this guarantee is provided by the BLAZE_ALWAYS_INLINE keyword,
// which uses platform-specific functionality to enforce the inlining of a function.
//
// This compilation switch enables/disables the BLAZE_ALWAYS_INLINE keyword. When disabled or in
// case the platform does not provide a keyword and/or modifier for a 100% inline guarantee, the
// BLAZE_ALWAYS_INLINE keyword uses the BLAZE_STRONG_INLINE keyword as fallback. Possible settings
// for the switch are:
//  - Deactivated: \b 0
//  - Activated  : \b 1
*/
#define BLAZE_USE_ALWAYS_INLINE 1
//*************************************************************************************************
