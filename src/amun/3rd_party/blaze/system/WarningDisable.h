//=================================================================================================
/*!
//  \file blaze/system/WarningDisable.h
//  \brief Deactivation of compiler specific warnings
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

#ifndef _BLAZE_SYSTEM_WARNINGDISABLE_H_
#define _BLAZE_SYSTEM_WARNINGDISABLE_H_


//=================================================================================================
//
//  MICROSOFT VISUAL STUDIO WARNINGS
//
//=================================================================================================

#if defined(_MSC_VER) && (_MSC_VER >= 1400)

   // Disables a 'deprecated' warning for some standard library functions. This warning
   // is emitted when you use some perfectly conforming library functions in a perfectly
   // correct way, and also by some of Microsoft's own standard library code. For more
   // information about this particular warning, see
   // http://msdn.microsoft.com/en-us/library/ttcz0bys(VS.80).aspx
#  pragma warning(disable:4996)

   // Disables a warning for a this pointer that is passed to a base class in the constructor
   // initializer list.
#  pragma warning(disable:4355)

   // Disables the warning for ignored C++ exception specifications.
#  pragma warning(disable:4290)

#endif




//=================================================================================================
//
//  INTEL WARNINGS
//
//=================================================================================================

#if defined(__INTEL_COMPILER) || defined(__ICL)

   // Disables a 'deprecated' warning for some standard library functions.
#  pragma warning(disable:1786)

#endif

#endif
