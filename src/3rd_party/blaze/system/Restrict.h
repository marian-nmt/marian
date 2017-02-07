//=================================================================================================
/*!
//  \file blaze/system/Restrict.h
//  \brief System settings for the restrict keyword
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

#ifndef _BLAZE_SYSTEM_RESTRICT_H_
#define _BLAZE_SYSTEM_RESTRICT_H_


//=================================================================================================
//
//  RESTRICT SETTINGS
//
//=================================================================================================

#include <blaze/config/Restrict.h>




//=================================================================================================
//
//  RESTRICT KEYWORD
//
//=================================================================================================

//*************************************************************************************************
/*!\def BLAZE_RESTRICT
// \brief Platform dependent setup of the restrict keyword.
// \ingroup system
*/
#if BLAZE_USE_RESTRICT

// Intel compiler
#  if defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) || defined(__ECC)
#    define BLAZE_RESTRICT __restrict

// GNU compiler
#  elif defined(__GNUC__)
#    define BLAZE_RESTRICT __restrict

// Microsoft visual studio
#  elif defined(_MSC_VER)
#    define BLAZE_RESTRICT

// All other compilers
#  else
#    define BLAZE_RESTRICT

#  endif
#else
#  define BLAZE_RESTRICT
#endif
//*************************************************************************************************

#endif
