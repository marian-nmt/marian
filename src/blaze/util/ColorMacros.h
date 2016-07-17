//=================================================================================================
/*!
//  \file blaze/util/ColorMacros.h
//  \brief Header file for color macros
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

#ifndef _BLAZE_UTIL_COLORMACROS_H_
#define _BLAZE_UTIL_COLORMACROS_H_


//=================================================================================================
//
//  COLOR MACRO SWITCH
//
//=================================================================================================

//! pe color output mode.
/*! This mode triggers the color output macros. */
#define BLAZE_COLOR_OUTPUT 0




//=================================================================================================
//
//  COLOR MACRO DEFINITIONS
//
//=================================================================================================

#if BLAZE_COLOR_OUTPUT

//! Switches the text color to black in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_BLACK         "\033[0;30m"

//! Switches the text color to red in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_RED           "\033[0;31m"

//! Switches the text color to green in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_GREEN         "\033[0;32m"

//! Switches the text color to brown in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_BROWN         "\033[0;33m"

//! Switches the text color to blue in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_BLUE          "\033[0;34m"

//! Switches the text color to magenta in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_MAGENTA       "\033[0;35m"

//! Switches the text color to cyan in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_CYAN          "\033[0;36m"

//! Switches the text color to white in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_WHITE         "\033[0;37m"

//! Switches the text color to a light black in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTBLACK    "\033[1;30m"

//! Switches the text color to a light red in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTRED      "\033[1;31m"

//! Switches the text color to a light green in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTGREEN    "\033[1;32m"

//! Switches the text color to yellow in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_YELLOW        "\033[1;33m"

//! Switches the text color to a light blue in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTBLUE     "\033[1;34m"

//! Switches the text color to a light magenta in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTMAGENTA  "\033[1;35m"

//! Switches the text color to a light cyan in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTCYAN     "\033[1;36m"

//! Switches the text color to a light white in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTWHITE    "\033[1;37m"

//! Switches the text color back to the default color.
#define BLAZE_OLDCOLOR      "\033[0m"

#else

//! Switches the text color to black in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_BLACK         ""

//! Switches the text color to red in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_RED           ""

//! Switches the text color to green in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_GREEN         ""

//! Switches the text color to brown in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_BROWN         ""

//! Switches the text color to blue in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_BLUE          ""

//! Switches the text color to magenta in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_MAGENTA       ""

//! Switches the text color to cyan in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_CYAN          ""

//! Switches the text color to white in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_WHITE         ""

//! Switches the text color to a light black in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTBLACK    ""

//! Switches the text color to a light red in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTRED      ""

//! Switches the text color to a light green in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTGREEN    ""

//! Switches the text color to yellow in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_YELLOW        ""

//! Switches the text color to a light blue in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTBLUE     ""

//! Switches the text color to a light magenta in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTMAGENTA  ""

//! Switches the text color to a light cyan in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTCYAN     ""

//! Switches the text color to a light white in case the BLAZE_COLOR_OUTPUT macro is set.
#define BLAZE_LIGHTWHITE    ""

//! Switches the text color back to the default color.
#define BLAZE_OLDCOLOR      ""

#endif

#endif
