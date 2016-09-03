//=================================================================================================
/*!
//  \file blaze/util/Template.h
//  \brief Header file for nested template disabiguation
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

#ifndef _BLAZE_UTIL_TEMPLATE_H_
#define _BLAZE_UTIL_TEMPLATE_H_


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Compiler specific patch for nested template disambiguation.
// \ingroup util
//
// The BLAZE_TEMPLATE is a patch for the Microsoft Visual C++ compiler that does not correctly
// parse definitions of nested templates of the following form:

   \code
   template< typename T >
   class Alloc {
    public:
      ...
      template< typename Other >
      class rebind {
       public:
         typedef Alloc<Other> other;
      };
      ...
   };

   typedef Alloc<int>  AI;
   typedef AI::template rebind<double>::other  Other;  // Compilation error with Visual C++
   \endcode

// In order to circumvent this compilation error, the BLAZE_TEMPLATE macro should be used
// instead the \a template keyword:

   \code
   ...
   typedef AI::BLAZE_TEMPLATE rebind<double>::other  Other;  // No compilation errors
   \endcode
*/
#if defined(_MSC_VER)
#  define BLAZE_TEMPLATE
#else
#  define BLAZE_TEMPLATE template
#endif
/*! \endcond */
//*************************************************************************************************

#endif
