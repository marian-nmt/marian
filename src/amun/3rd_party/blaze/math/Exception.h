//=================================================================================================
/*!
//  \file blaze/math/Exception.h
//  \brief Header file for the exception macros of the math module
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

#ifndef _BLAZE_MATH_EXCEPTION_H_
#define _BLAZE_MATH_EXCEPTION_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/Exception.h>




//=================================================================================================
//
//  EXCEPTION MACROS
//
//=================================================================================================

//*************************************************************************************************
/*!\def BLAZE_THROW_DIVISION_BY_ZERO
// \brief Macro for the emission of an exception on detection of a division by zero.
// \ingroup math
//
// This macro encapsulates the default way of \b Blaze to throw an exception on detection of
// a division by zero. Also, since it may be desirable to replace the type of exception by a
// custom exception type this macro provides an opportunity to customize the behavior.
//
// The macro excepts a single argument, which specifies the message of the exception:

   \code
   #define BLAZE_THROW_DIVISION_BY_ZERO( MESSAGE ) \
      BLAZE_THROW_RUNTIME_ERROR( MESSAGE )
   \endcode

// In order to customize the type of exception all that needs to be done is to define the macro
// prior to including any \a Blaze header file. This will override the \b Blaze default behavior.
// The following example demonstrates this by replacing \a std::runtime_error by a custom
// exception type:

   \code
   class DivisionByZero
   {
    public:
      DivisionByZero();
      explicit DivisionByZero( const std::string& message );
      // ...
   };

   #define BLAZE_THROW_DIVISION_BY_ZERO( MESSAGE ) \
      throw DivisionByZero( MESSAGE )

   #include <blaze/Blaze.h>
   \endcode

// \note It is recommended to define the macro such that a subsequent semicolon is required!
//
// \warning This macro is provided with the intention to assist in adapting \b Blaze to special
// conditions and environments. However, the customization of the type of exception via this
// macro may have an effect on the library. Thus be advised to use the macro with due care!
*/
#ifndef BLAZE_THROW_DIVISION_BY_ZERO
#  define BLAZE_THROW_DIVISION_BY_ZERO( MESSAGE ) BLAZE_THROW_RUNTIME_ERROR( MESSAGE )
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\def BLAZE_THROW_LAPACK_ERROR
// \brief Macro for the emission of an exception on detection of a LAPACK error.
// \ingroup math
//
// This macro encapsulates the default way of \b Blaze to throw an exception when encountering
// a LAPACK error (for instance when trying to invert a singular matrix). Also, since it may be
// desirable to replace the type of exception by a custom exception type this macro provides an
// opportunity to customize the behavior.
//
// The macro excepts a single argument, which specifies the message of the exception:

   \code
   #define BLAZE_THROW_LAPACK_ERROR( MESSAGE ) \
      BLAZE_THROW_RUNTIME_ERROR( MESSAGE )
   \endcode

// In order to customize the type of exception all that needs to be done is to define the macro
// prior to including any \a Blaze header file. This will override the \b Blaze default behavior.
// The following example demonstrates this by replacing \a std::runtime_error by a custom
// exception type:

   \code
   class LapackError
   {
    public:
      LapackError();
      explicit LapackError( const std::string& message );
      // ...
   };

   #define BLAZE_THROW_LAPACK_ERROR( MESSAGE ) \
      throw LapackError( MESSAGE )

   #include <blaze/Blaze.h>
   \endcode

// \note It is recommended to define the macro such that a subsequent semicolon is required!
//
// \warning This macro is provided with the intention to assist in adapting \b Blaze to special
// conditions and environments. However, the customization of the type of exception via this
// macro may have an effect on the library. Thus be advised to use the macro with due care!
*/
#ifndef BLAZE_THROW_LAPACK_ERROR
#  define BLAZE_THROW_LAPACK_ERROR( MESSAGE ) BLAZE_THROW_RUNTIME_ERROR( MESSAGE )
#endif
//*************************************************************************************************

#endif
