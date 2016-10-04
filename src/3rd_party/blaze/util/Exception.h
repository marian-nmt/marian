//=================================================================================================
/*!
//  \file blaze/util/Exception.h
//  \brief Header file for exception macros
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

#ifndef _BLAZE_UTIL_EXCEPTION_H_
#define _BLAZE_UTIL_EXCEPTION_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <stdexcept>




//=================================================================================================
//
//  EXCEPTION MACROS
//
//=================================================================================================

//*************************************************************************************************
/*!\def BLAZE_THROW
// \brief Macro for the error reporting mechanism of the \b Blaze library.
// \ingroup util
//
// This macro encapsulates the default, general way of the \b Blaze library to report errors of
// any kind by throwing an exception. Also, since under certain conditions and environments it
// may be desirable to replace exceptions by a different error reporting mechanism this macro
// provides an opportunity to customize the error reporting approach.
//
// The macro excepts a single argument, which specifies the exception to be thrown:

   \code
   #define BLAZE_THROW( EXCEPTION ) \
      throw EXCEPTION
   \endcode

// In order to customize the error reporing mechanism all that needs to be done is to define
// the macro prior to including any \a Blaze header file. This will cause the \b Blaze specific
// mechanism to be overridden. The following example demonstrates this by replacing exceptions
// by a call to a \a log() function and a direct call to abort:

   \code
   #define BLAZE_THROW( EXCEPTION ) \
      log( "..." ); \
      abort()

   #include <blaze/Blaze.h>
   \endcode

// \note It is possible to execute several statements instead of executing a single statement to
// throw an exception. Also note that it is recommended to define the macro such that a subsequent
// semicolon is required!
//
// \warning This macro is provided with the intention to assist in adapting \b Blaze to special
// conditions and environments. However, the customization of the error reporting mechanism via
// this macro can have a significant effect on the library. Thus be advised to use the macro
// with due care!
*/
#ifndef BLAZE_THROW
#  define BLAZE_THROW( EXCEPTION ) throw EXCEPTION
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\def BLAZE_THROW_BAD_ALLOC
// \brief Macro for the emission of a \a std::bad_alloc exception.
// \ingroup util
//
// This macro encapsulates the default way of \b Blaze to throw a \a std::bad_alloc exception.
// Also, since it may be desirable to replace the type of exception by a custom exception type
// this macro provides an opportunity to customize the behavior.

   \code
   #define BLAZE_THROW_BAD_ALLOC \
      BLAZE_THROW( std::bad_alloc() )
   \endcode

// In order to customize the type of exception all that needs to be done is to define the macro
// prior to including any \a Blaze header file. This will override the \b Blaze default behavior.
// The following example demonstrates this by replacing \a std::bad_alloc by a custom exception
// type:

   \code
   class BadAlloc
   {
    public:
      BadAlloc();
      // ...
   };

   #define BLAZE_THROW_BAD_ALLOC \
      throw BadAlloc()

   #include <blaze/Blaze.h>
   \endcode

// \note It is recommended to define the macro such that a subsequent semicolon is required!
//
// \warning This macro is provided with the intention to assist in adapting \b Blaze to special
// conditions and environments. However, the customization of the type of exception via this
// macro may have an effect on the library. Thus be advised to use the macro with due care!
*/
#ifndef BLAZE_THROW_BAD_ALLOC
#  define BLAZE_THROW_BAD_ALLOC BLAZE_THROW( std::bad_alloc() )
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\def BLAZE_THROW_LOGIC_ERROR
// \brief Macro for the emission of a \a std::logic_error exception.
// \ingroup util
//
// This macro encapsulates the default way of \b Blaze to throw a \a std::logic_error exception.
// Also, since it may be desirable to replace the type of exception by a custom exception type
// this macro provides an opportunity to customize the behavior.
//
// The macro excepts a single argument, which specifies the message of the exception:

   \code
   #define BLAZE_THROW_LOGIC_ERROR( MESSAGE ) \
      BLAZE_THROW( std::logic_error( MESSAGE ) )
   \endcode

// In order to customize the type of exception all that needs to be done is to define the macro
// prior to including any \a Blaze header file. This will override the \b Blaze default behavior.
// The following example demonstrates this by replacing \a std::logic_error by a custom exception
// type:

   \code
   class LogicError
   {
    public:
      LogicError();
      explicit LogicError( const std::string& message );
      // ...
   };

   #define BLAZE_THROW_LOGIC_ERROR( MESSAGE ) \
      throw LogicError( MESSAGE )

   #include <blaze/Blaze.h>
   \endcode

// \note It is recommended to define the macro such that a subsequent semicolon is required!
//
// \warning This macro is provided with the intention to assist in adapting \b Blaze to special
// conditions and environments. However, the customization of the type of exception via this
// macro may have an effect on the library. Thus be advised to use the macro with due care!
*/
#ifndef BLAZE_THROW_LOGIC_ERROR
#  define BLAZE_THROW_LOGIC_ERROR( MESSAGE ) BLAZE_THROW( std::logic_error( MESSAGE ) )
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\def BLAZE_THROW_INVALID_ARGUMENT
// \brief Macro for the emission of a \a std::invalid_argument exception.
// \ingroup util
//
// This macro encapsulates the default way of \b Blaze to throw a \a std::invalid_argument
// exception. Also, since it may be desirable to replace the type of exception by a custom
// exception type this macro provides an opportunity to customize the behavior.
//
// The macro excepts a single argument, which specifies the message of the exception:

   \code
   #define BLAZE_THROW_INVALID_ARGUMENT( MESSAGE ) \
      BLAZE_THROW( std::invalid_argument( MESSAGE ) )
   \endcode

// In order to customize the type of exception all that needs to be done is to define the macro
// prior to including any \a Blaze header file. This will override the \b Blaze default behavior.
// The following example demonstrates this by replacing \a std::invalid_argument by a custom
// exception type:

   \code
   class InvalidArgument
   {
    public:
      InvalidArgument();
      explicit InvalidArgument( const std::string& message );
      // ...
   };

   #define BLAZE_THROW_INVALID_ARGUMENT( MESSAGE ) \
      throw InvalidArgument( MESSAGE )

   #include <blaze/Blaze.h>
   \endcode

// \note It is recommended to define the macro such that a subsequent semicolon is required!
//
// \warning This macro is provided with the intention to assist in adapting \b Blaze to special
// conditions and environments. However, the customization of the type of exception via this
// macro may have an effect on the library. Thus be advised to use the macro with due care!
*/
#ifndef BLAZE_THROW_INVALID_ARGUMENT
#  define BLAZE_THROW_INVALID_ARGUMENT( MESSAGE ) BLAZE_THROW( std::invalid_argument( MESSAGE ) )
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\def BLAZE_THROW_LENGTH_ERROR
// \brief Macro for the emission of a \a std::length_error exception.
// \ingroup util
//
// This macro encapsulates the default way of \b Blaze to throw a \a std::length_error exception.
// Also, since it may be desirable to replace the type of exception by a custom exception type
// this macro provides an opportunity to customize the behavior.
//
// The macro excepts a single argument, which specifies the message of the exception:

   \code
   #define BLAZE_THROW_LENGTH_ERROR( MESSAGE ) \
      BLAZE_THROW( std::length_error( MESSAGE ) )
   \endcode

// In order to customize the type of exception all that needs to be done is to define the macro
// prior to including any \a Blaze header file. This will override the \b Blaze default behavior.
// The following example demonstrates this by replacing \a std::length_error by a custom
// exception type:

   \code
   class LengthError
   {
    public:
      LengthError();
      explicit LengthError( const std::string& message );
      // ...
   };

   #define BLAZE_THROW_LENGTH_ERROR( MESSAGE ) \
      throw LengthError( MESSAGE )

   #include <blaze/Blaze.h>
   \endcode

// \note It is recommended to define the macro such that a subsequent semicolon is required!
//
// \warning This macro is provided with the intention to assist in adapting \b Blaze to special
// conditions and environments. However, the customization of the type of exception via this
// macro may have an effect on the library. Thus be advised to use the macro with due care!
*/
#ifndef BLAZE_THROW_LENGTH_ERROR
#  define BLAZE_THROW_LENGTH_ERROR( MESSAGE ) BLAZE_THROW( std::length_error( MESSAGE ) )
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\def BLAZE_THROW_OUT_OF_RANGE
// \brief Macro for the emission of a \a std::out_of_range exception.
// \ingroup util
//
// This macro encapsulates the default way of \b Blaze to throw a \a std::out_of_range exception.
// Also, since it may be desirable to replace the type of exception by a custom exception type
// this macro provides an opportunity to customize the behavior.
//
// The macro excepts a single argument, which specifies the message of the exception:

   \code
   #define BLAZE_THROW_OUT_OF_RANGE( MESSAGE ) \
      BLAZE_THROW( std::out_of_range( MESSAGE ) )
   \endcode

// In order to customize the type of exception all that needs to be done is to define the macro
// prior to including any \a Blaze header file. This will override the \b Blaze default behavior.
// The following example demonstrates this by replacing \a std::out_of_range by a custom exception
// type:

   \code
   class OutOfRange
   {
    public:
      OutOfRange();
      explicit OutOfRange( const std::string& message );
      // ...
   };

   #define BLAZE_THROW_OUT_OF_RANGE( MESSAGE ) \
      throw OutOfRange( MESSAGE )

   #include <blaze/Blaze.h>
   \endcode

// \note It is recommended to define the macro such that a subsequent semicolon is required!
//
// \warning This macro is provided with the intention to assist in adapting \b Blaze to special
// conditions and environments. However, the customization of the type of exception via this
// macro may have an effect on the library. Thus be advised to use the macro with due care!
*/
#ifndef BLAZE_THROW_OUT_OF_RANGE
#  define BLAZE_THROW_OUT_OF_RANGE( MESSAGE ) BLAZE_THROW( std::out_of_range( MESSAGE ) )
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\def BLAZE_THROW_RUNTIME_ERROR
// \brief Macro for the emission of a \a std::runtime_error exception.
// \ingroup util
//
// This macro encapsulates the default way of \b Blaze to throw a \a std::runtime_error exception.
// Also, since it may be desirable to replace the type of exception by a custom exception type
// this macro provides an opportunity to customize the behavior.
//
// The macro excepts a single argument, which specifies the message of the exception:

   \code
   #define BLAZE_THROW_RUNTIME_ERROR( MESSAGE ) \
      BLAZE_THROW( std::runtime_error( MESSAGE ) )
   \endcode

// In order to customize the type of exception all that needs to be done is to define the macro
// prior to including any \a Blaze header file. This will override the \b Blaze default behavior.
// The following example demonstrates this by replacing \a std::runtime_error by a custom
// exception type:

   \code
   class RuntimeError
   {
    public:
      RuntimeError();
      explicit RuntimeError( const std::string& message );
      // ...
   };

   #define BLAZE_THROW_RUNTIME_ERROR( MESSAGE ) \
      throw RuntimeError( MESSAGE )

   #include <blaze/Blaze.h>
   \endcode

// \note It is recommended to define the macro such that a subsequent semicolon is required!
//
// \warning This macro is provided with the intention to assist in adapting \b Blaze to special
// conditions and environments. However, the customization of the type of exception via this
// macro may have an effect on the library. Thus be advised to use the macro with due care!
*/
#ifndef BLAZE_THROW_RUNTIME_ERROR
#  define BLAZE_THROW_RUNTIME_ERROR( MESSAGE ) BLAZE_THROW( std::runtime_error( MESSAGE ) )
#endif
//*************************************************************************************************

#endif
