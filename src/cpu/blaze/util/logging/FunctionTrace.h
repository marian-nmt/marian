//=================================================================================================
/*!
//  \file blaze/util/logging/FunctionTrace.h
//  \brief Header file for the FunctionTrace class
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

#ifndef _BLAZE_UTIL_LOGGING_FUNCTIONTRACE_H_
#define _BLAZE_UTIL_LOGGING_FUNCTIONTRACE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <new>
#include <string>
#include <blaze/system/Logging.h>
#include <blaze/system/Signature.h>
#include <blaze/util/NonCopyable.h>


namespace blaze {

namespace logging {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief RAII object for function tracing.
// \ingroup logging
//
// The FunctionTrace class is an auxiliary helper class for the tracing of function calls. It
// is implemented as a wrapper around the Logger class and is responsible for the atomicity of
// the logging operations of trace information.
*/
class FunctionTrace : private NonCopyable
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   FunctionTrace( const std::string& file, const std::string& function );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~FunctionTrace();
   //@}
   //**********************************************************************************************

   //**Forbidden operations************************************************************************
   /*!\name Forbidden operations */
   //@{
   void* operator new  ( std::size_t ) = delete;
   void* operator new[]( std::size_t ) = delete;
   void* operator new  ( std::size_t, const std::nothrow_t& ) noexcept = delete;
   void* operator new[]( std::size_t, const std::nothrow_t& ) noexcept = delete;

   void operator delete  ( void* ) noexcept = delete;
   void operator delete[]( void* ) noexcept = delete;
   void operator delete  ( void*, const std::nothrow_t& ) noexcept = delete;
   void operator delete[]( void*, const std::nothrow_t& ) noexcept = delete;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::string file_;      //!< The file name the traced function is contained in.
   std::string function_;  //!< The name of the traced function.
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  BLAZE_FUNCTION_TRACE MACRO
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Function trace macro.
// \ingroup logging
//
// This macro can be used to reliably trace function calls. In case function tracing is
// activated, the traces are logged via the Logger class. The following, short example
// demonstrates how the function trace macro is used:

   \code
   int main( int argc, char** argv )
   {
      BLAZE_FUNCTION_TRACE;

      // ...
   }
   \endcode

// The macro should be used as the very first statement inside the function in order to
// guarantee that logging the function trace is the very first and last action of the
// function call.\n
// Function tracing can be enabled or disabled via the BLAZE_USE_FUNCTION_TRACES macro.
// If function tracing is activated, the resulting log will contain trace information of
// the following form:

   \code
   [TRACE   ][000:00:00] + Entering function 'int main()' in file 'TraceDemo.cpp'
   [TRACE   ][000:00:10] - Leaving function 'int main()' in file 'TraceDemo.cpp'
   \endcode

// In case function tracing is deactivated, all function trace functionality is completely
// removed from the code, i.e. no function traces are logged and no overhead results from
// the BLAZE_FUNCTION_TRACE macro.
*/
#if BLAZE_USE_FUNCTION_TRACES
#  define BLAZE_FUNCTION_TRACE \
   blaze::logging::FunctionTrace BLAZE_FUNCTION_TRACE_OBJECT( __FILE__, BLAZE_SIGNATURE )
#else
#  define BLAZE_FUNCTION_TRACE
#endif
//*************************************************************************************************

} // namespace logging

} // namespace blaze

#endif
