//=================================================================================================
/*!
//  \file blaze/util/NonCopyable.h
//  \brief Base class for non-copyable class instances
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

#ifndef _BLAZE_UTIL_NONCOPYABLE_H_
#define _BLAZE_UTIL_NONCOPYABLE_H_


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Base class for non-copyable class instances.
// \ingroup util
//
// The NonCopyable class is intended to work as a base class for non-copyable classes. Both the
// copy constructor and the copy assignment operator are explicitly deleted in order to prohibit
// copy operations of the derived classes.\n
//
// \note It is not necessary to publicly derive from this class. It is sufficient to derive
// privately to prevent copy operations on the derived class.

   \code
   class A : private NonCopyable
   { ... };
   \endcode
*/
class NonCopyable
{
 protected:
   //**Constructor and destructor******************************************************************
   /*!\name Constructor and destructor */
   //@{
   inline NonCopyable()  {}  //!< Default constructor for the NonCopyable class.
   inline ~NonCopyable() {}  //!< Destructor of the NonCopyable class.
   //@}
   //**********************************************************************************************

   //**Copy constructor and copy assignment operator***********************************************
   /*!\name Copy constructor and copy assignment operator */
   //@{
   NonCopyable( const NonCopyable& ) = delete;             //!< Copy constructor (explicitly deleted)
   NonCopyable& operator=( const NonCopyable& ) = delete;  //!< Copy assignment operator (explicitly deleted)
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blaze

#endif
