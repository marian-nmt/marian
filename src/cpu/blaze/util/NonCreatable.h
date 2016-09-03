//=================================================================================================
/*!
//  \file blaze/util/NonCreatable.h
//  \brief Base class for non-creatable (static) classes
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

#ifndef _BLAZE_UTIL_NONCREATABLE_H_
#define _BLAZE_UTIL_NONCREATABLE_H_


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Base class for non-creatable (static) classes.
// \ingroup util
//
// The NonCreatable class is intended to work as a base class for non-creatable classes, i.e.
// classes that cannot be instantiated and exclusively offer static functions/data. Both the
// standard as well as the copy constructor and the copy assignment operator are declared
// private and left undefinded in order to prohibit the instantiation of objects of derived
// classes.\n
//
// \note It is not necessary to publicly derive from this class. It is sufficient to derive
// privately to prevent the instantiation of the derived class.

   \code
   class A : private NonCreatable
   { ... };
   \endcode
*/
class NonCreatable
{
 protected:
   //**Constructors and copy assignment operator***************************************************
   /*!\name Constructors and copy assignment operator */
   //@{
   NonCreatable() = delete;                                  //!< Constructor (explicitly deleted)
   NonCreatable( const NonCreatable& ) = delete;             //!< Copy constructor (explicitly deleted)
   NonCreatable& operator=( const NonCreatable& ) = delete;  //!< Copy assignment operator (explicitly deleted)
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blaze

#endif
