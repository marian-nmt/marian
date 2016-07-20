//=================================================================================================
/*!
//  \file blaze/util/typetraits/AddPointer.h
//  \brief Header file for the AddPointer type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_ADDPOINTER_H_
#define _BLAZE_UTIL_TYPETRAITS_ADDPOINTER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <type_traits>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Addition of a top level pointer.
// \ingroup type_traits
//
// The AddPointer type trait adds a top level pointer to the given type \a T. It has the same
// effect as \c blaze::RemovePointer<T>::Type*.

   \code
   blaze::AddPointer<int>::Type        // Results in 'int*'
   blaze::AddPointer<int const>::Type  // Results in 'int const*'
   blaze::AddPointer<int*>::Type       // Results in 'int**'
   blaze::AddPointer<int*&>::Type      // Results in 'int**'
   \endcode
*/
template< typename T >
struct AddPointer
{
 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   typedef typename std::add_pointer<T>::type  Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the AddPointer type trait.
// \ingroup type_traits
//
// The AddPointer_ alias declaration provides a convenient shortcut to access the nested \a Type
// of the AddPointer class template. For instance, given the type \a T the following two type
// definitions are identical:

   \code
   using Type1 = typename AddPointer<T>::Type;
   using Type2 = AddPointer_<T>;
   \endcode
*/
template< typename T >
using AddPointer_ = typename AddPointer<T>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
