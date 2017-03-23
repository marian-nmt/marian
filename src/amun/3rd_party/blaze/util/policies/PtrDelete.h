//=================================================================================================
/*!
//  \file blaze/util/policies/PtrDelete.h
//  \brief Header file for the PtrDelete policy classes.
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

#ifndef _BLAZE_UTIL_POLICIES_PTRDELETE_H_
#define _BLAZE_UTIL_POLICIES_PTRDELETE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <boost/checked_delete.hpp>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Pointer-delete policy class.
// \ingroup util
//
// The PtrDelete policy functor class applies a delete operation to the given argument. Note that
// the delete operation is NOT permitted for inclomplete types (i.e. declared but undefined data
// types). The attempt to apply a PtrDelete functor to a pointer to an object of incomplete type
// results in a compile time error!
*/
struct PtrDelete
{
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   template< typename Type >
   inline void operator()( Type ptr ) const;
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Implementation of the pointer-delete policy.
//
// \param ptr The pointer to delete.
// \return void
//
// This function applies a standard delete operation to the given argument. Note that the delete
// operation is NOT permitted for inclomplete types (i.e. declared but undefined data types). The
// attempt to use this function for a pointer to an object of incomplete type results in a compile
// time error!
*/
template< typename Type >
inline void PtrDelete::operator()( Type ptr ) const
{
   boost::checked_delete( ptr );
}
//*************************************************************************************************

} // namespace blaze

#endif
