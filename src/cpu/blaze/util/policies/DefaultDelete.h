//=================================================================================================
/*!
//  \file blaze/util/policies/DefaultDelete.h
//  \brief Header file for the DefaultDelete policy classes.
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

#ifndef _BLAZE_UTIL_POLICIES_DEFAULTDELETE_H_
#define _BLAZE_UTIL_POLICIES_DEFAULTDELETE_H_


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
/*!\brief Default C++ deletion policy class.
// \ingroup util
//
// The DefaultDelete deletion policy is the standard delete for resources allocated via the new
// operator. It uses delete or array delete (depending on the template argument) to free the
// resource:

   \code
   class Resource { ... };

   DefaultDelete<Resource> ptrDelete       // Uses delete to free resources
   DefaultDelete<Resource[]> arrayDelete;  // Uses array delete to free resources
   \endcode

// Note the explicit use of empty array bounds to configure DefaultDelete to use array delete
// instead of delete. Also note that the delete operation is NOT permitted for incomplete types
// (i.e. declared but undefined data types). The attempt to apply a DefaultDelete functor to a
// pointer or array to an object of incomplete type results in a compile time error!
*/
template< typename Type >
struct DefaultDelete
{
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline void operator()( Type* ptr ) const;
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Implementation of the default delete policy.
//
// \param ptr The pointer to delete.
// \return void
//
// This function frees the given pointer resource via delete. Note that the delete operation
// is NOT permitted for incomplete types (i.e. declared but undefined data types). The attempt
// to use this function for a pointer to an object of incomplete type results in a compile time
// error!
*/
template< typename Type >
inline void DefaultDelete<Type>::operator()( Type* ptr ) const
{
   boost::checked_delete( ptr );
}
//*************************************************************************************************




//=================================================================================================
//
//  SPECIALIZATION FOR ARRAYS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the DefaultDelete class template for arrays.
// \ingroup util
//
// This specialization of the DefaultDelete class template uses array delete to free the
// allocated resource.
*/
template< typename Type >
struct DefaultDelete<Type[]>
{
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline void operator()( Type* ptr ) const;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Implementation of the default array delete policy.
//
// \param ptr The pointer to delete.
// \return void
//
// This function frees the given array resource via array delete. Note that the delete operation
// is NOT permitted for incomplete types (i.e. declared but undefined data types). The attempt
// to use this function for a pointer to an object of incomplete type results in a compile time
// error!
*/
template< typename Type >
inline void DefaultDelete<Type[]>::operator()( Type* ptr ) const
{
   boost::checked_array_delete( ptr );
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
