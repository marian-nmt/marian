//=================================================================================================
/*!
//  \file blaze/math/constraints/StorageOrder.h
//  \brief Constraints on the storage order of matrix types
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

#ifndef _BLAZE_MATH_CONSTRAINTS_STORAGEORDER_H_
#define _BLAZE_MATH_CONSTRAINTS_STORAGEORDER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/typetraits/IsMatrix.h>
#include <blaze/math/typetraits/StorageOrder.h>


namespace blaze {

//=================================================================================================
//
//  MUST_BE_MATRIX_WITH_STORAGE_ORDER CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup math_constraints
//
// In case the given data type \a T is not a dense or sparse matrix type and in case the
// storage order of the given dense or sparse vector type \a T is not set to \a SO, a
// compilation error is created.
*/
#define BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER(T,SO) \
   static_assert( ::blaze::IsMatrix<T>::value && \
                  ::blaze::StorageOrder<T>::value == SO, "Invalid storage order detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MATRICES_MUST_HAVE_SAME_STORAGE_ORDER CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup math_constraints
//
// In case either of the two given data types \a T1 or \a T2 is not a matrix type and in case
// the storage order of both matrix types doesn't match, a compilation error is created.
*/
#define BLAZE_CONSTRAINT_MATRICES_MUST_HAVE_SAME_STORAGE_ORDER(T1,T2) \
   static_assert( ::blaze::IsMatrix<T1>::value && \
                  ::blaze::IsMatrix<T2>::value && \
                  ::blaze::StorageOrder<T1>::value == ::blaze::StorageOrder<T2>::value, "Invalid storage order failed" )
//*************************************************************************************************




//=================================================================================================
//
//  MATRICES_MUST_HAVE_DIFFERENT_STORAGE_ORDER CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup math_constraints
//
// In case either of the two given data types \a T1 or \a T2 is not a matrix type and in case
// the storage order of both matrix types does match, a compilation error is created.
*/
#define BLAZE_CONSTRAINT_MATRICES_MUST_HAVE_DIFFERENT_STORAGE_ORDER(T1,T2) \
   static_assert( ::blaze::IsMatrix<T1>::value && \
                  ::blaze::IsMatrix<T2>::value && \
                  ::blaze::StorageOrder<T1>::value != ::blaze::StorageOrder<T2>::value, "Invalid storage order detected" )
//*************************************************************************************************

} // namespace blaze

#endif
