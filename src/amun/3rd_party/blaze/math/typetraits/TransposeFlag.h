//=================================================================================================
/*!
//  \file blaze/math/typetraits/TransposeFlag.h
//  \brief Header file for the TransposeFlag type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_TRANSPOSEFLAG_H_
#define _BLAZE_MATH_TYPETRAITS_TRANSPOSEFLAG_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/TransposeFlag.h>
#include <blaze/math/typetraits/IsRowVector.h>
#include <blaze/math/typetraits/IsVector.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Evaluation of the transpose flag of a given matrix type.
// \ingroup math_type_traits
//
// Via this type trait it is possible to evaluate the transpose flag of a given vector type. In
// case the given type is a row vector type the nested boolean \a value is set to \a rowVector,
// in case it is a column vector type it is set to \a columnVector. If the given type is not a
// vector type a compilation error is created.

   \code
   typedef blaze::DynamicVector<int,blaze::rowVector>     RowVector;
   typedef blaze::DynamicVector<int,blaze::columnVector>  ColumnVector;

   blaze::TransposeFlag<RowVector>::value     // Evaluates to blaze::rowVector
   blaze::TransposeFlag<ColumnVector>::value  // Evaluates to blaze::columnVector
   blaze::TransposeFlag<int>::value           // Compilation error!
   \endcode
*/
template< typename T, typename = EnableIf_< IsVector<T> > >
struct TransposeFlag : public BoolConstant< ( IsRowVector<T>::value ? rowVector : columnVector ) >
{};
//*************************************************************************************************

} // namespace blaze

#endif
