//=================================================================================================
/*!
//  \file blaze/util/TypeTraits.h
//  \brief Header file for all type traits
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

#ifndef _BLAZE_UTIL_TYPETRAITS_H_
#define _BLAZE_UTIL_TYPETRAITS_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/typetraits/AddConst.h>
#include <blaze/util/typetraits/AddCV.h>
#include <blaze/util/typetraits/AddPointer.h>
#include <blaze/util/typetraits/AddReference.h>
#include <blaze/util/typetraits/AddVolatile.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blaze/util/typetraits/All.h>
#include <blaze/util/typetraits/Any.h>
#include <blaze/util/typetraits/CommonType.h>
#include <blaze/util/typetraits/Decay.h>
#include <blaze/util/typetraits/Extent.h>
#include <blaze/util/typetraits/GetMemberType.h>
#include <blaze/util/typetraits/HasMember.h>
#include <blaze/util/typetraits/HasSize.h>
#include <blaze/util/typetraits/HaveSameSize.h>
#include <blaze/util/typetraits/IsArithmetic.h>
#include <blaze/util/typetraits/IsArray.h>
#include <blaze/util/typetraits/IsAssignable.h>
#include <blaze/util/typetraits/IsBaseOf.h>
#include <blaze/util/typetraits/IsBoolean.h>
#include <blaze/util/typetraits/IsBuiltin.h>
#include <blaze/util/typetraits/IsCharacter.h>
#include <blaze/util/typetraits/IsClass.h>
#include <blaze/util/typetraits/IsComplex.h>
#include <blaze/util/typetraits/IsComplexDouble.h>
#include <blaze/util/typetraits/IsComplexFloat.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/typetraits/IsConstructible.h>
#include <blaze/util/typetraits/IsConvertible.h>
#include <blaze/util/typetraits/IsDestructible.h>
#include <blaze/util/typetraits/IsDouble.h>
#include <blaze/util/typetraits/IsEmpty.h>
#include <blaze/util/typetraits/IsFloat.h>
#include <blaze/util/typetraits/IsFloatingPoint.h>
#include <blaze/util/typetraits/IsInteger.h>
#include <blaze/util/typetraits/IsIntegral.h>
#include <blaze/util/typetraits/IsLong.h>
#include <blaze/util/typetraits/IsLongDouble.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsObject.h>
#include <blaze/util/typetraits/IsPod.h>
#include <blaze/util/typetraits/IsPointer.h>
#include <blaze/util/typetraits/IsReference.h>
#include <blaze/util/typetraits/IsSame.h>
#include <blaze/util/typetraits/IsShort.h>
#include <blaze/util/typetraits/IsSigned.h>
#include <blaze/util/typetraits/IsUnion.h>
#include <blaze/util/typetraits/IsUnsigned.h>
#include <blaze/util/typetraits/IsValid.h>
#include <blaze/util/typetraits/IsVectorizable.h>
#include <blaze/util/typetraits/IsVoid.h>
#include <blaze/util/typetraits/IsVolatile.h>
#include <blaze/util/typetraits/MakeSigned.h>
#include <blaze/util/typetraits/MakeUnsigned.h>
#include <blaze/util/typetraits/Rank.h>
#include <blaze/util/typetraits/RemoveAllExtents.h>
#include <blaze/util/typetraits/RemoveConst.h>
#include <blaze/util/typetraits/RemoveCV.h>
#include <blaze/util/typetraits/RemoveExtent.h>
#include <blaze/util/typetraits/RemovePointer.h>
#include <blaze/util/typetraits/RemoveReference.h>
#include <blaze/util/typetraits/RemoveVolatile.h>

#endif
