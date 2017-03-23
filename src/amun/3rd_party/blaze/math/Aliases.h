//=================================================================================================
/*!
//  \file blaze/math/Aliases.h
//  \brief Header file for auxiliary alias declarations
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

#ifndef _BLAZE_MATH_ALIASES_H_
#define _BLAZE_MATH_ALIASES_H_


namespace blaze {

//=================================================================================================
//
//  ALIAS DECLARATION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Alias declaration for nested \c BaseType type definitions.
// \ingroup aliases
//
// The BaseType_ alias declaration provides a convenient shortcut to access the nested
// \a BaseType type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::BaseType;
   using Type2 = BaseType_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using BaseType_ = typename T::BaseType;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c CompositeType type definitions.
// \ingroup aliases
//
// The CompositeType_ alias declaration provides a convenient shortcut to access the nested
// \a CompositeType type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::CompositeType;
   using Type2 = CompositeType_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using CompositeType_ = typename T::CompositeType;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c ConstIterator type definitions.
// \ingroup aliases
//
// The ConstIterator_ alias declaration provides a convenient shortcut to access the nested
// \a ConstIterator type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::ConstIterator;
   using Type2 = ConstIterator_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using ConstIterator_ = typename T::ConstIterator;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c ConstPointer type definitions.
// \ingroup aliases
//
// The ConstPointer_ alias declaration provides a convenient shortcut to access the nested
// \a ConstPointer type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::ConstPointer;
   using Type2 = ConstPointer_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using ConstPointer_ = typename T::ConstPointer;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c ConstReference type definitions.
// \ingroup aliases
//
// The ConstReference_ alias declaration provides a convenient shortcut to access the nested
// \a ConstReference type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::ConstReference;
   using Type2 = ConstReference_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using ConstReference_ = typename T::ConstReference;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c ElementType type definitions.
// \ingroup aliases
//
// The ElementType_ alias declaration provides a convenient shortcut to access the nested
// \a ElementType type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::ElementType;
   using Type2 = ElementType_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using ElementType_ = typename T::ElementType;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c Iterator type definitions.
// \ingroup aliases
//
// The Iterator_ alias declaration provides a convenient shortcut to access the nested
// \a Iterator type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::Iterator;
   using Type2 = Iterator_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using Iterator_ = typename T::Iterator;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c LeftOperand type definitions.
// \ingroup aliases
//
// The LeftOperand_ alias declaration provides a convenient shortcut to access the nested
// \a LeftOperand type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::LeftOperand;
   using Type2 = LeftOperand_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using LeftOperand_ = typename T::LeftOperand;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c Operand type definitions.
// \ingroup aliases
//
// The Operand_ alias declaration provides a convenient shortcut to access the nested \a Operand
// type definition of the given type \a T. The following code example shows both ways to access
// the nested type definition:

   \code
   using Type1 = typename T::Operand;
   using Type2 = Operand_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using Operand_ = typename T::Operand;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c OppositeType type definitions.
// \ingroup aliases
//
// The OppositeType_ alias declaration provides a convenient shortcut to access the nested
// \a OppositeType type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::OppositeType;
   using Type2 = OppositeType_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using OppositeType_ = typename T::OppositeType;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c Pointer type definitions.
// \ingroup aliases
//
// The Pointer_ alias declaration provides a convenient shortcut to access the nested
// \a Pointer type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::Pointer;
   using Type2 = Pointer_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using Pointer_ = typename T::Pointer;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c Reference type definitions.
// \ingroup aliases
//
// The Reference_ alias declaration provides a convenient shortcut to access the nested
// \a Reference type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::Reference;
   using Type2 = Reference_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using Reference_ = typename T::Reference;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c RepresentedType type definitions.
// \ingroup aliases
//
// The RepresentedType_ alias declaration provides a convenient shortcut to access the nested
// \a RepresentedType type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::RepresentedType;
   using Type2 = RepresentedType_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using RepresentedType_ = typename T::RepresentedType;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c ResultType type definitions.
// \ingroup aliases
//
// The ResultType_ alias declaration provides a convenient shortcut to access the nested
// \a ResultType type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::ResultType;
   using Type2 = ResultType_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using ResultType_ = typename T::ResultType;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c ReturnType type definitions.
// \ingroup aliases
//
// The ReturnType_ alias declaration provides a convenient shortcut to access the nested
// \a ReturnType type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::ReturnType;
   using Type2 = ReturnType_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using ReturnType_ = typename T::ReturnType;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c RightOperand type definitions.
// \ingroup aliases
//
// The RightOperand_ alias declaration provides a convenient shortcut to access the nested
// \a RightOperand type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::RightOperand;
   using Type2 = RightOperand_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using RightOperand_ = typename T::RightOperand;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c SIMDType type definitions.
// \ingroup aliases
//
// The SIMDType_ alias declaration provides a convenient shortcut to access the nested
// \a SIMDType type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::SIMDType;
   using Type2 = SIMDType_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using SIMDType_ = typename T::SIMDType;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c TransposeType type definitions.
// \ingroup aliases
//
// The TransposeType_ alias declaration provides a convenient shortcut to access the nested
// \a TransposeType type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::TransposeType;
   using Type2 = TransposeType_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using TransposeType_ = typename T::TransposeType;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Alias declaration for nested \c ValueType type definitions.
// \ingroup aliases
//
// The ValueType_ alias declaration provides a convenient shortcut to access the nested
// \a ValueType type definition of the given type \a T. The following code example shows
// both ways to access the nested type definition:

   \code
   using Type1 = typename T::ValueType;
   using Type2 = ValueType_<T>;

   BLAZE_CONSTRAINT_MUST_BE_STRICTLY_SAME_TYPE( Type1, Type2 );
   \endcode
*/
template< typename T >
using ValueType_ = typename T::ValueType;
//*************************************************************************************************

} // namespace blaze

#endif
