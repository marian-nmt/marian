//=================================================================================================
/*!
//  \file blaze/math/Epsilon.h
//  \brief Numerical epsilon value for floating point data types
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

#ifndef _BLAZE_MATH_EPSILON_H_
#define _BLAZE_MATH_EPSILON_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/constraints/FloatingPoint.h>
#include <blaze/util/Limits.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Negative epsilon value for floating point data types.
// \ingroup math
//
// The NegativeEpsilon class is a wrapper class around the functionality of the blaze::Limits
// class. It represents the negative smallest difference between two values of any floating
// point data type. In order to assign a negative epsilon value, the Epsilon class can be
// implicitly converted to the three built-in floating point data types float, double and
// long double.
//
// \note The NegativeEpsilon class is a helper class for the Epsilon class. It cannot be
// instantiated on its own, but can only be used by the Epsilon class.
*/
template< typename E >  // Positive epsilon type
class NegativeEpsilon
{
 public:
   //**Type definitions****************************************************************************
   typedef E  PositiveType;  //!< The positive epsilon type.
   //**********************************************************************************************

 private:
   //**Constructor*********************************************************************************
   /*!\name Constructor */
   //@{
   explicit inline NegativeEpsilon();
   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

 public:
   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Unary plus/minus operators******************************************************************
   /*!\name Unary plus/minus operators */
   //@{
   inline const NegativeEpsilon& operator+() const;
   inline const PositiveType     operator-() const;
   //@}
   //**********************************************************************************************

   //**Conversion operator*************************************************************************
   /*!\name Conversion operator */
   //@{
   template< typename T >
   inline operator const T() const;
   //@}
   //**********************************************************************************************

 private:
   //**Forbidden operations************************************************************************
   /*!\name Forbidden operations */
   //@{
   NegativeEpsilon& operator=( const NegativeEpsilon& );  //!< Copy assignment operator (private & undefined)
   void* operator&() const;                               //!< Address operator (private & undefined)
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   friend class Epsilon;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The default constructor of the NegativeEpsilon class.
*/
template< typename E >  // Positive epsilon type
inline NegativeEpsilon<E>::NegativeEpsilon()
{}
//*************************************************************************************************




//=================================================================================================
//
//  UNARY PLUS/MINUS OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the negative epsilon value for all floating point data types.
//
// \return The negative epsilon value.
*/
template< typename E >  // Positive epsilon type
inline const NegativeEpsilon<E>& NegativeEpsilon<E>::operator+() const
{
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the positive epsilon value for all floating point data types.
//
// \return The positive epsilon value.
*/
template< typename E >  // Positive epsilon type
inline const typename NegativeEpsilon<E>::PositiveType NegativeEpsilon<E>::operator-() const
{
   return PositiveType();
}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion operator to the required floating point data type.
//
// The conversion operator returns the negative epsilon value for the floating point
// data type \a T.
*/
template< typename E >  // Positive epsilon type
template< typename T >  // Floating point data type
inline NegativeEpsilon<E>::operator const T() const
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::epsilon();
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name NegativeEpsilon operators */
//@{
template< typename E, typename T >
inline bool operator==( const NegativeEpsilon<E>& lhs, const T& rhs );

template< typename E, typename T >
inline bool operator==( const T& lhs, const NegativeEpsilon<E>& rhs );

template< typename E, typename T >
inline bool operator!=( const NegativeEpsilon<E>& lhs, const T& rhs );

template< typename E, typename T >
inline bool operator!=( const T& lhs, const NegativeEpsilon<E>& rhs );

template< typename E, typename T >
inline bool operator<( const NegativeEpsilon<E>& lhs, const T& rhs );

template< typename E, typename T >
inline bool operator<( const T& lhs, const NegativeEpsilon<E>& rhs );

template< typename E, typename T >
inline bool operator>( const NegativeEpsilon<E>& lhs, const T& rhs );

template< typename E, typename T >
inline bool operator>( const T& lhs, const NegativeEpsilon<E>& rhs );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between a NegativeEpsilon object and a floating point value.
// \ingroup math
//
// \param rhs The right-hand side floating point value.
// \return \a true if the value is equal to the negative epsilon, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename E    // Positive epsilon type
        , typename T >  // Floating point data type
inline bool operator==( const NegativeEpsilon<E>& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::epsilon() == rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between a floating point value and a NegativeEpsilon object.
// \ingroup math
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is equal to the negative epsilon, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename E    // Positive epsilon type
        , typename T >  // Floating point data type
inline bool operator==( const T& lhs, const NegativeEpsilon<E>& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs == -Limits<T>::epsilon();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between a NegativeEpsilon object and a floating point value.
// \ingroup math
//
// \param rhs The right-hand side floating point value.
// \return \a true if the value is unequal to the negative epsilon, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename E    // Positive epsilon type
        , typename T >  // Floating point data type
inline bool operator!=( const NegativeEpsilon<E>& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::epsilon() != rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between a floating point value and a NegativeEpsilon object.
// \ingroup math
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is unequal to the negative epsilon, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename E    // Positive epsilon type
        , typename T >  // Floating point data type
inline bool operator!=( const T& lhs, const NegativeEpsilon<E>& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs != -Limits<T>::epsilon();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-than comparison between a NegativeEpsilon object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the value is greater than the negative epsilon, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename E    // Positive epsilon type
        , typename T >  // Floating point data type
inline bool operator<( const NegativeEpsilon<E>& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::epsilon() < rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-than comparison between a floating point value and a NegativeEpsilon object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is smaller than the negative epsilon, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename E    // Positive epsilon type
        , typename T >  // Floating point data type
inline bool operator<( const T& lhs, const NegativeEpsilon<E>& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs < -Limits<T>::epsilon();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-than comparison between a NegativeEpsilon object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the value is smaller than the negative epsilon, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename E    // Positive epsilon type
        , typename T >  // Floating point data type
inline bool operator>( const NegativeEpsilon<E>& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::epsilon() > rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-than comparison between a floating point value and a NegativeEpsilon object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is greater than the negative epsilon, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename E    // Positive epsilon type
        , typename T >  // Floating point data type
inline bool operator>( const T& lhs, const NegativeEpsilon<E>& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs > -Limits<T>::epsilon();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between a NegativeEpsilon object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the value is greater than or equal to the negative epsilon, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename E    // Positive epsilon type
        , typename T >  // Floating point data type
inline bool operator<=( const NegativeEpsilon<E>& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::epsilon() <= rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between a floating point value and a NegativeEpsilon object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is smaller than or equal to the negative epsilon, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename E    // Positive epsilon type
        , typename T >  // Floating point data type
inline bool operator<=( const T& lhs, const NegativeEpsilon<E>& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs <= -Limits<T>::epsilon();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-or-equal-than comparison between a NegativeEpsilon object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the value is smaller than or equal to the negative epsilon, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename E    // Positive epsilon type
        , typename T >  // Floating point data type
inline bool operator>=( const NegativeEpsilon<E>& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::epsilon() >= rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between a floating point value and a NegativeEpsilon object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is greater than or equal to the negative epsilon, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename E    // Positive epsilon type
        , typename T >  // Floating point data type
inline bool operator>=( const T& lhs, const NegativeEpsilon<E>& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs >= -Limits<T>::epsilon();
}
//*************************************************************************************************








//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Numerical epsilon value for floating point data types.
// \ingroup math
//
// The Epsilon class is a wrapper class around the functionality of the blaze::Limits class.
// It represents the smallest difference between two values of any floating point data type.
// In order to assign an epsilon value, the Epsilon class can be implicitly converted to the
// three built-in floating point data types float, double and long double.\n
// In order to handle epsilon values conveniently, the global Epsilon instance blaze::epsilon
// is provided, which can be used wherever a floating point data type is required.

   \code
   float f  =  epsilon;  // Assigns the positive epsilon for single precision values
   double d = -epsilon;  // Assigns the negative epsilon for double precision values
   \endcode
*/
class Epsilon
{
 public:
   //**Type definitions****************************************************************************
   typedef NegativeEpsilon<Epsilon>  NegativeType;  //!< The negative epsilon type.
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\name Constructor */
   //@{
   explicit inline Epsilon();
   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Unary plus/minus operators******************************************************************
   /*!\name Unary plus/minus operators */
   //@{
   inline const Epsilon&     operator+() const;
   inline const NegativeType operator-() const;
   //@}
   //**********************************************************************************************

   //**Conversion operators************************************************************************
   /*!\name Conversion operators */
   //@{
   template< typename T >
   inline operator const T() const;
   //@}
   //**********************************************************************************************

 private:
   //**Forbidden operations************************************************************************
   /*!\name Forbidden operations */
   //@{
   Epsilon& operator=( const Epsilon& );  //!< Copy assignment operator (private & undefined)
   void* operator&() const;               //!< Address operator (private & undefined)
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The default constructor of the Epsilon class.
*/
inline Epsilon::Epsilon()
{}
//*************************************************************************************************




//=================================================================================================
//
//  UNARY PLUS/MINUS OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the positive epsilon value for all floating point data types.
//
// \return The positive epsilon value.
*/
inline const Epsilon& Epsilon::operator+() const
{
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the negative epsilon value for all floating point data types.
//
// \return The negative epsilon value.
*/
inline const Epsilon::NegativeType Epsilon::operator-() const
{
   return NegativeType();
}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion operator to the required floating point data type.
//
// The conversion operator returns the smallest possible difference between values of the
// floating point data type \a T.
*/
template< typename T >
inline Epsilon::operator const T() const
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::epsilon();
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Epsilon operators */
//@{
template< typename T >
inline bool operator==( const Epsilon& lhs, const T& rhs );

template< typename T >
inline bool operator==( const T& lhs, const Epsilon& rhs );

template< typename T >
inline bool operator!=( const Epsilon& lhs, const T& rhs );

template< typename T >
inline bool operator!=( const T& lhs, const Epsilon& rhs );

template< typename T >
inline bool operator<( const Epsilon& lhs, const T& rhs );

template< typename T >
inline bool operator<( const T& lhs, const Epsilon& rhs );

template< typename T >
inline bool operator>( const Epsilon& lhs, const T& rhs );

template< typename T >
inline bool operator>( const T& lhs, const Epsilon& rhs );

template< typename T >
inline bool operator<=( const Epsilon& lhs, const T& rhs );

template< typename T >
inline bool operator<=( const T& lhs, const Epsilon& rhs );

template< typename T >
inline bool operator>=( const Epsilon& lhs, const T& rhs );

template< typename T >
inline bool operator>=( const T& lhs, const Epsilon& rhs );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between an Epsilon object and a floating point value.
// \ingroup math
//
// \param rhs The right-hand side floating point value.
// \return \a true if the floating point value is equal to epsilon, \a false if not.
//
// This operator works only for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >
inline bool operator==( const Epsilon& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::epsilon() == rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between a floating point value and an Epsilon object.
// \ingroup math
//
// \param lhs The left-hand side floating point value.
// \return \a true if the floating point value is equal to epsilon, \a false if not.
//
// This operator works only for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >
inline bool operator==( const T& lhs, const Epsilon& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs == Limits<T>::epsilon();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between an Epsilon object and a floating point value.
// \ingroup math
//
// \param rhs The right-hand side floating point value.
// \return \a true if the floating point value is unequal to epsilon, \a false if not.
//
// This operator works only for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >
inline bool operator!=( const Epsilon& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::epsilon() != rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between a floating point value and an Epsilon object.
// \ingroup math
//
// \param lhs The left-hand side floating point value.
// \return \a true if the floating point value is unequal to epsilon, \a false if not.
//
// This operator works only for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >
inline bool operator!=( const T& lhs, const Epsilon& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs != Limits<T>::epsilon();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-than comparison between an Epsilon object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the floating point value is greater than epsilon, \a false if not.
//
// This operator works only for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >
inline bool operator<( const Epsilon& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::epsilon() < rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-than comparison between a floating point value and an Epsilon object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the floating point value is smaller than epsilon, \a false if not.
//
// This operator works only for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >
inline bool operator<( const T& lhs, const Epsilon& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs < Limits<T>::epsilon();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-than comparison between an Epsilon object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the floating point value is smaller than epsilon, \a false if not.
//
// This operator works only for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >
inline bool operator>( const Epsilon& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::epsilon() > rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-than comparison between a floating point value and an Epsilon object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the floating point value is greater than epsilon, \a false if not.
//
// This operator works only for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >
inline bool operator>( const T& lhs, const Epsilon& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs > Limits<T>::epsilon();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between an Epsilon object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the floating point value is greater than or equal to epsilon, \a false if not.
//
// This operator works only for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >
inline bool operator<=( const Epsilon& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::epsilon() <= rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between a floating point value and an Epsilon object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the floating point value is smaller than or equal to epsilon, \a false if not.
//
// This operator works only for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >
inline bool operator<=( const T& lhs, const Epsilon& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs <= Limits<T>::epsilon();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-or-equal-than comparison between an Epsilon object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the floating point value is smaller than or equal to epsilon, \a false if not.
//
// This operator works only for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >
inline bool operator>=( const Epsilon& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::epsilon() >= rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between a floating point value and an Epsilon object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the floating point value is greater than or equal to epsilon, \a false if not.
//
// This operator works only for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >
inline bool operator>=( const T& lhs, const Epsilon& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs >= Limits<T>::epsilon();
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL EPSILON VALUE
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Global Epsilon instance.
// \ingroup math
//
// The blaze::epsilon instance can be used wherever a floating point data type is expected.
// It is implicitly converted to the corresponding floating point data type and represents
// the smallest possible difference between two values of the according data type.
*/
const Epsilon epsilon;
//*************************************************************************************************

} // namespace blaze

#endif
