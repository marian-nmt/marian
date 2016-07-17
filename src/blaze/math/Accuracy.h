//=================================================================================================
/*!
//  \file blaze/math/Accuracy.h
//  \brief Computation accuracy for floating point data types
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

#ifndef _BLAZE_MATH_ACCURACY_H_
#define _BLAZE_MATH_ACCURACY_H_


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
/*!\brief Negative computation accuracy for floating point data types.
// \ingroup math
//
// The NegativeAccuracy class is a wrapper class around the functionality of the blaze::Limits
// class. It represents the negative computation accuracy of the Blaze library for any floating
// point data type. In order to assign a negative accuracy value, the NegativeAccuracy class can
// be implicitly converted to the three built-in floating point data types float, double and long
// double.
//
// \note The NegativeAccuracy class is a helper class for the Accuracy class. It cannot be
// instantiated on its own, but can only be used by the Accuracy class.
*/
template< typename A >  // Positive accuracy type
class NegativeAccuracy
{
 public:
   //**Type definitions****************************************************************************
   typedef A  PositiveType;  //!< The positive accuracy type.
   //**********************************************************************************************

 private:
   //**Constructor*********************************************************************************
   /*!\name Constructor */
   //@{
   explicit inline NegativeAccuracy();
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
   inline const NegativeAccuracy& operator+() const;
   inline const PositiveType      operator-() const;
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
   NegativeAccuracy& operator=( const NegativeAccuracy& );  //!< Copy assignment operator (private & undefined)
   void* operator&() const;                                 //!< Address operator (private & undefined)
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   friend class Accuracy;
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
/*!\brief The default constructor of the NegativeAccuracy class.
*/
template< typename A >  // Positive accuracy type
inline NegativeAccuracy<A>::NegativeAccuracy()
{}
//*************************************************************************************************




//=================================================================================================
//
//  UNARY PLUS/MINUS OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the negative computation accuracy for all floating point data types.
//
// \return The negative computation accuracy.
*/
template< typename A >  // Positive accuracy type
inline const NegativeAccuracy<A>& NegativeAccuracy<A>::operator+() const
{
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the positive computation accuracy for all floating point data types.
//
// \return The positive computation accuracy.
*/
template< typename A >  // Positive accuracy type
inline const typename NegativeAccuracy<A>::PositiveType NegativeAccuracy<A>::operator-() const
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
// The conversion operator returns the negative computation accuracy for the floating point
// data type \a T.
*/
template< typename A >  // Positive accuracy type
template< typename T >  // Floating point data type
inline NegativeAccuracy<A>::operator const T() const
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::accuracy();
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Accuracy operators */
//@{
template< typename A, typename T >
inline bool operator==( const NegativeAccuracy<A>& lhs, const T& rhs );

template< typename A, typename T >
inline bool operator==( const T& lhs, const NegativeAccuracy<A>& rhs );

template< typename A, typename T >
inline bool operator!=( const NegativeAccuracy<A>& lhs, const T& rhs );

template< typename A, typename T >
inline bool operator!=( const T& lhs, const NegativeAccuracy<A>& rhs );

template< typename A, typename T >
inline bool operator<( const NegativeAccuracy<A>& lhs, const T& rhs );

template< typename A, typename T >
inline bool operator<( const T& lhs, const NegativeAccuracy<A>& rhs );

template< typename A, typename T >
inline bool operator>( const NegativeAccuracy<A>& lhs, const T& rhs );

template< typename A, typename T >
inline bool operator>( const T& lhs, const NegativeAccuracy<A>& rhs );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between a NegativeAccuracy object and a floating point value.
// \ingroup math
//
// \param rhs The right-hand side floating point value.
// \return \a true if the value is equal to the negative accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename A    // Positive accuracy type
        , typename T >  // Floating point data type
inline bool operator==( const NegativeAccuracy<A>& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::accuracy() == rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between a floating point value and a NegativeAccuracy object.
// \ingroup math
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is equal to the negative accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename A    // Positive accuracy type
        , typename T >  // Floating point data type
inline bool operator==( const T& lhs, const NegativeAccuracy<A>& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs == -Limits<T>::accuracy();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between a NegativeAccuracy object and a floating point value.
// \ingroup math
//
// \param rhs The right-hand side floating point value.
// \return \a true if the value is unequal to the negative accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename A    // Positive accuracy type
        , typename T >  // Floating point data type
inline bool operator!=( const NegativeAccuracy<A>& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::accuracy() != rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between a floating point value and a NegativeAccuracy object.
// \ingroup math
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is unequal to the negative accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename A    // Positive accuracy type
        , typename T >  // Floating point data type
inline bool operator!=( const T& lhs, const NegativeAccuracy<A>& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs != -Limits<T>::accuracy();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-than comparison between a NegativeAccuracy object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the value is greater than the negative accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename A    // Positive accuracy type
        , typename T >  // Floating point data type
inline bool operator<( const NegativeAccuracy<A>& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::accuracy() < rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-than comparison between a floating point value and a NegativeAccuracy object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is smaller than the negative accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename A    // Positive accuracy type
        , typename T >  // Floating point data type
inline bool operator<( const T& lhs, const NegativeAccuracy<A>& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs < -Limits<T>::accuracy();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-than comparison between a NegativeAccuracy object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the value is smaller than the negative accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename A    // Positive accuracy type
        , typename T >  // Floating point data type
inline bool operator>( const NegativeAccuracy<A>& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::accuracy() > rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-than comparison between a floating point value and a NegativeAccuracy object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is greater than the negative accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename A    // Positive accuracy type
        , typename T >  // Floating point data type
inline bool operator>( const T& lhs, const NegativeAccuracy<A>& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs > -Limits<T>::accuracy();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between a NegativeAccuracy object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the value is greater than or equal to the negative accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename A    // Positive accuracy type
        , typename T >  // Floating point data type
inline bool operator<=( const NegativeAccuracy<A>& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::accuracy() <= rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between a floating point value and a NegativeAccuracy object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is smaller than or equal to the negative accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename A    // Positive accuracy type
        , typename T >  // Floating point data type
inline bool operator<=( const T& lhs, const NegativeAccuracy<A>& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs <= -Limits<T>::accuracy();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-or-equal-than comparison between a NegativeAccuracy object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the value is smaller than or equal to the negative accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename A    // Positive accuracy type
        , typename T >  // Floating point data type
inline bool operator>=( const NegativeAccuracy<A>& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return -Limits<T>::accuracy() >= rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between a floating point value and a NegativeAccuracy object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is greater than or equal to the negative accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename A    // Positive accuracy type
        , typename T >  // Floating point data type
inline bool operator>=( const T& lhs, const NegativeAccuracy<A>& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs >= -Limits<T>::accuracy();
}
//*************************************************************************************************








//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Computation accuracy for floating point data types.
// \ingroup math
//
// The Accuracy class is a wrapper class around the functionality of the blaze::Limits class.
// It represents the computation accuracy of the Blaze library for any floating point data
// type. In order to assign an accuracy value, the Accuracy class can be implicitly converted
// to the three built-in floating point data types float, double and long double.\n
// In order to handle accuracy values conveniently, the global Accuracy instance blaze::accuracy
// is provided, which can be used wherever a floating point data value is required.

   \code
   float f  =  accuracy;  // Assigns the positive computation accuracy for single precision values
   double d = -accuracy;  // Assigns the negative computation accuracy for double precision values
   \endcode
*/
class Accuracy
{
 public:
   //**Type definitions****************************************************************************
   typedef NegativeAccuracy<Accuracy>  NegativeType;  //!< The negated accuracy type.
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\name Constructor */
   //@{
   explicit inline Accuracy();
   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Unary plus/minus operators******************************************************************
   /*!\name Unary plus/minus operators */
   //@{
   inline const Accuracy&    operator+() const;
   inline const NegativeType operator-() const;
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
   Accuracy& operator=( const Accuracy& );  //!< Copy assignment operator (private & undefined)
   void* operator&() const;                 //!< Address operator (private & undefined)
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
/*!\brief The default constructor of the Accuracy class.
*/
inline Accuracy::Accuracy()
{}
//*************************************************************************************************




//=================================================================================================
//
//  UNARY PLUS/MINUS OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the positive computation accuracy for all floating point data types.
//
// \return The positive computation accuracy.
*/
inline const Accuracy& Accuracy::operator+() const
{
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the negative computation accuracy for all floating point data types.
//
// \return The negative computation accuracy.
*/
inline const Accuracy::NegativeType Accuracy::operator-() const
{
   return NegativeType();
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
// The conversion operator returns the computation accuracy for the floating point data
// type \a T.
*/
template< typename T >  // Floating point data type
inline Accuracy::operator const T() const
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::accuracy();
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Accuracy operators */
//@{
template< typename T >
inline bool operator==( const Accuracy& lhs, const T& rhs );

template< typename T >
inline bool operator==( const T& lhs, const Accuracy& rhs );

template< typename T >
inline bool operator!=( const Accuracy& lhs, const T& rhs );

template< typename T >
inline bool operator!=( const T& lhs, const Accuracy& rhs );

template< typename T >
inline bool operator<( const Accuracy& lhs, const T& rhs );

template< typename T >
inline bool operator<( const T& lhs, const Accuracy& rhs );

template< typename T >
inline bool operator>( const Accuracy& lhs, const T& rhs );

template< typename T >
inline bool operator>( const T& lhs, const Accuracy& rhs );

template< typename T >
inline bool operator<=( const Accuracy& lhs, const T& rhs );

template< typename T >
inline bool operator<=( const T& lhs, const Accuracy& rhs );

template< typename T >
inline bool operator>=( const Accuracy& lhs, const T& rhs );

template< typename T >
inline bool operator>=( const T& lhs, const Accuracy& rhs );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between an Accuracy object and a floating point value.
// \ingroup math
//
// \param rhs The right-hand side floating point value.
// \return \a true if the floating point value is equal to the accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >  // Floating point data type
inline bool operator==( const Accuracy& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::accuracy() == rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between a floating point value and an Accuracy object.
// \ingroup math
//
// \param lhs The left-hand side floating point value.
// \return \a true if the floating point value is equal to the accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >  // Floating point data type
inline bool operator==( const T& lhs, const Accuracy& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs == Limits<T>::accuracy();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between an Accuracy object and a floating point value.
// \ingroup math
//
// \param rhs The right-hand side floating point value.
// \return \a true if the floating point value is unequal to the accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >  // Floating point data type
inline bool operator!=( const Accuracy& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::accuracy() != rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between a floating point value and an Accuracy object.
// \ingroup math
//
// \param lhs The left-hand side floating point value.
// \return \a true if the floating point value is unequal to the accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >  // Floating point data type
inline bool operator!=( const T& lhs, const Accuracy& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs != Limits<T>::accuracy();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-than comparison between an Accuracy object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the floatin point value is greater than the accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >  // Floating point data type
inline bool operator<( const Accuracy& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::accuracy() < rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-than comparison between a floating point value and an Accuracy object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the floating point value is smaller than the accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >  // Floating point data type
inline bool operator<( const T& lhs, const Accuracy& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs < Limits<T>::accuracy();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-than comparison between an Accuracy object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the floating point value is smaller than the accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >  // Floating point data type
inline bool operator>( const Accuracy& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::accuracy() > rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-than comparison between a floating point value and an Accuracy object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the floating point value is greater than the accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >  // Floating point data type
inline bool operator>( const T& lhs, const Accuracy& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs > Limits<T>::accuracy();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between an Accuracy object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the floating point value is greater than or equal to the accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >  // Floating point data type
inline bool operator<=( const Accuracy& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::accuracy() <= rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between a floating point value and an Accuracy object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is smaller than or equal to the accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >  // Floating point data type
inline bool operator<=( const T& lhs, const Accuracy& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs <= Limits<T>::accuracy();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-or-equal-than comparison between an Accuracy object and a floating point value.
//
// \param rhs The right-hand side floating point value.
// \return \a true if the value is smaller than or equal to the accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >  // Floating point data type
inline bool operator>=( const Accuracy& /*lhs*/, const T& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return Limits<T>::accuracy() >= rhs;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between a floating point value and an Accuracy object.
//
// \param lhs The left-hand side floating point value.
// \return \a true if the value is greater than or equal to the accuracy, \a false if not.
//
// This operator exclusively works for floating point data types. The attempt to compare any
// integral data type or user-defined class types will result in a compile time error.
*/
template< typename T >  // Floating point data type
inline bool operator>=( const T& lhs, const Accuracy& /*rhs*/ )
{
   BLAZE_CONSTRAINT_MUST_BE_FLOATING_POINT_TYPE( T );
   return lhs >= Limits<T>::accuracy();
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL ACCURACY VALUE
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Global Accuracy instance.
// \ingroup math
//
// The blaze::accuracy instance can be used wherever a floating point data type is expected.
// It is implicitly converted to the corresponding floating point data type and represents
// the computation accuracy of the Blaze library for the according data type.
*/
const Accuracy accuracy;
//*************************************************************************************************

} // namespace blaze

#endif
