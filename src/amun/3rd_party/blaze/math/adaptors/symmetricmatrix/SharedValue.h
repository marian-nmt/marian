//=================================================================================================
/*!
//  \file blaze/math/adaptors/symmetricmatrix/SharedValue.h
//  \brief Header file for the SharedValue class
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

#ifndef _BLAZE_MATH_ADAPTORS_SYMMETRICMATRIX_SHAREDVALUE_H_
#define _BLAZE_MATH_ADAPTORS_SYMMETRICMATRIX_SHAREDVALUE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <boost/shared_ptr.hpp>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Volatile.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Value shared among several positions within a symmetric matrix.
// \ingroup symmetric_matrix
//
// The SharedValue class template represents a single value of a symmetric matrix that is shared
// among several positions within the symmetric matrix. Changes to the value of one position
// are therefore applied to all positions sharing the same value.
*/
template< typename Type >  // Type of the shared value
class SharedValue
{
 public:
   //**Type definitions****************************************************************************
   typedef Type         ValueType;       //!< Type of the shared value.
   typedef Type&        Reference;       //!< Reference to the shared value.
   typedef const Type&  ConstReference;  //!< Reference-to-const to the shared value.
   typedef Type*        Pointer;         //!< Pointer to the shared value.
   typedef const Type*  ConstPointer;    //!< Pointer-to-const to the shared value.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline SharedValue();
   explicit inline SharedValue( const Type& value );
   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Access operators****************************************************************************
   /*!\name Access operators */
   //@{
   inline Reference      operator* ();
   inline ConstReference operator* () const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline Pointer base() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   mutable boost::shared_ptr<Type> value_;  //!< The shared value.
   //@}
   //**********************************************************************************************

   //**Forbidden operations************************************************************************
   /*!\name Forbidden operations */
   //@{
   void* operator&() const;  //!< Address operator (private & undefined)
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE  ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST         ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE      ( Type );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Default constructor for a SharedValue.
*/
template< typename Type >  // Type of the shared value
inline SharedValue<Type>::SharedValue()
   : value_()  // The shared value
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a SharedValue.
//
// \param value The value to be shared.
//
// This constructor creates a shared value as a copy of the given value.
*/
template< typename Type >  // Type of the shared value
inline SharedValue<Type>::SharedValue( const Type& value )
   : value_( new Type( value ) )  // The shared value
{}
//*************************************************************************************************




//=================================================================================================
//
//  ACCESS OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Direct access to the shared value.
//
// \return Reference to the shared value.
*/
template< typename Type >  // Type of the shared value
inline typename SharedValue<Type>::Reference SharedValue<Type>::operator*()
{
   if( !value_ )
      value_.reset( new Type() );
   return *value_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Direct access to the shared value.
//
// \return Reference to the shared value.
*/
template< typename Type >  // Type of the shared value
inline typename SharedValue<Type>::ConstReference SharedValue<Type>::operator*() const
{
   if( !value_ )
      value_.reset( new Type() );
   return *value_;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Low-level access to the underlying, shared value.
//
// \return Pointer to the shared value.
*/
template< typename Type >  // Type of the shared value
inline typename SharedValue<Type>::Pointer SharedValue<Type>::base() const noexcept
{
   return value_.get();
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name SharedValue operators */
//@{
template< typename T1, typename T2 >
inline bool operator==( const SharedValue<T1>& lhs, const SharedValue<T2>& rhs );

template< typename T1, typename T2 >
inline bool operator!=( const SharedValue<T1>& lhs, const SharedValue<T2>& rhs );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between two SharedValue objects.
// \ingroup symmetric_matrix
//
// \param lhs The left-hand side SharedValue object.
// \param rhs The right-hand side SharedValue object.
// \return \a true if both shared values refer to the same value, \a false if they don't.
*/
template< typename T1, typename T2 >
inline bool operator==( const SharedValue<T1>& lhs, const SharedValue<T2>& rhs )
{
   return ( lhs.base() == rhs.base() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between two SharedValue objects.
// \ingroup symmetric_matrix
//
// \param lhs The left-hand side SharedValue object.
// \param rhs The right-hand side SharedValue object.
// \return \a true if both shared values refer to different values, \a false if they don't.
*/
template< typename T1, typename T2 >
inline bool operator!=( const SharedValue<T1>& lhs, const SharedValue<T2>& rhs )
{
   return ( lhs.base() != rhs.base() );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name SharedValue global functions */
//@{
template< typename Type >
inline bool isDefault( const SharedValue<Type>& value );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the shared value is in default state.
// \ingroup symmetric_matrix
//
// \param value The given shared value.
// \return \a true in case the shared value is in default state, \a false otherwise.
//
// This function checks whether the given shared value is in default state. In case it is in
// default state, the function returns \a true, otherwise it returns \a false.
*/
template< typename Type >
inline bool isDefault( const SharedValue<Type>& value )
{
   using blaze::isDefault;

   return isDefault( *value );
}
//*************************************************************************************************

} // namespace blaze

#endif
