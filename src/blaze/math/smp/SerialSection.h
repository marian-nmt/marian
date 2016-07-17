//=================================================================================================
/*!
//  \file blaze/math/smp/SerialSection.h
//  \brief Header file for the serial section implementation
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

#ifndef _BLAZE_MATH_SMP_SERIALSECTION_H_
#define _BLAZE_MATH_SMP_SERIALSECTION_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Exception.h>
#include <blaze/util/Suffix.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Section to enforce the serial execution of operations.
// \ingroup smp
//
// The SerialSection class is an auxiliary helper class for the \a BLAZE_SERIAL_SECTION macro.
// It provides the functionality to detect whether a serial section is active, i.e. if the
// currently executed code is inside a serial section.
*/
template< typename T >
class SerialSection
{
 public:
   //**Constructor*********************************************************************************
   /*!\name Constructor */
   //@{
   inline SerialSection( bool activate );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~SerialSection();
   //@}
   //**********************************************************************************************

   //**Conversion operator*************************************************************************
   /*!\name Conversion operator */
   //@{
   inline operator bool() const;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   static bool active_;  //!< Activity flag for the serial section.
                         /*!< In case a serial section is active (i.e. the currently executed
                              code is inside a serial section), the flag is set to \a true,
                              otherwise it is \a false. */
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   friend bool isSerialSectionActive();
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  DEFINITION AND INITIALIZATION OF THE STATIC MEMBER VARIABLES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T >
bool SerialSection<T>::active_ = false;
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the SerialSection class.
//
// \param activate Activation flag for the serial section.
// \exception std::runtime_error Nested serial sections detected.
*/
template< typename T >
inline SerialSection<T>::SerialSection( bool activate )
{
   if( active_ ) {
      BLAZE_THROW_RUNTIME_ERROR( "Nested serial sections detected" );
   }

   active_ = activate;
}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor of the SerialSection class.
*/
template< typename T >
inline SerialSection<T>::~SerialSection()
{
   active_ = false;  // Resetting the activity flag
}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion operator to \a bool.
//
// The conversion operator returns \a true in case a serial section is active and \a false
// otherwise.
*/
template< typename T >
inline SerialSection<T>::operator bool() const
{
   return active_;
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name SerialSection functions */
//@{
inline bool isSerialSectionActive();
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether a serial section is active or not.
// \ingroup smp
//
// \return \a true if a serial section is active, \a false if not.
*/
inline bool isSerialSectionActive()
{
   return SerialSection<int>::active_;
}
//*************************************************************************************************








//=================================================================================================
//
//  SERIAL SECTION MACRO
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Section to enforce the serial execution of operations.
// \ingroup smp
//
// This macro provides the option to start a serial section to enforce the serial execution of
// operations. The following example demonstrates how a serial section is used:

   \code
   using blaze::rowMajor;
   using blaze::columnVector;

   blaze::DynamicMatrix<double,rowMajor> A;
   blaze::DynamicVector<double,columnVector> b, c, d, x, y, z;

   // ... Resizing and initialization

   // Start of a serial section
   // All operations executed within the serial section are guaranteed to be executed in
   // serial (even if a parallel execution would be possible and/or beneficial).
   BLAZE_SERIAL_SECTION {
      x = A * b;
      y = A * c;
      z = A * d;
   }
   \endcode

// Note that it is not allowed to use nested serial sections (i.e. a serial section within
// another serial section). In case the nested use of a serial section is detected, a
// \a std::runtime_error exception is thrown.
*/
#define BLAZE_SERIAL_SECTION \
   if( blaze::SerialSection<int> BLAZE_JOIN( serialSection, __LINE__ ) = true )
//*************************************************************************************************

} // namespace blaze

#endif
