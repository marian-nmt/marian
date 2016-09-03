//=================================================================================================
/*!
//  \file blaze/math/smp/ParallelSection.h
//  \brief Header file for the parallel section implementation
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

#ifndef _BLAZE_MATH_SMP_PARALLELSECTION_H_
#define _BLAZE_MATH_SMP_PARALLELSECTION_H_


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
/*!\brief Section for the debugging of the shared-memory parallelization.
// \ingroup smp
//
// The ParallelSection class is an auxiliary helper class for the \a BLAZE_PARALLEL_SECTION macro.
// It provides the functionality to detected whether a parallel section has been started and with
// that serves as a utility for debugging the shared-memory parallelization.
*/
template< typename T >
class ParallelSection
{
 public:
   //**Constructor*********************************************************************************
   /*!\name Constructor */
   //@{
   inline ParallelSection( bool activate );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~ParallelSection();
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
   static bool active_;  //!< Activity flag for the parallel section.
                         /*!< In case a parallel section is active (i.e. the currently executed
                              code is inside a parallel section), the flag is set to \a true,
                              otherwise it is \a false. */
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   friend bool isParallelSectionActive();
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
bool ParallelSection<T>::active_ = false;
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the ParallelSection class.
//
// \param activate Activation flag for the parallel section.
// \exception std::runtime_error Nested parallel sections detected.
*/
template< typename T >
inline ParallelSection<T>::ParallelSection( bool activate )
{
   if( active_ ) {
      BLAZE_THROW_RUNTIME_ERROR( "Nested parallel sections detected" );
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
/*!\brief Destructor of the ParallelSection class.
*/
template< typename T >
inline ParallelSection<T>::~ParallelSection()
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
// The conversion operator returns \a true in case a parallel section is active and \a false
// otherwise.
*/
template< typename T >
inline ParallelSection<T>::operator bool() const
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
/*!\name ParallelSection functions */
//@{
inline bool isParallelSectionActive();
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether a parallel section is active or not.
// \ingroup smp
//
// \return \a true if a parallel section is active, \a false if not.
*/
inline bool isParallelSectionActive()
{
   return ParallelSection<int>::active_;
}
//*************************************************************************************************








//=================================================================================================
//
//  PARALLEL SECTION MACRO
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Section for the debugging of the shared-memory parallelization.
// \ingroup smp
//
// During the shared-memory parallel (SMP) execution of an operation nested calls to the SMP
// assign functions are conceptually not allowed. In other words, it is not allowed to call a
// SMP assign function from within a non-SMP assign function. The BLAZE_PARALLEL_SECTION macro
// can be used to mark the start of a parallel section and with that detect nested SMP assign
// function calls. In case a nested use of a parallel section is detected, a \a std::runtime_error
// exception is thrown.\n
// Note that this macro is reserved for internal debugging purposes only and therefore must \b NOT
// be used explicitly! Using this macro might result in erroneous results, runtime or compilation
// errors.
*/
#define BLAZE_PARALLEL_SECTION \
   if( blaze::ParallelSection<int> BLAZE_JOIN( parallelSection, __LINE__ ) = true )
//*************************************************************************************************

} // namespace blaze

#endif
