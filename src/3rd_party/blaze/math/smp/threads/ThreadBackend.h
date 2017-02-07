//=================================================================================================
/*!
//  \file blaze/math/smp/threads/ThreadBackend.h
//  \brief Header file for the C++11 and Boost thread backend
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

#ifndef _BLAZE_MATH_SMP_THREADS_THREADBACKEND_H_
#define _BLAZE_MATH_SMP_THREADS_THREADBACKEND_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#if BLAZE_CPP_THREADS_PARALLEL_MODE
#  include <condition_variable>
#  include <mutex>
#  include <thread>
#elif BLAZE_BOOST_THREADS_PARALLEL_MODE
#  include <boost/thread/condition.hpp>
#  include <boost/thread/mutex.hpp>
#  include <boost/thread/thread.hpp>
#endif

#include <cstdlib>
#include <blaze/math/constraints/Expression.h>
#include <blaze/math/Functions.h>
#include <blaze/system/SMP.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/ThreadPool.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend system for the C++11 and Boost thread-based parallelization.
// \ingroup smp
//
// The ThreadBackend class template represents the backend system for the C++11 and Boost
// thread-based parallelization. It provides the functionality to manage a pool of active
// threads and to schedule (compound) assignment tasks for execution.\n
// This class must \b NOT be used explicitly! It is reserved for internal use only. Using
// this class explicitly might result in erroneous results and/or in undefined behavior.
*/
template< typename TT    // Type of the encapsulated thread
        , typename MT    // Type of the synchronization mutex
        , typename LT    // Type of the mutex lock
        , typename CT >  // Type of the condition variable
class ThreadBackend
{
 public:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   static inline size_t size  ();
   static inline void   resize( size_t n, bool block=false );
   static inline void   wait  ();
   //@}
   //**********************************************************************************************

   //**Thread execution functions******************************************************************
   /*!\name Thread execution functions */
   //@{
   template< typename Target, typename Source >
   static inline void scheduleAssign( Target& target, const Source& source );

   template< typename Target, typename Source >
   static inline void scheduleAddAssign( Target& target, const Source& source );

   template< typename Target, typename Source >
   static inline void scheduleSubAssign( Target& target, const Source& source );

   template< typename Target, typename Source >
   static inline void scheduleMultAssign( Target& target, const Source& source );

   template< typename Target, typename Source >
   static inline void scheduleDivAssign( Target& target, const Source& source );
   //@}
   //**********************************************************************************************

 private:
   //**Private class Assigner**********************************************************************
   /*!\brief Auxiliary functor for the threaded execution of a plain assignment.
   */
   template< typename Target    // Type of the target operand
           , typename Source >  // Type of the source operand
   struct Assigner
   {
      //**Constructor******************************************************************************
      /*!\brief Constructor for the Assigner class template.
      //
      // \param target The target operand to be assigned to.
      // \param source The source operand to be assigned to the target.
      */
      explicit inline Assigner( Target& target, const Source& source )
         : target_( target )  // The target operand
         , source_( source )  // The source operand
      {}
      //*******************************************************************************************

      //**Function call operator*******************************************************************
      /*!\brief Performs the assignment between the two given operands.
      //
      // \return void
      */
      inline void operator()() {
         assign( target_, source_ );
      }
      //*******************************************************************************************

      //**Member variables*************************************************************************
      Target       target_;  //!< The target operand.
      const Source source_;  //!< The source operand.
      //*******************************************************************************************

      //**Member variables*************************************************************************
      BLAZE_CONSTRAINT_MUST_BE_EXPRESSION_TYPE( Target );
      BLAZE_CONSTRAINT_MUST_BE_EXPRESSION_TYPE( Source );
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Private class AddAssigner*******************************************************************
   /*!\brief Auxiliary functor for the threaded execution of an addition assignment.
   */
   template< typename Target    // Type of the target operand
           , typename Source >  // Type of the source operand
   struct AddAssigner
   {
      //**Constructor******************************************************************************
      /*!\brief Constructor for the AddAssigner class template.
      //
      // \param target The target operand to be assigned to.
      // \param source The source operand to be added to the target.
      */
      explicit inline AddAssigner( Target& target, const Source& source )
         : target_( target )  // The target operand
         , source_( source )  // The source operand
      {}
      //*******************************************************************************************

      //**Function call operator*******************************************************************
      /*!\brief Performs the addition assignment between the two given operands.
      //
      // \return void
      */
      inline void operator()() {
         addAssign( target_, source_ );
      }
      //*******************************************************************************************

      //**Member variables*************************************************************************
      Target       target_;  //!< The target operand.
      const Source source_;  //!< The source operand.
      //*******************************************************************************************

      //**Member variables*************************************************************************
      BLAZE_CONSTRAINT_MUST_BE_EXPRESSION_TYPE( Target );
      BLAZE_CONSTRAINT_MUST_BE_EXPRESSION_TYPE( Source );
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Private class SubAssigner*******************************************************************
   /*!\brief Auxiliary functor for the threaded execution of a subtraction assignment.
   */
   template< typename Target    // Type of the target operand
           , typename Source >  // Type of the source operand
   struct SubAssigner
   {
      //**Constructor******************************************************************************
      /*!\brief Constructor for the SubAssigner class template.
      //
      // \param target The target operand to be assigned to.
      // \param source The source operand to be subtracted from the target.
      */
      explicit inline SubAssigner( Target& target, const Source& source )
         : target_( target )  // The target operand
         , source_( source )  // The source operand
      {}
      //*******************************************************************************************

      //**Function call operator*******************************************************************
      /*!\brief Performs the subtraction assignment between the two given operands.
      //
      // \return void
      */
      inline void operator()() {
         subAssign( target_, source_ );
      }
      //*******************************************************************************************

      //**Member variables*************************************************************************
      Target       target_;  //!< The target operand.
      const Source source_;  //!< The source operand.
      //*******************************************************************************************

      //**Member variables*************************************************************************
      BLAZE_CONSTRAINT_MUST_BE_EXPRESSION_TYPE( Target );
      BLAZE_CONSTRAINT_MUST_BE_EXPRESSION_TYPE( Source );
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Private class MultAssigner******************************************************************
   /*!\brief Auxiliary functor for the threaded execution of a multiplication assignment.
   */
   template< typename Target    // Type of the target operand
           , typename Source >  // Type of the source operand
   struct MultAssigner
   {
      //**Constructor******************************************************************************
      /*!\brief Constructor for the MultAssigner class template.
      //
      // \param target The target operand to be assigned to.
      // \param source The source operand to be multiplied with the target.
      */
      explicit inline MultAssigner( Target& target, const Source& source )
         : target_( target )  // The target operand
         , source_( source )  // The source operand
      {}
      //*******************************************************************************************

      //**Function call operator*******************************************************************
      /*!\brief Performs the multiplication assignment between the two given operands.
      //
      // \return void
      */
      inline void operator()() {
         multAssign( target_, source_ );
      }
      //*******************************************************************************************

      //**Member variables*************************************************************************
      Target       target_;  //!< The target operand.
      const Source source_;  //!< The source operand.
      //*******************************************************************************************

      //**Member variables*************************************************************************
      BLAZE_CONSTRAINT_MUST_BE_EXPRESSION_TYPE( Target );
      BLAZE_CONSTRAINT_MUST_BE_EXPRESSION_TYPE( Source );
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Private class DivAssigner*******************************************************************
   /*!\brief Auxiliary functor for the threaded execution of a division assignment.
   */
   template< typename Target    // Type of the target operand
           , typename Source >  // Type of the source operand
   struct DivAssigner
   {
      //**Constructor******************************************************************************
      /*!\brief Constructor for the DivAssigner class template.
      //
      // \param target The target operand to be assigned to.
      // \param source The source operand to be divided from the target.
      */
      explicit inline DivAssigner( Target& target, const Source& source )
         : target_( target )  // The target operand
         , source_( source )  // The source operand
      {}
      //*******************************************************************************************

      //**Function call operator*******************************************************************
      /*!\brief Performs the division assignment between the two given operands.
      //
      // \return void
      */
      inline void operator()() {
         divAssign( target_, source_ );
      }
      //*******************************************************************************************

      //**Member variables*************************************************************************
      Target       target_;  //!< The target operand.
      const Source source_;  //!< The source operand.
      //*******************************************************************************************

      //**Member variables*************************************************************************
      BLAZE_CONSTRAINT_MUST_BE_EXPRESSION_TYPE( Target );
      BLAZE_CONSTRAINT_MUST_BE_EXPRESSION_TYPE( Source );
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Initialization functions********************************************************************
   /*!\name Initialization functions */
   //@{
   static inline size_t initPool();
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   static ThreadPool<TT,MT,LT,CT> threadpool_;  //!< The pool of active threads of the backend system.
                                                /*!< It is initialized with the number of threads
                                                     specified via the environment variable
                                                     \c BLAZE_NUM_THREADS. However, it can be
                                                     explicitly resized to arbitrary numbers of
                                                     threads. */
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DEFINITION AND INITIALIZATION OF THE STATIC MEMBER VARIABLES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, typename MT, typename LT, typename CT >
ThreadPool<TT,MT,LT,CT> ThreadBackend<TT,MT,LT,CT>::threadpool_( initPool() );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the total number of threads managed by the thread backend system.
//
// \return The total number of threads of the thread backend system.
*/
template< typename TT    // Type of the encapsulated thread
        , typename MT    // Type of the synchronization mutex
        , typename LT    // Type of the mutex lock
        , typename CT >  // Type of the condition variable
inline size_t ThreadBackend<TT,MT,LT,CT>::size()
{
   return threadpool_.size();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Changes the total number of threads managed by the thread backend system.
//
// \param n The new number of threads \f$[1..\infty)\f$.
// \param block \a true if the function shall block, \a false if not.
// \return void
// \exception std::invalid_argument Invalid number of threads.
//
// This function changes the total number of threads managed by the thread backend system. If
// \a n is smaller than the current size of the thread pool, the according number of threads is
// removed from the backend system, otherwise new threads are added to the backend system. In
// case an invalid number of threads is specified, an \a std::invalid_argument exception is
// thrown. Via the \a block flag it is possible to block the function until the desired
// number of threads is available.
*/
template< typename TT    // Type of the encapsulated thread
        , typename MT    // Type of the synchronization mutex
        , typename LT    // Type of the mutex lock
        , typename CT >  // Type of the condition variable
inline void ThreadBackend<TT,MT,LT,CT>::resize( size_t n, bool block )
{
   return threadpool_.resize( n, block );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Waiting for all scheduled tasks to be completed.
//
// \return void
//
// This function blocks until all scheduled tasks have been completed.
*/
template< typename TT    // Type of the encapsulated thread
        , typename MT    // Type of the synchronization mutex
        , typename LT    // Type of the mutex lock
        , typename CT >  // Type of the condition variable
inline void ThreadBackend<TT,MT,LT,CT>::wait()
{
   threadpool_.wait();
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  THREAD EXECUTION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scheduling an assignment of the given operands for execution.
//
// \param target The target operand to be assigned to.
// \param source The target operand to be assigned to the target.
// \return void
//
// This function schedules a plain assignment of the two given operands for execution.
*/
template< typename TT        // Type of the encapsulated thread
        , typename MT        // Type of the synchronization mutex
        , typename LT        // Type of the mutex lock
        , typename CT >      // Type of the condition variable
template< typename Target    // Type of the target operand
        , typename Source >  // Type of the source operand
inline void ThreadBackend<TT,MT,LT,CT>::scheduleAssign( Target& target, const Source& source )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST( Target );
   threadpool_.schedule( Assigner<Target,Source>( target, source ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scheduling an addition assignment of the given operands for execution.
//
// \param target The target operand to be assigned to.
// \param source The target operand to be added to the target.
// \return void
//
// This function schedules an addition assignment of the two given operands for execution.
*/
template< typename TT        // Type of the encapsulated thread
        , typename MT        // Type of the synchronization mutex
        , typename LT        // Type of the mutex lock
        , typename CT >      // Type of the condition variable
template< typename Target    // Type of the target operand
        , typename Source >  // Type of the source operand
inline void ThreadBackend<TT,MT,LT,CT>::scheduleAddAssign( Target& target, const Source& source )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST( Target );
   threadpool_.schedule( AddAssigner<Target,Source>( target, source ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scheduling a subtraction assignment of the given operands for execution.
//
// \param target The target operand to be assigned to.
// \param source The target operand to be subtracted from the target.
// \return void
//
// This function schedules a subtraction assignment of the two given operands for execution.
*/
template< typename TT        // Type of the encapsulated thread
        , typename MT        // Type of the synchronization mutex
        , typename LT        // Type of the mutex lock
        , typename CT >      // Type of the condition variable
template< typename Target    // Type of the target operand
        , typename Source >  // Type of the source operand
inline void ThreadBackend<TT,MT,LT,CT>::scheduleSubAssign( Target& target, const Source& source )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST( Target );
   threadpool_.schedule( SubAssigner<Target,Source>( target, source ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scheduling a multiplication assignment of the given operands for execution.
//
// \param target The target operand to be assigned to.
// \param source The target operand to be multiplied with the target.
// \return void
//
// This function schedules a multiplication assignment of the two given operands for execution.
*/
template< typename TT        // Type of the encapsulated thread
        , typename MT        // Type of the synchronization mutex
        , typename LT        // Type of the mutex lock
        , typename CT >      // Type of the condition variable
template< typename Target    // Type of the target operand
        , typename Source >  // Type of the source operand
inline void ThreadBackend<TT,MT,LT,CT>::scheduleMultAssign( Target& target, const Source& source )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST( Target );
   threadpool_.schedule( MultAssigner<Target,Source>( target, source ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scheduling a division assignment of the given operands for execution.
//
// \param target The target operand to be assigned to.
// \param source The target operand to be divided from the target.
// \return void
//
// This function schedules a division assignment of the two given operands for execution.
*/
template< typename TT        // Type of the encapsulated thread
        , typename MT        // Type of the synchronization mutex
        , typename LT        // Type of the mutex lock
        , typename CT >      // Type of the condition variable
template< typename Target    // Type of the target operand
        , typename Source >  // Type of the source operand
inline void ThreadBackend<TT,MT,LT,CT>::scheduleDivAssign( Target& target, const Source& source )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST( Target );
   threadpool_.schedule( DivAssigner<Target,Source>( target, source ) );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  INITIALIZATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the initial number of threads of the thread pool.
//
// \return The initial number of threads.
//
// This function determines the initial number of threads based on the \c BLAZE_NUM_THREADS
// environment variable. In case the environment variable is not defined or not set, the
// function returns 1. Otherwise it returns the specified number of threads.
*/
#if (defined _MSC_VER)
#  pragma warning(push)
#  pragma warning(disable:4996)
#endif
template< typename TT    // Type of the encapsulated thread
        , typename MT    // Type of the synchronization mutex
        , typename LT    // Type of the mutex lock
        , typename CT >  // Type of the condition variable
inline size_t ThreadBackend<TT,MT,LT,CT>::initPool()
{
   const char* env = std::getenv( "BLAZE_NUM_THREADS" );

   if( env == nullptr )
      return 1UL;
   else return max( 1, atoi( env ) );
}
#if (defined _MSC_VER)
#  pragma warning(pop)
#endif
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  TYPE DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The type of the active thread backend system.
// \ingroup smp
//
// This type represents the active thread backend system. This backend system must be used to
// manage the active number of threads used to execute operations and to schedule tasks to be
// executed.
*/
#if BLAZE_CPP_THREADS_PARALLEL_MODE
typedef ThreadBackend< std::thread
                     , std::mutex
                     , std::unique_lock< std::mutex >
                     , std::condition_variable
                     >  TheThreadBackend;
#elif BLAZE_BOOST_THREADS_PARALLEL_MODE
typedef ThreadBackend< boost::thread
                     , boost::mutex
                     , boost::unique_lock< boost::mutex >
                     , boost::condition_variable
                     >  TheThreadBackend;
#endif
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COMPILE TIME CONSTRAINTS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
namespace {

BLAZE_STATIC_ASSERT( BLAZE_CPP_THREADS_PARALLEL_MODE || BLAZE_BOOST_THREADS_PARALLEL_MODE );

}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
