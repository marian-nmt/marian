//=================================================================================================
/*!
//  \file blaze/util/Thread.h
//  \brief Header file for the Thread class
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

#ifndef _BLAZE_UTIL_THREAD_H_
#define _BLAZE_UTIL_THREAD_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <functional>
#include <memory>
#include <blaze/util/Assert.h>
#include <blaze/util/NonCopyable.h>


namespace blaze {

//=================================================================================================
//
//  ::blaze NAMESPACE FORWARD DECLARATIONS
//
//=================================================================================================

template< typename TT, typename MT, typename LT, typename CT > class ThreadPool;




//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Implementation of a single thread of execution.
// \ingroup threads
//
// \section thread_general General
//
// The Thread template represents a thread of execution for the parallel execution of concurrent
// tasks. Each Thread object incorporates a single thread of execution, or Not-a-Thread, and at
// most one Thread object incorporates a given thread of execution since it is not possible to
// copy a Thread.\n
//
//
// \section thread_definition Class Definition
//
// The implementation of the Thread class template is based on the implementation of standard
// threads as provided by the C++11 standard or the Boost library. Via the four template
// parameters it is possible to configure a Thread instance as either a C++11 thread or as
// Boost thread:

   \code
   template< typename TT, typename MT, typename LT, typename CT >
   class Thread;
   \endcode

//  - TT: specifies the type of the encapsulated thread. This can either be \c std::thread,
//        \c boost::thread, or any other standard conforming thread type.
//  - MT: specifies the type of the used synchronization mutex. This can for instance be
//        \c std::mutex, \c boost::mutex, or any other standard conforming mutex type.
//  - LT: specifies the type of lock used in combination with the given mutex type. This
//        can be any standard conforming lock type, as for instance \c std::unique_lock,
//        \c boost::unique_lock.
//  - CT: specifies the type of the used condition variable. This can for instance be
//        \c std::condition_variable, \c boost::condition_variable, or any other standard
//        conforming condition variable type.
//
// Examples:

   \code
   typedef blaze::Thread< boost::thread
                        , boost::mutex
                        , boost::unique_lock<boost::mutex>
                        , boost::condition_variable >  BoostThread;

   typedef blaze::Thread< std::thread
                        , std::mutex
                        , std::unique_lock<std::mutex>
                        , std::condition_variable >  StdThread;
   \endcode

// For more information about the standard thread functionality, see [1] or [2] or the current
// documentation at the Boost homepage: www.boost.org.
//
//
// \section thread_setup Creating individual threads
//
// The Blaze library provides the functionality to create individual threads for specific tasks,
// or to create thread pools for the execution of a larger number of tasks (see the ThreadPool
// class description). The following example demonstrates the setup of individual threads to
// handle specific tasks. In this example, a function without arguments and a functor with two
// arguments are executed in parallel by two distinct threads:

   \code
   // Definition of a function with no arguments that returns void
   void function0() { ... }

   // Definition of a functor (function object) taking two arguments and returning void
   struct Functor2
   {
      void operator()( int a, int b ) { ... }
   };

   int main()
   {
      // Creating a new thread executing the zero argument function.
      StdThread thread1( function0 );

      // Waiting for the thread to finish its task
      thread1.join();

      // After the thread has completed its tasks, it is not possible to reassign it a
      // new task. Therefore it is necessary to create a new thread for optional follow
      // up tasks.

      // Creating a new thread executing the binary functor.
      StdThread thread2( Functor2(), 4, 6 );

      // Waiting for the second thread to finish its task
      thread2.join();
   }
   \endcode

// Note that the Thread class allows for up to five arguments for the given functions/functors.
// Also note that the two tasks are not executed in parallel since the join() function is used
// to wait for each thread's completion.
//
//
// \section thread_exception Throwing exceptions in a thread parallel environment
//
// It can happen that during the execution of a given task a thread encounters an erroneous
// situation and has to throw an exception. However, exceptions thrown in the usual way
// cannot be caught by a try-catch-block in the main thread of execution:

   \code
   // Definition of a function throwing a std::runtime_error during its execution
   void task()
   {
      ...
      throw std::runtime_error( ... );
      ...
   }

   // Creating a thread executing the throwing function. Although the setup, execution and
   // destruction of the thread are encapsuled inside a try-catch-block, the exception cannot
   // be caught and results in an abortion of the program.
   try {
      StdThread thread( task );
      thread.join();
   }
   catch( ... )
   {
      ...
   }
   \endcode

// For a detailed explanation how to portably transport exceptions between threads, see [1] or
// [2]. In case of the Boost library, the according Boost functionality as demonstrated in the
// following example has to be used. Note that any function/functor passed to a thread is
// responsible to handle exceptions in this way!

   \code
   #include <boost/bind.hpp>
   #include <boost/exception_ptr.hpp>

   // Definition of a function that happens to throw an exception. In order to throw the
   // exception, boost::enable_current_exception() is used in combination with throw.
   void throwException()
   {
      ...
      throw boost::enable_current_exception( std::runtime_error( ... ) );
      ...
   }

   // Definition of a thread function. The try-catch-block catches the exception and uses the
   // boost::current_exception() function to get a boost::exception_ptr object.
   void task( boost::exception_ptr& error )
   {
      try {
         throwException();
         error = boost::exception_ptr();
      }
      catch( ... ) {
         error = boost::current_exception();
      }
   }

   // The function that start a thread of execution can pass along a boost::exception_ptr object
   // that is set in case of an exception. Note that boost::current_exception() captures the
   // original type of the exception object. The exception can be thrown again using the
   // boost::rethrow_exception() function.
   void work()
   {
      boost::exception_ptr error;

      Thread<boost::thread> thread( boost::bind( task, boost::ref(error) ) );
      thread.join();

      if( error ) {
         std::cerr << " Exception during thread execution!\n\n";
         boost::rethrow_exception( error );
      }
   }
   \endcode

// \section thread_references References
//
// [1] A. Williams: C++ Concurrency in Action, Manning, 2012, ISBN: 978-1933988771\n
// [2] B. Stroustrup: The C++ Programming Language, Addison-Wesley, 2013, ISBN: 978-0321563842\n
*/
template< typename TT    // Type of the encapsulated thread
        , typename MT    // Type of the synchronization mutex
        , typename LT    // Type of the mutex lock
        , typename CT >  // Type of the condition variable
class Thread : private NonCopyable
{
 private:
   //**Type definitions****************************************************************************
   typedef TT                           ThreadType;      //!< Type of the encapsulated thread.
   typedef ThreadPool<TT,MT,LT,CT>      ThreadPoolType;  //!< Type of the managing thread pool.
   typedef std::unique_ptr<ThreadType>  ThreadHandle;    //!< Handle for a single thread.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit Thread( ThreadPoolType* pool );
   //@}
   //**********************************************************************************************

 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename Callable, typename... Args >
   explicit inline Thread( Callable func, Args&&... args );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~Thread();
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline bool joinable() const;
   inline void join();
   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline bool hasTerminated() const;
   //@}
   //**********************************************************************************************

   //**Thread execution functions******************************************************************
   /*!\name Thread execution functions */
   //@{
   void run();
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   volatile bool   terminated_;  //!< Thread termination flag.
                                 /*!< This flag value is used by the managing thread
                                      pool to learn whether the thread has terminated
                                      its execution. */
   ThreadPoolType* pool_;        //!< Handle to the managing thread pool.
   ThreadHandle    thread_;      //!< Handle to the thread of execution.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   friend class ThreadPool<TT,MT,LT,CT>;
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
/*!\brief Starting a thread in a thread pool.
//
// \param pool Handle to the managing thread pool.
//
// This function creates a new thread in the given thread pool. The thread is kept alive until
// explicitly killed by the managing thread pool.
*/
template< typename TT    // Type of the encapsulated thread
        , typename MT    // Type of the synchronization mutex
        , typename LT    // Type of the mutex lock
        , typename CT >  // Type of the condition variable
Thread<TT,MT,LT,CT>::Thread( ThreadPoolType* pool )
   : terminated_( false   )  // Thread termination flag
   , pool_      ( pool    )  // Handle to the managing thread pool
   , thread_    ( nullptr )  // Handle to the thread of execution
{
   thread_.reset( new ThreadType( std::bind( &Thread::run, this ) ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Starting a thread of execution on the given zero argument function/functor.
//
// \param func The given function/functor.
// \param args The arguments for the function/functor.
//
// This function creates a new thread of execution on the given function/functor. The given
// function/functor must be copyable, must be callable without arguments and must return void.
*/
template< typename TT         // Type of the encapsulated thread
        , typename MT         // Type of the synchronization mutex
        , typename LT         // Type of the mutex lock
        , typename CT >       // Type of the condition variable
template< typename Callable   // Type of the function/functor
        , typename... Args >  // Types of the function/functor arguments
inline Thread<TT,MT,LT,CT>::Thread( Callable func, Args&&... args )
   : pool_  ( nullptr )  // Handle to the managing thread pool
   , thread_( nullptr )  // Handle to the thread of execution
{
   thread_.reset( new ThreadType( func, std::forward<Args>( args )... ) );
}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor for the Thread class.
*/
template< typename TT    // Type of the encapsulated thread
        , typename MT    // Type of the synchronization mutex
        , typename LT    // Type of the mutex lock
        , typename CT >  // Type of the condition variable
Thread<TT,MT,LT,CT>::~Thread()
{}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether this is a thread of execution.
//
// \return \a true if this is a thread of execution, \a false otherwise.
//
// This function returns whether this thread is still executing the given task or if it has
// already finished the job. In case the thread is still execution, the function returns
// \a true, else it returns \a false.
*/
template< typename TT    // Type of the encapsulated thread
        , typename MT    // Type of the synchronization mutex
        , typename LT    // Type of the mutex lock
        , typename CT >  // Type of the condition variable
inline bool Thread<TT,MT,LT,CT>::joinable() const
{
   return thread_->joinable();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Waiting for a thread of execution to complete.
//
// \return void
//
// If the thread is still executing the given task, this function blocks until the thread's
// tasks is completed.
*/
template< typename TT    // Type of the encapsulated thread
        , typename MT    // Type of the synchronization mutex
        , typename LT    // Type of the mutex lock
        , typename CT >  // Type of the condition variable
inline void Thread<TT,MT,LT,CT>::join()
{
   thread_->join();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the thread has terminated its execution.
//
// \return \a true in case the thread is terminated, \a false otherwise.
//
// This function is used by the managing thread pool to learn whether the thread has finished
// its execution and can be destroyed.
*/
template< typename TT    // Type of the encapsulated thread
        , typename MT    // Type of the synchronization mutex
        , typename LT    // Type of the mutex lock
        , typename CT >  // Type of the condition variable
inline bool Thread<TT,MT,LT,CT>::hasTerminated() const
{
   return terminated_;
}
//*************************************************************************************************




//=================================================================================================
//
//  THREAD EXECUTION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Execution function for threads in a thread pool.
//
// This function is executed by any thread managed by a thread pool.
*/
template< typename TT    // Type of the encapsulated thread
        , typename MT    // Type of the synchronization mutex
        , typename LT    // Type of the mutex lock
        , typename CT >  // Type of the condition variable
void Thread<TT,MT,LT,CT>::run()
{
   // Checking the thread pool handle
   BLAZE_INTERNAL_ASSERT( pool_, "Uninitialized pool handle detected" );

   // Executing scheduled tasks
   while( pool_->executeTask() ) {}

   // Setting the termination flag
   terminated_ = true;
}
//*************************************************************************************************

} // namespace blaze

#endif
