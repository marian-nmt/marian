//=================================================================================================
/*!
//  \file blaze/util/threadpool/TaskQueue.h
//  \brief Task queue for the thread pool
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

#ifndef _BLAZE_UTIL_THREADPOOL_TASKQUEUE_H_
#define _BLAZE_UTIL_THREADPOOL_TASKQUEUE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <deque>
#include <blaze/util/threadpool/Task.h>


namespace blaze {

namespace threadpool {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Task queue for the thread pool.
// \ingroup threads
//
// The TaskQueue class represents the internal task container of a thread pool. It uses a FIFO
// (first in, first out) strategy to store and remove the assigned tasks.
*/
class TaskQueue
{
 private:
   //**Type definitions****************************************************************************
   typedef std::deque<Task>  Tasks;  //!< FIFO container for tasks.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   typedef Tasks::size_type  SizeType;  //!< Size type of the task queue.
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\name Constructor */
   //@{
   explicit inline TaskQueue();
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~TaskQueue();
   //@}
   //**********************************************************************************************

   //**Get functions*******************************************************************************
   /*!\name Get functions */
   //@{
   inline SizeType maxSize()  const;
   inline SizeType size()     const;
   inline bool     isEmpty()  const;
   //@}
   //**********************************************************************************************

   //**Element functions***************************************************************************
   /*!\name Element functions */
   //@{
   inline void push ( Task task );
   inline Task pop  ();
   inline void clear();
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline void swap( TaskQueue& tq ) noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Tasks tasks_;  //!< FIFO container for the contained tasks.
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
/*!\brief Default constructor for TaskQueue.
*/
inline TaskQueue::TaskQueue()
   : tasks_()  // FIFO container for the contained tasks
{}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Destructor for the TaskQueue class.
//
// The destructor destroys any remaining task in the task queue.
*/
TaskQueue::~TaskQueue()
{
   clear();
}
//*************************************************************************************************




//=================================================================================================
//
//  GET FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the maximum possible size of a task queue.
//
// \return The maximum possible size.
*/
inline TaskQueue::SizeType TaskQueue::maxSize() const
{
   return tasks_.max_size();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current size of the task queue.
//
// \return The current size.
//
// This function returns the number of the currently contained tasks.
*/
inline TaskQueue::SizeType TaskQueue::size() const
{
   return tasks_.size();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns \a true if the task queue has no elements.
//
// \return \a true if the task queue is empty, \a false if it is not.
*/
inline bool TaskQueue::isEmpty() const
{
   return tasks_.empty();
}
//*************************************************************************************************




//=================================================================================================
//
//  ELEMENT FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Adding a task to the end of the task queue.
//
// \param task The task to be added to the end of the task queue.
// \return void
//
// This function adds the given task to the end of the task queue. It runs in constant time.
*/
inline void TaskQueue::push( Task task )
{
   tasks_.push_back( task );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the task from the front of the task queue.
//
// \return The first task in the task queue.
*/
inline Task TaskQueue::pop()
{
   const Task task( tasks_.front() );
   tasks_.pop_front();
   return task;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Removing all tasks from the task queue.
//
// \return void
*/
inline void TaskQueue::clear()
{
   tasks_.clear();
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Swapping the contents of two task queues.
//
// \param tq The task queue to be swapped.
// \return void
// \exception no-throw guarantee.
*/
inline void TaskQueue::swap( TaskQueue& tq ) noexcept
{
   tasks_.swap( tq.tasks_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name TaskQueue operators */
//@{
inline void swap( TaskQueue& a, TaskQueue& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two task queues.
//
// \param a The first task queue to be swapped.
// \param b The second task queue to be swapped.
// \return void
// \exception no-throw guarantee.
*/
inline void swap( TaskQueue& a, TaskQueue& b ) noexcept
{
   a.swap( b );
}
//*************************************************************************************************

} // namespace threadpool

} // namespace blaze


#endif
