/*

Copyright (c) 2012 Jakob Progsch, Václav Zeman

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

   3. This notice may not be removed or altered from any source
   distribution.


This source code has been modified to have optional bounded size.
*/

#pragma once

#include <iostream>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

#include "common/logging.h"

namespace marian {

class ThreadPool {
 public:
    explicit ThreadPool(size_t threads, size_t bound /* bound on size, or 0 for unbounded */ = 0);

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();

    size_t getNumTasks() const {
      return tasks.size();
    }

    void wait_for_one(std::unique_lock<std::mutex>& lock) {
      waiting_threads++;
      sync_condition.notify_all();
      sync_condition.wait(lock, [this]{ return continue_work; });
      waiting_threads--;
    }

    void wait_for_others(std::unique_lock<std::mutex>& lock) {
      continue_work = false;
      sync_condition.wait(lock, [this]{
        return waiting_threads == workers.size() - 1;
      });
    }

    void notify_others() {
      continue_work = true;
      sync_condition.notify_all();
    }

    void join_all() {
      {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
      }
      bounded_condition.notify_all();
      condition.notify_all();
      for (std::thread &worker: workers) {
        worker.join();
      }
    }

 private:
    // need to keep track of threads so we can join them
    std::vector<std::thread> workers;
    // the task queue
    std::queue< std::function<void()> > tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::size_t bound;
    std::condition_variable bounded_condition;
    bool stop;
    std::condition_variable sync_condition;
    bool continue_work{true};
    size_t waiting_threads{0};
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads, size_t in_bound)
  : stop(false), bound(in_bound) {
    for (size_t i = 0; i < threads; ++i)
      workers.emplace_back(
          [this] {
              for(;;) {
                  std::function<void()> task;
                  {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock,
                        [this]{ return this->stop || !this->tasks.empty(); });
                    if (this->stop && this->tasks.empty()) {
                        return;
                    }
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                  }
                  this->bounded_condition.notify_one();
                  task();
              }
          }
      );
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
  using return_type = typename std::result_of<F(Args...)>::type;

  auto inner_task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
  auto outer_task = [inner_task]() -> return_type {
    try {
      return inner_task();
    }
    catch(const std::exception& e) {
      ABORT("Caught std::exception in sub-thread: {}", e.what());
    }
    catch(...) {
      ABORT("Caught unknown exception in sub-thread");
    }
  };

  auto task = std::make_shared<std::packaged_task<return_type()>>(outer_task);

  std::future<return_type> res = task->get_future();
  {
      std::unique_lock<std::mutex> lock(queue_mutex);
      this->bounded_condition.wait(lock, [this] { return this->tasks.size() < this->bound || this->bound == 0 || this->stop; });
      // don't allow enqueueing after stopping the pool
      if (stop) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }

      tasks.emplace([task](){
          (*task)();
      });
  }
  condition.notify_one();
  return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
  if(!stop)
    join_all();
}

}
