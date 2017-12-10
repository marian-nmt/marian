#pragma once
//#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace amunmt {
namespace GPU {

template<class T>
class Buffer
{
public:
    void add(const T &val) {
        while (true) {
            std::unique_lock<std::mutex> locker(mu);
            cond.wait(locker, [this](){return buffer_.size() < size_;});
            buffer_.push(val);
            locker.unlock();
            cond.notify_all();
            return;
        }
    }
    T remove() {
        while (true)
        {
            std::unique_lock<std::mutex> locker(mu);
            cond.wait(locker, [this](){return buffer_.size() > 0;});
            T val = buffer_.front();
            buffer_.pop();
            locker.unlock();
            cond.notify_all();
            return val;
        }
    }
    Buffer(unsigned int maxSize)
    :size_(maxSize)
    {}

    size_t size() const
    { return buffer_.size(); }

private:
   std::mutex mu;
   std::condition_variable cond;

    //std::deque<T> buffer_;
   std::queue<T> buffer_;
    const unsigned int size_;

};

}
}
