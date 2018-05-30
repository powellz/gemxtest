#ifndef __SEMAPHORE_H__
#define __SEMAPHORE_H__
#include <mutex>
#include <condition_variable>

namespace gemx{
class Semaphore {
public:
    Semaphore () = delete;
    Semaphore (const Semaphore & ) = delete;
    Semaphore (unsigned long long count_ = 0)
        : count(count_) {}

    inline void notify()
    {
        std::unique_lock<std::mutex> lock(mtx);
        count++;
        cv.notify_one();
    }

    inline void wait()
    {
        std::unique_lock<std::mutex> lock(mtx);
        while(count == 0){
            cv.wait(lock);
        }
        count--;
    }

private:
    std::mutex mtx;
    std::condition_variable cv;
    unsigned long long count;
};

}
#endif
