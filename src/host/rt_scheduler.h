#ifndef __RT_SCHEDULER_H__
#define __RT_SCHEDULER_H__

#include <queue>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <functional>
#include <vector>
#include <future>

#include <boost/thread/barrier.hpp>

#include "semaphore.h"
#include "thread_pool.h"
using namespace std;

namespace gemx{

template <typename DType>
class RunTimeScheduler{
public:
	RunTimeScheduler(unsigned nthreads)
	{
	    //_asio_limit = gemx::Semaphore(1000);
		_tpool = new ThreadPool(nthreads);
		std::atomic<unsigned int> cnt;
		cnt = 0;
		boost::barrier b(nthreads);
	    std::mutex mut;
	    std::vector< std::future<void> > results;
	    for (unsigned i = 0; i < nthreads; i++)
	    {
	        results.emplace_back(
                _tpool->enqueue([this, &mut, &b, &cnt]() {
                    {
                        std::lock_guard<std::mutex> guard(mut);
                        cout << "HI2U thread id " << gemx::getThreadId() << " mapped to " << cnt << endl;
                        _thrIDMap[gemx::getThreadIdStr()] = cnt++;
                    }
                    b.wait();
                })
	        );
	    }
	    for(auto && result: results)
	        result.get();

        cout << "Initialized threads!!!" << endl;
	    //_cv_arr.resize(nthreads);
	}

	bool empty()
	{
        std::lock_guard<std::mutex> guard(_m);
        return _results.empty();
	}

	DType dequeue()
	{
	    std::future<DType> ret;
	    {
            std::lock_guard<std::mutex> guard(_m);
            ret = std::move(_results.front());
            _results.pop();
	    }
        //ret.get();
        return ret.get();
	}

	void enqueue (const DType & d)
	{
        std::lock_guard<std::mutex> guard(_m);
	    //_results.emplace( _tpool->enqueue(f) );
	    _results.emplace( _tpool->enqueue( &RunTimeScheduler::do_work, _thrIDMap, d ));
	}

	~RunTimeScheduler()
	{
	    delete _tpool;
	}

protected:
	static int do_work( const unordered_map<string, unsigned> & threadIDMap, int i )
	{
        string s = gemx::getThreadIdStr();
        //std::cout << "hello " << i << std::endl;
        //std::this_thread::sleep_for(std::chrono::seconds(1));
        //std::cout << "world tid" << n << std::endl;
        unsigned m = threadIDMap.find(s)->second;
        cout << "ThreadID: " << m << "Job: " << i << endl;
        /*
        std::random_device r;
        std::default_random_engine e1(r());
        std::uniform_int_distribution<int> uniform_dist(1,m);
        int rand = uniform_dist(e1);
        std::this_thread::sleep_for (std::chrono::milliseconds(rand));
        */
        return i;
	}

	unordered_map<string, unsigned> _thrIDMap;
	ThreadPool * _tpool;
	mutex _m;
	queue< std::future<DType> > _results;
};

}
#endif
