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
#include "xhost.h"
using namespace std;

namespace gemx{

class XHostScheduler{
public:
	XHostScheduler(const vector<shared_ptr<XHost<void*>>> & host_vec)
	{
		unsigned nthreads = host_vec.size();
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
	virtual ~XHostScheduler()
	{
	    delete _tpool;
	}

	bool empty()
	{
        std::lock_guard<std::mutex> guard(_m);
        return _results.empty();
	}

	void* dequeue()
	{
	    std::future<void*> ret;
	    {
            std::lock_guard<std::mutex> guard(_m);
            ret = std::move(_results.front());
            _results.pop();
	    }
        //ret.get();
        return ret.get();
	}

	virtual void enqueue (void * in_ptr, unsigned num_bytes, void* out_ptr)
	{
        std::lock_guard<std::mutex> guard(this->_m);
	    //_results.emplace( _tpool->enqueue(f) );
	    this->_results.emplace( this->_tpool->enqueue( &XHostScheduler::run, _thrIDMap, in_ptr, num_bytes, out_ptr ));
	}

protected:
	static int run( const unordered_map<string, unsigned> & threadIDMap, void * in_ptr, unsigned num_bytes, void* out_ptr  )
	{
        string s = gemx::getThreadIdStr();
        //std::cout << "hello " << i << std::endl;
        //std::this_thread::sleep_for(std::chrono::seconds(1));
        //std::cout << "world tid" << n << std::endl;
        unsigned PE = threadIDMap.find(s)->second;
		_host_vec[PE]->SendToFPGA(in_ptr, in_ptr, num_bytes, false);
		_host_vec[PE]->Execute ( true);
		_host_vec[PE]->GetMat(out_ptr,true, true);
        return out_ptr;
	}

	unordered_map<string, unsigned> _thrIDMap;
	ThreadPool * _tpool;
	mutex _m;
	queue< std::future<void*> > _results;
	vector<shared_ptr<XHost<void*>>> _host_vec;

};

}
#endif
