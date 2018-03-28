#ifndef _XHOST_H_
#define _XHOST_H_
#include "assert.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include "CL/cl.h"
#include "CL/cl_ext.h"
#include <boost/compute.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/program.hpp>
#include <unordered_map>
#include "gemx_util.h"

using namespace std;
namespace gemx{
typedef enum {
    OpControl, OpGemv, OpGemm, OpTransp, OpSpmv, OpResult, OpFail, OpFcn
} OpType;

class kArgs {
public:
    virtual ~kArgs() {
    }
    virtual size_t sizeInBytes() = 0;
    virtual char* asByteArray() = 0;
};


//Base address will be the instruction memory region
class XCL_FPGA {
public:
    XCL_FPGA() = delete;
    XCL_FPGA(const string & xclbin, const string & kernelName, const vector<unsigned> & ddrBanks) {
        loadXclbin(xclbin, kernelName);
        _ddrbanks = ddrBanks;
    }

    ~XCL_FPGA() {
    }

    void loadXclbin(string p_XclbinFile, string p_KernelName) {
        // https://gitenterprise.xilinx.com/rkeryell/heterogeneous_examples/blob/master/vector_add/SDAccel-Boost.Compute/vector_add.cpp

        // Create the OpenCL context to attach resources on the device
        m_Context = move(boost::compute::system::default_context());
        // Create the OpenCL command queue to control the device
        //m_CommandQueue = move(boost::compute::system::default_queue());
        //boost::compute::command_queue queue(boost::compute::system::default_context(), boost::compute::system::default_device(), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
        boost::compute::command_queue queue(
                boost::compute::system::default_context(),
                boost::compute::system::default_device() /* CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/);
        m_CommandQueue = move(queue);
        // Construct an OpenCL program from the precompiled kernel file
        m_Program = move(
                boost::compute::program::create_with_binary_file(p_XclbinFile,
                        m_Context));
        m_Program.build();

        m_Kernel = move(boost::compute::kernel(m_Program, p_KernelName));
    }

    boost::compute::buffer createBuf(void *ptr, size_t sz_bytes) {
        cl_mem_ext_ptr_t l_bufExt;
        //l_bufExt.obj = NULL;
        l_bufExt.param = 0;
        l_bufExt.flags = _ddrbanks[0];
        l_bufExt.obj = ptr;
        // Buffers
        return boost::compute::buffer(m_Context, sz_bytes,
                CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,
                &l_bufExt);
    }

    bool copyToFpga(const boost::compute::buffer & buf, bool sync_send) {
        boost::compute::event l_event;
        //cout << "copyToFPGA" << endl;
        // Send the input data to the accelerator
        l_event = m_CommandQueue.enqueue_migrate_memory_objects(1, &(buf.get()),
                0);

        if (sync_send){
            l_event.wait();
        } else{
            m_waitInput.insert(l_event);
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::compute::buffer copyToFpga(void * buf, size_t sz_bytes,
            bool sync_send = false) {
        boost::compute::buffer cl_buf = createBuf(buf, sz_bytes);
        copyToFpga(cl_buf, sync_send);
        return cl_buf;
    }

    void copyFromFpga(const boost::compute::buffer & buf, bool sync_exec = true) {
        //cout << "copyFromFPGA" << endl;
        XTimer t;
        boost::compute::event l_readEvents =
                m_CommandQueue.enqueue_migrate_memory_objects(1, &(buf.get()),
                        CL_MIGRATE_MEM_OBJECT_HOST, m_waitOutput);
        //l_readEvents.wait();
        if ( sync_exec ){
            l_readEvents.wait();
            m_waitOutput.clear();
        } else{
            m_waitOutput.insert(l_readEvents);
        }
#ifdef GEMX_PERF_DBG
        cout << "copyFromFpga: " << t.elapsed() << endl;
#endif
    }
    void execKernel(const boost::compute::buffer & instr_buf, bool sync_exec = true) {
        boost::compute::extents<1> offset { 0 };
        boost::compute::extents<1> global { 1 };
        // Use only 1 CU
        boost::compute::extents<1> local { 1 };
        // Launch kernels
        m_Kernel.set_args(instr_buf, instr_buf);

        XTimer t;
        //boost::compute::event l_event = m_CommandQueue.enqueue_nd_range_kernel(
        //        m_Kernel, offset, global, local, m_waitInput);
        boost::compute::event l_event = m_CommandQueue.enqueue_task(m_Kernel, m_waitInput);


        if ( sync_exec ) {
            l_event.wait();
        } else{
            m_waitOutput.insert(l_event);
        }
        m_waitInput.clear();
#ifdef GEMX_PERF_DBG
        cout << "execKernel: " << t.elapsed() << endl;
#endif

    }

    void wait ()
    {
        //cout << "out wait sz: " << m_waitOutput.size() << endl;
        for (size_t i = 0; i < m_waitOutput.size(); i++){
            //cout << "OpenCL event status: " <<  m_waitOutput[i].status() << endl;
            m_waitOutput[i].wait();
            //cout << "OpenCL event status after wait: " <<  m_waitOutput[i].status() << endl;
        }
        m_waitInput.clear();
        m_waitOutput.clear();
    }

private:
    vector<unsigned> _ddrbanks;
    boost::compute::program m_Program;
    boost::compute::kernel m_Kernel;
    boost::compute::context m_Context;
    boost::compute::command_queue m_CommandQueue;
    boost::compute::wait_list m_waitInput, m_waitOutput;
};


template<typename HType>
class XHost{
public:
    XHost() = delete;
    XHost ( const string & xclbin, const string & kernelName, const string &device)
    {
        vector<unsigned> ddrBanks = this->getDDRBankFlags(device);
        _fpga = shared_ptr<XCL_FPGA>(new XCL_FPGA(xclbin, kernelName,ddrBanks));
    }

    virtual ~XHost(){}
    virtual vector<unsigned> getDDRBankFlags(const string & device){
            vector<unsigned>ddrBanks;

            unsigned ddr_flags;
            if ( device == "ku115"){
                ddrBanks = {XCL_MEM_DDR_BANK0, XCL_MEM_DDR_BANK2, XCL_MEM_DDR_BANK1, XCL_MEM_DDR_BANK3};
            }
            else if( device == "kcu1500" || device == "vcu1525"){
                ddrBanks = {XCL_MEM_DDR_BANK0, XCL_MEM_DDR_BANK1, XCL_MEM_DDR_BANK2, XCL_MEM_DDR_BANK3};
            }
            else if ( device == "vu9p"){
                ddrBanks = {XCL_MEM_DDR_BANK0, XCL_MEM_DDR_BANK3, XCL_MEM_DDR_BANK2, XCL_MEM_DDR_BANK1};
            }
            else if ( device == "vu9pf1"){
                ddrBanks = {XCL_MEM_DDR_BANK3, XCL_MEM_DDR_BANK2, XCL_MEM_DDR_BANK0, XCL_MEM_DDR_BANK1};
            }
            else{
                cerr << "Unsupported device! Options are ku115, kcu1500, vu9p, vu9pf1, vcu1525" << endl;
                assert( device == "ku115" || device == "kcu1500" || device == "vu9p" || device == "vu9pf1" || device == "vcu1525");
            }
            return ddrBanks;
    }
    virtual void Execute( bool sync_exec = true) = 0;

    bool AddMat(const HType & handle, void * mat_ptr, unsigned long long buf_sz) {
        if (_hostMat.find(handle) == _hostMat.end()) {
            _hostMat[handle] = mat_ptr;
            _hostMatSz[handle] = buf_sz;
            return true;
        }
        else if (_hostMatSz[handle] != buf_sz ){
            _hostMat[handle] = mat_ptr;
            _hostMatSz[handle] = buf_sz;
            this->_devHandle.erase(handle);
            //cout << "Erasing devhandle!" << endl;
            return true;
        }
        //cout << "Matrix " << handle << " already added!" << endl;
        return false;
    }

    void * GetMat(const HType & handle,
            bool queryFPGA = false, bool sync_get = true) {
        void * ret_ptr = nullptr;
        if (_hostMat.find(handle) != _hostMat.end()) {
            if (queryFPGA)
                GetFromFPGA(handle, sync_get);
            ret_ptr = _hostMat[handle];
        }
        return ret_ptr;
    }

    void Wait(){
        _fpga->wait();
    }

    void SendToFPGA(const HType & handle, void * mat_ptr, unsigned long long buf_sz,
            bool sync_send = false) {
        AddMat(handle, mat_ptr, buf_sz);
        SendToFPGA(handle, sync_send);
    }

    void SendToFPGA(const HType & handle, bool sync_send = false) {
        XTimer t;
        assert(_hostMat.find(handle) != _hostMat.end());

        //shared_ptr < Mat<T> > mat = _hostMat[handle];
        if (this->_devHandle.find(handle) != this->_devHandle.end()) {
            _fpga->copyToFpga(this->_devHandle[handle], sync_send);
        } else {
            this->_devHandle[handle] = _fpga->copyToFpga(_hostMat[handle], _hostMatSz[handle], sync_send);
        }
#ifdef GEMX_PERF_DBG
        cout << "SendToFPGA: " << t.elapsed() << endl;
#endif
    }

    void GetFromFPGA(const HType & handle, bool sync_get) {
        XTimer t;
        assert(this->_devHandle.find(handle) != this->_devHandle.end());
        _fpga->copyFromFpga(this->_devHandle[handle], sync_get);
#ifdef GEMX_PERF_DBG
        cout << "GetFromFPGA: " << t.elapsed() << endl;
#endif
    }

    int getBoardFreqMHz(unsigned int p_BoardId) {
      string l_freqCmd = "$XILINX_OPENCL/runtime/bin/xbsak query -d" + to_string(p_BoardId);;
      float l_freq = -1;
      char l_lineBuf[256];
      shared_ptr<FILE> l_pipe(popen(l_freqCmd.c_str(), "r"), pclose);
      if (!l_pipe) cout << ("ERROR: popen(" + l_freqCmd + ") failed");
      bool l_nextLine_isFreq = false;
      while (l_pipe && fgets(l_lineBuf, 256, l_pipe.get()) ) {
      string l_line(l_lineBuf);
      if (l_nextLine_isFreq) {
          string l_prefix, l_val, l_mhz;
          stringstream l_ss(l_line);
          l_ss >> l_prefix >> l_val >> l_mhz;
          l_freq = stof(l_val);
          assert(l_mhz == "MHz");
          break;
      } else if (l_line.find("OCL Frequency:") != string::npos) {
          l_nextLine_isFreq = true;
      }
      }
      if (l_freq == -1) {
      //if xbsak does not work, as happens on F1, put the XOCC achieved kernel frequcy here
      l_freq = -1;
      cout << "INFO: Failed to get board frequency by xbsak. This is normal for cpu and hw emulation, using -1 MHz for reporting.\n";
      }
      return((int)l_freq);
    }

protected:
    unordered_map<HType, void*  > _hostMat;
    unordered_map<HType, unsigned long long > _hostMatSz;
    unordered_map<HType, boost::compute::buffer> _devHandle;
    shared_ptr<XCL_FPGA> _fpga;
};


}


#endif
