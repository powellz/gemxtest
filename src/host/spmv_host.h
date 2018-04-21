#ifndef _SPMV_HOST_H_
#define _SPMV_HOST_H_

#include "xhost.h"
#include "gemm_host.h"
#include "gemx_util.h"

using namespace std;
namespace gemx {

class SpmvArgs: public kArgs {
public:
    virtual ~SpmvArgs() {
    }
    SpmvArgs() = delete;
    SpmvArgs ( unsigned int p_Aoffset, unsigned int p_Boffset, unsigned int p_Coffset, unsigned int M, unsigned int K, unsigned int Nnz) :
        m_spmv_args( { OpSpmv, p_Aoffset, p_Boffset, p_Coffset, M, K, Nnz, 0, 0, 0, 0, 0, 0, 0, 0, 0} ){
    }

    size_t sizeInBytes() {
        return sizeof(m_spmv_args);
    }
    char *asByteArray() {
        return reinterpret_cast<char*>(&m_spmv_args);
    }
protected:
    struct {
        int m_optype;
        unsigned int m_Aoffset, m_Boffset, m_Coffset, m_M, m_K, m_Nnz;
        unsigned int dummy[9];
    } m_spmv_args;
};

template<typename HType>
class SPMVHost : public GEMMHost<HType> {
public:
    SPMVHost() = delete;
    virtual ~SPMVHost(){
    }

    SPMVHost(const SPMVHost<HType> &) = delete;

    SPMVHost(const string & xclbin, const string & kernelName, const unsigned ddrBank, const string & device) : GEMMHost<HType> ( xclbin, kernelName, ddrBank, device)
    {
    }

    virtual bool AddGEMMOp(const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift) {
        cerr << "GEMM operation not supported" << endl;
        return false;
    }

    virtual bool AddGEMMOp(const HType & A, const HType & B, const HType & C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, unsigned int lda, unsigned int ldb, unsigned int ldc, unsigned int ldx, int postScale, int postShift) {
        cerr << "GEMM operation not supported" << endl;
        return false;
    }
    
    virtual void SendSpToFpga(int * row, int * col, double * data, unsigned int m, unsigned int k, unsigned int nnz,const HType & B, const HType & C){	
      vector<MtxRow> A;
      for (unsigned int i = 0; i<nnz; ++i){
	MtxRow l_row(row[i],col[i],data[i]);
	A.push_back(l_row);
      }
      
      this->SendSparseToFpga(&A,(unsigned long long)nnz+nnz*2*(sizeof(int)/sizeof(double)));
      this->SendToFPGA((short*)B, B,(unsigned long long) sizeof(short) * k);
      this->SendToFPGA((short*)C, C,(unsigned long long) sizeof(short) * m);
      
       unsigned long long A_off = 0, B_off = 0, C_off = 0;

        xclGetMemObjDeviceAddress(this->_devHandleSP.get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &A_off);
        xclGetMemObjDeviceAddress(this->_devHandle[B].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &B_off);
        xclGetMemObjDeviceAddress(this->_devHandle[C].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &C_off);

        cout << "A_dev_addr: " << A_off << " B_dev_addr: " << B_off << " C_dev_addr: " << C_off << endl;
        assert(A_off > this->_ddrDeviceBaseAddr);
        assert(B_off > this->_ddrDeviceBaseAddr);
        assert(C_off > this->_ddrDeviceBaseAddr);
        A_off -= this->_ddrDeviceBaseAddr;
        B_off -= this->_ddrDeviceBaseAddr;
        C_off -= this->_ddrDeviceBaseAddr;

        assert(A_off % this->PAGE_SIZE == 0);  
        assert(B_off % this->PAGE_SIZE == 0);
        assert(C_off % this->PAGE_SIZE == 0);

        A_off /= this->PAGE_SIZE;
        B_off /= this->PAGE_SIZE;
        C_off /= this->PAGE_SIZE;

        SpmvArgs args(A_off, B_off, C_off, m, k, nnz);
        this->AddInstr (&args);
    }
    
    virtual void SendSpToFpgaInt(int * row, int * col, double * data, unsigned int m, unsigned int k, unsigned int nnz, int* B, int* C){	
      vector<MtxRow> A;
      for (unsigned int i = 0; i<nnz; ++i){
	MtxRow l_row(row[i],col[i],data[i]);
	A.push_back(l_row);
      }
      
      this->SendSparseToFpga(&A,(unsigned long long)nnz+nnz*2*(sizeof(int)/sizeof(double)));
      this->SendToFPGA((short*)B, B,(unsigned long long) sizeof(int) * k);
      this->SendToFPGA((short*)C, C,(unsigned long long) sizeof(int) * m);
      
       unsigned long long A_off = 0, B_off = 0, C_off = 0;

        xclGetMemObjDeviceAddress(this->_devHandleSP.get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &A_off);
        xclGetMemObjDeviceAddress(this->_devHandle[(short*)B].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &B_off);
        xclGetMemObjDeviceAddress(this->_devHandle[(short*)C].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &C_off);

        cout << "A_dev_addr: " << A_off << " B_dev_addr: " << B_off << " C_dev_addr: " << C_off << endl;
        assert(A_off > this->_ddrDeviceBaseAddr);
        assert(B_off > this->_ddrDeviceBaseAddr);
        assert(C_off > this->_ddrDeviceBaseAddr);
        A_off -= this->_ddrDeviceBaseAddr;
        B_off -= this->_ddrDeviceBaseAddr;
        C_off -= this->_ddrDeviceBaseAddr;

        assert(A_off % this->PAGE_SIZE == 0);  
        assert(B_off % this->PAGE_SIZE == 0);
        assert(C_off % this->PAGE_SIZE == 0);

        A_off /= this->PAGE_SIZE;
        B_off /= this->PAGE_SIZE;
        C_off /= this->PAGE_SIZE;

        SpmvArgs args(A_off, B_off, C_off, m, k, nnz);
        this->AddInstr (&args);
    }
    

};

}


#endif