#ifndef _SPMV_HOST_H_
#define _SPMV_HOST_H_

#include "xhost.h"
#include "gemm_host.h"
#include "gemx_util.h"

using namespace std;
namespace gemx {

template < typename Tdata,  typename Tidx>
class SpMatUram
{
  private:
    unsigned int m_Rows, m_Cols, m_Nnz;
                Tdata *m_DataAddr;
                Tidx  *m_IdxAddr;
  public:
                static const unsigned int t_NumData = 48;
                static const unsigned int t_NumIdx = 48;
  public:
    SpMatUram(){}
    SpMatUram(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Nnz, Tdata *p_DataAddr)
      : m_Rows(p_Rows), m_Cols(p_Cols), m_Nnz(p_Nnz), m_DataAddr(p_DataAddr), m_IdxAddr((Tidx*)(p_DataAddr+16)) {
    }
    SpMatUram& operator=(const SpMatUram& p_Src) {
        for (unsigned int i = 0; i < m_Nnz; ++i) {
          getVal(i) = p_Src.getVal(i);
          getCol(i) = p_Src.getCol(i);
          getRow(i) = p_Src.getRow(i);
        }
        return *this;
    }
    inline Tdata &getVal(unsigned int p_id) {return m_DataAddr[(p_id/16)*t_NumData+(p_id%16)];}
    inline Tidx &getCol(unsigned int p_id) {return m_IdxAddr[(p_id/16)*t_NumIdx+(p_id%16)*2];}
    inline Tidx &getRow(unsigned int p_id) {return m_IdxAddr[(p_id/16)*t_NumIdx+(p_id%16)*2+1];}
    void 
    init(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Nnz, Tdata *p_DataAddr){
            m_Rows = p_Rows;
            m_Cols = p_Cols;
            m_Nnz = p_Nnz;
            m_DataAddr = p_DataAddr;
            m_IdxAddr = (Tidx*) (p_DataAddr+16);
    }    
    void
    fillFromVector(int* row,int* col, float* data) {
      for (unsigned int i = 0; i < m_Nnz; ++i) {
        getVal(i) = data[i];
        getCol(i) = col[i];
        getRow(i) = row[i];        
      }
    }
};
  
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
       
    virtual void SendSpToFpgaFloat(int * row, int * col, float * data, unsigned int m, unsigned int k, unsigned int nnz, float * B, float * C){   
       vector<float> tmpA(nnz*3);
       float* A = &tmpA[0];
       SpMatUram<float,int> MatA(m,k,nnz,A);
       MatA.fillFromVector(row,col,data);

       this->SendToFPGA((float*)A, A,(unsigned long long)(nnz+nnz*2)*sizeof(float));
       this->SendToFPGA((float*)B, B,(unsigned long long) sizeof(float) * k);
       this->SendToFPGA((float*)C, C,(unsigned long long) sizeof(float) * m);
      
       unsigned long long A_off = 0, B_off = 0, C_off = 0;

       xclGetMemObjDeviceAddress(this->_devHandle[A].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &A_off);
       xclGetMemObjDeviceAddress(this->_devHandle[B].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &B_off);
       xclGetMemObjDeviceAddress(this->_devHandle[C].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &C_off);

       //cout << "A_dev_addr: " << A_off << " B_dev_addr: " << B_off << " C_dev_addr: " << C_off << endl;
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
    
    virtual void SendSpToFpgaInt(int * row, int * col, float * data, unsigned int m, unsigned int k, unsigned int nnz, int * B, int * C){        
       vector<int> tmpA(nnz*3);
       int* A = &tmpA[0];
       SpMatUram<int,int> MatA(m,k,nnz,A);
       MatA.fillFromVector(row,col,data);

       this->SendToFPGA((float*)A, A,(unsigned long long)(nnz+nnz*2)*sizeof(int));
       this->SendToFPGA((float*)B, B,(unsigned long long) sizeof(int) * k);
       this->SendToFPGA((float*)C, C,(unsigned long long) sizeof(int) * m);
      
       unsigned long long A_off = 0, B_off = 0, C_off = 0;

       xclGetMemObjDeviceAddress(this->_devHandle[(float*)A].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &A_off);
       xclGetMemObjDeviceAddress(this->_devHandle[(float*)B].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &B_off);
       xclGetMemObjDeviceAddress(this->_devHandle[(float*)C].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &C_off);

       //cout << "A_dev_addr: " << A_off << " B_dev_addr: " << B_off << " C_dev_addr: " << C_off << endl;
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