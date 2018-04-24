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
    
    virtual void SendSpToFpga(int * row, int * col, double * data, unsigned int m, unsigned int k, unsigned int nnz, short * B, short * C){	
      cout << "SPMV API is not finished yet\n" << endl;
    }
        
    virtual void SendSpToFpgaInt(int * row, int * col, double * data, unsigned int m, unsigned int k, unsigned int nnz, int * B, int * C){	
      cout << "SPMV API is not finished yet\n" << endl;
    }
    
     virtual void SendSpToFpgaFloat(int * row, int * col, double * data, unsigned int m, unsigned int k, unsigned int nnz, float * B, float * C){	
      cout << "SPMV API is not finished yet\n" << endl;
    }
    

};

}


#endif