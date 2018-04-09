#ifndef _SPMV_HOST_H_
#define _SPMV_HOST_H_

#include "xhost.h"
#include "gemm_host.h"

using namespace std;
namespace gemx {


class SpmvArgs: public kArgs {
public:
    virtual ~SpmvArgs() {
    }
    SpmvArgs() = delete;
    SpmvArgs ( unsigned int p_Aoffset, unsigned int p_Boffset, unsigned int p_Coffset, unsigned int M, unsigned int K, unsigned int Nnz, unsigned int p_Cblocks, unsigned int p_DescPages) :
        m_spmv_args( { OpSpmv, p_Aoffset, p_Boffset, p_Coffset,M, K, Nnz, p_Cblocks, p_DescPages} ){
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
        unsigned int m_Aoffset, m_Boffset, m_Coffset, m_M, m_K, m_Nnz, m_Cblocks, m_DescPages;
        unsigned int dummy[7];
    } m_spmv_args;
};

template<typename HType>
class SpmvHost : public GEMMHost<HType> {
public:
    SpmvHost() = delete;
    virtual ~SpmvHost(){
    }

    SpmvHost(const SpmvHost<HType> &) = delete;

    SpmvHost( const string & xclbin, const string & kernelName, const string & device ) : GEMMHost<HType> ( xclbin, kernelName, device)
    {
    }

    virtual bool AddGEMMOp(const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift) {
        cerr << "GEMM operation not supported" << endl;
        return false;
    }

    virtual bool AddGEMMOp(const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, unsigned int lda, unsigned int ldb, unsigned int ldc, unsigned int ldx, int postScale, int postShift) {
        cerr << "GEMM operation not supported" << endl;
        return false;
    }

    virtual bool AddSpmvOp( )
    {

    }

};

}


#endif
