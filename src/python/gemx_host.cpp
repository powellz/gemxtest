#include "gemx_host.h"
#include <iostream>
#include <fstream>
using namespace std;
//#define GEMX_PERF_DBG

using namespace gemx;

template<typename T>
void print(char *name,T * A, int m, int n){
    ofstream myfile;
    string fName = name;
    fName += ".c";
    myfile.open (fName.c_str());

    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            myfile << A[i*m + j] << " ";
        }
        myfile << "\n";
    }
    myfile.close();
}

void MakeFCNHost(char *xclbin, char * kernName, char * device) {
    GEMXHostHandle<short*>::Instance().gh_ptr = shared_ptr< gemx::GEMMHost<short*> > (new gemx::FCNHost<short*>(xclbin, kernName, device) );
}

void MakeGEMMHost(char *xclbin, char * kernName, char * device) {
    GEMXHostHandle<short*>::Instance().gh_ptr = shared_ptr< gemx::GEMMHost<short*> > ( new gemx::GEMMHost<short*>(xclbin, kernName, device) );
}

void SendToFPGAShrt(short *A, unsigned long long num_elem, bool sync_send){
    gemx::XTimer t;
    GEMXHostHandle<short*>::Instance().gh_ptr->SendToFPGA((short*)A, A, sizeof(short) *num_elem, sync_send);
    //SendToFPGA( A, sizeof(short) * num_elem, sync_send);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["SendToFPGAShrt"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["SendToFPGAShrt"]++;
#endif
}

void SendToFPGAInt(int *A, unsigned long long num_elem, bool sync_send){
    gemx::XTimer t;
    GEMXHostHandle<short*>::Instance().gh_ptr->SendToFPGA((short*)A, A, sizeof(int) *num_elem, sync_send);
    //SendToFPGA(A, sizeof(int) *num_elem, sync_send);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["SendToFPGAInt"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["SendToFPGAInt"]++;
#endif

}

/*
void SendToFPGAShrt_dbg(char * name, short *A, int m, int n, bool sync_send){
    print<short>(name, A, m,n);
    GEMXHostHandle<short*>::Instance().gh_ptr->SendToFPGA(name, A, sizeof(short) * m * n, sync_send);
}

void SendToFPGAInt_dbg(char * name, int *A, int m, int n, bool sync_send){
    print<int>(name, A, m,n);
    GEMXHostHandle<short*>::Instance().gh_ptr->SendToFPGA(name, A, sizeof(int) * m * n, sync_send);
}
*/

void* GetFromFPGA(short *A, bool sync_get)
{
    gemx::XTimer t;
    void * ptr = GEMXHostHandle<short*>::Instance().gh_ptr->GetMat((short*)A, true, sync_get);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["GetFromFPGA"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["GetFromFPGA"]++;
#endif
    return ptr;
}

bool AddFCNOp(void * A, void * B, void *C, void * bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, short PReLUScale, short PReLUAlpha )
{
    gemx::XTimer t;
    //cout << C << " = " << A << " * " << B << " + " << bias << endl;
    gemx::FCNHost<short*>* fcn_ptr = static_cast< gemx::FCNHost<short*> *> (GEMXHostHandle<short*>::Instance().gh_ptr.get());
    bool ret = fcn_ptr->AddFCNOp((short*)A, (short*)B,(short*)C, (short*)bias, m,k,n, postScale, postShift, PReLUScale, PReLUAlpha);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["AddFCNOp"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["AddFCNOp"]++;
#endif
    return ret;
}

bool AddGEMMOp(void * A, void * B, void *C, void * bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift)
{
    //cout << C << " = " << A << " * " << B << " + " << bias << endl;
    return GEMXHostHandle<short*>::Instance().gh_ptr->AddGEMMOp((short*)A, (short*)B,(short*)C, (short*)bias, m,k,n, postScale, postShift);
}

void Execute (bool sync_exec)
{
    gemx::XTimer t;
    GEMXHostHandle<short*>::Instance().gh_ptr->Execute(sync_exec);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["Execute"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["Execute"]++;
#endif

}

void Wait ()
{
    gemx::XTimer t;
    GEMXHostHandle<short*>::Instance().gh_ptr->Wait();
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["Wait"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["Wait"]++;
#endif
}

void PrintStats(){
    for ( auto p : GEMXHostProfiler::Instance().func_time)
    {
        cout << p.first << ": " << (p.second * 1000.0) / GEMXHostProfiler::Instance().func_calls[p.first]  << " ms " << GEMXHostProfiler::Instance().func_calls[p.first] << " calls" << endl;
    }
}

int GetFreq(){
  return GEMXHostHandle<short*>::Instance().gh_ptr->getBoardFreqMHz(0);
}

//void DestroyGEMMHost(gemx::GEMMHost<short*> * ptr) {
//    delete ptr;
//}

void int16_gemm(short * A, short * B, short *X,
        short * C, unsigned int M, unsigned int K, unsigned int N ) {
    using namespace std;
    using namespace gemx;

    cout << "A_ptr: " << A << " B_ptr: " << B << " C_ptr: " << C << " X_ptr: " << X << endl;
    shared_ptr<GEMMHost<short*>> host_ptr = GEMXHostHandle<short*>::Instance().gh_ptr;

    host_ptr->SendToFPGA((short*)A, A, sizeof(short)*M*K);
    host_ptr->SendToFPGA((short*)B, B, sizeof(short)*K*N);
    host_ptr->SendToFPGA((short*)C, C, sizeof(short)*M*N);
    host_ptr->SendToFPGA((short*)X, X, sizeof(short)*M*N);

    host_ptr->AddGEMMOp(A, B, C, (short*)X, M,K,N, 1,0);

    host_ptr->Execute();

    host_ptr->GetMat(C, true, true);
}
