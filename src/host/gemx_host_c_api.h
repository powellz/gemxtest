// namespace

extern "C" {

void MakeFCNHost(char *xclbin, char* device, unsigned int nPE);
void MakeGEMMHost(char *xclbin, char* device, unsigned int nPE);
//void MakeSPMVHost(char *xclbin, char * kernName, char* device);

void SendToFPGAShrt(short *A,  unsigned long long num_elem, unsigned PE, bool sync_send);
void SendToFPGAInt(int *A,  unsigned long long num_elem, unsigned PE, bool sync_send);
//void SendToFPGAShrt_dbg( char * name, short *A, int m, int n, bool sync_send);
//void SendToFPGAInt_dbg( char * name, int *A, int m, int n, bool sync_send);

void* GetFromFPGA( short *A, unsigned PE, bool sync_get);

void Wait (unsigned PE);
void PrintStats();
bool AddFCNOp( void * A, void * B, void *C, void * bias,  unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, short PReLUScale, short PReLUAlpha, unsigned PE);
bool AddGEMMOp( void * A, void * B, void *C, void * bias,  unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, unsigned PE);

int GetFreq ();
void Execute (bool sync_exec, unsigned PE);

void int16_gemm(short * A, short * B, short * X, short *C, unsigned int M, unsigned int K, unsigned int N );

}
