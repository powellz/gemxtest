#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <stdio.h>  // fgets for popen

#include "gemx_pca_kernel.h"
#include "gemx_pca_fpga.h"
#include "gemx_pca_gen_bin.h"

//#define VERBOSE 0 

float getBoardFreqMHz(unsigned int p_BoardId) {
  std::string l_freqCmd = "$XILINX_OPENCL/runtime/bin/xbsak query -d" + std::to_string(p_BoardId);;
  float l_freq = -1;
  char l_lineBuf[256];
  std::shared_ptr<FILE> l_pipe(popen(l_freqCmd.c_str(), "r"), pclose);
  if (!l_pipe) std::cout << ("ERROR: popen(" + l_freqCmd + ") failed");
  bool l_nextLine_isFreq = false;
  while (l_pipe && fgets(l_lineBuf, 256, l_pipe.get()) ) {
    std::string l_line(l_lineBuf);
    //std::cout << "DEBUG: read line " << l_line << std::endl;
    if (l_nextLine_isFreq) {
      std::string l_prefix, l_val, l_mhz;
      std::stringstream l_ss(l_line);
      l_ss >> l_prefix >> l_val >> l_mhz;
      l_freq = std::stof(l_val);
      assert(l_mhz == "MHz");
      break;
    } else if (l_line.find("OCL Frequency:") != std::string::npos) {
      l_nextLine_isFreq = true;
    }
  }
  if (l_freq == -1) {
	//if xbsak does not work, user could put the XOCC achieved kernel frequcy here
	//l_freq = 200.2;
    std::cout << "INFO: Failed to get board frequency by xbsak. This is normal for cpu and hw emulation, using -1 MHz for reporting.\n";
  }
  return(l_freq);
}

int main(int argc, char **argv)
{
  //############  UI and PCA problem size  ############
  if (argc < 10) {
    std::cerr << "Usage:\n"
              <<  "  gemx_api_pca.exe <path/gemx.xclbin> [M K Nnz topK mtxFile A_handle B_handle C_handle]\n"
	      <<  "  Examples:\n"
	      <<  "    gemx_api_pca.exe   out_hw/gemx.xclbin 0 0 0 8 ../data/spmv/mario001.mtx A0 B0 C0 38496 38496 114648 8 none A0 C0 C1\n";
    exit(2);
  }
  if((argc - 2) % 8 != 0) {
        std::cerr << "  If enter more than one instructions, for each instruction, [M K Nnz topK mtxFile A_handle B_handle C_handle] could not be missing\n";
        exit(2);
  }
  unsigned int l_argIdx = 1;
  std::string l_xclbinFile(argv[l_argIdx]);
  unsigned int l_instrCount = ((argc-2)/8>1)?((argc-2)/8):1; //number of instructions
  if(l_instrCount > 15){
    std::cerr << "  Too many instructions at same time\n";
    exit(2);
  }
  unsigned int l_ddrW = GEMX_ddrWidth;
  unsigned int l_m[l_instrCount], l_k[l_instrCount], l_nnz[l_instrCount], l_topK[l_instrCount];
  std::string l_mtxFileName("none");
  GEMX_dataType l_norm = 0;
  GEMX_dataType l_minK = 0;
  
  printf("GEMX-pca C++ API example using accelerator image \n",
       l_xclbinFile.c_str());
       
  GenPca l_pca;
  //############  Client code - prepare the pca problem input  ############
  ProgramType l_program[GEMX_numKernels];  // Holds instructions and controls memory allocation

  for (int i=0; i<GEMX_numKernels; ++i) {
    l_argIdx = 2;
    for(int index = 0; index < l_instrCount; index++){
       l_m[index] = atoi(argv[l_argIdx++]);
       l_k[index] = atoi(argv[l_argIdx++]);
       l_nnz[index] = atoi(argv[l_argIdx++]);
       l_topK[index] = atoi(argv[l_argIdx++]);
       l_mtxFileName = argv[l_argIdx++];
       MtxFile l_mtxFile(l_mtxFileName);
       //The check() modifies the dimensions when loading from a matrix file. Please use 0 for l_M, l_K and l_NNZ when provding matrix file
       l_pca.check(l_m[index], l_k[index], l_nnz[index], l_mtxFile);
       std::string l_handleA = argv[l_argIdx++];
       std::string l_handleB = argv[l_argIdx++];
       std::string l_handleC = argv[l_argIdx++];
       l_pca.addInstr(l_program[i], l_norm, l_minK, l_m[index], l_k[index], l_nnz[index], l_topK[index], l_mtxFile, l_handleA, l_handleB, l_handleC, false);
    }
  }
  std::string kernelNames[GEMX_numKernels];
  gemx::MemDesc l_memDesc[GEMX_numKernels];
 
  for (int i=0; i<GEMX_numKernels; ++i) { 
    l_memDesc[i] = l_program[i].getMemDesc();
  }
  
  //############  Runtime reporting Infra  ############
  TimePointType l_tp[10];
  unsigned int l_tpIdx = 0;
  l_tp[l_tpIdx] = std::chrono::high_resolution_clock::now(); 

  //############  Run FPGA accelerator  ############
  // Init FPGA
  gemx::Fpga l_fpga;

  for (int i=0; i<GEMX_numKernels; ++i){
	kernelNames[i] = "gemxKernel_" + std::to_string(i);
  }
  if (l_fpga.loadXclbin(l_xclbinFile, kernelNames)) {
    std::cout << "INFO: created kernels" << std::endl;
  } else {
    std::cerr << "ERROR: failed to load " + l_xclbinFile + "\n";
    return EXIT_FAILURE;
  }
  showTimeData("loadXclbin", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

  //create buffers for transferring data to FPGA
  if (!l_fpga.createBuffers(l_memDesc)) {
    std::cerr << "ERROR: failed to create buffers for transffering data to FPGA DDR\n";
    return EXIT_FAILURE;
  }
  showTimeData("created buffers", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;
  
  // Transfer data to FPGA
  if (l_fpga.copyToFpga()) {
    (VERBOSE > 0) && std::cout << "INFO: transferred data to FPGA" << std::endl;
  } else {
    std::cerr << "ERROR: failed to copy data to FPGA DDR\n";
    return EXIT_FAILURE;
  }
  showTimeData("copyToFpga", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

  // Gemx kernel ops
  if (l_fpga.callKernels()) {
    (VERBOSE > 0) && std::cout << "INFO: Executed kernel" << std::endl;
  } else {
    std::cerr << "ERROR: failed to call kernels ";
	for (int i=0; i<GEMX_numKernels; ++i) {
		std::cerr << kernelNames[i] << " ";
	}
	std::cerr << "\n";
    return EXIT_FAILURE;
  }
  showTimeData("callKernel", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;

  // Transfer data back to host - due to lazy evaluation this is generally wheer the accelerator performs the work
  if (l_fpga.copyFromFpga()) {
    (VERBOSE > 0) && std::cout << "INFO: Transferred data from FPGA" << std::endl;
  } else {
    std::cerr << "ERROR: failed to copy data from FPGA DDR\n";
    return EXIT_FAILURE;
  }
  showTimeData("copyFromFpga", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;
  showTimeData("total", l_tp[0], l_tp[l_tpIdx]); l_tpIdx++;
  double l_timeApiInMs = -1;
  showTimeData("subtotalFpga", l_tp[2], l_tp[l_tpIdx], &l_timeApiInMs); l_tpIdx++; // Host->DDR, kernel, DDR->host
  
  //############  Get the exact kernel time from HW cycle counters on the accelerator  ############
  float l_boardFreqMHz = getBoardFreqMHz(0);
  
  //unsigned long int l_Ops = 2ull * l_NNZ;
  //unsigned long int theory_cycles = 2 * l_M / 16 + l_K / 16 + l_NNZ / 8;
  
  unsigned long int l_total_Ops = 0;
  unsigned long int l_total_theory_cycles = 0;
  for(int j=0;j<l_instrCount;++j){
    l_total_Ops += 2ull * l_nnz[j] + 4 * l_m[j];
    l_total_theory_cycles += 2 * l_m[j] / 16 + l_k[j] / 16 + l_nnz[j] / 8 + 4 * l_m[j] / 16;
  }
  double l_effCycles;
  KargsType l_kargsRes[GEMX_numKernels];
  KargsOpType l_op[GEMX_numKernels];
  gemx::InstrResArgs l_instrRes[GEMX_numKernels];
  unsigned long int l_cycleCount[GEMX_numKernels];
  unsigned long int l_maxCycleCount=0;
  double l_timeKernelInMs[GEMX_numKernels];
  double l_maxTimeKernelInMs=0;
  double l_perfKernelInGops[GEMX_numKernels];
  double l_totalPerfKernelInGops=0;
  double l_perfApiInGops;
  double l_timeMsAt100pctEff;
  //double l_effKernelPct;
  double l_effApiPct;
  
  for (int i=0; i<GEMX_numKernels; ++i) {
        l_cycleCount[i] = 0;
        for(int j=0;j<l_instrCount;++j){ //number of instructions
            l_op[i*l_instrCount+j] = l_kargsRes[i].load(l_program[i].getBaseResAddr(), j * l_kargsRes[i].getInstrWidth());
            //l_op[i*2+j] = l_kargsRes[i].load(l_program[i].getBaseResAddr(), j);
            assert(l_op[i*l_instrCount+j] == KargsType::OpResult);
            l_instrRes[i*l_instrCount+j] = l_kargsRes[i].getInstrResArgs();
            l_cycleCount[i] += l_instrRes[i*l_instrCount+j].getDuration();
            std::cout << std::string("cycles in kernel ") <<i<<"  "<< l_instrRes[i*l_instrCount+j].getDuration() <<std::endl;
        }
        l_maxCycleCount = (l_cycleCount[i] > l_maxCycleCount)? l_cycleCount[i]: l_maxCycleCount;
        l_timeKernelInMs[i] = l_cycleCount[i] / (l_boardFreqMHz * 1e6) * 1e3;
        l_maxTimeKernelInMs = (l_timeKernelInMs[i] > l_maxTimeKernelInMs)? l_timeKernelInMs[i]: l_maxTimeKernelInMs;
        l_perfKernelInGops[i] = l_total_Ops / (l_timeKernelInMs[i] * 1e-3) / 1e9;
        l_totalPerfKernelInGops += l_perfKernelInGops[i];
  }
  l_perfApiInGops = (l_total_Ops*GEMX_numKernels) / (l_timeApiInMs * 1e-3) / 1e9;
  l_timeMsAt100pctEff = l_total_theory_cycles / (l_boardFreqMHz * 1e6) * 1e3;
  //l_effKernelPct = 100 * l_timeMsAt100pctEff / l_maxTimeKernelInMs;
  l_effCycles = 100 * l_total_theory_cycles / l_maxCycleCount;
  l_effApiPct = 100 * l_timeMsAt100pctEff / l_timeApiInMs;
  // Show time, Gops in csv format
  std::cout <<"In each kernel, it ran "<<l_instrCount<<" instructions, size for matrices are: " <<"\n";
  for(int i=0;i<l_instrCount;++i){
     std::cout <<"m,k,nnz: "<<l_m[i]<<","<<l_k[i]<<","<<l_nnz[i]<<" \n";
  }
  std::cout << std::string("DATA_CSV:,DdrWidth,Freq,")
             + "KernelCycles,"
             + "TimeKernelMs,TimeApiMs,"
             + "EffKernelPct,EffApiPct,"
             + "PerfKernelGops,PerfApiGops\n"
            << "DATA_CSV:," <<  GEMX_ddrWidth << "," << l_boardFreqMHz << ","
            << l_maxCycleCount << ","
            << l_maxTimeKernelInMs << "," << l_timeApiInMs << ","
            << l_effCycles<<","<<l_effApiPct<<","
            << l_totalPerfKernelInGops << "," << l_perfApiInGops
            << std::endl;

  return EXIT_SUCCESS;
}