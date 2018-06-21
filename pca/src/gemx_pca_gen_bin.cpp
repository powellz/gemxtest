/**********
 * Copyright (c) 2017, Xilinx, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * **********/
/**
 *  @brief generate binary image from PCA command line inputs 
 *
 *  $DateTime: 2018/05/15 17:57:59 $
 */

// Fast compile and run:
//   make host

#include <stdio.h>
//#include <stdlib.h>
#include <string>
#include <vector>

#include <mm_malloc.h>
#include "gemx_pca_gen_bin.h"

int main(int argc, char** argv)
{
  if (argc < 3 ){
    printf("ERROR: passed %d arguments instead of %d, exiting\n",
           argc, 3);
    std::cout << "  Usage:\n    gemx_gen_bin.exe  <-write | -read> app.bin [op1 arg arg ...] [op2 arg arg ...] ... | -compare tol_rel tol_abs app_gold.bin app_out.bin\n"
              << "    Ops:\n"
              << "      pca   M  K  Nnz  TopK  mtxFile   HandleA HandleB HandleC\n"
              << "    Examples:\n"
              << "      gemx_gen_bin.exe -write app.bin pca 0 0 0 8 mario001.mtx A0 B0 C0 pca 38496 38496 114648 8 none A0 C0 C1\n"
              << "      gemx_gen_bin.exe -read app_gold.bin\n"
              << "      gemx_gen_bin.exe -read app_gold.bin\n"
              << "      gemx_gen_bin.exe -compare 1e-3 1e-9 app_gold.bin app_out.bin\n"
              << "\n";
    return EXIT_FAILURE;
  }
  
  std::string l_mode(argv[1]);
  bool l_write = l_mode == "-write";
  bool l_read = l_mode == "-read";
  bool l_compare = l_mode == "-compare";
  float l_TolRel = 0, l_TolAbs = 0;
  
  std::string l_binFile[2];
  
  if (l_read || l_write) {
    l_binFile[0] = argv[2];
    l_binFile[1] = l_binFile[0].substr(0, l_binFile[0].find_last_of(".")) + "_gold.bin";

    printf("GEMX:  %s %s %s\n",
           argv[0], l_mode.c_str(), l_binFile[0].c_str());
  } else if (l_compare) {
    std::stringstream l_TolRelS(argv[2]);
    std::stringstream l_TolAbsS(argv[3]);
    l_TolRelS >> l_TolRel;
    l_TolAbsS >> l_TolAbs;
    l_binFile[0] = argv[4];
    l_binFile[1] = argv[5];
    printf("GEMX:  %s %s %g %g %s %s\n",
           argv[0], l_mode.c_str(),
           l_TolRel, l_TolAbs,
           l_binFile[0].c_str(), l_binFile[1].c_str());
  } else {
    assert(0);
  }
  
  // Early assert for proper instruction length setting
  assert(sizeof(GEMX_dataType) * GEMX_ddrWidth* GEMX_argInstrWidth == GEMX_instructionSizeBytes); 

  ////////////////////////  TEST PROGRAM STARTS HERE  ////////////////////////
  
  GenControl l_control;
	GenPca l_pca;
	FloatType l_norm=0;
	FloatType l_minK=0;

  if (l_write) {
    ProgramType l_p[2];  // 0 - no golden, 1 with golden

    for(unsigned int wGolden = 0; wGolden < 2; ++wGolden) {
      
      unsigned int l_argIdx = 3;
      unsigned int l_instrCount = 0;
      
      while (l_argIdx < argc) {
        std::string l_opName(argv[l_argIdx++]);
        TimePointType l_t1 = std::chrono::high_resolution_clock::now(), l_t2;
        if (l_opName == "control") {
          bool l_isLastOp = atoi(argv[l_argIdx++]);
          bool l_noop = atoi(argv[l_argIdx++]);
          l_control.addInstr(l_p[wGolden], l_isLastOp, l_noop);
        } else if (l_opName == "pca") {
          unsigned int l_m = atoi(argv[l_argIdx++]);
          unsigned int l_k = atoi(argv[l_argIdx++]);
          unsigned int l_nnz = atoi(argv[l_argIdx++]);
          unsigned int l_topK = atoi(argv[l_argIdx++]);
          std::string l_mtxFileName(argv[l_argIdx++]);
          std::string l_handleA(argv[l_argIdx++]);
          std::string l_handleB(argv[l_argIdx++]);
          std::string l_handleC(argv[l_argIdx++]);
    			
					TimePointType l_tp[64];
    			unsigned int l_tpIdx = 0;
    			l_tp[l_tpIdx] = std::chrono::high_resolution_clock::now(); 
          MtxFile l_mtxFile(l_mtxFileName);
    			showTimeData("loadMtx", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;
          if (!l_pca.check(l_m, l_k, l_nnz, l_mtxFile)) exit(1);
          l_pca.addInstr(l_p[wGolden], l_norm, l_minK, l_m,  l_k, l_nnz, l_topK, l_mtxFile,
                          l_handleA, l_handleB, l_handleC, wGolden);
    			showTimeData("addInstr", l_tp[l_tpIdx], l_tp[l_tpIdx+1]); l_tpIdx++;
       } else {
         std::cerr << "ERROR: unknow op \"" << l_opName << "\"\n";
       }
       l_instrCount++;
       assert(l_instrCount < GEMX_numInstr - 1); // 1 is for the mandatory control instruction
       assert(l_argIdx <= argc);
       if (wGolden) {
         showTimeData("  " + l_opName + " with golden took ", l_t1, l_t2);
       } else {
         std::cout << "\n";
       }
     }
     // Fill noops (workaround for HLS issue with dataflow loops)
     while (l_instrCount < GEMX_numInstr - 1) {
       l_control.addInstr(l_p[wGolden], false, true);
       std::cout << "\n";
       l_instrCount++;
     }
     
     l_control.addInstr(l_p[wGolden], true, false);
     std::cout << "\n";
     l_instrCount++;
     assert(l_instrCount == GEMX_numInstr);
     
     l_p[wGolden].writeToBinFile(l_binFile[wGolden]);
   }

  } else if (l_read) {
    
    // Read file
    ProgramType l_p;
    l_p.readFromBinFile(l_binFile[0]);

    // Show cycle counts
    KargsType l_kargsRes;
    std::cout << "\nINFO:   format "
              << std::right << std::setw(4)  << "op"
              << std::right << std::setw(12) << "start"
              << std::right << std::setw(12) << "end"
              << std::right << std::setw(12) << "duration"
              << std::right << std::setw(14) << "ms@250MHz"
              << "\n";
    for (unsigned int l_pc = 0; l_pc < GEMX_numInstr; ++l_pc) {
      KargsOpType l_op = l_kargsRes.load(l_p.getBaseResAddr(), l_pc * l_kargsRes.getInstrWidth());
      assert(l_op == KargsType::OpResult || l_op == KargsType::OpControl); // OpControl is 0 which is ok
      gemx::InstrResArgs l_instrRes = l_kargsRes.getInstrResArgs();
      std::cout << "  DATA: cycles "
                << std::setw(4) << l_pc
                << std::setw(12) << l_instrRes.m_StartTime
                << std::setw(12) << l_instrRes.m_EndTime
                << std::setw(12) << l_instrRes.getDuration()
                << std::setw(14) << std::fixed << std::setprecision(6) << (l_instrRes.getDuration() / 250e6 * 1e3)
                << "\n";
    }
    std::cout << "\n";
    
    // Show all instructions
    KargsType l_kargs;
    unsigned int l_pc = 0;
    bool l_isLastOp = false;
    do {
      KargsOpType l_op = l_kargs.load(l_p.getBaseInstrAddr(), l_pc);
      switch(l_op) {
        case KargsType::OpControl: {
          ControlArgsType l_controlArgs = l_kargs.getControlArgs();
          l_isLastOp = l_controlArgs.getIsLastOp();
          bool l_noop = l_controlArgs.getNoop();
          assert(l_isLastOp || l_noop);
          break;
        }
        case KargsType::OpPca: {
          PcaArgsType l_pcaArgs = l_kargs.getPcaArgs();
          l_pca.show(l_p, l_pcaArgs);
          break;
        }
        default: {
          assert(false);
        }
      }
      l_pc += l_kargs.getInstrWidth();
    } while(!l_isLastOp);
    
  } else if (l_compare) {
    // Read files
    ProgramType l_p[2];
    l_p[0].readFromBinFile(l_binFile[0]);
    l_p[1].readFromBinFile(l_binFile[1]);
    
    // Compare all instructions
    KargsType l_kargs0, l_kargs1;
    unsigned int l_pc = 0;
    bool l_isLastOp = false;
    bool l_compareOk = true;
    do {
      KargsOpType l_op0 = l_kargs0.load(l_p[0].getBaseInstrAddr(), l_pc);
      KargsOpType l_op1 = l_kargs1.load(l_p[1].getBaseInstrAddr(), l_pc);
      if (l_op1 == KargsType::OpResult) {
        break;
      }
      assert(l_op0 == l_op1);
      switch(l_op0) {
        case KargsType::OpControl: {
          ControlArgsType l_controlArgs = l_kargs0.getControlArgs();
          l_isLastOp = l_controlArgs.getIsLastOp();
          break;
        }
        case KargsType::OpPca: {
          PcaArgsType l_pcaArgs = l_kargs0.getPcaArgs();
          bool l_opOk = l_pca.compare(l_TolRel, l_TolAbs, l_p[0], l_p[1], l_pcaArgs);
          l_compareOk = l_compareOk && l_opOk;
          break;
        }
        default: {
          assert(false);
        }
      }
      l_pc += l_kargs0.getInstrWidth();
    } while(!l_isLastOp);
    
    // Exit status from compare
    if (!l_compareOk) {
      return EXIT_FAILURE;
    }
  
  } else {
    assert(0); // Unknown user command
  }
  
  return EXIT_SUCCESS;
}

  
