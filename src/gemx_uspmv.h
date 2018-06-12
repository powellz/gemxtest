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
 *  @brief Sparse matrix vector multiply  C += A * B
 *  sparse matrix A is stored in Coordinate Formate (COO): 32-bit row, 32-bit col, nnz val
 *	vector B and C will be read from DDR and stored into URAM
 *
 *  $DateTime: 2018/06/06 09:20:31 $
 */

#ifndef GEMX_USPMV_H
#define GEMX_USPMV_H

#include "assert.h"
#include "hls_stream.h"
#include "gemx_types.h"
#include "gemx_kargs.h"

namespace gemx {

//assume sparse matrix input has been sorted along col indices.     
//assume index type and data type have some number of bits
template <
    typename t_FloatType,
		typename t_FloatEqIntType,
    unsigned int t_DdrWidth,       // DDR width in t_FloatType
		unsigned int t_NnzWords,			 // number of t_DdrWidth elements for block-wise A loader
    unsigned int t_KvectorBlocks,  // controls max size of K, max_K = t_KvectorBlocks * t_IdxDdrWidth
    unsigned int t_MvectorBlocks,  // controls max size of M, max_M = t_MvectorBlocks * t_IdxDdrWidth
		unsigned int t_NnzVectorBlocks, //controls max size of Nnz, max_Nnz = t_NnzVectorBlocks * t_DdrWidth
		unsigned int t_UspmvStages=1,
		unsigned int t_Interleaves=12 
  >
class Uspmv
{
 	private: 
		static const unsigned int t_FloatSize = sizeof(t_FloatType);
		static const unsigned int t_UramWidth = 8 / t_FloatSize;
		static const unsigned int t_NumUramPerDdr = t_DdrWidth / t_UramWidth; //number of URAM slices used to store one data DDR
		static const unsigned int t_DdrWidthMinusOne = t_DdrWidth -1;
    
  public:
		typedef WideType<t_FloatEqIntType, t_UramWidth> IdxUramWideType;
		typedef WideType<t_FloatType, t_UramWidth> DataUramWideType;

		typedef WideType<t_FloatEqIntType, t_DdrWidth> IdxDdrWideType;
		typedef WideType<t_FloatType, t_DdrWidth> DataDdrWideType;

		typedef WideType<IdxUramWideType, t_NumUramPerDdr> IdxUramWideType;
		typedef WideType<DataUramWideType, t_NumUramPerDdr> DataUramWideType;

    typedef hls::stream<DataDdrWideType> DataDdrWideStreamType;
		typedef hls::stream<IdxDdrWideType> IdxDdrWideStreamType;
		typedef hls::stream<bool>ControlStreamType;

  private:
		IdxUramWideType m_Acol[t_UspmvStages][t_NumUramPerDdr][t_KvectorBlocks];
		IdxUramWideType m_Arow[t_UspmvStages][t_NumUramPerDdr][t_MvectorBlocks];
		DataUramWideType m_Adata[t_UspmvStages][t_NumUramPerDdr][t_NnzVectorBlocks];
		t_FloatType m_Cdata[t_DdrWidth][t_Interleaves]
		
    static const unsigned int t_Debug = 0;

  private:
	//SPMV middle stage functions
		//read col indices from  URAM and form IdxWideStreamType
		//the col indices storage: starge col index, offset, offset, offset
		void
		loadA(DataDdrWideStream &p_outS, unsigned int p_nnzBlocks, unsigned int p_stageId) {
			DataDdrWideType l_val;
			#pragma HLS array_partition variable=l_val dim=1 complete
			DataUramWideType l_valUram[t_NumUramPerDdr];
			#pragma HLS array_partition variable=l_valUram dim=1 complete
			#pragma HLS array_partition variable=l_valUram dim=2 complete
			for (unsigned int i=0; i<p_nnzBlocks; ++i){
			#pragma HLS pipeline
				for (unsigned int b=0; b<t_NumUramPerDdr; ++b) {
					l_valUram[b] = m_Adata[p_stageId][b][i];
					for (unsigned int j=0; j<t_UramWidth; ++j) {
						l_val[b*t_UramWidth+j] = l_valUram[b][j];
					}
					p_outS.write(l_val);
				}
			}	
		}
		void
		loadCol(IdxDdrWideStreamType &p_outS, unsigned int p_nnzBlocks, unsigned int p_stageId) {
			IdxDdrWideType l_col;
			#pragma HLS array_partition variable=l_col dim=1 complete
			IdxUramWideType l_colUram[t_NumUramPerDdr];
			#pragma HLS array_partition variable=l_colUram dim=2 complete
			#pragma HLS array_partition variable=l_colUram dim=1 complete 
			for (unsigned int i=0; i<p_nnzBlocks; ++i){
			#pragma HLS pipeline
				for (unsigned int b=0; b<t_NumUramPerDdr; ++b) {
					l_colUram[b] = m_Acol[p_stageId][b][i];
					for (unsigned int j=0; j<t_UramWidth; ++j) {
						l_col[b*t_NumUramPerDdr+j] = l_colUram[b][j];
					}
				}
				p_outS.write(l_col);
			}
		}
		void
		pairB(IdxDdrWideStreamType &p_inIdxS, DataDdrWidStreamType &p_inDataS, DataDdrWideStream &p_outDataS, unsigned int p_nnzBlocks){
			unsigned int l_startCol=0;
			unsigned int l_endCol=t_DdrWidth;
			bool l_readCol=true;
			unsigned int l_nnzBlock=0;
			IdxDdrWideType l_col;
			DataDdrWideType l_b;
			DataDdrWideType l_val;
			BoolArr<t_DdrWidth> l_validVal(false);
			#pragma HLS array_partition variable=l_val dim=1 complete
		
			p_inDataS.read(l_b);	
			while (l_nnzBlock<p_nnzBlocks) {
			#pragma HLS PIPELINE
				if (l_readCol) {
					p_inIdxS.read(l_col);
					l_nnzBlock++;
				}
				else {
					for (unsigned int i=0; i<t_DdrWidth; ++i) {
						if (!l_validVal[i]) {
							l_col[i] -= t_DdrWidth;
						}
					}
					p_inDataS.read(l_b);
					l_startCol = l_endCol;
					l_endCol += t_DdrWidth;
				}
				for (unsigned int i=0; i<t_DdrWidth; ++i) {
					if (!l_validVal[i]) {
						l_validVal[i] = (l_col[i] < l_endCol);
						l_val[i] = l_b[l_col[i]];
					}
				}	
				l_readCol = l_validVal[t_DdrWidthMinusOne];
				if (l_validVal.Or()) {
					p_outDataS.write(l_val);
					l_validVal.Reset();
				}
			}
		}
		void
		multAB(DataDdrWideStream &p_inAs, DataDdrWideStream &p_inBs, DataDdrWideStream &p_outDataS, unsigned int p_nnzBlocks) {
			for (unsigned int i=0; i<p_nnzBlocks; ++i) {
			#pragma HLS pipeline
				DataDdrWideType l_valA;
				#pragma HLS array_partition variable=l_valA dim=1 complete
				DataDdrWideType l_valB;
				#pragma HLS array_partition varialble=l_valB dim=1 complete
				DataDdrWideType l_valOut;
				#pragma HLS array_partition variable=l_valOut dim=1 complete
				for (unsigned int j=0; j<t_DdrWidth; ++j) {
					l_valOut[j] = l_valA[j] * l_valB[j];
				}
				p_outDataS.write(l_valOut);
			}
		}
	public:
};

} // namespace
#endif

