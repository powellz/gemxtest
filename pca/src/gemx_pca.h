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
 *  @brief PCA (Principal component analysis) 
 *
 *  $DateTime: 2018/05/15 11:58:31 $
 *  $Author: Xilinx $
 */
#ifndef GEMX_PCA_H
#define GEMX_PCA_H
#include "hls_math.h"
#include "gemx_spmv.h"
namespace gemx {
////////////////////////////////////////
//class Pca (principal component analysis)
//add normalization stage to SPMV C = A*B
////////////////////////////////////////
template <
	typename t_FloatType,
	typename t_FloatEqIntType,
	unsigned int t_DdrWidth,
	unsigned int t_SpmvWidth,
	unsigned int t_kVectorBlocks,
	unsigned int t_mVectorBlocks,
	unsigned int t_MacGroups,
	unsigned int t_ColAddIdxBits,
	unsigned int t_NumCblocks,
	unsigned int t_FloatPerDesc
>
class Pca {
	public:
		typedef PcaArgs PcaArgsType;
		typedef typename Spmv<t_FloatType, t_FloatEqIntType, t_DdrWidth, t_SpmvWidth, t_kVectorBlocks, t_mVectorBlocks, t_MacGroups, t_ColAddIdxBits, t_NumCblocks, t_FloatPerDesc>::DdrWideType DdrWideType;
		typedef typename Spmv<t_FloatType, t_FloatEqIntType, t_DdrWidth, t_SpmvWidth, t_kVectorBlocks, t_mVectorBlocks, t_MacGroups, t_ColAddIdxBits, t_NumCblocks, t_FloatPerDesc>::DdrWideStreamType DdrWideStreamType;

	private:
		static const unsigned int t_MaxTopK=32;
		static const unsigned int t_NumDdrPerTopK = t_MaxTopK / t_DdrWidth;
		static const unsigned int t_numDescPerDdr = t_DdrWidth / t_FloatPerDesc;
		static const unsigned int t_RowsInCblock = t_SpmvWidth * t_MacGroups * t_mVectorBlocks * t_DdrWidth;
		static const unsigned int t_NumDdrPerSpmv = t_DdrWidth / t_SpmvWidth;
		static const unsigned int t_DdrWordsPerBlock = t_SpmvWidth * t_MacGroups / t_DdrWidth;
		Spmv<t_FloatType, t_FloatEqIntType, t_DdrWidth, t_SpmvWidth, t_kVectorBlocks, t_mVectorBlocks, t_MacGroups, t_ColAddIdxBits, t_NumCblocks, t_FloatPerDesc> m_Spmv;
		t_FloatType m_Norm;
		t_FloatType m_MinK;
		t_FloatType m_SortMem[t_DdrWidth][t_NumDdrPerTopK*t_NumCblocks*t_DdrWidth];
		

		static const unsigned int t_Debug_runPca = 0; 
		static const unsigned int t_Debug_calcNormC = 0;
	
	private:
		void
		normZeroOutB(DdrWideStreamType &p_inS, DdrWideStreamType &p_outS, unsigned int p_kBlocks) {
			LOOP_PCA_NORMB:for(unsigned int i=0; i<p_kBlocks; ++i) {
			#pragma HLS PIPELINE REWIND
				DdrWideType l_val, l_valZeroOut, l_valNorm;
				p_inS.read(l_val);
				for(unsigned int j=0; j<t_DdrWidth; ++j){
					t_FloatType l_absVal = (l_val[j] < 0)? -l_val[j]: l_val[j];
					if (l_absVal < m_MinK) {
						l_valZeroOut[j] = 0;
					}
					else {
						l_valZeroOut[j] = l_val[j];
					}

					if (m_Norm !=0){
						l_valNorm[j] = l_valZeroOut[j]/m_Norm;
					}
					else {
						l_valNorm[j] = l_valZeroOut[j];
					}
				}
				p_outS.write(l_valNorm);
			}
		}

		void
		calcNormC(DdrWideStreamType &p_inS, DdrWideStreamType &p_outS, unsigned int p_mgdBlocks) {
			unsigned int l_blocks = p_mgdBlocks * t_DdrWordsPerBlock;
			DdrWideType l_sum;
			for (unsigned int i=0; i<t_DdrWidth; ++i){
			#pragma HLS UNROLL
				l_sum[i] = 0;
			}

			//unsigned int l_totalWord=0;//only used for debugging
			LOOP_PCA_CALCNORMC:for (unsigned int l_block=0; l_block < l_blocks; ++l_block) {
			#pragma HLS PIPELINE
				DdrWideType l_val;
				p_inS.read(l_val);
				for (unsigned int i=0; i<t_DdrWidth; ++i) {
				#pragma HLS UNROLL
					if (l_val[i] != 0) {
						l_sum[i] += l_val[i] * l_val[i];
					}
				}
				p_outS.write(l_val);
			}

			for (unsigned int i=0; i<t_DdrWidth; ++i) {
			#pragma HLS PIPELINE
				if (l_sum[i] != 0) {
					m_Norm += l_sum[i];
				}
			}
			
			DdrWideType l_zero(0);
			for (unsigned int i=0; i<t_MaxTopK; ++i){
				p_outS.write(l_zero);
			}

		}

		void 
		cmpStep(
			t_FloatType p_leftIn,
			t_FloatType &p_cur,
			t_FloatType &p_leftOut
		) {
			#pragma HLS inline self
			//p_leftOut = (p_leftIn > p_cur)? p_cur: p_leftIn;
			//p_cur = (p_leftIn > p_cur)? p_leftIn: p_cur;

			if (p_leftIn > p_cur) {
				p_leftOut = p_cur;
				p_cur = p_leftIn;
			}
			else if (p_leftIn < p_cur) {
				p_leftOut = p_leftIn;
			}
		}
		
		void
		sortTopKunit(DdrWideStreamType &p_inS, unsigned int p_mgdBlocks, unsigned int p_Block) {
			unsigned int l_blocks = p_mgdBlocks * t_DdrWordsPerBlock;
									 l_blocks += t_MaxTopK;
			DdrWideType l_absV;
			#pragma HLS ARRAY_PARTITION variable=l_absV dim=1 complete

			WideType<t_FloatType, t_MaxTopK> m_TopKs[t_DdrWidth];
			#pragma HLS ARRAY_PARTITION variable=m_TopKs dim=1 complete
			#pragma HLS ARRAY_PARTITION variable=m_TopKs dim=2 complete

			WideType<t_FloatType, t_MaxTopK+1> l_topKs[t_DdrWidth];
			#pragma HLS ARRAY_PARTITION variable=l_topKs dim=1 complete
			#pragma HLS ARRAY_PARTITION variable=l_topKs dim=2 complete
			for (unsigned int n=0; n<t_MaxTopK; ++n) {	
			#pragma HLS PIPELINE
				for (unsigned int i=0; i<t_DdrWidth; ++i) {
					(void)m_TopKs[i].shift(0);
					(void)l_topKs[i].shift(0);
				}
			}

			for(unsigned int i=0; i<l_blocks; ++i) {
			#pragma HLS PIPELINE
				DdrWideType l_val;
				p_inS.read(l_val);
				for(unsigned int j=0; j<t_DdrWidth; ++j){
					l_absV[j] = (l_val[j] < 0)? -l_val[j]: l_val[j];
				}
			
				LOOP_ZEROOUTK_SORT:for (unsigned int n=0; n<t_DdrWidth; ++n) {
					l_topKs[n][0] = l_absV[n];
					for (unsigned int k=0; k<t_MaxTopK; ++k) {
						cmpStep(l_topKs[n][k], m_TopKs[n][k], l_topKs[n][k+1]);
					}
				}	
			}

			//output partially sorted TopK
			for (unsigned int i=0; i< t_DdrWidth; ++i) {
				for (unsigned int j=0; j<t_NumDdrPerTopK; ++j){
				#pragma HLS PIPELINE
					DdrWideType l_val;
					unsigned int l_offset = p_Block * t_DdrWidth * t_NumDdrPerTopK + i*t_NumDdrPerTopK +j;
					for (unsigned int k=0; k<t_DdrWidth; ++k) {
						m_SortMem[k][l_offset] = m_TopKs[i][j*t_DdrWidth+k];
					}
				}
			}
		}

		void
		getMinK(unsigned int p_Blocks, unsigned int p_TopK) {
			WideType<t_FloatType, t_MaxTopK> l_topKs;
			WideType<t_FloatType, t_MaxTopK+1> l_helpKs(0);
			#pragma HLS ARRAY_PARTITION variable=l_topKs dim=1 complete
			#pragma HLS ARRAY_PARTITION variable=l_helpKs dim=1 complete

				if (t_Debug_runPca) {
					for (unsigned int i=0; i<p_Blocks; ++i) {
						for (unsigned int j=0; j<t_DdrWidth; ++j) {
							for (unsigned int k=0; k<p_TopK; ++k) {
								unsigned int bank = (i*t_DdrWidth*t_MaxTopK+j*t_MaxTopK+k) % t_DdrWidth;
								unsigned int offset = (i*t_DdrWidth*t_MaxTopK+j*t_MaxTopK+k) / t_DdrWidth;
								std::cout << "DEBUG:runPca " << "l_topKs["<<i<<"]["<<j<< "][" <<k << "]=" << std::setw(GEMX_FLOAT_WIDTH) << m_SortMem[bank][offset] << std::endl;
							}
						}
					}
				}
			//init l_topKs
			for (unsigned int i=0; i<t_NumDdrPerTopK; ++i) {
				for (unsigned int j=0; j<t_DdrWidth; ++j) {
					l_topKs[i*t_DdrWidth+j] = m_SortMem[j][i];
				}
			}

			//sorting
			unsigned int l_ques=p_Blocks*t_DdrWidth-1;

			for (unsigned int i=1; i<=l_ques; ++i) {
				t_FloatType l_minK = l_topKs[p_TopK-1];
				unsigned int offset=0;
				unsigned int bank=0;
				bool l_exit = false;
				while (!l_exit) {
					t_FloatType l_dat=m_SortMem[bank][i*t_NumDdrPerTopK+offset];
					if ((l_dat < l_minK) && (bank == 0) && (offset == 0)) {
						l_exit = true;	
					}
					else { 
						if (l_dat > l_minK) {
						#pragma HLS PIPELINE
							l_helpKs[0] = l_dat;
							for (unsigned int k=0; k<t_MaxTopK; ++k) {
								cmpStep(l_helpKs[k], l_topKs[k], l_helpKs[k+1]);
							}
						}
						bank = (bank+1)%t_DdrWidth;
						offset += (bank+1)/t_DdrWidth;
						l_exit = (offset == t_NumDdrPerTopK);
					}
				}
				//if (t_Debug_runPca) {
					//for (unsigned int j=0; j<p_TopK; ++j) {
						//std::cout << "DEBUG:runPca " << "l_topKs["<<j<< "]=" << std::setw(GEMX_FLOAT_WIDTH) << l_topKs[j] << std::endl;
					//}
				//}
			}

			for (unsigned int i=0; i<p_TopK; ++i) {
			#pragma HLS PIPELINE
				for (unsigned int k=0; k<t_MaxTopK; ++k) {
					cmpStep(l_helpKs[k], l_topKs[k], l_helpKs[k+1]);
				}
				l_helpKs[0] = 0;
			}

			m_MinK = l_topKs[p_TopK-1];
		}

	public:
		void
		loadNormB(DdrWideType *p_bAddr, unsigned int p_kBlocks) {
		#pragma HLS DATAFLOW
			DdrWideStreamType l_bS;
			#pragma HLS DATA_PACK variable=l_bS
			#pragma HLS STREAM variable=l_bS depth=4	
			
			DdrWideStreamType l_normBs;
			#pragma HLS DATA_PACK variable=l_normBs
			#pragma HLS STREAM variable=l_normBs depth=4	

			m_Spmv.loadB2Stream(p_bAddr, l_bS, p_kBlocks);
			normZeroOutB(l_bS, l_normBs, p_kBlocks);
			m_Spmv.storeBFromStream(l_normBs, p_kBlocks);
		}

		void
		storeCandCalcNorm(DdrWideType *p_cAddr, unsigned int p_mgdBlocks, unsigned int p_Block) {
		#pragma HLS DATAFLOW
			DdrWideStreamType l_cS;
			#pragma HLS DATA_PACK variable=l_cS
			#pragma HLS STREAM variable=l_cS depth=4
			DdrWideStreamType l_cZeroOutS;
			#pragma HLS DATA_PACK variable=l_cZeroOutS
			#pragma HLS STREAM variable=l_cZeroOutS depth=4

			m_Spmv.storeCandStreaming(p_cAddr, l_cS, p_mgdBlocks);
			calcNormC(l_cS, l_cZeroOutS, p_mgdBlocks);
			sortTopKunit(l_cZeroOutS, p_mgdBlocks, p_Block);
		}

		void
		runPca(
			DdrWideType *p_DdrRd,
			DdrWideType *p_DdrWr,
			PcaArgsType &p_Args,
			t_FloatType &p_Norm,
			t_FloatType &p_MinK
		){
      #pragma HLS inline off
			#pragma HLS ARRAY_PARTITION variable=m_SortMem dim=1 complete	
			assert(t_NumDdrPerTopK * t_DdrWidth == t_MaxTopK);

			m_Norm = p_Norm;
			m_MinK = p_MinK;
			// Load entire B into BRAM
			const unsigned int l_kBlocks = p_Args.m_K / t_DdrWidth;
      assert(l_kBlocks * t_DdrWidth == p_Args.m_K);
      assert(l_kBlocks <= t_kVectorBlocks * t_SpmvWidth);
      DdrWideType *l_bAddr = p_DdrRd + p_Args.m_Boffset * DdrWideType::per4k();
			unsigned int l_topK;
			l_topK = p_Args.m_TopK;
      loadNormB(l_bAddr, l_kBlocks); // in DDR units

			// Load C block descriptors
			const unsigned int l_Cblocks = p_Args.m_Cblocks;
      const unsigned int l_descDdrWords = (l_Cblocks + t_numDescPerDdr - 1) / t_numDescPerDdr;
      DdrWideType *l_dAddr = p_DdrRd + p_Args.m_Aoffset * DdrWideType::per4k();
      m_Spmv.loadD(l_dAddr, l_descDdrWords);  // in descriptor units
			m_Norm = 0;
			for (unsigned int l_Cblock = 0; l_Cblock < l_Cblocks; ++l_Cblock) {
				SpmvAdesc l_desc = m_Spmv.getDesc(l_Cblock);
				unsigned int l_nnz = l_desc.getNnz();
				const unsigned int t_mgdBlocks = t_RowsInCblock / (t_SpmvWidth * t_MacGroups);
				assert(t_mgdBlocks *  (t_SpmvWidth * t_MacGroups) == t_RowsInCblock);
				const unsigned int l_mgdBlocks = (l_Cblock < l_Cblocks - 1) ?
																						t_mgdBlocks :
																						(p_Args.m_M % t_RowsInCblock) / (t_SpmvWidth * t_MacGroups);
				assert((l_mgdBlocks == t_mgdBlocks) ||
							 (l_mgdBlocks *  (t_SpmvWidth * t_MacGroups) == (p_Args.m_M % t_RowsInCblock)));

				// Load C
				DdrWideType *l_cAddr = p_DdrWr + p_Args.m_Coffset * DdrWideType::per4k() +
															 l_Cblock * (t_RowsInCblock / t_DdrWidth) ;
				m_Spmv.initC(l_mgdBlocks);

				unsigned int l_blockAoffset = l_desc.getOffset();
				DdrWideType *l_aAddr = p_DdrRd + (p_Args.m_Aoffset + p_Args.m_DescPages + l_blockAoffset) *
															 DdrWideType::per4k();
				const unsigned int l_numWordsA = l_nnz * t_NumDdrPerSpmv / t_DdrWidth;
				assert(l_numWordsA * t_DdrWidth == l_nnz * t_NumDdrPerSpmv);
				m_Spmv.multA(l_aAddr, l_numWordsA);

				// Store C
				storeCandCalcNorm(l_cAddr, l_mgdBlocks, l_Cblock);
			}
			getMinK(l_Cblocks, l_topK);
			p_Norm = hls::sqrtf(m_Norm);
			p_MinK = m_MinK;
			if (t_Debug_runPca) {
				std::cout << "DEBUG:runPca " << "p_Norm = " << std::setw(GEMX_FLOAT_WIDTH) << p_Norm << std::endl;
				std::cout << "DEBUG:runPca " << "p_Mink = " << std::setw(GEMX_FLOAT_WIDTH) << p_MinK << std::endl;
			}
		}
};
}
#endif
