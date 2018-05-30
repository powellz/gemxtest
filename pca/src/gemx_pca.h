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
		static const unsigned int t_MaxTopK=256;
		static const unsigned int t_numDescPerDdr = t_DdrWidth / t_FloatPerDesc;
		static const unsigned int t_RowsInCblock = t_SpmvWidth * t_MacGroups * t_mVectorBlocks * t_DdrWidth;
		static const unsigned int t_NumDdrPerSpmv = t_DdrWidth / t_SpmvWidth;
		static const unsigned int t_DdrWordsPerBlock = t_SpmvWidth * t_MacGroups / t_DdrWidth;
		Spmv<t_FloatType, t_FloatEqIntType, t_DdrWidth, t_SpmvWidth, t_kVectorBlocks, t_mVectorBlocks, t_MacGroups, t_ColAddIdxBits, t_NumCblocks, t_FloatPerDesc> m_Spmv;
		t_FloatType m_Norm;
		t_FloatType m_MinK;

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
				//l_totalWord++;
				//t_Debug_calcNormC && std::cout <<"DEBUG: calcNormC " << "After adding entry " << int(l_totalWord)
					//		<< " with value " << l_val << "\n"
						//	<< " l_sum = " << l_sum << std::endl;
			}
			for (unsigned int i=0; i<t_DdrWidth; ++i) {
			#pragma HLS PIPELINE
				if (l_sum[i] != 0) {
					m_Norm += l_sum[i];
				}
			}
			t_Debug_calcNormC && std::cout <<"DEBUG: calcNormC " << "add l_sum together = " << m_Norm << std::endl;
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
			else {
				p_leftOut = p_leftIn;
			}
		}
		
		void
		zeroOutK(DdrWideStreamType &p_inS, unsigned int p_mgdBlocks, unsigned int p_TopK, bool p_isFirstCall) {
			unsigned int l_blocks = p_mgdBlocks * t_DdrWordsPerBlock;
			DdrWideType l_absV;
			#pragma HLS ARRAY_PARTITION variable=l_absV dim=1 complete

			static WideType<t_FloatType, t_MaxTopK> m_TopKs;
			#pragma HLS ARRAY_PARTITION variable=m_TopKs dim=1 complete

			WideType<t_FloatType, t_MaxTopK+1> l_topKs;
			#pragma HLS ARRAY_PARTITION variable=l_topKs dim=1 complete
			
			if (p_isFirstCall) {
				for (unsigned int i=0; i<t_MaxTopK; ++i) {
				#pragma HLS UNROLL
					m_TopKs[i] = 0;
				}
			}

			for (unsigned int i=0; i<t_MaxTopK; ++i) {
			#pragma HLS UNROLL
				l_topKs[i] = 0;
			}

			for(unsigned int i=0; i<l_blocks; ++i) {
				DdrWideType l_val;
				p_inS.read(l_val);
				BoolArr<t_DdrWidth> l_isZero(false);
				for (unsigned int j=0; j<t_DdrWidth; ++j) {
				#pragma HLS UNROLL
					l_isZero[j] = (l_val[j] == 0);
				}
				for(unsigned int j=0; j<t_DdrWidth; ++j){
				#pragma HLS UNROLL
					l_absV[j] = (l_val[j] < 0)? -l_val[j]: l_val[j];
				}
			
				unsigned int l_id=0;
				if (!l_isZero.Or()) {
					LOOP_ZEROOUTK_SORT:for (unsigned int n=0; n<t_DdrWidth; ++n) {
						t_FloatType l_cmpVal = l_absV[0];
						l_topKs[0] = l_absV[0];

						if (l_cmpVal > m_TopKs[p_TopK-1]){
							for (unsigned int k=0; k<t_MaxTopK; ++k) {
							#pragma HLS UNROLL
								cmpStep(l_topKs[k], m_TopKs[k], l_topKs[k+1]);
							}
						}
						(void)l_absV.unshift();
					}	
				}
			}

			//if ((l_topKs[0] > 4.1333) && (m_TopKs[16] < 4.1333)) {
				//t_Debug_runPca && std::cout << "DEBUG:runPca " << "l_topKs[0] = " << std::setw(GEMX_FLOAT_WIDTH) << l_topKs[0] << "m_TopKs[16] = " << m_TopKs[16] <<  std::endl;
			//}

			//flush out pipeline
			for (unsigned int n=0; n<t_DdrWidth; ++n) {
				l_topKs[0] = 0;
				for (unsigned int k=0; k<t_MaxTopK; ++k) {
					cmpStep(l_topKs[k], m_TopKs[k], l_topKs[k+1]);
				}
			}
			m_MinK = m_TopKs[p_TopK-1];
			if (t_Debug_runPca) {
				for (unsigned int i=0; i<p_TopK; ++i) {
					std::cout << "DEBUG:runPca " << "m_TopKs[" << i << "]=" << std::setw(GEMX_FLOAT_WIDTH) << m_TopKs[i] << std::endl;
				}
			}
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
		storeCandCalcNorm(DdrWideType *p_cAddr, unsigned int p_mgdBlocks, unsigned int p_TopK, bool p_isFirstCall) {
		#pragma HLS DATAFLOW
			DdrWideStreamType l_cS;
			#pragma HLS DATA_PACK variable=l_cS
			#pragma HLS STREAM variable=l_cS depth=4
			DdrWideStreamType l_cZeroOutS;
			#pragma HLS DATA_PACK variable=l_cZeroOutS
			#pragma HLS STREAM variable=l_cZeroOutS depth=4

			m_Spmv.storeCandStreaming(p_cAddr, l_cS, p_mgdBlocks);
			calcNormC(l_cS, l_cZeroOutS, p_mgdBlocks);
			zeroOutK(l_cZeroOutS, p_mgdBlocks, p_TopK, p_isFirstCall);
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
				bool l_isFirstCall = (l_Cblock == 0);
				storeCandCalcNorm(l_cAddr, l_mgdBlocks, l_topK, l_isFirstCall);
			}
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
