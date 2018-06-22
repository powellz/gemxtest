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

typedef enum<MUL, ST> OpCode;
class Instruction {
public:
	OpCode m_Op;
	uint8_t m_StageId;
	unsigned int m_NnzBlocks;
	unsigned int m_KmBlocks; //blocks of data along K or M dimension. K should equal to M.
public:
	Instruction() {}
	Instruction (OpCode p_op, uint8_t t_StageId, unsigned int p_nnzBlocks, unsigned int p_kmBlocks)
		: m_Op(p_op), m_StageId(t_StageId), m_NnzBlocks(p_nnzBlocks) {}
};
 
template <
	typename t_FloatType,
	typename t_FloatEqIntType,
	unsigned int t_NumBanks,
	unsigned int t_Interleaves
>
class UspmvC
{
	private:
		t_FloatType m_ValC;
		t_FloatEqIntType m_Row;
	public:
		UspmvC(){}
		UspmvC(t_FloatType p_C, t_FloatEqIntType p_row)
			:m_ValC(p_C), m_Row(p_row) {}
		t_FloatEqIntType getRow() {return m_Row;}
		void setRow(t_FloatEqIntType p_row) {m_Row=p_row;}
		t_FloatType &getC() {return m_ValC;}
		t_FloatEqIntType getRowBank() {return (m_Row % t_NumBanks);}
		t_FloatEqIntType getRowGroup() {return ((m_Row / t_NumBanks) % t_Interleaves);}
		t_FloatEqIntType getRowOffset() {return m_Row / (t_NumBanks*t_Interleaves);}
    void setRowOffsetIntoRow(t_FloatEqIntType p_rowOffset) {m_Row = p_rowOffset;}
		void
		print(std::ostream& os) {
			os << std::setw(GEMX_FLOAT_WIDTH) << getRow() << " "
				 << std::setw(GEMX_FLOAT_WIDTH) << getC();
		}
};
template <typename T1, typename T2, unsigned int T3, unsigned int T4>
std::ostream& operator<<(std::ostream& os, UspmvC<T1, T2, T3, T4>& p_val) {
	p_val.print(os);
	return(os);
}
	
//assume sparse matrix input has been sorted along col indices.     
//assume index type and data type have some number of bits
template <
  typename t_FloatType,
	typename t_FloatEqIntType,
  unsigned int t_DdrWidth,       // DDR width in t_FloatType
	unsigned int t_NnzWords,			 // number of t_DdrWidth elements for block-wise A loader
  unsigned int t_KvectorBlocks,  // controls max size of K, max_K = t_KvectorBlocks * t_DdrWidth
  unsigned int t_MvectorBlocks,  // controls max size of M, max_M = t_MvectorBlocks * t_DdrWidth
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
		static const unsigned int t_Moffsets = t_MvectorBlocks / t_Interleaves;
		static const unsigned int t_NumAddrPerDdr = 16;
		static const unsigned int t_AddrBlocks = (t_UspmvStages + t_NumAddrPerDdr - 1) / t_NumAddrPerDdr;
   	static const unsigned int t_NumParams = 2; 
  public:
		typedef UspmvC<t_FloatType, t_FloatEqIntType, t_DdrWidth, t_Interleaves> UspmvCType;

		typedef WideType<t_FloatEqIntType, t_UramWidth> IdxUramWideType;
		typedef WideType<t_FloatType, t_UramWidth> DataUramWideType;

		typedef WideType<t_FloatEqIntType, t_DdrWidth> IdxDdrWideType;
		typedef WideType<t_FloatType, t_DdrWidth> DataDdrWideType;
		typedef WideType<UspmvCType, t_DdrWidth> UspmvCDdrWideType;

		typedef WideType<IdxUramWideType, t_NumUramPerDdr> IdxUramWideType;
		typedef WideType<DataUramWideType, t_NumUramPerDdr> DataUramWideType;

		typedef t_FloatEqIntType ParamType;
		typedef WideType<ParamType, t_NumParams> WideParamType;

		typedef unsigned int AddrType;
		typedef <AddrTyps, t_NumAddrPerDdr> AddrDdrWideType;

    typedef hls::stream<DataDdrWideType> DataDdrWideStreamType;
		typedef hls::stream<IdxDdrWideType> IdxDdrWideStreamType;
		typedef hls::stream<UspmvCDdrWideType> UspmvCDdrWideStreamType;
		typedef hls::stream<UspmvCType> UspmvCStreamType;
		typedef hls::stream<bool> ControlStreamType;

		typedef hls::stream<Instruction> InstrStreamType;
		typedef hls::stream<WideParamType> WideParamStreamType;

		typedef UspmvArgs UspmvArgsType;

  private:
		IdxUramWideType m_Acol[t_UspmvStages][t_NumUramPerDdr][t_KvectorBlocks];
		IdxUramWideType m_Arow[t_UspmvStages][t_NumUramPerDdr][t_MvectorBlocks];
		DataUramWideType m_Adata[t_UspmvStages][t_NumUramPerDdr][t_NnzVectorBlocks];
		t_FloatType m_Cdata[t_UspmvStages][t_DdrWidth][t_Interleaves][t_Moffsets];
		
    static const unsigned int t_Debug = 0;

  private:
	//SPMV middle stage functions
		//read col indices from  URAM and form IdxWideStreamType
		//the col indices storage: starge col index, offset, offset, offset
		void
		loadRow(IdxDdrWideStreamType &p_outS, unsigned int p_nnzBlocks, unsigned int t_StageId) {
			IdxDdrWideType l_row;
			#pragma HLS array_partition variable=l_row dim=1 complete
			IdxUramWideType l_rowUram[t_NumUramPerDdr];
			#pragma HLS array_partition variable=l_rowUram dim=1 complete
			#pragma HLS array_partition variable=l_rowUram dim=2 complete
			for (unsigned int i=0; i<p_nnzBlocks; ++i) {
			#pragma HLS pipeline
				for (unsigned int b=0; b<t_NumUramPerDdr; ++b) {
					l_rowUram[b] = m_Arow[t_StageId][b][i];
					for (unsigned int j=0; j<t_UramWidth; ++j) {
						l_row[b*t_NumUramPerDdr+j] = l_rowUram[b][j];
					}
				}
				p_outS.write(l_row);
			}
		}
		void
		loadA(DataDdrWideStream &p_outS, unsigned int p_nnzBlocks, unsigned int t_StageId) {
			DataDdrWideType l_val;
			#pragma HLS array_partition variable=l_val dim=1 complete
			DataUramWideType l_valUram[t_NumUramPerDdr];
			#pragma HLS array_partition variable=l_valUram dim=1 complete
			#pragma HLS array_partition variable=l_valUram dim=2 complete
			for (unsigned int i=0; i<p_nnzBlocks; ++i){
			#pragma HLS pipeline
				for (unsigned int b=0; b<t_NumUramPerDdr; ++b) {
					l_valUram[b] = m_Adata[t_StageId][b][i];
					for (unsigned int j=0; j<t_UramWidth; ++j) {
						l_val[b*t_UramWidth+j] = l_valUram[b][j];
					}
					p_outS.write(l_val);
				}
			}	
		}
		void
		loadCol(IdxDdrWideStreamType &p_outS, unsigned int p_nnzBlocks, unsigned int t_StageId) {
			IdxDdrWideType l_col;
			#pragma HLS array_partition variable=l_col dim=1 complete
			IdxUramWideType l_colUram[t_NumUramPerDdr];
			#pragma HLS array_partition variable=l_colUram dim=2 complete
			#pragma HLS array_partition variable=l_colUram dim=1 complete 
			for (unsigned int i=0; i<p_nnzBlocks; ++i){
			#pragma HLS pipeline
				for (unsigned int b=0; b<t_NumUramPerDdr; ++b) {
					l_colUram[b] = m_Acol[t_StageId][b][i];
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
		void
		formC(IdxDdrWidstreamType &p_inRowS, DataDdrWideStreamType &p_inValS, UspmvCDdrWideStreamType &p_outS, unsigned int p_nnzBlocks) {
			for (unsigned int i=0; i<p_nnzBlocks; ++i) {
			#pragma HLS pipeline
				UspmvCDdrWideType l_valOut;
				#pragma HLS array_partition variable=l_val dim=1 complete
				IdxDdrWideType l_idx;
				#pragma HLS array_partition variable=l_idx dim=1 complete
				DataDdrWideType l_valC;
				#pragma HLS array_partition variable=l_valC dim=1 complete
				l_idx = p_inRowS.read();
				l_valC = p_inValS.read();
				for (unsigned int b=0; b<t_DdrWidth; ++b) {
					l_valOut[b].getC()=l_valC[b];
					l_valOut[b].setRow(l_idx[b]);
				}
				p_outS.write(l_valOut);
			}
		}
    
    void
    xBarRowSplit(UspmvCDdrWideStreamType &p_inS,
                 UspmvCStreamType p_outDataS[t_DdrWidth][t_DdrWidth],
                 ControlStreamType &p_outCntS, unsigned int p_nnzBlocks) {
        #pragma HLS data_pack variable=p_inS
        #pragma HLS STREAM    variable=p_inS
        #pragma HLS data_pack variable=p_outDataS
        #pragma HLS STREAM    variable=p_outDataS

        LOOP_XC_WORDS:for(unsigned int i = 0; i < p_nnzBlocks; ++i) {
          #pragma HLS PIPELINE
          UspmvCDdrWideType l_val = p_inS.read();
          #pragma HLS array_partition variable=l_val COMPLETE

          LOOP_XC_W:for (int w = 0; w < t_DdrWidth; ++w) {
            #pragma HLS UNROLL
            unsigned int l_rowBank = l_val[w].getRowBank();
            p_outDataS[w][l_rowBank].write(l_val[w]);
            t_Debug && std::cout << "DEBUG: xBarColSplit " << " read " << l_val[w]
                                 << "  and sent it to col bank " << l_colBank << "\n";
          }
        }
        p_outCntS.write(true);
      }
      
			void
      xBarRowMerge(UspmvCStreamType p_inDataS[t_DdrWidth][t_DdrWidth],
                   ControlStreamType &p_inCntS,
                   UspmvCStreamType p_outDataS[t_DdrWidth],
                   ControlStreamType p_outCntS[t_DdrWidth])
      {
        #pragma HLS data_pack variable=p_inDataS
        #pragma HLS STREAM    variable=p_inDataS
        #pragma HLS data_pack variable=p_outDataS
        #pragma HLS STREAM    variable=p_outDataS

        bool l_exit = false, l_preDone = false;
        BoolArr<t_DdrWidth> l_activity(true);
        LOOP_XRM_WHILE:while (!l_exit) {
          #pragma HLS PIPELINE
          if (l_preDone  && !l_activity.Or()) {
            l_exit = true;
          }
          bool l_unused;
          if (p_inCntS.read_nb(l_unused)) {
            l_preDone = true;
          }
          l_activity.Reset();
          
          LOOP_XRM_BANK_MERGE:for (int b = 0; b < t_DdrWidth; ++b) {
            #pragma HLS UNROLL
            unsigned int l_idx = 0;
            LOOP_XRM_IDX:for (int bb = 0; bb < t_DdrWidth; ++bb) {
              #pragma HLS UNROLL
              unsigned int l_w = (bb + b ) % t_DdrWidth;
              if (!p_inDataS[l_w][b].empty()) {
                l_idx = l_w;
                break;
              }
            }

            UspmvCType l_val;
            if (p_inDataS[l_idx][b].read_nb(l_val)) {
              p_outDataS[b].write(l_val);
              l_activity[b] = true;
              t_Debug && std::cout << "DEBUG: xBarRowMerge bank " << b 
                                   << " read input position " << l_idx
                                   << " value " << l_val
                                   << "  and sent it to its bank\n" << std::flush;
            }
          }
        }
        LOOP_XRM_SEND_EXIT:for (int b = 0; b < t_DdrWidth; ++b) {
          #pragma HLS UNROLL
          p_outCntS[b].write(true);
        }
      }
    
			void
    	rowInterleave(UspmvCStreamType &p_inDataS,
                  ControlStreamType &p_inCntS,
                  UspmvCStreamType p_outDataS[t_Interleaves],
                  ControlStreamType &p_outCntS
                  ) { 
        #pragma HLS data_pack variable=p_inDataS
        #pragma HLS STREAM    variable=p_inDataS
        #pragma HLS data_pack variable=p_outDataS
        #pragma HLS STREAM    variable=p_outDataS

        bool l_exit = false, l_preDone = false;
        bool l_activity = true;

        LOOP_RI_WHILE:while (!l_exit) {
          #pragma HLS LOOP_TRIPCOUNT min=1 max=36870
          #pragma HLS PIPELINE
          
          if (l_preDone && !l_activity && p_inDataS.empty()) {
             l_exit = true;
          }
          bool l_unused = false;
          if (p_inCntS.read_nb(l_unused)) {
            l_preDone = true;
          }
          l_activity = false;
          
          UspmvCType l_val;
          if (p_inDataS.read_nb(l_val)) { 
            unsigned int l_rowGroup = l_val.getRowGroup();
            unsigned int l_rowOffset = l_val.getRowOffset();
            assert(l_rowOffset < t_Moffset);
            l_val.setRowOffsetIntoRow(l_rowOffset);
            p_outDataS[l_rowGroup].write(l_val);
            l_activity = true;
            t_Debug && std::cout << "DEBUG: rowInterleave bank " << t_BankId << " read " << l_val
                               << "  and sent it to row group " << l_rowGroup << "\n" << std::flush;
          }
        }
        p_outCntS.write(true);
      }
    
			void
    	rowUnit(
						UspmvCStreamType p_inDataS[t_Interleaves], 
            ControlStreamType &p_inCntS, 
						SpmvCStreamType p_outDataS[t_Interleaves],
						ControlStreamType &p_outCntS,
						unsigned int t_BankId) {

					UspmvCType gVal[t_Interleaves];
					#pragma HLS array_partition variable=gVal COMPLETE
					UspmvCType cVal[t_Interleaves];
					#pragma HLS array_partition variable=cVal COMPLETE
					
					LOOP_RU_INIT:for (int g = 0; g < t_Interleaves; ++g) {
						#pragma HLS UNROLL
						unsigned int l_initRow = t_BankId + g * t_DdrWidth;
						gVal[g] = UspmvCType(0, l_initRow);
					}
							
					bool l_exit = false, l_preDone = false;
					bool l_activity = true;
					unsigned int l_idleCounter = 0;

					LOOP_RU_WHILE:while (!l_exit) {
						#pragma HLS LOOP_TRIPCOUNT min=1 max=3072
						#pragma HLS PIPELINE
						
						if (l_preDone && !l_activity && streamsAreEmpty<UspmvCStreamType, t_Interleaves>(p_inDataS)) {
							 l_exit = true;
						}
						bool l_unused = false;
						if (p_inCntS.read_nb(l_unused)) {
							l_preDone = true;
						}
						l_activity = false;
						 
						LOOP_RU_G_CALC:for (int g = 0; g < t_Interleaves; ++g) {
							#pragma HLS UNROLL
							if (p_inDataS[g].read_nb(cVal[g])) {
								t_Debug && std::cout << "DEBUG: rowUnit " << t_BankId << " slot " << g
																		 << " read " << abVal[g] << "\n";
								t_Debug && std::cout << "DEBUG: rowUnit " << t_BankId << " slot " << g
												 << "  multiplied " << cVal[g].getC() << " = "
												 << abVal[g].getA() << " * " << abVal[g].getB() << "\n";
								l_activity = true;
							} else {
								cVal[g].getC() = 0;
							}
							if (cVal[g].getRow() != gVal[g].getRow()) {
								p_outDataS[g].write(gVal[g]);
								t_Debug && std::cout << "DEBUG: rowUnit " << t_BankId << " slot " << g
												 << "  sent out " << gVal[g] << "\n";
								gVal[g] = cVal[g];
							} else {
								gVal[g].getC() += cVal[g].getC();
								t_Debug && std::cout << "DEBUG: rowUnit " << t_BankId << " slot " << g
												 << "  added " << cVal[g] << " to local C " << gVal[g].getC() << "\n";
							}
						}
					}
					LOOP_RU_G_FLUSH:for (int g = 0; g < t_Interleaves; ++g) {
						if (gVal[g].getC() != 0) {
							p_outDataS[g].write(gVal[g]);
							t_Debug && std::cout << "DEBUG: rowUnit " << t_BankId << " slot " << g
												 << "  flushed out " << gVal[g] << "\n";
						}
					}
					p_outCntS.write(true);
				}
		
    void
    aggUnit(UspmvCStreamType p_inDataS[t_Interleaves], ControlStreamType &p_inCntS, unsigned int t_BankId, unsigned int t_StageId) {
				//init m_Cdata with 0s
				for (unsigned int i=0; i<t_Moffsets; ++i) {
				#pragma HLS pipeline
					for (unsigned int j=0; j<t_Interleaves; ++j) {
						for (unsigned int k=0; k<t_DdrWidth; ++k) {
							m_Cdata[t_StageId][k][j][i] = 0;
						}
					}
				}				

        bool l_exit = false;
        bool l_preDone = false;
        BoolArr<t_Interleaves> l_activity(true);
        #pragma HLS array_partition variable=l_activity COMPLETE

        UspmvCType cVal[t_Interleaves];
        #pragma HLS array_partition variable=cVal COMPLETE
        
        LOOP_AU_WHILE:while (!l_exit) {
          #pragma HLS LOOP_TRIPCOUNT min=1 max=604
          #pragma HLS PIPELINE II=12
          
          if (l_preDone && !l_activity.Or() && streamsAreEmpty<UspmvCStreamType, t_Interleaves>(p_inDataS) ) {
            l_exit = true;
          }
          bool l_unused = false;
          if (p_inCntS.read_nb(l_unused)) {
            l_preDone = true;
          }
          l_activity.Reset();
          
          LOOP_AU_G_CALC:for (int g = 0; g < t_Interleaves; ++g) {
            #pragma HLS UNROLL
            if (p_inDataS[g].read_nb(cVal[g])) {
              unsigned int l_rowOffset = cVal[g].getRow();
							t_FloatType l_valC = cVal[g].getC();
							if (l_valC != 0) {
								m_Cdata[t_StageId][t_BandId][g][l_rowOffset] += l_valC;
							}
              t_Debug && std::cout << "DEBUG: aggUnit " << t_BankId << " slot " << g
                                   << "  added " << cVal[g] << " to m_C "
                                   << getCref(t_BankId, g, l_rowOffset) << "\n";
              l_activity[g] = true;
              //assert(cVal[g].getRowBank() == t_BankId);
              //assert(cVal[g].getRowGroup() == g);
            }
          }
          
        }
      }

public:
		void 
		spmvDecode(InstrStreamType &p_inS, InstrStreamType &p_outS, 
							 WideParamStreamType &p_outParamS, 
								unsigned int t_StageId) {
			bool l_exit=false;
			do {
			#pragma HLS pipeline
				Instruction l_instr;
				l_instr = p_inS.read();
				l_exit = (l_instr.m_Op == Instruction::ST);
				uint8_t l_stageId = l_instr.m_StageId;
				ParamType l_nnzBlocks = l_instr.m_NnzBlocks;
				ParamType l_kmBlocks = l_instr.m_KmBlocks;
				WideParamType l_val;
				#pragma HLS array_partition variable=l_val dim=1 complete
				if (l_stageId == t_StageId) {
					l_val[0] = l_nnzBlocks;
					l_val[1] = l_nnzBlocks;
					p_outParamS.write(l_val);
				}
				else {
					p_outS.write(l_instr);
				}
			} while (!l_exit);
		}
		void
		spmvStFwdA (DataDdrWideStreamType &p_inDataS, ControlStreamType &p_inCntS,
								DataDdrWideStreamType &p_OutDataS,ControlStreamType &p_outCntS,
								ParamType p_nnzBlocks,
								unsigned int t_StageId){
			//store col
			for (unsigned int i=0; i<p_nnzBlocks; ++i) {
			#pragma HLS pipeline
				DataDdrWideType l_val;
				l_val = p_inDataS.read();
				WideConv<DataDdrWideType, IdxDdrWideType> l_conv;
				IdxDdrWideType l_col = l_conv.convert(l_val);
				#pragma HLS array_partition variable=l_col dim=1 complete
				IdxUramWideType l_uramCol[t_NumUramPerDdr];
				#pragma HLS array_partition variable=l_uramCol dim=1 complete
				#pragma HLS array_partition variable = l_uramCol dim=2 complete
				for (unsigned int j=0; j<t_NumUramPerDdr; ++j) {
					for (unsigned int k=0; k<t_UramWidth; ++k) {
						l_uramCol[j][k] = l_col[j*t_UramWidth+k];
					}
					m_Acol[t_StageId][j][i] = l_uramCol[j];
				}
			}
			//store row
			for (unsigned int i=0; i<p_nnzBlocks; ++i) {
			#pragma HLS pipeline
				DataDdrWideType l_val;
				l_val = p_inDataS.read();
				WideConv<DataDdrWideType, IdxDdrWideType> l_conv;
				IdxDdrWideType l_row = l_conv.convert(l_val);
				#pragma HLS array_partition variable=l_row dim=1 complete
				IdxUramWideType l_uramRow[t_NumUramPerDdr];
				#pragma HLS array_partition variable=l_uramRow dim=1 complete
				#pragma HLS array_partition variable = l_uramRow dim=2 complete
				for (unsigned int j=0; j<t_NumUramPerDdr; ++j) {
					for (unsigned int k=0; k<t_UramWidth; ++k) {
						l_uramRow[j][k] = l_row[j*t_UramWidth+k];
					}
					m_Arow[t_StageId][j][i] = l_uramRow[j];
				}
			}
			//store val 
			for (unsigned int i=0; i<p_nnzBlocks; ++i) {
			#pragma HLS pipeline
				DataDdrWideType l_val;
				l_val = p_inDataS.read();
				#pragma HLS array_partition variable=l_val dim=1 complete
				DataUramWideType l_uramVal[t_NumUramPerDdr];
				#pragma HLS array_partition variable=l_uramVal dim=1 complete
				#pragma HLS array_partition variable = l_uramVal dim=2 complete
				for (unsigned int j=0; j<t_NumUramPerDdr; ++j) {
					for (unsigned int k=0; k<t_UramWidth; ++k) {
						l_uramVal[j][k] = l_val[j*t_UramWidth+k];
					}
					m_AData[t_StageId][j][i] = l_uramVal[j];
				}
			}
			//forward the rest
			bool l_exit=false;
			bool l_final=false;
			while (!l_exit) { 
			#pragma HLS pipeline
				l_exit = p_inDataS.empty() && l_final;
				DataDdrWideType l_val;
				if (!p_inDataS.read(l_val)) {
					p_outDataS.write(l_val);
				}
				bool l_unused;
				if (p_inCntS.read_nb(l_unused)) {
					l_final=true;
				}
			}
			p_outCntS.write(true);
		}
 
		void
		spmvCompute(DataDdrWideStreamType &p_inBs, ParamType p_nnzBlocks, unsigned int t_StageId) {
		#pragma HLS dataflow
			static const unsigned int t_FifoDeep=16;
			static const unsigned int t_FifoShallow = 1;
			
			IdxDdrWideStreamType l_idx2pairB;
			#pragma HLS data_pack variable=l_idx2pairB
			#pragma HLS stream variable=l_idx2pairB depth=t_FifoShallow
			DataDdrWideStreamType l_dataA2multAB;
			#pragma HLS data_pack variable=l_dataA2multAB
			#pragma HLS stream variable=l_dataA2multAB depth=t_FifoShallow
			DataDdrWideStreamType l_dataB2multAB;
			#pragma HLS data_pack variable=l_dataB2multAB
			#pragma HLS stream variable=l_dataB2multAB depth=t_FifoShallow
			IdxDdrWideStreamType l_idx2formC;
			#pragma HLS data_pack variable=l_idx2formC
			#pragma HLS stream variable=l_idx2formC depth=t_FifoShallow
			DataDdrWideType l_data2formC;
			#pragma HLS data_pack variable=l_data2formC
			#pragma HLS stream variable=l_data2formC depth=t_FifoShallow
			UspmvCDdrWideStreamType l_data2xBarRowSplit;
			#pragma HLS data_pack variable=l_data2xBarRowSplit
			#pragma HLS stream variable=l_data2xBarRowSplit depth=t_FifoShallow
			UspmvCStreamType l_data2xBarRowMerge[t_DdrWidth][t_DdrWidth];
			#pragma HLS data_pack variable=l_data2xBarRowMerge
			#pragma HLS stream variable=l_data2xBarRowMerge depth=t_FifoDeep
			ControlStreamType l_cnt2xBarRowMerge;
			#pragma HLS data_pack variable=l_cnt2xBarRowMerge
			#pragma HLS stream variable=l_cnt2xBarRowMerge depth=t_FifoShallow
			UspmvCStreamType l_data2rowInterleave[t_DdrWidth];
			#pragma HLS data_pack variable=l_data2rowInterleave
			#pragma HLS stream variable=l_data2rowInterleave depth=t_FifoShallow
			ControlStreamType l_cnt2rowInterleave[t_DdrWidth];
			#pragma HLS data_pack variable=l_cnt2rowInterleave
			#pragma HLS stream variable=l_cnt2rowInterleave depth=t_FifoShallow
			UspmvCStreamType l_data2rowUnit[t_DdrWidth][t_Interleaves];
			#pragma HLS data_pack variable=l_data2rowUnit
			#pragma HLS stream variable=l_data2rowUnit depth=t_FifoDeep
			ControlStreamType l_cnt2rowUnit[t_DdrWidth];
			#pragma HLS data_pack variable=l_cnt2rowUnit
			#pragma HLS stream variable=l_cnt2rowUnit depth=t_FifoShallow
			UspmvCStreamType l_data2aggUnit[t_DdrWidth][t_Interleaves];
			#pragma HLS data_pack variable=l_data2aggUnit
			#pragma HLS stream variable=l_data2aggUnit depth=t_FifoDeep
			ControlStreamType l_cnt2aggUnit[t_DdrWidth]
			#pragma HLS data_pack variable=l_cnt2aggUnit
			#pragma HLS stream variable=l_cnt2aggUnit depth=t_FifoShallow

			
			loadCol(l_idx2pairB, p_nnzBlocks, t_StageId) 
			pairB(l_idx2pairB, p_inBs, DataDdrWideStream l_dataB2multAB, p_nnzBlocks)
			loadA(l_dataA2multAB, p_nnzBlocks, t_StageId); 
			multAB(l_dataA2multAB, l_dataB2multAB, l_data2formC, p_nnzBlocks) 
			loadRow(l_idx2formC, p_nnzBlocks, t_StageId); 
			formC(l_idx2formC, l_data2formC, l_data2xBarRowSplit, p_nnzBlocks); 
      xBarRowSplit(l_data2xBarRowSplit, l_data2xBarRowMerge, l_cnt2xBarRowMerge, p_nnzBlocks);
      xBarRowMerge(l_data2xBarRowMerge, l_cnt2RowMerge, l_data2rowInterleave, l_cnt2RowInterleave);

      LOOP_W_RU:for(int w = 0; w < t_SpmvWidth; ++w) {
        #pragma HLS UNROLL
        rowInterleave(l_data2rowInterleave[w], l_cnt2rowInterleave[w], l_data2rowUnit[w], l_cnt2rowUnit[w])
        rowUnit(l_data2rowUnit[w], l_cnt2rowUnit[w], l_data2aggUnit[w], l_cnt2aggUnit[w], w);
        aggUnit(l_data2aggUnit[w], l_cnt2aggUnit[w], w);
     }
		}

		//spmvStreamC: read m_Cdata and stream it out
		void
		spmvStreamC(DataDdrWideStreamType &p_outCs, ParamType p_mBlocks, unsigned int t_StageId){
			unsigned int l_offsets = p_mBlocks / t_Interleaves;
			assert(l_offset*t_Interleaves == p_mBlocks);
			for (unsigned int i=0; i<l_offsets; ++i) {
				for (unsigned int j=0; j<t_Interleaves; ++j) {
				#pragma HLS pipeline
					DdrWideType l_val;
					#pragma HLS array_partition variable=l_val dim=1 complete
					for (unsigned int k=0; k<t_DdrWidth; ++k) {
						l_val[k] = m_Cdata[t_StageId][k][j][i];	
					}
					p_outCs.write(l_val);
				}
			}
		}
		
		void
		spmvStoreA(DataDdrWideStreamType &p_inDataS, ControlStreamType &p_inCntS, ParamType p_nnzBlocks,
								unsigned int t_StageId){
			//store col
			for (unsigned int i=0; i<p_nnzBlocks; ++i) {
			#pragma HLS pipeline
				DataDdrWideType l_val;
				l_val = p_inDataS.read();
				WideConv<DataDdrWideType, IdxDdrWideType> l_conv;
				IdxDdrWideType l_col = l_conv.convert(l_val);
				#pragma HLS array_partition variable=l_col dim=1 complete
				IdxUramWideType l_uramCol[t_NumUramPerDdr];
				#pragma HLS array_partition variable=l_uramCol dim=1 complete
				#pragma HLS array_partition variable = l_uramCol dim=2 complete
				for (unsigned int j=0; j<t_NumUramPerDdr; ++j) {
					for (unsigned int k=0; k<t_UramWidth; ++k) {
						l_uramCol[j][k] = l_col[j*t_UramWidth+k];
					}
					m_Acol[t_StageId][j][i] = l_uramCol[j];
				}
			}
			//store row
			for (unsigned int i=0; i<p_nnzBlocks; ++i) {
			#pragma HLS pipeline
				DataDdrWideType l_val;
				l_val = p_inDataS.read();
				WideConv<DataDdrWideType, IdxDdrWideType> l_conv;
				IdxDdrWideType l_row = l_conv.convert(l_val);
				#pragma HLS array_partition variable=l_row dim=1 complete
				IdxUramWideType l_uramRow[t_NumUramPerDdr];
				#pragma HLS array_partition variable=l_uramRow dim=1 complete
				#pragma HLS array_partition variable = l_uramRow dim=2 complete
				for (unsigned int j=0; j<t_NumUramPerDdr; ++j) {
					for (unsigned int k=0; k<t_UramWidth; ++k) {
						l_uramRow[j][k] = l_row[j*t_UramWidth+k];
					}
					m_Arow[t_StageId][j][i] = l_uramRow[j];
				}
			}
			//store val 
			for (unsigned int i=0; i<p_nnzBlocks; ++i) {
			#pragma HLS pipeline
				DataDdrWideType l_val;
				l_val = p_inDataS.read();
				#pragma HLS array_partition variable=l_val dim=1 complete
				DataUramWideType l_uramVal[t_NumUramPerDdr];
				#pragma HLS array_partition variable=l_uramVal dim=1 complete
				#pragma HLS array_partition variable = l_uramVal dim=2 complete
				for (unsigned int j=0; j<t_NumUramPerDdr; ++j) {
					for (unsigned int k=0; k<t_UramWidth; ++k) {
						l_uramVal[j][k] = l_val[j*t_UramWidth+k];
					}
					m_AData[t_StageId][j][i] = l_uramVal[j];
				}
			}
			//forward the rest
			bool l_exit=false;
			bool l_final=false;
			while (!l_exit) { 
			#pragma HLS pipeline
				l_exit = p_inDataS.empty() && l_final;
				bool l_unused;
				if (p_inCntS.read_nb(l_unused)) {
					l_final=true;
				}
			}
		}
		
		void
		spmvDecodeLast (InstrStreamType &p_inS, WideParamStreamType &p_outParamS, unsigned int t_StageId) {
			bool l_exit=false;
			do {
			#pragma HLS pipeline
				Instruction l_instr;
				l_instr = p_inS.read();
				l_exit = (l_instr.m_Op == Instruction::ST);
				uint8_t l_stageId = l_instr.m_StageId;
				ParamType l_nnzBlocks = l_instr.m_NnzBlocks;
				ParamType l_kmBlocks = l_instr.m_KmBlocks;
				WideParamType l_val;
				#pragma HLS array_partition variable=l_val dim=1 complete
				if (l_stageId == t_StageId) {
					l_val[0] = l_nnzBlocks;
					l_val[1] = l_kmBlocks;
					p_outParamS.write(l_val);
				}
			} while (!l_exit);
		}
		
		void 
		uspmvStageMidSeq(
			DataDdrWideStreamType &p_inAdataS,
			ControlStreamType &p_inAcntS,
			DataDdrWideStreamType &p_inBs,
			WideParamStreamType &p_inParamS,
			DataDdrWideStreamType &p_outAdataS,
			ControlStreamType &p_outAcntS,
			DataDdrWideStreamType &p_outCs,
			unsigned int t_StageId
		) {
			WideParamType l_val = p_inParamS.read();
			p_nnzBlocks = l_val[0];
			p_mBlocks = l_val[1];	
			spmvStFwdA (p_inAdataS, p_inAcntS, p_OutAdataS, p_outAcntS, p_nnzBlocks, t_StageId);
			spmvCompute(p_inBs, p_nnzBlocks, t_StageId);
			spmvStreamC(p_outCs, p_mBlocks, t_StageId);
		}
		
		void 
		uspmvStageLastSeq(
			DataDdrWideStreamType &p_inAdataS,
			ControlStreamType &p_inAcntS,
			DataDdrWideStreamType &p_inBs,
			WideParamStreamType &p_inParamS,
			DataDdrWideStreamType &p_outCs,
			unsigned int t_StageId
		) {
			WideParamType l_val = p_inParamS.read();
			p_nnzBlocks = l_val[0];
			p_mBlocks = l_val[1];	
			spmvStoreA (p_inAdataS, p_inAcntS, p_nnzBlocks, t_StageId);
			spmvCompute(p_inBs, p_nnzBlocks, t_StageId);
			spmvStreamC(p_outCs, p_mBlocks, t_StageId);
		}

		void
		loadAB(
			DataDdrWideType *p_rdAddr,
			DataDdrWideType *p_bAddr,
			unsigned int p_aOffsetBase,
			unsigned int p_nnzOffsetBase,
			unsigned int p_kBlocks,
			InstrStreamType &p_outInstrS,
			DataDdrWideStreamType &p_outAs,
			ControlStreamType &p_outAcntS,
			DataDdrWideStreamType &p_outBs
		) {
			//read all Nnzs to construct instructions
			AddrDdrWideType *l_nnzAddr = p_rdAddr + p_nnzOffsetBase * AddrDdrWideType::per4k();
			WideType<AddrType, t_UspmvStages> l_nnzVals;
			WideType<AddrType, t_UspmvStages> l_nnzBlocks;
			#pragma HLS array_partition variable=l_nnzVals dim=1 complete
			#pragma HLS array_partition variable=l_nnzBlocks dim=1 complete
			for (unsigned int i=0; i<t_AddrBlocks; ++i){
			#pragma HLS pipeline
				AddrDdrWideType l_addr = l_nnzAddr[i] 
				#pragma HLS array_partition variable=l_addr dim=1 complete
				for (unsigned int j=0; j<t_NumAddrPerDdr; ++j) {
					if ((i*t_NumAddrPerDdr+j) < t_UspmvStages) {
						l_nnzVals[i*t_NumAddrPerDdr+j] = l_addr[j];
					}
				}
			}
			for (unsigned int i=0; i<t_UspmvStages; ++i) {
			#pragma HLS UNROLL
				l_nnzBlocks[i] = (l_nnzVals[i] / t_DdrWidth)*3;
			}

			for (unsigned int i=0; i<t_UspmvStages; ++i) {
			#pragma HLS pipeline
				Instruction l_instr(Instruction::MUL, i, l_nnzBlocks[i], p_kBlocks);
				p_outInstrS.write(l_instr);
			}
			Instruction l_stInstr(Instruction::ST, t_UspmvStages, 0, p_kBlocks);
			p_outInstrS.write(l_stInstr);

			//load A offsets
			AddrDdrWideType *l_aAddr = p_rdAddr + p_aOffsetBase * AddrDdrWideType::per4k();
			WideType<AddrType, t_UspmvStages> l_aOffsets;
			#pragma HLS array_partition variable=l_Aoffsets dim=1 complete
			for (unsigned int i=0; i<t_AddrBlocks; ++i) {
			#pragma HLS pipeline
				AddrDdrWideType l_addr = l_aAddr[i];
				for (unsigned int j=0; j<t_NumAddrPerDdr; ++j) {
					if ((i*t_NumAddrPerDdr+j) < t_UspmvStages) {
						l_aOffsets[i*t_NumAddrPerDdr+j] = l_addr[j];
					}
				}
			}

			//load A
			for (unsigned int i=0; i<t_UspmvStages; ++i) {
				DataDdrWideType *l_aData	= p_rdAddr + l_aOffsets[i] * DataDdrWideType::per4k();
				for (unsigned int j=0; j<l_nnzBlocks[i]; ++j) {
				#pragma HLS pipeline
					DataDdrWideType l_data = l_aData[j];
					p_outAs.write(l_data);
				} 
			}
			p_outAcntS.write(true);
		
			//loadB
			for (unsigned int i=0; i<p_kBlocks; ++i) {
			#pragma HLS pipeline
				DataDdrWideType l_data = p_bAddr[i];
				p_outBs.write(l_data);
			}		
		}

		storeC(
			DataDdrWideStreamType &p_inCs,
			DataDdrWideType *p_cAddr,
			unsigned int p_kBlocks
		){
			for (unsigned int i=0; i<p_kBlocks; ++i){
			#pragma HLS pipeline
				DataDdrWideType l_data = p_inCs.read();
				p_cAddr[i] = l_data;
			}
		}
		
		void
		streamUspmv(
			DataDdrWideType *p_rdAddr,
			DataDdrWideType *p_bAddr,
			DataDdrWideType *p_cAddr,
			unsigned int p_aOffsetBase,
			unsigned int p_nnzOffsetBase,
			unsigned int p_kBlocks
		)
		{
			InstrStreamType l_instS[t_UspmvStages];
			DataDdrWideStreamType l_aDataS[t_UspmvStages];
			ControlStreamType l_aCntS[t_UspmvStages];
			DataDdrWideStreamType l_bS[t_UspmvStages];
			WideParamStreamType l_paramS[t_UspmvStages];
			DataDdrWideStreamType l_cS;
		#pragma HLS dataflow
			loadAB(p_rdAddr, p_bAddr, p_aOffsetBase, p_nnzOffsetBase, p_kBlocks, 
						 l_instS[0], l_aDataS[0], l_aCntS[0], l_bS[0]);
			for (unsigned int i=0; i<t_UspmvStages-1; ++i) {
			#pragma HLS unroll
				spmvDecode(l_instS[i], l_instS[i+1], l_paramS[i], i);
				uspmvStageMidSeq( l_aDataS[i], l_aCntS[i], l_bS[i], l_paramS[i],
													l_aDataS[i+1],l_aCntS[i+1], l_bS[i+1], i);
			}
			
			spmvDecodeLast (l_instS[t_UspmStages-1], l_paramS[t_UspmvStages-1], t_UspmvStages-1);
			uspmvStageLastSeq(l_aDataS[t_UspmvStages-1], l_aCntS[t_UspmvStages-1], l_bS[t_UspmvStages-1],
												l_paramS[t_UspmvStages-1], l_cS, t_UspmvStages-1);
			storeC(l_cS, p_cAddr, p_kBlocks);	
		}	
		void
		runUspmv(
			DataDdrWideType *p_DdrRd,
			DataDdrWideType *p_DdrWr,
			UspmvArgsType &p_Args
		) {
			t_Debug && std::cout << "\nrunSpmv START K/M=" << p_Args.m_K << std::endl;
			#pragma HLS inline off
			
			DataDdrWideType *l_bAddr = p_DdrRd + p_Args.m_Boffset * DataDdrWideType::per4k();
			DataDdrWideType *l_cAddr = p_DdrRd + p_Args.m_Coffset * DataDdrWideType::per4k();
			const unsigned int l_kBlocks = p_Args.m_K/t_DdrWidth;
			assert(l_kBlocks * t_DdrWidth == p_Args.m_K);

		}
};

} // namespace
#endif

