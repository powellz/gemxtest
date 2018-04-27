#!/bin/bash

export PYTHONPATH=./src/python
export XILINX_OPENCL=$1/xbinst
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$XILINX_OPENCL/runtime/lib/x86_64
export PATH=$XILINX_OPENCL/runtime/bin:$PATH
unset XILINX_SDACCEL
unset XILINX_SDX
unset XCL_EMULATION_MODE
