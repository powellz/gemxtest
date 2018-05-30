#!/bin/bash

export PYTHONPATH=./src/python
export XILINX_OPENCL=$1/xbinst
unset XILINX_SDACCEL
unset XILINX_SDX
unset XCL_EMULATION_MODE
