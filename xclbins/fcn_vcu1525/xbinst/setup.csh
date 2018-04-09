setenv XILINX_OPENCL /home/xteng/work/HEAD_WORK/OPENCL_APPS_DEV/src/git/gemx/gemx/out_hw/xbinst
if ( ! $?LD_LIBRARY_PATH ) then
    setenv LD_LIBRARY_PATH $XILINX_OPENCL/runtime/lib/x86_64
else
    setenv LD_LIBRARY_PATH $XILINX_OPENCL/runtime/lib/x86_64:$LD_LIBRARY_PATH
endif
if ( ! $?PATH ) then
    setenv PATH $XILINX_OPENCL/runtime/bin
else
    setenv PATH $XILINX_OPENCL/runtime/bin:$PATH
endif
unsetenv XILINX_SDACCEL
unsetenv XILINX_SDX
unsetenv XCL_EMULATION_MODE
