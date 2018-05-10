#!/bin/env bash
. $RDI_REGR_SCRIPTS/init.sh
WRAPPER=$RDI_ROOT/hierdesign/fisInfra/sprite/bin/rdiWrapper
### hwt_check
. /proj/sdxbf/prod/bin/hwt_check.sh

#| Run in normal shell mode
pragma_normal

sdxbf_root_path=/proj/dsamgr/prod
if [ ! -z "$SDXBF_ROOT" ];then
   sdxbf_root_path=$SDXBF_ROOT
fi
hwt_start_test
pre_board_check=$?

echo "Running Test on Board"
echo "Setting XCL_BUILDSYSTEM to 1"
export XCL_BUILDSYSTEM=1

#| Capture the environment before the test runs - useful for debugging
env > env_board.out

#| This function is used to end the script
script_exit () {
  hwt_end_test $1
  exit $?
}

#| Get the test meta-data used in this script - we get all data in one call
#| for performance reasons
echo "Reading test meta-data"
tmd_list=`${RDI_REGR_SCRIPTS}/tmd get test_name user.dsa user.board_check`
if [ $? -ne 0 ];then
   echo "ERROR: Failed to import the test meta-data"
   script_exit 1
fi

#| Create an array of the TMD data (setting IFS set the delimeter for this cmd)
IFS='|' read -ra TMD_DATA <<< "$tmd_list"

#| Get the test name
test_name=${TMD_DATA[0]}
if [ -z "$test_name" ];then
   test_name='unknown'
fi
echo " -> test_name : $test_name"

#| Get the DSA Name - ERROR if this string is not found in the output
dsa_name=${TMD_DATA[1]}
if [ -z "$dsa_name" ];then
   echo "ERROR: DSA name (dsa_name) not found in test meta-data"
   script_exit 1
fi
echo " -> dsa_name : $dsa_name"

#| Get the board check flag from the user section
board_check=${TMD_DATA[2]}
if [ -z "$board_check" ];then
   board_check=0
fi
echo " -> board_check : $board_check"

#| Make sure the board is in a good state before running the test
if [ $board_check == 1 ];then
   if [ $pre_board_check -ne 0 ];then
      echo "ERROR: Board check failed - could not run test"
      script_exit 1
   fi
fi

## Code for running on board
## Runing in xbinst
pushd out_hw/xbinst
# Turn on device profiling
export SDACCEL_TIMELINE_REPORT=true
export SDACCEL_DEVICE_PROFILE=true
export LD_LIBRARY_PATH=$XILINX_SDX/lnx64/tools/opencv:$LD_LIBRARY_PATH
test -z ${DSA_PLATFORM} && script_exit 1
test -z ${DSA} && script_exit 1
#export HALLIB=libxcldrv.so
export HALLIB=libxclgemdrv.so
if [[ $DSA_PLATFORM =~ .*_minotaur-.* || $DSA_PLATFORM =~ .*_aws-.* ]]; then
  export HALLIB=libawsbmdrv.so
#elif [[ $DSA_PLATFORM =~ .*_4_0 || $DSA_PLATFORM =~ .*_4_1 ]]; then
 # export HALLIB=libxclngdrv.so
elif [[ $DSA_PLATFORM =~ .*_3_3 ]]; then
  /bin/cp -prf $XILINX_SDX/lib/lnx64.0/liblmx6.0.so runtime/lib/x86_64/lib/
  export HALLIB=libxcldrv.so
fi
echo "Unsetting XILINX_SDACCEL and XILINX_SDX"
#unset XILINX_SDACCEL
#unset XILINX_SDX
export XILINX_OPENCL=$PWD
export LD_LIBRARY_PATH=$XILINX_OPENCL/runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${XILINX_SDX}/../../../../internal_platforms/${DSA_PLATFORM}/sw/lib/x84_64:$XILINX_OPENCL/runtime/lib/x86_64/:$LD_LIBRARY_PATH
unset XILINX_SDACCEL
unset XILINX_SDX

## Commented by praveen on 07/14/2017
# 2016.4 patch
#if [[ $DSA_PLATFORM =~ .*_3_3 ]]; then
#  echo Overriding XILINX_SDX for 2016.4
#  export XILINX_SDX=/proj/xbuilds/2016.4_sdx_0310_1/installs/lin64/SDx/2016.4
#fi

TIME=$(which time)
echo "$TIME -o time.log ./out_host/gemx_host.exe out_hw/gemx.xclbin out_host/app.bin out_hw/app_out.bin 2>&1 | tee output.log"
GEMX_HOST_DIR=../../out_host
GEMX_HW_DIR=..
$TIME -o time.log ${GEMX_HOST_DIR}/gemx_host.exe ${GEMX_HW_DIR}/gemx.xclbin ${GEMX_HOST_DIR}/app.bin ${GEMX_HW_DIR}/app_out.bin 2>&1 | tee output.log
boardTestRC=${PIPESTATUS[0]}
dmesg > dmesg.log

# Check correctness
echo INFO: translating ${GEMX_HW_DIR}/app_out0.bin to ${GEMX_HW_DIR}/app_out0.txt
${GEMX_HOST_DIR}/gemx_gen_bin.exe -read ${GEMX_HW_DIR}/app_out0.bin > ${GEMX_HW_DIR}/app_out0.txt
echo INFO: Board performance data
head -22 ${GEMX_HW_DIR}/app_out0.txt
echo INFO: Comparing ${GEMX_HOST_DIR}/app_gold.bin ${GEMX_HW_DIR}/app_out0.bin
cmp -i 8192 ${GEMX_HOST_DIR}/app_gold.bin ${GEMX_HW_DIR}/app_out0.bin || ${GEMX_HOST_DIR}/gemx_gen_bin.exe -compare 1e-3 3e-6  ${GEMX_HOST_DIR}/app_gold.bin ${GEMX_HW_DIR}/app_out0.bin && echo INFO: Host Testbench ended Correctness test Status PASS || script_exit 1

popd

# Additional HW commands
type run_board_custom.sh && run_board_custom.sh
type ../run_board_custom.sh && ../run_board_custom.sh

#| Make sure the board is in a good state after running the test
script_exit $boardTestRC
exit $?
