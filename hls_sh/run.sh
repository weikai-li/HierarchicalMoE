#!/bin/bash
VITIS_VERSION="2021.1"

export VITIS_PYTHON_27_LIBRARY_PATH="$XILINX_VITIS/aietools/tps/lnx64/target_aie_ml/chessdir/python-2.7.13/lib"
export C_INCLUDE_PATH="$XILINX_HLS/lnx64/tools/gcc/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/:$C_INCLUDE_PATH" 
export C_INCLUDE_PATH="$XILINX_HLS/include/:$C_INCLUDE_PATH" 
export C_INCLUDE_PATH="/opt/merlin/sources/merlin-compiler/trunk/source-opt/include/apint_include/:$C_INCLUDE_PATH" 
export C_INCLUDE_PATH="$XILINX_HLS/include:$C_INCLUDE_PATH"
export PATH=/mnt/software/xilinx/Vitis_HLS/${VITIS_VERSION}/bin:$PATH

while getopts "p:c:" opt
do
    case "$opt" in
        p ) port=${OPTARG} ;;
	c ) cmd="$OPTARG" ;;
    esac
done

if [ -z $port ] || [ -z "$cmd" ]
then
    echo "Argument missing"
    exit 1
fi

setsid redis-server --port $port --daemonize yes

mkdir -p /tmp/home
export HOME=/tmp/home

export PATH=$XILINX_XRT/bin:$PATH
export PATH=$XILINX_VIVADO/bin:$PATH
export PATH=$XILINX_VITIS/bin:$XILINX_VITIS/runtime/bin:$PATH

$cmd
