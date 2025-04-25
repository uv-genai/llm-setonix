#!/usr/bin/env bash
echo `hostname`
source $HOME/.opt/llama.cpp-rpc/activate-llama.cpp-rpc.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 rpc-server --host 0.0.0.0 -p 6666
