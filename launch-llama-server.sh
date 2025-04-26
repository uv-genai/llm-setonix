export MODEL=$1
export RPC_NODE_FILE=$2
export NODES=`cat $RPC_NODE_FILE | tr '\n' ','`
echo " --host 0.0.0.0 --port 8080 -m $MODEL --rpc=\"${NODES::-1}\"" 
