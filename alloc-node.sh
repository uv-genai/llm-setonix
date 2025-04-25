#1 job max for regular queues
#salloc -p gpu-dev --gres=gpu:8 --account=pawsey0001-gpu 
salloc -p gpu --reservation=PAWSEY_GPU_COS_TESTING -A pawsey0001-gpu --exclusive
