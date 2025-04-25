#!/bin/bash
#SBATCH -o SLURM_Logs/%x_%j_master.out
#SBATCH -e SLURM_Logs/%x_%j_master.err
#SBATCH -D ./
#SBATCH -J llama.cpp 
#SBATCH --nodes 2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --partition="gpu"
#SBATCH --exclusive
#SBATCH -A pawsey0001-gpu
#SBATCH --reservation=PAWSEY_GPU_COS_TESTING
#SBATCH --wait-all-nodes=1
srun ./launch-rpc-server.sh
#srun -N 2 --ntasks=2 --ntasks-per-node=1 --reservation=PAWSEY_GPU_COS_TESTING -A pawsey0001-gpu --gres=gpu:8 --partition=gpu --mem=0 ./launch-rpc-server.sh


