#!/bin/sh
#SBATCH --ntasks-per-node 2 
#SBATCH -t 00:01:00
#SBATCH --gres=gpu:2
#SBATCH --constraint="rhel8"

cd /lustre/ssingh37/Acads/CMSC818x/wait-free-backprop
 
module load cuda
module load cudnn/gcc
module load openmpi/gcc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_ROOT/lib64:$CUDA_HOME/lib64:/lustre/ssingh37/Acads/CMSC818x/nccl/build/lib
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eno1 mpirun ./multi_GPU_wfbp
