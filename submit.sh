#!/bin/sh
#SBATCH --ntasks 2
#SBATCH --ntasks-per-node 1 
#SBATCH -t 10:00
#SBATCH --gres=gpu:2
#SBATCH --constraint="rhel8"
#cd /homes/cmsc818x-1uz9/wait-free-backprop/nccl 

NCCL_DEBUG=INFO mpirun -n 2 ./multi_gpu 
