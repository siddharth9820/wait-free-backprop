#!/bin/sh
#SBATCH -N 1
#SBATCH -t 10:00
#SBATCH --gres=gpu:2
#SBATCH --constraint="rhel8"

cd /homes/cmsc818x-1uz9/wait-free-backprop/nccl 

mpirun -n 1 ./demo 
