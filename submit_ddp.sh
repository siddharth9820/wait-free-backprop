#!/bin/bash
#SBATCH --ntasks-per-node 2 
#SBATCH -t 00:02:00
#SBATCH --constraint="rhel8"
#SBATCH --gres=gpu:2

#Define module command, etc
. ~/.profile
#Load the pytorch module
# module load python
module load pytorch

#Number of processes per node to launch (20 for CPU, 2 for GPU)
NPROC_PER_NODE=2

#This command to run your pytorch script
#You will want to replace this
COMMAND="pytorchDDP.py --batches 192 --cuda --dist_backend gloo"

#We want names of master and slave nodes
MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
#Make sure this node (MASTER) comes first
HOSTLIST="$MASTER $SLAVES"
echo $HOSTLIST

#Get a random unused port on this host(MASTER) between 2000 and 9999
#First line gets list of unused ports
#2nd line restricts between 2000 and 9999
#3rd line gets single random port from the list
# MPORT=`ss -tan | awk '{print $4}' | cut -d':' -f2 | \
#         grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | \
#         sort | uniq | shuf`

MPORT='10000'

#Launch the pytorch processes, first on master (first in $HOSTLIST) then
#on the slaves

RANK=1
for node in $SLAVES; do
        ssh -q $node "module load pytorch && cd /lustre/ssingh37/Acads/CMSC818x/wait-free-backprop && \
                python -m torch.distributed.launch \
                --nproc_per_node=$NPROC_PER_NODE \
                --nnodes=$SLURM_JOB_NUM_NODES \
                --node_rank=$RANK \
                --master_addr="$MASTER" --master_port="$MPORT" \
                $COMMAND" &
        RANK=$((RANK+1))
done

RANK=0
cd /lustre/ssingh37/Acads/CMSC818x/wait-free-backprop
python -m torch.distributed.launch \
--nproc_per_node=$NPROC_PER_NODE \
--nnodes=$SLURM_JOB_NUM_NODES \
--node_rank=$RANK \
--master_addr="$MASTER" --master_port="$MPORT" \
$COMMAND  
