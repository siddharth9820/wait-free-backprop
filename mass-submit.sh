#!/bin/bash

function build_submit() {
    bin=$1
    run_dir=$(pwd)
    NCCL_HOME=$(command cd ..; pwd)/build/lib

    # find number of nodes, rounding up
    script_name=submit-$bin.sh
    echo "#!/bin/bash

#SBATCH -t 00:30
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 2
#SBATCH --gres=gpu:2
#SBATCH --constraint=rhel8

cd $run_dir

for m in cuda cudnn/gcc openmpi/gcc ; do
    module load \$m
done

export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CUDNN_ROOT/lib64:\$CUDA_HOME/lib64:\$NCCL_HOME

echo running ${bin}
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eno1\
    nvprof --profile-child-processes -o ${bin}-p%p.nvvp\
    mpirun ./${bin}"\
    > $script_name
    chmod +x $script_name
    sbatch $script_name
}

for bin in multi_gpu multi_GPU_wfbp multi_GPU_VanillaBP ; do
    build_submit $bin
done
