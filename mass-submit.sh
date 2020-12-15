#!/bin/bash

run_profiling=$1

function build_submit() {
    bin=$1
    n_gpus=$2
    run_dir=$(pwd)
    NCCL_HOME=$(command cd ..; pwd)/build/lib

    if [ "$run_profiling" = true ] ; then
        PROFILING_CMD="nvprof --profile-child-processes -o ${bin}-p%p.nvvp"
    fi

    script_name=submit-${bin}-${n_gpus}.sh
    echo "#!/bin/bash

#SBATCH -t 05:00
#SBATCH --ntasks ${n_gpus}
#SBATCH --ntasks-per-node 2
#SBATCH --gres=gpu:2
#SBATCH --constraint=rhel8

cd $run_dir

for m in cuda cudnn/gcc openmpi/gcc ; do
    module load \$m
done

export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CUDNN_ROOT/lib64:\$CUDA_HOME/lib64:${NCCL_HOME}

echo running ${bin} with ${n_gpus} gpus
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eno1\
    ${PROFILING_CMD}\
    mpirun ./${bin}"\
    > ${script_name}
    chmod +x ${script_name}
    sbatch ${script_name}
}

for bin in multi_GPU_wfbp multi_GPU_VanillaBP ; do
    for n_gpus in 1 2 4 6 8 10 12 ; do
        build_submit $bin $n_gpus
    done
done
