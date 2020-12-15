# Data Parallelism for Distributed Deep Learning
Course Project for CMSC818X : Introduction to Parallel Computing at University of Maryland, College Park

By Joy Kitson, Onur Cankur and Siddharth Singh

## Build

### Dependencies
The first step to building this project is to install its dependencies. Assuming you are on a system 
with the (Modules)[https://modules.readthedocs.io/en/latest/module.html] package, such as the
Deepthought2 computing cluster (DT@), most of these can simply be loaded directly. On DT2, the
neccessary modules are `cuda`, `cudnn/gcc`, and `openmpi/gcc`. These can be load by running

```module load <package name>```

There is one library which our project depends on which unfortunately cannot be installed this way on 
DT2, NCCL. This must instead be installed directly from source. You can clone the NCCL sourcecode from
its (git repo)[https://github.com/NVIDIA/nccl]. We recommend cloning it at the same level as your
local copy of this repo. From there, just follow the
(build instructions)[https://github.com/NVIDIA/nccl#build] for the library.

If you don't have root on the machine you're installing NCCL on, you'll need to add NCCL to your
`LD_LIBRARY_PATH`, rather than follow the
(installation instructions)[https://github.com/NVIDIA/nccl#install] from the repo. In bash, you can
do this by running

```export LD_LIBRARY_PATH=<path to NCCL>/build/lib:${LD_LIBRARY_PATH}```

In either case, you'll also need to set `NCCL_HOME` in a similar fashion

```export NCCL_HOME=<path to NCCL>/build```

in order for our `makefile` to work correctly. Note that it may be helpful to add these exports,
along with the module loads from earlier, to your `.bashrc` (or, on DT2, your `.bashrc.mine`) if you
plan on using this code frequently.

### The Makefile
Now that you have all that setup, you can begin actually building our code. Assumining you've already
cloned this repo, and followed the instructions above, all you should need to do now is `cd` into the
top level of your local copy of this repo, and type

```make```

This should build three executables: `single_gpu`, `multi_GPU_wfbp`, and `multi_GPU_VanillaBP`. In
order to clear these executables, along with all other generated files, simly run

```make clean```

## Run

### Submit
Once you have everything properly built, it's time to run the code. One way to do so is using the
`submit.sh` script. This is a normal singleton batch script, and you'll need to edit it manually
to determine which executable to run. You can do this by changing the executable mentioned after
`mpirun`. You'll also need to change the export statement for `LD_LIBRARY_PATH`, so that the third
last entry matches the path to your NCCL installation. In addition, you'll need to change the path
after the `cd` on line 7 to match the path to your local copy of this repo. Other configuration changes,
such as adjusting the number of nodes the job will run on, also require manually editing the script,
particularly the `#SBATCH` lines at the top of the file.

Once you have the script confiured to your liking, you can submit it like any other batch script with

```sbatch submit.sh```

In order to obtain single GPU timings, use the following command 

```mpirun -np 1 ./multi_GPU_VanillaBP```

For sanity check on MNIST, use the following command

```mpirun -np 1 ./single_gpu```

### Submit DDP

Similarly, `submit-ddp.sh` is a singleton batch script for running `pytorchDDP.py`. In order for
this script to run, properly you'll need to change the path after the `cd` on line 53 to point to
your local copy of this repo. THis script also requires manual changes to adjust the number of nodes
run on and other settings, often by editing the `#SBATCH` lines at the top of the file.

### Mass Submit
If you installed NCCL at the same level as your local copy of this repo, `mass-submit.sh` provides
an easy interface for running a number of jobs at once. By default, it is configured to expect NCCL
to be built there, and should run smoothly without editing if that is the case.

By default, `mass-submit.sh` will run both `multi_GPU_wfbp`, and `multi_GPU_VanillaBP` on 2, 4, 6, 8,
10, and 12 GPUs without profiling, when run in the top level directory of this repo. In order to run
with profiling, just use

```./mass-submit.sh true```

instead of the usual no-argument run. This script works by generating a number of boilerplate
singleton batch scripts, sinilar to `submit.sh` and `submit-ddp.sh` and then running the sbatch
command on them.

# Plots
To see the code used to generate the plots in the final report, see `scaling-plots.ipynb`. This notebook
should contain all the code _and_ data necessary to gnerate these plots.
