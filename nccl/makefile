CC := nvcc

# make this an environment variable later
NCCL_HOME := /homes/cmsc818x-1uz3/nccl/build

INCL_PATHS := -I$(CUDA_HOME)/include/ -I$(NCCL_HOME)/include
LIBS := -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib -lcudart -lrt -lmpi -lnccl

SRCS := demo.c
OBJS := $(subst .c,.o,$(SRCS))
BIN := demo

$(BIN): $(SRCS)
	$(CC) $< -o $@ -ccbin mpicc $(INCL_PATHS) $(LIBS)

.PHONY:setup
setup:
	set LD_LIBRARY_PATH="/homes/cmsc818x-1uz3/nccl/build/lib/:"$LD_LIBRARY_PATH
