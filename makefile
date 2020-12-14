# TODO: Move the build rules for the cudnn_layers dir into their own makefile
LAYER_SRCS = convolution.cpp generic_layer.cpp fc.cpp relu.cpp
LAYER_OBJS = $(subst .cpp,.o,$(LAYER_SRCS))
LAYER_HEADERS = cudnn_layers/convolution.h

MNIST_SRC = dataloader.cpp 
MNIST_OBJS = $(subst .cpp,.o,$(MNIST_SRCS))

MAIN_SRCS = single_gpu.cpp multi_GPU_wfbp.cpp 
MAIN_OBJS = $(subst .cpp,.o,$(MAIN_SRCS))
BINS = $(subst .cpp,,$(MAIN_SRCS))

# Override this by setting the correspodning environment variable
NCCL_HOME ?= /lustre/ssingh37/Acads/CMSC818x/nccl/build

CC = nvcc -ccbin mpic++
FLAGS = --std=c++11 -arch=sm_35 -lmpi -lm -lcudnn -lcublas -lrt -lcudart -lnccl
CFLAGS = -I$(CUDNN_INCDIR) -I$(NCCL_HOME)/include -Imnist-loader/include
LDFLAGS = -L$(CUDNN_LIBDIR)64 -L$(NCCL_HOME)/lib

.PHONY: all
all: $(BINS)

$(BINS) : % : %.o $(LAYER_OBJS) dataloader
	$(CC) $(CFLAGS) $(LDFLAGS) $(FLAGS) $< $(LAYER_OBJS) dataloader.o mnist-loader/src/mnist_loader.o $(NVIDIA_FLAGS) -o $@

$(MAIN_OBJS) : %.o : %.cpp $(LAYER_OBJS)  makefile
	$(CC) -c $(CFLAGS) $(FLAGS) $< -o $@

$(LAYER_OBJS): %.o: cudnn_layers/%.cpp cudnn_layers/%.h makefile
	$(CC) -c $(CFLAGS) $(FLAGS) $< -o $@

dataloader: dataloader.cpp dataloader.h mnist makefile
	$(CC) -c $(CFLAGS) $(FLAGS) dataloader.cpp -o dataloader.o

mnist: 
	cd mnist-loader && make

.PHONY: clean
clean:
	rm *.o $(BINS) && cd mnist-loader && make clean
