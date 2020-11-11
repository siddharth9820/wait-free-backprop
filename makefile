# TODO: Move the build rules for the cudnn_layers dir into their own makefile
LAYER_SRCS = convolution.cpp generic_layer.cpp fc.cpp relu.cpp
LAYER_OBJS = $(subst .cpp,.o,$(LAYER_SRCS))
LAYER_HEADERS = cudnn_layers/convolution.h

MAIN_SRCS = single_gpu.cpp multi_gpu.cpp
MAIN_OBJS = $(subst .cpp,.o,$(MAIN_SRCS))
BINS = $(subst .cpp,,$(MAIN_SRCS))

CC = nvcc -ccbin mpic++
FLAGS = --std=c++11 -arch=sm_35 -lmpi -lm -lcudnn -lcublas -lrt -lcudart 
CFLAGS = -I$(CUDNN_INCDIR)
LDFLAGS = -L$(CUDNN_LIBDIR)64

.PHONY: all
all: $(BINS)

$(BINS) : % : %.o $(LAYER_OBJS) 
	$(CC) $(CFLAGS) $(LDFLAGS) $(FLAGS) $< $(LAYER_OBJS) $(NVIDIA_FLAGS) -o $@

$(MAIN_OBJS) : %.o : %.cpp $(LAYER_OBJS) makefile
	$(CC) -c $(CFLAGS) $(FLAGS) $< -o $@

$(LAYER_OBJS): %.o: cudnn_layers/%.cpp cudnn_layers/%.h makefile
	$(CC) -c $(CFLAGS) $(FLAGS) $< -o $@

clean:
	rm *.o $(BINS)
