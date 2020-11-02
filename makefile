home = /cfarhomes/ssingh37/cudnn
layers = convolution.o generic_layer.o
layer_headers = $(home)/cudnn_layers/convolution.h

cc = nvcc
flags = -arch=sm_35 -std=c++11
nvidia_flags = -lcudnn -lcublas
CFLAGS = -I$(CUDNN_INCDIR)
LDFLAGS = -L$(CUDNN_LIBDIR)

all : single_gpu.o $(layers) 
	$(cc) $(CFLAGS) $(LDFLAGS) $(flags) $(home)/single_gpu.o $(layers) $(nvidia_flags)

single_gpu.o : $(home)/single_gpu.cpp $(layers)
	$(cc) -c $(CFLAGS) $(flags) $(home)/single_gpu.cpp

$(layers): %.o: $(home)/cudnn_layers/%.cpp $(home)/cudnn_layers/%.h
	$(cc) -c $(CFLAGS) $(flags) $< -o $@

clean:
	rm *.o