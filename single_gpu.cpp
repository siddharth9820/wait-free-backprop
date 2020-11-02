#include "cudnn_layers/generic_layer.h"
#include "cudnn_layers/convolution.h"
#include <iostream>
#include "common.h"

int main(int argc, const char* argv[])
{
    int filter_size[3] = {3, 3, 3};     //HWC
    int input_shape[4] = {64, 1, 10, 10}; //NCHW
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    Layer * conv_layer = new Convolution(filter_size, input_shape, cudnn);
    
    std::cout << conv_layer->get_type() << std::endl;
    int output_shape[4];
    conv_layer->get_output_shape(output_shape);

    std::cout << "Output Shape" << std::endl;
    for(int i=0;i<4;i++)std::cout << output_shape[i] << " " ;
    std::cout << std::endl;

    std::cout << "Workspace Size: " << conv_layer->get_workspace_size() << std::endl;
    std::cout << "Input Size: " << conv_layer->get_input_size() << std::endl;
    std::cout << "Output Size: " << conv_layer->get_output_size() << std::endl;
    conv_layer->allocate_internal_memory();

    std::normal_distribution<float> distribution(MU,SIGMA);
    std::default_random_engine generator;

    float * input, * output, * d_input, * d_output;
    int input_size = conv_layer->get_input_size(), output_size = conv_layer->get_output_size();
    input = (float*)malloc(input_size);
    for(int i=0;i<input_size/sizeof(float);i++) input[i] = distribution(generator);
    
    output = (float*)malloc(output_size);

    checkCUDA(cudaMalloc(&d_input, input_size));
    checkCUDA(cudaMalloc(&d_output, output_size));
    
    checkCUDA(cudaMemcpy(d_input,input,input_size,cudaMemcpyHostToDevice));

    conv_layer->forward(d_input, d_output);

    checkCUDA(cudaMemcpy(output,d_output,output_size,cudaMemcpyDeviceToHost));
    for(int i=0;i<output_size/sizeof(float);i++) std::cout << output[i] << " ";
    std::cout << std::endl;


}