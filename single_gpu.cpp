#include "cudnn_layers/generic_layer.h"
#include "cudnn_layers/convolution.h"
#include "cudnn_layers/fc.h"
#include <iostream>
#include "common.h"

void check_conv(cudnnHandle_t cudnn, int filter_size[], int input_shape[])
{
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

    float * input, * output, * grad_output, * grad_input, * d_input, * d_output, * d_grad_input, * d_grad_output;
    int input_size = conv_layer->get_input_size(), output_size = conv_layer->get_output_size();
    
    input = (float*)malloc(input_size);
    grad_output = (float*)malloc(output_size);

    for(int i=0;i<input_size/sizeof(float);i++){
        input[i] = distribution(generator);
    }

    for(int i=0;i<output_size/sizeof(float);i++){
        grad_output[i] = distribution(generator);
    }
    output = (float*)malloc(output_size);
    grad_input = (float*)malloc(input_size);

    checkCUDA(cudaMalloc(&d_input, input_size));
    checkCUDA(cudaMalloc(&d_output, output_size));
    checkCUDA(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
    
    checkCUDA(cudaMalloc(&d_grad_input, input_size));
    checkCUDA(cudaMalloc(&d_grad_output, output_size));
    checkCUDA(cudaMemcpy(d_grad_output, grad_output, output_size, cudaMemcpyHostToDevice));
    
    conv_layer->forward(d_input, d_output);
    conv_layer->backward(d_grad_output, d_grad_input, d_input);

    checkCUDA(cudaMemcpy(grad_input, d_grad_input, input_size, cudaMemcpyDeviceToHost));
}

void check_FC(cublasHandle_t cublas, int output_dim, int input_shape[])
{
    Layer * fc_layer = new FC(output_dim, input_shape, cublas);
    
    std::cout << fc_layer->get_type() << std::endl;
    int output_shape[4];
    fc_layer->get_output_shape(output_shape);

    std::cout << "Output Shape" << std::endl;
    for(int i=0;i<4;i++)std::cout << output_shape[i] << " " ;
    std::cout << std::endl;

    std::cout << "Workspace Size: " << fc_layer->get_workspace_size() << std::endl;
    std::cout << "Input Size: " << fc_layer->get_input_size() << std::endl;
    std::cout << "Output Size: " << fc_layer->get_output_size() << std::endl;
    fc_layer->allocate_internal_memory();

    std::normal_distribution<float> distribution(MU,SIGMA);
    std::default_random_engine generator;

    float * input, * output, * grad_output, * grad_input, * d_input, * d_output, * d_grad_input, * d_grad_output;
    int input_size = fc_layer->get_input_size(), output_size = fc_layer->get_output_size();
    
    input = (float*)malloc(input_size);
    grad_output = (float*)malloc(output_size);

    for(int i=0;i<input_size/sizeof(float);i++){
        input[i] = distribution(generator);
    }

    for(int i=0;i<output_size/sizeof(float);i++){
        grad_output[i] = distribution(generator);
    }
    output = (float*)malloc(output_size);
    grad_input = (float*)malloc(input_size);

    checkCUDA(cudaMalloc(&d_input, input_size));
    checkCUDA(cudaMalloc(&d_output, output_size));
    checkCUDA(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
    
    checkCUDA(cudaMalloc(&d_grad_input, input_size));
    checkCUDA(cudaMalloc(&d_grad_output, output_size));
    checkCUDA(cudaMemcpy(d_grad_output, grad_output, output_size, cudaMemcpyHostToDevice));
    
    fc_layer->forward(d_input, d_output);
    fc_layer->backward(d_grad_output, d_grad_input, d_input);

    // checkCUDA(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // for(int i=0;i<output_size/sizeof(float);i++) std::cout << output[i] << " ";
    // std::cout << std::endl;

    checkCUDA(cudaMemcpy(grad_input, d_grad_input, input_size, cudaMemcpyDeviceToHost));
    
    // for(int i=0;i<input_size/sizeof(float);i++) std::cout << grad_input[i] << " ";
    // std::cout << std::endl;

}



int main(int argc, const char* argv[])
{
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    int filter_size[3] = {3, 3, 3};        //HWC
    int input_shape[4] = {64, 1, 10, 10};  //NCHW
    //check_conv(cudnn, filter_size, input_shape);
    check_FC(cublas, 100, input_shape);
}